use crate::statespace::Indexer;

pub struct Graph {
    r: f32,
    size: usize,
    adjs: [u64; 63],
    vulns: [f32; 63],
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Change {
    /// No vector entry greatly changed values.
    Minor,
    /// Some vector entry was changed significantly.
    Major,
}

pub enum CreationError {
    NotStronglyConnected,
    NonexistentNode,
}

impl Graph {
    /// Attempts to create a graph out of the given adjacency matrix, errors if
    /// the graph is not strongly connected, or if some node is connected to a
    /// non-existent node.
    pub fn new(adjs: [u64; 63], r: f32) -> Result<Graph, CreationError> {
        let size = adjs.iter().take_while(|&&adj| adj != 0).count();
        if adjs.iter().any(|&adj| adj >> size != 0) {
            return Err(CreationError::NonexistentNode);
        }
        if !strongly_connected(&adjs[..size]) {
            return Err(CreationError::NotStronglyConnected);
        }
        Ok(Graph {
            r,
            size,
            adjs,
            vulns: calc_vuln(&adjs),
        })
    }

    pub fn len(&self) -> usize {
        self.size
    }

    /// Runs a single Gauss-Siedel step through all elements in `indexer`'s
    /// range, entries are updated in an arbitrary order. Does not update the
    /// very first, or very last entries in the state-space, if they happen to
    /// be included in `indexer`'s range.
    ///
    /// Returns whether any entry was changed significantly.
    pub fn step_division(&self, indexer: &mut Indexer<'_, [f32; 3]>) -> Change {
        assert!(self.size >= 8);

        let topbits = indexer.div_idx() as u64 * indexer.div_len();
        let mut weights = [0.0; _];
        {
            let mut s = topbits;
            while s != 0 {
                let bit = s.trailing_zeros() as usize;
                s &= s - 1;
                self.adjust_neighbours(&mut weights, topbits - s, bit);
            }
        }

        let mut chng = match indexer.div_idx() {
            0 => Change::Minor,
            _ => self.update_entries(&weights, indexer, 0),
        };
        for i in 1..indexer.div_len() {
            let bit = i.trailing_zeros() as usize;
            let i = i ^ i >> 1;
            if indexer.div_idx() == 0xff && i == indexer.div_len() - 1 {
                std::hint::cold_path();
                weights.fill(0.0);
                continue;
            }
            self.adjust_neighbours(&mut weights, topbits | i, bit);
            match self.update_entries(&weights, indexer, i) {
                Change::Minor => {}
                Change::Major => chng = Change::Major,
            }
        }
        chng
    }

    /// Assuming `weights` describe the transition probabilities from
    /// `state ^ (1 << bit)`, adjusts the weights to the transition
    /// probabilities of `state`.
    fn adjust_neighbours(&self, weights: &mut [f32; 63], state: u64, bit: usize) {
        debug_assert!(state < 1 << self.size);
        debug_assert!(bit < self.size);

        let mut adjs = self.adjs[bit];
        let vuln = self.vulns[bit];

        let epidemic = (state >> bit) & 1 != 0;
        let x = if epidemic { -1.0 } else { 1.0 } / adjs.count_ones() as f32;
        let y = -self.r * x;
        weights[bit] = if epidemic {
            vuln - weights[bit] / self.r
        } else {
            (vuln - weights[bit]) * self.r
        };
        while adjs != 0 {
            let adj = adjs.trailing_zeros() as usize;
            weights[adj] += if (state >> adj) & 1 != 0 { x } else { y };
            adjs &= adjs - 1;
        }
    }

    /// Updates entry `indexer.at(idx)` using the transition probabilities in
    /// `weights`, which are assumed to match the indexed entry.
    ///
    /// Returns whether the change was large enough to be judged significant.
    fn update_entries(
        &self,
        weights: &[f32; 63],
        indexer: &mut Indexer<'_, [f32; 3]>,
        idx: u64,
    ) -> Change {
        const OVER_RLX: f32 = 1.5;

        debug_assert!(idx < 1 << self.size);

        let w = &weights[..self.size];
        let sum_w = w.iter().sum::<f32>();

        let [mut sum_p, mut sum_t, mut sum_ct] = [0.0; 3];
        for (&[p, t, ct], &w) in indexer.neighbours(idx).zip(w) {
            sum_p += p * w;
            sum_t += t * w;
            sum_ct += ct * w;
        }
        let [prev_p, prev_t, prev_ct] = *indexer.at(idx);
        let new_p = prev_p + OVER_RLX * (sum_p / sum_w - prev_p);

        let norm = self.size as f32
            + (self.r - 1.0) * (idx.count_ones() + indexer.div_idx().count_ones()) as f32;

        sum_t += norm;
        sum_ct += norm * new_p;

        let new_t = prev_t + OVER_RLX * (sum_t / sum_w - prev_t);
        let new_ct = prev_ct + OVER_RLX * (sum_ct / sum_w - prev_ct);

        *indexer.at(idx) = [new_p, new_t, new_ct];

        // A change is regarded as minor if the ULP difference between the old
        // and new values is less than `0x40`.
        match [(prev_p, new_p), (prev_t, new_t), (prev_ct, new_ct)]
            .map(|(prev, new)| prev.to_bits().abs_diff(new.to_bits()))
        {
            [..0x40, ..0x40, ..0x40] => Change::Minor,
            _ => Change::Major,
        }
    }
}

fn calc_vuln(adjs: &[u64; 63]) -> [f32; 63] {
    let mut vulns = [0.0; _];
    for i in 0..adjs.len() {
        let strength = 1.0 / adjs[i].count_ones() as f32;
        let mut adjs = adjs[i];
        while adjs != 0 {
            vulns[adjs.trailing_zeros() as usize] += strength;
            adjs &= adjs - 1;
        }
    }
    vulns
}

fn strongly_connected(adjs: &[u64]) -> bool {
    for i in 0..adjs.len() {
        let mut seen = 0;
        let mut frontier: u64 = 1 << i;
        while let Some(node) = frontier.lowest_one() {
            seen |= 1 << node;
            frontier &= frontier - 1;
            frontier |= adjs[node as usize] & !seen;
        }
        if seen != (1 << adjs.len()) - 1 {
            return false;
        }
    }
    true
}

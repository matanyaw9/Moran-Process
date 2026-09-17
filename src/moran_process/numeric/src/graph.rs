use super::data::Data;
use std::slice;

pub struct Graph {
    r: f32,
    size: usize,
    adjs: [u64; 63],
    vulns: [f32; 63],
}

#[allow(dead_code)]
pub enum Shape {
    Complete,
    Cycle,
    Star,
    Tree,
}

impl Graph {
    /// Creates a `Graph` out of the given cross-language representation.
    ///
    /// # Safety
    /// `nbrs` and `offsets` must be valid as per the definition in `GraphCore` with respect to
    /// `size`.
    pub unsafe fn from_ffi(size: usize, nbrs: *const u32, offsets: *const u32, r: f32) -> Graph {
        assert!(size < 64);
        let mut adjs = [0; _];
        // SAFETY: Caller guarantees pointer validity
        let offsets = unsafe { slice::from_raw_parts(offsets, size + 1) };
        let nbrs = unsafe { slice::from_raw_parts(nbrs, offsets[size] as usize) };
        for (i, &[x, y]) in offsets.array_windows().enumerate() {
            for &nbr in &nbrs[x as usize..y as usize] {
                adjs[i] |= 1 << nbr;
            }
        }
        Graph {
            r,
            size,
            adjs,
            vulns: calc_vuln(&adjs),
        }
    }

    /// Creates a `Graph` out of the given shape.
    pub fn from_shape(size: usize, shape: Shape, r: f32) -> Graph {
        assert!(size < 64);
        let adjs = std::array::from_fn(|i| match shape {
            _ if i >= size => 0,
            Shape::Complete => (1 << size) - (1 << i) - 1,
            Shape::Cycle => (1 << ((i + size - 1) % size)) | (1 << ((i + 1) % size)),
            Shape::Star if i == 0 => (1 << size) - 2,
            Shape::Star => 1,
            Shape::Tree if i == 0 => 6 & ((1 << size) - 1),
            Shape::Tree => ((6 << (i * 2)) | (1 << ((i - 1) / 2))) & ((1 << size) - 1),
        });
        Graph {
            r,
            size,
            adjs,
            vulns: calc_vuln(&adjs),
        }
    }

    /// Attempts to create a graph out of the given text representation.
    pub fn from_text(text: &str, r: f32) -> Option<Graph> {
        let size = match text.chars().filter(|&c| c == ';').count() {
            s @ ..63 => s + 1,
            _ => return None,
        };
        let mut adjs = [0; _];
        let mut i = 0;
        for num in text.split_inclusive([',', ';']) {
            let n = num.trim_end_matches([',', ';']).parse::<usize>().ok()?;
            adjs[i] |= 1 << n;
            i += num.ends_with(';') as usize;
        }
        Some(Graph {
            r,
            size,
            adjs,
            vulns: calc_vuln(&adjs),
        })
    }

    pub fn len(&self) -> usize {
        self.size
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Run a single Gauss-Siedel step through the `idx`th division, out of 256 (zero-indexed),
    /// entries are updates in an arbitrary order. Does not update the very first, or very last
    /// entries of the data, if the section given is `0` or `255` respectively.
    ///
    /// Returns the ∞-norm of the pairwise ULP distances between the old and new values of the
    /// division.
    pub fn step_division(&self, prob: &Data, time: &Data, idx: u8) -> u32 {
        assert!(self.size >= 8);

        let topbits = (idx as u64) << (self.size - 8);
        let division = 1 << (self.size - 8);

        let mut weights = [0.0; _];
        {
            let mut s = topbits;
            while s != 0 {
                let idx = s.trailing_zeros() as usize;
                s &= s - 1;
                self.adjust_neighbours(&mut weights, topbits - s, idx);
            }
        }

        let mut diff = match idx {
            0 => 0,
            _ => self.update_entries(&weights, prob, time, topbits),
        };
        let last = idx == 0xff;
        let twothirds = 2 * division / 3;

        for i in 1..if last { twothirds } else { division } {
            let state = (i ^ (i >> 1)) | topbits;
            self.adjust_neighbours(&mut weights, state, i.trailing_zeros() as usize);
            diff = diff.max(self.update_entries(&weights, prob, time, state));
        }
        if !last {
            return diff;
        }
        std::hint::cold_path();
        weights.fill(0.0);
        prob.set((1 << self.size) - 1, 1.0);
        for i in twothirds + 1..division {
            let state = (i ^ (i >> 1)) | topbits;
            self.adjust_neighbours(&mut weights, state, i.trailing_zeros() as usize);
            diff = diff.max(self.update_entries(&weights, prob, time, state));
        }
        diff
    }

    /// Assuming `weights` describe the transition probabilities from `state ^ (1 << idx)`, adjusts
    /// the weights to the transition probabilities of `state`.
    fn adjust_neighbours(&self, weights: &mut [f32; 63], state: u64, idx: usize) {
        debug_assert!(state < 1 << self.size);
        debug_assert!(idx < self.size);

        let mut adjs = self.adjs[idx];
        let vuln = self.vulns[idx];

        let epidemic = (state >> idx) & 1 != 0;
        let x = if epidemic { -1.0 } else { 1.0 } / adjs.count_ones() as f32;
        let y = -self.r * x;
        weights[idx] = if epidemic {
            vuln - weights[idx] / self.r
        } else {
            (vuln - weights[idx]) * self.r
        };
        while adjs != 0 {
            let adj = adjs.trailing_zeros() as usize;
            weights[adj] += if (state >> adj) & 1 != 0 { x } else { y };
            adjs &= adjs - 1;
        }
    }

    /// Updates the probability and time entries at index `state` using the
    /// transition probabilities in `weights`. Returns the greater of the
    /// two differences between the old and new values in ULPs.
    fn update_entries(&self, weights: &[f32; 63], prob: &Data, time: &Data, state: u64) -> u32 {
        const OVER_RLX: f32 = 1.5;

        debug_assert!(state < 1 << self.size);

        let prev_prob = prob.get(state);
        let prev_time = time.get(state);
        let w = &weights[..self.size];
        let w_sum = w.iter().sum::<f32>();

        let delta_prob = OVER_RLX
            * (w.iter()
                .enumerate()
                .map(|(i, p)| p * prob.get((1 << i) ^ state))
                .sum::<f32>()
                / w_sum
                - prev_prob);

        let delta_time = OVER_RLX
            * (w.iter()
                .enumerate()
                .map(|(i, p)| p * time.get((1 << i) ^ state))
                .sum::<f32>()
                .algebraic_add(self.size as f32 + (self.r - 1.0) * state.count_ones() as f32)
                / w_sum
                - prev_time);

        prob.set(state, prev_prob + delta_prob);
        time.set(state, prev_time + delta_time);
        let p = prev_prob
            .to_bits()
            .abs_diff((prev_prob + delta_prob).to_bits());
        let t = prev_time
            .to_bits()
            .abs_diff((prev_time + delta_time).to_bits());
        p.max(t)
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

use super::Action;
use super::data::Data;
use std::slice;

pub struct Graph {
    r: f32,
    size: usize,
    nodes: [Node; 63],
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
        let mut nodes = [Node { adjs: 0, vuln: 0.0 }; _];
        // SAFETY: Caller guarantees pointer validity
        let offsets = unsafe { slice::from_raw_parts(offsets, size + 1) };
        let nbrs = unsafe { slice::from_raw_parts(nbrs, offsets[size] as usize) };
        for (i, &[x, y]) in offsets.array_windows().enumerate() {
            for &nbr in &nbrs[x as usize..y as usize] {
                nodes[i].adjs |= 1 << nbr;
            }
        }
        calc_vuln(&mut nodes);
        Graph { r, size, nodes }
    }

    /// Creates a `Graph` out of the given shape.
    pub fn from_shape(size: usize, shape: Shape, r: f32) -> Graph {
        assert!(size < 64);
        let mut nodes = std::array::from_fn(|i| match shape {
            _ if i >= size => 0,
            Shape::Complete => (1 << size) - (1 << i) - 1,
            Shape::Cycle => (1 << ((i + size - 1) % size)) | (1 << ((i + 1) % size)),
            Shape::Star if i == 0 => (1 << size) - 2,
            Shape::Star => 1,
            Shape::Tree if i == 0 => 6 & ((1 << size) - 1),
            Shape::Tree => ((6 << (i * 2)) | (1 << ((i - 1) / 2))) & ((1 << size) - 1),
        })
        .map(|adjs| Node { adjs, vuln: 0.0 });
        calc_vuln(&mut nodes);
        Graph { r, size, nodes }
    }

    /// Attempts to create a graph out of the given text representation.
    pub fn from_text(text: &str, r: f32) -> Option<Graph> {
        let size = match text.chars().filter(|&c| c == ';').count() {
            s @ ..63 => s + 1,
            _ => return None,
        };
        let mut nodes = [Node { adjs: 0, vuln: 0.0 }; _];
        let mut i = 0;
        for num in text.split_inclusive([',', ';']) {
            let n = num.trim_end_matches([',', ';']).parse::<usize>().ok()?;
            nodes[i].adjs |= 1 << n;
            i += num.ends_with(';') as usize;
        }
        calc_vuln(&mut nodes);
        Some(Graph { r, size, nodes })
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
    pub fn step_division(&self, x: &Data, idx: u8, action: Action) -> f32 {
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

        let mut change = 0.0;
        if idx != 0 {
            change = self.update_entry(&weights, x, topbits, action);
        }
        let last = idx == 0xff;
        let twothirds = 2 * division / 3;

        for i in 1..if last { twothirds } else { division } {
            let state = (i ^ (i >> 1)) | topbits;
            self.adjust_neighbours(&mut weights, state, i.trailing_zeros() as usize);
            change += self.update_entry(&weights, x, state, action);
        }
        if !last {
            return change;
        }
        std::hint::cold_path();
        weights.fill(0.0);
        for i in twothirds + 1..division {
            let state = (i ^ (i >> 1)) | topbits;
            self.adjust_neighbours(&mut weights, state, i.trailing_zeros() as usize);
            change += self.update_entry(&weights, x, state, action);
        }
        change
    }

    /// Assuming `weights` describe the transition probabilities from `state ^ (1 << idx)`, adjusts
    /// the weights to the transition probabilities of `state`.
    fn adjust_neighbours(&self, weights: &mut [f32; 63], state: u64, idx: usize) {
        debug_assert!(state < 1 << self.size);
        debug_assert!(idx < self.size);

        let Node { mut adjs, vuln } = self.nodes[idx];
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

    /// Updates the entry at index `state` using the transition probabilities in `weights`.
    fn update_entry(&self, weights: &[f32; 63], x: &Data, state: u64, action: Action) -> f32 {
        const OVER_RLX: f32 = 1.5;

        debug_assert!(state < 1 << self.size);

        let old = x.get(state);
        let w = &weights[..self.size];
        let c = OVER_RLX
            * (w.iter()
                .enumerate()
                .map(|(i, p)| p * x.get((1 << i) ^ state))
                .sum::<f32>()
                .algebraic_add(match action {
                    Action::FixationProb => 0.0,
                    Action::AbsrobTime => {
                        self.len() as f32 + (self.r - 1.0) * state.count_ones() as f32
                    }
                })
                / w.iter().sum::<f32>()
                - old);
        x.set(state, old + c);
        c
    }
}

#[derive(Clone, Copy)]
struct Node {
    /// bitboard of neighbouring nodes
    adjs: u64,
    /// ∑_{(v, u) ∈ V(G)} 1 / deg(v)
    vuln: f32,
}

fn calc_vuln(nodes: &mut [Node; 63]) {
    for i in 0..nodes.len() {
        let strength = 1.0 / nodes[i].adjs.count_ones() as f32;
        let mut adjs = nodes[i].adjs;
        while adjs != 0 {
            nodes[adjs.trailing_zeros() as usize].vuln += strength;
            adjs &= adjs - 1;
        }
    }
}

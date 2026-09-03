use super::data::DataRef;
use std::slice;

pub struct Graph {
    r: f64,
    size: usize,
    nodes: [Node; 63],
}

impl Graph {
    /// Creates a `Graph` out of the given cross-language representation.
    ///
    /// # Safety
    /// `nbrs` and `offsets` must be valid as per the definition in `GraphCore` with respect to
    /// `size`.
    pub unsafe fn from_ffi(size: usize, nbrs: *const u32, offsets: *const u32, r: f64) -> Graph {
        assert!(size < 64);
        let mut nodes = [Node { adjs: 0, vuln: 0.0 }; _];
        // SAFETY: Caller guarantees pointer validity
        let offsets = unsafe { slice::from_raw_parts(offsets, size + 1) };
        let nbrs = unsafe { slice::from_raw_parts(nbrs, offsets[size] as usize) };
        for (i, &[x, y]) in offsets.array_windows().enumerate() {
            let strength = 1.0 / (y - x) as f64;
            for &nbr in &nbrs[x as usize..y as usize] {
                nodes[i].adjs |= 1 << nbr;
                nodes[nbr as usize].vuln += strength;
            }
        }
        Graph { r, size, nodes }
    }

    pub fn len(&self) -> usize {
        self.size
    }

    /// Run a single Gauss-Siedel step through the entries in range `start..start + (1 << bits)`,
    /// in some order. Skips the very first and very last entries of the data, if they happen to be
    /// included in the range.
    pub fn step_section(&self, x: DataRef, start: u64, bits: usize) -> f64 {
        assert!(start.is_multiple_of(1 << bits));
        assert!(bits <= self.size);
        assert!(start <= (1 << self.size) - (1 << bits));

        let mut weights = [0.0; _];
        {
            let mut s = start;
            while s != 0 {
                let idx = s.trailing_zeros() as usize;
                s &= s - 1;
                self.adjust_neighbours(&mut weights, start - s, idx);
            }
        }

        let mut change = 0.0;
        if start != 0 && start != (1 << self.size) - 1 {
            change = self.update_entry(&weights, x, start);
        }
        let last = start == (1 << self.size) - (1 << bits);
        let twothirds = (2 << bits) / 3;
        for i in 1..if last { twothirds } else { 1 << bits } {
            let state = (i ^ (i >> 1)) | start;
            self.adjust_neighbours(&mut weights, state, i.trailing_zeros() as usize);
            change += self.update_entry(&weights, x, state);
        }
        if !last {
            return change;
        }
        std::hint::cold_path();
        weights.fill(0.0);
        for i in twothirds + 1..1 << bits {
            let state = (i ^ (i >> 1)) | start;
            self.adjust_neighbours(&mut weights, state, i.trailing_zeros() as usize);
            change += self.update_entry(&weights, x, state);
        }
        change
    }

    fn adjust_neighbours(&self, weights: &mut [f64; 63], state: u64, idx: usize) {
        debug_assert!(state < 1 << self.size);
        debug_assert!(idx < self.size);

        let Node { mut adjs, vuln } = self.nodes[idx];
        let epidemic = (state >> idx) & 1 != 0;
        let x = if epidemic { -1.0 } else { 1.0 } / adjs.count_ones() as f64;
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

    fn update_entry(&self, weights: &[f64; 63], x: DataRef, state: u64) -> f64 {
        const OVER_RLX: f64 = 1.5;

        debug_assert!(state < 1 << self.size);

        let old = x.get(state);
        let w = &weights[..self.size];
        // TODO: this assumes we're computing fixation probability.
        // add a flag to choose between fixation probability and
        // absorption time: .sum::<f64>().algebraic_add(...)
        let c = OVER_RLX
            * (w.iter()
                .enumerate()
                .map(|(i, p)| p * x.get((1 << i) ^ state))
                .sum::<f64>()
                / w.iter().sum::<f64>()
                - old);
        x.set(state, old + c);
        c.abs()
    }
}

#[derive(Clone, Copy)]
struct Node {
    /// bitboard of neighbouring nodes
    adjs: u64,
    /// ∑_{(v, u) ∈ V(G)} 1 / deg(v)
    vuln: f64,
}

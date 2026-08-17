fn main() {
    const N: usize = 20;
    let g = Graph::<N>::mammal(2.0);

    print!("{:?}", g);

    // SAFETY: bit-pattern zero is a valid `f64` and denotes value `0.0`
    let mut x = unsafe { Box::<[f64; 1 << N]>::new_zeroed().assume_init() };

    for i in 0..500 {
        let change = gauss_seidel_step(&g, &mut x);
        let fix_prob = (0..N).map(|i| x[1 << i]).sum::<f64>() / N as f64;
        println!("step {: >3}: {: <20} diff: {}", i, fix_prob, change);

        if change < 3e-15 * 2.0f64.powi(N as _) {
            break;
        }
    }
}

const OVER_RLX: f64 = 1.5;

fn gauss_seidel_step<const N: usize, const M: usize>(g: &Graph<N>, x: &mut [f64; M]) -> f64 {
    const {
        assert!(1 << N == M);
    }
    x[0] = 0.0;
    x[M - 1] = 1.0;

    let mut change = 0.0;
    for (mut prev, seg) in [(0, 1..M * 2 / 3), (M - 1, M * 2 / 3 + 1..M)] {
        let seg = seg.map(|i| i ^ (i >> 1));
        let mut weights = [0.0; N];
        for state in seg {
            g.adjust_neighbours(
                &mut weights,
                prev as u32,
                (state ^ prev).trailing_zeros() as u8,
            );
            let c = OVER_RLX
                * (weights
                    .iter()
                    .enumerate()
                    .map(|(i, p)| p * x[1 << i ^ state])
                    .sum::<f64>()
                    / weights.iter().sum::<f64>()
                    - x[state]);
            x[state] += c;
            change += c.abs();
            prev = state;
        }
    }
    change
}

pub struct Graph<const N: usize> {
    r: f64,
    adj_mat: [u32; N],
    /// vuln[u] = ∑_{(v, u) ∈ V} 1 / deg(v)
    vuln: [f64; N],
}

impl<const N: usize> Graph<N> {
    /// Given `weights` stores the connection weights of `state`, adjusts
    /// `weights` to store the connection weights of `state ^ 1 << idx`.
    pub fn adjust_neighbours(&self, weights: &mut [f64; N], state: u32, idx: u8) {
        let idx = idx as usize;
        debug_assert!(state < 1 << N);
        debug_assert!(idx < N);

        let neighbours = self.adj_mat[idx];
        let conn_stength = 1.0 / neighbours.count_ones() as f64;

        if state >> idx & 1 == 0 {
            let mut n = neighbours & !state;
            while n != 0 {
                weights[n.trailing_zeros() as usize] += conn_stength * self.r;
                n &= n - 1;
            }
            n = neighbours & state;
            while n != 0 {
                weights[n.trailing_zeros() as usize] -= conn_stength;
                n &= n - 1;
            }
            weights[idx] = self.vuln[idx] - weights[idx] / self.r;
        } else {
            let mut n = neighbours & !state;
            while n != 0 {
                weights[n.trailing_zeros() as usize] -= conn_stength * self.r;
                n &= n - 1;
            }
            n = neighbours & state;
            while n != 0 {
                weights[n.trailing_zeros() as usize] += conn_stength;
                n &= n - 1;
            }
            weights[idx] = (self.vuln[idx] - weights[idx]) * self.r;
        }
    }

    pub fn complete(r: f64) -> Self {
        Graph::new(r, std::array::from_fn(|i| (1 << N) - (1 << i) - 1))
    }

    pub fn mammal(r: f64) -> Self {
        Graph::new(
            r,
            std::array::from_fn(|i| {
                (3u32.unbounded_shl(2 * i as u32 + 1)
                    | i.checked_sub(1).map_or(0, |i| 1 << (i / 2)))
                    & ((1 << N) - 1)
            }),
        )
    }

    fn new(r: f64, adj_mat: [u32; N]) -> Self {
        let mut vuln = [0.0; _];
        for mut node in adj_mat {
            let strength = 1.0 / node.count_ones() as f64;
            while node != 0 {
                vuln[node.trailing_zeros() as usize] += strength;
                node &= node - 1;
            }
        }
        Graph { r, adj_mat, vuln }
    }
}

impl<const N: usize> std::fmt::Debug for Graph<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "r = {}", self.r)?;
        for &line in &self.adj_mat {
            for i in 0..N - 1 {
                write!(f, "{}, ", (line >> i) & 1)?;
            }
            writeln!(f, "{}", line >> (N - 1))?;
        }
        Ok(())
    }
}

fn main() {
    const N: usize = 15;
    let g = Graph::<N>::mammal(2.0);

    print!("{:?}", g);

    // SAFETY: bit-pattern zero is a valid `f64` and denotes value `0.0`
    let mut x = unsafe { Box::<[f64; 1 << N]>::new_zeroed().assume_init() };

    for i in 0..400 {
        gauss_seidel_step(&g, &mut x);
        let fix_prob = (0..N).map(|i| x[1 << i]).sum::<f64>() / N as f64;
        println!("step {: >3}: {}", i, fix_prob);
    }
}

fn gauss_seidel_step<const N: usize, const M: usize>(g: &Graph<N>, x: &mut [f64; M]) {
    const {
        assert!(1 << N == M);
    }
    x[0] = 0.0;
    x[(1 << N) - 1] = 1.0;
    for i in 1..x.len() - 1 {
        let state = i as u32;
        let neighs = g.neighbours(state);
        x[i] = neighs
            .iter()
            .enumerate()
            .map(|(i, p)| p * x[1 << i ^ state as usize])
            .sum::<f64>()
            / neighs.iter().sum::<f64>();
    }
}

pub struct Graph<const N: usize> {
    r: f64,
    adj_mat: [u32; N],
}

impl<const N: usize> Graph<N> {
    /// Probabilities to go from `state` to all of its neighbours.
    /// `g.neighbours(s)[i]` is the probability to go from state `s` to
    /// `s ^ (1 << i)`.
    pub fn neighbours(&self, state: u32) -> [f64; N] {
        debug_assert!(state < 1 << N);
        let mut res = [0.0; _];
        let denum = N as f64 + (self.r - 1.0) * state.count_ones() as f64;
        for (bit, place) in (0..).map(|i| 1 << i).zip(&mut res) {
            let epidemic = state & bit == 0;
            let mut sum = 0.0;

            let mut attks = state ^ if epidemic { 0 } else { (1 << N) - 1 };
            while attks != 0 {
                let j = attks.trailing_zeros() as usize;
                attks &= attks - 1;
                if self.adj_mat[j] & bit != 0 {
                    sum += 1.0 / self.adj_mat[j].count_ones() as f64;
                }
            }
            *place = sum * if epidemic { self.r } else { 1.0 } / denum;
        }
        res
    }

    pub fn complete(r: f64) -> Self {
        Graph {
            r,
            adj_mat: std::array::from_fn(|i| (1 << N) - (1 << i) - 1),
        }
    }

    pub fn mammal(r: f64) -> Self {
        Graph {
            r,
            adj_mat: std::array::from_fn(|i| {
                ((1 << N) - 1)
                    & if i == 0 {
                        6
                    } else {
                        3u32.unbounded_shl(2 * i as u32 + 1) + (1 << ((i - 1) / 2))
                    }
            }),
        }
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

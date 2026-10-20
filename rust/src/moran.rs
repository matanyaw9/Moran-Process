//! Multi-type Moran process on a weighted, possibly directed graph.
//!
//! W[i, j] > 0 means i can place offspring on j. Type 0 is the wild type (fitness 1);
//! mutant type k is type k + 1 internally and reported as k.

use std::ops::Range;

pub const WILD_TYPE: i8 = -1;   
pub const CENSORED: i8 = -2; // TODO might be wasteful, we can see n_steps

/// One run: winner (mutant k, WILD_TYPE or CENSORED), Moran steps, and n_changes, the
/// steps in which a node changed type. Both engines give all three the same distribution.
type Outcome = (i8, u64, u64);

/// Outcomes of all runs, one column each, so they map straight onto numpy arrays.
#[derive(Default)]
pub struct Runs {  // TODO Those Vecs can be boxes, since we know how many runs we do.
    pub winner: Vec<i8>,
    pub steps: Vec<u64>,
    pub n_changes: Vec<u64>,
}

impl Runs {
    fn push(&mut self, (winner, steps, n_changes): Outcome) {
        self.winner.push(winner);
        self.steps.push(steps);
        self.n_changes.push(n_changes);
    }
}

#[derive(Clone, Copy)]
pub enum Rule {
    Bd,
    Db,
}

/// xoshiro256++ seeded through splitmix64, as in the v1 C++ engine.
pub struct Rng([u64; 4]);

impl Rng {
    pub fn new(seed: u64) -> Self {
        let mut s = seed;
        let mut splitmix = || {
            s = s.wrapping_add(0x9e3779b97f4a7c15);
            let z = (s ^ (s >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            let z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            z ^ (z >> 31)
        };
        Rng([splitmix(), splitmix(), splitmix(), splitmix()])
    }

    fn next(&mut self) -> u64 {
        let s = &mut self.0;
        let out = s[0].wrapping_add(s[3]).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        out
    }

    /// Uniform in [0, 1).
    fn uniform(&mut self) -> f64 {
        (self.next() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Unbiased integer in [0, n), Lemire's method.
    fn below(&mut self, n: usize) -> usize {
        let n = n as u64;
        let mut m = self.next() as u128 * n as u128;
        if (m as u64) < n {
            let threshold = n.wrapping_neg() % n;
            while (m as u64) < threshold {
                m = self.next() as u128 * n as u128;
            }
        }
        (m >> 64) as usize
    }

    /// Trials up to and including the first success, success probability p.
    fn geometric(&mut self, p: f64) -> f64 {
        if p >= 1.0 {
            return 1.0;
        }
        1.0 + ((1.0 - self.uniform()).ln() / (-p).ln_1p()).floor()
    }

    /// Index k drawn with probability weight(k) / total.
    fn pick(&mut self, len: usize, total: f64, weight: impl Fn(usize) -> f64) -> usize {
        let mut u = self.uniform() * total;
        let mut last = 0;
        for k in 0..len {
            let w = weight(k);
            if w > 0.0 {
                if u < w {
                    return k;
                }
                u -= w;
                last = k;
            }
        }
        last // only reached through float round-off
    }
}

/// CSR adjacency, either out-neighbors or in-neighbors (the transpose). Zero-weight
/// edges are dropped. `w` is None when all weights are equal (the unweighted case).
struct Adj {
    ptr: Vec<u32>,
    idx: Vec<u32>,
    w: Option<Vec<f32>>,
}

impl Adj {
    fn new(indptr: &[i32], indices: &[i32], weights: &[f64], transpose: bool) -> Self {
        let n = indptr.len() - 1;
        let edges = || {
            (0..n)
                .flat_map(|i| (indptr[i] as usize..indptr[i + 1] as usize).map(move |k| (i, k)))
                .filter(|&(_, k)| weights[k] > 0.0)
                .map(|(i, k)| {
                    let j = indices[k] as usize;
                    if transpose { (j, i, weights[k]) } else { (i, j, weights[k]) }
                })
        };
        let mut ptr = vec![0u32; n + 1];
        for (row, _, _) in edges() {
            ptr[row + 1] += 1;
        }
        for i in 0..n {
            ptr[i + 1] += ptr[i];
        }
        let mut next: Vec<usize> = ptr.iter().map(|&p| p as usize).collect();
        let mut idx = vec![0; ptr[n] as usize];
        let mut w = vec![0.0; ptr[n] as usize];
        for (row, col, x) in edges() {
            idx[next[row]] = col as u32;
            w[next[row]] = x as f32;
            next[row] += 1;
        }
        let weighted = w.windows(2).any(|p| p[0] != p[1]);
        Adj { ptr, idx, w: weighted.then_some(w) }
    }

    /// Replaces each row's weights by their running sum, for binary-search draws.
    fn cumulative(mut self) -> Self {
        if let Some(w) = &mut self.w {
            for row in self.ptr.windows(2) {
                let mut acc = 0.0f64; // f64 so long rows do not drift
                for k in row[0] as usize..row[1] as usize {
                    acc += w[k] as f64;
                    w[k] = acc as f32;
                }
            }
        }
        self
    }

    fn range(&self, i: usize) -> Range<usize> {
        self.ptr[i] as usize..self.ptr[i + 1] as usize
    }

    fn weight(&self, k: usize) -> f64 {
        self.w.as_ref().map_or(1.0, |w| w[k] as f64)
    }
}

/// Nodes grouped by type, so drawing a reproducer by fitness costs O(types), not O(n).
struct Population {
    kind: Vec<u8>,
    pools: Vec<Vec<u32>>,
    slot: Vec<u32>, // position of each node inside its pool
}

impl Population {
    fn new(n: usize, n_types: usize) -> Self {
        Population { kind: vec![0; n], pools: vec![Vec::new(); n_types], slot: vec![0; n] }
    }

    /// Everyone wild type, then each mutant type on distinct uniformly random nodes.
    fn place(&mut self, n_mutants: &[u32], rng: &mut Rng) {
        let n = self.kind.len();
        self.kind.fill(0);
        for pool in &mut self.pools {
            pool.clear();
        }
        self.pools[0].extend(0..n as u32);
        self.slot.iter_mut().enumerate().for_each(|(i, s)| *s = i as u32);
        for (k, &count) in n_mutants.iter().enumerate() {
            for _ in 0..count {
                let node = self.pools[0][rng.below(self.pools[0].len())];
                self.set(node as usize, k as u8 + 1);
            }
        }
    }

    fn set(&mut self, node: usize, t: u8) {
        let s = self.slot[node] as usize;
        let old = &mut self.pools[self.kind[node] as usize];
        old.swap_remove(s);
        if let Some(&moved) = old.get(s) {
            self.slot[moved as usize] = s as u32;
        }
        let new = &mut self.pools[t as usize];
        self.slot[node] = new.len() as u32;
        new.push(node as u32);
        self.kind[node] = t;
    }

    fn fixed(&self) -> Option<usize> {
        self.pools.iter().position(|p| p.len() == self.kind.len())
    }
}

/// One run from the current placement. A step whose
/// reproducer has no out-neighbors (Bd) or whose dying node has no in-neighbors (dB)
/// changes nothing but still counts.
fn run(adj: &Adj, rule: Rule, fitness: &[f64], pop: &mut Population, rng: &mut Rng, max_steps: u64) -> Outcome {
    let n = pop.kind.len();
    if let Some(t) = pop.fixed() {
        return (t as i8 - 1, 0, 0);
    }
    let mut changes = 0;
    for step in 1..=max_steps {
        let total: f64 = pop.pools.iter().zip(fitness).map(|(p, f)| p.len() as f64 * f).sum();
        if total == 0.0 {
            return (CENSORED, max_steps, changes); // only fitness-0 types left, nothing can ever change
        }
        let (target, t) = match rule {
            Rule::Bd => {
                let t = rng.pick(fitness.len(), total, |k| pop.pools[k].len() as f64 * fitness[k]);
                let parent = pop.pools[t][rng.below(pop.pools[t].len())] as usize;
                let r = adj.range(parent);
                if r.is_empty() {
                    continue;
                }
                let k = match &adj.w {
                    None => rng.below(r.len()),
                    Some(cum) => {
                        let cum = &cum[r.clone()];
                        let row_total = cum[cum.len() - 1] as f64;
                        if row_total == 0.0 {
                            continue;
                        }
                        let u = rng.uniform() * row_total;
                        cum.partition_point(|&c| c as f64 <= u).min(r.len() - 1)
                    }
                };
                (adj.idx[r.start + k] as usize, t as u8)
            }
            Rule::Db => {
                let dead = rng.below(n);
                let r = adj.range(dead);
                let weight =
                    |k: usize| fitness[pop.kind[adj.idx[r.start + k] as usize] as usize] * adj.weight(r.start + k);
                let in_total: f64 = (0..r.len()).map(weight).sum();
                if in_total == 0.0 {
                    continue;
                }
                let parent = adj.idx[r.start + rng.pick(r.len(), in_total, weight)] as usize;
                (dead, pop.kind[parent])
            }
        };
        if pop.kind[target] != t {
            pop.set(target, t);
            changes += 1;
            if pop.pools[t as usize].len() == n {
                return (t as i8 - 1, step, changes);
            }
        }
    }
    (CENSORED, max_steps, changes)
}

/// Fenwick tree over per-node rates: O(log n) update and weighted draw.
struct Fenwick {
    tree: Vec<f64>,
    total: f64,
}

impl Fenwick {
    fn build(&mut self, vals: &[f64]) {
        let n = vals.len();
        self.tree.clear();
        self.tree.push(0.0);
        self.tree.extend_from_slice(vals);
        for i in 1..=n {
            let j = i + (i & i.wrapping_neg());
            if j <= n {
                self.tree[j] += self.tree[i];
            }
        }
        self.total = vals.iter().sum();
    }

    fn add(&mut self, i: usize, delta: f64) {
        self.total += delta;
        let mut i = i + 1;
        while i < self.tree.len() {
            self.tree[i] += delta;
            i += i & i.wrapping_neg();
        }
    }

    /// The i with prefix(i) <= u < prefix(i + 1).
    fn find(&self, mut u: f64) -> usize {
        let n = self.tree.len() - 1;
        let (mut pos, mut step) = (0, 1 << n.ilog2());
        while step > 0 {
            if pos + step <= n && self.tree[pos + step] <= u {
                pos += step;
                u -= self.tree[pos];
            }
            step >>= 1;
        }
        pos.min(n - 1)
    }
}

/// Bookkeeping for Bd with null steps skipped. Edge i -> j is active when i and j differ
/// in type. Node i's rate f_i * (active out-weight) / (total out-weight) is, divided by
/// the total fitness F, the chance that a step picks i and changes something.
struct Active {
    inc: Adj,             // in-neighbors: who is affected when a node flips
    row_total: Vec<f64>,  // total out-weight of each node
    count: Vec<u32>,      // active out-edges, exact, so "none left" never drifts
    weight: Vec<f64>,     // active out-weight
    rate: Vec<f64>,
    tree: Fenwick,
    n_live: usize,        // nodes with rate > 0; 0 means the process is stuck
    flips: usize,
}

impl Active {
    fn new(out: &Adj, indptr: &[i32], indices: &[i32], weights: &[f64]) -> Self {
        let n = indptr.len() - 1;
        Active {
            inc: Adj::new(indptr, indices, weights, true),
            row_total: (0..n).map(|i| out.range(i).map(|k| out.weight(k)).sum()).collect(),
            count: vec![0; n],
            weight: vec![0.0; n],
            rate: vec![0.0; n],
            tree: Fenwick { tree: Vec::with_capacity(n + 1), total: 0.0 },
            n_live: 0,
            flips: 0,
        }
    }

    fn reset(&mut self, out: &Adj, fitness: &[f64], kind: &[u8]) {
        for i in 0..kind.len() {
            self.recount(i, out, kind);
            self.rate[i] = self.rate_of(i, fitness, kind);
        }
        self.n_live = self.rate.iter().filter(|&&r| r > 0.0).count();
        self.tree.build(&self.rate);
        self.flips = 0;
    }

    fn recount(&mut self, i: usize, out: &Adj, kind: &[u8]) {
        let (mut c, mut w) = (0, 0.0);
        for k in out.range(i) {
            if kind[out.idx[k] as usize] != kind[i] {
                c += 1;
                w += out.weight(k);
            }
        }
        self.count[i] = c;
        self.weight[i] = w;
    }

    fn rate_of(&self, i: usize, fitness: &[f64], kind: &[u8]) -> f64 {
        if self.count[i] == 0 { 0.0 } else { fitness[kind[i] as usize] * self.weight[i] / self.row_total[i] }
    }

    fn update_rate(&mut self, i: usize, fitness: &[f64], kind: &[u8]) {
        let (old, new) = (self.rate[i], self.rate_of(i, fitness, kind));
        self.n_live = self.n_live + (new > 0.0) as usize - (old > 0.0) as usize;
        self.tree.add(i, new - old);
        self.rate[i] = new;
    }

    /// Node j has just changed from type `old` to kind[j].
    fn flip(&mut self, j: usize, old: u8, out: &Adj, fitness: &[f64], kind: &[u8]) {
        self.recount(j, out, kind);
        self.update_rate(j, fitness, kind);
        for k in self.inc.range(j) {
            let src = self.inc.idx[k] as usize;
            let (was, now) = (kind[src] != old, kind[src] != kind[j]);
            if src == j || was == now {
                continue;
            }
            let w = self.inc.weight(k);
            if now {
                self.count[src] += 1;
                self.weight[src] += w;
            } else {
                self.count[src] -= 1;
                self.weight[src] = if self.count[src] == 0 { 0.0 } else { self.weight[src] - w };
            }
            self.update_rate(src, fitness, kind);
        }
        // Incremental tree updates accumulate round-off; a rebuild every n flips is O(1) amortized.
        self.flips += 1;
        if self.flips >= kind.len() {
            self.tree.build(&self.rate);
            self.flips = 0;
        }
    }

    fn draw(&mut self, rng: &mut Rng) -> usize {
        loop {
            let i = self.tree.find(rng.uniform() * self.tree.total);
            if self.rate[i] > 0.0 {
                return i;
            }
            self.tree.build(&self.rate); // round-off landed on a zero-rate node
        }
    }
}

/// Bd that only simulates steps that change the population. The null steps in between
/// are geometric with success probability (sum of rates) / F, so `steps` has the same
/// distribution as in `run`, and censoring at max_steps is unchanged.
fn run_skipping(out: &Adj, act: &mut Active, fitness: &[f64], pop: &mut Population, rng: &mut Rng, max_steps: u64) -> Outcome {
    let n = pop.kind.len();
    if let Some(t) = pop.fixed() {
        return (t as i8 - 1, 0, 0);
    }
    act.reset(out, fitness, &pop.kind);
    let (mut steps, mut changes) = (0u64, 0u64);
    loop {
        if act.n_live == 0 {
            return (CENSORED, max_steps, changes); // no step can ever change anything again
        }
        let total_fitness: f64 = pop.pools.iter().zip(fitness).map(|(p, f)| p.len() as f64 * f).sum();
        let gap = rng.geometric(act.tree.total / total_fitness);
        if gap > (max_steps - steps) as f64 {
            return (CENSORED, max_steps, changes);
        }
        steps += gap as u64;

        let parent = act.draw(rng);
        let t = pop.kind[parent];
        let r = out.range(parent);
        let differs = |k: usize| if pop.kind[out.idx[r.start + k] as usize] != t { out.weight(r.start + k) } else { 0.0 };
        let target = out.idx[r.start + rng.pick(r.len(), act.weight[parent], differs)] as usize;
        let old = pop.kind[target];
        pop.set(target, t);
        changes += 1;
        if pop.pools[t as usize].len() == n {
            return (t as i8 - 1, steps, changes);
        }
        act.flip(target, old, out, fitness, &pop.kind);
    }
}

/// `n_repeats` independent runs on the CSR graph (indptr, indices, weights).
/// Mutant type k (at most 127 types) starts on n_mutants[k] random nodes with fitness[k].
/// `skip_null` (Bd only) uses `run_skipping`: same output distribution, fewer steps simulated.
#[allow(clippy::too_many_arguments)]
pub fn simulate(
    indptr: &[i32],
    indices: &[i32],
    weights: &[f64],
    n_mutants: &[u32],
    fitness: &[f64],
    rule: Rule,
    skip_null: bool,
    n_repeats: usize,
    max_steps: u64,
    seed: u64,
) -> Runs {
    assert!(!skip_null || matches!(rule, Rule::Bd), "null-step skipping is Bd only");
    let adj = Adj::new(indptr, indices, weights, matches!(rule, Rule::Db));
    let fitness: Vec<f64> = std::iter::once(1.0).chain(fitness.iter().copied()).collect();
    let mut pop = Population::new(indptr.len() - 1, fitness.len());
    let mut rng = Rng::new(seed);
    let mut runs = Runs::default();
    if skip_null {
        let mut act = Active::new(&adj, indptr, indices, weights);
        for _ in 0..n_repeats {
            pop.place(n_mutants, &mut rng);
            runs.push(run_skipping(&adj, &mut act, &fitness, &mut pop, &mut rng, max_steps));
        }
    } else {
        let adj = if matches!(rule, Rule::Bd) { adj.cumulative() } else { adj };
        for _ in 0..n_repeats {
            pop.place(n_mutants, &mut rng);
            runs.push(run(&adj, rule, &fitness, &mut pop, &mut rng, max_steps));
        }
    }
    runs
}

#[cfg(test)]
mod tests {
    use super::*;

    fn csr(n: usize, edges: &[(usize, usize)]) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
        let mut sorted = edges.to_vec();
        sorted.sort();
        let mut indptr = vec![0i32; n + 1];
        for &(i, _) in &sorted {
            indptr[i + 1] += 1;
        }
        for i in 0..n {
            indptr[i + 1] += indptr[i];
        }
        let indices = sorted.iter().map(|&(_, j)| j as i32).collect();
        (indptr, indices, vec![1.0; sorted.len()])
    }

    fn undirected(edges: &[(usize, usize)]) -> Vec<(usize, usize)> {
        edges.iter().flat_map(|&(i, j)| [(i, j), (j, i)]).collect()
    }

    fn complete(n: usize) -> Vec<(usize, usize)> {
        (0..n).flat_map(|i| (0..n).filter(move |&j| j != i).map(move |j| (i, j))).collect()
    }

    fn star(n: usize) -> Vec<(usize, usize)> {
        (1..n).map(|j| (0, j)).collect()
    }

    type Engine = (Rule, bool);
    const BD: [Engine; 2] = [(Rule::Bd, false), (Rule::Bd, true)];
    const ALL: [Engine; 3] = [(Rule::Bd, false), (Rule::Bd, true), (Rule::Db, false)];

    /// Fraction of runs won by mutant type 0, asserting nothing was censored.
    fn rho((p, i, w): &(Vec<i32>, Vec<i32>, Vec<f64>), r: f64, (rule, skip): Engine, repeats: usize) -> f64 {
        let winner = simulate(p, i, w, &[1], &[r], rule, skip, repeats, 1_000_000, 42).winner;
        assert!(winner.iter().all(|&x| x != CENSORED));
        winner.iter().filter(|&&x| x == 0).count() as f64 / repeats as f64
    }

    fn assert_close(got: f64, want: f64, repeats: usize) {
        let se = (want * (1.0 - want) / repeats as f64).sqrt();
        assert!((got - want).abs() < 4.0 * se, "got {got}, want {want} +- {se}");
    }

    #[test]
    fn complete_graph_matches_moran_formula() {
        let (n, r, reps) = (10, 2.0_f64, 20_000);
        let want = (1.0 - 1.0 / r) / (1.0 - r.powi(-(n as i32)));
        for e in BD {
            assert_close(rho(&csr(n, &complete(n)), r, e, reps), want, reps);
        }
    }

    #[test]
    fn directed_cycle_matches_moran_formula() {
        let (n, r, reps) = (8, 1.5_f64, 20_000);
        let cycle: Vec<_> = (0..n).map(|i| (i, (i + 1) % n)).collect();
        let want = (1.0 - 1.0 / r) / (1.0 - r.powi(-(n as i32)));
        for e in BD {
            assert_close(rho(&csr(n, &cycle), r, e, reps), want, reps);
        }
    }

    // Undirected star whose leaf -> hub edges weigh 0 acts as the directed star, so
    // rho = 1/5. Ignoring weights would give the undirected star's much larger rho.
    #[test]
    fn zero_weight_edges_are_never_used() {
        let (p, i, mut w) = csr(5, &undirected(&star(5)));
        for k in 0..w.len() {
            w[k] = if (p[0] as usize..p[1] as usize).contains(&k) { i[k] as f64 } else { 0.0 };
        }
        for e in ALL {
            assert_close(rho(&(p.clone(), i.clone(), w.clone()), 3.0, e, 20_000), 0.2, 20_000);
        }
    }

    #[test]
    fn neutral_rho_is_one_over_n() {
        let star = undirected(&star(6));
        for e in ALL {
            assert_close(rho(&csr(6, &star), 1.0, e, 20_000), 1.0 / 6.0, 20_000);
        }
    }

    // Hub -> leaves: the hub is the only source, so a mutant fixes iff it starts there,
    // whatever r is. Reading W in the wrong direction would never fix (censored).
    #[test]
    fn directed_star_fixes_only_from_hub() {
        for e in ALL {
            assert_close(rho(&csr(5, &star(5)), 3.0, e, 20_000), 0.2, 20_000);
        }
    }

    #[test]
    fn zero_fitness_mutant_never_wins() {
        let (p, i, w) = csr(10, &complete(10));
        for skip in [false, true] {
            let winner = simulate(&p, &i, &w, &[3, 2], &[0.0, 1.5], Rule::Bd, skip, 5_000, 1_000_000, 7).winner;
            assert!(winner.iter().all(|&x| x == WILD_TYPE || x == 1));
        }
    }

    #[test]
    fn censored_when_max_steps_hit() {
        let (p, i, w) = csr(50, &complete(50));
        for skip in [false, true] {
            let runs = simulate(&p, &i, &w, &[2], &[1.0], Rule::Bd, skip, 100, 1, 0);
            assert!(runs.winner.iter().all(|&x| x == CENSORED));
            assert!(runs.steps.iter().zip(&runs.n_changes).all(|(&s, &c)| s == 1 && c <= 1));
        }
    }

    /// (steps, n_changes) of the runs that ended with `winner`.
    fn ending_in(runs: &Runs, winner: i8) -> (Vec<u64>, Vec<u64>) {
        let keep = |col: &[u64]| -> Vec<u64> {
            runs.winner.iter().zip(col).filter(|(w, _)| **w == winner).map(|(_, &x)| x).collect()
        };
        (keep(&runs.steps), keep(&runs.n_changes))
    }

    fn mean(x: &[u64]) -> f64 {
        x.iter().sum::<u64>() as f64 / x.len() as f64
    }

    /// Two-sample Kolmogorov-Smirnov statistic: the largest gap between the empirical CDFs.
    fn ks(a: &[u64], b: &[u64]) -> f64 {
        let (mut a, mut b) = (a.to_vec(), b.to_vec());
        a.sort_unstable();
        b.sort_unstable();
        let (mut i, mut j, mut d) = (0, 0, 0.0f64);
        while i < a.len() && j < b.len() {
            let x = a[i].min(b[j]);
            while i < a.len() && a[i] == x {
                i += 1;
            }
            while j < b.len() && b[j] == x {
                j += 1;
            }
            d = d.max((i as f64 / a.len() as f64 - j as f64 / b.len() as f64).abs());
        }
        d
    }

    // Plain and skipping Bd must agree on the whole distribution of steps and n_changes,
    // separately for runs ending in fixation and in extinction, and on rho. KS limit is
    // the 0.001 significance level. Print the table with:
    //   cargo test engines_agree -- --nocapture
    #[test]
    fn engines_agree_on_distributions() {
        let tree: Vec<_> = (1..15).map(|j| ((j - 1) / 2, j)).collect();
        let mut weighted = csr(8, &(0..8).flat_map(|i| [(i, (i + 1) % 8), (i, (i + 3) % 8)]).collect::<Vec<_>>());
        for k in 0..weighted.2.len() {
            weighted.2[k] = 1.0 + (k % 3) as f64;
        }
        let graphs = [
            ("undirected star N=10", csr(10, &undirected(&star(10))), 2.0),
            ("binary tree N=15", csr(15, &undirected(&tree)), 1.2),
            ("weighted directed N=8", weighted, 1.5),
        ];
        let reps = 20_000;
        println!(
            "\n{:<22} {:>4} {:<10} {:>13} {:>21} {:>19} {:>9} {:>9} {:>7}",
            "graph", "r", "ending", "runs", "mean steps", "mean n_changes", "KS steps", "KS chg", "limit"
        );
        println!("{:<42} {:>13} {:>21} {:>19}", "", "plain / skip", "plain / skip", "plain / skip");
        for (name, (p, i, w), r) in graphs {
            let [plain, skip] = [false, true].map(|s| simulate(&p, &i, &w, &[1], &[r], Rule::Bd, s, reps, 1 << 40, 11));
            for (ending, who) in [("fixation", 0), ("extinction", WILD_TYPE)] {
                let ((sa, ca), (sb, cb)) = (ending_in(&plain, who), ending_in(&skip, who));
                let limit = 1.95 * ((sa.len() + sb.len()) as f64 / (sa.len() * sb.len()) as f64).sqrt();
                let (ks_steps, ks_changes) = (ks(&sa, &sb), ks(&ca, &cb));
                println!(
                    "{name:<22} {r:>4} {ending:<10} {:>6} / {:<6} {:>9.1} / {:<9.1} {:>8.2} / {:<8.2} {ks_steps:>9.4} {ks_changes:>9.4} {limit:>7.4}",
                    sa.len(), sb.len(), mean(&sa), mean(&sb), mean(&ca), mean(&cb)
                );
                assert!(ks_steps < limit && ks_changes < limit, "{name}, {ending}: distributions differ");
            }
            let (pa, pb) = [&plain, &skip].map(|x| x.winner.iter().filter(|&&w| w == 0).count() as f64 / reps as f64).into();
            let se = (2.0 * pa * (1.0 - pa) / reps as f64).sqrt();
            assert!((pa - pb).abs() < 4.0 * se, "{name}: rho {pa} vs {pb}");
        }
    }
}

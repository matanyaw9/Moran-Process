use std::slice;
use std::time::Duration;

#[unsafe(no_mangle)]
extern "C" fn compute(
    n_nodes: u64,
    nbrs: *const u32,
    offsets: *const u32,
    r: f64,
    timeout_secs: f64,
    res: *mut f64,
    action: u8,
) {
    if !std::arch::is_x86_feature_detected!("bmi2") {
        panic!("CPU does not have pext instruction :(")
    }
    let nodes = unsafe { create_nodes(n_nodes, nbrs, offsets) };
    let res = unsafe { slice::from_raw_parts_mut(res, n_nodes as usize) };
    (match action {
        0 => solve::<false>,
        1 => solve::<true>,
        _ => panic!("bad action"),
    })(&nodes, r, res, Duration::from_secs_f64(timeout_secs))
}

/// Create nodes out of the specified graph given in `n_nodes`, `nbrs`, and
/// `offsets`.
///
/// # Safety
/// `nbrs` and `offsets` must be valid as per the conditions of `GraphCore`.
unsafe fn create_nodes(n_nodes: u64, nbrs: *const u32, offsets: *const u32) -> Box<[Node]> {
    let n_nodes = n_nodes as usize;
    let mut nodes = vec![Node { adjs: 0, vuln: 0.0 }; n_nodes].into_boxed_slice();
    // SAFETY: Caller guarantees pointer validity
    let offsets = unsafe { slice::from_raw_parts(offsets, n_nodes + 1) };
    let nbrs = unsafe { slice::from_raw_parts(nbrs, offsets[offsets.len() - 1] as usize) };

    for (i, &[x, y]) in offsets.array_windows().enumerate() {
        let strength = 1.0 / (y - x) as f64;
        for &nbr in &nbrs[x as usize..y as usize] {
            nodes[i].adjs |= 1 << nbr;
            nodes[nbr as usize].vuln += strength;
        }
    }
    nodes
}

/// Computes either the fixation probability, or abosorption time of the given
/// graph. Bails after `timeout_ns` nanoseconds if did not already converge on
/// a solution.
fn solve<const TIME: bool>(nodes: &[Node], r: f64, res: &mut [f64], timeout: Duration) {
    const MAX_STEPS: u64 = 500;

    debug_assert!(nodes.len() == res.len());

    let mut x = vec![0.0; 1 << nodes.len()].into_boxed_slice();
    let start = std::time::Instant::now();
    let mut steps = 0;
    loop {
        let change = gauss_seidel_step::<TIME>(nodes, r, &mut x);
        steps += 1;
        if steps >= MAX_STEPS
            || change < 3e-15 * 2.0f64.powi(nodes.len() as _)
            || start.elapsed() > timeout
        {
            break;
        }
    }
    for (i, r) in res.iter_mut().enumerate() {
        *r = x[1 << i];
    }
}

/// Run a single Gauss Seidel iteration through the vector `x`.
#[inline(always)]
fn gauss_seidel_step<const TIME: bool>(nodes: &[Node], r: f64, x: &mut [f64]) -> f64 {
    const OVER_RLX: f64 = 1.5;

    debug_assert!(1 << nodes.len() == x.len());

    x[0] = 0.0;
    x[x.len() - 1] = if TIME { 0.0 } else { 1.0 };

    let mut change = 0.0;
    for seg in [1..x.len() * 2 / 3, x.len() * 2 / 3 + 1..x.len()] {
        let mut weights = [0.0; _];
        for i in seg {
            let state = i ^ (i >> 1);
            adjust_neighbours(
                nodes,
                r,
                &mut weights,
                state as u64,
                i.trailing_zeros() as usize,
            );
            let w = &mut weights[..nodes.len()];
            let c = OVER_RLX
                * (w.iter()
                    .enumerate()
                    .map(|(i, p)| p * x[1 << i ^ state])
                    .sum::<f64>()
                    .algebraic_add(if TIME {
                        nodes.len() as f64 + (r - 1.0) * state.count_ones() as f64
                    } else {
                        0.0
                    })
                    / w.iter().sum::<f64>()
                    - x[state]);
            x[state] += c;
            change += c.abs();
        }
    }
    change
}

/// Assuming `weights[..nodes.len()]` stores the connection weights of
/// `state ^ 1 << idx`, adjusts `weights[..nodes.len()]` to store the
/// connection weights of `state`, using the graph information given in `nodes`
/// and `r`.
#[inline(always)]
fn adjust_neighbours(nodes: &[Node], r: f64, weights: &mut [f64; 64], state: u64, idx: usize) {
    debug_assert!(state < 1 << nodes.len());
    debug_assert!(idx < nodes.len());

    let Node { adjs: mut n, vuln } = nodes[idx];

    let epidemic = state >> idx & 1 != 0;
    let x = if epidemic { -1.0 } else { 1.0 } / n.count_ones() as f64;
    let y = -r * x;
    weights[idx] = if epidemic {
        vuln - weights[idx] / r
    } else {
        (vuln - weights[idx]) * r
    };
    // SAFETY: CPU feature BMI2 was checked for in main function
    let mut changes = unsafe { std::arch::x86_64::_pext_u64(state, n) };
    while n != 0 {
        weights[n.trailing_zeros() as usize] += if changes & 1 != 0 { x } else { y };
        changes >>= 1;
        n &= n - 1;
    }
}

#[derive(Clone, Copy)]
struct Node {
    /// bitboard of neighbouring nodes
    adjs: u64,
    /// ∑_{(v, u) ∈ V(G)} 1 / deg(v)
    vuln: f64,
}

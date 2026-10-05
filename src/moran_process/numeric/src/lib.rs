pub mod graph;
mod statespace;
use crate::graph::{Graph, Phase};
use crate::statespace::StateSpace;
use std::num::NonZero;
use std::slice;

#[unsafe(no_mangle)]
extern "C" fn compute(
    size: u64,
    nbrs: *const u32,
    offsets: *const u32,
    r: f32,
    res: *mut f32,
    thrds: u64,
) {
    let size = size as _;
    let thrds = thrds as _;

    // SAFETY: Caller responsible for passing legal inputs
    let g = unsafe { Graph::from_ffi(size, nbrs, offsets, r) };
    let res = unsafe { slice::from_raw_parts_mut(res, 3 * size) };
    crunch(&g, thrds, res)
}

/// Main computation function. Probabilities are written to `res[..g.len()]`,
/// average times to homogeneity to `res[g.len()..2 * g.len()]`, and average
/// times to fixation to `res[2 * g.len()..]`.
pub fn crunch(g: &Graph, thrds: usize, res: &mut [f32]) {
    assert!(res.len() == 3 * g.len());

    let thrds = match thrds {
        0 => std::thread::available_parallelism().map_or(1, NonZero::get),
        _ => thrds,
    };
    let mut space = StateSpace::new_with(g.len(), |i| match (1 << g.len()) - i {
        1 => [1.0, 0.0],
        _ => [0.0, 0.0],
    });
    let (p, t) = res.split_at_mut(g.len());
    let (t, ct) = t.split_at_mut(g.len());

    for phase in [Phase::First, Phase::Second] {
        let cruncher = || {
            let mut idxr = space.get_indexer();
            while let Some(mut i) = idxr {
                let chng = g.step_division(phase, &mut i);
                idxr = space.next_indexer(i, chng);
            }
        };
        std::thread::scope(|s| {
            for _ in 1..thrds {
                s.spawn(cruncher);
            }
            cruncher();
        });
        match phase {
            Phase::First => {
                let data = space.data();
                for (i, (p, t)) in p.iter_mut().zip(&mut *t).enumerate() {
                    [*p, *t] = data[1 << i];
                }
                for datum in space.data() {
                    datum[1] = 0.0;
                }
                space.reset_schedule();
            }
            Phase::Second => {
                let data = space.data();
                for (i, ct) in ct.iter_mut().enumerate() {
                    *ct = data[1 << i][1] / data[1 << i][0];
                }
            }
        }
    }
}

pub mod data;
pub mod graph;
pub mod schedule;
use crate::data::Data;
use crate::graph::Graph;
use crate::schedule::Schedule;
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

    let prob = &Data::new(g.len());
    let time = &Data::new(g.len());
    let schd = Schedule::new();
    let cruncher = || {
        let mut section = schd.first();
        while let Some(s) = section {
            let diff = g.step_division(prob, time, s);
            section = schd.next(s, diff);
        }
    };
    let thrds = match thrds {
        0 => std::thread::available_parallelism().map_or(1, NonZero::get),
        _ => thrds,
    };

    std::thread::scope(|s| {
        for _ in 1..thrds {
            s.spawn(cruncher);
        }
        cruncher();
    });

    let (p, t) = res.split_at_mut(g.len());
    let (t, ct) = t.split_at_mut(g.len());

    for (i, ((p, t), ct)) in p.iter_mut().zip(t).zip(ct).enumerate() {
        *p = prob.get(1 << i);
        *t = time.get(1 << i);
        *ct = f32::NAN; // TODO: compute me
    }
}

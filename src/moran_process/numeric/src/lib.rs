pub mod data;
pub mod graph;
pub mod schedule;
use crate::data::Data;
use crate::graph::Graph;
use crate::schedule::Schedule;
use std::slice;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Action {
    Prob,
    Time { cond: bool },
}

#[unsafe(no_mangle)]
extern "C" fn compute(
    size: u64,
    nbrs: *const u32,
    offsets: *const u32,
    r: f32,
    res: *mut f32,
    action: u64,
    thrds: u64,
) {
    let size = size as _;
    let action = match action {
        0 => Action::Prob,
        1 => Action::Time { cond: false },
        2 => Action::Time { cond: true },
        _ => panic!("bad action"),
    };
    let thrds = thrds as _;

    // SAFETY: Caller responsible for passing legal inputs
    let g = unsafe { Graph::from_ffi(size, nbrs, offsets, r) };
    let res = unsafe { slice::from_raw_parts_mut(res, size) };
    crunch(&g, action, thrds, res)
}

/// Main computation function, result is written to `res`.
pub fn crunch(g: &Graph, action: Action, thrds: usize, res: &mut [f32]) {
    assert!(res.len() == g.len());

    let x = &Data::new(g.len());
    let schd = Schedule::new();
    let cruncher = || {
        let mut section = schd.first();
        while let Some(s) = section {
            let diff = g.step_division(x, s, action);
            section = schd.next(s, diff);
        }
    };
    let thrds = match thrds {
        0 => std::thread::available_parallelism().map_or(1, |n| n.get()),
        _ => thrds,
    };

    std::thread::scope(|s| {
        for _ in 1..thrds {
            s.spawn(cruncher);
        }
        cruncher();
    });
    for (i, r) in res.iter_mut().enumerate() {
        *r = x.get(1 << i);
    }
}

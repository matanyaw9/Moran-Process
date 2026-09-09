pub mod data;
pub mod graph;
pub mod schedule;
use crate::data::Data;
use crate::graph::Graph;
use crate::schedule::Schedule;
use std::slice;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Action {
    FixationProb,
    AbsrobTime,
}

#[unsafe(no_mangle)]
extern "C" fn compute(
    size: usize,
    nbrs: *const u32,
    offsets: *const u32,
    r: f64,
    res: *mut f64,
    action: u8,
) {
    let action = match action {
        0 => Action::FixationProb,
        1 => Action::AbsrobTime,
        _ => panic!("bad action"),
    };

    // SAFETY: Caller responsible for passing legal inputs
    let g = unsafe { Graph::from_ffi(size, nbrs, offsets, r) };
    let res = unsafe { slice::from_raw_parts_mut(res, size) };
    crunch(&g, action, res)
}

pub fn crunch(g: &Graph, action: Action, res: &mut [f64]) {
    // TODO: get or calculate number of threads to use
    const THRDS: usize = 6;

    assert!(res.len() == g.len());

    let d = Data::new(g.len(), action);
    let x = d.reference();
    let schd = Schedule::new(3e-15 * 2f64.powi(g.len() as _));
    let cruncher = || {
        let mut section = schd.first();
        while let Some(s) = section {
            let change = g.step_division(x, s, action);
            section = schd.next(s, change);
        }
    };
    std::thread::scope(|s| {
        for _ in 1..THRDS {
            s.spawn(cruncher);
        }
        cruncher();
    });
    for (i, r) in res.iter_mut().enumerate() {
        *r = x.get(1 << i);
    }
}

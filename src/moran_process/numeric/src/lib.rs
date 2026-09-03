mod data;
mod graph;
use crate::data::Data;
use crate::graph::Graph;

#[unsafe(no_mangle)]
extern "C" fn compute(
    size: usize,
    nbrs: *const u32,
    offsets: *const u32,
    r: f64,
    res: *mut f64,
    action: u8,
) {
    // TODO: reintroduce action
    _ = action;

    // SAFETY: Caller responsible for passing legal inputs
    let g = unsafe { Graph::from_ffi(size, nbrs, offsets, r) };
    let d = Data::new(g.len());
    let x = d.reference();

    const MAX_STEPS: u64 = 500;
    for _ in 0..MAX_STEPS {
        let change = g.step_section(x, 0, g.len());
        if change < 3e-15 * 2f64.powi(g.len() as _) {
            break;
        }
    }

    for i in 0..g.len() {
        let val = x.get(1 << i);
        // SAFETY: Caller responsible for passing legal inputs
        unsafe {
            res.add(i).write(val);
        }
    }
}

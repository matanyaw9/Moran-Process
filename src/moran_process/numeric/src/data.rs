use super::Action;
use std::sync::atomic::{AtomicU64, Ordering};

pub struct Data(Box<[AtomicU64]>);

impl Data {
    /// Create a new state-space vector, initialised for the given action.
    pub fn new(size: usize, action: Action) -> Data {
        const ZERO: u64 = 0f64.to_bits();
        const ONE: u64 = 1f64.to_bits();
        Data(
            std::iter::repeat_n(ZERO, (1 << size) - 1)
                .chain([match action {
                    Action::FixationProb => ONE,
                    Action::AbsrobTime => ZERO,
                }])
                .map(AtomicU64::new)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        )
    }

    pub fn get(&self, idx: u64) -> f64 {
        f64::from_bits(self.0[idx as usize].load(Ordering::Relaxed))
    }

    pub fn set(&self, idx: u64, val: f64) {
        self.0[idx as usize].store(val.to_bits(), Ordering::Relaxed);
    }
}

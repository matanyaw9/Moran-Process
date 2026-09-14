use super::Action;
use std::sync::atomic::{AtomicU32, Ordering};

pub struct Data(Box<[AtomicU32]>);

impl Data {
    /// Create a new state-space vector, initialised for the given action.
    pub fn new(size: usize, action: Action) -> Data {
        const ZERO: u32 = 0f32.to_bits();
        const ONE: u32 = 1f32.to_bits();
        Data(
            std::iter::repeat_n(ZERO, (1 << size) - 1)
                .chain([match action {
                    Action::FixationProb => ONE,
                    Action::AbsrobTime => ZERO,
                }])
                .map(AtomicU32::new)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        )
    }

    pub fn get(&self, idx: u64) -> f32 {
        f32::from_bits(self.0[idx as usize].load(Ordering::Relaxed))
    }

    pub fn set(&self, idx: u64, val: f32) {
        self.0[idx as usize].store(val.to_bits(), Ordering::Relaxed);
    }
}

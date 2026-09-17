use std::sync::atomic::{AtomicU32, Ordering};

pub struct Data(Box<[AtomicU32]>);

impl Data {
    /// Create a new state-space vector, zero initialised
    pub fn new(size: usize) -> Data {
        Data(
            std::iter::repeat_n(0, 1 << size)
                .map(AtomicU32::new)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        )
    }

    pub fn clear(&mut self) {
        self.0.fill_with(|| AtomicU32::new(0));
    }

    pub fn get(&self, idx: u64) -> f32 {
        f32::from_bits(self.0[idx as usize].load(Ordering::Relaxed))
    }

    pub fn set(&self, idx: u64, val: f32) {
        self.0[idx as usize].store(val.to_bits(), Ordering::Relaxed);
    }
}

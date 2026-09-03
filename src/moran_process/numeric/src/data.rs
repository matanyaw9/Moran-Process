use std::sync::atomic::{AtomicU64, Ordering};

pub struct Data(Box<[AtomicU64]>);

#[derive(Clone, Copy)]
pub struct DataRef<'data>(&'data [AtomicU64]);

impl Data {
    pub fn new(size: usize) -> Data {
        let mut v = Vec::with_capacity(1 << size);
        v.resize_with(v.capacity() - 1, || AtomicU64::new(0));
        // TODO: this assumes we're computing fixation probability.
        // add a flag to choose beterrn fixation probability and
        // absorption time
        v.push(AtomicU64::new(1f64.to_bits()));
        Data(v.into_boxed_slice())
    }

    pub fn reference(&self) -> DataRef<'_> {
        DataRef(&self.0)
    }
}

impl<'data> DataRef<'data> {
    pub fn get(self, idx: u64) -> f64 {
        f64::from_bits(self.0[idx as usize].load(Ordering::Relaxed))
    }

    pub fn set(self, idx: u64, val: f64) {
        self.0[idx as usize].store(val.to_bits(), Ordering::Relaxed);
    }
}

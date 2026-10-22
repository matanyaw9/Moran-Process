use crate::graph::Change;
#[cfg(debug_assertions)]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::{
    marker::PhantomData,
    sync::{Condvar, Mutex, MutexGuard},
};

// TODO: see whether we can make the schedule lock-free

// TODO: make all `StateSpace`'s contents live on the heap, remove
// `next_indexer` and make it a consuming method on `Indexer`
#[repr(align(64))]
pub struct StateSpace<T> {
    /// This pointer is actually the guts of a `Box<[T]>`, we store it as raw
    /// mutable pointer to allow shared mutable access to (disjoint parts of)
    /// its contents, using more raw pointers.
    data: *mut [T],
    schedule: Mutex<Schedule>,
    no_work: Condvar,
    // No blanket `Send` and `Sync`, invariant on `T`.
    //
    // `StateSpace<T>` must be invariant on `T` since `&StateSpace<T>` is
    // covariant on `StateSpace<T>` and allows write access (by means of
    // `get_indexer`). https://counterexamples.org/general-covariance.html
}

impl<T> Drop for StateSpace<T> {
    fn drop(&mut self) {
        // SAFETY: `self.data` was created via `Box::into_raw`, no other
        // references to `self.data` exist at this point.
        drop(unsafe { Box::from_raw(self.data) });
    }
}

// SAFETY: `StateSpace<T>` acts as a container of `T`s; it therefore can only
// be moved across threads if the values themselves are allowed to.
unsafe impl<T: Send> Send for StateSpace<T> {}

// SAFETY: `&StateSpace<T>` provides access to `&mut T`s and `&T`s (via
// `get_indexer`), hence `&StateSpace<T>` is `Sync` iff both `T`s (`T: Send`)
// and `&T`s (`T: Sync`) can be moved accross threads.
unsafe impl<T: Send + Sync> Sync for StateSpace<T> {}

pub struct Indexer<'space, T> {
    // TODO: think whether we can relax the `Send`/`Sync` constraints
    data: *mut T,
    log2_div_len: u8,
    division: u8,
    _ph: PhantomData<&'space StateSpace<T>>,
    // `!Send`, `!Sync`, invariant on `T`.
}

pub struct Neighbours<'idxr, T> {
    data: *mut T,
    state: u64,
    curbit: u8,
    endbit: u8,
    _ph: PhantomData<&'idxr [T]>,
    // `!Send`, `!Sync`, invariant on `T`.
}

impl<T> StateSpace<T> {
    pub fn new_with(size: usize, f: impl FnMut(usize) -> T) -> Self {
        assert!(size >= 8);
        let data = Box::into_raw((0..1 << size).map(f).collect::<Vec<_>>().into_boxed_slice());
        Self {
            data,
            schedule: Mutex::new(Schedule {
                changes: 0b1111,
                sides: [Side {
                    epoch: 0,
                    queued: !0,
                    done: 0,
                }; _],
            }),
            no_work: Condvar::new(),
        }
    }

    pub fn data(&mut self) -> &mut [T] {
        // SAFETY: `self.data` points to a valid, initialised `[T]` of the
        // correct length. Since the only ways to get access to `self.data`'s
        // contents are through this method, and other methods that take
        // `&self`, we know no one else aliases the memory.
        unsafe { &mut *self.data }
    }

    pub fn indexer(&self) -> Option<Indexer<'_, T>> {
        self.create_indexer(self.schedule.lock().unwrap())
    }

    pub fn next_indexer(&self, idxr: Indexer<'_, T>, change: Change) -> Option<Indexer<'_, T>> {
        assert!(std::ptr::eq(idxr.data, self.data.cast()));

        let mut guard = self.schedule.lock().unwrap();
        let i = (idxr.division >= 0x80) as usize;
        guard.sides[i].done |= 1 << (idxr.division << 1 >> 2);

        if change == Change::Major {
            guard.changes |= 1 << i;
        }
        if guard.sides[i].done == !0 {
            #[cfg(debug_assertions)]
            println!(
                "({:0>4}, {:0>4}) significant changes {:0>4b}",
                guard.sides[0].epoch, guard.sides[1].epoch, guard.changes,
            );
            let bit = guard.changes & 1 << i;
            guard.changes &= 0b1010 >> i;
            guard.changes |= bit << 2;
            guard.sides[i] = Side {
                queued: !0,
                done: 0,
                epoch: guard.sides[i].epoch + 1,
            };
            self.no_work.notify_all();
        }
        self.create_indexer(guard)
    }

    fn create_indexer(&self, mut guard: MutexGuard<'_, Schedule>) -> Option<Indexer<'_, T>> {
        let (i, ctz) = 'outer: loop {
            if guard.changes == 0 {
                return None;
            }
            // attempt to take a queued task from a lagging/levelled side
            for i in [0, 1] {
                if guard.sides[i].epoch <= guard.sides[1 - i].epoch
                    && let Some(ctz) = guard.sides[i].queued.lowest_one()
                {
                    break 'outer (i, ctz);
                }
            }
            // also attempt to take a queued task from the leading side, if that
            // task does not depend on a different task from the lagging side
            for i in [0, 1] {
                if guard.sides[i].epoch == guard.sides[1 - i].epoch + 1
                    && let Some(ctz) =
                        (guard.sides[i].queued & guard.sides[1 - i].done).lowest_one()
                {
                    break 'outer (i, ctz);
                }
            }
            #[cfg(debug_assertions)]
            {
                static WAITS: AtomicUsize = AtomicUsize::new(0);
                println!("threadsleep {}", WAITS.fetch_add(1, Ordering::Relaxed));
            }
            guard = self.no_work.wait(guard).unwrap();
        };
        guard.sides[i].queued &= !(1 << ctz);
        let pairity = (ctz.count_ones() ^ guard.sides[i].epoch ^ i as u32) & 1;
        let log2_div_len = self.data.len().ilog2() as u8 - 8;
        let division = ((i as u32) << 7 | ctz << 1 | pairity) as u8;
        Some(Indexer {
            data: self.data.cast(),
            log2_div_len,
            division,
            _ph: PhantomData,
        })
    }
}

impl<'s, T> Indexer<'s, T> {
    pub fn div_idx(&self) -> u8 {
        self.division
    }

    pub fn div_len(&self) -> u64 {
        1 << self.log2_div_len
    }

    pub fn at(&mut self, idx: u64) -> &mut T {
        assert!(idx >> self.log2_div_len == 0);
        let idx = (self.division as usize) << self.log2_div_len ^ idx as usize;
        // SAFETY: `idx` was asserted to be within bounds
        unsafe { &mut *self.data.add(idx) }
    }

    pub fn neighbours(&mut self, idx: u64) -> Neighbours<'_, T> {
        assert!(idx >> self.log2_div_len == 0);
        Neighbours {
            data: self.data,
            state: idx | (self.division as u64) << self.log2_div_len,
            curbit: 0,
            endbit: self.log2_div_len + 8,
            _ph: PhantomData,
        }
    }
}

impl<'i, T> Iterator for Neighbours<'i, T> {
    type Item = &'i T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.curbit >= self.endbit {
            return None;
        }
        let state = self.state as usize ^ 1 << self.curbit;
        self.curbit += 1;
        // SAFETY: `self` was constructed after the state was asserted to be in
        // bounds
        Some(unsafe { &*self.data.add(state) })
    }
}

/// State of the yielded-out `Indexer`s.
struct Schedule {
    /// A 4-bit vector of the significance of change, of the two most recent
    /// epochs, in the two sides.
    /// - bit `0`: side `0`, current epoch
    /// - bit `1`: side `1`, current epoch
    /// - bit `2`: side `0`, previous epoch
    /// - bit `3`: side `1`, previous epoch
    changes: u8,
    sides: [Side; 2],
}

#[derive(Clone, Copy)]
struct Side {
    epoch: u32,
    queued: u64,
    done: u64,
    // invariants:
    //   queued ∩ done = ∅
    //   doneᶜ ≠ 0
}

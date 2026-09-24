use crate::graph::Change;
use std::{
    marker::PhantomData,
    ops::{Index, IndexMut},
    sync::{
        Condvar, Mutex, MutexGuard,
        atomic::{AtomicUsize, Ordering},
    },
};

// TODO: see whether we can make the schedule lock-free

#[repr(align(64))]
pub struct StateSpace<T> {
    data: Box<[T]>,
    schedule: Mutex<Schedule>,
    no_work: Condvar,

    // Remove blanket `Send` and `Sync`, make invariant on `T`.
    //
    // `StateSpace<T>` must be invariant on `T` since `&StateSpace<T>` is
    // covariant on `StateSpace<T>` and allows write access (by means of
    // `get_indexer`). If `StateSpace<T>` were covariant on `T`, one could:
    // `&StateSpace<Cat>` -> `&StateSpace<Animal>` -> `Indexer<'_, Animal>` and
    // store an animal in a state-space of cats.
    _ph: PhantomData<*mut T>,
}

// SAFETY: `StateSpace<T>` acts as a container of `T`s; it therefore can only
// be moved across threads if the values themselves are allowed to.
unsafe impl<T: Send> Send for StateSpace<T> {}

// SAFETY: `&StateSpace<T>` provides access to `&mut T`s and `&T`s (via
// `get_indexer`), hence `&StateSpace<T>` is `Sync` iff both `T`s (`T: Send`)
// and `&T`s (`T: Sync`) can be moved accross threads.
unsafe impl<T: Send + Sync> Sync for StateSpace<T> {}

pub struct Indexer<'s, T> {
    // `!Send`, `!Sync`, invariant on `T`.

    // TODO: think whether we can relax the `Send`/`Sync` constraints
    data: *mut T,
    log2_div_len: u8,
    division: u8,
    _ph: PhantomData<&'s StateSpace<T>>,
}

pub struct NeighbourIter<'i, T> {
    // `!Send`, `!Sync`, invariant on `T`.
    data: *mut T,
    state: u64,
    curbit: u8,
    endbit: u8,
    _ph: PhantomData<&'i [T]>,
}

impl<T> StateSpace<T> {
    pub fn new_with(size: usize, f: impl FnMut(usize) -> T) -> Self {
        assert!(size >= 8);
        Self {
            data: (0..1 << size).map(f).collect::<Vec<_>>().into_boxed_slice(),
            schedule: Mutex::new(Schedule {
                changes: 0b1111,
                sides: [Side {
                    epoch: 0,
                    queued: !0,
                    done: 0,
                }; _],
            }),
            no_work: Condvar::new(),
            _ph: PhantomData,
        }
    }

    pub fn reset_schedule(&mut self) {
        *self.schedule.get_mut().unwrap() = Schedule {
            changes: 0b1111,
            sides: [Side {
                epoch: 0,
                queued: !0,
                done: 0,
            }; _],
        };
    }

    pub fn all_data(&mut self) -> &mut [T] {
        &mut self.data
    }

    pub fn get_indexer(&self) -> Option<Indexer<'_, T>> {
        self.create_indexer(self.schedule.lock().unwrap())
    }

    pub fn next_indexer(&self, idxr: Indexer<'_, T>, change: Change) -> Option<Indexer<'_, T>> {
        assert!(std::ptr::eq(idxr.data, self.data.as_ptr()));

        let mut guard = self.schedule.lock().unwrap();
        let i = (idxr.division >= 0x80) as usize;
        guard.sides[i].done |= 1 << (idxr.division << 1 >> 2);

        if change == Change::Significant {
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
            // TODO: ensure writes via this pointer do not violate the aliasing
            // model
            data: self.data.as_ptr().cast_mut(),
            log2_div_len,
            division,
            _ph: PhantomData,
        })
    }
}

impl<'s, T> Index<u64> for Indexer<'s, T> {
    type Output = T;

    fn index(&self, idx: u64) -> &Self::Output {
        assert!(idx >> self.log2_div_len == 0);
        let idx = (self.division as usize) << self.log2_div_len ^ idx as usize;
        // SAFETY: `idx` was asserted to be within bounds
        unsafe { &*self.data.add(idx) }
    }
}

impl<'s, T> IndexMut<u64> for Indexer<'s, T> {
    fn index_mut(&mut self, idx: u64) -> &mut Self::Output {
        assert!(idx >> self.log2_div_len == 0);
        let idx = (self.division as usize) << self.log2_div_len ^ idx as usize;
        // SAFETY: `idx` was asserted to be within bounds
        unsafe { &mut *self.data.add(idx) }
    }
}

impl<'s, T> Indexer<'s, T> {
    pub fn div_idx(&self) -> u8 {
        self.division
    }

    pub fn div_size(&self) -> u64 {
        1 << self.log2_div_len
    }

    pub fn neighbours(&mut self, state: u64) -> NeighbourIter<'_, T> {
        assert!(state >> self.log2_div_len == 0);
        NeighbourIter {
            data: self.data,
            state: state | (self.division as u64) << self.log2_div_len,
            curbit: 0,
            endbit: self.log2_div_len + 8,
            _ph: PhantomData,
        }
    }
}

impl<'i, T> Iterator for NeighbourIter<'i, T> {
    type Item = &'i T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.curbit < self.endbit {
            let state = self.state as usize ^ 1 << self.curbit;
            self.curbit += 1;
            // SAFETY: `self` was constructed after the state was asserted to
            // be in bounds
            Some(unsafe { &*self.data.add(state) })
        } else {
            None
        }
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

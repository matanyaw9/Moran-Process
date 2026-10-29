use crate::graph::Change;
#[cfg(debug_assertions)]
use std::sync::atomic::AtomicUsize;
use std::{
    marker::PhantomData,
    sync::{
        Condvar, Mutex, MutexGuard,
        atomic::{AtomicBool, Ordering},
    },
};

#[repr(align(64))]
pub struct StateSpace<T> {
    /// This pointer is actually the guts of a `Box<[T]>`, we store it as raw
    /// mutable pointer to allow shared mutable access to (disjoint parts of)
    /// its contents, using more raw pointers.
    data: *mut [T],
    regimen: Regimen,
    // No blanket `Send` and `Sync`, invariant on `T`.
    //
    // `StateSpace<T>` must be invariant on `T` since `&StateSpace<T>` is
    // covariant on `StateSpace<T>` and allows write access (by means of
    // `get_indexer`). https://counterexamples.org/general-covariance.html
}

impl<T> StateSpace<T> {
    /// Create a new `StateSpace` for a graph of size `size`, initially
    /// populatedwith the values given by `f`.
    pub fn new_with(size: usize, f: impl FnMut(usize) -> T) -> Self {
        Self {
            data: Box::into_raw((0..1 << size).map(f).collect::<Vec<_>>().into_boxed_slice()),
            regimen: match size {
                ..MULTITHREAD_THRESHOLD => Regimen::SingleThread {
                    began: AtomicBool::new(false),
                },
                MULTITHREAD_THRESHOLD.. => Regimen::MultiThread {
                    dispatcher: Mutex::new(Dispatcher {
                        changes: 0b1111,
                        epochs: [0; _],
                        queued: [!0; _],
                        done: [0; _],
                    }),
                    no_work: Condvar::new(),
                },
            },
        }
    }

    pub fn data(&mut self) -> &mut [T] {
        // SAFETY: `self.data` points to a valid, initialised `[T]` of the
        // correct length. Since the only ways to get access to `self.data`'s
        // contents are through this method, and other methods that take
        // `&self`, we know no one else aliases the memory.
        unsafe { &mut *self.data }
    }

    /// Request an `Indexer` to go thorugh.
    pub fn indexer(&self) -> Option<Indexer<'_, T>> {
        match &self.regimen {
            Regimen::SingleThread { began } => {
                (!began.swap(true, Ordering::Relaxed)).then(|| Indexer {
                    data: self.data.cast(),
                    log2_div_len: self.data.len().ilog2() as u8,
                    division: Division::Whole,
                    _ph: PhantomData,
                })
            }
            Regimen::MultiThread {
                dispatcher,
                no_work,
            } => self.create_indexer(dispatcher.lock().unwrap(), no_work),
        }
    }

    /// Report that an `Indexer` has been iterated through, request a new
    /// `Indexer`.
    pub fn next_indexer(&self, idxr: Indexer<'_, T>, change: Change) -> Option<Indexer<'_, T>> {
        assert!(std::ptr::eq(idxr.data, self.data.cast()));
        match &self.regimen {
            Regimen::SingleThread { .. } => (change == Change::Major).then(|| Indexer {
                data: self.data.cast(),
                log2_div_len: self.data.len().ilog2() as u8,
                division: Division::Whole,
                _ph: PhantomData,
            }),
            Regimen::MultiThread {
                dispatcher,
                no_work,
            } => {
                let mut guard = dispatcher.lock().unwrap();
                let side = match idxr.division {
                    Division::Whole => unreachable!(),
                    Division::Part(..0x80) => 0,
                    Division::Part(0x80..) => 1,
                };
                guard.done[side] |= 1
                    << match idxr.division {
                        Division::Whole => unreachable!(),
                        Division::Part(div) => div << 1 >> 2,
                    };

                if change == Change::Major {
                    guard.changes |= 1 << side;
                }
                if guard.done[side] == !0 {
                    #[cfg(debug_assertions)]
                    println!(
                        "({:0>4}, {:0>4}) significant changes {:0>4b}",
                        guard.epochs[0], guard.epochs[1], guard.changes,
                    );
                    let bit = guard.changes & 1 << side;
                    guard.changes &= 0b1010 >> side;
                    guard.changes |= bit << 2;
                    guard.epochs[side] += 1;
                    guard.queued[side] = !0;
                    guard.done[side] = 0;
                    no_work.notify_all();
                }
                self.create_indexer(guard, no_work)
            }
        }
    }

    fn create_indexer(
        &self,
        mut guard: MutexGuard<'_, Dispatcher>,
        no_work: &Condvar,
    ) -> Option<Indexer<'_, T>> {
        let (i, ctz) = 'outer: loop {
            if guard.changes == 0 {
                return None;
            }
            // attempt to take a queued task from a lagging/levelled side
            for i in [0, 1] {
                if guard.epochs[i] <= guard.epochs[1 - i]
                    && let Some(ctz) = guard.queued[i].lowest_one()
                {
                    break 'outer (i, ctz);
                }
            }
            // also attempt to take a queued task from the leading side, if that
            // task does not depend on a different task from the lagging side
            for i in [0, 1] {
                if guard.epochs[i] == guard.epochs[1 - i] + 1
                    && let Some(ctz) = (guard.queued[i] & guard.done[1 - i]).lowest_one()
                {
                    break 'outer (i, ctz);
                }
            }
            #[cfg(debug_assertions)]
            {
                static WAITS: AtomicUsize = AtomicUsize::new(0);
                println!("threadsleep {}", WAITS.fetch_add(1, Ordering::Relaxed));
            }
            guard = no_work.wait(guard).unwrap();
        };
        guard.queued[i] &= !(1 << ctz);
        let pairity = (ctz.count_ones() ^ guard.epochs[i] ^ i as u32) & 1;
        Some(Indexer {
            data: self.data.cast(),
            log2_div_len: self.data.len().ilog2() as u8 - 8,
            division: Division::Part(((i as u32) << 7 | ctz << 1 | pairity) as u8),
            _ph: PhantomData,
        })
    }
}

pub struct Indexer<'space, T> {
    data: *mut T,
    log2_div_len: u8,
    division: Division,
    _ph: PhantomData<&'space StateSpace<T>>,
    // No blanket `Send` and `Sync`, invariant on `T`.
}

impl<'s, T> Indexer<'s, T> {
    /// Returns whether the mutant extinction state included in this
    /// `Indexer`'s range.
    pub fn first(&self) -> bool {
        matches!(self.division, Division::Whole | Division::Part(0))
    }

    /// Returns whether the mutant fixation state included in this `Indexer`'s
    /// range.
    pub fn last(&self) -> bool {
        matches!(self.division, Division::Whole | Division::Part(0xff))
    }

    pub fn topbits(&self) -> u64 {
        match self.division {
            Division::Part(div) => (div as u64) << self.log2_div_len,
            Division::Whole => 0,
        }
    }

    pub fn len(&self) -> u64 {
        1 << self.log2_div_len
    }

    /// Access the `idx`th entry of this `Indexer`. Importantly, the entry at
    /// index `0` is the first state *within this `Indexer`'s range*, and *is
    /// not* necessarily the no-mutants state.
    pub fn at(&mut self, idx: u64) -> &mut T {
        assert!(idx >> self.log2_div_len == 0);
        let idx = (idx | self.topbits()) as _;
        // SAFETY: `idx` was asserted to be within bounds
        unsafe { &mut *self.data.add(idx) }
    }

    /// An iterator over all neighbouring states of `self.at(idx)`.
    pub fn neighbours(&self, idx: u64) -> Neighbours<'_, T> {
        assert!(idx >> self.log2_div_len == 0);
        Neighbours {
            data: self.data,
            state: idx | self.topbits(),
            curbit: 0,
            endbit: self.log2_div_len
                + match self.division {
                    Division::Part(_) => 8,
                    Division::Whole => 0,
                },
            _ph: PhantomData,
        }
    }
}

pub struct Neighbours<'idxr, T> {
    data: *const T,
    state: u64,
    curbit: u8,
    endbit: u8,
    _ph: PhantomData<&'idxr Indexer<'idxr, T>>,
    // No blanket `Send` and `Sync`, covariant on `T`.
}

impl<'i, T> Iterator for Neighbours<'i, T> {
    type Item = &'i T;

    fn next(&mut self) -> Option<Self::Item> {
        (self.curbit < self.endbit).then(|| {
            let state = self.state as usize ^ 1 << self.curbit;
            self.curbit += 1;
            // SAFETY: `self` was constructed after the state was asserted to be in
            // bounds
            unsafe { &*self.data.add(state) }
        })
    }
}

const MULTITHREAD_THRESHOLD: usize = 16;

enum Regimen {
    SingleThread {
        began: AtomicBool,
    },
    MultiThread {
        dispatcher: Mutex<Dispatcher>,
        no_work: Condvar,
    },
}

/// Management value keeping track of the state of the yielded-out `Indexer`s.
struct Dispatcher {
    /// A 4-bit vector of the significance of change, of the two most recent
    /// epochs, in the two sides.
    /// - bit `0`: side `0`, current epoch
    /// - bit `1`: side `1`, current epoch
    /// - bit `2`: side `0`, previous epoch
    /// - bit `3`: side `1`, previous epoch
    changes: u8,
    epochs: [u32; 2],
    /// Invariants:
    ///   queued[0] ∩ done[0] = ∅ ∧ queued[1] ∩ done[1] = ∅
    queued: [u64; 2],
    /// Invariants:
    ///   done[0] ≠ U ∧ done[1] ≠ U
    done: [u64; 2],
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Division {
    Whole,
    Part(u8),
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

// SAFETY: `&mut Indexer<'_, T>` provides access both to `&mut T` and `&T`, so
// so `T` needs to be both `Send` and `Sync` for us to be able pass
// `Indexer<'_, T>`s between threads.
unsafe impl<T: Send + Sync> Send for Indexer<'_, T> {}

// SAFETY: `&Indexer<'_, T>` provides access to `&T`s, the latter therefore
// need to be thread-safe for the former to be.
unsafe impl<T: Sync> Sync for Indexer<'_, T> {}

// SAFETY: `&mut Neighbours<'_, T>` provides access to `&T` - hence `T` must be
// `Sync` for `Neighbours<'_, T>` to be `Send`.
unsafe impl<T: Sync> Send for Neighbours<'_, T> {}

// SAFETY: `&Neighbours<'_, T>` is useless and provides no access to anything.
unsafe impl<T> Sync for Neighbours<'_, T> {}

use super::graph::Change;
#[cfg(debug_assertions)]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, MutexGuard};

// TODO: see whether we can make the schedule lock-free

pub struct Schedule(Mutex<WorkState>, Condvar);

struct WorkState {
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
    //   done ≠ u64::MAX
}

impl Schedule {
    pub fn new() -> Schedule {
        Schedule(
            Mutex::new(WorkState {
                changes: 0b1111,
                sides: [Side {
                    epoch: 0,
                    queued: u64::MAX,
                    done: 0,
                }; _],
            }),
            Condvar::new(),
        )
    }

    /// Ask the scheduler for a new division to compute. A result of `None`
    /// indicates that the computation is already complete.
    pub fn first(&self) -> Option<u8> {
        let guard = self.0.lock().unwrap();
        self.get_division(guard)
    }

    /// Ask the scheduler for a division to compute, providing the previously
    /// computed division, and whether any entry wherein was significantly
    /// changed. A result of `None` indicates that the computation is already
    /// complete.
    pub fn next(&self, prev: u8, change: Change) -> Option<u8> {
        let mut guard = self.0.lock().unwrap();

        let i = (prev >= 0x80) as usize;
        guard.sides[i].done |= 1 << (prev << 1 >> 2);

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
            (guard.sides[i].queued, guard.sides[i].done) = (!0, 0);
            guard.sides[i].epoch += 1;
            self.1.notify_all();
        }
        self.get_division(guard)
    }

    fn get_division(&self, mut guard: MutexGuard<'_, WorkState>) -> Option<u8> {
        while guard.changes != 0 {
            // attempt to take a queued task from a lagging/levelled side
            for i in [0, 1] {
                if guard.sides[i].epoch <= guard.sides[1 - i].epoch
                    && let Some(ctz) = guard.sides[i].queued.lowest_one()
                {
                    guard.sides[i].queued &= !(1 << ctz);
                    let pairity = (ctz.count_ones() ^ guard.sides[i].epoch ^ i as u32) & 1;
                    return Some(((i as u32) << 7 | ctz << 1 | pairity) as u8);
                }
            }
            // also attempt to take a queued task from the leading side, if that
            // task does not depend on a different task from the lagging side
            for i in [0, 1] {
                if guard.sides[i].epoch == guard.sides[1 - i].epoch + 1
                    && let Some(ctz) =
                        (guard.sides[i].queued & guard.sides[1 - i].done).lowest_one()
                {
                    guard.sides[i].queued &= !(1 << ctz);
                    let pairity = (ctz.count_ones() ^ guard.sides[i].epoch ^ i as u32) & 1;
                    return Some(((i as u32) << 7 | ctz << 1 | pairity) as u8);
                }
            }
            #[cfg(debug_assertions)]
            {
                static WAITS: AtomicUsize = AtomicUsize::new(0);
                println!("threadsleep {}", WAITS.fetch_add(1, Ordering::Relaxed));
            }
            guard = self.1.wait(guard).unwrap();
        }
        None
    }
}

impl Default for Schedule {
    fn default() -> Self {
        Self::new()
    }
}

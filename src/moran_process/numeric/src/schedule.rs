use std::sync::{Condvar, Mutex, MutexGuard};

// TODO: see whether we can make the schedule lock-free

pub struct Schedule(Mutex<WorkState>, Condvar);

struct WorkState {
    cease: bool,
    sides: [Side; 2],
}

#[derive(Clone, Copy)]
struct Side {
    ulp_diffs: [u32; 2],
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
                cease: false,
                sides: [Side {
                    ulp_diffs: [!0; 2],
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
    /// computed division, and the ULP change of the maximally-changed scalar.
    /// A result of `None` indicates that the computation is already complete.
    pub fn next(&self, prev: u8, diff: u32) -> Option<u8> {
        let mut guard = self.0.lock().unwrap();

        let i = (prev >= 0x80) as usize;
        guard.sides[i].done |= 1 << ((prev >> 1) & 0x3f);
        let c = &mut guard.sides[i].ulp_diffs[1];
        *c = diff.max(*c);
        if guard.sides[i].done == u64::MAX {
            let diff = (0..=1)
                .flat_map(|i| guard.sides[i].ulp_diffs)
                .max()
                .unwrap();
            guard.cease = diff <= 0x3f;
            guard.sides[i].ulp_diffs = [guard.sides[i].ulp_diffs[1], 0];
            (guard.sides[i].queued, guard.sides[i].done) = (u64::MAX, 0);
            guard.sides[i].epoch += 1;
            self.1.notify_all();
        }
        self.get_division(guard)
    }

    fn get_division(&self, mut guard: MutexGuard<'_, WorkState>) -> Option<u8> {
        while !guard.cease {
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

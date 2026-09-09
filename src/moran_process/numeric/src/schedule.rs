use std::sync::{Condvar, Mutex, MutexGuard};

// TODO: see whether we can make the schedule lock-free
pub struct Schedule(Mutex<WorkState>, Condvar);

struct WorkState {
    cease: bool,
    threshold: f64,
    sides: [Side; 2],
}

#[derive(Clone, Copy)]
struct Side {
    changes: [f64; 2],
    epoch: u32,
    queued: u64,
    done: u64,
    // invariants:
    //   queued ∩ done = ∅
    //   done ≠ u64::MAX
}

impl Schedule {
    pub fn new(threshold: f64) -> Schedule {
        Schedule(
            Mutex::new(WorkState {
                cease: false,
                threshold,
                sides: [Side {
                    changes: [f64::MAX / 4.0; 2],
                    epoch: 0,
                    queued: u64::MAX,
                    done: 0,
                }; _],
            }),
            Condvar::new(),
        )
    }

    pub fn first(&self) -> Option<u8> {
        let guard = self.0.lock().unwrap();
        self.get_section(guard)
    }

    pub fn next(&self, prev: u8, change: f64) -> Option<u8> {
        let mut guard = self.0.lock().unwrap();

        let i = (prev >= 0x80) as usize;
        guard.sides[i].done |= 1 << (prev >> 1);
        guard.sides[i].changes[1] += change;
        if guard.sides[i].done == u64::MAX {
            let tot_change = guard.sides[0]
                .changes
                .iter()
                .chain(&guard.sides[1].changes)
                .sum::<f64>();
            guard.cease = tot_change <= guard.threshold;
            guard.sides[i].changes = [guard.sides[i].changes[1], 0.0];
            (guard.sides[i].queued, guard.sides[i].done) = (u64::MAX, 0);
            guard.sides[i].epoch += 1;
            self.1.notify_all();
        }
        self.get_section(guard)
    }

    fn get_section(&self, mut guard: MutexGuard<'_, WorkState>) -> Option<u8> {
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

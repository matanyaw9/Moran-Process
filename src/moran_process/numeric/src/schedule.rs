use std::sync::{Condvar, Mutex, MutexGuard};

pub struct Schedule(Mutex<WorkState>, Condvar);

struct WorkState {
    cease: bool,
    threshold: f64,

    l_changes: [f64; 2],
    l_epoch: u32,
    l_queued: u64,
    l_done: u64,

    r_changes: [f64; 2],
    r_epoch: u32,
    r_queued: u64,
    r_done: u64,
    // invariants:
    //   r_queued ∩ r_done = ∅
    //   l_queued ∩ l_done = ∅
}

impl Schedule {
    pub fn new(threshold: f64) -> Schedule {
        Schedule(
            Mutex::new(WorkState {
                cease: false,
                threshold,
                l_changes: [f64::MAX / 4.0; 2],
                l_epoch: 0,
                l_queued: !0,
                l_done: 0,
                r_changes: [f64::MAX / 4.0; 2],
                r_epoch: 0,
                r_queued: !0,
                r_done: 0,
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

        // TODO: DRY this
        if prev < 0x80 {
            guard.l_done |= 1 << (prev >> 1);
            guard.l_changes[1] += change;
            if guard.l_done == !0 {
                let tot_change = guard.l_changes.iter().chain(&guard.r_changes).sum::<f64>();
                guard.cease = tot_change <= guard.threshold;
                guard.l_changes = [guard.l_changes[1], 0.0];
                (guard.l_queued, guard.l_done) = (!0, 0);
                guard.l_epoch += 1;
                self.1.notify_all();
            }
        } else {
            guard.r_done |= 1 << (prev >> 1 & 0x3f);
            guard.r_changes[1] += change;
            if guard.r_done == !0 {
                let tot_change = guard.l_changes.iter().chain(&guard.r_changes).sum::<f64>();
                guard.cease = tot_change <= guard.threshold;
                guard.r_changes = [guard.r_changes[1], 0.0];
                (guard.r_queued, guard.r_done) = (!0, 0);
                guard.r_epoch += 1;
                self.1.notify_all();
            }
        }
        self.get_section(guard)
    }

    fn get_section(&self, mut guard: MutexGuard<'_, WorkState>) -> Option<u8> {
        loop {
            if guard.cease {
                break None;
            }
            // TODO: DRY this
            if guard.l_epoch <= guard.r_epoch
                && let Some(ctz) = guard.l_queued.lowest_one()
            {
                guard.l_queued &= !(1 << ctz);
                let pairity = (ctz.count_ones() ^ guard.l_epoch) & 1;
                break Some((ctz << 1 | pairity) as u8);
            }
            if guard.r_epoch <= guard.l_epoch
                && let Some(ctz) = guard.r_queued.lowest_one()
            {
                guard.r_queued &= !(1 << ctz);
                let pairity = (ctz.count_ones() ^ guard.r_epoch ^ 1) & 1;
                break Some((0x80 | ctz << 1 | pairity) as u8);
            }
            if guard.l_epoch == guard.r_epoch + 1
                && let Some(ctz) = (guard.l_queued & guard.r_done).lowest_one()
            {
                guard.l_queued &= !(1 << ctz);
                let pairity = (ctz.count_ones() ^ guard.l_epoch) & 1;
                break Some((ctz << 1 | pairity) as u8);
            }
            if guard.r_epoch == guard.l_epoch + 1
                && let Some(ctz) = (guard.r_queued & guard.l_done).lowest_one()
            {
                guard.r_queued &= !(1 << ctz);
                let pairity = (ctz.count_ones() ^ guard.r_epoch ^ 1) & 1;
                break Some((0x80 | ctz << 1 | pairity) as u8);
            }
            guard = self.1.wait(guard).unwrap();
        }
    }
}

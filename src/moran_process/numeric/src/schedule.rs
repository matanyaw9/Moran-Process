use std::sync::Mutex;

// TODO: this whole implementation is smelly
// and bad and ugly and should be reworked.

pub struct Schedule(Mutex<Inner>);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Portion {
    pub start: u64,
    pub bits: usize,
}

struct Inner {
    done: bool,
    threshold: f64,
    graph_size: usize,
    epoch: usize,
    changes: [f64; 4],
    workers: Box<[Elem]>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Elem {
    Layer0,
    Layer0inFlight,
    Layer1,
    Layer1inFlight,
    Waiting,
}

impl Schedule {
    pub fn new(threshold: f64, graph_size: usize, work_slots: usize) -> Schedule {
        assert!(work_slots.is_power_of_two());
        assert!(graph_size >= work_slots.ilog2() as usize + 2);

        Schedule(Mutex::new(Inner {
            done: false,
            threshold,
            graph_size,
            epoch: 0,
            changes: [f64::MAX / 4.0; _],
            workers: vec![Elem::Layer0; work_slots].into_boxed_slice(),
        }))
    }

    pub fn first(&self) -> Option<Portion> {
        loop {
            let mut inner = self.0.lock().unwrap();
            if inner.done {
                return None;
            }

            for (i, elem) in inner.workers.iter().enumerate() {
                if let Elem::Layer0 = elem {
                    inner.workers[i] = Elem::Layer0inFlight;
                    let task = inner.epoch * inner.workers.len() + i;
                    return Some(task_to_portion(task, inner.graph_size, inner.workers.len()));
                }
            }
            for (i, elem) in inner.workers.iter().enumerate() {
                if let Elem::Layer1 = elem {
                    inner.workers[i] = Elem::Layer1inFlight;
                    let task = (inner.epoch + 1) * inner.workers.len() + i;
                    return Some(task_to_portion(task, inner.graph_size, inner.workers.len()));
                }
            }
            drop(inner);
            // TODO: perhaps we can wake the threads on demand?
            println!("threadsleep");
            std::thread::yield_now();
        }
    }

    pub fn next(&self, prev: Portion, change: f64) -> Option<Portion> {
        let mut inner = self.0.lock().unwrap();
        if inner.done {
            return None;
        }

        inner.changes[3] += change;
        let worker_slots = inner.workers.len();
        match &mut inner.workers[portion_to_idx(prev, worker_slots)] {
            el @ Elem::Layer0inFlight => *el = Elem::Layer1,
            el @ Elem::Layer1inFlight => *el = Elem::Waiting,
            _ => unreachable!(),
        }

        for &elem in &inner.workers {
            if let Elem::Layer0 | Elem::Layer0inFlight = elem {
                drop(inner);
                return self.first();
            }
        }

        if inner.changes.iter().sum::<f64>() < inner.threshold {
            inner.done = true;
            return None;
        }
        inner.epoch += 1;
        inner.changes.rotate_left(1);
        inner.changes[3] = 0.0;
        for elem in &mut inner.workers {
            *elem = match elem {
                Elem::Layer0 | Elem::Layer0inFlight => unreachable!(),
                Elem::Layer1 => Elem::Layer0,
                Elem::Layer1inFlight => Elem::Layer0inFlight,
                Elem::Waiting => Elem::Layer1,
            };
        }
        drop(inner);
        self.first()
    }
}

fn task_to_portion(task: usize, graph_size: usize, work_slots: usize) -> Portion {
    let bits = work_slots.ilog2() as usize + 2;
    let rem = task as u64 % (1 << bits);
    Portion {
        start: ((rem << 1) % (1 << bits) + (rem.count_ones() as u64 & 1)) << (graph_size - bits),
        bits: graph_size - bits,
    }
}

fn portion_to_idx(p: Portion, work_slots: usize) -> usize {
    (p.start as usize >> p.bits >> 1) % work_slots
}

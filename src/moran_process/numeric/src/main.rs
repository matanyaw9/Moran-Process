mod graph;
mod parse;
mod statespace;
use crate::graph::Graph;
use crate::statespace::StateSpace;

macro_rules! abort {
    ($($arg:tt)*) => {{
        eprintln!($($arg)*);
        std::process::exit(1)
    }};
}

pub fn main() -> std::io::Result<()> {
    let mut parser = parse::Parser::new();
    loop {
        let line = match parser.next() {
            Ok(None) => break,
            Ok(Some(line)) => line,
            Err(parse::Error::Io(err)) => abort!("could not read line: {err}"),
            Err(parse::Error::Parse(pos)) => abort!("parsing error at position {pos}"),
        };
        let g = match Graph::new(line.adjs, line.r) {
            Ok(g) => g,
            Err(graph::CreationError::NonexistentNode) => {
                abort!("adjacency matrix refers to a non-existent node")
            }
            Err(graph::CreationError::NotStronglyConnected) => {
                abort!("adjacency matrix is not strongly-connected")
            }
        };
        if line
            .report
            .as_deref()
            .unwrap_or(&[])
            .iter()
            .any(|&rep| rep >= 1 << g.len())
        {
            abort!("bad requested report state")
        }
        let thrds = match line.threads {
            None => std::thread::available_parallelism().map_or(1, std::num::NonZero::get),
            Some(t) => t,
        };
        let res = crunch(&g, thrds, line.report.as_deref());
        print!("[");
        if let &[first, ref rest @ ..] = &res[..] {
            print!("{}", first);
            for res in rest {
                print!(",{}", res);
            }
        }
        println!("]");
    }
    Ok(())
}

/// Main computation function. Return value is a list of (fixation probability,
/// absorption time, fixation time) triples. If `reports` is `Some`, index `n`
/// of the output corresponds to the state at `retports.unwrap()[n]`,
/// otherwise, return value is of length `g.len()`, and contains values for
/// states `[1, 2, 4, ..., 2^(g.len() - 1)]`.
fn crunch(g: &Graph, thrds: usize, reports: Option<&[u64]>) -> Vec<Result> {
    let mut space = StateSpace::new_with(g.len(), |i| {
        [(i == (1 << g.len()) - 1) as u32 as f32, 0.0, 0.0]
    });

    std::thread::scope(|s| {
        let cruncher = || {
            let mut idxr = space.indexer();
            while let Some(mut i) = idxr {
                let chng = g.step_division(&mut i);
                idxr = space.next_indexer(i, chng);
            }
        };
        for _ in 1..thrds {
            s.spawn(cruncher);
        }
        cruncher();
    });

    let data = space.data();
    reports
        .unwrap_or(&[])
        .iter()
        .copied()
        .chain((0..if reports.is_none() { g.len() } else { 0 }).map(|i| 1 << i))
        .map(|i| {
            let datum = data[i as usize];
            Result {
                prob: datum[0],
                time: datum[1],
                ctime: datum[2] / datum[0],
            }
        })
        .collect::<Vec<_>>()
}

#[derive(Clone, Copy)]
struct Result {
    prob: f32,
    time: f32,
    ctime: f32,
}

impl std::fmt::Display for Result {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{{\"prob\":{},\"time\":{},\"ctime\":{}}}",
            self.prob, self.time, self.ctime
        )
    }
}

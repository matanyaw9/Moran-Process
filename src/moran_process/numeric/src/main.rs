use numeric::graph::{Graph, Shape};

macro_rules! abort {
    ($($arg:tt)*) => {{
        eprintln!($($arg)*);
        std::process::exit(1)
    }};
}

pub fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    let args = args.iter().map(String::as_str).collect::<Vec<_>>();

    let g = match args[..] {
        [_, shape @ ("complete" | "cycle" | "star" | "tree"), size, r] => {
            let shape = match shape {
                "complete" => Shape::Complete,
                "cycle" => Shape::Cycle,
                "star" => Shape::Star,
                "tree" => Shape::Tree,
                _ => unreachable!(),
            };
            let size = size.parse::<usize>().unwrap_or_else(|_| abort!("bad size"));
            let r = r.parse::<f32>().unwrap_or_else(|_| abort!("bad r"));
            Graph::from_shape(size, shape, r)
        }
        [_, "--file", filepath, r] => {
            let r = r.parse::<f32>().unwrap_or_else(|_| abort!("bad r"));
            let text =
                std::fs::read_to_string(filepath).unwrap_or_else(|_| abort!("could not read file"));
            Graph::from_text(&text, r).unwrap_or_else(|| abort!("bad file contents"))
        }
        _ => abort!(
            "Usage: {} ((complete | cycle | star | tree) <size> | --file <pathname>) <r>",
            args[0]
        ),
    };

    let mut res = vec![0.0; 3 * g.len()];
    numeric::crunch(&g, 0, &mut res);
    println!("prob            time            ctime");
    for i in 0..g.len() {
        println!(
            "{: <16}{: <16}{}",
            res[i],
            res[i + g.len()],
            res[i + 2 * g.len()]
        );
    }
    println!(
        "means\n{: <16}{: <16}{}",
        res[..g.len()].iter().sum::<f32>() / g.len() as f32,
        res[g.len()..2 * g.len()].iter().sum::<f32>() / g.len() as f32,
        res[2 * g.len()..].iter().sum::<f32>() / g.len() as f32
    );
}

use numeric::graph::{Graph, Shape};

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
            let Ok(size) = size.parse::<usize>() else {
                badexit("bad size")
            };
            let Ok(r) = r.parse::<f32>() else {
                badexit("bad r")
            };
            Graph::from_shape(size, shape, r)
        }
        [_, "--file", filepath, r] => {
            let Ok(r) = r.parse::<f32>() else {
                badexit("bad r")
            };
            let Ok(text) = std::fs::read_to_string(filepath) else {
                badexit("could not read file")
            };
            let Some(g) = Graph::from_text(&text, r) else {
                badexit("bad file contents")
            };
            g
        }
        _ => badexit(&format!(
            "Usage: {} ((complete | cycle | star | tree) <size> | --file <pathname>) <r>",
            args[0]
        )),
    };

    let mut res = vec![0.0; 3 * g.len()];
    numeric::crunch(&g, 0, &mut res);
    println!("prob    \ttime    \tctime");
    for ((&p, &t), &ct) in res[..g.len()]
        .iter()
        .zip(&res[g.len()..2 * g.len()])
        .zip(&res[2 * g.len()..])
    {
        println!("{p}\t{t}\t{ct}");
    }
}

fn badexit(msg: &str) -> ! {
    eprintln!("{}", msg);
    std::process::exit(1)
}

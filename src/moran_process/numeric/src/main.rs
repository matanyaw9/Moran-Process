use numeric::Action;
use numeric::graph::{Graph, Shape};

pub fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    let args = args.iter().map(String::as_str).collect::<Vec<_>>();

    let (action, g) = match args[..] {
        [
            _,
            action @ ("prob" | "time"),
            shape @ ("complete" | "cycle" | "star" | "tree"),
            size,
            r,
        ] => {
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
            (action, Graph::from_shape(size, shape, r))
        }
        [_, action @ ("prob" | "time"), "--file", filepath, r] => {
            let Ok(r) = r.parse::<f32>() else {
                badexit("bad r")
            };
            let Ok(text) = std::fs::read_to_string(filepath) else {
                badexit("could not read file")
            };
            let Some(g) = Graph::from_text(&text, r) else {
                badexit("bad file contents")
            };
            (action, g)
        }
        _ => badexit(&format!(
            "Usage: {} (prob | time) ((complete | cycle | star | tree) <size> | --file <pathname>) <r>",
            args[0]
        )),
    };
    let action = match action {
        "prob" => Action::FixationProb,
        "time" => Action::AbsrobTime,
        _ => unreachable!(),
    };

    let mut res = vec![0.0; g.len()];
    numeric::crunch(&g, action, 0, &mut res);
    for r in res {
        println!("{r}");
    }
}

fn badexit(msg: &str) -> ! {
    eprintln!("{}", msg);
    std::process::exit(1)
}

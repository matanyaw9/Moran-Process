use numeric::Action;
use numeric::graph::{Graph, Shape};

pub fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    let [_, action, shape, size, r] = &args[..] else {
        eprintln!("Usage: {} <action> <shape> <size> <r>", args[0]);
        return;
    };
    let action = match action.as_str() {
        "prob" => Action::FixationProb,
        "time" => Action::AbsrobTime,
        _ => {
            eprintln!("bad action");
            return;
        }
    };
    let shape = match shape.as_str() {
        "complete" => Shape::Complete,
        "star" => Shape::Star,
        "tree" => Shape::Tree,
        _ => {
            eprintln!("bad shape");
            return;
        }
    };
    let Ok(size) = size.parse::<usize>() else {
        eprintln!("bad size");
        return;
    };
    let Ok(r) = r.parse::<f64>() else {
        eprintln!("bad r");
        return;
    };

    let g = Graph::from_shape(size, shape, r);
    let mut res = vec![0.0; g.len()];
    numeric::crunch(&g, action, &mut res);

    for r in res {
        println!("{r}");
    }
}

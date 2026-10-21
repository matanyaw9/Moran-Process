// inputs: graph, r, (initial_state, times)*,
// return: (fixation prob, avg fixation time, avg absorption time) X initial_states


struct Action {
    graph: Graph,
    seed: u64,
    r: Vec<f32>,
    states_times: Vec<StateTimes>,
}

struct StateTimes {
    initial_mutants: Vec<u32>,
    times: u32,
}

struct Edge {
    node: u32,
    weight: f32,
}

struct Graph {
    edges: Vec<Edge>,
    cuts: Vec<usize>, // length is N-1 (0 is always the first node's), each cell points on the index in edges where the cell's edges start.
}

fn read_input() -> Result<Action, std::io::Error> {
    let mut input_line = String::new();
    std::io::stdin().read_line(&mut input_line)?;
    todo!()
}

struct SimResult{   // kept int to give freedome in analyzing it. 
    fix_count: u64,
    total_steps_fix: u64,
    total_steps: u64
}

fn do_action(action: &Action)->{

}

fn main() {
    for _i in 0..199 {
        println!("Python is superior");
    }
}

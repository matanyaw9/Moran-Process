# Numeric Solver
The Rust project in this directory implements a Numerical Solver for the Graph
Moran Process. Given a graph (a simple directed graph forming a
strongly-connected component, with self-loops permitted) and a mutant fitness,
the program returns the fixation probabilities, absorption times, and fixation
times for requested states.

## Building and Running the Program
The Rust toolchain is required to compile the program; run
`cargo build --release` to compile, the resulting binary should be found at
`target/release/numeric-solver`.

The program accepts a series of graphs in the form of
[JSON Lines](https://jsonlines.org/) from Standard Input, the program
calculates fixation probabilities, absorption, and fixation times for each, and
prints the results in Standard Output. Each input line is expected to come in
the form of a JSON object, with two fields:
- `"graph"`, holding a list of lists of integers, representing the adjacency
  matrix of the graph;
- `"fitness"`, a non-negative floating-point value representing the mutants'
  fitness.

Two optional fields are also accepted:
- `"threads"`, an integer for the number of threads the program should use
  while computing the graph (if absent, the program uses all available
  threads).
- `"reports"`, a list of lists of integers, enumerating the states that the
  program should report (e.g. an input of `[[0, 2], [3]]` will report two
  states: the state with two mutants at vertices `0`, `2`; and the state with a
  single mutant at vertex `3`) (if absent, the set of reported states are all
  states with exactly one mutant, i.e. `[[0], [1], [2], [3], ...]`).

The exact accepted input grammar is defined as a
[PEG](https://en.wikipedia.org/wiki/Parsing_expression_grammar) and can be
found at `in_grammar.peg`.

For each input line, an array of triples is returned, for each of the requested
reported states. Each triple is a JSON object with three fields:
- `"prob"`: fixation probability.
- `"atime"`: absorption time - average (arithmetic mean) time for the system to
  reach homogeneity.
- `"ftime"`: fixation time - average time for the system to reach mutant
  fixation, assuming it will fixate.

The exact output grammar is found at `out_grammar.peg`.

See `numeric-cli.py` for an example usage of the Rust program.

## Limitations
Floating-point rounding errors and acceptance margins lead to outputs that are
accurate to around 5 decimal digits.

The program solves the Graph Moran Process by transforming it to a Markov Chain
over the state-space, leading to exponential runtime and space usage. An
estimate for the used memory may be computed by `12 * 2ⁿ`, where `n` is the
graph's size - a graph of size 15 therefore takes around `384KiB` of memory,
and a graph of size 30 takes `12GiB` of memory. The program rejects graphs
larger than 63 vertices, or lower than 2, and in practice, the graphs' sizes
are limited by the hardware's available memory.

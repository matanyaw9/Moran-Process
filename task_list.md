# My tasks for this project:

# Moran Process Project Tasks (March 16, 2026)

## Tasks
- [X] Organise the files  
- [ ] Make the Process an Abstract Class
- [ ] Write a multi color Moran Process Simulation in C++
- [ ] Write the Moran Process simulation in C++
- [ ] Make the Analysis file fit large datasets
- [ ] Make the Analysis notebook just load figures that I will create with a function
- [ ] Give it normal CLI interface

## Desired Flow
1. Create a graph zoo
2. Run a simulation on a selected graph zoo
3. Analyze the simulation results
4. Create ML models from the simulation results
5. Create Evolutionary Algorithm to get extreme time/prob to fixation Graphs


# Desired Files Architecture: 

moran-process/
├── pyproject.toml         # Packaging and dependency configuration
├── uv.lock                # Your locked dependencies
├── README.md              # Project documentation
├── task_list.md           # Your to-do lists
├── submit_main.sh         # Top-level cluster submission script
├── tests/                 # Unit tests (sitting OUTSIDE of src/)
├── notebooks/             # Purely for Jupyter exploration (moved from analysis/)
├── simulation_data/       # Data, logs, and outputs (kept out of version control usually)
└── src/                   # The actual codebase!
    └── moran_process/     # Your main Python package
        ├── __init__.py
        ├── cli.py         # The new Typer/Click interface
        ├── core/          # population_graph.py, base_process.py
        ├── simulations/   # process_run.py, and future C++ code
        ├── pipeline/      # process_lab.py, worker_wrapper.py
        └── analysis/      # visualization.py, data_utils.py
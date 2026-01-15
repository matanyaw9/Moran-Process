Written by Matanya Wiener

This project intends to research evolutionary graphs, in particular respiratory organ shaped graphs, to see whether they have special properties that interfere with inner evolutionary processes (like pathogene processes).

## Project Structure

```
.
├── analysis/                    # Analysis notebooks
│   ├── df_analysis.ipynb       # Graph database and experiment analysis
│   └── compare_random_vs_respiratory.ipynb  # Comparative analysis
├── tests/                       # Test scripts and examples
│   ├── run_random_graphs_test.py
│   ├── test_csv_append.py
│   ├── example_usage.py
│   └── testing_scripts.ipynb
├── simulation_data/             # Experiment results and graph database
│   ├── graph_database.csv
│   ├── respiratory_runs.csv
│   └── random_graphs_*.csv
├── old_files/                   # Archived code
├── population_graph.py          # Graph topology definitions
├── process_run.py               # Single Moran process simulation
├── process_lab.py               # Batch experiment runner
├── main.py                      # Respiratory graph experiments
└── run_random_graphs.py         # Random graph experiments
```

## Core Components

- **PopulationGraph** - Defines various graph topologies (complete, cycle, mammalian, avian, fish, random)
- **ProcessRun** - Simulates a single Moran process on a graph
- **ProcessLab** - Manages multiple process runs and stores results

## Running Experiments

```bash
# Run respiratory graph experiments
python main.py

# Run random graph experiments
python run_random_graphs.py

# Run quick tests
python tests/run_random_graphs_test.py
```

## Analysis

Open Jupyter notebooks in the `analysis/` directory to visualize and analyze results.
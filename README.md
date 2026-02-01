# Moran Process Simulation

Written by Matanya Wiener

This project researches evolutionary dynamics on different graph topologies, particularly respiratory organ-shaped graphs, to investigate whether they have special properties that affect evolutionary processes.

## Project Structure

```
.
├── analysis/                    # Analysis notebooks and utilities
│   ├── analysis_utils.py       # Data loading and analysis utilities
│   ├── df_analysis.ipynb       # Graph database and experiment analysis
│   ├── compare_random_vs_respiratory.ipynb  # Comparative analysis
│   └── experiment_analysis.ipynb  # Additional experiment analysis
├── tests/                       # Test scripts and examples
│   ├── test_*.py               # Unit tests for core functionality
│   ├── example_usage.py        # Usage examples
│   └── testing_scripts.ipynb  # Interactive testing
├── simulation_data/             # Experiment results and graph database
│   ├── graph_database.csv      # Database of all graph topologies
│   ├── *_results.csv           # Experiment result files
│   └── tmp/                    # Temporary batch processing files
├── old_files/                   # Archived code
├── population_graph.py          # Graph topology definitions
├── process_run.py               # Single Moran process simulation
├── process_lab.py               # Batch experiment runner with HPC support
├── main.py                      # Main experiment runner
├── run_random_graphs.py         # Random graph experiments
└── worker_wrapper.py            # HPC job wrapper
```

## Core Components

- **PopulationGraph** - Graph topology definitions (complete, cycle, mammalian lung, avian, fish, random)
- **ProcessRun** - Single Moran process simulation engine
- **ProcessLab** - Batch experiment manager with CSV export and HPC job submission

## Quick Start

```bash
# Install dependencies
pip install -e .

# Run a simple experiment
python tests/example_usage.py

# Run full experiments
python main.py
python run_random_graphs.py

# Analyze results
jupyter notebook analysis/df_analysis.ipynb
```

## Features

- **Multiple Graph Topologies** - Complete, cycle, respiratory organ models, and random graphs
- **Batch Processing** - Run large-scale experiments with automatic result storage
- **HPC Support** - Submit jobs to SLURM clusters for parallel processing
- **Comprehensive Analysis** - Jupyter notebooks for visualization and statistical analysis
- **Automatic Data Management** - Graph database with topology hashing and result tracking
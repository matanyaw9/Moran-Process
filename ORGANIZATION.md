# Project Organization

This document describes the organization of the Moran Process simulation project.

## Directory Structure

### Root Directory
Contains core simulation code and main experiment runners:
- `population_graph.py` - Graph topology definitions and database management
- `process_run.py` - Single Moran process simulation engine
- `process_lab.py` - Batch experiment runner with CSV export
- `main.py` - Main experiment script for respiratory graphs
- `run_random_graphs.py` - Experiment script for random graphs

### `/analysis/`
Contains Jupyter notebooks for data analysis and visualization:
- `df_analysis.ipynb` - Analyzes graph database and experiment results
- `compare_random_vs_respiratory.ipynb` - Compares random vs respiratory graphs

### `/tests/`
Contains test scripts and example code:
- `run_random_graphs_test.py` - Quick test with fewer repeats
- `test_csv_append.py` - Tests CSV append functionality
- `example_usage.py` - Demonstrates ProcessLab features
- `testing_scripts.ipynb` - Interactive testing notebook

### `/simulation_data/`
Stores all experiment results and graph database:
- `graph_database.csv` - Database of all graph topologies with properties
- `respiratory_runs.csv` - Results from respiratory graph experiments
- `random_graphs_grand_experiments.csv` - Full random graph experiments
- `random_graphs_experiments_test.csv` - Test run results

### `/old_files/`
Archived code from previous versions

## Workflow

1. **Define Graphs** - Use `PopulationGraph` class to create graph topologies
2. **Run Experiments** - Use `ProcessLab.run_comparative_study()` to run simulations
3. **Save Results** - Results automatically saved to CSV if `output_path` provided
4. **Analyze** - Use notebooks in `/analysis/` to visualize and interpret results

## Key Features

- **Automatic CSV Appending** - New results append to existing files
- **Graph Database** - Automatically tracks all graph topologies with WL hashing
- **Flexible Experiment Runner** - ProcessLab handles batch simulations efficiently
- **Comprehensive Analysis** - Jupyter notebooks for visualization and statistics

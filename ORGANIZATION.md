# Project Organization

This document describes the organization of the Moran Process simulation project.

## Directory Structure

### Root Directory
Contains core simulation code and experiment runners:
- `population_graph.py` - Graph topology definitions and database management
- `process_run.py` - Single Moran process simulation engine
- `process_lab.py` - Batch experiment runner with CSV export and HPC job submission
- `main.py` - Main experiment script for respiratory graphs
- `run_random_graphs.py` - Random graph experiment script
- `worker_wrapper.py` - HPC job wrapper for SLURM clusters
- `pyproject.toml` - Project dependencies and configuration

### `/analysis/`
Analysis tools and notebooks:
- `analysis_utils.py` - Utility functions for data loading and environment setup
- `df_analysis.ipynb` - Graph database and experiment result analysis
- `compare_random_vs_respiratory.ipynb` - Comparative analysis between topologies
- `experiment_analysis.ipynb` - Additional experiment analysis and visualization

### `/tests/`
Test scripts and examples:
- `test_*.py` - Unit tests for core functionality (CSV handling, data loading, graph serialization, job submission)
- `example_usage.py` - Demonstrates ProcessLab usage patterns
- `testing_scripts.ipynb` - Interactive testing and development notebook

### `/simulation_data/`
Experiment results and graph database:
- `graph_database.csv` - Database of all graph topologies with properties and WL hashes
- `*_results.csv` - Experiment result files from various runs
- `tmp/` - Temporary files for batch processing and HPC jobs

### `/old_files/`
Archived code from previous development iterations

## Key Features

### Graph Topologies
- **Complete graphs** - Fully connected networks
- **Cycle graphs** - Ring topologies
- **Respiratory organ models** - Mammalian lung, avian, and fish gill structures
- **Random graphs** - Erdős–Rényi and other random topologies

### Experiment Management
- **Automatic CSV handling** - Results append to existing files with proper headers
- **Graph database** - Automatic tracking of topologies with Weisfeiler-Lehman hashing
- **Batch processing** - Run multiple parameter combinations efficiently
- **HPC integration** - Submit large experiments to SLURM clusters

### Analysis Pipeline
- **Data loading utilities** - Consistent data access across notebooks
- **Visualization tools** - Fixation probability plots, evolutionary trade-off maps
- **Statistical analysis** - Comparative studies across topologies and parameters

## Workflow

1. **Define Experiments** - Configure graphs and parameters in main scripts
2. **Run Simulations** - Use ProcessLab for local runs or HPC job submission
3. **Store Results** - Automatic CSV export with graph database updates
4. **Analyze Data** - Use analysis notebooks for visualization and interpretation

## Development Notes

- All notebooks should be run from the project root directory
- Use `analysis_utils.py` for consistent data loading
- Test new features with scripts in `/tests/` before production runs
- Graph database automatically prevents duplicate topology storage
# Analysis Directory

This directory contains Jupyter notebooks for analyzing simulation results.

## Files

- **df_analysis.ipynb** - Analysis of graph database and experiment results
  - Visualizes fixation probabilities
  - Compares median steps to fixation
  - Creates histograms and scatter plots
  
- **compare_random_vs_respiratory.ipynb** - Comparative analysis between random graphs and respiratory topology graphs
  - Loads and combines data from multiple CSV files
  - Creates evolutionary trade-off maps
  - Compares dynamics across topologies and selection coefficients

## Usage

Open notebooks with Jupyter:

```bash
jupyter notebook analysis/df_analysis.ipynb
jupyter notebook analysis/compare_random_vs_respiratory.ipynb
```

## Data Sources

These notebooks typically read from:
- `simulation_data/graph_database.csv` - Graph topology database
- `simulation_data/respiratory_runs.csv` - Respiratory graph experiment results
- `simulation_data/random_graphs_grand_experiments.csv` - Random graph experiment results
- `simulation_data/random_graphs_experiments_test.csv` - Test experiment results

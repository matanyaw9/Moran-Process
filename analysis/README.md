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

- **analysis_utils.py** - Utility functions for loading data and setting up the analysis environment

## Usage

### Running Notebooks

Open notebooks with Jupyter from the **project root directory**:

```bash
# From the project root (Moran-Process/)
jupyter notebook analysis/df_analysis.ipynb
jupyter notebook analysis/compare_random_vs_respiratory.ipynb
```

Or navigate to the analysis directory first:

```bash
cd analysis
jupyter notebook df_analysis.ipynb
jupyter notebook compare_random_vs_respiratory.ipynb
```

### Data Loading

The notebooks use `analysis_utils.py` to automatically handle data loading:

```python
from analysis_utils import setup_analysis_environment, load_all_data

# Setup environment and load all data
setup_analysis_environment()
data = load_all_data()

# Access specific datasets
df_graphs = data['graphs']              # Graph database
df_respiratory = data['respiratory']    # Respiratory experiments
df_random_full = data['random_full']    # Full random experiments
df_all = data['all_experiments']        # Combined data
```

## Data Sources

These notebooks read from `../simulation_data/`:
- `graph_database.csv` - Graph topology database
- `respiratory_runs.csv` - Respiratory graph experiment results
- `random_graphs_grand_experiments.csv` - Random graph experiment results
- `random_graphs_experiments_test.csv` - Test experiment results

## Troubleshooting

If you get file not found errors:

1. **Check your working directory**: Run `pwd` (Linux/Mac) or `cd` (Windows) to see where you are
2. **Run from project root**: Make sure you're in the main `Moran-Process/` directory
3. **Check data files exist**: Verify that `simulation_data/` contains the CSV files
4. **Use the utility functions**: The `analysis_utils.py` provides helpful error messages

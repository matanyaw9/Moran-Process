# Tests Directory

This directory contains test scripts and example code for the Moran Process simulation project.

## Files

- **run_random_graphs_test.py** - Quick test version of random graph experiments with fewer repeats
- **test_csv_append.py** - Test script to verify CSV append functionality
- **example_usage.py** - Example demonstrating the ProcessLab output_path feature
- **testing_scripts.ipynb** - Jupyter notebook for interactive testing and experimentation
- **test_data_loading.py** - Test script that uses analysis_utils to verify data loading
- **test_simple_data_loading.py** - Simple test script for data loading without dependencies

## Usage

Run test scripts from the project root directory:

```bash
# From project root (Moran-Process/)
python tests/run_random_graphs_test.py
python tests/test_csv_append.py
python tests/example_usage.py

# Test data loading
python tests/test_simple_data_loading.py      # Simple test (recommended)
python tests/test_data_loading.py             # Test using analysis_utils
```

Or run from the tests directory:

```bash
cd tests
python run_random_graphs_test.py
python test_simple_data_loading.py
```

## Data Loading Tests

Two test scripts are provided to verify data loading works correctly:

### test_simple_data_loading.py (Recommended)
- Standalone script with no external dependencies
- Tests loading CSV files from simulation_data/
- Works from any directory
- Provides clear error messages

### test_data_loading.py
- Uses the analysis_utils module
- Tests the same utilities used by analysis notebooks
- More comprehensive but has more dependencies

## Jupyter Notebook Testing

Open the interactive testing notebook:

```bash
jupyter notebook tests/testing_scripts.ipynb
```

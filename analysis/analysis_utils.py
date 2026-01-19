"""
Utility functions for analysis notebooks
"""
import os
import sys
import pandas as pd

def get_data_path():
    """
    Get the correct path to the simulation_data directory.
    Works from any subdirectory in the project.
    """
    current_dir = os.getcwd()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try different possible locations
    possible_paths = [
        'simulation_data',           # From project root
        '../simulation_data',        # From analysis/ or tests/
        '../../simulation_data',     # From deeper subdirectories
    ]
    
    # Also try relative to script location
    script_parent = os.path.dirname(script_dir)  # Go up from analysis/
    possible_paths.append(os.path.join(script_parent, 'simulation_data'))
    
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            return path
    
    # If none found, default to ../simulation_data and let the error handling deal with it
    return '../simulation_data'

def load_experiment_data(csv_filename):
    """
    Load experiment data from the simulation_data directory.
    
    Args:
        csv_filename (str): Name of the CSV file to load
        
    Returns:
        pd.DataFrame: Loaded data, or empty DataFrame if file not found
    """
    data_dir = get_data_path()
    file_path = os.path.join(data_dir, csv_filename)
    
    try:
        df = pd.read_csv(file_path)
        print(f"✓ Loaded {csv_filename}: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"✗ Could not find {file_path}")
        print(f"  Current directory: {os.getcwd()}")
        print(f"  Looking for data in: {os.path.abspath(data_dir)}")
        return pd.DataFrame()

# TODO Not sure I want this function... 
def load_all_data():
    """
    Load all common data files used in analysis.
    
    Returns:
        dict: Dictionary with DataFrames for different data types
    """
    data = {}
    
    # Load graph database
    data['graphs'] = load_experiment_data('graph_database.csv')
    
    # Load experiment results
    data['respiratory'] = load_experiment_data('respiratory_runs.csv')
    data['random_test'] = load_experiment_data('random_graphs_experiments_test.csv')
    data['random_full'] = load_experiment_data('random_graphs_grand_experiments.csv')
    
    # Combine experiment data
    experiment_dfs = [df for df in [data['respiratory'], data['random_test'], data['random_full']] if not df.empty]
    if experiment_dfs:
        data['all_experiments'] = pd.concat(experiment_dfs, ignore_index=True)
        print(f"✓ Combined experiments: {data['all_experiments'].shape}")
    else:
        data['all_experiments'] = pd.DataFrame()
        print("✗ No experiment data found")
    
    return data

def setup_analysis_environment():
    """
    Set up the analysis environment by adding parent directory to path
    and importing necessary modules.
    """
    # Add parent directory to Python path for imports
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    print(f"Analysis environment ready!")
    print(f"Current directory: {os.getcwd()}")
    print(f"Data directory: {os.path.abspath(get_data_path())}")
    
    return parent_dir
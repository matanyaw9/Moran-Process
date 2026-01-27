"""
Utility functions for analysis notebooks
"""
import os
import sys
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import shutil



COLOR_DICT = {
    'Random': 'lightgray',     
    'Avian': "#2DB806",       
    'Fish': '#1f77b4',        
    'Mammalian': "#833105",   
    'Complete': 'black',       
    'Other': 'yellow'          
}


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
    file_path = os.path.join(data_dir, os.path.basename(csv_filename))
    
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

def aggregate_results(batch_dir, save_to_dir, delete_temp=False):
    """
    Aggregate CSV result files from a batch directory into a single file.
        
    Args:
        batch_dir (str): Directory containing results subdirectory with CSV files
        save_to_dir (str): Directory to save the aggregated results
        delete_temp (bool): Whether to delete the batch directory after aggregation
            
    Returns:
        pd.DataFrame: Aggregated results, or None if no files found
    """
    batch_name = os.path.basename(batch_dir).removeprefix("batch_")
    output_file = os.path.join(save_to_dir, f"{batch_name}_results.csv")
    if os.path.exists(output_file):
        print(f"File {os.path.abspath(output_file)} already exitst! Not aggregating...")
        return load_experiment_data(output_file)
    
    results_path = os.path.join(batch_dir, "results")
    
    # 1. Find all CSV files in the results directory
    # Using glob handles the pattern matching for 'result_job_*.csv'
    all_files = glob.glob(os.path.join(results_path, "result_job_*.csv"))
    
    if not all_files:
        print(f"No result files found in {results_path}")
        return None

    print(f"Found {len(all_files)} files. Aggregating...")

    # 2. Read each CSV into a list of DataFrames
    df_list = []
    for filename in all_files:
        df = pd.read_csv(filename)
        df_list.append(df)

    # 3. Concatenate all DataFrames into one
    master_df = pd.concat(df_list, ignore_index=True)

    # 4. Optional: Save the master file to the batch root
    output_file = os.path.join(save_to_dir, f"{batch_name}_results.csv")
    master_df.to_csv(output_file, index=False)
    
    print(f"Success! Aggregated {len(master_df)} total rows.")
    print(f"Master file saved at: {output_file}")

    if delete_temp:
        shutil.rmtree(batch_dir)
        print(f"Deleted temporary batch directory: {batch_dir}")

    
    return master_df

def plot_property_effect(df, x_prop, y_outcome='prob_fixation', color_dict=COLOR_DICT):
    """
    Plots a specific graph property against an evolutionary outcome.
    Faceted by 'r' to show how the effect varies with selection strength.
    """
    plt.figure(figsize=(11,8))
    is_prob = (y_outcome == 'prob_fixation')
    ylabel = "Probability of Fixation ($P_{fix}$)" if is_prob else "Median Steps (Time)"
    sns.scatterplot(
        data=df,
        x=x_prop,
        y=y_outcome,
        hue='category',     # Color by Category
        style='r',          # Shape by r
        palette=color_dict,
        s=120,              # Marker size
        alpha=0.85,         # Transparency
        edgecolor='w',      # White edge to make points pop
        linewidth=0.5
    )
    # Add Neutral Limit Line if plotting Probability
    if is_prob:
        avg_n = df['n_nodes'].mean()
        plt.axhline(1/avg_n, color='black', linestyle=':', label=f'Neutral (1/N)')

    plt.title(f'Effect of {x_prop.replace('_', ' ').title()} on {ylabel}', fontsize=16)
    plt.xlabel(x_prop.replace('_', ' ').title())
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.4)
    
    # Legend handling: Place outside
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    
    plt.tight_layout()
    plt.show()

# print(len(GRAPH_PROPS))
# plot_property_effect(analysis_df, 'median_steps', 'prob_fixation')
# # --- EXAMPLES OF USAGE ---
# for prop in GRAPH_PROPS:
#     plot_property_effect(analysis_df, prop, 'median_steps')
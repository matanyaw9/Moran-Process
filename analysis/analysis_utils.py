"""
Utility functions for analysis notebooks
"""
import os
import sys
import pandas as pd
import numpy as np
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


def plot_hybrid_density(df, x_prop, y_outcome='prob_fixation', color_dict=COLOR_DICT, density_threshold=100):
    """
    Hybrid Plot:
    - Standard Scatter for sparse X-values.
    - Violin + Jittered Scatter for dense X-values (> density_threshold points).
    """
    plt.figure(figsize=(11, 8))
    
    # --- 1. Labeling & Setup ---
    is_prob = (y_outcome == 'prob_fixation')
    ylabel = "Probability of Fixation ($P_{fix}$)" if is_prob else "Median Steps (Time)"
    
    # Copy data to avoid warnings
    plot_df = df.copy()
    
    # --- 2. X-Axis Processing ---
    # We need numeric positions for the violins. 
    # If X is strings/categories, we map them to integers 0, 1, 2...
    is_numeric_x = pd.api.types.is_numeric_dtype(plot_df[x_prop])
    
    if is_numeric_x:
        # Round numeric data to group nearby points (e.g., 0.1 and 0.10001 become 0.1)
        plot_df['x_plot'] = plot_df[x_prop].round(3)
    else:
        # Map categories to integers
        unique_cats = sorted(plot_df[x_prop].unique())
        cat_map = {val: i for i, val in enumerate(unique_cats)}
        plot_df['x_plot'] = plot_df[x_prop].map(cat_map)

    # --- 3. Identify Dense Locations ---
    # Count points for each unique X value
    counts = plot_df['x_plot'].value_counts()
    dense_x_values = counts[counts > density_threshold].index.tolist()
    
    # Calculate an appropriate width for violins (avoid overlapping)
    unique_x_sorted = sorted(plot_df['x_plot'].unique())
    if len(unique_x_sorted) > 1:
        # Find minimum distance between neighbors to set width safely
        min_dist = min(np.diff(unique_x_sorted))
        violin_width = min_dist * 0.8
    else:
        violin_width = 0.5

    # --- 4. Draw Violins (Background Layer) ---
    # We loop ONLY through the dense X values and plant a violin there
    for x_val in dense_x_values:
        subset = plot_df[plot_df['x_plot'] == x_val]
        
        # plt.violinplot allows placing a violin at a specific 'positions'
        parts = plt.violinplot(
            dataset=subset[y_outcome],
            positions=[x_val],
            widths=violin_width,
            # native_scale=True,
            showmeans=False,
            showextrema=False # Hide the min/max lines, just show the blob
        )
        
        # Style the violin body to be neutral gray
        for pc in parts['bodies']:
            pc.set_facecolor('whitesmoke')
            pc.set_edgecolor('lightgray')
            pc.set_alpha(1) 

    # --- 5. Prepare Scatter Data (Jitter Layer) ---
    # For dense columns -> Add Jitter
    # For sparse columns -> Keep exact X
    
    # Define jitter function
    def apply_jitter(row):
        if row['x_plot'] in dense_x_values:
            # Jitter width should be relative to the violin width
            noise = np.random.uniform(-violin_width * 0.15, violin_width * 0.15)
            return row['x_plot'] + noise
        else:
            return row['x_plot']

    plot_df['x_jittered'] = plot_df.apply(apply_jitter, axis=1)

    # --- 6. Draw Main Scatter Plot (Top Layer) ---
    sns.scatterplot(
        data=plot_df,
        x='x_jittered',
        y=y_outcome,
        hue='category',
        style='r',
        palette=color_dict,
        s=80 if len(plot_df) > 1000 else 120, # Adjust size slightly if huge data
        alpha=0.85,
        edgecolor='w',
        linewidth=0.5,
        zorder=2
    )

    # --- 7. Final Formatting ---
    
    # If we mapped categories to integers, restore the labels
    if not is_numeric_x:
        plt.xticks(ticks=range(len(unique_cats)), labels=unique_cats)
    
    # Add Neutral Limit Line
    if is_prob:
        avg_n = df['n_nodes'].mean()
        plt.axhline(1/avg_n, color='black', linestyle=':', label=f'Neutral (1/N)')

    plt.title(f'Effect of {x_prop.replace("_", " ").title()} on {ylabel}', fontsize=16)
    plt.xlabel(x_prop.replace('_', ' ').title())
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.4)
    
    # Legend
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    
    plt.tight_layout()
    plt.show()


def plot_hexbin_property_effect(df, x_prop, y_outcome='prob_fixation', color_dict=COLOR_DICT, gridsize=30, cmap='Greys'):
    """
    Plots a hexbin density map for 'Random' graphs, overlaid with specific 'Animal' data points.
    - Random graphs -> Hexbin Density (Background)
    - Simulated Animals -> Scatter Points (Foreground)
    """
    plt.figure(figsize=(11, 8))
    
    # 1. Setup Labels
    is_prob = (y_outcome == 'prob_fixation')
    ylabel = "Probability of Fixation ($P_{fix}$)" if is_prob else "Median Steps (Time)"
    
    # 2. Split Data: Background (Random) vs Foreground (Animals)
    # Adjust 'Random' string if your category names are different
    random_mask = df['category'].str.contains('Random', case=False, na=False)
    random_df = df[random_mask]
    animal_df = df[~random_mask]
    
    # 3. Plot Hexbin (The Density Cloud) - ONLY for Random data
    # mincnt=1 ensures we don't plot hexagons for empty space
    hb = plt.hexbin(
        x=random_df[x_prop], 
        y=random_df[y_outcome], 
        gridsize=gridsize, 
        cmap=cmap, 
        mincnt=1,
        edgecolors='none',
        alpha=0.6,
        label='Random Density'
    )
    
    # Add a colorbar for the density
    cb = plt.colorbar(hb, label='Count of Random Graphs')
    
    # 4. Plot Scatter Overlay (The Specific Animals)
    sns.scatterplot(
        data=animal_df,
        x=x_prop,
        y=y_outcome,
        hue='category',     # Keep your color logic
        style='r',          # Keep your shape logic
        palette=color_dict,
        s=120,              # Make them big and visible
        alpha=1.0,          # No transparency for animals
        edgecolor='w',      # White edge to make them pop against the hexbins
        linewidth=1,
        zorder=10           # Force on top of everything
    )

    # 5. Shared Elements (Lines, Grid, Labels)
    if is_prob:
        avg_n = df['n_nodes'].mean()
        plt.axhline(1/avg_n, color='black', linestyle=':', linewidth=2, label='Neutral (1/N)')

    plt.title(f'Effect of {x_prop.replace("_", " ").title()} on {ylabel}', fontsize=16)
    plt.xlabel(x_prop.replace('_', ' ').title())
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # 6. Legend Handling
    # We want the legend to show the Animal categories + r styles
    plt.legend(bbox_to_anchor=(1.15, 1), loc='upper left', borderaxespad=0.)
    
    plt.tight_layout()
    plt.show()
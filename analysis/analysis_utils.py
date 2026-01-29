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
import textwrap


COLOR_DICT = {
    'Random': 'lightgray',     
    'Avian': "#2DB806",       
    'Fish': '#1f77b4',        
    'Mammalian': "#833105",   
    'Complete': 'black',       
    'Other': 'yellow'          
}
# Paste your dictionary here (or ensure it's in the global scope)
GRAPH_PROPERTY_DESCRIPTION = {
    'n_nodes': "The total number of vertices (individuals) in the graph.",
    'n_edges': "The total number of connections (links) between nodes in the graph.",
    'density': "The ratio of actual edges to the maximum possible number of edges (0 = empty, 1 = fully connected).",
    'diameter': "The longest shortest path between any pair of nodes (the 'width' of the network).",
    'avg_degree': "The average number of connections a node has.",
    'average_clustering': "A measure of how much nodes tend to cluster together (how likely a node's neighbors are also neighbors).",
    'average_shortest_path_length': "The average number of steps required to get from one node to any other node.",
    'degree_assortativity': "The correlation between a node's degree and the degree of its neighbors (positive = high-degree nodes connect to other high-degree nodes).",
    'avg_betweenness_centrality': "The average frequency that nodes act as a bridge along the shortest path between two other nodes.",
    'max_degree': "The highest number of connections held by a single node in the graph (the 'hub' size).",
    'min_degree': "The lowest number of connections held by a single node.",
    'degree_std': "The standard deviation of degrees; measures how much variation there is in connectivity (high = mixture of hubs and leaves).",
    'transitivity': "The overall probability that two neighbors of a node are connected (similar to clustering but calculated globally).",
    'radius': "The minimum eccentricity in the graph (the shortest distance from the 'center' of the graph to the furthest node).",
    'avg_degree_centrality': "The average fraction of the total possible nodes that any given node is connected to.",
    'max_degree_centrality': "The highest centrality score; indicates the most central or well-connected node relative to network size.",
    'max_betweenness_centrality': "The score of the node that acts as the most critical bridge or bottleneck in the network.",
    'avg_closeness_centrality': "The average speed at which nodes can access all other nodes (inverse of average distance).",
    'max_closeness_centrality': "The score of the node that can reach all other nodes in the fewest number of steps."
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
    ylabel = "Probability of Fixation ($P_{fix}$)" if is_prob else y_outcome.title()
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
    Hybrid Plot with Correlation & Description Patches.
    """
    plt.figure(figsize=(11, 8.5)) # Increased height slightly for the subtitle
    
    # --- 1. Labeling & Setup ---
    is_prob = (y_outcome == 'prob_fixation')
    ylabel = "Probability of Fixation ($P_{fix}$)" if is_prob else y_outcome.title()
    
    # PATCH 1: Calculate Correlation (Pearson)
    clean_df = df[[x_prop, y_outcome, 'r']].dropna()

    # TODO I want different correlations for different r values: 
    r_groups = clean_df.groupby('r')
    corrs_by_r = r_groups.apply(lambda g: g[x_prop].corr(g[y_outcome]))

    # Construct the text string for the box
    stats_lines = [f"Pearson Correlation"]
    stats_lines.append("-" * 15)
    for r_val, r_corr in corrs_by_r.items():
        stats_lines.append(f"r={r_val}: {r_corr:.3f}")

    stats_text = "\n".join(stats_lines)

    
    # Copy data for plotting
    plot_df = df.copy()
    
    # --- 2. X-Axis Processing ---
    is_numeric_x = pd.api.types.is_numeric_dtype(plot_df[x_prop])
    
    if is_numeric_x:
        plot_df['x_plot'] = plot_df[x_prop].round(3)
    else:
        unique_cats = sorted(plot_df[x_prop].unique())
        cat_map = {val: i for i, val in enumerate(unique_cats)}
        plot_df['x_plot'] = plot_df[x_prop].map(cat_map)

    # --- 3. Identify Dense Locations ---
    counts = plot_df['x_plot'].value_counts()
    dense_x_values = counts[counts > density_threshold].index.tolist()
    
    # === SMART WIDTH CALCULATION ===
    unique_x_sorted = sorted(plot_df['x_plot'].unique())
    
    if len(unique_x_sorted) > 1:
        diffs = np.diff(unique_x_sorted)
        total_span = unique_x_sorted[-1] - unique_x_sorted[0]
        if total_span == 0: total_span = 1.0 
        
        # Threshold: 2% of total span
        min_valid_gap_threshold = total_span * 0.02 
        valid_gaps = diffs[diffs > min_valid_gap_threshold]
        
        if len(valid_gaps) > 0:
            dist_basis = np.min(valid_gaps)
        else:
            dist_basis = total_span * 0.1
            
        violin_width = dist_basis * 0.7 
    else:
        violin_width = 0.5

    # --- 4. Draw Violins (Background) ---
    for x_val in dense_x_values:
        subset = plot_df[plot_df['x_plot'] == x_val]
        parts = plt.violinplot(
            dataset=subset[y_outcome],
            positions=[x_val],
            widths=violin_width,
            showmeans=False,
            showextrema=False
        )
        for pc in parts['bodies']:
            pc.set_facecolor('whitesmoke')
            pc.set_edgecolor('lightgray')
            pc.set_alpha(1) 

    # --- 5. Draw Scatter (Foreground) ---
    def apply_jitter(row):
        if row['x_plot'] in dense_x_values:
            noise = np.random.uniform(-violin_width * 0.15, violin_width * 0.15)
            return row['x_plot'] + noise
        else:
            return row['x_plot']

    plot_df['x_jittered'] = plot_df.apply(apply_jitter, axis=1)

    sns.scatterplot(
        data=plot_df,
        x='x_jittered',
        y=y_outcome,
        hue='category',
        style='r',
        palette=color_dict,
        s=80 if len(plot_df) > 1000 else 120,
        alpha=0.85,
        edgecolor='w',
        linewidth=0.5,
        zorder=2
    )

    # --- 6. Final Formatting ---
    if not is_numeric_x:
        plt.xticks(ticks=range(len(unique_cats)), labels=unique_cats)
    
    if is_prob:
        avg_n = df['n_nodes'].mean()
        plt.axhline(1/avg_n, color='black', linestyle=':', label=f'Neutral (1/N)')

    # Titles & Labels
    plt.xlabel(x_prop.replace('_', ' ').title())
    plt.ylabel(ylabel)
    
    # PATCH 2: Add Description as Subtitle
    # Fetch description, default to empty string if not found
    desc_text = GRAPH_PROPERTY_DESCRIPTION.get(x_prop, "")
    
    # Use the Suptitle for the main title, and standard title for description to get distinct sizing
    plt.suptitle(f'Effect of {x_prop.replace("_", " ").title()} on {ylabel}', fontsize=16, y=0.96)
    
    # Wrap text so it doesn't run off the screen
    wrapped_desc = "\n".join(textwrap.wrap(desc_text, width=80))
    plt.title(wrapped_desc, fontsize=10, style='italic', color='#555555', pad=15)

    # Add Correlation Text Box
    # First, capture the legend object
    leg = plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

    # Add the correlation text box below the legend
    plt.gca().text(
        1.02, 0.4, # Horizontal matches legend; Vertical adjusted manually or via transform
        stats_text, 
        transform=plt.gca().transAxes, 
        fontsize=10, 
        verticalalignment='top', # Anchor to the top of the text block
        horizontalalignment='left',
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor="lightgray")
    )

    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    
    plt.tight_layout()
    # plt.subplots_adjust(right=0.8) # Make room on the right
    plt.show()
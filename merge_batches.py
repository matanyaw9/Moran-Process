
# merge_batches.py
# Here is a script that takes the outputs of two batches and merges them into a single batch for merged analysis. 


import os
from pathlib import Path
import shutil
import joblib
import pickle

ROOT = Path(os.getcwd()) 

# Now define your paths relative to ROOT
DATA_DIR = ROOT / "simulation_data"

NEW_BATCH_NAME = "merged_batch_04"
BATCH_1 = "batch_large_test_30_02"
BATCH_2 = "batch_extreme_graphs_02"

# Function to copy files from one batch to the new batch
def concat_csv_files(csv_path_1, csv_path_2, new_file_path):
    """
    Concatenates two CSV files with the same columns into a single file.
    Uses shutil to avoid loading the entire files into memory, while 
    ensuring the header from the second file is skipped.
    """
    with open(new_file_path, 'wb') as f_out: # Open in binary mode for shutil
        
        # 1. Copy the entirety of the first file (including its header)
        with open(csv_path_1, 'rb') as f_in1:
            shutil.copyfileobj(f_in1, f_out)
            
        # 2. Copy the second file, skipping its header row
        with open(csv_path_2, 'rb') as f_in2:
            # Advance the file pointer past the first line (the header)
            next(f_in2, None) 
            
            # Copy the remaining data
            shutil.copyfileobj(f_in2, f_out)

def merge_graph_zoos(zoo_path_1, zoo_path_2, new_zoo_path):
    """
    This function should take 2 graph zoos and merge them into a single graph zoo in the new batch directory. 
    The function should ensure that the graph names are unique across both zoos to avoid overwriting.
    """
    # Load with appropriate method based on file extension
    zoo_1 = joblib.load(zoo_path_1) if str(zoo_path_1).endswith('.joblib') else pickle.load(open(zoo_path_1, 'rb'))
    zoo_2 = joblib.load(zoo_path_2) if str(zoo_path_2).endswith('.joblib') else pickle.load(open(zoo_path_2, 'rb'))
    
    all_graphs = zoo_1 + zoo_2
    wl_hashes = set([graph.wl_hash for graph in all_graphs])
    if len(wl_hashes) != len(all_graphs):
        updated_output_zoo = []
        seen_hashes = set()
        for graph in all_graphs:
            if graph.wl_hash not in seen_hashes:
                updated_output_zoo.append(graph)
                seen_hashes.add(graph.wl_hash)
    else:
        updated_output_zoo = all_graphs

    joblib.dump(updated_output_zoo, new_zoo_path)

    
if __name__ == "__main__":

    NEW_BATCH_DIR = DATA_DIR / NEW_BATCH_NAME
    BATCH_1_DIR = DATA_DIR / BATCH_1
    BATCH_2_DIR = DATA_DIR / BATCH_2

    os.makedirs(NEW_BATCH_DIR, exist_ok=True)
    print(f"Created new batch directory: {NEW_BATCH_DIR}")

    # Define the paths to the full_results.csv files in each batch
    full_results_1 = BATCH_1_DIR / "full_results.csv"
    full_results_2 = BATCH_2_DIR / "full_results.csv"
    new_full_results_path = NEW_BATCH_DIR / "full_results.csv"
    concat_csv_files(full_results_1, full_results_2, new_full_results_path)
    print(f"Merged full results saved to: {new_full_results_path}")


    graph_props_1 = BATCH_1_DIR / "graph_props.csv"
    graph_props_2 = BATCH_2_DIR / "graph_props.csv"
    new_graph_props_path = NEW_BATCH_DIR / "graph_props.csv"
    concat_csv_files(graph_props_1, graph_props_2, new_graph_props_path)
    print(f"Merged graph properties saved to: {new_graph_props_path}")

    # Define the paths to the graph zoos in each batch
    graph_zoo_1 = BATCH_1_DIR / "tmp" /  "graphs.pkl"
    graph_zoo_2 = BATCH_2_DIR / "tmp" / "graph_zoo.joblib"
    new_graph_zoo_path = NEW_BATCH_DIR / "tmp" / "graph_zoo.joblib"
    os.makedirs(NEW_BATCH_DIR / "tmp", exist_ok=True)
    merge_graph_zoos(graph_zoo_1, graph_zoo_2, new_graph_zoo_path)
    print(f"Merged graph zoo saved to: {new_graph_zoo_path}")
import os
from pathlib import Path
import sys
import joblib
import pickle
import polars as pl


from moran_process.core import population_graph
sys.modules['population_graph'] = population_graph

def merge_csv_safely(csv_path_1: Path, csv_path_2: Path, new_file_path: Path):
    """
    Concatenates two CSVs safely by aligning their column NAMES, not physical positions.
    Uses out-of-core processing to prevent memory crashes.
    """
    if not csv_path_1.exists() or not csv_path_2.exists():
        print(f"Missing files. Skipping merge for {new_file_path.name}")
        return

    print(f"Merging {new_file_path.name}...")
    
    # Lazy scan both files
    df1 = pl.scan_csv(csv_path_1)
    df2 = pl.scan_csv(csv_path_2)
    
    # how="diagonal" safely aligns columns by name. 
    # If a column exists in one but not the other, it fills with nulls instead of shifting data.
    merged_lazy = pl.concat([df1, df2], how="diagonal")
    
    # sink_csv streams the result directly to disk
    merged_lazy.sink_csv(new_file_path)

def merge_graph_zoos(zoo_path_1: Path, zoo_path_2: Path, new_zoo_path: Path):
    """
    Merges two graph zoos, ensuring unique graphs based on wl_hash.
    """
    print(f"Merging graph zoos...")
    # Load with appropriate method based on file extension
    zoo_1 = joblib.load(zoo_path_1) if str(zoo_path_1).endswith('.joblib') else pickle.load(open(zoo_path_1, 'rb'))
    zoo_2 = joblib.load(zoo_path_2) if str(zoo_path_2).endswith('.joblib') else pickle.load(open(zoo_path_2, 'rb'))
    
    all_graphs = zoo_1 + zoo_2
    
    # Use a dictionary to keep only unique graphs by their hash
    unique_graphs = {graph.wl_hash: graph for graph in all_graphs}
    updated_output_zoo = list(unique_graphs.values())

    joblib.dump(updated_output_zoo, new_zoo_path)


def merge_batches(batch_1_name: str, batch_2_name: str, new_batch_name: str):
    """
    Main entry point to merge two simulation batches.
    """
    ROOT = Path(os.getcwd()) 
    DATA_DIR = ROOT / "simulation_data"
    
    NEW_BATCH_DIR = DATA_DIR / new_batch_name
    BATCH_1_DIR = DATA_DIR / batch_1_name
    BATCH_2_DIR = DATA_DIR / batch_2_name
    
    # Validate inputs
    if not BATCH_1_DIR.exists() or not BATCH_2_DIR.exists():
        raise FileNotFoundError("One or both of the source batch directories do not exist.")

    # Prepare output directories
    NEW_BATCH_DIR.mkdir(parents=True, exist_ok=True)
    (NEW_BATCH_DIR / "tmp").mkdir(exist_ok=True)
    print(f"Created new batch directory: {NEW_BATCH_DIR}")

    # 1. Merge the CSVs safely
    merge_csv_safely(BATCH_1_DIR / "full_results.csv", 
                     BATCH_2_DIR / "full_results.csv", 
                     NEW_BATCH_DIR / "full_results.csv")
    
    merge_csv_safely(BATCH_1_DIR / "graph_props.csv", 
                     BATCH_2_DIR / "graph_props.csv", 
                     NEW_BATCH_DIR / "graph_props.csv")

    # 2. Find and merge the Zoos (handling both .pkl and .joblib extensions)
    # Search batch 1
    zoo_1_files = list((BATCH_1_DIR / "tmp").glob("graph*.*"))
    # Search batch 2
    zoo_2_files = list((BATCH_2_DIR / "tmp").glob("graph*.*"))
    
    if zoo_1_files and zoo_2_files:
        new_graph_zoo_path = NEW_BATCH_DIR / "tmp" / "graph_zoo.joblib"
        merge_graph_zoos(zoo_1_files[0], zoo_2_files[0], new_graph_zoo_path)
        print(f"Merged graph zoo saved to: {new_graph_zoo_path}")
    else:
        print("Warning: Could not find graph zoos in one or both of the tmp directories.")
        
    print(f"Successfully merged {batch_1_name} and {batch_2_name} into {new_batch_name}!")


if __name__ == "__main__":
    # Example usage:
    merge_batches("batch_large_test_30_02", "batch_extreme_graphs_02", "merged_batch_06")

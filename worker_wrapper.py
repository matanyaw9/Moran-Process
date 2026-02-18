import argparse
import os
import sys
import pickle
import pandas as pd
from datetime import datetime
from population_graph import PopulationGraph
from process_run import ProcessRun
import joblib

def load_data(zoo_path, task_manifest_path):
    """Loads the Graph Zoo and the Task Manifest."""
    
    # 1. Load the Graph Zoo (The "Frozen" Graphs)
    if not os.path.exists(zoo_path):
        raise FileNotFoundError(f"Could not find {zoo_path}")
    
    with open(zoo_path, "rb") as f:
        graph_zoo = joblib.load(f)
    print(f"[Worker] Loaded {len(graph_zoo)} graphs from Zoo.")

    # 2. Load the Task Manifest (The "Huge Table")
    if not os.path.exists(task_manifest_path):
        raise FileNotFoundError(f"Could not find task_manifest.csv at {task_manifest_path}")
        
    manifest_df = pd.read_csv(task_manifest_path)
    print(f"[Worker] Loaded Manifest with {len(manifest_df)} total tasks.")
    
    return graph_zoo, manifest_df

def run_worker_slice(batch_dir, zoo_path, manifest_path, worker_index):
    """
    Main Logic:
    1. Calculate which rows of the CSV belong to this job.
    2. Run simulations for those rows.
    3. Save a unique CSV file.
    """
    print(f"--- Worker Started | Job Index: {worker_index} ---")
    
    # 1. Load Data
    graph_zoo, manifest_df = load_data(zoo_path, manifest_path)
    
    # 2. Calculate My Slice
    # LSF Job Indices are 1-based (1, 2, 3...)
    # We convert to 0-based for array slicing

    
    # Handle the last job (which might not have a full chunk)

    my_tasks = manifest_df[manifest_df['worker_id'] == worker_index]

    print('='*60)
    print(my_tasks)
    print('='*60)


    if my_tasks.empty:
        print(f"[Worker] No tasks found. Exiting.")
        return

    print(f"[Worker] Processing {my_tasks['n_repeats'].sum()} simulations.")
    # 3. Run The Simulations
    results_buffer = []
    # Iterate over the rows in my slice
    # iterrows is slow, but fine for 500 tasks. itertuples is faster.
    for row in my_tasks.itertuples():
        try:
            # A. Get Parameters from the Table
            # row.graph_idx corresponds to the list index in graphs.pkl
            target_graph = graph_zoo[row.graph_idx]
            r_val = row.r_value
            n_repeats = row.n_repeats
            
            for rep in range(n_repeats):
                # B. Initialize Simulation
                sim = ProcessRun(population_graph=target_graph, selection_coefficient=r_val)
                sim.initialize_random_mutant()
                
                # C. Run
                raw_result = sim.run()
                
                # D. Save Record
                record = {
                    "task_id": row.task_id,
                    "job_id": worker_index,
                    "wl_hash": target_graph.wl_hash,
                    "graph_name": target_graph.name,
                    "r": r_val,
                    "fixation": raw_result["fixation"],
                    "steps": raw_result["steps"],
                    "initial_mutants": raw_result["initial_mutants"],
                    "duration": raw_result["duration"],

                    # Add any other graph properties you need for analysis
                    # **target_graph.metadata # (Optional) expands WL hash, etc.

                }
                results_buffer.append(record)
            
        except Exception as e:
            print(f"ERROR in Task {row.task_id}: {e}")
            # Continue to next task so one failure doesn't kill the job
            continue

    # 4. Save My Results
    if results_buffer:
        results_df = pd.DataFrame(results_buffer)
        
        # Save to: batch_dir/results/result_job_5.csv
        filename = f"result_job_{worker_index}.csv"
        save_path = os.path.join(batch_dir, "results", filename)
        
        results_df.to_csv(save_path, index=False)
        print(f"--- Worker Finished. Saved {len(results_df)} rows to {filename} ---")
    else:
        print("--- Worker Finished. No results generated. ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--zoo-path", required=True, help="Path to the graph zoo pickle file")
    parser.add_argument("--manifest-path", required=True, help="Path to the task manifest CSV file")
    parser.add_argument("--batch-dir", required=True, help="Path to the batch directory (containing graphs.pkl)")

    
    # Optional: Allow manually passing job-index for testing locally
    # On the cluster, we will look at the env variable LSB_JOBINDEX
    parser.add_argument("--job-index", type=int, default=None)
    
    args = parser.parse_args()
    
    # Get Job Index
    job_idx = args.job_index
    if job_idx is None:
        # Try to get from LSF Environment
        env_idx = os.environ.get("LSB_JOBINDEX")
        if env_idx:
            job_idx = int(env_idx)
        else:
            print("ERROR: Could not find job index! (Set --job-index or run via bsub)")
            sys.exit(1)
            
    run_worker_slice(args.batch_dir, args.zoo_path, args.manifest_path, job_idx)
    # print(f"job number {job_idx} is running!:\t Batch dir: {args.batch_dir}\t Chunk Size: {args.chunk_size}")

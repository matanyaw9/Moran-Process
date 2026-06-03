import pandas as pd
import time
import os 
import glob
import shutil
import pickle
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import itertools
import joblib

from moran_process.core.population_graph import PopulationGraph
from moran_process.simulations.moran_simulation_process import MoranProcess
from moran_process.analysis.analysis_utils import create_batch_info

def _parse_memory_mb(memory) -> int:
    """Convert a human-readable memory string to MB (integer) for LSF rusage.

    Accepts: "2GB", "2G", "512MB", "512M", or a bare integer string/int (treated as MB).
    Examples: "2GB" -> 2048, "8G" -> 8192, "512MB" -> 512, "2048" -> 2048.
    """
    if isinstance(memory, int):
        return memory
    s = str(memory).strip().upper()
    if s.endswith("GB") or s.endswith("G"):
        factor = 1024
        num = s.rstrip("GB").rstrip("G")
    elif s.endswith("MB") or s.endswith("M"):
        factor = 1
        num = s.rstrip("MB").rstrip("M")
    else:
        factor = 1
        num = s
    return int(float(num) * factor)


class ProcessLab:
    """ Manages multiple process runs and stores their results"""
    def __init__(self):                     
        """
        """
    def run_comparative_study(self, graphs_zoo, r_values, n_repeats=100, print_time=True, output_path=None):
        """
        Run comparative study across multiple graphs and selection coefficients.
        
        :param graphs: List of instantiated PopulationGraph objects
        :param r_values: List of floats (selection coefficients)
        :param n_repeats: Number of repetitions per configuration
        :param print_time: Whether to print timing information for each run
        :param output_path: Optional path to save results CSV. If provided, results will be 
                           appended to existing file or create new file. Can be absolute or 
                           relative path (e.g., 'simulation_data/results.csv')
        :return: DataFrame with all results
        """
        all_results = []
        
        # Total iterations for progress bar
        total_sims = len(graphs_zoo) * len(r_values) * n_repeats
        
        print(f"--- Starting Study: {len(graphs_zoo)} Graphs x {len(r_values)} r-vals x {n_repeats}  = {total_sims} repeats ---")
        
        # We can optimize by converting graphs to adjacency lists ONCE
        for graph_obj in graphs_zoo:
            # Convert once per graph; reused across all r values and repeats
            graph_core = graph_obj.to_simulation_struct()

            for r in r_values:
                for _ in range(n_repeats):
                    sim = MoranProcess(graph_core=graph_core, selection_coefficient=r)
                    sim.initialize_random_mutant()
                    raw_result = sim.run()

                    record = {
                        **graph_obj.metadata,
                        "r": r,
                        "fixation": raw_result["fixation"],
                        "steps": raw_result["steps"],
                        "duration": raw_result["duration"],
                    }
                    all_results.append(record)
                    if print_time:
                        seconds = raw_result['duration']
                        print(f"Graph: {graph_obj.name}, r: {r}, Fixation: {raw_result['fixation']}, n_nodes: {graph_obj.number_of_nodes()}, Steps: {raw_result['steps']}, Time: {seconds:.4f}s")
        
        print('Done.')
        df = pd.DataFrame(all_results)
        
        # Save to CSV if output_path is provided
        if output_path:
            ProcessLab.save_results(df, output_path)
        
        return df
    
    @staticmethod
    def save_results(df, output_path):
        """
        Save results to CSV file, appending to existing file if it exists.
        
        :param df: DataFrame with results to save
        :param output_path: Path to CSV file
        """
        # Ensure directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Append to existing CSV if it exists, otherwise create new
        if os.path.exists(output_path):
            existing_df = pd.read_csv(output_path)
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            print(f"Appending {len(df)} new rows to existing CSV with {len(existing_df)} rows")
            combined_df.to_csv(output_path, index=False)
            print(f"Total rows in CSV: {len(combined_df)}")
        else:
            df.to_csv(output_path, index=False)
            print(f"Created new CSV file with {len(df)} rows")
        
        print(f"Results saved to: {output_path}")
     

    # --- HPC SUBMISSION ENGINE ---
    def submit_jobs(self,
                    zoo_path,
                    n_graphs,
                    r_values,
                    batch_name,
                    batch_dir,
                    n_repeats=10,
                    n_requested_jobs=1,
                    queue="short",
                    memory="2GB",
                    graph_types=None,
                    node_sizes=None,
                    description="",
                    notes="",
                    batch_seed=None,
                    ):
        """
        1. Dumps all graphs to 'graphs.pkl'
        2. Creates 'task_manifest.csv' (The Huge Table)
        3. Submits an LSF Job Array where each worker takes a 'chunk' of the table.
        """
        print("entered ProcessLab.submit_jobs ")

        # Create subdirs for logs and results
        if os.path.exists(batch_dir):
            print(f"Warning: Batch directory {batch_name} already exists. Appending/Overwriting.")

        tmp_dir = os.path.join(batch_dir, 'tmp')
        os.makedirs(tmp_dir, exist_ok=True)
        results_dir = os.path.join(tmp_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        logs_dir = os.path.join(batch_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)

        register_graphs_job(zoo_path, batch_name, batch_dir)

        print(f"--- Preparing Batch {batch_name} ---")

        # 3. Generate Task Manifest (The Huge Table)
        # We expand the loops into a list of rows
        manifest_path = os.path.join(tmp_dir, "task_manifest.csv")
        manifest_df = ProcessLab._create_task_list(n_graphs,
                                                   r_values,
                                                   n_repeats,
                                                   n_requested_jobs,
                                                   output_path=manifest_path,
                                                   batch_seed=batch_seed)

        print(f"Created manifest with {len(manifest_df)} rows.")

        # 4. Load the full zoo once here (login node), convert to per-worker shards.
        # Workers receive a small list[GraphCore] shard (~50 graphs) instead of
        # the full zoo (50k graphs). This is the main RAM fix.
        print(f"[ProcessLab] Loading zoo from {zoo_path} ...")
        with open(zoo_path, "rb") as f:
            graph_zoo = joblib.load(f)
        print(f"[ProcessLab] Zoo loaded: {len(graph_zoo)} graphs.")

        zoo_shards_dir = os.path.join(tmp_dir, "zoo_shards")
        manifest_df = ProcessLab._write_zoo_shards(manifest_df, graph_zoo, zoo_shards_dir)
        del graph_zoo  # free the full zoo; shards are on disk now

        manifest_df.to_csv(manifest_path, index=False)
        print(f"[ProcessLab] Manifest updated with local_graph_idx → {manifest_path}")

        # 5. Submit LSF job array.
        # Each worker receives --zoo-shard-dir and constructs its own shard path
        # using $LSB_JOBINDEX, so no global zoo path is needed at runtime.
        python_exec = sys.executable
        memory_mb = _parse_memory_mb(memory)

        cmd_job = [
            "bsub",
            "-q", queue,
            "-J", f"batch_{batch_name}[1-{n_requested_jobs}]",
            "-o", os.path.join(logs_dir, "job_%J_%I.out"),
            "-e", os.path.join(logs_dir, "job_%J_%I.err"),
            "-R", f"rusage[mem={memory_mb}]",
            "-env", "OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, PYTHONPATH=src",
        ]

        cmd_process = [
            python_exec, "-u",
            "-m", "moran_process.pipeline.worker_wrapper",
            "--zoo-shard-dir", str(zoo_shards_dir),
            "--manifest-path", str(manifest_path),
            "--batch-dir", str(tmp_dir),
        ]
        cmd = cmd_job + cmd_process

        print(f"Submitting: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode == 0:
            print("✅ Command launched and finished successfully.")
        else:
            print(f"❌ Command failed with return code {result.returncode}")
        print(f"Error message: {result.stderr}")
        print(f"Batch submitted! \n > Logs: {logs_dir} \n > Results: {results_dir}")

        create_batch_info(
            batch_dir=batch_dir,
            name=batch_name,
            description=description,
            graph_types=graph_types,
            r_values=r_values,
            n_repeats=n_repeats,
            node_sizes=node_sizes,
            notes=notes,
            n_graphs=n_graphs,
            total_simulations=n_graphs * len(r_values) * n_repeats,
            n_requested_jobs=n_requested_jobs,
            queue=queue,
            memory_mb=memory_mb,
            zoo_path=zoo_path,
            batch_seed=batch_seed,
        )


    # @staticmethod
    # def _create_task_list(n_graphs, r_values, n_jobs, n_repeats):
    #     """Create CSV task list for job array execution."""
    #     tasks = []
    #     task_id = 0
    #     simulations_per_worker = math.ceil((n_graphs * len(r_values) * n_repeats) / n_jobs) 
    #     simulations = simulations_per_worker
    #     for graph_idx in range(n_graphs):
    #         for r in r_values:
    #             repeats = min(simulations, n_repeats)
    #             tasks.append({
    #                 'task_id': task_id,
    #                 'graph_idx': graph_idx,
    #                 'r': r,
    #                 'repeats': min(repeats)
    #             })
    #             task_id += 1
    #             simulations -= repeats
        
    #     return pd.DataFrame(tasks)


    def _create_task_list(n_graphs, r_values, repeats_per_config, num_workers,
                          output_path="task_manifest.csv", batch_seed=None):
        """
        Allocates simulations to workers as evenly as possible.
        Splits a single configuration across multiple workers if necessary.

        batch_seed: integer seed for reproducible batches. A per-task seed is derived
                    from a batch-level RNG so the batch can be exactly replayed by
                    storing batch_seed in batch_info.json. None = random (no seeds stored).
        """
        import numpy as np
        # None → seeds from OS entropy; int → deterministic. Either way, seeds are
        # stored in the manifest so any batch can be replayed from its manifest alone.
        task_rng = np.random.default_rng(batch_seed)

        # 1. Generate all unique configurations (Graph X, r Y)
        configs = list(itertools.product(range(n_graphs), r_values))
        num_configs = len(configs)
        
        # 2. Calculate total work and fair share
        total_sims = num_configs * repeats_per_config
        base_share = total_sims // num_workers
        remainder = total_sims % num_workers
        
        tasks = []
        
        # Trackers for our position in the configurations list
        current_config_idx = 0
        # How many repeats of the current config are still waiting to be assigned?
        repeats_left_in_current_config = repeats_per_config 
        task_id = 0
        # 3. Assign work to each worker
        for worker_id in range(num_workers):
            
            # Calculate exactly how many repeats this worker should handle
            # (Distribute the remainder: first few workers get +1 simulation)
            worker_target = base_share + (1 if worker_id < remainder else 0)
            
            while worker_target > 0 and current_config_idx < num_configs:
                graph_idx, r = configs[current_config_idx]
                
                # How many can we take from the current config?
                # Either all that are left in this config, or just enough to fill the worker.
                take = min(worker_target, repeats_left_in_current_config)
                
                # Add the row to our manifest
                tasks.append({
                    'task_id': task_id,
                    'worker_id': worker_id+1,
                    'graph_idx': graph_idx,
                    'r_value': r,
                    'n_repeats': take,
                    'seed': int(task_rng.integers(0, 2**31)),
                })
                
                # Update counters
                worker_target -= take
                repeats_left_in_current_config -= take
                
                # If we used up this configuration, move to the next one
                if repeats_left_in_current_config == 0:
                    current_config_idx += 1
                    repeats_left_in_current_config = repeats_per_config
                
                task_id += 1
        # 4. Create DataFrame and save
        manifest = pd.DataFrame(tasks)
        manifest.to_csv(output_path, index=False)

        print(f"Manifest created! Total Sims: {total_sims}. Distributed across {num_workers} workers.")
        return manifest

    @staticmethod
    def _write_zoo_shards(manifest_df, graph_zoo, shards_dir):
        """Write one GraphCore shard per worker to shards_dir.

        Each shard is a list[GraphCore] containing only the graphs that worker
        needs, converted from PopulationGraph at submission time so workers
        never load NetworkX objects. The manifest is returned with an added
        `local_graph_idx` column (0-based index into the shard) alongside the
        original global `graph_idx` for debugging.
        """
        os.makedirs(shards_dir, exist_ok=True)
        local_idx_map = {}  # (worker_id, global_graph_idx) -> local_graph_idx

        n_workers = manifest_df['worker_id'].nunique()
        print(f"[ProcessLab] Creating {n_workers} zoo shards (GraphCore / CSR format)...")

        for worker_id, group in manifest_df.groupby('worker_id'):
            global_idxs = sorted(group['graph_idx'].unique())
            for local_i, global_i in enumerate(global_idxs):
                local_idx_map[(worker_id, global_i)] = local_i

            shard = [graph_zoo[g].to_simulation_struct() for g in global_idxs]
            shard_path = os.path.join(shards_dir, f"zoo_worker_{worker_id}.pkl")
            joblib.dump(shard, shard_path)

            if worker_id % 100 == 0 or worker_id == 1:
                print(f"  [Shards] {worker_id}/{n_workers} — {len(shard)} graphs → {os.path.basename(shard_path)}")

        manifest_df = manifest_df.copy()
        manifest_df['local_graph_idx'] = [
            local_idx_map[(r.worker_id, r.graph_idx)]
            for r in manifest_df.itertuples()
        ]

        print(f"[ProcessLab] All {n_workers} shards written to {shards_dir}")
        return manifest_df

def register_graphs_job(graph_zoo_path, batch_name, batch_dir, queue='short', memory="8GB"):

    print("entered ProcessLab.register_graphs_job ")
    logs_dir = os.path.join(batch_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    python_exec = sys.executable
    memory_mb = _parse_memory_mb(memory)

    cmd_job = [
            "bsub",
            "-q", queue,
            "-J", f"batch_{batch_name}_register_graphs",
            "-o", os.path.join(logs_dir, "job_%J_register_graphs.out"), # Log stdout
            "-e", os.path.join(logs_dir, "job_%J_register_graphs.err"), # Log stderr
            "-R", f"rusage[mem={memory_mb}]",
            "-env", "PYTHONPATH=src",  
        ]

    cmd_process = [
            python_exec, "-u", "-m", "moran_process.core.population_graph",
            "--register",
            "--batch-dir", str(batch_dir),
            "--graph-zoo-path", str(graph_zoo_path)
        ]
    cmd = cmd_job + cmd_process
    # cmd = cmd_process + ['--job-index', '1']

    print(f"Submitting: {' '.join(cmd)}")
    subprocess.run(cmd)



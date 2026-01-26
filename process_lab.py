import pandas as pd
import time
import os 
import pickle
import math
import subprocess
from datetime import datetime
from pathlib import Path
from population_graph import PopulationGraph
from process_run import ProcessRun

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
            # Pre-compute adjacency for speed
            # adj_list = [list(graph_obj.graph.neighbors(n)) for n in range(graph_obj.N)]
            
            for r in r_values:
                # Run Repeats
                for _ in range(n_repeats):
                    # Initialize Engine
                    sim = ProcessRun(population_graph=graph_obj, selection_coefficient=r)
                    sim.initialize_random_mutant() # You might want to seed this for reproducibility
                    
                    # Run
                    raw_result = sim.run()
                    
                    # MERGE METADATA HERE
                    # This is the "secret sauce" to robust analysis
                    record = {
                        **graph_obj.metadata, # Expands: N, graph_name, depth...
                        "r": r,
                        **raw_result          # Expands: fixation, steps...
                    }
                    all_results.append(record)
                    if print_time: 
                        seconds = raw_result['duration']
                        print(f"Graph: {graph_obj.name}, r: {r}, Fixation: {raw_result['fixation']}, N: {graph_obj.number_of_nodes()}, Steps: {raw_result['steps']}, Time: {seconds:.4f}s")
        
        print('Done.')
        df = pd.DataFrame(all_results)
        
        # Save to CSV if output_path is provided
        if output_path:
            save_results(df, output_path)
        
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
    def submit_jobs(self, graphs_zoo, r_values, n_repeats=1000, n_jobs=50, queue="new-short", memory="2048"):
        """
        1. Dumps all graphs to 'graphs.pkl'
        2. Creates 'task_manifest.csv' (The Huge Table)
        3. Submits an LSF Job Array where each worker takes a 'chunk' of the table.
        """
        # 1. Prepare Batch Directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_dir = os.path.join(self.output_dir, f"batch_{timestamp}")
        
        # Create subdirs for logs and results
        logs_dir = os.path.join(batch_dir, "logs")
        results_dir = os.path.join(batch_dir, "results")
        os.makedirs(logs_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        print(f"--- Preparing Batch {timestamp} ---")

        # 2. Serialize Graphs (The Zoo)
        # We save the LIST of graphs to one file. It's faster for the cluster to read.
        zoo_path = os.path.join(batch_dir, "graphs.pkl")
        with open(zoo_path, "wb") as f:
            pickle.dump(graphs_zoo, f)
        print(f"Serialized {len(graphs_zoo)} graphs to {zoo_path}")

        # 3. Generate Task Manifest (The Huge Table)
        # We expand the loops into a list of rows
        manifest_df = _create_task_list(graphs_zoo, r_values, n_repeats)
        
        manifest_path = os.path.join(batch_dir, "task_manifest.csv")
        manifest_df.to_csv(manifest_path, index=False)
        print(f"Created Manifest with {len(manifest_df)} rows.")

        # 4. Calculate Slicing (Chunk Size)
        # If we have 50,000 tasks and 50 jobs, chunk_size = 1000.
        total_tasks = len(manifest_df)
        chunk_size = math.ceil(total_tasks / n_jobs)
        print(f"Splitting into {n_jobs} jobs. Chunk size: {chunk_size} tasks per worker.")

        # 5. Submit LSF Job Array
        # We pass the batch_dir and chunk_size. 
        # The worker will use its LSB_JOBINDEX to calculate its start/end.
        # start = (ID - 1) * chunk_size
        
        worker_script = "worker_wrapper.py" # Must be in current dir
        
        cmd = [
            "bsub",
            "-q", queue,
            "-J", f"batch_{timestamp}[1-{n_jobs}]", # Array 1..N
            "-o", os.path.join(logs_dir, "job_%J_%I.out"), # Log stdout
            "-e", os.path.join(logs_dir, "job_%J_%I.err"), # Log stderr
            "-R", f"rusage[mem={memory}]",
            "python", worker_script,
            "--batch-dir", batch_dir,
            "--chunk-size", str(chunk_size)
        ]

        print(f"Submitting: {' '.join(cmd)}")
        subprocess.run(cmd)
        print(f"Batch submitted! \n > Logs: {logs_dir} \n > Results: {results_dir}")


    @staticmethod
    def _create_task_list(graphs, r_values, n_repeats):
        """Create CSV task list for job array execution."""
        tasks = []
        task_id = 0
        
        for graph_idx, graph_obj in enumerate(graphs):
            for r in r_values:
                for repeat in range(n_repeats):
                    tasks.append({
                        'task_id': task_id,
                        'graph_idx': graph_idx,
                        'r_value': r,
                        'repeat': repeat
                    })
                    task_id += 1
        
        return pd.DataFrame(tasks)
        
    
    # def _build_lsf_command(self, n_jobs, graph_file, r_values, repeats_per_job, 
    #                       output_dir, queue, memory, walltime, temp_dir):
    #     """Generate LSF bsub command string."""
    #     os.makedirs(output_dir, exist_ok=True)
    #     logs_dir = Path(temp_dir) / "logs"
    #     logs_dir.mkdir(exist_ok=True)
        
    #     cmd_parts = [
    #         "bsub",
    #         "-J", f"processlab[1-{n_jobs}]",
    #         "-n", "1",
    #         "-M", memory,
    #         "-W", walltime,
    #         "-o", str(logs_dir / "job_%J_%I.out"),
    #         "-e", str(logs_dir / "job_%J_%I.err")
    #     ]
        
    #     if queue:
    #         cmd_parts.extend(["-q", queue])
        
    #     r_values_str = " ".join(map(str, r_values))
    #     worker_cmd = (
    #         f"python worker_wrapper.py "
    #         f"--graph-file {graph_file} "
    #         f"--r-values {r_values_str} "
    #         f"--repeats-per-job {repeats_per_job} "
    #         f"--output-dir {output_dir}"
    #     )
        
    #     cmd_parts.append(worker_cmd)
    #     return " ".join(cmd_parts)

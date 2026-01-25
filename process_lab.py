import pandas as pd
import time
import os 
from datetime import datetime
from population_graph import PopulationGraph
from process_run import ProcessRun
from hpc.serialization import GraphSerializer, SerializationError

class ProcessLab:
    """ Manages multiple process runs and stores their results"""
    def __init__(self):
        """
        """
    def run_comparative_study(self, graphs, r_values, n_repeats=100, print_time=True, output_path=None):
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
        total_sims = len(graphs) * len(r_values) * n_repeats
        
        print(f"--- Starting Study: {len(graphs)} Graphs x {len(r_values)} r-vals x {n_repeats}  = {total_sims} repeats ---")
        
        # We can optimize by converting graphs to adjacency lists ONCE
        for graph_obj in graphs:
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
     
     
    def submit_jobs(self, graphs: list[PopulationGraph], r_values: list[float], 
                   n_repeats: int, n_jobs: int | None = None, 
                   output_path: str | None = None, repeats_per_job: int | None = None,
                   queue: str | None = None, memory: str = "4GB", walltime: str = "2:00",
                   temp_dir: str | None = None) -> dict[str, Any]:
        """Submit comparative study as LSF job array using CSV task list."""
        
        # Calculate total simulations and job distribution
        total_simulations = len(graphs) * len(r_values) * n_repeats
        
        if repeats_per_job is None:
            repeats_per_job = max(1, total_simulations // (n_jobs or 10))
        if n_jobs is None:
            n_jobs = math.ceil(total_simulations / repeats_per_job)
        
        # Create temp directory
        if temp_dir is None:
            temp_dir = tempfile.mkdtemp(prefix="processlab_hpc_")
        else:
            os.makedirs(temp_dir, exist_ok=True)
        
        temp_path = Path(temp_dir)
        
        try:
            # Create task list CSV
            task_list = self._create_task_list(graphs, r_values, n_repeats)
            task_file = temp_path / "task_list.csv"
            task_list.to_csv(task_file, index=False)
            
            # Serialize graphs for reference
            graph_file = temp_path / "graphs.pkl"
            with open(graph_file, 'wb') as f:
                pickle.dump(graphs, f)
            
            # Generate and execute LSF command
            job_command = self._build_lsf_command(
                n_jobs, str(task_file), str(graph_file), repeats_per_job,
                str(temp_path / "results"), queue, memory, walltime, temp_dir
            )
            
            result = subprocess.run(job_command, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"LSF command failed: {result.stderr}")
            
            # Parse job ID
            import re
            job_id_match = re.search(r'Job <(\d+)>', result.stdout)
            job_id = job_id_match.group(1) if job_id_match else None
            
            tracking_info = {
                'job_id': job_id,
                'job_array_size': n_jobs,
                'total_simulations': total_simulations,
                'repeats_per_job': repeats_per_job,
                'temp_dir': temp_dir,
                'task_file': str(task_file),
                'graph_file': str(graph_file),
                'output_path': output_path,
                'submission_time': datetime.now().isoformat()
            }
            
            print(f"Successfully submitted LSF job array:")
            print(f"  Job ID: {job_id}")
            print(f"  Array size: {n_jobs} jobs")
            print(f"  Total simulations: {total_simulations}")
            print(f"  Simulations per job: ~{repeats_per_job}")
            print(f"  Task list: {task_file}")
            print(f"  Temporary directory: {temp_dir}")
            
            return tracking_info
            
        except Exception as e:
            print(f"ERROR: Job submission failed: {e}")
            if temp_dir and os.path.exists(temp_dir):
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise
    
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
        

    
    def _build_lsf_command(self, n_jobs, graph_file, r_values, repeats_per_job, 
                          output_dir, queue, memory, walltime, temp_dir):
        """Generate LSF bsub command string."""
        os.makedirs(output_dir, exist_ok=True)
        logs_dir = Path(temp_dir) / "logs"
        logs_dir.mkdir(exist_ok=True)
        
        cmd_parts = [
            "bsub",
            "-J", f"processlab[1-{n_jobs}]",
            "-n", "1",
            "-M", memory,
            "-W", walltime,
            "-o", str(logs_dir / "job_%J_%I.out"),
            "-e", str(logs_dir / "job_%J_%I.err")
        ]
        
        if queue:
            cmd_parts.extend(["-q", queue])
        
        r_values_str = " ".join(map(str, r_values))
        worker_cmd = (
            f"python worker_wrapper.py "
            f"--graph-file {graph_file} "
            f"--r-values {r_values_str} "
            f"--repeats-per-job {repeats_per_job} "
            f"--output-dir {output_dir}"
        )
        
        cmd_parts.append(worker_cmd)
        return " ".join(cmd_parts)

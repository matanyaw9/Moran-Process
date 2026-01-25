import pandas as pd
import time
import os 
import subprocess
import tempfile
import pickle
import math
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from population_graph import PopulationGraph
from process_run import ProcessRun

class ProcessLab:
    """Manages multiple process runs and stores their results"""
    
    def run_comparative_study(self, graphs, r_values, n_repeats=100, print_time=True, output_path=None):
        """
        Run comparative study across multiple graphs and selection coefficients.
        
        :param graphs: List of instantiated PopulationGraph objects
        :param r_values: List of floats (selection coefficients)
        :param n_repeats: Number of repetitions per configuration
        :param print_time: Whether to print timing information for each run
        :param output_path: Optional path to save results CSV
        :return: DataFrame with all results
        """
        all_results = []
        total_sims = len(graphs) * len(r_values) * n_repeats
        
        print(f"--- Starting Study: {len(graphs)} Graphs x {len(r_values)} r-vals x {n_repeats} = {total_sims} repeats ---")
        
        for graph_obj in graphs:
            for r in r_values:
                for _ in range(n_repeats):
                    sim = ProcessRun(population_graph=graph_obj, selection_coefficient=r)
                    sim.initialize_random_mutant()
                    raw_result = sim.run()
                    
                    record = {
                        **graph_obj.metadata,
                        "r": r,
                        **raw_result
                    }
                    all_results.append(record)
                    
                    if print_time: 
                        print(f"Graph: {graph_obj.name}, r: {r}, Fixation: {raw_result['fixation']}, "
                              f"N: {graph_obj.number_of_nodes()}, Steps: {raw_result['steps']}, "
                              f"Time: {raw_result['duration']:.4f}s")
        
        print('Done.')
        df = pd.DataFrame(all_results)
        
        if output_path:
            self._save_results(df, output_path)
        
        return df
    
    def _save_results(self, df, output_path):
        """Save results to CSV file, appending to existing file if it exists."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
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

    def submit_jobs(self, graphs: List[PopulationGraph], r_values: List[float], 
                   n_repeats: int, n_jobs: Optional[int] = None, 
                   output_path: Optional[str] = None, repeats_per_job: Optional[int] = None,
                   queue: Optional[str] = None, memory: str = "4GB", walltime: str = "2:00",
                   temp_dir: Optional[str] = None) -> Dict[str, Any]:
        """Submit comparative study as LSF job array."""
        
        # Calculate job distribution
        if repeats_per_job is None:
            repeats_per_job = max(1, n_repeats // (n_jobs or 10))
        if n_jobs is None:
            n_jobs = math.ceil(n_repeats / repeats_per_job)
        
        # Create temp directory
        if temp_dir is None:
            temp_dir = tempfile.mkdtemp(prefix="processlab_hpc_")
        else:
            os.makedirs(temp_dir, exist_ok=True)
        
        temp_path = Path(temp_dir)
        
        try:
            # Serialize graphs
            graph_file = temp_path / "graphs.pkl"
            with open(graph_file, 'wb') as f:
                pickle.dump(graphs, f)
            
            # Generate and execute LSF command
            job_command = self._build_lsf_command(
                n_jobs, str(graph_file), r_values, repeats_per_job,
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
                'total_repeats': n_repeats,
                'repeats_per_job': repeats_per_job,
                'temp_dir': temp_dir,
                'graph_file': str(graph_file),
                'output_path': output_path,
                'submission_time': datetime.now().isoformat()
            }
            
            print(f"Successfully submitted LSF job array:")
            print(f"  Job ID: {job_id}")
            print(f"  Array size: {n_jobs} jobs")
            print(f"  Total work: {len(graphs)} graphs × {len(r_values)} r-values × {n_repeats} repeats")
            print(f"  Temporary directory: {temp_dir}")
            
            return tracking_info
            
        except Exception as e:
            print(f"ERROR: Job submission failed: {e}")
            if temp_dir and os.path.exists(temp_dir):
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise
    
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
    
    def aggregate_results(self, temp_dir: str, output_path: str, cleanup: bool = True) -> Dict[str, Any]:
        """Merge individual worker CSV files into master dataset."""
        temp_path = Path(temp_dir)
        result_files = list(temp_path.glob("results_job_*.csv"))
        
        if not result_files:
            print(f"WARNING: No result files found in {temp_dir}")
            return {'total_files_found': 0, 'total_files_processed': 0, 'total_rows': 0}
        
        print(f"Found {len(result_files)} result files to aggregate")
        
        all_dataframes = []
        for csv_file in sorted(result_files):
            try:
                df = pd.read_csv(csv_file)
                if not df.empty:
                    all_dataframes.append(df)
                    print(f"Processed {csv_file.name}: {len(df)} rows")
            except Exception as e:
                print(f"ERROR: Failed to process {csv_file}: {e}")
        
        if not all_dataframes:
            raise RuntimeError("No valid CSV files could be processed")
        
        # Combine and save
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        combined_df = combined_df.drop_duplicates()  # Simple deduplication
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_csv(output_path, index=False)
        print(f"Successfully aggregated {len(combined_df)} rows to {output_path}")
        
        # Cleanup
        if cleanup:
            for file_path in result_files:
                file_path.unlink()
            if not any(temp_path.iterdir()):
                temp_path.rmdir()
        
        return {
            'total_files_found': len(result_files),
            'total_files_processed': len(all_dataframes),
            'total_rows': len(combined_df),
            'output_path': output_path
        }


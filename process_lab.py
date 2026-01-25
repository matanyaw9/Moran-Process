import pandas as pd
import time
import os 
import subprocess
import tempfile
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from population_graph import PopulationGraph
from process_run import ProcessRun
from hpc.serialization import GraphSerializer, SerializationError
from hpc.job_distribution import JobDistributor


class LSFConfig:
    """
    Configuration class for LSF parameters with validation and defaults.
    """
    
    DEFAULT_MEMORY = "4GB"
    DEFAULT_WALLTIME = "2:00"
    DEFAULT_CORES = 1
    
    def __init__(self, queue: Optional[str] = None, memory: str = DEFAULT_MEMORY, 
                 walltime: str = DEFAULT_WALLTIME, cores: int = DEFAULT_CORES, 
                 **additional_options):
        """
        Initialize LSF configuration with validation.
        
        :param queue: LSF queue name (optional)
        :param memory: Memory per job (e.g., "4GB", "8GB")
        :param walltime: Wall time limit (e.g., "2:00", "4:30")
        :param cores: Number of cores per job
        :param additional_options: Additional LSF parameters
        """
        self.queue = queue
        self.memory = memory
        self.walltime = walltime
        self.cores = cores
        self.additional_options = additional_options
        
        # Validate parameters
        self._validate()
    
    def _validate(self):
        """Validate all LSF configuration parameters."""
        # Validate memory format
        if not self.memory or not isinstance(self.memory, str):
            raise ValueError("Memory must be a non-empty string")
        
        memory_pattern = r'^\d+(\.\d+)?(GB|MB|KB)$'
        if not re.match(memory_pattern, self.memory, re.IGNORECASE):
            raise ValueError(f"Invalid memory format: {self.memory}. Use format like '4GB', '512MB'")
        
        # Validate walltime format
        if not self.walltime or not isinstance(self.walltime, str):
            raise ValueError("Walltime must be a non-empty string")
        
        walltime_pattern = r'^\d{1,2}:\d{2}(:\d{2})?$'
        if not re.match(walltime_pattern, self.walltime):
            raise ValueError(f"Invalid walltime format: {self.walltime}. Use format like '2:00' or '1:30:00'")
        
        # Validate queue name if provided
        if self.queue is not None:
            if not isinstance(self.queue, str) or not self.queue.strip():
                raise ValueError("Queue name must be a non-empty string if provided")
        
        # Validate cores
        if not isinstance(self.cores, int) or self.cores < 1:
            raise ValueError("Cores must be a positive integer")
    
    def to_bsub_args(self) -> List[str]:
        """
        Convert configuration to bsub command arguments.
        
        :return: List of bsub command arguments
        """
        args = []
        
        # Core resource requirements
        args.extend(["-n", str(self.cores)])
        args.extend(["-M", self.memory])
        args.extend(["-W", self.walltime])
        
        # Queue specification (optional)
        if self.queue:
            args.extend(["-q", self.queue])
        
        # Additional options
        for key, value in self.additional_options.items():
            if key.startswith('-'):
                args.extend([key, str(value)])
            else:
                args.extend([f"-{key}", str(value)])
        
        return args
    
    @classmethod
    def get_defaults(cls) -> 'LSFConfig':
        """
        Get default LSF configuration.
        
        :return: LSFConfig with default values
        """
        return cls()
    
    def __str__(self) -> str:
        """String representation of configuration."""
        config_str = f"LSFConfig(memory={self.memory}, walltime={self.walltime}, cores={self.cores}"
        if self.queue:
            config_str += f", queue={self.queue}"
        if self.additional_options:
            config_str += f", additional_options={self.additional_options}"
        config_str += ")"
        return config_str

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

    # --- HPC EXECUTION METHODS ---
    
    def submit_jobs(self, graphs: List[PopulationGraph], r_values: List[float], 
                   n_repeats: int, n_jobs: Optional[int] = None, 
                   output_path: Optional[str] = None, repeats_per_job: Optional[int] = None,
                   lsf_config: Optional[LSFConfig] = None, temp_dir: Optional[str] = None,
                   # Legacy parameters for backward compatibility
                   queue: Optional[str] = None, memory: str = "4GB", walltime: str = "2:00", 
                   **lsf_options) -> Dict[str, Any]:
        """
        Submit comparative study as LSF job array with configurable parameters.
        
        :param graphs: List of PopulationGraph objects
        :param r_values: List of selection coefficients  
        :param n_repeats: Total number of repeats across all jobs
        :param n_jobs: Number of LSF jobs (auto-calculated if None)
        :param output_path: Path for final aggregated results
        :param repeats_per_job: Repeats per job (auto-calculated if None)
        :param lsf_config: LSFConfig object with LSF parameters (preferred)
        :param temp_dir: Directory for temporary files (auto-created if None)
        :param queue: LSF queue name (legacy, use lsf_config instead)
        :param memory: Memory per job (legacy, use lsf_config instead)
        :param walltime: Wall time limit (legacy, use lsf_config instead)
        :param lsf_options: Additional LSF parameters (legacy, use lsf_config instead)
        :return: Job tracking information including job IDs
        """
        # Validate input parameters
        if not graphs:
            raise ValueError("Graphs list cannot be empty")
        if not r_values:
            raise ValueError("r_values list cannot be empty")
        if n_repeats <= 0:
            raise ValueError("n_repeats must be positive")
        
        # Handle LSF configuration - prefer lsf_config over legacy parameters
        if lsf_config is None:
            # Create LSFConfig from legacy parameters for backward compatibility
            lsf_config = LSFConfig(
                queue=queue,
                memory=memory,
                walltime=walltime,
                **lsf_options
            )
        
        # Initialize job distributor
        distributor = JobDistributor()
        
        # Calculate job distribution
        if n_jobs is None:
            n_jobs, calculated_repeats_per_job = distributor.calculate_job_count(
                len(graphs), len(r_values), n_repeats, repeats_per_job
            )
            if repeats_per_job is None:
                repeats_per_job = calculated_repeats_per_job
        else:
            if repeats_per_job is None:
                repeats_per_job = max(1, n_repeats // n_jobs)
        
        # Create temporary directory for job files
        if temp_dir is None:
            temp_dir = tempfile.mkdtemp(prefix="processlab_hpc_")
        else:
            os.makedirs(temp_dir, exist_ok=True)
        
        temp_path = Path(temp_dir)
        
        try:
            # Serialize graphs
            graph_file = temp_path / "graphs.pkl"
            serialization_metadata = self._serialize_graphs(graphs, str(graph_file))
            
            # Generate LSF command
            job_command = self._generate_lsf_command(
                n_jobs=n_jobs,
                graph_file=str(graph_file),
                r_values=r_values,
                repeats_per_job=repeats_per_job,
                output_dir=str(temp_path / "results"),
                lsf_config=lsf_config,
                temp_dir=temp_dir
            )
            
            # Execute LSF command
            job_info = self._execute_lsf_command(job_command)
            
            # Prepare tracking information
            tracking_info = {
                'job_id': job_info.get('job_id'),
                'job_array_size': n_jobs,
                'total_repeats': n_repeats,
                'repeats_per_job': repeats_per_job,
                'n_graphs': len(graphs),
                'n_r_values': len(r_values),
                'temp_dir': temp_dir,
                'graph_file': str(graph_file),
                'output_path': output_path,
                'submission_time': datetime.now().isoformat(),
                'lsf_command': job_command,
                'lsf_config': str(lsf_config),
                'serialization_metadata': serialization_metadata
            }
            
            print(f"Successfully submitted LSF job array:")
            print(f"  Job ID: {job_info.get('job_id')}")
            print(f"  Array size: {n_jobs} jobs")
            print(f"  Total work: {len(graphs)} graphs × {len(r_values)} r-values × {n_repeats} repeats")
            print(f"  Repeats per job: {repeats_per_job}")
            print(f"  LSF config: {lsf_config}")
            print(f"  Temporary directory: {temp_dir}")
            
            return tracking_info
            
        except Exception as e:
            print(f"ERROR: Job submission failed: {e}")
            # Clean up temporary files on failure
            if temp_dir and os.path.exists(temp_dir):
                import shutil
                try:
                    shutil.rmtree(temp_dir)
                except Exception as cleanup_error:
                    print(f"WARNING: Failed to clean up temporary directory: {cleanup_error}")
            raise
    
    def _validate_lsf_params(self, queue: Optional[str], memory: str, walltime: str):
        """
        Validate LSF configuration parameters.
        
        :param queue: LSF queue name (can be None)
        :param memory: Memory specification (e.g., "4GB", "512MB")
        :param walltime: Wall time specification (e.g., "2:00", "24:00")
        :raises: ValueError if parameters are invalid
        """
        # Validate memory format
        if not memory or not isinstance(memory, str):
            raise ValueError("Memory must be a non-empty string")
        
        # Check memory format (e.g., "4GB", "512MB")
        import re
        memory_pattern = r'^\d+(\.\d+)?(GB|MB|KB)$'
        if not re.match(memory_pattern, memory, re.IGNORECASE):
            raise ValueError(f"Invalid memory format: {memory}. Use format like '4GB', '512MB'")
        
        # Validate walltime format
        if not walltime or not isinstance(walltime, str):
            raise ValueError("Walltime must be a non-empty string")
        
        # Check walltime format (e.g., "2:00", "24:00", "1:30:00")
        walltime_pattern = r'^\d{1,2}:\d{2}(:\d{2})?$'
        if not re.match(walltime_pattern, walltime):
            raise ValueError(f"Invalid walltime format: {walltime}. Use format like '2:00' or '1:30:00'")
        
        # Validate queue name if provided
        if queue is not None:
            if not isinstance(queue, str) or not queue.strip():
                raise ValueError("Queue name must be a non-empty string if provided")
    
    def _generate_lsf_command(self, n_jobs: int, graph_file: str, r_values: List[float],
                             repeats_per_job: int, output_dir: str, lsf_config: LSFConfig,
                             temp_dir: str) -> str:
        """
        Generate LSF bsub command string with proper job array syntax.
        
        :param n_jobs: Number of jobs in the array
        :param graph_file: Path to serialized graph file
        :param r_values: List of selection coefficients
        :param repeats_per_job: Number of repeats per job
        :param output_dir: Directory for output files
        :param lsf_config: LSF configuration object
        :param temp_dir: Temporary directory for logs
        :return: Complete bsub command string
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create logs directory
        logs_dir = Path(temp_dir) / "logs"
        logs_dir.mkdir(exist_ok=True)
        
        # Build bsub command components
        cmd_parts = ["bsub"]
        
        # Job array specification
        cmd_parts.extend(["-J", f"processlab[1-{n_jobs}]"])
        
        # Add LSF configuration arguments
        cmd_parts.extend(lsf_config.to_bsub_args())
        
        # Output and error files
        cmd_parts.extend(["-o", str(logs_dir / "job_%J_%I.out")])
        cmd_parts.extend(["-e", str(logs_dir / "job_%J_%I.err")])
        
        # Worker script and arguments
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
    
    def _execute_lsf_command(self, command: str) -> Dict[str, Any]:
        """
        Execute LSF bsub command with error handling.
        
        :param command: Complete bsub command string
        :return: Dictionary with job information
        :raises: RuntimeError if LSF command fails
        """
        try:
            print(f"Executing LSF command: {command}")
            
            # Execute bsub command
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=60  # 60 second timeout for job submission
            )
            
            if result.returncode != 0:
                error_msg = f"LSF command failed with return code {result.returncode}"
                if result.stderr:
                    error_msg += f"\nSTDERR: {result.stderr}"
                if result.stdout:
                    error_msg += f"\nSTDOUT: {result.stdout}"
                raise RuntimeError(error_msg)
            
            # Parse job ID from bsub output
            job_id = self._parse_job_id(result.stdout)
            
            return {
                'job_id': job_id,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'return_code': result.returncode,
                'command': command
            }
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("LSF command timed out after 60 seconds")
        except subprocess.SubprocessError as e:
            raise RuntimeError(f"Failed to execute LSF command: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error executing LSF command: {e}")
    
    def _parse_job_id(self, bsub_output: str) -> Optional[str]:
        """
        Parse job ID from bsub command output.
        
        :param bsub_output: Output from bsub command
        :return: Job ID string or None if not found
        """
        import re
        
        # LSF typically outputs: "Job <12345> is submitted to queue <normal>."
        # or "Job <12345> is submitted to default queue <normal>."
        job_id_pattern = r'Job <(\d+)> is submitted'
        match = re.search(job_id_pattern, bsub_output)
        
        if match:
            return match.group(1)
        
        # Alternative pattern for different LSF versions
        alt_pattern = r'Job (\d+) submitted'
        alt_match = re.search(alt_pattern, bsub_output)
        
        if alt_match:
            return alt_match.group(1)
        
        print(f"WARNING: Could not parse job ID from bsub output: {bsub_output}")
        return None
    
    def _serialize_graphs(self, graphs, filepath):
        """
        Serialize graph zoo to pickle file with metadata preservation.
        
        :param graphs: List of PopulationGraph objects
        :param filepath: Output pickle file path
        :raises: SerializationError if graphs cannot be serialized
        :return: Serialization metadata dictionary
        """
        try:
            metadata = GraphSerializer.serialize_graphs(graphs, filepath)
            print(f"Successfully serialized {len(graphs)} graphs to {filepath}")
            return metadata
        except SerializationError as e:
            print(f"ERROR: Failed to serialize graphs: {e}")
            raise
        except Exception as e:
            print(f"ERROR: Unexpected error during graph serialization: {e}")
            raise SerializationError(f"Unexpected serialization error: {e}")
    
    def _deserialize_graphs(self, filepath):
        """
        Deserialize graph zoo from pickle file.
        
        :param filepath: Path to pickle file containing serialized graphs
        :return: Tuple of (graphs_list, metadata_dict)
        :raises: SerializationError if deserialization fails
        """
        try:
            graphs, metadata = GraphSerializer.deserialize_graphs(filepath)
            print(f"Successfully deserialized {len(graphs)} graphs from {filepath}")
            return graphs, metadata
        except SerializationError as e:
            print(f"ERROR: Failed to deserialize graphs: {e}")
            raise
        except Exception as e:
            print(f"ERROR: Unexpected error during graph deserialization: {e}")
            raise SerializationError(f"Unexpected deserialization error: {e}")
    
    def aggregate_results(self, temp_dir: str, output_path: str, cleanup: bool = True) -> Dict[str, Any]:
        """
        Merge individual worker CSV files into master dataset.
        
        :param temp_dir: Directory containing worker result files
        :param output_path: Path for final aggregated CSV
        :param cleanup: Whether to remove temporary files after aggregation
        :return: Summary statistics (total simulations, missing files)
        :raises: ValueError if parameters are invalid
        :raises: RuntimeError if aggregation fails
        """
        # Validate input parameters
        if not temp_dir or not isinstance(temp_dir, str):
            raise ValueError("temp_dir must be a non-empty string")
        if not output_path or not isinstance(output_path, str):
            raise ValueError("output_path must be a non-empty string")
        
        temp_path = Path(temp_dir)
        if not temp_path.exists():
            raise ValueError(f"Temporary directory does not exist: {temp_dir}")
        
        # Create output directory if it doesn't exist
        output_file_path = Path(output_path)
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Find all CSV result files in temp directory
            result_files = list(temp_path.glob("results_job_*.csv"))
            
            if not result_files:
                print(f"WARNING: No result files found in {temp_dir}")
                return {
                    'total_files_found': 0,
                    'total_files_processed': 0,
                    'total_rows': 0,
                    'missing_files': [],
                    'corrupted_files': [],
                    'output_path': output_path,
                    'aggregation_time': datetime.now().isoformat()
                }
            
            print(f"Found {len(result_files)} result files to aggregate")
            
            # Track aggregation statistics
            all_dataframes = []
            processed_files = []
            corrupted_files = []
            total_rows = 0
            
            # Process each CSV file
            for csv_file in sorted(result_files):
                try:
                    # Read CSV file
                    df = pd.read_csv(csv_file)
                    
                    if df.empty:
                        print(f"WARNING: Empty CSV file: {csv_file}")
                        continue
                    
                    # Validate required columns exist
                    required_columns = ['wl_hash', 'graph_name', 'r', 'fixation', 'steps']
                    missing_columns = [col for col in required_columns if col not in df.columns]
                    if missing_columns:
                        print(f"WARNING: CSV file missing required columns {missing_columns}: {csv_file}")
                        corrupted_files.append(str(csv_file))
                        continue
                    
                    all_dataframes.append(df)
                    processed_files.append(str(csv_file))
                    total_rows += len(df)
                    
                    print(f"Processed {csv_file.name}: {len(df)} rows")
                    
                except Exception as e:
                    print(f"ERROR: Failed to process {csv_file}: {e}")
                    corrupted_files.append(str(csv_file))
                    continue
            
            if not all_dataframes:
                raise RuntimeError("No valid CSV files could be processed")
            
            # Combine all dataframes
            print(f"Combining {len(all_dataframes)} dataframes...")
            combined_df = pd.concat(all_dataframes, ignore_index=True)
            
            # Apply duplicate prevention and data validation
            combined_df, validation_summary = self._validate_and_deduplicate_results(combined_df)
            
            # Save aggregated results
            combined_df.to_csv(output_path, index=False)
            print(f"Successfully aggregated {len(combined_df)} rows to {output_path}")
            
            # Cleanup temporary files if requested
            if cleanup:
                self._cleanup_temp_files(temp_path, processed_files)
            
            # Return summary statistics
            summary = {
                'total_files_found': len(result_files),
                'total_files_processed': len(processed_files),
                'total_rows': total_rows,
                'final_rows': len(combined_df),
                'missing_files': [],  # Will be populated by missing file detection
                'corrupted_files': corrupted_files,
                'output_path': output_path,
                'aggregation_time': datetime.now().isoformat(),
                'cleanup_performed': cleanup,
                **validation_summary  # Include validation results
            }
            
            # Generate and display detailed summary report
            self._generate_aggregation_summary_report(summary, temp_dir)
            
            return summary
            
        except Exception as e:
            print(f"ERROR: Result aggregation failed: {e}")
            raise RuntimeError(f"Failed to aggregate results: {e}")
    
    def _cleanup_temp_files(self, temp_path: Path, processed_files: List[str]):
        """
        Safely remove temporary files after successful aggregation.
        
        :param temp_path: Path to temporary directory
        :param processed_files: List of files that were successfully processed
        """
        try:
            import shutil
            
            # Only remove files that were successfully processed
            for file_path in processed_files:
                file_to_remove = Path(file_path)
                if file_to_remove.exists():
                    file_to_remove.unlink()
                    print(f"Removed temporary file: {file_to_remove.name}")
            
            # Remove logs directory if it exists and is empty
            logs_dir = temp_path / "logs"
            if logs_dir.exists() and not any(logs_dir.iterdir()):
                logs_dir.rmdir()
                print("Removed empty logs directory")
            
            # Remove temp directory if it's empty
            if temp_path.exists() and not any(temp_path.iterdir()):
                temp_path.rmdir()
                print(f"Removed empty temporary directory: {temp_path}")
            
        except Exception as e:
            print(f"WARNING: Failed to cleanup some temporary files: {e}")
            # Don't raise exception for cleanup failures
    
    def _validate_and_deduplicate_results(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Apply data validation and duplicate prevention to aggregated results.
        
        :param df: Combined DataFrame from all worker results
        :return: Tuple of (cleaned_df, validation_summary)
        """
        original_rows = len(df)
        validation_summary = {
            'original_rows': original_rows,
            'duplicates_removed': 0,
            'invalid_rows_removed': 0,
            'data_corruption_detected': False,
            'validation_errors': []
        }
        
        try:
            # 1. Data validation - check for corrupted or invalid data
            print("Validating data integrity...")
            
            # Check for required columns
            required_columns = ['wl_hash', 'graph_name', 'r', 'fixation', 'steps']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                validation_summary['validation_errors'].append(f"Missing required columns: {missing_columns}")
                validation_summary['data_corruption_detected'] = True
            
            # Check for null values in critical columns
            null_counts = df[required_columns].isnull().sum()
            if null_counts.any():
                null_columns = null_counts[null_counts > 0].to_dict()
                validation_summary['validation_errors'].append(f"Null values found: {null_columns}")
                # Remove rows with null values in critical columns
                before_null_removal = len(df)
                df = df.dropna(subset=required_columns)
                removed_null = before_null_removal - len(df)
                validation_summary['invalid_rows_removed'] += removed_null
                if removed_null > 0:
                    print(f"Removed {removed_null} rows with null values in critical columns")
            
            # Check for invalid data types
            if 'r' in df.columns:
                non_numeric_r = pd.to_numeric(df['r'], errors='coerce').isnull().sum()
                if non_numeric_r > 0:
                    validation_summary['validation_errors'].append(f"Non-numeric r values: {non_numeric_r}")
                    # Convert to numeric, invalid values become NaN
                    df['r'] = pd.to_numeric(df['r'], errors='coerce')
                    df = df.dropna(subset=['r'])
                    validation_summary['invalid_rows_removed'] += non_numeric_r
                    print(f"Removed {non_numeric_r} rows with invalid r values")
            
            if 'steps' in df.columns:
                non_numeric_steps = pd.to_numeric(df['steps'], errors='coerce').isnull().sum()
                if non_numeric_steps > 0:
                    validation_summary['validation_errors'].append(f"Non-numeric steps values: {non_numeric_steps}")
                    df['steps'] = pd.to_numeric(df['steps'], errors='coerce')
                    df = df.dropna(subset=['steps'])
                    validation_summary['invalid_rows_removed'] += non_numeric_steps
                    print(f"Removed {non_numeric_steps} rows with invalid steps values")
            
            # Check for invalid fixation values (should be boolean or 0/1)
            if 'fixation' in df.columns:
                invalid_fixation = ~df['fixation'].isin([True, False, 0, 1, 'True', 'False'])
                if invalid_fixation.any():
                    invalid_count = invalid_fixation.sum()
                    validation_summary['validation_errors'].append(f"Invalid fixation values: {invalid_count}")
                    df = df[~invalid_fixation]
                    validation_summary['invalid_rows_removed'] += invalid_count
                    print(f"Removed {invalid_count} rows with invalid fixation values")
            
            # 2. Duplicate prevention
            print("Checking for duplicate entries...")
            
            # Define columns that should be unique for each simulation run
            # We'll consider duplicates based on core simulation parameters
            duplicate_check_columns = ['wl_hash', 'graph_name', 'r']
            
            # Add additional columns if they exist (for more precise duplicate detection)
            optional_duplicate_columns = ['initial_mutants', 'job_id', 'repeat_id']
            for col in optional_duplicate_columns:
                if col in df.columns:
                    duplicate_check_columns.append(col)
            
            # Find duplicates
            before_dedup = len(df)
            
            # Method 1: Remove exact duplicates (all columns identical)
            df_exact_dedup = df.drop_duplicates()
            exact_duplicates_removed = before_dedup - len(df_exact_dedup)
            
            if exact_duplicates_removed > 0:
                print(f"Removed {exact_duplicates_removed} exact duplicate rows")
                validation_summary['duplicates_removed'] += exact_duplicates_removed
                df = df_exact_dedup
            
            # Method 2: Check for logical duplicates (same simulation parameters)
            # This is more conservative - we only remove if we're confident it's a duplicate
            if len(duplicate_check_columns) >= 3:  # Only if we have enough columns for reliable detection
                logical_duplicates = df.duplicated(subset=duplicate_check_columns, keep='first')
                logical_duplicate_count = logical_duplicates.sum()
                
                if logical_duplicate_count > 0:
                    print(f"Found {logical_duplicate_count} potential logical duplicates")
                    # For now, we'll be conservative and not remove logical duplicates automatically
                    # Instead, we'll report them for user review
                    validation_summary['validation_errors'].append(
                        f"Potential logical duplicates detected: {logical_duplicate_count} rows"
                    )
            
            # 3. Final validation summary
            final_rows = len(df)
            total_removed = original_rows - final_rows
            
            print(f"Data validation complete:")
            print(f"  Original rows: {original_rows}")
            print(f"  Final rows: {final_rows}")
            print(f"  Total removed: {total_removed}")
            print(f"  Duplicates removed: {validation_summary['duplicates_removed']}")
            print(f"  Invalid rows removed: {validation_summary['invalid_rows_removed']}")
            
            if validation_summary['validation_errors']:
                print(f"  Validation warnings: {len(validation_summary['validation_errors'])}")
                for error in validation_summary['validation_errors']:
                    print(f"    - {error}")
            
            return df, validation_summary
            
        except Exception as e:
            print(f"ERROR: Data validation failed: {e}")
            validation_summary['validation_errors'].append(f"Validation process failed: {e}")
            validation_summary['data_corruption_detected'] = True
            # Return original dataframe if validation fails
            return df, validation_summary
    
    def _generate_aggregation_summary_report(self, summary: Dict[str, Any], temp_dir: str):
        """
        Generate and display detailed aggregation status information.
        
        :param summary: Summary statistics from aggregation process
        :param temp_dir: Temporary directory path for additional analysis
        """
        print("\n" + "="*60)
        print("RESULT AGGREGATION SUMMARY REPORT")
        print("="*60)
        
        # Basic statistics
        print(f"Aggregation completed at: {summary['aggregation_time']}")
        print(f"Output file: {summary['output_path']}")
        print(f"Temporary directory: {temp_dir}")
        print()
        
        # File processing statistics
        print("FILE PROCESSING:")
        print(f"  Files found: {summary['total_files_found']}")
        print(f"  Files processed successfully: {summary['total_files_processed']}")
        print(f"  Files corrupted/failed: {len(summary['corrupted_files'])}")
        
        if summary['corrupted_files']:
            print("  Corrupted files:")
            for file_path in summary['corrupted_files']:
                print(f"    - {Path(file_path).name}")
        
        # Missing file detection and reporting
        missing_files = self._detect_missing_files(temp_dir, summary)
        summary['missing_files'] = missing_files
        
        if missing_files:
            print(f"  Missing files: {len(missing_files)}")
            print("  Missing file details:")
            for missing_info in missing_files:
                print(f"    - Job {missing_info['job_id']}: {missing_info['expected_filename']}")
        else:
            print("  Missing files: 0")
        
        print()
        
        # Data processing statistics
        print("DATA PROCESSING:")
        print(f"  Total rows collected: {summary['total_rows']}")
        print(f"  Final rows after validation: {summary['final_rows']}")
        
        if 'duplicates_removed' in summary:
            print(f"  Duplicate rows removed: {summary['duplicates_removed']}")
        if 'invalid_rows_removed' in summary:
            print(f"  Invalid rows removed: {summary['invalid_rows_removed']}")
        
        data_loss_pct = ((summary['total_rows'] - summary['final_rows']) / summary['total_rows'] * 100) if summary['total_rows'] > 0 else 0
        print(f"  Data retention: {100 - data_loss_pct:.1f}%")
        
        # Validation warnings
        if 'validation_errors' in summary and summary['validation_errors']:
            print()
            print("VALIDATION WARNINGS:")
            for i, error in enumerate(summary['validation_errors'], 1):
                print(f"  {i}. {error}")
        
        # Success/failure assessment
        print()
        print("AGGREGATION STATUS:")
        
        success_rate = (summary['total_files_processed'] / summary['total_files_found'] * 100) if summary['total_files_found'] > 0 else 0
        
        if success_rate == 100 and not summary['corrupted_files'] and not missing_files:
            status = "✓ COMPLETE SUCCESS"
            print(f"  {status}")
            print("  All files processed successfully with no data issues.")
        elif success_rate >= 90:
            status = "✓ SUCCESS WITH MINOR ISSUES"
            print(f"  {status}")
            print(f"  {success_rate:.1f}% of files processed successfully.")
        elif success_rate >= 70:
            status = "⚠ PARTIAL SUCCESS"
            print(f"  {status}")
            print(f"  {success_rate:.1f}% of files processed successfully.")
            print("  Review missing/corrupted files for potential data recovery.")
        else:
            status = "✗ SIGNIFICANT ISSUES"
            print(f"  {status}")
            print(f"  Only {success_rate:.1f}% of files processed successfully.")
            print("  Manual investigation recommended.")
        
        # Cleanup status
        if summary.get('cleanup_performed', False):
            print("  Temporary files cleaned up successfully.")
        else:
            print("  Temporary files preserved for manual review.")
        
        print("="*60)
        print()
    
    def _detect_missing_files(self, temp_dir: str, summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect missing result files based on expected job patterns.
        
        :param temp_dir: Temporary directory containing result files
        :param summary: Current aggregation summary
        :return: List of missing file information
        """
        missing_files = []
        
        try:
            temp_path = Path(temp_dir)
            
            # Get list of existing result files
            existing_files = list(temp_path.glob("results_job_*.csv"))
            existing_job_ids = set()
            
            # Extract job IDs from existing files
            import re
            job_id_pattern = r'results_job_(\d+)_'
            
            for file_path in existing_files:
                match = re.search(job_id_pattern, file_path.name)
                if match:
                    existing_job_ids.add(int(match.group(1)))
            
            # Try to determine expected job range
            # This is a heuristic approach since we don't have the original job submission info
            if existing_job_ids:
                min_job_id = min(existing_job_ids)
                max_job_id = max(existing_job_ids)
                
                # Check for gaps in the sequence
                expected_job_ids = set(range(min_job_id, max_job_id + 1))
                missing_job_ids = expected_job_ids - existing_job_ids
                
                for job_id in sorted(missing_job_ids):
                    missing_files.append({
                        'job_id': job_id,
                        'expected_filename': f'results_job_{job_id}_*.csv',
                        'reason': 'Missing from expected sequence'
                    })
            
            # Also check for files that couldn't be processed (in corrupted_files)
            for corrupted_file in summary.get('corrupted_files', []):
                file_path = Path(corrupted_file)
                match = re.search(job_id_pattern, file_path.name)
                if match:
                    job_id = int(match.group(1))
                    # Only add if not already in missing_files
                    if not any(mf['job_id'] == job_id for mf in missing_files):
                        missing_files.append({
                            'job_id': job_id,
                            'expected_filename': file_path.name,
                            'reason': 'File corrupted or unreadable'
                        })
            
        except Exception as e:
            print(f"WARNING: Could not detect missing files: {e}")
        
        return missing_files

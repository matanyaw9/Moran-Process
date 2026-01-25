"""
Worker execution logic for HPC compute nodes.

This module contains the ProcessLabWorker class for executing simulation batches
on compute nodes in HPC environments. It handles graph loading, work assignment
calculation, simulation execution, and result saving.

Requirements: 3.4, 3.5
"""

import os
import sys
import logging
import pandas as pd
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from population_graph import PopulationGraph
from process_run import ProcessRun
from hpc.serialization import GraphSerializer, SerializationError
from hpc.job_distribution import JobDistributor


class WorkerExecutionError(Exception):
    """Custom exception for worker execution failures."""
    
    def __init__(self, message: str, job_id: Optional[int] = None, 
                 error_context: Optional[Dict[str, Any]] = None):
        """
        Initialize WorkerExecutionError with detailed context.
        
        Args:
            message: Error description
            job_id: LSF job ID for context
            error_context: Additional context information
        """
        super().__init__(message)
        self.job_id = job_id
        self.error_context = error_context or {}
        self.timestamp = datetime.now().isoformat()


class WorkerErrorHandler:
    """
    Centralized error handling and logging for worker operations.
    
    Provides comprehensive error classification, logging, and recovery
    mechanisms for different types of worker failures.
    
    Requirements: 3.6, 7.2
    """
    
    # Error classification constants
    RECOVERABLE_ERRORS = (
        ConnectionError,
        TimeoutError,
        OSError,  # File system temporary issues
        MemoryError  # Might be recoverable with smaller batch
    )
    
    FATAL_ERRORS = (
        SerializationError,
        ValueError,  # Invalid parameters
        ImportError,  # Missing dependencies
        KeyError,  # Missing required data
        AttributeError  # Code structure issues
    )
    
    def __init__(self, job_id: Optional[int] = None):
        """
        Initialize error handler with job context.
        
        Args:
            job_id: LSF job ID for error context
        """
        self.job_id = job_id
        self.logger = logging.getLogger(f'worker_error_handler_{job_id or "unknown"}')
        self.error_count = 0
        self.error_history: List[Dict[str, Any]] = []
    
    def classify_error(self, error: Exception) -> str:
        """
        Classify error as recoverable or fatal.
        
        Args:
            error: Exception to classify
            
        Returns:
            'recoverable' or 'fatal'
        """
        if isinstance(error, self.RECOVERABLE_ERRORS):
            return 'recoverable'
        elif isinstance(error, self.FATAL_ERRORS):
            return 'fatal'
        elif isinstance(error, WorkerExecutionError):
            # Check if underlying error is recoverable
            if hasattr(error, '__cause__') and error.__cause__:
                return self.classify_error(error.__cause__)
            return 'fatal'  # Default for WorkerExecutionError
        else:
            # Unknown errors are treated as fatal for safety
            return 'fatal'
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None, 
                  operation: str = "unknown") -> Dict[str, Any]:
        """
        Log error with comprehensive context and classification.
        
        Args:
            error: Exception that occurred
            context: Additional context information
            operation: Description of operation that failed
            
        Returns:
            Error record dictionary
        """
        self.error_count += 1
        context = context or {}
        
        error_record = {
            'error_id': self.error_count,
            'job_id': self.job_id,
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'classification': self.classify_error(error),
            'context': context,
            'traceback': traceback.format_exc()
        }
        
        self.error_history.append(error_record)
        
        # Log with appropriate level based on classification
        if error_record['classification'] == 'recoverable':
            self.logger.warning(f"RECOVERABLE ERROR in {operation}: {error}")
            self.logger.warning(f"Error context: {context}")
        else:
            self.logger.error(f"FATAL ERROR in {operation}: {error}")
            self.logger.error(f"Error context: {context}")
            self.logger.error(f"Full traceback:\n{error_record['traceback']}")
        
        return error_record
    
    def handle_error(self, error: Exception, context: Dict[str, Any] = None,
                    operation: str = "unknown", raise_on_fatal: bool = True) -> bool:
        """
        Handle error with logging and classification.
        
        Args:
            error: Exception that occurred
            context: Additional context information
            operation: Description of operation that failed
            raise_on_fatal: Whether to re-raise fatal errors
            
        Returns:
            True if error is recoverable, False if fatal
            
        Raises:
            WorkerExecutionError: If error is fatal and raise_on_fatal is True
        """
        error_record = self.log_error(error, context, operation)
        
        is_recoverable = error_record['classification'] == 'recoverable'
        
        if not is_recoverable and raise_on_fatal:
            raise WorkerExecutionError(
                f"Fatal error in {operation}: {error}",
                job_id=self.job_id,
                error_context=error_record
            )
        
        return is_recoverable
    
    def get_error_summary(self) -> Dict[str, Any]:
        """
        Get summary of all errors encountered.
        
        Returns:
            Dictionary with error statistics and history
        """
        if not self.error_history:
            return {'total_errors': 0, 'recoverable': 0, 'fatal': 0, 'history': []}
        
        recoverable_count = sum(1 for e in self.error_history 
                               if e['classification'] == 'recoverable')
        fatal_count = len(self.error_history) - recoverable_count
        
        return {
            'total_errors': len(self.error_history),
            'recoverable': recoverable_count,
            'fatal': fatal_count,
            'history': self.error_history,
            'job_id': self.job_id
        }


class ProcessLabWorker:
    """
    Handles job execution logic on compute nodes.
    
    This class manages the complete workflow for a single worker job:
    1. Load serialized graphs from file
    2. Calculate work assignment based on job index
    3. Execute assigned simulation batch
    4. Save results to unique CSV file
    
    Enhanced with comprehensive error handling and logging.
    
    Requirements: 3.4, 3.5, 3.6, 7.2
    """
    
    def __init__(self, args: argparse.Namespace):
        """
        Initialize ProcessLabWorker with parsed command-line arguments.
        
        Args:
            args: Parsed arguments from argparse containing:
                - graph_file: Path to serialized graph zoo
                - r_values: List of selection coefficients
                - repeats_per_job: Number of repeats to execute
                - output_dir: Directory for output files
                - verbose: Enable verbose logging (optional)
        """
        self.args = args
        self.logger = logging.getLogger(f'processlab_worker')
        
        # Initialize error handler
        self.error_handler = WorkerErrorHandler()
        
        # Initialize core attributes
        self.graphs: Optional[List[PopulationGraph]] = None
        self.graph_metadata: Optional[Dict[str, Any]] = None
        self.job_distributor = JobDistributor()
        
        # Work assignment attributes (set during execute_job)
        self.job_index: Optional[int] = None
        self.graph_index: Optional[int] = None
        self.r_index: Optional[int] = None
        self.repeat_start: Optional[int] = None
        self.repeat_end: Optional[int] = None
        self.assigned_graph: Optional[PopulationGraph] = None
        self.assigned_r_value: Optional[float] = None
        
        self.logger.info("ProcessLabWorker initialized successfully")
    
    def load_graphs(self) -> None:
        """
        Load serialized graphs from pickle file with comprehensive error handling.
        
        Raises:
            WorkerExecutionError: If graph loading fails
        """
        try:
            self.logger.info(f"Loading graphs from: {self.args.graph_file}")
            
            # Validate file before attempting to load
            graph_path = Path(self.args.graph_file)
            if not graph_path.exists():
                raise FileNotFoundError(f"Graph file not found: {self.args.graph_file}")
            
            if graph_path.stat().st_size == 0:
                raise ValueError(f"Graph file is empty: {self.args.graph_file}")
            
            self.graphs, self.graph_metadata = GraphSerializer.deserialize_graphs(
                self.args.graph_file
            )
            
            # Validate loaded data
            if not self.graphs:
                raise ValueError("No graphs loaded from file")
            
            self.logger.info(f"Successfully loaded {len(self.graphs)} graphs")
            self.logger.debug(f"Graph metadata: {self.graph_metadata}")
            
            # Log graph details if verbose
            if hasattr(self.args, 'verbose') and self.args.verbose:
                for i, graph in enumerate(self.graphs):
                    self.logger.debug(f"Graph {i}: {graph.name} (N={graph.N})")
            
        except Exception as e:
            context = {
                'graph_file': self.args.graph_file,
                'file_exists': Path(self.args.graph_file).exists(),
                'operation_stage': 'graph_loading'
            }
            self.error_handler.handle_error(e, context, "load_graphs")
    
    def calculate_work_assignment(self, job_index: int) -> Tuple[int, int, int, int]:
        """
        Calculate work assignment for this job with enhanced error handling.
        
        Args:
            job_index: LSF job index (1-based)
            
        Returns:
            Tuple of (graph_index, r_index, repeat_start, repeat_end)
            
        Raises:
            WorkerExecutionError: If work assignment calculation fails
        """
        try:
            self.logger.info(f"Calculating work assignment for job {job_index}")
            
            # Validate inputs
            if not self.graphs:
                raise ValueError("Graphs not loaded - call load_graphs() first")
            
            if job_index < 1:
                raise ValueError(f"Invalid job_index: {job_index} (must be >= 1)")
            
            n_graphs = len(self.graphs)
            n_r_values = len(self.args.r_values)
            repeats_per_job = self.args.repeats_per_job
            
            self.logger.debug(f"Work parameters: {n_graphs} graphs, {n_r_values} r_values, "
                            f"{repeats_per_job} repeats_per_job")
            
            # Calculate assignment using job distributor
            graph_index, r_index, repeat_start, repeat_end = (
                self.job_distributor.calculate_work_assignment(
                    job_index, n_graphs, n_r_values, repeats_per_job
                )
            )
            
            # Validate assignment
            if graph_index >= n_graphs:
                raise ValueError(f"Invalid graph_index {graph_index} >= {n_graphs}")
            if r_index >= n_r_values:
                raise ValueError(f"Invalid r_index {r_index} >= {n_r_values}")
            
            # Store assignment details
            self.graph_index = graph_index
            self.r_index = r_index
            self.repeat_start = repeat_start
            self.repeat_end = repeat_end
            self.assigned_graph = self.graphs[graph_index]
            self.assigned_r_value = self.args.r_values[r_index]
            
            self.logger.info(f"Work assignment calculated:")
            self.logger.info(f"  Graph: {self.assigned_graph.name} (index {graph_index})")
            self.logger.info(f"  R-value: {self.assigned_r_value} (index {r_index})")
            self.logger.info(f"  Repeat range: {repeat_start} to {repeat_end-1}")
            self.logger.info(f"  Total repeats: {repeat_end - repeat_start}")
            
            return graph_index, r_index, repeat_start, repeat_end
            
        except Exception as e:
            context = {
                'job_index': job_index,
                'n_graphs': len(self.graphs) if self.graphs else 0,
                'n_r_values': len(self.args.r_values),
                'repeats_per_job': self.args.repeats_per_job,
                'operation_stage': 'work_assignment'
            }
            self.error_handler.handle_error(e, context, "calculate_work_assignment")
    
    def run_simulation_batch(self) -> List[Dict[str, Any]]:
        """
        Execute the assigned simulation batch with detailed error handling.
        
        Returns:
            List of result dictionaries, one per simulation repeat
            
        Raises:
            WorkerExecutionError: If simulation execution fails
        """
        try:
            if not self.assigned_graph or self.assigned_r_value is None:
                raise ValueError(
                    "Work assignment not calculated - call calculate_work_assignment() first"
                )
            
            results = []
            n_repeats = self.repeat_end - self.repeat_start
            failed_repeats = []
            
            self.logger.info(f"Starting simulation batch: {n_repeats} repeats")
            self.logger.info(f"Graph: {self.assigned_graph.name}, R-value: {self.assigned_r_value}")
            
            # Execute simulations with individual error handling
            for repeat_idx in range(self.repeat_start, self.repeat_end):
                try:
                    # Initialize simulation engine
                    sim = ProcessRun(
                        population_graph=self.assigned_graph,
                        selection_coefficient=self.assigned_r_value
                    )
                    
                    # Initialize with random mutant
                    initial_mutants = sim.initialize_random_mutant()
                    
                    # Run simulation
                    raw_result = sim.run()
                    
                    # Validate simulation result
                    if not isinstance(raw_result, dict):
                        raise ValueError(f"Invalid simulation result type: {type(raw_result)}")
                    
                    required_keys = ['fixation', 'steps', 'duration']
                    missing_keys = [key for key in required_keys if key not in raw_result]
                    if missing_keys:
                        raise ValueError(f"Missing required result keys: {missing_keys}")
                    
                    # Merge metadata with results (following ProcessLab pattern)
                    record = {
                        **self.assigned_graph.metadata,  # Graph metadata (N, name, etc.)
                        "r": self.assigned_r_value,
                        **raw_result,  # Simulation results (fixation, steps, etc.)
                        "job_id": self.job_index,
                        "repeat_id": repeat_idx
                    }
                    
                    results.append(record)
                    
                    # Log progress for verbose mode
                    if hasattr(self.args, 'verbose') and self.args.verbose:
                        self.logger.debug(
                            f"Repeat {repeat_idx}: fixation={raw_result['fixation']}, "
                            f"steps={raw_result['steps']}, duration={raw_result['duration']:.4f}s"
                        )
                    
                except Exception as e:
                    # Log individual repeat failure but continue with batch
                    context = {
                        'repeat_idx': repeat_idx,
                        'graph_name': self.assigned_graph.name,
                        'r_value': self.assigned_r_value,
                        'operation_stage': 'individual_simulation'
                    }
                    
                    # Check if error is recoverable
                    is_recoverable = not self.error_handler.handle_error(
                        e, context, f"simulation_repeat_{repeat_idx}", raise_on_fatal=False
                    )
                    
                    failed_repeats.append({
                        'repeat_idx': repeat_idx,
                        'error': str(e),
                        'recoverable': is_recoverable
                    })
                    
                    # If too many failures, abort batch
                    if len(failed_repeats) > n_repeats * 0.5:  # More than 50% failed
                        raise WorkerExecutionError(
                            f"Too many simulation failures: {len(failed_repeats)}/{n_repeats}",
                            job_id=self.job_index,
                            error_context={'failed_repeats': failed_repeats}
                        )
            
            # Check if we have any successful results
            if not results:
                raise WorkerExecutionError(
                    "All simulation repeats failed",
                    job_id=self.job_index,
                    error_context={'failed_repeats': failed_repeats}
                )
            
            self.logger.info(f"Completed simulation batch: {len(results)} successful results")
            
            if failed_repeats:
                self.logger.warning(f"Failed repeats: {len(failed_repeats)}/{n_repeats}")
                for failure in failed_repeats:
                    self.logger.warning(f"  Repeat {failure['repeat_idx']}: {failure['error']}")
            
            # Log summary statistics
            if results:
                fixations = sum(1 for r in results if r['fixation'])
                avg_steps = sum(r['steps'] for r in results) / len(results)
                avg_duration = sum(r['duration'] for r in results) / len(results)
                
                self.logger.info(f"Batch summary:")
                self.logger.info(f"  Successful: {len(results)}/{n_repeats} repeats")
                self.logger.info(f"  Fixations: {fixations}/{len(results)} ({100*fixations/len(results):.1f}%)")
                self.logger.info(f"  Average steps: {avg_steps:.1f}")
                self.logger.info(f"  Average duration: {avg_duration:.4f}s")
            
            return results
            
        except Exception as e:
            if isinstance(e, WorkerExecutionError):
                raise
            context = {
                'graph_name': self.assigned_graph.name if self.assigned_graph else 'unknown',
                'r_value': self.assigned_r_value,
                'repeat_range': f"{self.repeat_start}-{self.repeat_end}",
                'operation_stage': 'simulation_batch'
            }
            self.error_handler.handle_error(e, context, "run_simulation_batch")
    
    def save_results(self, results: List[Dict[str, Any]]) -> str:
        """
        Save simulation results to unique CSV file with error handling.
        
        Args:
            results: List of result dictionaries from simulation batch
            
        Returns:
            Path to saved CSV file
            
        Raises:
            WorkerExecutionError: If file saving fails
        """
        try:
            if not results:
                raise ValueError("No results to save")
            
            # Generate unique filename using job ID and timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results_job_{self.job_index}_{timestamp}.csv"
            output_path = Path(self.args.output_dir) / filename
            
            self.logger.info(f"Saving {len(results)} results to: {output_path}")
            
            # Ensure output directory exists
            try:
                output_path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                raise OSError(f"Cannot create output directory {output_path.parent}: {e}")
            
            # Convert results to DataFrame and save
            df = pd.DataFrame(results)
            
            # Validate DataFrame before saving
            if df.empty:
                raise ValueError("Results DataFrame is empty")
            
            # Save with error handling
            try:
                df.to_csv(output_path, index=False)
            except Exception as e:
                raise OSError(f"Failed to write CSV file: {e}")
            
            # Verify file was created successfully
            if not output_path.exists():
                raise OSError(f"Output file was not created: {output_path}")
            
            file_size = output_path.stat().st_size
            if file_size == 0:
                raise OSError(f"Output file is empty: {output_path}")
            
            # Verify file content by reading back a sample
            try:
                test_df = pd.read_csv(output_path, nrows=1)
                if test_df.empty:
                    raise ValueError("Saved CSV file appears to be empty")
            except Exception as e:
                self.logger.warning(f"Could not verify saved file content: {e}")
            
            self.logger.info(f"Successfully saved results ({file_size} bytes)")
            self.logger.info(f"Output file: {output_path}")
            
            return str(output_path)
            
        except Exception as e:
            context = {
                'n_results': len(results) if results else 0,
                'output_dir': self.args.output_dir,
                'job_index': self.job_index,
                'operation_stage': 'result_saving'
            }
            self.error_handler.handle_error(e, context, "save_results")
    
    def execute_job(self, job_index: int) -> Dict[str, Any]:
        """
        Execute complete job workflow with comprehensive error handling and logging.
        
        This is the main entry point that orchestrates the complete worker execution:
        1. Load graphs from serialized file
        2. Calculate work assignment based on job index
        3. Execute assigned simulation batch
        4. Save results to unique CSV file
        
        Enhanced with detailed error context and failure reporting.
        
        Args:
            job_index: LSF job index (1-based)
            
        Returns:
            Dictionary with job execution summary
            
        Raises:
            WorkerExecutionError: If any step of job execution fails
        """
        start_time = datetime.now()
        self.job_index = job_index
        self.error_handler.job_id = job_index
        
        try:
            self.logger.info(f"=== Starting job execution for job {job_index} ===")
            
            # Step 1: Load graphs
            self.logger.info("Step 1: Loading graphs")
            self.load_graphs()
            
            # Step 2: Calculate work assignment
            self.logger.info("Step 2: Calculating work assignment")
            graph_idx, r_idx, repeat_start, repeat_end = self.calculate_work_assignment(job_index)
            
            # Step 3: Execute simulation batch
            self.logger.info("Step 3: Executing simulation batch")
            results = self.run_simulation_batch()
            
            # Step 4: Save results
            self.logger.info("Step 4: Saving results")
            output_file = self.save_results(results)
            
            # Calculate execution time
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            # Get error summary
            error_summary = self.error_handler.get_error_summary()
            
            # Prepare execution summary
            summary = {
                'job_index': job_index,
                'graph_name': self.assigned_graph.name,
                'graph_index': graph_idx,
                'r_value': self.assigned_r_value,
                'r_index': r_idx,
                'repeat_start': repeat_start,
                'repeat_end': repeat_end,
                'n_repeats_executed': len(results),
                'n_fixations': sum(1 for r in results if r['fixation']),
                'output_file': output_file,
                'execution_time_seconds': execution_time,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'success': True,
                'error_summary': error_summary
            }
            
            self.logger.info(f"=== Job {job_index} completed successfully ===")
            self.logger.info(f"Execution time: {execution_time:.2f} seconds")
            self.logger.info(f"Results saved to: {output_file}")
            
            if error_summary['total_errors'] > 0:
                self.logger.info(f"Errors encountered: {error_summary['total_errors']} "
                               f"({error_summary['recoverable']} recoverable, "
                               f"{error_summary['fatal']} fatal)")
            
            return summary
            
        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            # Get final error summary
            error_summary = self.error_handler.get_error_summary()
            
            error_summary_dict = {
                'job_index': job_index,
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__,
                'execution_time_seconds': execution_time,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'error_summary': error_summary
            }
            
            self.logger.error(f"=== Job {job_index} failed ===")
            self.logger.error(f"Error: {e}")
            self.logger.error(f"Error type: {type(e).__name__}")
            self.logger.error(f"Execution time: {execution_time:.2f} seconds")
            
            if error_summary['total_errors'] > 0:
                self.logger.error(f"Total errors during execution: {error_summary['total_errors']}")
                self.logger.error(f"Error breakdown: {error_summary['recoverable']} recoverable, "
                                f"{error_summary['fatal']} fatal")
            
            raise WorkerExecutionError(
                f"Job {job_index} execution failed: {e}",
                job_id=job_index,
                error_context=error_summary_dict
            )
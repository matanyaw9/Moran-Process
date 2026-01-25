#!/usr/bin/env python3
"""
LSF worker script for ProcessLab distributed execution.

This script serves as the entry point for compute nodes in HPC environments.
It reads the LSF job index, loads serialized graphs, and executes assigned
simulation batches using the ProcessLabWorker class.

Usage:
    python worker_wrapper.py --graph-file graphs.pkl --r-values 1.0 1.1 1.2 
                            --repeats-per-job 25 --output-dir results/

Environment Variables:
    LSB_JOBINDEX: LSF job array index (1-based) used for work assignment

Requirements: 3.1, 3.2
"""

import argparse
import os
import sys
import logging
from pathlib import Path
from typing import List

# Add current directory to Python path to import local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hpc.worker import ProcessLabWorker


def setup_logging(job_index: int) -> logging.Logger:
    """
    Set up logging for the worker with job-specific configuration.
    
    Args:
        job_index: LSF job index for log identification
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(f'processlab_worker_{job_index}')
    logger.setLevel(logging.INFO)
    
    # Create console handler if not already exists
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            f'[Job {job_index}] %(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the worker script.
    
    Returns:
        Parsed arguments namespace
        
    Raises:
        SystemExit: If required arguments are missing or invalid
    """
    parser = argparse.ArgumentParser(
        description="ProcessLab HPC worker for distributed simulation execution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python worker_wrapper.py --graph-file graphs.pkl --r-values 1.0 1.1 1.2 \\
                            --repeats-per-job 25 --output-dir results/
    
    python worker_wrapper.py --graph-file /tmp/graphs.pkl --r-values 0.9 1.0 1.1 \\
                            --repeats-per-job 50 --output-dir /scratch/results/
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--graph-file', 
        required=True,
        type=str,
        help='Path to serialized graph zoo pickle file'
    )
    
    parser.add_argument(
        '--r-values', 
        required=True, 
        nargs='+', 
        type=float,
        help='List of selection coefficients to process (space-separated floats)'
    )
    
    parser.add_argument(
        '--repeats-per-job', 
        required=True, 
        type=int,
        help='Number of repeats this job should execute'
    )
    
    parser.add_argument(
        '--output-dir', 
        required=True,
        type=str,
        help='Directory for output CSV files'
    )
    
    # Optional arguments
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose logging output'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true', 
        help='Validate arguments and work assignment without running simulations'
    )
    
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace, logger: logging.Logger) -> None:
    """
    Validate parsed command-line arguments.
    
    Args:
        args: Parsed arguments from argparse
        logger: Logger instance for error reporting
        
    Raises:
        SystemExit: If validation fails
    """
    # Validate repeats_per_job
    if args.repeats_per_job <= 0:
        logger.error(f"repeats-per-job must be positive, got: {args.repeats_per_job}")
        sys.exit(1)
    
    # Validate graph file exists and is readable
    graph_path = Path(args.graph_file)
    if not graph_path.exists():
        logger.error(f"Graph file not found: {args.graph_file}")
        sys.exit(1)
    
    if not graph_path.is_file():
        logger.error(f"Graph file path is not a file: {args.graph_file}")
        sys.exit(1)
    
    if graph_path.stat().st_size == 0:
        logger.error(f"Graph file is empty: {args.graph_file}")
        sys.exit(1)
    
    # Validate r_values are reasonable
    for i, r_val in enumerate(args.r_values):
        if r_val <= 0:
            logger.error(f"r_values must be positive, got r_values[{i}] = {r_val}")
            sys.exit(1)
        if r_val > 100:
            logger.warning(f"Large r_value detected: r_values[{i}] = {r_val}")
    
    # Validate output directory can be created
    output_path = Path(args.output_dir)
    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError) as e:
        logger.error(f"Cannot create output directory {args.output_dir}: {e}")
        sys.exit(1)
    
    logger.info("Argument validation completed successfully")


def get_job_index(logger: logging.Logger) -> int:
    """
    Read LSF job index from environment variable with error handling.
    
    Args:
        logger: Logger instance for error reporting
        
    Returns:
        Job index as integer (1-based as used by LSF)
        
    Raises:
        SystemExit: If LSB_JOBINDEX is missing or invalid
    """
    try:
        job_index_str = os.environ.get('LSB_JOBINDEX')
        
        if job_index_str is None:
            logger.error("LSB_JOBINDEX environment variable not found")
            logger.error("This script must be run within an LSF job array")
            sys.exit(1)
        
        job_index = int(job_index_str)
        
        if job_index < 1:
            logger.error(f"Invalid LSB_JOBINDEX value: {job_index} (must be >= 1)")
            sys.exit(1)
        
        logger.info(f"Successfully read job index: {job_index}")
        return job_index
        
    except ValueError as e:
        logger.error(f"Failed to parse LSB_JOBINDEX as integer: {job_index_str}")
        logger.error(f"Error details: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error reading LSB_JOBINDEX: {e}")
        sys.exit(1)


def main() -> None:
    """
    Main entry point for the worker wrapper script.
    
    This function orchestrates the complete worker execution:
    1. Parse command-line arguments
    2. Read LSF job index from environment
    3. Set up logging
    4. Validate all inputs
    5. Create and execute ProcessLabWorker
    
    Enhanced with comprehensive error handling and detailed failure reporting.
    
    Exits with code 0 on success, 1 on any error.
    """
    # Parse arguments first (before logging setup to get job index)
    try:
        args = parse_arguments()
    except SystemExit as e:
        # argparse calls sys.exit on error, we catch it to add context
        print("ERROR: Failed to parse command-line arguments", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Unexpected error parsing arguments: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Get job index for logging setup
    # We need a temporary logger for this step
    temp_logger = logging.getLogger('temp')
    temp_handler = logging.StreamHandler(sys.stderr)
    temp_handler.setLevel(logging.ERROR)
    temp_logger.addHandler(temp_handler)
    temp_logger.setLevel(logging.ERROR)
    
    try:
        job_index = get_job_index(temp_logger)
    except SystemExit:
        # get_job_index already logged the error
        sys.exit(1)
    except Exception as e:
        temp_logger.error(f"Unexpected error getting job index: {e}")
        sys.exit(1)
    
    # Set up proper logging with job index
    try:
        logger = setup_logging(job_index)
    except Exception as e:
        print(f"ERROR: Failed to set up logging for job {job_index}: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Set logging level based on verbose flag
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    
    logger.info("=== ProcessLab HPC Worker Starting ===")
    logger.info(f"Job index: {job_index}")
    logger.info(f"Graph file: {args.graph_file}")
    logger.info(f"R-values: {args.r_values}")
    logger.info(f"Repeats per job: {args.repeats_per_job}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Log system information for debugging
    logger.debug(f"Python version: {sys.version}")
    logger.debug(f"Working directory: {os.getcwd()}")
    logger.debug(f"Python path: {sys.path[:3]}...")  # First 3 entries
    
    try:
        # Validate all arguments
        logger.info("Validating arguments...")
        validate_arguments(args, logger)
        
        # Handle dry-run mode
        if args.dry_run:
            logger.info("=== DRY RUN MODE ===")
            logger.info("Arguments validated successfully")
            logger.info("Work assignment would be calculated here")
            logger.info("No simulations will be executed")
            logger.info("=== DRY RUN COMPLETE ===")
            sys.exit(0)
        
        # Create and execute worker
        logger.info("Creating ProcessLabWorker instance")
        try:
            worker = ProcessLabWorker(args)
        except Exception as e:
            logger.error(f"Failed to create ProcessLabWorker: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            raise
        
        logger.info(f"Executing job {job_index}")
        try:
            execution_summary = worker.execute_job(job_index)
            
            # Log execution summary
            logger.info("=== Job Execution Summary ===")
            logger.info(f"Success: {execution_summary.get('success', False)}")
            logger.info(f"Execution time: {execution_summary.get('execution_time_seconds', 0):.2f}s")
            logger.info(f"Repeats executed: {execution_summary.get('n_repeats_executed', 0)}")
            logger.info(f"Fixations: {execution_summary.get('n_fixations', 0)}")
            logger.info(f"Output file: {execution_summary.get('output_file', 'unknown')}")
            
            # Log error summary if present
            error_summary = execution_summary.get('error_summary', {})
            if error_summary.get('total_errors', 0) > 0:
                logger.info(f"Errors during execution: {error_summary['total_errors']} "
                           f"({error_summary.get('recoverable', 0)} recoverable, "
                           f"{error_summary.get('fatal', 0)} fatal)")
            
        except Exception as e:
            logger.error(f"Worker execution failed: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            raise
        
        logger.info("=== ProcessLab HPC Worker Completed Successfully ===")
        sys.exit(0)
        
    except KeyboardInterrupt:
        logger.error("Worker execution interrupted by user (SIGINT)")
        logger.error("This may indicate a timeout or manual cancellation")
        sys.exit(1)
    except MemoryError:
        logger.error("Worker execution failed due to insufficient memory")
        logger.error("Consider reducing repeats-per-job or using more memory")
        sys.exit(1)
    except OSError as e:
        logger.error(f"Worker execution failed due to system error: {e}")
        logger.error("This may indicate file system issues or resource limits")
        sys.exit(1)
    except ImportError as e:
        logger.error(f"Worker execution failed due to missing dependency: {e}")
        logger.error("Check that all required Python packages are installed")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Worker execution failed with unexpected error: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        
        # Log additional context for debugging
        if hasattr(e, '__traceback__'):
            import traceback
            logger.error("Full traceback:")
            for line in traceback.format_exc().splitlines():
                logger.error(line)
        
        # Log system state for debugging
        logger.error("=== System State at Failure ===")
        logger.error(f"Working directory: {os.getcwd()}")
        logger.error(f"Available memory: {_get_memory_info()}")
        logger.error(f"Disk space: {_get_disk_space_info(args.output_dir if 'args' in locals() else '.')}")
        
        sys.exit(1)


def _get_memory_info() -> str:
    """Get basic memory information for debugging."""
    try:
        import psutil
        memory = psutil.virtual_memory()
        return f"{memory.available / (1024**3):.1f}GB available / {memory.total / (1024**3):.1f}GB total"
    except ImportError:
        return "psutil not available"
    except Exception as e:
        return f"error getting memory info: {e}"


def _get_disk_space_info(path: str) -> str:
    """Get basic disk space information for debugging."""
    try:
        import shutil
        total, used, free = shutil.disk_usage(path)
        return f"{free / (1024**3):.1f}GB free / {total / (1024**3):.1f}GB total"
    except Exception as e:
        return f"error getting disk info: {e}"


if __name__ == "__main__":
    main()
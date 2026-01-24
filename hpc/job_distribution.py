"""
Job distribution and work assignment algorithms for HPC execution.

This module contains algorithms for:
- Calculating optimal job counts based on graphs, r_values, and repeats
- Distributing work across LSF job arrays
- Mapping job indices to specific (graph, r_value, repeat_range) assignments
"""

import math
from typing import Tuple, Optional


class JobDistributor:
    """
    Handles job distribution and work assignment calculations for HPC execution.
    """
    
    def __init__(self):
        """Initialize JobDistributor."""
        pass
    
    def calculate_job_count(self, n_graphs: int, n_r_values: int, n_repeats: int, 
                           repeats_per_job: Optional[int] = None) -> Tuple[int, int]:
        """
        Calculate total number of jobs needed based on graphs, r_values, and repeats.
        
        Args:
            n_graphs: Number of graphs in the study
            n_r_values: Number of r_values (selection coefficients) 
            n_repeats: Total number of repeats requested
            repeats_per_job: Target repeats per job (auto-calculated if None)
            
        Returns:
            Tuple of (n_jobs, actual_repeats_per_job)
            
        Raises:
            ValueError: If parameters are invalid
        """
        # Validate input parameters
        if n_graphs <= 0:
            raise ValueError("Number of graphs must be positive")
        if n_r_values <= 0:
            raise ValueError("Number of r_values must be positive")
        if n_repeats <= 0:
            raise ValueError("Number of repeats must be positive")
        if repeats_per_job is not None and repeats_per_job <= 0:
            raise ValueError("Repeats per job must be positive")
        
        # Calculate total combinations
        total_combinations = n_graphs * n_r_values
        
        # Handle edge case: single job scenario
        if n_repeats == 1 or total_combinations == 1:
            if repeats_per_job is None:
                return 1, n_repeats
            else:
                # Even with specified repeats_per_job, we need at least 1 job
                n_jobs = max(1, math.ceil(n_repeats / repeats_per_job))
                actual_repeats_per_job = math.ceil(n_repeats / n_jobs)
                return n_jobs, actual_repeats_per_job
        
        # Auto-calculate repeats_per_job if not specified
        if repeats_per_job is None:
            # Default strategy: aim for reasonable job count
            # Use heuristic: target around 10-50 repeats per job for good balance
            if n_repeats <= 50:
                repeats_per_job = n_repeats  # Single job for small studies
            elif n_repeats <= 500:
                repeats_per_job = 10  # 10 repeats per job for medium studies
            else:
                repeats_per_job = 25  # 25 repeats per job for large studies
        
        # Calculate number of jobs needed
        # We need enough jobs to handle all repeats across all combinations
        n_jobs = math.ceil(n_repeats / repeats_per_job)
        
        # Ensure we have at least one job per combination if needed
        # This handles cases where repeats_per_job is very large
        min_jobs_needed = math.ceil(n_repeats / repeats_per_job)
        n_jobs = max(n_jobs, min_jobs_needed)
        
        # Calculate actual repeats per job (may be less than requested due to rounding)
        actual_repeats_per_job = math.ceil(n_repeats / n_jobs)
        
        return n_jobs, actual_repeats_per_job
    
    def distribute_work(self, n_graphs: int, n_r_values: int, n_repeats: int, n_jobs: int) -> dict:
        """
        Distribute repeats across jobs optimally using round-robin assignment.
        
        This method distributes the total repeats across all (graph, r_value) combinations
        using a round-robin approach to ensure balanced work distribution.
        
        Args:
            n_graphs: Number of graphs in the study
            n_r_values: Number of r_values (selection coefficients)
            n_repeats: Total number of repeats to distribute
            n_jobs: Number of jobs to distribute work across
            
        Returns:
            Dictionary with distribution information:
            {
                'total_combinations': int,
                'repeats_per_job': int,
                'total_work_units': int,
                'distribution_strategy': str
            }
            
        Raises:
            ValueError: If parameters are invalid
        """
        # Validate input parameters
        if n_graphs <= 0:
            raise ValueError("Number of graphs must be positive")
        if n_r_values <= 0:
            raise ValueError("Number of r_values must be positive")
        if n_repeats <= 0:
            raise ValueError("Number of repeats must be positive")
        if n_jobs <= 0:
            raise ValueError("Number of jobs must be positive")
        
        # Calculate total combinations
        total_combinations = n_graphs * n_r_values
        
        # Calculate repeats per job (rounded up to ensure all repeats are covered)
        repeats_per_job = math.ceil(n_repeats / n_jobs)
        
        # Calculate total work units that will be processed
        # This may be slightly more than n_repeats due to rounding
        total_work_units = n_jobs * repeats_per_job
        
        # Determine distribution strategy
        if total_combinations >= n_jobs:
            strategy = "round_robin_combinations"
        else:
            strategy = "multiple_jobs_per_combination"
        
        return {
            'total_combinations': total_combinations,
            'repeats_per_job': repeats_per_job,
            'total_work_units': total_work_units,
            'distribution_strategy': strategy,
            'excess_work': total_work_units - n_repeats
        }
    
    def calculate_work_assignment(self, job_index: int, n_graphs: int, n_r_values: int, 
                                 repeats_per_job: int) -> Tuple[int, int, int, int]:
        """
        Calculate work assignment for a specific job using round-robin distribution.
        
        Maps job index to specific (graph, r_value, repeat_range) assignment using
        round-robin approach across the Cartesian product of (graphs × r_values).
        
        Args:
            job_index: LSF job index (1-based, as used by LSB_JOBINDEX)
            n_graphs: Number of graphs in the study
            n_r_values: Number of r_values (selection coefficients)
            repeats_per_job: Number of repeats this job should execute
            
        Returns:
            Tuple of (graph_index, r_index, repeat_start, repeat_end)
            - graph_index: 0-based index into graphs list
            - r_index: 0-based index into r_values list  
            - repeat_start: Starting repeat number (0-based)
            - repeat_end: Ending repeat number (exclusive, for range())
            
        Raises:
            ValueError: If parameters are invalid
        """
        # Validate input parameters
        if job_index < 1:
            raise ValueError("Job index must be >= 1 (LSF uses 1-based indexing)")
        if n_graphs <= 0:
            raise ValueError("Number of graphs must be positive")
        if n_r_values <= 0:
            raise ValueError("Number of r_values must be positive")
        if repeats_per_job <= 0:
            raise ValueError("Repeats per job must be positive")
        
        # Convert to 0-based indexing for calculations
        job_idx_0based = job_index - 1
        
        # Calculate total combinations
        total_combinations = n_graphs * n_r_values
        
        # Determine which (graph, r_value) combination this job handles
        # Use round-robin assignment across combinations
        combination_index = job_idx_0based % total_combinations
        
        # Map combination index to (graph_index, r_index)
        graph_index = combination_index // n_r_values
        r_index = combination_index % n_r_values
        
        # Calculate repeat range for this job
        # Jobs cycle through combinations, so we need to determine which "round" this is
        round_number = job_idx_0based // total_combinations
        repeat_start = round_number * repeats_per_job
        repeat_end = repeat_start + repeats_per_job
        
        # Validate that indices are within bounds
        if graph_index >= n_graphs:
            raise ValueError(f"Calculated graph_index {graph_index} >= n_graphs {n_graphs}")
        if r_index >= n_r_values:
            raise ValueError(f"Calculated r_index {r_index} >= n_r_values {n_r_values}")
        
        return graph_index, r_index, repeat_start, repeat_end
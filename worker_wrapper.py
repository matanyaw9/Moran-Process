#!/usr/bin/env python3
"""
Simple LSF worker script for ProcessLab distributed execution.
"""

import argparse
import os
import sys
import pickle
import pandas as pd
import math
from datetime import datetime
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from population_graph import PopulationGraph
from process_run import ProcessRun


def main():
    parser = argparse.ArgumentParser(description="ProcessLab HPC worker")
    parser.add_argument('--graph-file', required=True, help='Path to serialized graphs')
    parser.add_argument('--r-values', required=True, nargs='+', type=float, help='Selection coefficients')
    parser.add_argument('--repeats-per-job', required=True, type=int, help='Repeats per job')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    
    args = parser.parse_args()
    
    # Get job index from LSF environment
    job_index = int(os.environ['LSB_JOBINDEX'])
    
    print(f"Job {job_index}: Loading graphs from {args.graph_file}")
    
    # Load graphs
    with open(args.graph_file, 'rb') as f:
        graphs = pickle.load(f)
    
    # Calculate work assignment using simple round-robin
    n_graphs = len(graphs)
    n_r_values = len(args.r_values)
    total_combinations = n_graphs * n_r_values
    
    # Map job to (graph, r_value) combination
    combination_index = (job_index - 1) % total_combinations
    graph_index = combination_index // n_r_values
    r_index = combination_index % n_r_values
    
    assigned_graph = graphs[graph_index]
    assigned_r_value = args.r_values[r_index]
    
    print(f"Job {job_index}: Processing {assigned_graph.name} with r={assigned_r_value}")
    
    # Run simulations
    results = []
    for repeat_idx in range(args.repeats_per_job):
        try:
            sim = ProcessRun(population_graph=assigned_graph, selection_coefficient=assigned_r_value)
            sim.initialize_random_mutant()
            raw_result = sim.run()
            
            record = {
                **assigned_graph.metadata,
                "r": assigned_r_value,
                **raw_result,
                "job_id": job_index,
                "repeat_id": repeat_idx
            }
            results.append(record)
            
        except Exception as e:
            print(f"Job {job_index}: Error in repeat {repeat_idx}: {e}")
    
    # Save results
    if results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results_job_{job_index}_{timestamp}.csv"
        output_path = Path(args.output_dir) / filename
        
        os.makedirs(args.output_dir, exist_ok=True)
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        
        print(f"Job {job_index}: Saved {len(results)} results to {output_path}")
    else:
        print(f"Job {job_index}: No results to save")


if __name__ == "__main__":
    main()
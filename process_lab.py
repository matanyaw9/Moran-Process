import pandas as pd
import time
import os 
from datetime import datetime
from population_graph import PopulationGraph
from process_run import ProcessRun

class ProcessLab:
    """ Manages multiple process runs and stores their results"""
    def __init__(self):
        """
        """
    def run_comparative_study(self, graphs, r_values, n_repeats=100, print_time=False):
        """
        :param graphs: List of instantiated PopulationGraph objects
        :param r_values: List of floats
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
        return pd.DataFrame(all_results)

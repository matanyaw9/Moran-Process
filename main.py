# main.py
from population_graph import PopulationGraph
from process_lab import ProcessLab
import pandas as pd
import os
import time

OUTPUT_FILE_NAME = '500_nodes_study_results.csv'


    # 1. DEFINE THE GRAPH ZOO
    # We instantiate them here so we can inspect them before running
graph_zoo = [
    PopulationGraph.complete_graph(N=511),
    # PopulationGraph.cycle_graph(N=31),
    PopulationGraph.mammalian_lung_graph(branching_factor=2, depth=8), # N = 511
    PopulationGraph.avian_graph(n_rods=15, rod_length=34),
    PopulationGraph.fish_graph(n_rods=7, rod_length=24)
    
]

def main():
    # 2. DEFINE PARAMETERS
    r_values = [1.0, 1.1]
    # r_values = [1.1]

    repeats = 100  # Higher repeats for smoother stats
    
    # 3. RUN EXPERIMENT
    lab = ProcessLab()
    df = lab.run_comparative_study(graph_zoo, r_values, n_repeats=repeats, print_time=True)
    
    # 4. SAVE & ANALYZE
    # The dataframe now automatically contains 'N', 'depth', 'graph_type', etc.
    data_dir = "simulation_data"
    df.to_csv(os.path.join(data_dir, OUTPUT_FILE_NAME), index=False)
    
if __name__ == "__main__":
    start_time = time.perf_counter()
    for graph in graph_zoo:
        print(f"Graph Name: {graph.name}\t\tGraph Size: {graph.number_of_nodes()}")
    
    main()

    end_time = time.perf_counter()
    print(f"Whole thing took {(end_time-start_time):.4f} seconds")

    

    # for graph in graph_zoo:
    #     graph.draw()

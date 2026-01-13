# main.py
from population_graph import PopulationGraph
from process_lab import ProcessLab
import pandas as pd
import os
import time

OUTPUT_FILE_NAME = 'version_check.csv'


    # 1. DEFINE THE GRAPH ZOO
    # We instantiate them here so we can inspect them before running
graph_zoo = [
    PopulationGraph.complete_graph(N=31),
    # PopulationGraph.cycle_graph(N=31),
    PopulationGraph.mammalian_lung_graph(branching_factor=2, depth=4), # N = 511
    PopulationGraph.avian_graph(n_rods=4, rod_length=7),
    PopulationGraph.fish_graph(n_rods=3, rod_length=3)
    
]

def main():
    # 1. DEFINE PARAMETERS
    r_values = [1.0, 1.1]
    repeats = 500  # Higher repeats for smoother stats
    
    # 2. RUN EXPERIMENT
    lab = ProcessLab()
    df = lab.run_comparative_study(graph_zoo, r_values, n_repeats=repeats, print_time=True)
    
    # 3. SAVE & ANALYZE
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

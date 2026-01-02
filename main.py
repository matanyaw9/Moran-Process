# main.py
from population_graph import PopulationGraph
from process_lab import ProcessLab
import pandas as pd
import os

    # 1. DEFINE THE GRAPH ZOO
    # We instantiate them here so we can inspect them before running
graph_zoo = [
    PopulationGraph.complete_graph(N=31),
    PopulationGraph.cycle_graph(N=31),
    PopulationGraph.mammalian_lung_graph(branching_factor=2, depth=4), # N ~ 30
    PopulationGraph.avian_graph(n_rods=4, rod_length=7),
    PopulationGraph.fish_graph(n_rods=3, rod_length=3)
    
]

def main():
    # 2. DEFINE PARAMETERS
    r_values = [1.0, 1.1, 1.2, 2.0]
    # r_values = [1.1]

    repeats = 1000  # Higher repeats for smoother stats
    
    # 3. RUN EXPERIMENT
    lab = ProcessLab()
    df = lab.run_comparative_study(graph_zoo, r_values, n_repeats=repeats)
    
    # 4. SAVE & ANALYZE
    # The dataframe now automatically contains 'N', 'depth', 'graph_type', etc.
    filename = "comparative_study_results.csv"
    data_dir = "simulation_data"
    df.to_csv(os.path.join(data_dir, filename), index=False)
    
if __name__ == "__main__":
    # main()
    

    for graph in graph_zoo:
        graph.draw()

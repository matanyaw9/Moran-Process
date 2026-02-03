# main.py

from population_graph import PopulationGraph
from process_lab import ProcessLab
import pandas as pd
import os
import time

EXPERIMENTS_CSV = 'respiratory_runs.csv'

    # 1. DEFINE THE GRAPH ZOO
    # We instantiate them here so we can inspect them before running
graph_zoo = [
    # PopulationGraph.complete_graph(N=31),
    # PopulationGraph.cycle_graph(N=31),
    PopulationGraph.mammalian_lung_graph(branching_factor=2, depth=4), # N = 511
    PopulationGraph.avian_graph(n_rods=4, rod_length=7),
    PopulationGraph.fish_graph(n_rods=3, rod_length=3)
]


def main():
    """
    Main experiment runner for random graphs.
    Similar structure to main.py but for random graphs.
    """
    
    # 1. DEFINE PARAMETERS
    BATCH_NAME = None
    n_nodes = list(range(30, 33))
    # edge_counts = list(range(29, 35))  
    n_graphs_per_combination = 500  # Number of random graphs per edge count
    r_values = [1.2 ]  # Same r values as main.py
    n_repeats = 10_000  # Same as main.py for consistency
    n_jobs = 1000
    
    # 2. GENERATE RANDOM GRAPHS
    print("="*60)
    print("RANDOM GRAPH EXPERIMENT")
    print("="*60)
    print(f"Configuration:")
    print(f"  Nodes per graph: {n_nodes}")
    print(f"  Edge counts: n_nodes -1 to +4")
    print(f"  Graphs per edge count: {n_graphs_per_combination}")
    print(f"  r values: {r_values}")
    print(f"  Repeats per configuration: {n_repeats}")
    n_random_configs = len(n_nodes) * 5 * len(r_values) * n_graphs_per_combination
    n_graps_total = n_random_configs + len(graph_zoo)    
    print(f"  In Total: {n_graps_total} graphs")
    print(f"  In Total: {n_graps_total * n_repeats} simulations")

    print("="*60)
    
    for nn in n_nodes: 
        edge_counts = range(nn-1, nn+5)
        for ne in edge_counts:
            for i in range(n_graphs_per_combination):
                graph_zoo.append(PopulationGraph.random_connected_graph(n_nodes=nn, 
                                                                    n_edges=ne, 
                                                                    name = f'random_n{nn}_e{ne}_{i}'))
    
    print(f"Number of graphs: {len(graph_zoo)}")
    # 3. DISPLAY GRAPH INFORMATION
    print("\n" + "="*60)
    print("GENERATED GRAPHS:")
    print("="*60)
    for graph in graph_zoo:
        print(f"Graph: {graph.name:30s} | Nodes: {graph.N:3d} | Edges: {graph.graph.number_of_edges():3d} | Density: {graph.graph.number_of_edges() / (graph.N * (graph.N - 1) / 2):.3f}")
    
    # 4. RUN EXPERIMENT AND SAVE RESULTS
    print("\n" + "="*60)
    print("RUNNING EXPERIMENTS")
    print("="*60)
    
    lab = ProcessLab()
    
    lab.submit_jobs(
        graph_zoo, 
        r_values, 
        n_repeats=n_repeats, 
        batch_name=BATCH_NAME,
        n_jobs=n_jobs
    )
    
if __name__ == "__main__":
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    print(f"Whole thing took {(end_time-start_time):.4f} seconds")


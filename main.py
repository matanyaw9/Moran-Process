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
    n_nodes = 31
    edge_counts = list(range(30, 32))  
    n_graphs_per_edge_count = 10  # Number of random graphs per edge count
    r_values = [1.0, 1.1, ]  # Same r values as main.py
    n_repeats = 100  # Same as main.py for consistency
    
    # 2. GENERATE RANDOM GRAPHS
    print("="*60)
    print("RANDOM GRAPH EXPERIMENT")
    print("="*60)
    print(f"Configuration:")
    print(f"  Nodes per graph: {n_nodes}")
    print(f"  Edge counts: {edge_counts}")
    print(f"  Graphs per edge count: {n_graphs_per_edge_count}")
    print(f"  r values: {r_values}")
    print(f"  Repeats per configuration: {n_repeats}")
    print("="*60)
    
    for n_edges in edge_counts:
        for i in range(n_graphs_per_edge_count):
            graph_zoo.append(PopulationGraph.random_connected_graph(n_nodes=n_nodes, 
                                                                    n_edges=n_edges, 
                                                                    name = f'random_n{n_nodes}_e{n_edges}_{i}'))
    
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
    
    df = lab.submit_jobs(
        graph_zoo, 
        r_values, 
        n_repeats=n_repeats, 
        batch_name="big_batch_test",
        n_jobs=100
    )
    
if __name__ == "__main__":
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    print(f"Whole thing took {(end_time-start_time):.4f} seconds")


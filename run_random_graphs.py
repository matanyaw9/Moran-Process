# run_random_graphs.py
# Script to run experiments on random graphs with 31 nodes and varying edge counts

from population_graph import PopulationGraph
from process_lab import ProcessLab
import pandas as pd
import os
import time
import numpy as np

EXPERIMENTS_CSV = 'random_graphs_experiments.csv'
GRAPH_DATABASE_CSV = 'simulation_data/graph_database.csv'

def generate_random_graphs(n_nodes=31, edge_counts=[30, 31, 32], n_graphs_per_edge_count=10):
    """
    Generate random connected graphs with specified node count and edge counts.
    
    Args:
        n_nodes (int): Number of nodes in each graph
        edge_counts (list): List of edge counts to generate graphs for
        n_graphs_per_edge_count (int): Number of random graphs to generate for each edge count
        
    Returns:
        list: List of PopulationGraph objects
    """
    graph_zoo = []
    
    print(f"Generating {len(edge_counts)} x {n_graphs_per_edge_count} = {len(edge_counts) * n_graphs_per_edge_count} random graphs...")
    counter = 0
    for edge_count in edge_counts:
        print(f"\nGenerating {n_graphs_per_edge_count} graphs with {n_nodes} nodes and {edge_count} edges...")
        
        for i in range(n_graphs_per_edge_count):
            # Use different seed for each graph to ensure variety
            seed = int(time.time() * 1000000) % (2**32) + i
            counter += 1
            
            try:
                graph = PopulationGraph.random_connected_graph(
                    n_nodes=n_nodes,
                    n_edges=edge_count,
                    name=f'random_{counter}',
                    seed=seed,
                    register_in_db=True  # Register in database
                )
                graph_zoo.append(graph)
                print(f"  Created: {graph.name} (WL hash: {graph.wl_hash[:8]}...)")
                
            except ValueError as e:
                print(f"  Error creating graph with {edge_count} edges: {e}")
                continue
    
    print(f"\nSuccessfully generated {len(graph_zoo)} random graphs")
    return graph_zoo


def main():
    """
    Main experiment runner for random graphs.
    Similar structure to main.py but for random graphs.
    """
    
    # 1. DEFINE PARAMETERS
    n_nodes = 31
    edge_counts = [30, 31, 32]  # Sparse, balanced, slightly denser
    n_graphs_per_edge_count = 10  # Number of random graphs per edge count
    r_values = [1.0, 1.1, 1.2, 2.0]  # Same r values as main.py
    repeats = 1000  # Same as main.py for consistency
    
    # 2. GENERATE RANDOM GRAPHS
    print("="*60)
    print("RANDOM GRAPH EXPERIMENT")
    print("="*60)
    print(f"Configuration:")
    print(f"  Nodes per graph: {n_nodes}")
    print(f"  Edge counts: {edge_counts}")
    print(f"  Graphs per edge count: {n_graphs_per_edge_count}")
    print(f"  r values: {r_values}")
    print(f"  Repeats per configuration: {repeats}")
    print("="*60)
    
    graph_zoo = generate_random_graphs(
        n_nodes=n_nodes,
        edge_counts=edge_counts,
        n_graphs_per_edge_count=n_graphs_per_edge_count
    )
    
    if not graph_zoo:
        print("ERROR: No graphs were generated. Exiting.")
        return
    
    # 3. DISPLAY GRAPH INFORMATION
    print("\n" + "="*60)
    print("GENERATED GRAPHS:")
    print("="*60)
    for graph in graph_zoo:
        print(f"Graph: {graph.name:30s} | Nodes: {graph.N:3d} | Edges: {graph.graph.number_of_edges():3d} | Density: {graph.graph.number_of_edges() / (graph.N * (graph.N - 1) / 2):.3f}")
    
    # 4. RUN EXPERIMENT
    print("\n" + "="*60)
    print("RUNNING EXPERIMENTS")
    print("="*60)
    
    lab = ProcessLab()
    df = lab.run_comparative_study(graph_zoo, r_values, n_repeats=repeats, print_time=False)
    
    # 5. SAVE RESULTS
    data_dir = "simulation_data"
    os.makedirs(data_dir, exist_ok=True)
    
    output_path = os.path.join(data_dir, EXPERIMENTS_CSV)
    df.to_csv(output_path, index=False)
    
    print("\n" + "="*60)
    print("RESULTS SAVED")
    print("="*60)
    print(f"Experiment results: {output_path}")
    print(f"Graph database: {GRAPH_DATABASE_CSV}")
    
    # 6. DISPLAY SUMMARY STATISTICS
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    # Group by graph type and r value
    summary = df.groupby(['graph_name', 'r']).agg({
        'fixation': 'mean',
        'steps': 'mean',
        'duration': 'mean'
    }).reset_index()
    
    print("\nFixation probabilities by graph and r value:")
    print(summary[['graph_name', 'r', 'fixation']].to_string(index=False))
    
    # Overall statistics
    print(f"\nOverall fixation rate: {df['fixation'].mean():.4f}")
    print(f"Average steps to absorption: {df['steps'].mean():.2f}")
    print(f"Total simulation time: {df['duration'].sum():.2f}s")
    
    # Database statistics
    print("\n" + "="*60)
    print("GRAPH DATABASE STATISTICS")
    print("="*60)
    stats = PopulationGraph.get_graph_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    start_time = time.perf_counter()
    
    main()
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    print("\n" + "="*60)
    print(f"TOTAL EXECUTION TIME: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print("="*60)

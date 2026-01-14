# run_random_graphs_test.py
# Quick test version with fewer repeats

from population_graph import PopulationGraph
from process_lab import ProcessLab
import pandas as pd
import os
import time

EXPERIMENTS_CSV = 'random_graphs_experiments_test.csv'
RANDOM_GRAPH_DRAWS = 'random_graph_draws'
DATA_DIR  = 'simulation_data'

def main():
    """Test version with fewer graphs and repeats"""
    
    # 1. DEFINE PARAMETERS (reduced for testing)
    n_nodes = 31
    edge_counts = [30, 31, 32]
    n_graphs_per_edge_count = 2  # Only 2 graphs per edge count for testing
    r_values = [1.0, 1.2]  # Only 2 r values for testing
    repeats = 1000  # Only 10 repeats for testing
    
    print("="*60)
    print("RANDOM GRAPH EXPERIMENT (TEST VERSION)")
    print("="*60)
    print(f"Configuration:")
    print(f"  Nodes per graph: {n_nodes}")
    print(f"  Edge counts: {edge_counts}")
    print(f"  Graphs per edge count: {n_graphs_per_edge_count}")
    print(f"  r values: {r_values}")
    print(f"  Repeats per configuration: {repeats}")
    total_sims = len(edge_counts) * n_graphs_per_edge_count * len(r_values) * repeats
    print(f"  Total simulations: {total_sims}")
    print("="*60)
    
    # 2. GENERATE RANDOM GRAPHS
    graph_zoo = []
    counter = 0
    os.makedirs(RANDOM_GRAPH_DRAWS, exist_ok=True)
    for edge_count in edge_counts:
        print(f"\nGenerating {n_graphs_per_edge_count} graphs with {n_nodes} nodes and {edge_count} edges...")
        
        for i in range(n_graphs_per_edge_count):
            seed = 1000 + counter
            counter += 1
            
            graph = PopulationGraph.random_connected_graph(
                name = f'random_{counter}',
                n_nodes=n_nodes,
                n_edges=edge_count,
                seed=seed,
                register_in_db=True
            )
            graph_zoo.append(graph)
            print(f"  Created: {graph.name}")
            graph.draw(filename=os.path.join(DATA_DIR,RANDOM_GRAPH_DRAWS, f'{graph.name}.png'))
            

    
    print(f"\nGenerated {len(graph_zoo)} graphs")
    
    # 3. RUN EXPERIMENT AND SAVE RESULTS
    print("\n" + "="*60)
    print("RUNNING EXPERIMENTS")
    print("="*60)
    
    lab = ProcessLab()
    output_path = os.path.join(data_dir, EXPERIMENTS_CSV)
    
    df = lab.run_comparative_study(
        graph_zoo, 
        r_values, 
        n_repeats=repeats, 
        print_time=True,
        output_path=output_path
    )
    
    # 4. SUMMARY
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total simulations: {len(df)}")
    print(f"Overall fixation rate: {df['fixation'].mean():.4f}")
    print(f"Average steps: {df['steps'].mean():.2f}")


if __name__ == "__main__":
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    print(f"\nTotal time: {end_time - start_time:.2f} seconds")

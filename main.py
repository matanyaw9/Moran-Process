# main.py

"""
In this file, we create the graph zoo by first creating the graphs we're interested in and then creating lots of 
random graphs. We give the newley created batch a name.
"""

import argparse
from population_graph import PopulationGraph
from process_lab import ProcessLab
from datetime import datetime
import os
import time
import joblib




BATCH_NAME = 'toy_example'

EXPERIMENTS_CSV = 'respiratory_runs.csv'

    # 1. DEFINE THE GRAPH ZOO
    # We instantiate them here so we can inspect them before running
graph_zoo = [
    # PopulationGraph.complete_graph(n_nodes=31),
    # PopulationGraph.cycle_graph(n_nodes=31),
    PopulationGraph.mammalian_lung_graph(branching_factor=2, depth=4),
    PopulationGraph.avian_graph(n_rods=7, rod_length=4),
    PopulationGraph.avian_graph(n_rods=4, rod_length=7),
    PopulationGraph.fish_graph(n_rods=3, rod_length=3)
]


def print_configuration(n_nodes, min_edges, max_edges, n_graphs_per_combination, 
                         r_values, n_repeats, n_random_configs, n_graphs_total):
    """
    Print experiment configuration details.
    """
    print("="*60)
    print("RANDOM GRAPH EXPERIMENT")
    print("="*60)
    print(f"Configuration:")
    print(f"  Nodes per graph: {n_nodes}")
    print(f"  Edge counts: {min_edges} to {max_edges}")
    print(f"  Graphs per edge count: {n_graphs_per_combination}")
    print(f"  r values: {r_values}")
    print(f"  Repeats per configuration: {n_repeats}")
    print(f"  In Total: {n_graphs_total} graphs")
    print(f"  In Total: {n_graphs_total * n_repeats} simulations")
    print("="*60)


def generate_random_graphs(n_nodes:int, edge_range:int, n_graphs_per_combination:int, forbidden_wl_hashes: set[str]=set()):
    """
    Generate random connected graphs and add them to the graph zoo.
    
    Args:
        graph_zoo: List of existing PopulationGraph objects
        n_nodes: List of node counts to generate graphs for
        edge_range: Range of edge counts relative to node count
        n_graphs_per_combination: Number of random graphs to generate per (n_nodes, n_edges) combination
    
    Returns:
        Updated graph_zoo list with random graphs added
    """
    new_random_graph_zoo = []
    occupied_wl = forbidden_wl_hashes.copy()

    for nn in n_nodes: 
        min_e = nn - 1
        max_e = nn + edge_range - 1  # if edge_range = 1 -> range will be only [nn-1]
        edge_counts = range(min_e, max_e)
        
        for ne in edge_counts:
            for i in range(n_graphs_per_combination):
                wl_hash = None 
                while wl_hash is None or wl_hash in occupied_wl:
                    new_random_graph = PopulationGraph.random_connected_graph(
                        n_nodes=nn, 
                        n_edges=ne, 
                        name=f'random_n{nn}_e{ne}_{i}'
                    )
                    wl_hash = new_random_graph.wl_hash
                new_random_graph_zoo.append(new_random_graph)
                occupied_wl.add(wl_hash)
    
    print(f"Number of graphs: {len(new_random_graph_zoo)}")
    print("\n" + "="*60)
    print("GENERATED GRAPHS:")
    print("="*60)

    for graph in new_random_graph_zoo:
        print(f"Graph: {graph.name:30s} | Nodes: {graph.n_nodes:3d} | Edges: {graph.graph.number_of_edges():3d} | Density: {graph.graph.number_of_edges() / (graph.n_nodes * (graph.n_nodes - 1) / 2):.3f}")
    
    return new_random_graph_zoo


def main(batch_name=False):
    """
    Main experiment runner for random graphs.
    Similar structure to main.py but for random graphs.
    """
    # 1. Toy Examples
    n_nodes = list(range(29, 34))
    edge_range = 5
    n_graphs_per_combination = 0  # Number of random graphs per n_edge X n_nodes
    
    r_values = [1.1]  
    n_repeats = 10  
    n_jobs = 4
    

    #  # 1. Small test Run
    # n_nodes = list(range(29, 34))
    # n_graphs_per_combination = 5  # Number of random graphs per n_edge X n_nodes
    # r_values = [1.1 ]  
    # n_repeats = 10  
    # n_jobs = 250
    # edge_range = 3
    
    # # Extreme Graphs  
    # graph_zoo = joblib.load('./tmp_winning_graphs/extreme_graph_zoo.joblib')   
    # n_nodes = list(range(29, 34))
    # edge_range = 5
    # n_graphs_per_combination = 0  # Number of random graphs per n_edge X n_nodes
    
    # r_values = [1.1 ]  
    # n_repeats = 10_000  
    # n_jobs = 1_000

    
    # # DEFAULT PARAMS    
    # n_nodes = list(range(29, 34))
    # edge_range = 5
    # n_graphs_per_combination = 500  # Number of random graphs per n_edge X n_nodes
    
    # r_values = [1.1 ]  
    # n_repeats = 10_000  
    # n_jobs = 1_000
    

    output_dir = os.path.join('simulation_data')
    os.makedirs(output_dir, exist_ok=True)
    # 1. Prepare Batch Directory
    batch_name = batch_name or BATCH_NAME or datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = os.path.join(output_dir, f"batch_{batch_name}")
    os.makedirs(batch_dir, exist_ok=True)
    min_edges = min(n_nodes) - 1 
    max_edges = max(n_nodes) + edge_range - 2

    # 2. PRINT CONFIGURATION
    n_random_configs = len(n_nodes) * edge_range * len(r_values) * n_graphs_per_combination
    n_graphs_total = n_random_configs + len(graph_zoo)    
    print_configuration(n_nodes, min_edges, max_edges, n_graphs_per_combination, 
                        r_values, n_repeats, n_random_configs, n_graphs_total)

    # 3. GENERATE RANDOM GRAPHS
    if n_random_configs:
        graph_zoo_hashes = set([graph.wl_hash for graph in graph_zoo])
        random_graphs = generate_random_graphs(n_nodes, edge_range, n_graphs_per_combination, forbidden_wl_hashes=graph_zoo_hashes)
        graph_zoo.extend(random_graphs)
    # 4. RUN EXPERIMENT AND SAVE RESULTS
    print("\n" + "="*60, "RUNNING EXPERIMENTS", "="*60, sep='\n')

    # 5. SERIALIZE THE GRAPHS
    tmp_dir = os.path.join(batch_dir, 'tmp')
    os.makedirs(tmp_dir, exist_ok=True)
    zoo_path = os.path.join(tmp_dir, "graph_zoo.joblib")
    with open(zoo_path, "wb") as f:
        joblib.dump(graph_zoo, f)

    print(f"Serialized {len(graph_zoo)} graphs to {zoo_path}")
    
    lab = ProcessLab()
    
    lab.submit_jobs(
        zoo_path=zoo_path, 
        n_graphs=len(graph_zoo),
        r_values=r_values, 
        batch_name=batch_name,
        batch_dir=batch_dir,
        n_repeats=n_repeats, 
        n_jobs=n_jobs
    )
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-name", required=False, type=str, help="The name of the batch")
    args = parser.parse_args()

    start_time = time.perf_counter()
  
    main(args.batch_name)
    end_time = time.perf_counter()
    print(f"Whole thing took {(end_time-start_time):.4f} seconds")


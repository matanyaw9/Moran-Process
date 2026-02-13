# main.py
import argparse
from population_graph import PopulationGraph
from process_lab import ProcessLab
from datetime import datetime
import os
import time

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



BATCH_NAME = 'large_test_30_02'

def main(batch_name=False):
    """
    Main experiment runner for random graphs.
    Similar structure to main.py but for random graphs.
    """

    #  # 1. Small test Run
    # n_nodes = list(range(29, 34))
    # n_graphs_per_combination = 5  # Number of random graphs per n_edge X n_nodes
    # r_values = [1.1 ]  
    # n_repeats = 10  
    # n_jobs = 250
    # edge_range = 3
    
    
    # 1. DEFINE PARAMETERS
    n_nodes = list(range(29, 34))
    n_graphs_per_combination = 500  # Number of random graphs per n_edge X n_nodes
    r_values = [1.1 ]  
    n_repeats = 10_000  
    n_jobs = 1_000
    edge_range = 5
    

    output_dir = os.path.join('simulation_data')
    os.makedirs(output_dir, exist_ok=True)
    # 1. Prepare Batch Directory
    batch_name = batch_name or BATCH_NAME or datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = os.path.join(output_dir, f"batch_{batch_name}")
    os.makedirs(batch_dir, exist_ok=True)
    min_edges = min(n_nodes) - 1 
    max_edges = max(n_nodes) + edge_range - 2


    # 2. GENERATE RANDOM GRAPHS
    print("="*60)
    print("RANDOM GRAPH EXPERIMENT")
    print("="*60)
    print(f"Configuration:")
    print(f"  Nodes per graph: {n_nodes}")
    print(f"  Edge counts: {min_edges} to {max_edges}")
    print(f"  Graphs per edge count: {n_graphs_per_combination}")
    print(f"  r values: {r_values}")
    print(f"  Repeats per configuration: {n_repeats}")
    n_random_configs = len(n_nodes) * edge_range * len(r_values) * n_graphs_per_combination
    n_graps_total = n_random_configs + len(graph_zoo)    
    print(f"  In Total: {n_graps_total} graphs")
    print(f"  In Total: {n_graps_total * n_repeats} simulations")

    print("="*60)
    existing_graphs = set([graph.wl_hash for graph in graph_zoo])

    for nn in n_nodes: 
        min_e = nn-1
        max_e = nn + edge_range - 1 # if edge_range = 1 -> range will be only [nn-1]
        edge_counts = range(min_e, max_e)
        # edge_counts = [30]
        for ne in edge_counts:
            for i in range(n_graphs_per_combination):
                wl_hash = None 
                while wl_hash is None or wl_hash in existing_graphs:
                    new_random_graph = PopulationGraph.random_connected_graph(n_nodes=nn, 
                                                                              n_edges=ne, 
                                                                              name = f'random_n{nn}_e{ne}_{i}',
                                                                              )
                    wl_hash = new_random_graph.wl_hash
                graph_zoo.append(new_random_graph)
                existing_graphs.add(wl_hash)
    
    print(f"Number of graphs: {len(graph_zoo)}")
    # 3. DISPLAY GRAPH INFORMATION
    print("\n" + "="*60)
    print("GENERATED GRAPHS:")
    print("="*60)

    # PopulationGraph.batch_register(graph_zoo_path=zoo_path, batch_dir=batch_dir)

    for graph in graph_zoo:
        print(f"Graph: {graph.name:30s} | Nodes: {graph.n_nodes:3d} | Edges: {graph.graph.number_of_edges():3d} | Density: {graph.graph.number_of_edges() / (graph.n_nodes * (graph.n_nodes - 1) / 2):.3f}")
    
    # 4. RUN EXPERIMENT AND SAVE RESULTS
    print("\n" + "="*60)
    print("RUNNING EXPERIMENTS")
    print("="*60)
    
    lab = ProcessLab()
    
    lab.submit_jobs(
        graph_zoo, 
        r_values, 
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


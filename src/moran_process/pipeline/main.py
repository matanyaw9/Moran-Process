# main.py

"""
Batch Creation and Runnning
In this file, we create the graph zoo by first creating the graphs we're interested in and then creating lots of 
random graphs. We give the newley created batch a name.
"""

import argparse
from datetime import datetime
import logging
import os
import sys
import time
import joblib
import numpy as np
from pathlib import Path
from moran_process.core.population_graph import PopulationGraph
from moran_process.pipeline.process_lab import ProcessLab

log = logging.getLogger(__name__)

GRAPH_ZOO_SEED = 42  # controls random graph topology generation; fix for reproducible zoos

BATCH_NAME = 'testing_large_batch_17_03_2026-03'

ROOT = Path(os.getcwd()) 

# Directory constants                                                                                                                                                                                                        
SIMULATION_DATA_DIR = ROOT / "simulation_data"
GRAPH_ZOOS_DIR = ROOT / "graph_zoos"
TMP_DIR_NAME = "tmp"


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
    log.info("Random graph experiment configuration:")
    log.info("  Nodes per graph: %s", n_nodes)
    log.info("  Edge counts: %s to %s", min_edges, max_edges)
    log.info("  Graphs per edge count: %s", n_graphs_per_combination)
    log.info("  r values: %s", r_values)
    log.info("  Repeats per configuration: %s", n_repeats)
    log.info("  In total: %s graphs", n_graphs_total)
    log.info("  In total: %s simulations", n_graphs_total * n_repeats)


def generate_random_graphs(n_nodes:int, edge_range:int, n_graphs_per_combination:int,
                           forbidden_wl_hashes: set[str]=set(), rng=None):
    """
    Generate random connected graphs and add them to the graph zoo.

    Args:
        n_nodes: List of node counts to generate graphs for
        edge_range: Range of edge counts relative to node count
        n_graphs_per_combination: Number of random graphs to generate per (n_nodes, n_edges) combination
        forbidden_wl_hashes: Set of wl_hashes already in the zoo (dedup guard)
        rng: numpy Generator to derive graph seeds from; defaults to GRAPH_ZOO_SEED

    Returns:
        Updated graph_zoo list with random graphs added
    """
    if rng is None:
        rng = np.random.default_rng(GRAPH_ZOO_SEED)

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
                    graph_seed = int(rng.integers(0, 2**32))
                    new_random_graph = PopulationGraph.random_connected_graph(
                        n_nodes=nn,
                        n_edges=ne,
                        name=f'random_n{nn}_e{ne}_{i}',
                        seed=graph_seed,
                    )
                    wl_hash = new_random_graph.wl_hash
                new_random_graph_zoo.append(new_random_graph)
                occupied_wl.add(wl_hash)
    
    log.info("Generated %d random graphs.", len(new_random_graph_zoo))
    for graph in new_random_graph_zoo:
        density = graph.graph.number_of_edges() / (graph.n_nodes * (graph.n_nodes - 1) / 2)
        log.debug("Graph: %-30s | Nodes: %3d | Edges: %3d | Density: %.3f",
                  graph.name, graph.n_nodes, graph.graph.number_of_edges(), density)

    return new_random_graph_zoo


def main(batch_name=False, engine="cpp"):
    """
    Main experiment runner for random graphs.
    Similar structure to main.py but for random graphs.
    """
    # # 1. Toy Examples
    # n_nodes = list(range(29, 34))
    # edge_range = 3
    # n_random_graphs_per_combination = 0  # Number of random graphs per n_edge X n_nodes
    
    # r_values = [1.1]  
    # n_repeats = 10  
    # n_jobs = 5
    

    # # Extreme Graphs  
    # graph_zoo = []      # I intentionally overwrite graph_zoo
    # for fname in os.listdir(GRAPH_ZOOS_DIR):
    #     if not (fname.startswith("extreme_") and fname.endswith(".joblib")):
    #         continue
    #     fpath = GRAPH_ZOOS_DIR / fname
    #     zoo = joblib.load(fpath)
    #     graph_zoo.extend(zoo)
    
    # n_nodes = list(range(29, 34))
    # edge_range = 5
    # n_random_graphs_per_combination = 0  # Number of random graphs per n_edge X n_nodes
    # r_values = [1.1 ]  
    # n_repeats = 20_000  
    # n_jobs = 1_000

    
    # DEFAULT PARAMS    
    n_nodes = list(range(29, 34))
    edge_range = 5
    n_random_graphs_per_combination = 500  # Number of random graphs per n_edge X n_nodes
    
    r_values = [1.1 ]  
    n_repeats = 10_000  
    n_jobs = 1000

    # # 100 node graphs   
    # graph_zoo = []      # I intentionally overwrite graph_zoo
    # n_nodes = [100]
    # edge_range = 4
    # n_random_graphs_per_combination = 50  # Number of random graphs per n_edge X n_nodes
    
    # r_values = [1.1 ]  
    # n_repeats = 10_000  
    # n_jobs = 1000
    

    SIMULATION_DATA_DIR.mkdir(exist_ok=True)
    # 1. Prepare Batch Directory
    batch_name = batch_name or BATCH_NAME or datetime.now().strftime("%Y%m%d_%H%M%S")
    BATCH_DIR = SIMULATION_DATA_DIR / batch_name
    BATCH_DIR.mkdir(exist_ok=True)
    min_edges = min(n_nodes) - 1 
    max_edges = max(n_nodes) + edge_range - 2

    # 2. PRINT CONFIGURATION
    n_random_configs = len(n_nodes) * edge_range * len(r_values) * n_random_graphs_per_combination
    n_graphs_total = n_random_configs + len(graph_zoo)
    print_configuration(n_nodes, min_edges, max_edges, n_random_graphs_per_combination,
                        r_values, n_repeats, n_random_configs, n_graphs_total)

    # Snapshot the biological graph specs BEFORE we extend the zoo with random
    # graphs, so batch_info records exactly which named graphs were hand-built.
    biological_graph_specs = [
        {"name": g.name, "category": g.category, "params": g.params}
        for g in graph_zoo
    ]

    # 3. GENERATE RANDOM GRAPHS
    if n_random_configs:
        graph_zoo_hashes = set([graph.wl_hash for graph in graph_zoo])
        rng = np.random.default_rng(GRAPH_ZOO_SEED)
        random_graphs = generate_random_graphs(n_nodes, edge_range, n_random_graphs_per_combination,
                                               forbidden_wl_hashes=graph_zoo_hashes, rng=rng)
        graph_zoo.extend(random_graphs)
    # 4. RUN EXPERIMENT AND SAVE RESULTS
    log.info("Running experiments")

    # 5. SERIALIZE THE GRAPHS
    tmp_dir = BATCH_DIR / TMP_DIR_NAME
    tmp_dir.mkdir(exist_ok=True)
    zoo_path = tmp_dir / "graph_zoo.joblib"
    with open(zoo_path, "wb") as f:
        joblib.dump(graph_zoo, f)

    log.info("Serialized %d graphs to %s", len(graph_zoo), zoo_path)

    # The recipe to rebuild this exact zoo: random-graph knobs + the seed that
    # drives topology generation + the hand-built biological graphs. Recorded
    # verbatim in batch_info.json under the 'zoo' section.
    zoo_config = {
        "graph_zoo_seed": GRAPH_ZOO_SEED,
        "random_graph_config": {
            "n_nodes": n_nodes,
            "edge_range": edge_range,
            "n_graphs_per_combination": n_random_graphs_per_combination,
        },
        "biological_graphs": biological_graph_specs,
    }

    lab = ProcessLab()

    lab.submit_jobs(
        zoo_path=zoo_path,
        n_graphs=len(graph_zoo),
        r_values=r_values,
        batch_name=batch_name,
        batch_dir=BATCH_DIR,
        n_repeats=n_repeats,
        n_requested_jobs=n_jobs,
        engine=engine,
        zoo_config=zoo_config,
    )
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-name", required=False, type=str, help="The name of the batch")
    parser.add_argument("--engine", choices=["cpp", "python"], default="cpp",
                        help="Simulation engine: 'cpp' (fast, default) or 'python' (reference)")
    args = parser.parse_args()

    # Configure logging once, at the entry point. stdout -> terminal / log file.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    start_time = time.perf_counter()

    main(args.batch_name, engine=args.engine)
    end_time = time.perf_counter()
    log.info("Whole thing took %.4f seconds", end_time - start_time)


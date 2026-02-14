
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from process_lab import ProcessLab
from population_graph import PopulationGraph




if __name__ == "__main__":



    graph_zoo = [
    # PopulationGraph.complete_graph(N=10),
    # PopulationGraph.cycle_graph(N=31),
    PopulationGraph.mammalian_lung_graph(branching_factor=2, depth=4), # N = 511
    PopulationGraph.avian_graph(n_rods=4, rod_length=7),
    PopulationGraph.fish_graph(n_rods=3, rod_length=3)
]

    r_values = [1.0, 1.2, 1.3, 1.4]
    repeats = 100
    
    lab = ProcessLab()
    lab.submit_jobs(graph_zoo=graph_zoo, r_values=r_values, n_repeats=repeats, n_jobs=50)
    
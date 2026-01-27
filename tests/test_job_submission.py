
from process_lab import ProcessLab
from population_graph import PopulationGraph




if __name__ == "__main__":
    graph_zoo = [
    PopulationGraph.complete_graph(N=10),
    # PopulationGraph.cycle_graph(N=31),
    # PopulationGraph.mammalian_lung_graph(branching_factor=2, depth=4), # N = 511
    # PopulationGraph.avian_graph(n_rods=4, rod_length=7),
    # PopulationGraph.fish_graph(n_rods=3, rod_length=3)
]

    r_values = [1.0]
    repeats = 2
    
    lab = ProcessLab()
    lab.submit_jobs(graphs_zoo=graph_zoo, r_values=r_values, n_repeats=2, n_jobs=1)
    
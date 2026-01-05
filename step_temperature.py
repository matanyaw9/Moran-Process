from population_graph import PopulationGraph
from process_run import ProcessRun

N = 10
sim = ProcessRun(PopulationGraph.complete_graph(N=N), selection_coefficient=1.1)
sim.initialize_random_mutant(n_mutants=5)
# history = sim.run(track_history=True)['history']
# history
            
import numpy as np
import random
from population_graph import PopulationGraph

class ProcessRun:
    def __init__(self, population_graph: PopulationGraph, selection_coefficient=1.0, max_steps=1_000_000):
        """
        Initialize a process run on a given population graph.
        :param graph: An instance of PopulationGraph.
        selection_coefficient (r): Fitness of mutant (>1 is advantageous)
        """
        self.pop_graph = population_graph
        self.r = selection_coefficient
        self.max_steps = max_steps
        
        # Current state: 0 = Wild Type, 1 = Mutant
        # We use a simple list/array mapped to node indices
        self.n_nodes = self.pop_graph.n_nodes
        self.state = np.zeros(self.n_nodes, dtype=int)

        # Converts NetworkX graph to a list of lists: adj_list[0] = [neighbors of 0]
        self.adj_list = [list(self.pop_graph.graph.neighbors(n)) for n in range(self.n_nodes)]
    
    def initialize_random_mutant(self, seed=None):
        """Places a single mutant at a random node."""
        self.state.fill(0) # Reset to all wild type
        if seed is not None:
            random.seed(seed)
        
        random_node = random.randint(0, self.n_nodes - 1)
        self.state[random_node] = 1
        return random_node

    def step(self):
        """
        Executes one Moran step:
        1. Select individual to reproduce (weighted by fitness).
        2. Select a neighbor to replace (uniformly random).
        """
        # 1. Calculate fitness weights
        # Wild type (0) has fitness 1.0, Mutant (1) has fitness r
        # We can sum total fitness to normalize probabilities
        current_fitness_map = np.where(self.state == 1, self.r, 1.0)
        total_fitness = np.sum(current_fitness_map)
        probs = current_fitness_map / total_fitness
        
        # Select Reproducer
        reproducer_idx = np.random.choice(self.n_nodes, p=probs)
        
        # Select Neighbor to Die
        # Note: If a node has no neighbors (isolated), nothing happens
        neighbors = self.adj_list[reproducer_idx]
        if len(neighbors) > 0:
            victim_idx = random.choice(neighbors)
            # The victim takes the state of the reproducer
            self.state[victim_idx] = self.state[reproducer_idx]

    def run(self):
        """
        Runs the simulation until fixation or extinction.
        Returns: Dictionary with result details.
        """
        steps = 0
        fixation = False
        
        while steps < self.max_steps:
            # Check current counts
            mutant_count = np.sum(self.state)
            
            # EXTINCTION CHECK
            if mutant_count == 0:
                break # Extinction
            
            # FIXATION CHECK
            if mutant_count == self.n_nodes:
                fixation = True
                break # Fixation
            
            # Run one step
            self.step()
            steps += 1
        return {
            "fixation": fixation,
            "steps": steps,
            "mutant_count": int(np.sum(self.state)),
            "selection_coeff": self.r
        }

# --- TEST BLOCK ---
if __name__ == "__main__":
    print("--- Testing ProcessRun Class ---")
    
    # 1. Setup Graph (Complete Graph to test 1/N theory)
    graph = PopulationGraph()
    N = 15
    experiments = 100
    # graph.generate_mammalian_lung_graph()
    graph.generate_complete_graph(N)
    
    # 2. Setup Process (Neutral drift, r=1.0)
    # Theory: Probability of fixation should be 1/N = 1/20 = 0.05 (5%)
    sim = ProcessRun(graph, selection_coefficient=1.1)
    
    print(f"\nRunning {experiments} simulations on Complete Graph (N={sim.n_nodes})...")
    # print(f'\nRunning {experiments} simulations on Mammalian Lung Graph (N={sim.n_nodes})...')
    fixation_count = 0
    
    for i in range(experiments):
        sim.initialize_random_mutant()
        result = sim.run()
        if result["fixation"]:
            fixation_count += 1
            
    print(f"Fixations: {fixation_count}/{experiments}")
    print(f"Theoretical Probability: {1/sim.n_nodes:.2f} ({100/sim.n_nodes:.3f}%)")
    
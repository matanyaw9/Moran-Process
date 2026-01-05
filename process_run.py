import numpy as np
import random
import time
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
        self.n_nodes = self.pop_graph.number_of_nodes()
        self.state = np.zeros(self.n_nodes, dtype=int)

        # Converts NetworkX graph to a list of lists: adj_list[0] = [neighbors of 0]
        self.adj_list = [list(self.pop_graph.graph.neighbors(n)) for n in range(self.n_nodes)]
    
    def initialize_random_mutant(self, n_mutants=1, seed=None):
        """Places a single mutant at a random node."""
        self.state.fill(0) # Reset to all wild type
        if seed is not None:
            random.seed(seed)
        
        if n_mutants > self.n_nodes:
            raise ValueError("Number of mutants exceeds number of nodes in the graph.")        
        random_nodes = random.sample(range(self.n_nodes), k=n_mutants)
        self.state[random_nodes] = 1
        return random_nodes

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

    def run(self, track_history=False):
        """
        Runs the simulation until fixation or extinction.
        Returns: Dictionary with result details.
        """
        start_time = time.perf_counter()  # <--- Start Timer
        steps = 0
        fixation = False
        initial_mutants = np.sum(self.state)
        history = []
        
        while steps < self.max_steps:

            # Check current counts
            mutant_count = np.sum(self.state)
            
            if track_history:
                history.append(mutant_count)

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
        end_time = time.perf_counter()  # <--- Stop Timer
        
        result = {
            "fixation": fixation,
            "steps": steps,
            "initial_mutants": initial_mutants,
            "selection_coeff": self.r,
            'duration': end_time - start_time
        }
        if track_history: result['history'] = np.array(history)
        return result

    


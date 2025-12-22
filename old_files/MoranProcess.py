from evograph import EvoGraph

class MoranProcess:
    """
    This class implements the Moran process on an evolutionary graph. 

    """
    def __init__(self, evo_graph, fitness_resident=1.0, fitness_mutant=1.0):
        """
        Initializes the MoranProcess with an evolutionary graph and fitness values for residents and mutants.
        
        Parameters:
        evo_graph (EvoGraph): An instance of the EvoGraph class representing the population structure.
        fitness_resident (float): The fitness value of resident individuals.
        fitness_mutant (float): The fitness value of mutant individuals.
        """
        self.time_step = 0
        self.evo_graph = evo_graph
        self.fitness_resident = fitness_resident
        self.fitness_mutant = fitness_mutant

    
    def simulate_step(self):
        pass

    def run_simulation(self, num_steps):
        pass


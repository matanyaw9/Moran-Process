import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

class PopulationGraph:
    """This class is a container of a networkx graph. Used for Evolutionary Graph Theory"""
    def __init__(self, n_nodes=0):
        """
        Initialize an empty graph contaner 
        :param n_nodes: Optional Initial size (used if we want ot pre-allocate, though usually handled by generators).
        """
        self.graph = nx.Graph()
        self.n_nodes = n_nodes

    
    # --- GENERATORS ---

    def generate_complete_graph(self, n):
        """
        Creates a fully connected graph (everyone connected to everyone). 
        """
        self.n_nodes = n
        self.graph = nx.complete_graph(n)

    def generate_cycle_graph(self, n):
        """
        Creates a ring graph.
        """
        self.n_nodes = n
        self.graph = nx.cycle_graph(n)
    
    def generate_mammalian_lung_graph(self, branching_factor=2, depth=3):
        self.graph = nx.balanced_tree(branching_factor, depth)
        self.n_nodes = self.graph.number_of_nodes()

    # --- UTULITIES ---
    def to_adjacency_matrix(self):
        """
        Converts the graph into a NumPy matrix.        
        """
        if self.graph is None: 
            raise ValueError("Graph not initialized.")
        return nx.to_numpy_array(self.graph)
    
    def draw(self):
        """Quick visualization helper"""
        if self.graph is None:
            return
        nx.draw(self.graph, with_labels=True)
        plt.show()

if __name__ == "__main__":
    print("--- Testing Population Graph Class")
    pop = PopulationGraph()
    pop.generate_complete_graph(8)
    pop.draw()


        

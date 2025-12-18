import numpy as np

class EvoGraph:
    """
    This class represents an evolutionary graph structure used in evolutionary algorithms.
    The graph consists of nodes and edges, where nodes represent individuals in a population and edges 
    represent relationships or interactions between them.
    The graph itself will be represented using numpy matrices. 

    """

    def __init__(self, num_nodes, type_graph='complete'):
        """
        Initializes the EvoGraph with a specified number of nodes.
        
        Parameters:
        num_nodes (int): The number of nodes in the graph.
        """
        self.num_nodes = num_nodes
        self.adjacency_matrix = np.ones((num_nodes, num_nodes), dtype=int)

# TODO: Implement different types of graphs (e.g., cycle, star, lungs, etc.)

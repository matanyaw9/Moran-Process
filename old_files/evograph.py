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

class Node: 
    """
    This class represents a node in the evolutionary graph. Each node corresponds to an individual in the population.
    It will be eventually represented in a matrix form within the EvoGraph class.
    """
    def __init__(self):
        self._value = None
        self._edges = [] # a list of nodes

    def get_value(self):
        return self._value
    def set_value(self, value):
        self._value = value
    def get_edges(self):
        return self._edges
    def add_edge(self, node):
        self._edges.append(node)
    def remove_edge(self, node):
        self._edges.remove(node)
    def clear_edges(self):
        self._edges = []
    def num_edges(self):
        return len(self._edges)
    

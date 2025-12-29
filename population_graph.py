import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
# import pydot
from networkx.drawing.nx_pydot import graphviz_layout

class PopulationGraph:
    """This class is a container of a networkx graph. Used for Evolutionary Graph Theory"""
    def __init__(self, graph: nx.Graph | None = None):
        """
        Initialize an empty graph contaner 
        :param n_nodes: Optional Initial size (used if we want ot pre-allocate, though usually handled by generators).
        """
        self.graph_type = None
        self.graph = graph if graph is not None else nx.Graph()


    
    # --- FACTORY METHODS
    @classmethod
    def complete_graph(cls, n:int):
        """
        Creates a fully connected graph (everyone connected to everyone). 
        """
        return cls(nx.complete_graph(n))

    @classmethod
    def cycle_graph(cls, n:int):
        """
        Creates a ring graph.
        """
        return cls(nx.cycle_graph(n))
    
    @classmethod
    def mammalian_lung_graph(cls, branching_factor:int=2, depth:int=3):
        """Generates a tree shaped population graph mimicking mammalian lung topology."""
        return cls(nx.balanced_tree(branching_factor, depth))

    @classmethod
    def avian_graph(cls, n_rods: int, rod_length: int, directed: bool = False):
        """
        Generates a graph mimicking Avian Lungs topology. 

        Structure: 
        - n_rods: Number of parallel 'parabronchi' (linear paths). 
        - rods_length: Number of nodes in each rod.
        - Connectivity: All rods connect to an 'Inlet' and 'Outlet'. 
          A 'Circuit node connects Outlet back to Inlet to close the loop. 

        Args:
            n_rods (int): Number of parallel paths. 
            rod_length (int): Number of nodes per path. 
            directed (bool): If True, returns a DiGraph (unidirectional flow).
        """
        G = nx.DiGraph() if directed else nx.Graph()

        inlet = "Inlet"
        outlet = "Outlet"
        circuit_node = "Circuit"

        G.add_nodes_from([inlet, outlet, circuit_node])
        # Flow: Outlet -> Circuit -> Inlet
        G.add_edge(outlet, circuit_node)
        G.add_edge(circuit_node, inlet)

        for i in range(n_rods):
            rod_nodes = [(i, j) for j in range(rod_length)]
            G.add_nodes_from(rod_nodes)
            intra_rod_edges = list(zip(rod_nodes[:-1], rod_nodes[1:]))
            G.add_edges_from(intra_rod_edges)
            G.add_edge(inlet, rod_nodes[0])
            G.add_edge(rod_nodes[-1], outlet)
        
        # Optimization Tip for NumPy:
        # Convert string labels ('Inlet', (0,1)) to integers (0, 1, 2...)
        # This speeds up your ProcessRun matrix lookups significantly.
        G = nx.convert_node_labels_to_integers(G)
        return cls(G)

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
        # pos = graphviz_layout(self.graph, prog="twopi")
        nx.draw(self.graph, with_labels=True)
        plt.show()

    # --- Getters ---
    def get_as_numpy(self):
        """Returns the graph as a numpy adjacency matrix."""
        return nx.to_numpy_array(self.graph)
    
    def get_neighbors(self, node):
        return list(self.graph.neighbors(node))
    
    def get_nodes(self):
        return list(self.graph.nodes)
    
    def number_of_nodes(self):
        return self.graph.number_of_nodes()
    


# --- TEST BLOCK ---
if __name__ == "__main__":
    print("--- Testing Population Graph Class")
    pop = PopulationGraph()
    pop.mammalian_lung_graph(branching_factor=2, depth=4)
    centrality = nx.eigenvector_centrality(pop.graph)
    print("Centrality: ", centrality)
    # print(pop.degree)
    pop.draw()


        

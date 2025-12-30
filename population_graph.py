import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
# import pydot
from networkx.drawing.nx_pydot import graphviz_layout

class PopulationGraph:
    """This class is a container of a networkx graph. Used for Evolutionary Graph Theory"""
    def __init__(self, graph: nx.Graph | None = None, title: str = ''):
        """
        Initialize an empty graph contaner 
        :param n_nodes: Optional Initial size (used if we want ot pre-allocate, though usually handled by generators).
        """
        self.graph = graph if graph is not None else nx.Graph()
        self.title = title

    # --- FACTORY METHODS ---
    @classmethod
    def complete_graph(cls, n:int):
        """
        Creates a fully connected graph (everyone connected to everyone). 
        """
        return cls(nx.complete_graph(n), title='complete')

    @classmethod
    def cycle_graph(cls, n:int):
        """
        Creates a ring graph.
        """
        return cls(nx.cycle_graph(n), title='cycle')
    
    @classmethod
    def mammalian_lung_graph(cls, branching_factor:int=2, depth:int=3):
        """Generates a tree shaped population graph mimicking mammalian lung topology."""
        G = nx.balanced_tree(branching_factor, depth)
        
        pos = {}
        def assign_pos(node, x_min, x_max, cur_depth):
            x = (x_min + x_max) / 2
            y = -cur_depth
            pos[node] = np.array([x, y])

            children = [n for n in G.neighbors(node) if n > node]
            if not children: return

            width = (x_max - x_min) / len(children)
            for i, child in enumerate(children):
                assign_pos(child, x_min + i * width, x_min+ (i+1)* width, cur_depth+1)
            
        assign_pos(0, 0, 100, 0)
        nx.set_node_attributes(G, pos, 'pos')
        return cls(G, 'mammalian')

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
        return cls(G, title='avian')
    
    @classmethod
    def fish_graph(cls, n_rods: int, rod_length: int):
        """Generates a graph mimicking Fish Gills topology. """
        G = nx.Graph()
        main_rod = list(range(n_rods))
        G.add_edges_from(list(zip(main_rod[:-1], main_rod[1:])))
        G.add_nodes_from(main_rod)
        for i in range(n_rods):
            c_rod = [f"r{i}c{j}" for j in range(rod_length)]
            r_rod = [f"r{i}r{j}" for j in range(rod_length)]
            l_rod = [f"r{i}l{j}" for j in range(rod_length)]
            G.add_nodes_from(c_rod)
            G.add_nodes_from(r_rod)
            G.add_nodes_from(l_rod)
            G.add_edges_from(list(zip(c_rod[:-1], c_rod[1:])))
            G.add_edges_from(list(zip(c_rod, r_rod)))
            G.add_edges_from(list(zip(c_rod, l_rod)))
            G.add_edge(c_rod[0], main_rod[i])
        G = nx.convert_node_labels_to_integers(G)
        

        return cls(G, title='fish')


    # --- UTULITIES ---
    def to_adjacency_matrix(self):
        """
        Converts the graph into a NumPy matrix.        
        """
        if self.graph is None: 
            raise ValueError("Graph not initialized.")
        return nx.to_numpy_array(self.graph)
    
    # --- Visualisation ---
    def draw(self, ax=None):
        """Draws the graph using its stored biological layout."""
        if self.graph is None: return

        pos = nx.get_node_attributes(self.graph, 'pos')
        if not pos: 
            pos = nx.spring_layout(self.graph, seed=42)
        
        if ax is None:
            plt.figure(figsize=(10, 8))
            ax = plt.gca()
        
        nx.draw(self.graph, pos=pos, ax=ax, 
                with_labels=False,
                node_size=50,
                node_color='skyblue', 
                edge_color='#555555',
                width=1.5)
        ax.set_title(self.title, fontsize=14)
        # Turn off axis for cleaner look
        ax.axis('off')
        # plt.title(self.title if self.title else "Population Graph")
        plt.show()


        
        # # Also set the window (figure) title to match
        # manager = getattr(plt.gcf().canvas, "manager", None)
        # if manager is not None and hasattr(manager, "set_window_title"):
        #     manager.set_window_title(self.title if self.title else "Population Graph")
        
        # nx.draw_spring(self.graph, with_labels=True)
        # plt.show()

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
    mammalian = PopulationGraph.mammalian_lung_graph(branching_factor=2, depth=8)
    mammalian.draw()
    # avian = PopulationGraph.avian_graph(n_rods=5, rod_length=8)
    # fish = PopulationGraph.fish_graph(n_rods=3, rod_length=5)
    # complete = PopulationGraph.complete_graph(10)
    # cyrcular = PopulationGraph.cycle_graph(10)

    # # Now let's draw all of them: 
    # avian.draw()
    # fish.draw()
    # complete.draw()
    # cyrcular.draw()

        

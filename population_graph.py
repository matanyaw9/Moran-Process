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
        
        inlet, outlet, circuit = "Inlet", "Outlet", "Circuit"
        G.add_nodes_from([inlet, outlet, circuit])
        
        # 1. Define Macro Layout
        pos = {}
        x_start, x_end = 0, rod_length + 1
        y_center = 0
        
        pos[inlet]   = np.array([x_start - 1, y_center])
        pos[outlet]  = np.array([x_end + 1, y_center])
        pos[circuit] = np.array([(x_start + x_end)/2, y_center - (n_rods/2) - 2]) # Loop below

        # Connect the loop
        G.add_edge(outlet, circuit)
        G.add_edge(circuit, inlet)

        # 2. Generate Parallel Rods
        for i in range(n_rods):
            # Center rods around Y=0
            y = (i - (n_rods - 1) / 2) * 1.0
            
            # Connect Inlet
            first_node = f"r{i}_0"
            G.add_edge(inlet, first_node)
            
            for j in range(rod_length):
                node_id = f"r{i}_{j}"
                x = x_start + j + 0.5
                pos[node_id] = np.array([x, y])
                
                # Internal Edges
                if j > 0:
                    prev_node = f"r{i}_{j-1}"
                    G.add_edge(prev_node, node_id)
            
            # Connect Outlet
            last_node = f"r{i}_{rod_length-1}"
            G.add_edge(last_node, outlet)

        # 3. Store pos & Convert labels
        nx.set_node_attributes(G, pos, 'pos')
        G = nx.convert_node_labels_to_integers(G)
        return cls(G, 'Avian Lung (Parabronchi)')
    
    @classmethod
    def fish_graph(cls, n_rods: int, rod_length: int):
        """Generates a 'Comb' structure: Vertical arch, horizontal filaments."""
        G = nx.Graph()
        pos = {}
        
        main_rod_x = 0
        rod_spacing_y = 4.0 # Vertical distance between filaments
        
        main_nodes = [f"main_{i}" for i in range(n_rods)]
        
        for i in range(n_rods):
            # Main Arch Node
            main_id = main_nodes[i]
            y_base = i * rod_spacing_y
            pos[main_id] = np.array([main_rod_x, y_base])
            
            # Generate Filaments (c) and Lamellae (r/l)
            c_nodes = [f"r{i}c{j}" for j in range(rod_length)]
            r_nodes = [f"r{i}r{j}" for j in range(rod_length)]
            l_nodes = [f"r{i}l{j}" for j in range(rod_length)]
            
            # Connect Main -> First Filament Node
            G.add_edge(main_id, c_nodes[0])
            
            for j in range(rod_length):
                x = main_rod_x + (j + 1) * 1.0
                
                # Positions: c is center, r is above, l is below
                pos[c_nodes[j]] = np.array([x, y_base])
                pos[r_nodes[j]] = np.array([x, y_base + 0.5])
                pos[l_nodes[j]] = np.array([x, y_base - 0.5])
                
                # Edges: Ladder structure
                G.add_edge(c_nodes[j], r_nodes[j]) # Center to Upper
                G.add_edge(c_nodes[j], l_nodes[j]) # Center to Lower
                if j > 0:
                    G.add_edge(c_nodes[j-1], c_nodes[j]) # Linear filament

        # Connect Main Arch vertically
        for k in range(n_rods - 1):
            G.add_edge(main_nodes[k], main_nodes[k+1])

        nx.set_node_attributes(G, pos, 'pos')
        G = nx.convert_node_labels_to_integers(G)
        return cls(G, 'Fish Gills')


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
    avian = PopulationGraph.avian_graph(n_rods=5, rod_length=8)
    avian.draw()
    fish = PopulationGraph.fish_graph(n_rods=8, rod_length=16)
    fish.draw()
    complete = PopulationGraph.complete_graph(10)
    complete.draw()
    cyrcular = PopulationGraph.cycle_graph(10)
    cyrcular.draw()


        

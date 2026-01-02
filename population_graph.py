import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
# import pydot

class PopulationGraph:
    """This class is a container of a networkx graph. Used for Evolutionary Graph Theory"""
    def __init__(self, graph: nx.Graph, name: str, graph_type: str, params: dict = None):
        self.graph = graph
        self.name = name  # e.g., "Mammalian_Depth4"
        self.graph_type = graph_type  # e.g., "Tree", "Complete"
        self.params = params or {}  # Store {depth: 4, branching: 2} for reproducibility
        
        # Pre-calculate static metrics (Vital for analysis later)
        self.N = self.graph.number_of_nodes()
        self.is_directed = self.graph.is_directed()
        
    @property
    def metadata(self):
        """Returns a flat dictionary of graph properties for the dataframe."""
        return {
            "graph_name": self.name,
            "graph_type": self.graph_type,
            "N": self.N,
            **self.params  # Unpack specific params like 'depth' or 'n_rods'
        }
    # --- FACTORY METHODS ---
    @classmethod
    def complete_graph(cls, N:int):
        """
        Creates a fully connected graph (everyone connected to everyone). 
        """
        name=f'complete_n{N}'
        return cls(nx.complete_graph(N), name=name, graph_type="Complete")

    @classmethod
    def cycle_graph(cls, N:int):
        """
        Creates a ring graph.
        """
        name=f'cycle_n{N}'
        return cls(nx.cycle_graph(N), name=name, graph_type='Cycle')
    
    @classmethod
    def mammalian_lung_graph(cls, branching_factor:int=2, depth:int=3, name='mammalian'):
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
        name = f"mammalian_b{branching_factor}_d{depth}"
        return cls(G, name=name, graph_type="Mammalian", 
                   params={"branching": branching_factor, "depth": depth})

    @classmethod
    def avian_graph(cls, n_rods: int, rod_length: int, directed: bool = False, name="avian"):
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
        name = f'avian_r{n_rods}_l{rod_length}'
        return cls(G, name, graph_type='Avian', params={"n_rods": n_rods, "rods_length": rod_length} )
    
    @classmethod
    def fish_graph(cls, n_rods: int, rod_length: int, name='fish'):
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
        name = f'fish_r{n_rods}_l{rod_length}'
        return cls(G, name, graph_type='Fish', params={'n_rods': n_rods, 'rod_length': rod_length})


    # --- UTULITIES ---
    def to_adjacency_matrix(self):
        """
        Converts the graph into a NumPy matrix.        
        """
        if self.graph is None: 
            raise ValueError("Graph not initialized.")
        return nx.to_numpy_array(self.graph)
    
# --- VISUALIZATION ---
    def draw(self, ax=None, filename=''):
        """Draws the graph using its stored biological layout."""
        if self.graph is None: return

        # 1. Coordinate Retrieval
        pos = nx.get_node_attributes(self.graph, 'pos')
        if not pos: 
            pos = nx.spring_layout(self.graph, seed=42)
        
        # 2. Canvas Setup
        created_internally = False  # Track if we created the figure
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = plt.gca()
            created_internally = True
        
        # 3. Drawing
        nx.draw(self.graph, pos=pos, ax=ax, 
                with_labels=False,
                node_size=50,
                node_color='skyblue', 
                edge_color='#555555',
                width=1.5)
        
        ax.set_title(self.name, fontsize=14)
        ax.axis('off')

        # 4. Saving Logic (Robust)
        if filename:
            # Retrieve the immediate parent
            root_fig = ax.get_figure()
            
            # CHECK: If it's a SubFigure (which has no savefig), get the REAL parent
            # SubFigures have a .figure attribute pointing to the top-level Figure
            if not hasattr(root_fig, 'savefig') and hasattr(root_fig, 'figure'):
                root_fig = root_fig.figure

            # Save
            if root_fig is not None:
                root_fig.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"Saved graph to {filename}")
            
            # CLEANUP: Only close if WE created the figure. 
            # If the user passed 'ax', they manage the lifecycle.
            if created_internally:
                plt.close(root_fig)
        elif created_internally:
            # Only show if we created it; otherwise let caller control show()
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
    mammalian = PopulationGraph.mammalian_lung_graph(branching_factor=2, depth=8)
    mammalian.draw(filename="./simulation_data/mammal.png")
    avian = PopulationGraph.avian_graph(n_rods=5, rod_length=8)
    avian.draw(filename="./simulation_data/avian.png")
    fish = PopulationGraph.fish_graph(n_rods=8, rod_length=16)
    fish.draw(filename="./simulation_data/fish.png")
    complete = PopulationGraph.complete_graph(10)
    complete.draw(filename="./simulation_data/complete.png")
    cyrcular = PopulationGraph.cycle_graph(10)
    cyrcular.draw(filename="./simulation_data/cycle.png")


        

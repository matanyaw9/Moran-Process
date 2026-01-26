import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path
import warnings
import pickle
# import pydot
warnings.filterwarnings("ignore", message="The hashes produced for graphs")


# COLOR_DICT = {
#     'Random': 'lightgray',          # The Baseline
#     'Mammalian': '#d62728',         # Red
#     'Avian': '#1f77b4',             # Blue
#     'Fish': '#2ca02c',              # Green
#     'Complete': 'black',            # Fully Connected
#     'Other': '#9467bd'               # Purple
# }
COLOR_DICT = {
    'Random': 'lightgray',     
    'Avian': "#2DB806",       
    'Fish': '#1f77b4',        
    'Mammalian': "#833105",   
    'Complete': 'black',       
    'Other': 'yellow'          
}

GRAPH_PROPS = ['n_nodes', 'n_edges', 'density', 'diameter', 'avg_degree', 
     'average_clustering', 'average_shortest_path_length', 
     'degree_assortativity', 'avg_betweenness_centrality', 'max_degree', 'min_degree', 'degree_std', 'transitivity', 'radius', 'avg_degree_centrality', 'max_degree_centrality', 'max_betweenness_centrality', 'avg_closeness_centrality', 'max_closeness_centrality']

class PopulationGraph:
    """This class is a container of a networkx graph. Used for Evolutionary Graph Theory"""
    
    # Class-level database
    _database_path = "simulation_data/graph_database.csv"
    _database = None
    
    def __init__(self, graph: nx.Graph, 
                 name: str, 
                 category: str, 
                 params: dict|None = None,
                 register_in_db=True,):
        self.graph = graph
        self.name = name  # e.g., "Mammalian_Depth4"
        self.category = category  # e.g., "Tree", "Complete"
        self.params = params or {}  # Store {depth: 4, branching: 2} for reproducibility
        
        # Pre-calculate static metrics (Vital for analysis later)
        self.N = self.graph.number_of_nodes()
        self.is_directed = self.graph.is_directed()
        
        # Calculate WL hash and check database
        self.wl_hash = nx.weisfeiler_lehman_graph_hash(self.graph)
        if register_in_db:
            self._register_in_database()
        
    def calculate_graph_properties(self, save_graph6=True):
        """Calculate comprehensive graph properties for database storage."""
        G = self.graph

        # Basic properties
        properties = {
            'wl_hash': self.wl_hash,
            'graph_name': self.name,
            'category': self.category,
            'n_nodes': self.N,
            'n_edges': G.number_of_edges(),
            'is_directed': self.is_directed,
            'density': nx.density(G),
            'is_connected': nx.is_connected(G) if not self.is_directed else nx.is_weakly_connected(G),
        }

        # 1. Graph6 Generation (Added as requested)
        # header=False strips the '>>graph6<<' prefix to save CSV space
        if save_graph6:
            try:
                graph6_str = nx.to_graph6_bytes(G, header=False).decode('ascii').strip()
            except Exception:
                graph6_str = None
            properties['graph6_string'] = graph6_str

        # Add parameters (like rods in avian lungs)
        properties.update(self.params)
        
        # Only calculate expensive metrics for connected graphs
        if properties['is_connected']:
            try:
                # --- FAST METRICS (Always run) ---
                
                # Degree statistics
                degrees = [d for n, d in G.degree()]
                properties['avg_degree'] = np.mean(degrees)
                properties['max_degree'] = max(degrees)
                properties['min_degree'] = min(degrees)
                properties['degree_std'] = np.std(degrees)
                
                # Assortativity (Caught separately to handle division-by-zero warnings)
                try:
                    properties['degree_assortativity'] = nx.degree_assortativity_coefficient(G)
                except Exception:
                    properties['degree_assortativity'] = None

                # Clustering & Transitivity (Usually fast enough)
                properties['average_clustering'] = nx.average_clustering(G)
                properties['transitivity'] = nx.transitivity(G)

                # --- SLOW METRICS (Guarded by N) ---
                
                # CRITICAL FIX: Diameter is O(N^2). It will freeze your computer for N > 1000.
                if self.N <= 500:
                    properties['diameter'] = nx.diameter(G)
                    properties['radius'] = nx.radius(G)
                    properties['average_shortest_path_length'] = nx.average_shortest_path_length(G)
                else:
                    properties['diameter'] = None 
                    properties['radius'] = None
                    properties['average_shortest_path_length'] = None

                # --- CENTRALITIES ---
                
                # Degree Centrality (Fast)
                degree_cent = nx.degree_centrality(G)
                properties['avg_degree_centrality'] = np.mean(list(degree_cent.values()))
                properties['max_degree_centrality'] = max(degree_cent.values())

                # Betweenness Centrality (Slow - O(NM))
                # Use built-in approximation 'k' for large graphs
                if self.N <= 100:
                    between_cent = nx.betweenness_centrality(G)
                else:
                    between_cent = nx.betweenness_centrality(G, k=50) # Sample 50 nodes
                
                properties['avg_betweenness_centrality'] = np.mean(list(between_cent.values()))
                properties['max_betweenness_centrality'] = max(between_cent.values())

                # Closeness Centrality (Slow - O(NM))
                # NetworkX does NOT support 'k' sampling for closeness automatically!
                # We must implement manual sampling.
                if self.N <= 200:
                    close_cent = nx.closeness_centrality(G)
                    properties['avg_closeness_centrality'] = np.mean(list(close_cent.values()))
                    properties['max_closeness_centrality'] = max(close_cent.values())
                else:
                    # Manual sampling: pick 50 random nodes and compute closeness for them
                    sample_nodes = list(G.nodes())
                    np.random.shuffle(sample_nodes)
                    sample_nodes = sample_nodes[:50]
                    
                    close_vals = [nx.closeness_centrality(G, u=n) for n in sample_nodes]
                    properties['avg_closeness_centrality'] = np.mean(close_vals)
                    properties['max_closeness_centrality'] = max(close_vals)

            except (nx.NetworkXError, ZeroDivisionError) as e:
                print(f"Warning: Could not calculate some properties for {self.name}: {e}")
        
        return properties
    
    @classmethod
    def _load_database(cls):
        """Load the graph database from CSV file."""
        if cls._database is None:
            # Ensure directory exists
            Path(cls._database_path).parent.mkdir(parents=True, exist_ok=True)
            
            if os.path.exists(cls._database_path):
                cls._database = pd.read_csv(cls._database_path)
            else:
                cls._database = pd.DataFrame()
        return cls._database
    
    @classmethod
    def _save_database(cls):
        """Save the graph database to CSV file."""
        if cls._database is not None:
            cls._database.to_csv(cls._database_path, index=False)
    
    def _register_in_database(self):
        """Register this graph in the database if not already present."""
        PopulationGraph._load_database()
        
        # Check if graph already exists - if it does, no need to calc properties
        if not PopulationGraph._database.empty and self.wl_hash in PopulationGraph._database['wl_hash'].values:
            print(f"Graph with WL hash {self.wl_hash} already exists in database")
            return
        
        # Calculate properties and add to database
        properties = self.calculate_graph_properties()
        
        # Convert to DataFrame row and append
        new_row = pd.DataFrame([properties])
        if PopulationGraph._database.empty:
            PopulationGraph._database = new_row
        else:
            PopulationGraph._database = pd.concat([PopulationGraph._database, new_row], ignore_index=True)
        
        # Save to file
        PopulationGraph._save_database()
        print(f"Added graph {self.name} (WL hash: {self.wl_hash}) to database")
    
    @classmethod
    def get_database(cls):
        """Get the current graph database."""
        return cls._load_database().copy()
    
    @classmethod
    def find_similar_graphs(cls, wl_hash):
        """Find graphs with the same WL hash (isomorphic graphs)."""
        db = cls._load_database()
        if db.empty:
            return pd.DataFrame()
        return db[db['wl_hash'] == wl_hash]
    
    @classmethod
    def get_graph_stats(cls)->dict: 
        """Get summary statistics of the graph database."""
        db = cls._load_database()
        if db.empty:
            print("Database is empty")
            return {}
        
        stats = {
            'total_graphs': len(db),
            'unique_topologies': db['wl_hash'].nunique(),
            'graph_types': db['category'].value_counts().to_dict(),
            'size_range': f"{db['n_nodes'].min()}-{db['n_nodes'].max()} nodes",
            'avg_density': db['density'].mean() if 'density' in db.columns else None
        }
        return stats
    
    def register_in_database(self):
        """Manually register this graph in the database (useful if created with register_in_db=False)."""
        self._register_in_database()
        
    @property
    def metadata(self):
        """Returns a flat dictionary of graph properties for the dataframe."""
        return {
            "wl_hash": self.wl_hash,
            "graph_name": self.name
        }
    
    # --- FACTORY METHODS ---
    @classmethod
    def complete_graph(cls, N:int, register_in_db: bool = True):
        """
        Creates a fully connected graph (everyone connected to everyone). 
        """
        name=f'complete_n{N}'
        return cls(nx.complete_graph(N), name=name, category="Complete", register_in_db=register_in_db)

    @classmethod
    def cycle_graph(cls, N:int, register_in_db: bool = True):
        """
        Creates a ring graph.
        """
        name=f'cycle_n{N}'
        return cls(nx.cycle_graph(N), name=name, category='Cycle', register_in_db=register_in_db)
    
    @classmethod
    def mammalian_lung_graph(cls, branching_factor:int=2, depth:int=3, name='mammalian', register_in_db: bool = True):
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
        return cls(G, name=name, category="Mammalian", 
                   params={"branching": branching_factor, "depth": depth}, register_in_db=register_in_db)

    @classmethod
    def avian_graph(cls, n_rods: int, rod_length: int, directed: bool = False, name="avian", register_in_db: bool = True):
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
        return cls(G, name, category='Avian', params={"n_rods": n_rods, "rods_length": rod_length}, register_in_db=register_in_db)
    
    @classmethod
    def fish_graph(cls, n_rods: int, rod_length: int, name='fish', register_in_db: bool = True):
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
        return cls(G, name, category='Fish', params={'n_rods': n_rods, 'rod_length': rod_length}, register_in_db=register_in_db)

    @classmethod
    def random_connected_graph(cls, n_nodes: int, 
                                n_edges: int|None = None, 
                                name: str|None = None, 
                                seed: int|None = None, 
                                register_in_db: bool = True):
        """
        Creates a random connected graph with specified nodes and edges.
        
        Args:
            n_nodes (int): Number of nodes in the graph
            n_edges (int, optional): Number of edges. If None, generates random number
                                   between n_nodes-1 (minimum for connectivity) and 
                                   n_nodes*(n_nodes-1)/2 (complete graph)
            seed (int, optional): Random seed for reproducibility
            register_in_db (bool): Whether to register this graph in the database
            
        Returns:
            PopulationGraph: Random connected graph
        """
        if seed is not None:
            np.random.seed(seed)
        
        if n_nodes < 1:
            raise ValueError("Number of nodes must be at least 1")
        
        # Calculate edge bounds
        min_edges = n_nodes - 1  # Minimum for connectivity (spanning tree)
        max_edges = n_nodes * (n_nodes - 1) // 2  # Complete graph
        
        # Generate random number of edges if not specified
        if n_edges is None:
            n_edges = np.random.randint(min_edges, max_edges + 1)
        
        # Validate edge count
        if n_edges < min_edges:
            raise ValueError(f"Need at least {min_edges} edges for connectivity with {n_nodes} nodes")
        if n_edges > max_edges:
            raise ValueError(f"Maximum {max_edges} edges possible with {n_nodes} nodes")
        
        # Start with a random spanning tree to ensure connectivity
        G = nx.random_labeled_tree(n_nodes, seed=seed)
        
        # Add additional random edges if needed
        current_edges = G.number_of_edges()
        edges_to_add = n_edges - current_edges
        
        if edges_to_add > 0:
            # Generate all possible edges using numpy - much more efficient
            i_indices, j_indices = np.triu_indices(n_nodes, k=1)
            all_possible_edges = np.column_stack((i_indices, j_indices))
            
            # Convert existing edges to numpy array for efficient comparison
            existing_edges = np.array(list(G.edges()))
            
            # Find available edges using numpy set operations
            # Create a view for efficient comparison
            all_edges_view = all_possible_edges.view([('', all_possible_edges.dtype)] * 2).ravel()
            existing_edges_view = existing_edges.view([('', existing_edges.dtype)] * 2).ravel()
            
            # Get mask of available edges
            available_mask = ~np.isin(all_edges_view, existing_edges_view)
            available_edges = all_possible_edges[available_mask]
            
            # Randomly select additional edges
            if len(available_edges) >= edges_to_add:
                selected_indices = np.random.choice(
                    len(available_edges), 
                    size=edges_to_add, 
                    replace=False
                )
                for idx in selected_indices:
                    G.add_edge(*available_edges[idx])
        
        if not name: 
            name = f'random_n{n_nodes}_e{n_edges}'
            if seed is not None:
                name += f'_s{seed}'
            
        return cls(G, name, category='Random', 
                   params={'n_nodes': n_nodes, 'n_edges': n_edges, 'seed': seed}, register_in_db=register_in_db)


    # --- UTULITIES ---
    def to_adjacency_matrix(self):
        """
        Converts the graph into a NumPy matrix.        
        """
        if self.graph is None: 
            raise ValueError("Graph not initialized.")
        return nx.to_numpy_array(self.graph)
    
# --- VISUALIZATION ---
    def draw(self, ax=None, filename='', descriptive=True):
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

        # --- ADDED: Descriptive Stats ---
        if descriptive:
            N = self.graph.number_of_nodes()
            E = self.graph.number_of_edges()
            stats_text = f"Nodes (N): {N}\nEdges (E): {E}"
            
            # Place text in bottom-right corner (0.98, 0.02) relative to axes
            ax.text(0.98, 0.02, stats_text, 
                    transform=ax.transAxes, 
                    horizontalalignment='right',
                    verticalalignment='bottom',
                    fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))
        # --------------------------------

        ax.axis('off')

        # 4. Saving Logic (Robust)
        if filename:
            # Retrieve the immediate parent
            root_fig = ax.get_figure()
            if root_fig is None: raise RuntimeError("Could not retrieve figure from axis.")
            
            # CHECK: If it's a SubFigure (which has no savefig), get the REAL parent
            # SubFigures have a .figure attribute pointing to the top-level Figure
            if not hasattr(root_fig, 'savefig') and hasattr(root_fig, 'figure'):
                root_fig = root_fig.figure

            # Save
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

    def get_wl_hash(self):
        return nx.weisfeiler_lehman_graph_hash(self.graph)
    

    # --- HPC SERIALIZATION ---
    def save(self, filepath: str):
        """
        Serializes the entire PopulationGraph object to a file.
        Used to send the graph topology to HPC worker nodes.
        """
        # Ensure the directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        # print(f"Graph serialized to: {filepath}") # Optional logging

    @staticmethod
    def load(filepath: str):
        """
        Static method to load a PopulationGraph object from a file.
        Usage: graph = PopulationGraph.load('graphs/avian_1.pkl')
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Graph file not found: {filepath}")
            
        with open(filepath, 'rb') as f:
            graph_obj = pickle.load(f)
            
        if not isinstance(graph_obj, PopulationGraph):
            raise TypeError(f"Loaded object is not a PopulationGraph. Got: {type(graph_obj)}")
            
        return graph_obj



# --- TEST BLOCK ---
if __name__ == "__main__":
    print("--- Testing Population Graph Class with Database and register_in_db parameter")
    
    # Clear any existing database for fresh test
    if os.path.exists("simulation_data/graph_database.csv"):
        os.remove("simulation_data/graph_database.csv")
    
    print("\n1. Creating graphs with database registration...")
    mammalian = PopulationGraph.mammalian_lung_graph(branching_factor=2, depth=4)
    complete = PopulationGraph.complete_graph(10)
    
    print("\n2. Creating graphs WITHOUT database registration...")
    # These won't be added to database initially
    temp_cycle = PopulationGraph.cycle_graph(8, register_in_db=False)
    temp_random = PopulationGraph.random_connected_graph(12, 20, seed=42, register_in_db=False)
    
    print(f"Temp cycle WL hash: {temp_cycle.wl_hash}")
    print(f"Temp random WL hash: {temp_random.wl_hash}")
    
    print("\n3. Database stats after initial creation:")
    stats = PopulationGraph.get_graph_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n4. Manually registering temp graphs...")
    temp_cycle.register_in_database()
    temp_random.register_in_database()
    
    print("\n5. Final database stats:")
    stats = PopulationGraph.get_graph_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n6. Testing duplicate detection with register_in_db=False...")
    # This should still detect the duplicate even without registering
    duplicate_complete = PopulationGraph.complete_graph(10, register_in_db=False)
    print(f"Duplicate complete graph WL hash: {duplicate_complete.wl_hash}")
    
    print("\n7. Sample database entries:")
    db = PopulationGraph.get_database()
    if not db.empty:
        print(db[['name', 'category', 'wl_hash', 'n_nodes', 'n_edges']].head())
        
    print(f"\nDatabase saved to: {PopulationGraph._database_path}")


        

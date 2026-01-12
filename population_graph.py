import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path
# import pydot

class PopulationGraph:
    """This class is a container of a networkx graph. Used for Evolutionary Graph Theory"""
    
    # Class-level database
    _database_path = "simulation_data/graph_database.csv"
    _database = None
    
    def __init__(self, graph: nx.Graph, name: str, graph_type: str, params: dict|None = None):
        self.graph = graph
        self.name = name  # e.g., "Mammalian_Depth4"
        self.graph_type = graph_type  # e.g., "Tree", "Complete"
        self.params = params or {}  # Store {depth: 4, branching: 2} for reproducibility
        
        # Pre-calculate static metrics (Vital for analysis later)
        self.N = self.graph.number_of_nodes()
        self.is_directed = self.graph.is_directed()
        
        # Calculate WL hash and check database
        self.wl_hash = self._calculate_wl_hash()
        self._register_in_database()
        
    def _calculate_wl_hash(self):
        """Calculate Weisfeiler-Lehman hash for graph isomorphism detection."""
        return nx.weisfeiler_lehman_graph_hash(self.graph)
    
    def _calculate_graph_properties(self):
        """Calculate comprehensive graph properties for database storage."""
        G = self.graph
        
        # Basic properties
        properties = {
            'wl_hash': self.wl_hash,
            'name': self.name,
            'graph_type': self.graph_type,
            'n_nodes': self.N,
            'n_edges': G.number_of_edges(),
            'is_directed': self.is_directed,
            'density': nx.density(G),
            'is_connected': nx.is_connected(G) if not self.is_directed else nx.is_weakly_connected(G),
        }
        
        # Add parameters
        properties.update(self.params)
        
        # Only calculate expensive metrics for connected graphs
        if properties['is_connected']:
            try:
                # Centrality measures (sample a few nodes for large graphs)
                nodes_sample = list(G.nodes())[:min(100, self.N)]  # Limit for performance
                
                # Degree centrality
                degree_cent = nx.degree_centrality(G)
                properties['avg_degree_centrality'] = np.mean(list(degree_cent.values()))
                properties['max_degree_centrality'] = max(degree_cent.values())
                
                # Betweenness centrality (sample for large graphs)
                if self.N <= 100:
                    between_cent = nx.betweenness_centrality(G)
                    properties['avg_betweenness_centrality'] = np.mean(list(between_cent.values()))
                    properties['max_betweenness_centrality'] = max(between_cent.values())
                else:
                    between_cent = nx.betweenness_centrality(G, k=min(50, self.N))
                    properties['avg_betweenness_centrality'] = np.mean(list(between_cent.values()))
                    properties['max_betweenness_centrality'] = max(between_cent.values())
                
                # Closeness centrality
                if self.N <= 100:
                    close_cent = nx.closeness_centrality(G)
                    properties['avg_closeness_centrality'] = np.mean(list(close_cent.values()))
                    properties['max_closeness_centrality'] = max(close_cent.values())
                
                # Structural properties
                properties['diameter'] = nx.diameter(G)
                properties['radius'] = nx.radius(G)
                properties['average_shortest_path_length'] = nx.average_shortest_path_length(G)
                
                # Clustering
                properties['average_clustering'] = nx.average_clustering(G)
                properties['transitivity'] = nx.transitivity(G)
                
                # Degree statistics
                degrees = [d for n, d in G.degree()]
                properties['avg_degree'] = np.mean(degrees)
                properties['max_degree'] = max(degrees)
                properties['min_degree'] = min(degrees)
                properties['degree_std'] = np.std(degrees)
                
                # Assortativity
                properties['degree_assortativity'] = nx.degree_assortativity_coefficient(G)
                
            except (nx.NetworkXError, ZeroDivisionError) as e:
                # Handle cases where metrics can't be calculated
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
        
        # Check if graph already exists
        if not PopulationGraph._database.empty and self.wl_hash in PopulationGraph._database['wl_hash'].values:
            print(f"Graph with WL hash {self.wl_hash} already exists in database")
            return
        
        # Calculate properties and add to database
        properties = self._calculate_graph_properties()
        
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
    def get_graph_stats(cls):
        """Get summary statistics of the graph database."""
        db = cls._load_database()
        if db.empty:
            return "Database is empty"
        
        stats = {
            'total_graphs': len(db),
            'unique_topologies': db['wl_hash'].nunique(),
            'graph_types': db['graph_type'].value_counts().to_dict(),
            'size_range': f"{db['n_nodes'].min()}-{db['n_nodes'].max()} nodes",
            'avg_density': db['density'].mean() if 'density' in db.columns else None
        }
        return stats
        
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

    @classmethod
    def random_connected_graph(cls, n_nodes: int, n_edges: int|None = None, seed: int|None = None):
        """
        Creates a random connected graph with specified nodes and edges.
        
        Args:
            n_nodes (int): Number of nodes in the graph
            n_edges (int, optional): Number of edges. If None, generates random number
                                   between n_nodes-1 (minimum for connectivity) and 
                                   n_nodes*(n_nodes-1)/2 (complete graph)
            seed (int, optional): Random seed for reproducibility
            
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
        
        name = f'random_n{n_nodes}_e{n_edges}'
        if seed is not None:
            name += f'_s{seed}'
            
        return cls(G, name, graph_type='Random', 
                   params={'n_nodes': n_nodes, 'n_edges': n_edges, 'seed': seed})


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

    def get_wl_hash(self):
        return nx.weisfeiler_lehman_graph_hash(self.graph)
    


# --- TEST BLOCK ---
if __name__ == "__main__":
    print("--- Testing Population Graph Class with Database")
    
    # Clear any existing database for fresh test
    if os.path.exists("simulation_data/graph_database.csv"):
        os.remove("simulation_data/graph_database.csv")
    
    print("\n1. Creating biological graphs...")
    mammalian = PopulationGraph.mammalian_lung_graph(branching_factor=2, depth=4)
    mammalian.draw(filename="./simulation_data/mammal.png")
    
    avian = PopulationGraph.avian_graph(n_rods=5, rod_length=8)
    avian.draw(filename="./simulation_data/avian.png")
    
    fish = PopulationGraph.fish_graph(n_rods=4, rod_length=6)
    fish.draw(filename="./simulation_data/fish.png")
    
    complete = PopulationGraph.complete_graph(10)
    complete.draw(filename="./simulation_data/complete.png")
    
    cyrcular = PopulationGraph.cycle_graph(10)
    cyrcular.draw(filename="./simulation_data/cycle.png")
    
    print("\n2. Creating random graphs...")
    # Test random connected graphs
    random_graph1 = PopulationGraph.random_connected_graph(15, 25, seed=42)
    random_graph1.draw(filename="./simulation_data/random1.png")
    
    random_graph2 = PopulationGraph.random_connected_graph(20, seed=123)  # Random edges
    random_graph2.draw(filename="./simulation_data/random2.png")
    
    print(f"Random graph 1: {random_graph1.N} nodes, {random_graph1.graph.number_of_edges()} edges")
    print(f"Random graph 2: {random_graph2.N} nodes, {random_graph2.graph.number_of_edges()} edges")
    
    print("\n3. Testing duplicate detection...")
    # Create another complete graph with same size - should detect duplicate
    complete_duplicate = PopulationGraph.complete_graph(10)
    
    # Create another random graph with same seed - should detect duplicate
    random_duplicate = PopulationGraph.random_connected_graph(15, 25, seed=42)
    
    print("\n4. Database statistics:")
    stats = PopulationGraph.get_graph_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n5. Sample database entries:")
    db = PopulationGraph.get_database()
    if not db.empty:
        print(db[['name', 'graph_type', 'wl_hash', 'n_nodes', 'n_edges', 'density']].head())
        
    print(f"\nDatabase saved to: {PopulationGraph._database_path}")


        

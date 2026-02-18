import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path
import warnings
import pickle
import argparse
import joblib
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


GRAPH_PROPS = ['n_nodes', 'n_edges', 'density', 'diameter', 'avg_degree', 
     'average_clustering', 'average_shortest_path_length', 
     'degree_assortativity', 'avg_betweenness_centrality', 'max_degree', 'min_degree', 'degree_std', 'transitivity', 'radius', 'avg_degree_centrality', 'max_degree_centrality', 'max_betweenness_centrality', 'avg_closeness_centrality', 'max_closeness_centrality']

class PopulationGraph:
    """This class is a container of a networkx graph. Used for Evolutionary Graph Theory"""
    
    # Class-level database
    def __init__(self, graph: nx.Graph, 
                 name: str, 
                 category: str, 
                 params: dict|None = None,
                 labeled_edges = False,
                 ):
        self.graph = graph
        self.name = name  # e.g., "Mammalian_Depth4"
        self.category = category  # e.g., "Tree", "Complete"
        self.params = params or {}  # Store {depth: 4, branching: 2} for reproducibility
        
        # Pre-calculate static metrics (Vital for analysis later)
        self.n_nodes = self.graph.number_of_nodes()
        self.is_directed = self.graph.is_directed()
        self.labeled_edges = labeled_edges
        
        # Calculate WL hash and check database
        self.wl_hash = nx.weisfeiler_lehman_graph_hash(self.graph)

        # If requested, assign a 'label' attribute to every edge
        if self.labeled_edges:
            edge_labels = {}
            for u, v in self.graph.edges():
                # Create a consistent string label "u_v"
                # If undirected, sort nodes to ensure "0_1" is same as "1_0"
                if not self.is_directed:
                    n1, n2 = sorted((u, v))
                    label = f"{n1}_{n2}"
                else:
                    label = f"{u}_{v}"
                edge_labels[(u, v)] = label
            
            # Efficiently batch-update the graph
            nx.set_edge_attributes(self.graph, edge_labels, "label")
        
    def calculate_graph_properties(self):
        """Calculate comprehensive graph properties for database storage."""
        G = self.graph

        # Basic properties
        properties = {
            'wl_hash': self.wl_hash,
            'graph_name': self.name,
            'category': self.category,
            'n_nodes': self.n_nodes,
            'n_edges': G.number_of_edges(),
            'is_directed': self.is_directed,
            'density': nx.density(G),
            'is_connected': nx.is_connected(G) if not self.is_directed else nx.is_weakly_connected(G),
        }

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
                if self.n_nodes <= 500:
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
                if self.n_nodes <= 100:
                    between_cent = nx.betweenness_centrality(G)
                else:
                    between_cent = nx.betweenness_centrality(G, k=50) # Sample 50 nodes
                
                properties['avg_betweenness_centrality'] = np.mean(list(between_cent.values()))
                properties['max_betweenness_centrality'] = max(between_cent.values())

                # Closeness Centrality (Slow - O(NM))
                # NetworkX does NOT support 'k' sampling for closeness automatically!
                # We must implement manual sampling.
                if self.n_nodes <= 200:
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
    def batch_register(cls, graph_zoo_path, batch_dir):
        """Registers graphs from a pickle file in one go to avoid I/O thrashing.
        
        Args:
            graph_zoo_path: Path to pickle file containing list of PopulationGraph objects
            batch_dir: Directory where graph_props.csv will be saved
        """
        # Load existing batch database
        if type(graph_zoo_path) is str: 
            graph_props_path = os.path.join(batch_dir, 'graph_props.csv')
            new_rows = []
            existing_hashes = set()
            with open(graph_zoo_path, "rb") as f:
                graph_zoo = joblib.load(f)
        
        elif type(graph_zoo_path) is list: 
            graph_zoo = graph_zoo_path
        else: 
            raise ValueError("graph_zoo_path must be a list or a string path")
        
        print(f"Batch processing {len(graph_zoo)} graphs...")
        
        for graph in graph_zoo:
            # Skip if already exists
            if graph.wl_hash in existing_hashes:
                continue
            
            # Calculate metrics only for new unique graphs
            props = graph.calculate_graph_properties()
            new_rows.append(props)
            existing_hashes.add(graph.wl_hash)
            
        if new_rows:
            pd.DataFrame(new_rows).to_csv(graph_props_path, index=False)
            print(f"Entered {len(new_rows)} graph props to {graph_props_path}.")
        else:
            print("No new graphs to add.")
    

    @property
    def metadata(self):
        """Returns a flat dictionary of graph properties for the dataframe."""
        return {
            "wl_hash": self.wl_hash,
            "graph_name": self.name
        }
    
    # --- FACTORY METHODS ---
    @classmethod
    def complete_graph(cls, n_nodes:int, labeled_edges: bool = False):
        """
        Creates a fully connected graph (everyone connected to everyone). 
        """
        name=f'complete_n{n_nodes}'
        return cls(nx.complete_graph(n_nodes), name=name, category="Complete", labeled_edges=labeled_edges)

    @classmethod
    def cycle_graph(cls, n_nodes:int, labeled_edges: bool = False):
        """
        Creates a ring graph.
        """
        name=f'cycle_n{n_nodes}'
        return cls(nx.cycle_graph(n_nodes), name=name, category='Cycle', labeled_edges=labeled_edges)
    
    @classmethod
    def mammalian_lung_graph(cls, branching_factor:int=2, depth:int=3, name='mammalian', labeled_edges: bool = False):
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
               params={"branching": branching_factor, "depth": depth},
               labeled_edges=labeled_edges)

    @classmethod
    def avian_graph(cls, n_rods: int, rod_length: int, directed: bool = False, name="avian", labeled_edges: bool = False):
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
        return cls(G, name, category='Avian', params={"n_rods": n_rods, "rods_length": rod_length}, labeled_edges=labeled_edges)
    
    @classmethod
    def fish_graph(cls, n_rods: int, rod_length: int, name='fish', labeled_edges: bool = False):
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
        return cls(G, name, category='Fish', params={'n_rods': n_rods, 'rod_length': rod_length}, labeled_edges=labeled_edges)


    @classmethod
    def random_connected_graph(cls, n_nodes: int, 
                                n_edges: int | None = None, 
                                name: str | None = None, 
                                seed: int | None = None,
                                labeled_edges: bool = False):
        """
        Creates a random connected graph efficiently using local RNG for stability.
        """
        # 1. Use a local Generator. This fixes the seeding issue without polluting global state.
        rng = np.random.default_rng(seed)
        
        if n_nodes < 1:
            raise ValueError("Number of nodes must be at least 1")

        # 2. Calculate bounds
        min_edges = n_nodes - 1
        max_edges = n_nodes * (n_nodes - 1) // 2
        
        if n_edges is None:
            n_edges = rng.integers(min_edges, max_edges + 1)
        
        if not (min_edges <= n_edges <= max_edges):
            raise ValueError(f"Edges must be between {min_edges} and {max_edges} for {n_nodes} nodes.")

        # 3. Generate the backbone (Spanning Tree)
        # We derive a seed for NetworkX from our local rng to maintain full reproducibility
        tree_seed = int(rng.integers(0, 2**32))
        G = nx.random_labeled_tree(n_nodes, seed=tree_seed)
        
        # 4. Efficiently add the remaining edges (Rejection Sampling)
        # Your previous method generated N^2 edges (huge memory). 
        # This method generates batches of random pairs, which is much faster for biological/sparse graphs.
        edges_needed = n_edges - (n_nodes - 1)
        
        if edges_needed > 0:
            # For very dense graphs (near complete), use the complement approach
            if n_edges > 0.9 * max_edges:
                # Add all edges then remove random ones (faster for near-complete)
                G_complete = nx.complete_graph(n_nodes)
                edges_to_remove = max_edges - n_edges
                edges_list = list(G_complete.edges())
                # Use rng.choice without replacement
                remove_indices = rng.choice(len(edges_list), size=edges_to_remove, replace=False)
                # Rebuild G (it's faster to start full and prune than add 90% of edges one by one)
                G = G_complete
                G.remove_edges_from([edges_list[i] for i in remove_indices])
            else:
                # For sparse/medium graphs (Respiratory logic) -> Add random edges
                while edges_needed > 0:
                    # Generate a batch of potential edges (u, v)
                    # We generate 2x what we need to account for collisions/existing edges
                    batch_size = max(edges_needed * 2, 100)
                    u_list = rng.integers(0, n_nodes, size=batch_size)
                    v_list = rng.integers(0, n_nodes, size=batch_size)
                    
                    for u, v in zip(u_list, v_list):
                        if u != v and not G.has_edge(u, v):
                            G.add_edge(u, v)
                            edges_needed -= 1
                            if edges_needed == 0:
                                break

        if not name:
            name = f'random_n{n_nodes}_e{n_edges}'
            if seed is not None:
                name += f'_s{seed}'

        return cls(G, name, category='Random', 
               params={'n_nodes': n_nodes, 'n_edges': n_edges, 'seed': seed},
               labeled_edges=labeled_edges)

    def mutate_graph(self, name=None, seed=None):
        # 1. Initialize the local RNG
        rng = np.random.default_rng(seed)
        
        G = self.graph.copy()
        edges = list(G.edges())
        
        if not edges:
            return self

        # 2. Remove a random edge using RNG
        idx_to_remove = rng.integers(len(edges))
        u_rem, v_rem = edges[idx_to_remove]
        edge_label = G.get_edge_data(u_rem, v_rem).get('label', f"{u_rem}_{v_rem}") if self.labeled_edges else None
        G.remove_edge(u_rem, v_rem)

        # 3. Get connected components
        if self.is_directed:
            comps = list(nx.weakly_connected_components(G))
        else:
            comps = list(nx.connected_components(G))

        # 4. Branching Logic
        if len(comps) > 1:
            # CASE A: Graph was split. Bridge the two components.
            # IMPORTANT: We must SORT the list because set->list conversion 
            # is non-deterministic in Python (hash randomization). 
            # Without sorting, the seed won't work across different program runs.
            comp_a = sorted(list(comps[0]))
            comp_b = sorted(list(comps[1]))
            
            # Pick one random node from each distinct group using RNG
            u = rng.choice(comp_a)
            v = rng.choice(comp_b)
            G.add_edge(u, v)
            
        else:
            # CASE B: Graph is still connected (edge was part of a cycle).
            # Add a random edge elsewhere.
            nodes = sorted(list(G.nodes())) # Sort for deterministic indexing
            
            # Create an efficient lookup for existing edges
            # (check once, update later)
            existing_edges = set(G.edges())

            for _ in range(100): 
                # rng.choice needs 1D array or int, passing list works but is slower.
                # Better to pick indices if nodes are standard integers, 
                # but if nodes are strings, choice(nodes) is fine.
                u, v = rng.choice(nodes, size=2, replace=False)
                
                # Check undirected existence (u,v) or (v,u)
                if not G.has_edge(u, v):
                    if self.labeled_edges and edge_label is not None:
                        G.add_edge(u, v, label=edge_label)
                        break
                    G.add_edge(u, v)
                    break
            else:
                 # If we fail to find a valid swap (e.g. complete graph), restore original
                 G.add_edge(u_rem, v_rem)
                 print(f"Warning: Failed to mutate {self.name} after 100 attempts. Returning original graph.")
                 # Optional: Warn user or just return un-mutated graph

        if not name:
            name = f"{self.name}_mutated"
            
        return PopulationGraph(G, name, self.category, params=self.params.copy(), labeled_edges=self.labeled_edges)

    # --- UTULITIES ---
    def to_adjacency_matrix(self):
        """
        Converts the graph into a NumPy matrix.        
        """
        if self.graph is None: 
            raise ValueError("Graph not initialized.")
        return nx.to_numpy_array(self.graph)
    
    # --- VISUALIZATION ---
    def draw(self, ax=None, filename='', descriptive=True, with_labels=False):
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
        
        with_edge_labels = self.labeled_edges
        # 3. Drawing
        nx.draw(self.graph, pos=pos, ax=ax, 
            node_size=50,
            node_color='skyblue', 
            with_labels=with_labels,
            edge_color='#555555',
            width=1.5)
        
        if with_edge_labels:
            edge_labels = nx.get_edge_attributes(self.graph, 'label')
            nx.draw_networkx_edge_labels(self.graph, pos, edge_labels, ax=ax, font_size=8)
        
        ax.set_title(self.name, fontsize=14)

        # --- ADDED: Descriptive Stats ---
        if descriptive:
            n_nodes = self.graph.number_of_nodes()
            n_edges = self.graph.number_of_edges()
            stats_text = f"Nodes (N): {n_nodes}\nEdges (E): {n_edges}"
            
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



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register graphs in the database.")
    parser.add_argument("--batch-dir", required=True, help="Directory for graph_props.csv")
    parser.add_argument("--graph-zoo-path", required=True, help="Path to pickle file with graphs")
    parser.add_argument("--register", action="store_true", help="Enable registration")

    args = parser.parse_args()

    if args.register:
        print("Registering graphs in database...")
        PopulationGraph.batch_register(args.graph_zoo_path, args.batch_dir)

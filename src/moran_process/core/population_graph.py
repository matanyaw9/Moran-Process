import networkx as nx
import numpy as np
import pandas as pd
import os
from pathlib import Path
import warnings
import pickle
import argparse
import joblib

# matplotlib is imported lazily inside draw() (the only consumer) so importing
# this module does not pull the plotting stack. GraphCore now lives in the
# dependency-light core.graph_core module; re-exported here so existing imports
# and older pickled shards (`population_graph.GraphCore`) keep resolving.
from moran_process.core.graph_core import GraphCore

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


GRAPH_PROPS = [
    "n_nodes",
    "n_edges",
    "density",
    "diameter",
    "avg_degree",
    "average_clustering",
    "average_shortest_path_length",
    "degree_assortativity",
    "avg_betweenness_centrality",
    "max_degree",
    "min_degree",
    "degree_std",
    "transitivity",
    "radius",
    "avg_degree_centrality",
    "max_degree_centrality",
    "max_betweenness_centrality",
    "avg_closeness_centrality",
    "max_closeness_centrality",
]


class PopulationGraph:
    """This class is a container of a networkx graph. Used for Evolutionary Graph Theory"""

    # Class-level database
    def __init__(
        self,
        graph: nx.Graph,
        name: str,
        category: str,
        params: dict | None = None,
        labeled_edges=False,
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
            "wl_hash": self.wl_hash,
            "graph_name": self.name,
            "category": self.category,
            "n_nodes": self.n_nodes,
            "n_edges": G.number_of_edges(),
            "is_directed": self.is_directed,
            "density": nx.density(G),
            "is_connected": (
                nx.is_connected(G)
                if not self.is_directed
                else nx.is_weakly_connected(G)
            ),
        }

        properties.update(self.params)

        # Only calculate expensive metrics for connected graphs
        if properties["is_connected"]:
            try:
                # --- FAST METRICS (Always run) ---

                # Degree statistics
                degrees = [d for n, d in G.degree()]
                properties["avg_degree"] = np.mean(degrees)
                properties["max_degree"] = max(degrees)
                properties["min_degree"] = min(degrees)
                properties["degree_std"] = np.std(degrees)

                # Assortativity (Caught separately to handle division-by-zero warnings)
                try:
                    properties["degree_assortativity"] = (
                        nx.degree_assortativity_coefficient(G)
                    )
                except Exception:
                    properties["degree_assortativity"] = None

                # Clustering & Transitivity (Usually fast enough)
                properties["average_clustering"] = nx.average_clustering(G)
                properties["transitivity"] = nx.transitivity(G)

                # --- SLOW METRICS (Guarded by N) ---

                # Distances need every node to reach every other. A connected undirected
                # graph gives that for free; a digraph needs strong connectivity, which
                # the directed star (one source, N-1 sinks) does not have. Without this
                # guard nx.diameter raises and aborts the whole try block, taking the six
                # centrality columns down with it even though they compute fine.
                distances_defined = not self.is_directed or nx.is_strongly_connected(G)

                # CRITICAL FIX: Diameter is O(N^2). It will freeze your computer for N > 1000.
                if self.n_nodes <= 500 and distances_defined:
                    properties["diameter"] = nx.diameter(G)
                    properties["radius"] = nx.radius(G)
                    properties["average_shortest_path_length"] = (
                        nx.average_shortest_path_length(G)
                    )
                else:
                    properties["diameter"] = None
                    properties["radius"] = None
                    properties["average_shortest_path_length"] = None

                # --- CENTRALITIES ---

                # Degree Centrality (Fast)
                degree_cent = nx.degree_centrality(G)
                properties["avg_degree_centrality"] = np.mean(
                    list(degree_cent.values())
                )
                properties["max_degree_centrality"] = max(degree_cent.values())

                # Betweenness Centrality (Slow - O(NM))
                # Use built-in approximation 'k' for large graphs
                if self.n_nodes <= 100:
                    between_cent = nx.betweenness_centrality(G)
                else:
                    between_cent = nx.betweenness_centrality(G, k=50)  # Sample 50 nodes

                properties["avg_betweenness_centrality"] = np.mean(
                    list(between_cent.values())
                )
                properties["max_betweenness_centrality"] = max(between_cent.values())

                # Closeness Centrality (Slow - O(NM))
                # NetworkX does NOT support 'k' sampling for closeness automatically!
                # We must implement manual sampling.
                if self.n_nodes <= 200:
                    close_cent = nx.closeness_centrality(G)
                    properties["avg_closeness_centrality"] = np.mean(
                        list(close_cent.values())
                    )
                    properties["max_closeness_centrality"] = max(close_cent.values())
                else:
                    # Manual sampling: pick 50 random nodes and compute closeness for them
                    sample_nodes = list(G.nodes())
                    np.random.shuffle(sample_nodes)
                    sample_nodes = sample_nodes[:50]

                    close_vals = [nx.closeness_centrality(G, u=n) for n in sample_nodes]
                    properties["avg_closeness_centrality"] = np.mean(close_vals)
                    properties["max_closeness_centrality"] = max(close_vals)

            except (nx.NetworkXError, ZeroDivisionError) as e:
                print(
                    f"Warning: Could not calculate some properties for {self.name}: {e}"
                )

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
            graph_props_path = os.path.join(batch_dir, "graph_props.csv")
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
        return {"wl_hash": self.wl_hash, "graph_name": self.name}

    # --- FACTORY METHODS ---
    @classmethod
    def complete_graph(cls, n_nodes: int, labeled_edges: bool = False):
        """
        Creates a fully connected graph (everyone connected to everyone).
        """
        name = f"complete_n{n_nodes}"
        return cls(
            nx.complete_graph(n_nodes),
            name=name,
            category="Complete",
            labeled_edges=labeled_edges,
        )

    @classmethod
    def cycle_graph(cls, n_nodes: int, directed: bool = False, labeled_edges: bool = False):
        """
        Creates a ring graph.

        With directed=True every edge points one way around the ring
        (0 -> 1 -> ... -> n-1 -> 0). That leaves every node with in-degree 1 and
        out-degree 1, which is the isothermal condition, so it neither amplifies nor
        suppresses: rho equals the well-mixed Moran value (1 - 1/r)/(1 - 1/r**N). It is
        the natural control against the directed graphs that do deviate.

        Args:
            n_nodes (int): Number of nodes in the ring.
            directed (bool): If True, returns a DiGraph circulating one way.
        """
        G = nx.cycle_graph(n_nodes, create_using=nx.DiGraph if directed else None)

        # Store the ring layout, as star_graph and line_graph already do for
        # theirs. Without a 'pos' attribute draw() falls back to spring_layout,
        # which lays a cycle out as tangled spaghetti and hides the one property
        # that matters about it: that every node is identical.
        angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
        pos = {i: np.array([np.cos(a), np.sin(a)]) for i, a in enumerate(angles)}
        nx.set_node_attributes(G, pos, "pos")

        name = f"directed_cycle_n{n_nodes}" if directed else f"cycle_n{n_nodes}"
        return cls(
            G,
            name=name,
            category="Cycle",
            params={"n_nodes": n_nodes},
            labeled_edges=labeled_edges,
        )

    @classmethod
    def line_graph(cls, n_nodes: int, directed: bool = False, labeled_edges: bool = False):
        """
        Creates a path (line) graph: nodes arranged in a linear chain.
        Known theoretical suppressor of selection in evolutionary graph theory.

        With directed=True every edge points downstream (0 -> 1 -> ... -> n-1). Node 0
        becomes a source that nothing can overwrite, so a mutant fixates iff it is born
        there: rho = 1/N at every r, selection fully abolished. Same verdict as the
        directed star, but it fixates far faster (a front sweeping a chain, rather than
        a hub coupon-collecting its leaves).

        Note the directed variant is weakly but NOT strongly connected, so its distance
        metrics are undefined and recorded as None.

        Args:
            n_nodes (int): Number of nodes in the chain.
            directed (bool): If True, returns a DiGraph flowing one way.
        """
        G = nx.path_graph(n_nodes, create_using=nx.DiGraph if directed else None)

        pos = {i: np.array([float(i), 0.0]) for i in range(n_nodes)}
        nx.set_node_attributes(G, pos, "pos")

        name = f"directed_line_n{n_nodes}" if directed else f"line_n{n_nodes}"
        return cls(
            G,
            name=name,
            category="Line",
            params={"n_nodes": n_nodes},
            labeled_edges=labeled_edges,
        )

    @classmethod
    def star_graph(
        cls, n_nodes: int, directed: bool = False, labeled_edges: bool = False
    ):
        """
        Creates a star graph: one central hub connected to all other nodes.
        Node 0 is the hub; nodes 1..n_nodes-1 are leaves.
        Classic selective amplifier in evolutionary graph theory (Lieberman et al. 2005).

        With directed=True every edge points hub -> leaf. That makes the hub a source
        (in-degree 0, so nothing can overwrite it) and every leaf a sink (out-degree 0,
        so it can never propagate), which abolishes selection entirely: a mutant fixates
        iff it is born on the hub, so rho = 1/N at every r. It is the exact counterpart
        of the undirected star -- same nodes, same edges, amplifier becomes suppressor.

        Note the directed variant is weakly but NOT strongly connected, so distance
        metrics (diameter, radius, average shortest path) are undefined for it.

        Args:
            n_nodes (int): Total nodes (1 hub + n_nodes-1 leaves).
            directed (bool): If True, returns a DiGraph with edges hub -> leaf.
        """
        # nx.star_graph(k) produces k+1 nodes (hub + k leaves), so pass n_nodes-1.
        # create_using=DiGraph emits exactly the (0, i) edges, i.e. hub -> leaf.
        G = nx.star_graph(n_nodes - 1, create_using=nx.DiGraph if directed else None)

        # Layout: hub at center, leaves on a unit circle
        angles = np.linspace(0, 2 * np.pi, n_nodes - 1, endpoint=False)
        pos = {0: np.array([0.0, 0.0])}
        for i, angle in enumerate(angles):
            pos[i + 1] = np.array([np.cos(angle), np.sin(angle)])
        nx.set_node_attributes(G, pos, "pos")

        name = f"directed_star_n{n_nodes}" if directed else f"star_n{n_nodes}"
        return cls(
            G,
            name=name,
            category="Star",
            params={"n_nodes": n_nodes},
            labeled_edges=labeled_edges,
        )

    @classmethod
    def grid_graph(cls, width: int, height: int, labeled_edges: bool = False):
        """
        Creates a rectangular grid (2D lattice) of size width x height.

        Every node is connected to its left/right and top/bottom neighbors only
        (4-connectivity), so when drawn it looks like a filled rectangle.

        Args:
            width (int): Number of columns.
            height (int): Number of rows.
        """
        if width < 1 or height < 1:
            raise ValueError("width and height must both be at least 1")

        # nx.grid_2d_graph(rows, cols) yields the exact left/right/top/bottom
        # lattice; nodes are (row, col) tuples.
        G = nx.grid_2d_graph(height, width)

        # Layout: column -> x, row -> -y so row 0 sits at the top, matching the
        # screen-oriented convention used by mammalian_lung_graph.
        pos = {(r, c): np.array([float(c), -float(r)]) for r, c in G.nodes()}
        nx.set_node_attributes(G, pos, "pos")

        # Compact integer labels (and keep pos) for hashing/simulation layers.
        G = nx.convert_node_labels_to_integers(G)
        name = f"grid_w{width}_h{height}"
        return cls(
            G,
            name=name,
            category="Grid",
            params={"width": width, "height": height},
            labeled_edges=labeled_edges,
        )

    @classmethod
    def mammalian_lung_graph(
        cls,
        branching_factor: int = 2,
        depth: int = 3,
        directed: bool = False,
        name="mammalian",
        labeled_edges: bool = False,
    ):
        """Generates a tree shaped population graph mimicking mammalian lung topology.

        With directed=True every edge points parent -> child, i.e. top to bottom, the
        direction air travels on inhalation. The root is then the only source (in-degree
        0, so nothing can overwrite it) and every leaf a sink, which abolishes selection:
        a mutant fixates iff it is born at the root, so rho = 1/N at every r. It is the
        fastest of the single-source suppressors, because a tree sweeps 2**k fronts in
        parallel and so fixates in time set by depth rather than by node count.

        Note the directed variant is weakly but NOT strongly connected, so its distance
        metrics are undefined and recorded as None.

        Args:
            branching_factor (int): Children per internal node.
            depth (int): Levels below the root.
            directed (bool): If True, returns a DiGraph flowing root -> leaves.
        """
        G = nx.balanced_tree(
            branching_factor, depth, create_using=nx.DiGraph if directed else None
        )

        pos = {}

        def assign_pos(node, x_min, x_max, cur_depth):
            x = (x_min + x_max) / 2
            y = -cur_depth
            pos[node] = np.array([x, y])

            children = [n for n in G.neighbors(node) if n > node]
            if not children:
                return

            width = (x_max - x_min) / len(children)
            for i, child in enumerate(children):
                assign_pos(
                    child, x_min + i * width, x_min + (i + 1) * width, cur_depth + 1
                )

        assign_pos(0, 0, 100, 0)
        nx.set_node_attributes(G, pos, "pos")
        prefix = "directed_mammalian" if directed else "mammalian"
        name = f"{prefix}_b{branching_factor}_d{depth}"
        return cls(
            G,
            name=name,
            category="Mammalian",
            params={"branching": branching_factor, "depth": depth},
            labeled_edges=labeled_edges,
        )

    @classmethod
    def avian_graph(
        cls,
        n_rods: int,
        rod_length: int,
        directed: bool = False,
        name="avian",
        labeled_edges: bool = False,
    ):
        """
        Generates a graph mimicking Avian Lungs topology.

        Structure:
        - n_rods: Number of parallel 'parabronchi' (linear paths).
        - rods_length: Number of nodes in each rod.
        - Connectivity: every parabronchus runs from the posterior air sac to the
          anterior air sac; the trachea closes the loop from anterior back to
          posterior.

        With directed=True every edge points downstream, so air (and offspring)
        flow one way only: posterior air sac -> parabronchi -> anterior air sac
        -> trachea -> posterior air sac. Node identifiers are anatomical and are
        preserved as a 'label' node attribute after the integer relabelling, so
        draw(with_labels=True) shows them.

        Args:
            n_rods (int): Number of parallel paths.
            rod_length (int): Number of nodes per path.
            directed (bool): If True, returns a DiGraph (unidirectional flow).
        """
        G = nx.DiGraph() if directed else nx.Graph()

        inlet, outlet, circuit = "posterior-air-sac", "anterior-air-sac", "trachea"
        G.add_nodes_from([inlet, outlet, circuit])

        # 1. Define Macro Layout
        pos = {}
        x_start, x_end = 0, rod_length + 1
        y_center = 0

        pos[inlet] = np.array([x_start - 1, y_center])
        pos[outlet] = np.array([x_end + 1, y_center])
        pos[circuit] = np.array(
            [(x_start + x_end) / 2, y_center - (n_rods / 2) - 2]
        )  # Loop below

        # Connect the loop
        G.add_edge(outlet, circuit)
        G.add_edge(circuit, inlet)

        # 2. Generate Parallel Rods
        for i in range(n_rods):
            # Center rods around Y=0
            y = (i - (n_rods - 1) / 2) * 1.0

            # Connect Inlet
            first_node = f"parabronchus_{i}_0"
            G.add_edge(inlet, first_node)

            for j in range(rod_length):
                node_id = f"parabronchus_{i}_{j}"
                x = x_start + j + 0.5
                pos[node_id] = np.array([x, y])

                # Internal Edges
                if j > 0:
                    prev_node = f"parabronchus_{i}_{j-1}"
                    G.add_edge(prev_node, node_id)

            # Connect Outlet
            last_node = f"parabronchus_{i}_{rod_length-1}"
            G.add_edge(last_node, outlet)

        # 3. Store pos & Convert labels
        # The simulation layer indexes nodes as range(n) (see to_simulation_struct),
        # so the anatomical identifiers cannot survive as node keys. label_attribute
        # keeps them as a node attribute instead, which draw() picks up and which the
        # WL hash ignores (weisfeiler_lehman_graph_hash reads node attrs only when
        # node_attr is passed), so relabelling does not invalidate any existing batch.
        nx.set_node_attributes(G, pos, "pos")
        G = nx.convert_node_labels_to_integers(G, label_attribute="label")
        prefix = "directed_avian" if directed else "avian"
        name = f"{prefix}_r{n_rods}_l{rod_length}"
        return cls(
            G,
            name,
            category="Avian",
            params={"n_rods": n_rods, "rods_length": rod_length},
            labeled_edges=labeled_edges,
        )

    @classmethod
    def fish_graph(
        cls,
        n_rods: int,
        rod_length: int,
        fillaments: int = 2,
        name="fish",
        labeled_edges: bool = False,
    ):
        """Generates a 'Comb' structure: Vertical arch, horizontal filaments.

        Each of the n_rods gill filaments hangs off the vertical arch as a chain of
        rod_length center nodes, and every center node carries `fillaments` lamellae
        split between its two sides (the upper side takes the extra one when the
        count is odd). fillaments=2 is the original one-up-one-down comb, so the
        default reproduces graphs built before this parameter existed, wl_hash
        included; only an explicit value creates a new topology.

        N = n_rods * (1 + rod_length * (1 + fillaments))

        Args:
            n_rods (int): Number of filaments along the arch.
            rod_length (int): Center nodes per filament.
            fillaments (int): Lamellae per center node, TOTAL across both sides.
        """
        if fillaments < 0:
            raise ValueError("fillaments must be non-negative")

        G = nx.Graph()
        pos = {}

        main_rod_x = 0
        n_up = (fillaments + 1) // 2  # odd counts put the extra lamella on top
        n_down = fillaments // 2

        # Lamellae fan out from their center node instead of stacking on one
        # vertical line. They all attach to that single node, so drawn colinear
        # their edges would overlap and a star would read as a path.
        def lamella_offsets(count, sign):
            if count == 0:
                return []
            radius = max(0.5, 0.18 * count)
            angles = (
                np.array([np.pi / 2])
                if count == 1
                else np.linspace(np.pi / 2 - 0.55, np.pi / 2 + 0.55, count)
            )
            return [
                np.array([radius * np.cos(a), sign * radius * np.sin(a)])
                for a in angles
            ]

        up_offsets = lamella_offsets(n_up, 1.0)
        down_offsets = lamella_offsets(n_down, -1.0)

        # Vertical distance between filaments, widened so tall fans on adjacent
        # rods do not collide (4.0 is the historical spacing, kept as the floor).
        reach = max([abs(o[1]) for o in up_offsets + down_offsets], default=0.5)
        rod_spacing_y = max(4.0, 2 * reach + 2.0)

        # Same idea along the filament: a wide fan is wider than the historical
        # 1.0 gap between center nodes, and neighbouring fans would interleave.
        # Collapses back to exactly 1.0 for one lamella per side.
        span = max([abs(o[0]) for o in up_offsets + down_offsets], default=0.0)
        c_spacing = max(1.0, 3.0 * span)

        main_nodes = [f"main_{i}" for i in range(n_rods)]

        for i in range(n_rods):
            # Main Arch Node
            main_id = main_nodes[i]
            y_base = i * rod_spacing_y
            pos[main_id] = np.array([main_rod_x, y_base])

            # Filament backbone (c); lamellae hang off each of its nodes
            c_nodes = [f"r{i}c{j}" for j in range(rod_length)]

            # Connect Main -> First Filament Node
            G.add_edge(main_id, c_nodes[0])

            for j in range(rod_length):
                x = main_rod_x + (j + 1) * c_spacing
                center = np.array([x, y_base])
                pos[c_nodes[j]] = center

                for m, offset in enumerate(up_offsets):
                    node_id = f"r{i}u{j}_{m}"
                    pos[node_id] = center + offset
                    G.add_edge(c_nodes[j], node_id)

                for m, offset in enumerate(down_offsets):
                    node_id = f"r{i}d{j}_{m}"
                    pos[node_id] = center + offset
                    G.add_edge(c_nodes[j], node_id)

                if j > 0:
                    G.add_edge(c_nodes[j - 1], c_nodes[j])  # Linear filament

        # Connect Main Arch vertically
        for k in range(n_rods - 1):
            G.add_edge(main_nodes[k], main_nodes[k + 1])

        nx.set_node_attributes(G, pos, "pos")
        G = nx.convert_node_labels_to_integers(G)
        name = f"fish_r{n_rods}_l{rod_length}_f{fillaments}"
        return cls(
            G,
            name,
            category="Fish",
            params={
                "n_rods": n_rods,
                "rod_length": rod_length,
                "fillaments": fillaments,
            },
            labeled_edges=labeled_edges,
        )

    @classmethod
    def random_connected_graph(
        cls,
        n_nodes: int,
        n_edges: int | None = None,
        name: str | None = None,
        seed: int | None = None,
        labeled_edges: bool = False,
    ):
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
            raise ValueError(
                f"Edges must be between {min_edges} and {max_edges} for {n_nodes} nodes."
            )

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
                remove_indices = rng.choice(
                    len(edges_list), size=edges_to_remove, replace=False
                )
                # Rebuild G (it's faster to start full and prune than add 90% of edges one by one)
                G = G_complete
                G.remove_edges_from([edges_list[i] for i in remove_indices])
            else:
                # For sparse/medium graphs (Respiratory logic) -> Add random edges
                while edges_needed > 0:
                    # Generate a batch of potential edges (u, v)
                    # We generate 2x what we need to account for collisions/existing edges
                    batch_size = max(edges_needed * 2, 100)
                    # .tolist() rather than iterating the arrays directly: it yields
                    # Python ints, where iterating yields np.int64. Both hash and compare
                    # equal to an int, so the node dict looks clean and G.nodes reports
                    # plain ints -- but the numpy object is what gets stored in the edge
                    # tuple. NetworkX algorithms that compare a node against a tuple then
                    # broadcast instead of comparing, and raise "truth value of an array
                    # with more than one element is ambiguous". nx.minimum_cycle_basis
                    # does exactly that, so it fails on every graph built here.
                    u_list = rng.integers(0, n_nodes, size=batch_size).tolist()
                    v_list = rng.integers(0, n_nodes, size=batch_size).tolist()

                    for u, v in zip(u_list, v_list):
                        if u != v and not G.has_edge(u, v):
                            G.add_edge(u, v)
                            edges_needed -= 1
                            if edges_needed == 0:
                                break

        if not name:
            name = f"random_n{n_nodes}_e{n_edges}"
            if seed is not None:
                name += f"_s{seed}"

        return cls(
            G,
            name,
            category="Random",
            params={"n_nodes": n_nodes, "n_edges": n_edges, "seed": seed},
            labeled_edges=labeled_edges,
        )

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
        edge_label = (
            G.get_edge_data(u_rem, v_rem).get("label", f"{u_rem}_{v_rem}")
            if self.labeled_edges
            else None
        )
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

            # Pick one random node from each distinct group using RNG.
            # int() because rng.choice returns np.int64, which would land in the edge
            # tuple and break node-vs-tuple comparisons downstream (see
            # random_connected_graph). Node keys are integers in every factory.
            u = int(rng.choice(comp_a))
            v = int(rng.choice(comp_b))
            G.add_edge(u, v)

        else:
            # CASE B: Graph is still connected (edge was part of a cycle).
            # Add a random edge elsewhere.
            nodes = sorted(list(G.nodes()))  # Sort for deterministic indexing

            # Create an efficient lookup for existing edges
            # (check once, update later)
            existing_edges = set(G.edges())

            for _ in range(100):
                # rng.choice needs 1D array or int, passing list works but is slower.
                # Better to pick indices if nodes are standard integers,
                # but if nodes are strings, choice(nodes) is fine.
                # int() for the same reason as above: rng.choice yields np.int64.
                u, v = (int(x) for x in rng.choice(nodes, size=2, replace=False))

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
                print(
                    f"Warning: Failed to mutate {self.name} after 100 attempts. Returning original graph."
                )
                # Optional: Warn user or just return un-mutated graph

        if not name:
            name = f"{self.name}_mutated"

        return PopulationGraph(
            G,
            name,
            self.category,
            params=self.params.copy(),
            labeled_edges=self.labeled_edges,
        )

    # --- UTULITIES ---
    def to_adjacency_matrix(self):
        """
        Converts the graph into a NumPy matrix.
        """
        if self.graph is None:
            raise ValueError("Graph not initialized.")
        return nx.to_numpy_array(self.graph)

    # --- VISUALIZATION ---
    def draw(
        self,
        ax=None,
        filename="",
        descriptive=True,
        with_labels=False,
        title=None,
        node_color="skyblue",
        node_size=50,
    ):
        """Draws the graph using its stored biological layout.

        ``node_color`` and ``node_size`` go straight to ``nx.draw``; the
        defaults reproduce the plain view. They exist so ``draw_colored_graph``
        can reuse the layout, label and save logic below instead of copying it.
        """
        import matplotlib.pyplot as plt  # lazy: keep matplotlib off the module-import path

        if self.graph is None:
            return

        # 1. Coordinate Retrieval
        pos = nx.get_node_attributes(self.graph, "pos")
        if not pos:
            pos = nx.spring_layout(self.graph, seed=42)

        # 2. Canvas Setup
        created_internally = False  # Track if we created the figure
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = plt.gca()
            created_internally = True

        with_edge_labels = self.labeled_edges

        # Node keys are integers after convert_node_labels_to_integers, so factories
        # that carry meaningful names (avian: parabronchi, air sacs, trachea) stash
        # them in a 'label' node attribute. Prefer those over the integer keys;
        # graphs without the attribute fall back to the keys as before.
        label_kwargs = {}
        if with_labels:
            node_labels = nx.get_node_attributes(self.graph, "label")
            if node_labels:
                label_kwargs = {
                    "labels": node_labels,
                    "font_size": 8,
                    # names are far wider than the 50pt markers. Sit them above the
                    # node rather than on it, so they do not cover the edges (and,
                    # on the directed variant, the arrowheads); the translucent box
                    # keeps them readable where rods run close together.
                    "verticalalignment": "bottom",
                    "bbox": dict(
                        boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7
                    ),
                }

        # 3. Drawing
        nx.draw(
            self.graph,
            pos=pos,
            ax=ax,
            node_size=node_size,
            node_color=node_color,
            with_labels=with_labels,
            edge_color="#555555",
            width=1.5,
            **label_kwargs,
        )

        if with_edge_labels:
            edge_labels = nx.get_edge_attributes(self.graph, "label")
            nx.draw_networkx_edge_labels(
                self.graph, pos, edge_labels, ax=ax, font_size=8
            )

        if title:
            ax.set_title(title, fontsize=14)
        else:
            ax.set_title(self.name, fontsize=14)

        # --- ADDED: Descriptive Stats ---
        if descriptive:
            n_nodes = self.graph.number_of_nodes()
            n_edges = self.graph.number_of_edges()
            stats_text = f"Nodes (N): {n_nodes}\nEdges (E): {n_edges}"

            # Place text in bottom-right corner (0.98, 0.02) relative to axes
            ax.text(
                0.98,
                0.02,
                stats_text,
                transform=ax.transAxes,
                horizontalalignment="right",
                verticalalignment="bottom",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9),
            )
        # --------------------------------

        ax.axis("off")

        # 4. Saving Logic (Robust)
        if filename:
            # Retrieve the immediate parent
            root_fig = ax.get_figure()
            if root_fig is None:
                raise RuntimeError("Could not retrieve figure from axis.")

            # CHECK: If it's a SubFigure (which has no savefig), get the REAL parent
            # SubFigures have a .figure attribute pointing to the top-level Figure
            if not hasattr(root_fig, "savefig") and hasattr(root_fig, "figure"):
                root_fig = root_fig.figure

            # Save
            root_fig.savefig(filename, dpi=300, bbox_inches="tight")
            print(f"Saved graph to {filename}")

            # CLEANUP: Only close if WE created the figure.
            # If the user passed 'ax', they manage the lifecycle.
            if created_internally:
                plt.close(root_fig)
        elif created_internally:
            # Only show if we created it; otherwise let caller control show()
            plt.show()

    def draw_colored_graph(
        self,
        values,
        cmap="viridis",
        label="",
        center=None,
        half_width=None,
        vmin_floor=None,
        node_size=300,
        ax=None,
        filename="",
        title=None,
        with_labels=False,
        descriptive=False,
        bad_color="lightgray",
    ):
        """Draw the graph with each node colored by a per-node number.

        ``values`` is any sequence of length ``n_nodes`` indexed by node id
        (win fraction, mean steps to fixation, degree, ...) and ``label`` says
        what the number means: it becomes the colorbar label.

        ``center`` picks between the two shapes these quantities come in. Left
        as None the scale runs linearly from min to max, which is what a
        magnitude such as fixation time wants. Given a value, the scale becomes
        a ``TwoSlopeNorm`` centered there and the color answers "above or below
        the reference?" rather than "how big?". A win fraction wants
        ``center=1/n_nodes`` with a diverging cmap, because under neutral drift
        the only interesting contrast is deviation from 1/N.

        ``half_width`` fixes the +/- range around ``center`` instead of fitting
        it to the largest deviation. Auto-fitting is wrong whenever the values
        are noisy estimates of the reference itself: a perfectly neutral result
        would have its scale zoomed into pure Monte Carlo scatter and come out
        looking like signal. Pass a known noise floor (or a shared range, to put
        several panels on one scale) to stop that.

        ``vmin_floor`` bounds the low end of that range from below, for a
        quantity that cannot go there. A win fraction is a probability, so a
        scale reaching to -0.9 (which is what ``center=1/21`` and a node winning
        every trial produce) spends most of its low half on values that cannot
        occur. Clamping only the low end keeps ``center`` on the middle color,
        at the price of the two sides covering unequal ranges of value.

        NaN is allowed and means "undefined at this node", which is a real case
        rather than a mistake: the mean time to takeover *conditional on a node
        winning* does not exist at a node that never won, and on a single-source
        directed graph that is every node but one. Those nodes are drawn in
        ``bad_color`` and left out of the scale, so one undefined entry cannot
        collapse the whole colorbar.

        Returns the ``Normalize`` in use, so several panels can be redrawn on a
        shared scale.
        """
        import matplotlib.pyplot as plt  # lazy: keep matplotlib off the module-import path
        import matplotlib.colors as mcolors

        vals = np.asarray(values, dtype=float)
        if vals.shape != (self.n_nodes,):
            raise ValueError(
                f"values must hold one number per node: expected shape "
                f"({self.n_nodes},), got {vals.shape}"
            )

        if np.all(np.isnan(vals)):
            raise ValueError(
                f"every value is NaN, so there is nothing to scale a colormap to "
                f"({self.name}, {self.n_nodes} nodes)"
            )

        if center is None:
            norm = mcolors.Normalize(vmin=np.nanmin(vals), vmax=np.nanmax(vals))
        else:
            if half_width is not None:
                max_dev = float(half_width)
            else:
                max_dev = float(np.nanmax(np.abs(vals - center)))
            if max_dev == 0:
                # TwoSlopeNorm needs vmin < vcenter < vmax, so a constant vector
                # (every node exactly at the reference) still needs a half-width.
                max_dev = abs(float(center)) or 1.0
            vmin = center - max_dev
            if vmin_floor is not None:
                if vmin_floor >= center:
                    raise ValueError(
                        f"vmin_floor must sit below center, got "
                        f"vmin_floor={vmin_floor} and center={center}"
                    )
                vmin = max(vmin, float(vmin_floor))
            norm = mcolors.TwoSlopeNorm(
                vmin=vmin, vcenter=center, vmax=center + max_dev
            )

        # Copy before set_bad: get_cmap hands back the registered instance, and
        # mutating that would change the colormap for every other figure.
        colormap = plt.get_cmap(cmap).copy()
        colormap.set_bad(bad_color)
        # Index by node id, not by position: nx.draw colors nodes in G.nodes()
        # order, which is insertion order and not guaranteed to be sorted.
        node_colors = colormap(norm(vals[list(self.graph.nodes())]))

        created_internally = ax is None
        if created_internally:
            _, ax = plt.subplots(figsize=(8, 7))
        fig = ax.get_figure()

        # Reuse draw() for layout, labels and the axis. Passing ax keeps it from
        # saving or showing, so the colorbar lands before anything is written.
        self.draw(
            ax=ax,
            descriptive=descriptive,
            with_labels=with_labels,
            title=title,
            node_color=node_colors,
            node_size=node_size,
        )

        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label=label, shrink=0.8)

        if filename:
            # A SubFigure has no savefig; its .figure is the real top-level one.
            root_fig = fig if hasattr(fig, "savefig") else fig.figure
            root_fig.savefig(filename, dpi=300, bbox_inches="tight")
            print(f"Saved graph to {filename}")
            if created_internally:
                plt.close(root_fig)
        elif created_internally:
            plt.show()

        return norm

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

    def to_simulation_struct(self) -> GraphCore:
        """Convert to a compact CSR GraphCore for HPC serialization.

        Builds adjacency once from NetworkX and packs it into two flat int32
        arrays so workers never need to inflate the full NetworkX graph.
        """
        G = self.graph
        n = self.n_nodes
        adj = [list(G.neighbors(i)) for i in range(n)]
        offsets = np.zeros(n + 1, dtype=np.int32)
        for i in range(n):
            offsets[i + 1] = offsets[i] + len(adj[i])
        nbrs = np.empty(int(offsets[n]), dtype=np.int32)
        for i in range(n):
            nbrs[offsets[i] : offsets[i + 1]] = adj[i]
        return GraphCore(
            n_nodes=n, nbrs=nbrs, offsets=offsets, wl_hash=self.wl_hash, name=self.name
        )

    # --- HPC SERIALIZATION ---
    def save(self, filepath: str):
        """
        Serializes the entire PopulationGraph object to a file.
        Used to send the graph topology to HPC worker nodes.
        """
        # Ensure the directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "wb") as f:
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

        with open(filepath, "rb") as f:
            graph_obj = pickle.load(f)

        if not isinstance(graph_obj, PopulationGraph):
            raise TypeError(
                f"Loaded object is not a PopulationGraph. Got: {type(graph_obj)}"
            )

        return graph_obj


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register graphs in the database.")
    parser.add_argument(
        "--batch-dir", required=True, help="Directory for graph_props.csv"
    )
    parser.add_argument(
        "--graph-zoo-path", required=True, help="Path to pickle file with graphs"
    )
    parser.add_argument("--register", action="store_true", help="Enable registration")

    args = parser.parse_args()

    if args.register:
        print("Registering graphs in database...")
        PopulationGraph.batch_register(args.graph_zoo_path, args.batch_dir)

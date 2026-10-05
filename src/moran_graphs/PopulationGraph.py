import networkx as nx
import numpy as np

class PopulationGraph:
    "Container of networkx graph. Used for respiratory graph factories."

    def __init__(self, graph:nx.Graph, name:str, category):
        self.name = name
        self.graph = graph
        self.category = category

    
    @classmethod    
    def complete_graph(cls, n_nodes: int) -> "PopulationGraph":
        """
        Creates a Fully Connected graph (everyone connected to everyone).
        """
        name = f"complete_n{n_nodes}"
        return cls(
            nx.complete_graph(n_nodes),
            name=name,
            category="Complete",
        )

    @classmethod
    def cycle_graph(cls, n_nodes: int, directed: bool = False)-> "PopulationGraph":
        """
        Creates a ring graph.

        Args:
            n_nodes (int): Number of nodes in the ring.
            directed (bool): If True, returns a DiGraph circulating one way.
        """
        G = nx.cycle_graph(n_nodes, create_using=nx.DiGraph if directed else None)

        angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
        pos = {i: np.array([np.cos(a), np.sin(a)]) for i, a in enumerate(angles)}
        nx.set_node_attributes(G, pos, "pos")

        name = f"directed_cycle_n{n_nodes}" if directed else f"cycle_n{n_nodes}"
        return cls(
            G,
            name=name,
            category="Cycle",
        )

    @classmethod
    def line_graph(cls, n_nodes: int, directed: bool = False)-> "PopulationGraph":
        """
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
        )

    @classmethod
    def star_graph(
        cls, n_nodes: int, directed: bool = False)-> "PopulationGraph":
        """
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
        )


    @classmethod
    def grid_graph(cls, width: int, height: int)-> "PopulationGraph":
        """
        Creates a rectangular grid (2D lattice) of size width x height.

        Every node is connected to its left/right and top/bottom neighbors only
        (4-connectivity).

        Args:
            width (int): Number of columns.
            height (int): Number of rows.
        """
        if width < 1 or height < 1:
            raise ValueError("width and height must both be at least 1")

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
        )

    @classmethod
    def mammalian_lung_graph(
        cls,
        depth: int,
        branching_factor: int = 2,
        directed: bool = False,
        name=None,
    )-> "PopulationGraph":
        """Generates a tree shaped population graph mimicking mammalian lung topology.

        With directed=True every edge points parent -> child, i.e. The root is then the only source.

        Args:
            branching_factor (int): Children per internal node.
            depth (int): Levels below the root.
            directed (bool): If True, returns a DiGraph flowing root -> leaves.
        """
        G = nx.balanced_tree(
            branching_factor, depth, create_using=nx.DiGraph if directed else None
        )
        if name is None:
            name = f"mammalian_b{branching_factor}_d{depth}"

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
        return cls(
            G,
            name=name,
            category="Mammalian",
        )

    @classmethod
    def avian_graph(
        cls,
        n_parabronchi: int,
        l_parabronchi: int,
        directed: bool = False,
        name="avian",
    )-> "PopulationGraph": 
        """
        Generates a graph mimicking Avian Lungs topology.

        Structure:
        - n_parabronchi: Number of parallel 'parabronchi' (linear paths).
        - l_parabronchi: Number of nodes in each parabronchi.
        - Connectivity: every parabronchus runs from the posterior air sac to the
          anterior air sac; the trachea closes the loop from anterior back to
          posterior.

        With directed=True every edge points downstream, so air (and offspring)
        flow one way only: posterior air sac -> parabronchi -> anterior air sac
        -> trachea -> posterior air sac. Node identifiers are anatomical and are
        preserved as a 'label' node attribute after the integer relabelling, so
        draw(with_labels=True) shows them.

        Args:
            n_parabronchi (int): Number of parallel paths.
            l_parabronchi (int): Number of nodes per path.
            directed (bool): If True, returns a DiGraph (unidirectional flow).
        """
        G = nx.DiGraph() if directed else nx.Graph()

        inlet, outlet, circuit = "posterior-air-sac", "anterior-air-sac", "trachea"
        G.add_nodes_from([inlet, outlet, circuit])

        # 1. Define Macro Layout
        pos = {}
        x_start, x_end = 0, l_parabronchi + 1
        y_center = 0

        pos[inlet] = np.array([x_start - 1, y_center])
        pos[outlet] = np.array([x_end + 1, y_center])
        pos[circuit] = np.array(
            [(x_start + x_end) / 2, y_center - (n_parabronchi / 2) - 2]
        )  # Loop below

        # Connect the loop
        G.add_edge(outlet, circuit)
        G.add_edge(circuit, inlet)

        # 2. Generate Parallel Rods
        for i in range(n_parabronchi):
            # Center rods around Y=0
            y = (i - (n_parabronchi - 1) / 2) * 1.0

            rod = [f"parabronchus_{i}_{j}" for j in range(l_parabronchi)]
            for j, node_id in enumerate(rod):
                pos[node_id] = np.array([x_start + j + 0.5, y])

            # inlet -> rod[0] -> ... -> rod[-1] -> outlet; follows G's directedness
            nx.add_path(G, [inlet, *rod, outlet])

        # 3. Store pos & Convert labels
        # The simulation layer indexes nodes as range(n) (see to_simulation_struct),
        # so the anatomical identifiers cannot survive as node keys. label_attribute
        # keeps them as a node attribute instead, which draw() picks up and which the
        # WL hash ignores (weisfeiler_lehman_graph_hash reads node attrs only when
        # node_attr is passed), so relabelling does not invalidate any existing batch.
        nx.set_node_attributes(G, pos, "pos")
        G = nx.convert_node_labels_to_integers(G, label_attribute="label")
        prefix = "directed_avian" if directed else "avian"
        name = f"{prefix}_r{n_parabronchi}_l{l_parabronchi}"
        return cls(
            G,
            name,
            category="Avian",
        )

    @classmethod
    def fish_graph(
        cls,
        n_rods: int,
        rod_length: int,
        fillaments: int = 2,
        directed: bool = False,
        name="fish",
    )-> "PopulationGraph":
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
            directed (bool): If True, returns a DiGraph flowing out from the root,
                the middle arch node (main_{(n_rods - 1) // 2}): along the arch
                away from the root, main -> filament tip, and filament node ->
                its lamellae.
        """
        G = nx.DiGraph() if directed else nx.Graph()
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

            for j, c_node in enumerate(c_nodes):
                center = np.array([main_rod_x + (j + 1) * c_spacing, y_base])
                pos[c_node] = center

                lamellae = [f"r{i}u{j}_{m}" for m in range(n_up)] + [
                    f"r{i}d{j}_{m}" for m in range(n_down)
                ]
                for node_id, offset in zip(lamellae, up_offsets + down_offsets):
                    pos[node_id] = center + offset
                nx.add_star(G, [c_node, *lamellae])

            nx.add_path(G, c_nodes)  # Linear filament

        # Connect Main Arch vertically
        if directed:
            # Two paths out of the middle node, so the arch flows up and down from it
            root = (n_rods - 1) // 2
            nx.add_path(G, main_nodes[root::-1])
            nx.add_path(G, main_nodes[root:])
        else:
            nx.add_path(G, main_nodes)

        nx.set_node_attributes(G, pos, "pos")
        G = nx.convert_node_labels_to_integers(G)
        prefix = "directed_fish" if directed else "fish"
        name = f"{prefix}_r{n_rods}_l{rod_length}_f{fillaments}"
        return cls(
            G,
            name,
            category="Fish",
        )

    # TODO ensure this is correctly uniformly random. 
    @classmethod
    def random_graph(
        cls,
        n_nodes: int,
        n_edges: int,
        seed=None,
        max_tries: int = 100,
        mixing_steps: int | None = None,
    )-> "PopulationGraph":
        """
        Samples a graph uniformly from all CONNECTED labeled graphs with N nodes
        and E edges (a disconnected population can never fixate).

        - E = N - 1: the graph is a tree, drawn exactly via a random Pruefer sequence.
        - Otherwise: draw G(N, E) and reject until connected, which is exact. Near
          the tree limit almost every draw is disconnected (N=60, E=64 connects
          ~0 times in 2000), so after max_tries it falls back to an MCMC chain
          whose stationary distribution is uniform on the same set.

        Args:
            n_nodes (int): Number of nodes N.
            n_edges (int): Number of edges E, with N - 1 <= E <= N(N - 1) / 2.
            seed: None, an int or a np.random.Generator (pass a Generator to draw
                several independent graphs from one stream).
            max_tries (int): G(N, E) draws before switching to MCMC.
            mixing_steps (int): MCMC steps; defaults to 20 * E.
        """
        max_edges = n_nodes * (n_nodes - 1) // 2
        if not n_nodes - 1 <= n_edges <= max_edges:
            raise ValueError(
                f"a connected graph on {n_nodes} nodes needs "
                f"{n_nodes - 1} <= n_edges <= {max_edges}, got {n_edges}"
            )
        rng = np.random.default_rng(seed)

        if n_edges == n_nodes - 1:
            G = nx.random_labeled_tree(n_nodes, seed=rng)
        else:
            for _ in range(max_tries):
                G = nx.gnm_random_graph(n_nodes, n_edges, seed=rng)
                if nx.is_connected(G):
                    break
            else:
                G = cls._connected_gnm_mcmc(
                    n_nodes, n_edges, rng, mixing_steps or 20 * n_edges
                )

        return cls(G, f"random_n{n_nodes}_e{n_edges}", category="Random")

    @staticmethod
    def _connected_gnm_mcmc(n_nodes, n_edges, rng, n_steps):
        """Edge-swap chain on connected graphs with n_nodes nodes and n_edges edges.

        Each step moves one uniformly chosen edge to one uniformly chosen non-edge
        and undoes it if that disconnects the graph. The proposal is symmetric, so
        the chain's stationary distribution is uniform over connected graphs.
        """

        def random_non_edge():
            while True:
                u, v = rng.choice(n_nodes, size=2, replace=False)
                if not G.has_edge(u, v):
                    return int(u), int(v)

        # Start connected: a random spanning tree plus random extra edges.
        G = nx.random_labeled_tree(n_nodes, seed=rng)
        while G.number_of_edges() < n_edges:
            G.add_edge(*random_non_edge())

        edges = list(G.edges)
        for _ in range(n_steps):
            i = rng.integers(n_edges)
            old = edges[i]
            new = random_non_edge()
            G.remove_edge(*old)
            G.add_edge(*new)
            # G was connected, so it still is iff old's endpoints are still joined.
            if nx.has_path(G, *old):
                edges[i] = new
            else:
                G.remove_edge(*new)
                G.add_edge(*old)
        return G

    @classmethod
    def sample_random_graphs(
        cls, n_nodes: int, n_edges: int, M: int, seed=None, distinct=True, **kwargs
    ):
        """
        Samples M random connected graphs with N nodes and E edges (see random_graph).

        With distinct=True no two returned graphs are isomorphic. Candidates are
        bucketed by Weisfeiler-Lehman hash and only same-bucket graphs are compared
        with a full isomorphism test. Raises if M distinct graphs are not found
        within 100 * M draws (e.g. there are only 2 trees on 4 nodes).

        Extra kwargs are passed to random_graph.
        """
        rng = np.random.default_rng(seed)
        graphs = []
        buckets = {}  # WL hash -> graphs already kept with that hash

        for _ in range(100 * M):
            if len(graphs) == M:
                break
            pg = cls.random_graph(n_nodes, n_edges, seed=rng, **kwargs)
            if distinct:
                bucket = buckets.setdefault(nx.weisfeiler_lehman_graph_hash(pg.graph), [])
                if any(nx.is_isomorphic(pg.graph, h) for h in bucket):
                    continue
                bucket.append(pg.graph)
            pg.name = f"{pg.name}_{len(graphs)}"
            graphs.append(pg)
        else:
            if len(graphs) < M:
                raise RuntimeError(
                    f"found only {len(graphs)} of {M} distinct graphs with "
                    f"{n_nodes} nodes and {n_edges} edges"
                )
        return graphs

    def draw(
            self,
            ax=None,
            descriptive=True,
            with_labels=False,
            title=None,
            node_color="skyblue",
            node_size=50,
        ):
            """Draws the graph using its stored biological layout and returns the Axes.

            Draws onto ``ax`` if given, otherwise onto a new figure. Showing or
            saving is left to the caller (``ax.figure.savefig(...)``).
            ``node_color`` and ``node_size`` go straight to ``nx.draw``.
            """
            import matplotlib.pyplot as plt  # lazy: keep matplotlib off the module-import path

            if self.graph is None:
                return

            # 1. Coordinate Retrieval
            pos = nx.get_node_attributes(self.graph, "pos")
            if not pos:
                pos = nx.spring_layout(self.graph, seed=42)

            # 2. Canvas Setup
            if ax is None:
                _, ax = plt.subplots(figsize=(10, 8))

            # with_edge_labels = self.labeled_edges

            # # Node keys are integers after convert_node_labels_to_integers, so factories
            # # that carry meaningful names (avian: parabronchi, air sacs, trachea) stash
            # # them in a 'label' node attribute. Prefer those over the integer keys;
            # # graphs without the attribute fall back to the keys as before.
            # label_kwargs = {}
            # if with_labels:
            #     node_labels = nx.get_node_attributes(self.graph, "label")
            #     if node_labels:
            #         label_kwargs = {
            #             "labels": node_labels,
            #             "font_size": 8,
            #             # names are far wider than the 50pt markers. Sit them above the
            #             # node rather than on it, so they do not cover the edges (and,
            #             # on the directed variant, the arrowheads); the translucent box
            #             # keeps them readable where rods run close together.
            #             "verticalalignment": "bottom",
            #             "bbox": dict(
            #                 boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7
            #             ),
            #         }

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
                # **label_kwargs,
            )

            # if with_edge_labels:
            #     edge_labels = nx.get_edge_attributes(self.graph, "label")
            #     nx.draw_networkx_edge_labels(
            #         self.graph, pos, edge_labels, ax=ax, font_size=8
            #     )

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
            return ax

    def csr(self):
        """W[i, j] = weight of edge i -> j (1 if unweighted); row i is node i."""
        return nx.to_scipy_sparse_array(self.graph, format="csr")

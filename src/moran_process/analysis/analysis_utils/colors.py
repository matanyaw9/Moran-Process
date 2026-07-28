"""
Category colors, graph-property metadata, and shared plotting constants.

Pure data plus two small helpers; imports nothing else from this package, so it
sits at the bottom of the dependency graph (``plots`` pulls from here).
"""

import seaborn as sns
import matplotlib.colors as mcolors
import hashlib

__all__ = [
    "CATEGORY_COLOR_DICT",
    "GRAPH_PROPERTY_DESCRIPTION",
    "GRAPH_PROPERTY_COLUMNS",
    "DEFAULT_FIG_SIZE",
    "generate_robust_color_dict",
]


CATEGORY_COLOR_DICT = {
    # Biological (Earthy/Natural)
    "Mammalian": "#8C510A",  # Deep Brown
    "Avian": "#2E7D32",  # Forest Green
    "Fish": "#084182",  # Dark Blue
    # Structural
    "Random": "#E0E0E0",
    "Complete": "#000000",
    "Cycle": "#5C6BC0",
    "Star": "#FFE656",
    # --- PROBABILITY (Blues/Purples) ---
    "maximize LR Fixation Probability": "#08519C",  # Navy Blue
    "maximize XGBOOST Fixation Probability": "#6BAED6",  # Soft Sky Blue
    "minimize LR Fixation Probability": "#54278F",  # Deep Indigo
    "minimize XGBOOST Fixation Probability": "#9E9AC8",  # Lavender
    # --- TIME (Reds/Oranges) ---
    "maximize LR Fixation Time": "#A50F15",  # Blood Red
    "maximize XGBOOST Fixation Time": "#FC9272",  # Salmon
    "minimize LR Fixation Time": "#D94801",  # Burnt Orange
    "minimize XGBOOST Fixation Time": "#FDBB84",  # Peach
}

# The GA now targets RESIDUALS, so its categories carry a target suffix -- e.g.
# "maximize XGBOOST Fixation Probability (delta)". Without these keys they would miss the
# hand-picked colors above and fall through to auto-generated husl. Rather than restating
# 8 more hex codes, each suffixed name inherits its unsuffixed base color, so a GA target
# keeps one color whether or not it was fitted on residuals.
_GA_TARGET_SUFFIXES = {
    "Fixation Probability": " (delta)",
    "Fixation Time": " (log-ratio)",
}
CATEGORY_COLOR_DICT.update(
    {
        f"{base}{suffix}": color
        for base, color in list(CATEGORY_COLOR_DICT.items())
        for metric, suffix in _GA_TARGET_SUFFIXES.items()
        if base.endswith(metric)
    }
)

# Use a defaultdict to return 'lightgray' for unknown categories
# Paste your dictionary here (or ensure it's in the global scope)
GRAPH_PROPERTY_DESCRIPTION = {
    "n_nodes": "The total number of vertices (individuals) in the graph.",
    "n_edges": "The total number of connections (links) between nodes in the graph.",
    "density": "The ratio of actual edges to the maximum possible number of edges (0 = empty, 1 = fully connected).",
    "diameter": "The longest shortest path between any pair of nodes (the 'width' of the network).",
    "avg_degree": "The average number of connections a node has.",
    "average_clustering": "A measure of how much nodes tend to cluster together (how likely a node's neighbors are also neighbors).",
    "average_shortest_path_length": "The average number of steps required to get from one node to any other node.",
    "degree_assortativity": "The correlation between a node's degree and the degree of its neighbors (positive = high-degree nodes connect to other high-degree nodes).",
    "avg_betweenness_centrality": "The average frequency that nodes act as a bridge along the shortest path between two other nodes.",
    "max_degree": "The highest number of connections held by a single node in the graph (the 'hub' size).",
    "min_degree": "The lowest number of connections held by a single node.",
    "degree_std": "The standard deviation of degrees; measures how much variation there is in connectivity (high = mixture of hubs and leaves).",
    "transitivity": "The overall probability that two neighbors of a node are connected (similar to clustering but calculated globally).",
    "radius": "The minimum eccentricity in the graph (the shortest distance from the 'center' of the graph to the furthest node).",
    "avg_degree_centrality": "The average fraction of the total possible nodes that any given node is connected to.",
    "max_degree_centrality": "The highest centrality score; indicates the most central or well-connected node relative to network size.",
    "max_betweenness_centrality": "The score of the node that acts as the most critical bridge or bottleneck in the network.",
    "avg_closeness_centrality": "The average speed at which nodes can access all other nodes (inverse of average distance).",
    "max_closeness_centrality": "The score of the node that can reach all other nodes in the fewest number of steps.",
}


GRAPH_PROPERTY_COLUMNS = [
    "n_nodes",
    "n_edges",
    "is_directed",
    "density",
    "is_connected",
    "avg_degree",
    "max_degree",
    "min_degree",
    "degree_std",
    "degree_assortativity",
    "average_clustering",
    "transitivity",
    "diameter",
    "radius",
    "average_shortest_path_length",
    "avg_degree_centrality",
    "max_degree_centrality",
    "avg_betweenness_centrality",
    "max_betweenness_centrality",
    "avg_closeness_centrality",
    "max_closeness_centrality",
]

DEFAULT_FIG_SIZE = (8.7, 6)


def _sort_categories(categories):
    """Return categories sorted: Avian/Fish/Mammalian first, Random last, rest alphabetically."""
    BIOLOGICAL = ["Avian", "Fish", "Mammalian"]
    LAST = ["Random"]
    cat_set = set(categories)
    bio = [c for c in BIOLOGICAL if c in cat_set]
    last = [c for c in LAST if c in cat_set]
    middle = sorted(c for c in cat_set if c not in BIOLOGICAL and c not in LAST)
    return bio + middle + last


def generate_robust_color_dict(df, existing_colors, default_palette="husl"):
    """Build a category -> color dict covering every category in df['category'].

    Args:
        df: DataFrame with a 'category' column
        existing_colors: base mapping (e.g. CATEGORY_COLOR_DICT); known categories keep their color
        default_palette: seaborn palette name used to generate colors for unknown categories
    """
    # 1. Get unique, non-null categories
    categories = sorted(df["category"].dropna().unique().tolist())

    # 2. Separate known and unknown categories
    known_cats = [c for c in categories if c in existing_colors]
    unknown_cats = [c for c in categories if c not in existing_colors]

    # 3. Initialize the final dictionary with known colors
    final_color_dict = {c: existing_colors[c] for c in known_cats}

    # 4. Handle unknown categories
    if unknown_cats:
        num_unknown = len(unknown_cats)

        # Strategy A: If few unknowns, generate a nicely spaced palette
        if num_unknown <= 20:
            # 'husl' creates perceptually distinct colors
            new_colors = sns.color_palette(default_palette, n_colors=num_unknown)

            for i, cat in enumerate(unknown_cats):
                final_color_dict[cat] = mcolors.to_hex(new_colors[i])

        # Strategy B: If many unknowns, use a deterministic hash to pick colors
        # This prevents the palette from becoming an indistinguishable rainbow
        else:
            # Use a very large palette to draw from
            large_palette = sns.color_palette("hls", 50)

            for cat in unknown_cats:
                # Create a deterministic integer from the category name
                hash_val = int(hashlib.md5(cat.encode("utf-8")).hexdigest(), 16)
                # Pick a color from the palette based on the hash
                color_idx = hash_val % len(large_palette)
                final_color_dict[cat] = mcolors.to_hex(large_palette[color_idx])

    return final_color_dict

"""Structural properties of a graph, computed once per graph by the batch properties job."""

import warnings

import networkx as nx
import numpy as np
import scipy.sparse as sp

# Above this size, betweenness and closeness are estimated from SAMPLE fixed-seed sources
# and distance metrics are skipped (all-pairs BFS in Python gets slow).
EXACT_MAX_N = 2000
SAMPLE = 100


def graph_from_W(W) -> nx.Graph:
    """The networkx graph with weights W (W[i, j] = weight of i -> j), directed iff W is not symmetric."""
    W = sp.csr_array(W)
    directed = (W != W.T).nnz > 0
    return nx.from_scipy_sparse_array(W, create_using=nx.DiGraph if directed else nx.Graph)


def wl_hash(W) -> str:
    """Weisfeiler-Lehman hash of the graph with weights W (see graph_from_W).

    Weights enter the hash only if they are not all equal, so an unweighted graph
    hashes the same whatever its weight.
    """
    W = sp.csr_array(W)
    G = graph_from_W(W)
    weighted = W.nnz > 0 and not np.all(W.data == W.data[0])
    with warnings.catch_warnings():
        # networkx 3.5 changed WL hashes (so they differ from v1 batches) and warns on every call.
        warnings.simplefilter("ignore", UserWarning)
        return nx.weisfeiler_lehman_graph_hash(G, edge_attr="weight" if weighted else None)


def _or_nan(fn, *args, **kwargs):
    """fn's value, or NaN where networkx says it is undefined for this graph."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # e.g. assortativity of a regular graph
            return float(fn(*args, **kwargs))
    except (nx.NetworkXException, ZeroDivisionError, ValueError):
        return float("nan")


def properties(G: nx.Graph) -> dict:
    """The metrics of the old moran-process graph_props table, for a networkx graph."""
    n = G.number_of_nodes()
    directed = G.is_directed()
    connected = nx.is_weakly_connected(G) if directed else nx.is_connected(G)
    degrees = np.array([d for _, d in G.degree()])
    nan = float("nan")
    props = {
        "n_nodes": n,
        "n_edges": G.number_of_edges(),
        "is_directed": directed,
        "density": nx.density(G),
        "is_connected": connected,
        "avg_degree": degrees.mean(),
        "max_degree": int(degrees.max()),
        "min_degree": int(degrees.min()),
        "degree_std": degrees.std(),
        "degree_assortativity": _or_nan(nx.degree_assortativity_coefficient, G),
        "average_clustering": _or_nan(nx.average_clustering, G),
        "transitivity": _or_nan(nx.transitivity, G),  # undefined (NaN) on directed graphs
    }

    # Distances need every node to reach every other, which a digraph like the
    # directed star (one source, N - 1 sinks) does not give.
    reachable = connected and (not directed or nx.is_strongly_connected(G))
    if reachable and n <= EXACT_MAX_N:
        ecc = nx.eccentricity(G).values()
        props.update(
            diameter=max(ecc),
            radius=min(ecc),
            average_shortest_path_length=nx.average_shortest_path_length(G),
        )
    else:
        props.update(diameter=nan, radius=nan, average_shortest_path_length=nan)

    degree_c = list(nx.degree_centrality(G).values())
    exact = n <= EXACT_MAX_N
    between_c = list(nx.betweenness_centrality(G, k=None if exact else SAMPLE, seed=0).values())
    if exact:
        close_c = list(nx.closeness_centrality(G).values())
    else:
        nodes = np.random.default_rng(0).choice(list(G), size=SAMPLE, replace=False)
        close_c = [nx.closeness_centrality(G, u=u) for u in nodes]
    props.update(
        avg_degree_centrality=np.mean(degree_c),
        max_degree_centrality=max(degree_c),
        avg_betweenness_centrality=np.mean(between_c),
        max_betweenness_centrality=max(between_c),
        avg_closeness_centrality=np.mean(close_c),
        max_closeness_centrality=max(close_c),
    )
    return props

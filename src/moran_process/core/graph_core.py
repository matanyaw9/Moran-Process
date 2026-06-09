"""Compact CSR graph representation for HPC simulation workers.

`GraphCore` lives in its own dependency-light module (numpy only) so that
unpickling a zoo shard inside a worker does NOT drag in the heavy
PopulationGraph / NetworkX / matplotlib stack. The full PopulationGraph (used to
build, analyse and draw graphs) stays in ``population_graph.py`` and re-exports
``GraphCore`` for backwards compatibility with older pickled shards.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(eq=False)
class GraphCore:
    """Compact CSR representation of a graph for HPC simulation workers.

    Replaces the heavy NetworkX/PopulationGraph objects in zoo shards.
    Node i's neighbours are nbrs[offsets[i] : offsets[i+1]].
    """
    n_nodes: int
    nbrs: np.ndarray    # int32, concatenated neighbour lists (length 2E for undirected)
    offsets: np.ndarray # int32, length N+1
    wl_hash: str
    name: str

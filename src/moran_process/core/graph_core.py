"""Compact CSR graph representation for HPC simulation workers.

`GraphCore` lives in its own dependency-light module (numpy only) so that
unpickling a zoo shard inside a worker does NOT drag in the heavy
PopulationGraph / NetworkX / matplotlib stack. The full PopulationGraph (used to
build, analyse and draw graphs) stays in ``population_graph.py`` and re-exports
``GraphCore`` for backwards compatibility with older pickled shards.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(eq=False)
class GraphCore:
    """Compact CSR representation of a graph for HPC simulation workers.

    Replaces the heavy NetworkX/PopulationGraph objects in zoo shards.
    Node i's neighbours are nbrs[offsets[i] : offsets[i+1]].
    """

    n_nodes: int
    nbrs: np.ndarray  # int32, concatenated neighbour lists (length 2E for undirected)
    offsets: np.ndarray  # int32, length N+1
    wl_hash: str
    name: str

    def save(self, file):
        """Write the adjacency lists as text: nodes separated by ';', neighbours by ','.

        Only the topology is stored; wl_hash and name are not.
        """
        parts = []
        for i in range(len(self.offsets) - 1):
            [start, end] = self.offsets[i : i + 2]
            parts.append(",".join(str(adj) for adj in self.nbrs[start:end]))

        with open(file, "w") as f:
            f.write(";".join(parts))

    @staticmethod
    def load(file):
        """Inverse of `save`. The name is taken from the file stem, wl_hash is left empty."""
        with open(file, "r") as f:
            parts = f.read().strip().split(";")

        # An empty part is a node with no neighbours, so it must be kept.
        adj = [[int(v) for v in part.split(",") if v] for part in parts]
        offsets = np.zeros(len(adj) + 1, dtype=np.int32)
        offsets[1:] = np.cumsum([len(a) for a in adj])
        nbrs = np.array([v for a in adj for v in a], dtype=np.int32)

        return GraphCore(
            n_nodes=len(adj),
            nbrs=nbrs,
            offsets=offsets,
            wl_hash="",
            name=Path(file).stem,
        )

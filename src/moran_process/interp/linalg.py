import numpy as np
import itertools
from enum import Enum

from moran_process.core.graph_core import GraphCore


class Genesis(Enum):
    spontaneous = 0
    temperature = 1


class TransitionMatrix:
    # TODO: change this to a SciPy sparse matrix
    #
    # transition matrix; m[i, j] represents
    # probability to go from state i to state j.
    #
    # state i is to be understood as a boolean vector,
    # where the active bits encode the vertices which
    # contain a mutant. e.g. state 6 (0b0110) has
    # mutants in vertices no. 1, 2.
    mat: np.ndarray
    graph: GraphCore

    def __init__(self, g: GraphCore, r: float):
        n = g.n_nodes
        m = np.zeros((2**n, 2**n))

        for i in range(2**n):
            denum = (r - 1) * i.bit_count() + n
            for src in range(n):
                src_mut = (i >> src) & 1 != 0
                neighs = g.nbrs[g.offsets[src] : g.offsets[src + 1]]
                for dst in neighs:
                    j = i | (1 << dst) if src_mut else i & ~(1 << dst)
                    m[i, j] += (r if src_mut else 1) / denum / len(neighs)

        self.mat = m
        self.graph = g

    def fixation_prob(self, gen: Genesis) -> np.float64:
        q = self.mat[1:-1, 1:-1]
        r = self.mat[1:-1, -1]
        gv = self._genesis_vec(gen)[1:-1]
        return gv @ np.linalg.solve(np.identity(q.shape[1]) - q, r)

    def fixation_time(self, gen: Genesis) -> np.float64:
        q = self.mat[1:-1, 1:-1]
        r = self.mat[1:-1, -1]
        inv_row_sums = 1 / np.sum(self.mat[1:-1, 1:], axis=1)
        # don't use `*=`, that'll mutate self.mat!
        q = q * inv_row_sums.reshape((-1, 1))
        r = r * inv_row_sums
        gv = self._genesis_vec(gen)[1:-1]
        return gv @ np.linalg.solve(
            np.linalg.matrix_power(np.identity(q.shape[1]) - q, 2), r
        )

    def absorb_time(self, gen: Genesis) -> np.float64:
        q = self.mat[1:-1, 1:-1]
        r = self.mat[1:-1, 1] + self.mat[1:-1, -1]
        gv = self._genesis_vec(gen)[1:-1]
        return gv @ np.linalg.solve(
            np.linalg.matrix_power(np.identity(q.shape[1]) - q, 2), r
        )

    def _genesis_vec(self, gen: Genesis) -> np.ndarray:
        g = self.graph
        v = np.zeros(2**g.n_nodes)
        node_prob = 1 / g.n_nodes
        match gen:
            case Genesis.spontaneous:
                for src in range(g.n_nodes):
                    v[1 << src] = node_prob
            case Genesis.temperature:
                for src in range(g.n_nodes):
                    neighs = g.nbrs[g.offsets[src], g.offsets[src + 1]]
                    for dst in neighs:
                        v[1 << dst] += node_prob / len(neighs)
            case _:
                assert False
        return v

import numpy as np
from scipy import sparse
import itertools
from enum import Enum

from moran_process.core.graph_core import GraphCore


class Genesis(Enum):
    spontaneous = 0
    temperature = 1


class TransitionMatrix:
    # within the stored matrices, each state i is to be understood as a boolean
    # vector, where the active bits encode the vertices which contain a mutant.
    # e.g. state 6 (0b0110) has mutants in vertices no. 1, 2.

    # transient-to-transient matrix. dimensions are (2ⁿ-2, 2ⁿ-2).
    # q[i, j] represents the probability to go from state i-1 to state j-1.
    q: sparse.csr_array
    # transient-to-mutent-extinction matrix. dimensions are (2ⁿ-2, 1).
    # r_none[i, 0] represents the probability to go from state i-1 to state 0.
    r_none: sparse.csc_array
    # transient-to-mutant-takeover matrix. dimenstions are (2ⁿ-2, 1).
    # r_all[i, 0] represents the probability to go from state i-1 to state
    # 2ⁿ-1.
    r_all: sparse.csc_array
    graph: GraphCore

    def __init__(self, g: GraphCore, r: float):
        n = g.n_nodes
        q = sparse.lil_array((2**n - 2, 2**n - 2))
        r_none = sparse.lil_array((2**n - 2, 1))
        r_all = sparse.lil_array((2**n - 2, 1))

        for i in range(1, 2**n - 1):
            denum = (r - 1) * i.bit_count() + n
            for src in range(n):
                src_mut = (i >> src) & 1 != 0
                neighs = g.nbrs[g.offsets[src] : g.offsets[src + 1]]
                src_denum = denum * len(neighs)
                for dst in neighs:
                    j = i | (1 << dst) if src_mut else i & ~(1 << dst)
                    if j == 0:
                        r_none[i - 1, 0] += 1 / src_denum
                    elif j == 2**n - 1:
                        r_all[i - 1, 0] += r / src_denum
                    else:
                        q[i - 1, j - 1] += (r if src_mut else 1) / src_denum

        self.q = q.tocsr()
        self.r_none = r_none.tocsc()
        self.r_all = r_all.tocsc()
        self.graph = g

    def fixation_prob(self, gen: Genesis) -> np.float64:
        q = self.q
        r = self.r_all
        gv = self._genesis_vec(gen)
        iq = sparse.identity(q.shape[1]) - q
        res = gv @ sparse.linalg.spsolve(iq, r)
        return res[0]

    def fixation_time(self, gen: Genesis) -> np.float64:
        # TODO: this toarray() call makes for a big
        # allocation. see if this can be rewritten sparsely.
        row_sums = 1 - self.r_none.toarray()
        q = self.q / row_sums
        gv = self._genesis_vec(gen)
        iq = sparse.identity(q.shape[1]) - q
        # TODO: this np.ones call also makes a huge allocation,
        # see if it can be done better. same goes for the
        # call in absorb_time.
        res = gv @ sparse.linalg.spsolve(iq, np.ones((iq.shape[1], 1)))
        return res[0]

    def absorb_time(self, gen: Genesis) -> np.float64:
        q = self.q
        gv = self._genesis_vec(gen)
        iq = sparse.identity(q.shape[1]) - q
        res = gv @ sparse.linalg.spsolve(iq, np.ones((iq.shape[1], 1)))
        return res[0]

    def _genesis_vec(self, gen: Genesis) -> sparse.csr_array:
        g = self.graph
        v = sparse.lil_array((1, 2**g.n_nodes - 2))
        node_prob = 1 / g.n_nodes
        match gen:
            case Genesis.spontaneous:
                for src in range(g.n_nodes):
                    v[0, (1 << src) - 1] = node_prob
            case Genesis.temperature:
                for src in range(g.n_nodes):
                    neighs = g.nbrs[g.offsets[src], g.offsets[src + 1]]
                    for dst in neighs:
                        v[0, (1 << dst) - 1] += node_prob / len(neighs)
            case _:
                assert False
        return v.tocsr()

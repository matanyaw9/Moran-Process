import numpy as np
import scipy.sparse as sp

from moran_graphs import _engine

WILD_TYPE = -1
CENSORED = -2


def simulate(
    W, n_mutants, fitness, n_repeats, max_steps, seed, update_rule="Bd", skip_null_steps=False
):
    """Runs a multi-type Moran process n_repeats times.

    W[i, j] is the weight of edge i -> j (i can place offspring on j), e.g.
    PopulationGraph.csr(). The wild type has fitness 1; mutant type k
    starts on n_mutants[k] uniformly random nodes and has fitness fitness[k].

    Returns per-run arrays: winner (int8: k, WILD_TYPE, or CENSORED if max_steps was
    hit), steps (uint64, Moran steps), n_changes (uint64, steps in which a node changed
    type; steps - n_changes are the null steps), and censored.
    update_rule="dB" raises NotImplementedError.

    skip_null_steps=True simulates only steps between two different types and draws the
    number of skipped steps, so results have the same distribution, usually much faster.
    """
    W = sp.csr_array(W, dtype=np.float64, copy=True)
    W.sum_duplicates()  # the engine assumes one entry per edge
    W.eliminate_zeros()  # so an all-ones W takes the engine's unweighted fast path
    n = W.shape[0]
    if W.shape != (n, n) or (W.data < 0).any():
        raise ValueError("W must be square with non-negative weights")
    if len(n_mutants) != len(fitness) or len(fitness) > 127:
        raise ValueError("need one fitness per mutant type, at most 127 types")
    if sum(n_mutants) > n or min(fitness, default=0) < 0:
        raise ValueError("more mutants than nodes, or negative fitness")

    winner, steps, n_changes = _engine.simulate(
        W.indptr.astype(np.int32, copy=False),
        W.indices.astype(np.int32, copy=False),
        W.data,
        list(n_mutants),
        [float(f) for f in fitness],
        update_rule,
        skip_null_steps,
        n_repeats,
        max_steps,
        seed,
    )
    return {"winner": winner, "steps": steps, "n_changes": n_changes, "censored": winner == CENSORED}


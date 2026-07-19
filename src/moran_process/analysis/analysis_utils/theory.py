"""Exact Moran reference values for the complete (fully connected) graph.

This is the baseline every other topology is measured against: a graph amplifies
selection if it fixes more often than the complete graph of the same size at the
same r, and suppresses it if it fixes less often.

Why the complete graph is solvable at all
-----------------------------------------
All its nodes are interchangeable, so the state collapses to the mutant COUNT and
the process becomes a birth-death walk on 0..N. That is N+1 states instead of the
2^N states a general graph needs, and first-step analysis turns it into a
tridiagonal linear system. No other topology has this property, which is exactly
why every other graph has to be simulated.

Two finite-N details, both of which the usual textbook asymptotics drop
----------------------------------------------------------------------
1. Neighbor replacement. ``MoranProcess.step`` picks ``victim = choice(neighbors)``,
   so a node never overwrites itself: on a complete graph the death pool is the N-1
   other nodes, not all N. This rescales every time by (N-1)/N (3% at N=31) and
   leaves fixation probability untouched (rho depends only on q_i/p_i, in which the
   pool size cancels).
2. No asymptotics. The familiar (r+1)/(r-1) * N log N fixation time is a large-N
   expansion; it diverges as r -> 1+ while the true value stays finite. At N=31,
   r=1.1 it overshoots by 2.8x. We solve the chain exactly instead, which also
   removes the need for an r == 1 special case.

Validated against the simulated ``complete_n31`` graphs of batch
2026_06_21-respiratory-vs-random-redo: simulated / exact = 1.000 at every r.
See ``scripts/validate_theory.py``.
"""

import numpy as np
from scipy.linalg import solve_banded

__all__ = [
    "analytic_moran_fc_fixation_prob",
    "analytic_moran_fc_fixation_time",
    "analytic_moran_fc_absorption_time",
]


def _transition_rates(n, r):
    """Up/down probabilities of the mutant-count walk, for states i = 1..n-1.

        p_i = P(i -> i+1) = [r*i / (r*i + n-i)] * [(n-i) / (n-1)]
        q_i = P(i -> i-1) = [(n-i) / (r*i + n-i)] * [i / (n-1)]

    First bracket: the reproducer is drawn with probability proportional to fitness.
    Second: its offspring lands on a neighbor of the opposite type. The (n-1) is the
    neighbor convention (no self-replacement).

    The leftover 1 - p_i - q_i is a null event, where a node overwrites a same-type
    neighbor. It changes nothing but still costs a step, which is why it has to stay
    in the time equations below.
    """
    i = np.arange(1, n, dtype=float)
    total_fitness = r * i + (n - i)
    p = (r * i / total_fitness) * ((n - i) / (n - 1))
    q = ((n - i) / total_fitness) * (i / (n - 1))
    return p, q


def _solve_chain(n, r, rhs):
    """Solve (p_i + q_i) x_i - p_i x_{i+1} - q_i x_{i-1} = rhs_i, with x_0 = x_n = 0.

    Both time quantities are this one system under a different right-hand side:
    rhs = 1 gives the absorption time, rhs = phi gives E[T * 1{fixation}].
    """
    p, q = _transition_rates(n, r)
    ab = np.zeros((3, n - 1))
    ab[0, 1:] = -p[:-1]  # super-diagonal
    ab[1, :] = p + q  # diagonal
    ab[2, :-1] = -q[1:]  # sub-diagonal
    return solve_banded((1, 1), ab, rhs)


def _fixation_probs(n, r):
    """phi_i = P(fixation | i mutants), for i = 1..n-1.

    Exact in closed form because q_i/p_i = 1/r for every i, which collapses the
    general birth-death sum into a geometric series.
    """
    i = np.arange(1, n, dtype=float)
    if np.isclose(r, 1.0):
        return i / n
    x = 1.0 / r
    return (1 - x**i) / (1 - x**n)


def _elementwise(scalar_fn, n, r):
    """Apply a scalar (n, r) -> float function over broadcast arrays.

    Each distinct (n, r) pair is solved once. A results frame has tens of thousands
    of rows but only a handful of distinct (n, r) cells, so this turns ~60k linear
    solves into ~20.
    """
    n_arr, r_arr = np.broadcast_arrays(
        np.asarray(n, dtype=float), np.asarray(r, dtype=float)
    )
    if np.any(r_arr < 1):
        raise ValueError("r must be >= 1")
    pairs, inverse = np.unique(
        np.stack([n_arr.ravel(), r_arr.ravel()]), axis=1, return_inverse=True
    )
    values = np.array([scalar_fn(int(nn), float(rr)) for nn, rr in pairs.T])
    out = values[inverse].reshape(n_arr.shape)
    return out if out.ndim else out[()]


def analytic_moran_fc_fixation_prob(n, r):
    r"""Fixation probability of one mutant of fitness r on a complete graph of size n.

        rho(n, r) = (1 - 1/r) / (1 - 1/r^n),   and 1/n at r == 1.

    Exact, not an approximation. Vectorized over both arguments: pass scalars, or an
    array of sizes against a scalar r (the reference curve in the plots), or the
    n_nodes and r columns of a results frame.
    """
    n_arr, r_arr = np.broadcast_arrays(
        np.asarray(n, dtype=float), np.asarray(r, dtype=float)
    )
    if np.any(r_arr < 1):
        raise ValueError("r must be >= 1 for this to work")

    neutral = np.isclose(r_arr, 1.0)
    # The dummy r=2 keeps the neutral cells finite so the 0/0 never materializes;
    # np.where discards those entries anyway.
    x = 1.0 / np.where(neutral, 2.0, r_arr)
    rho = (1 - x) / (1 - x**n_arr)
    out = np.where(neutral, 1.0 / n_arr, rho)
    return out if out.ndim else out[()]


def analytic_moran_fc_fixation_time(n, r):
    """Expected steps to fixation on a complete graph, CONDITIONED on the mutant winning.

    This is the quantity the simulations report as ``mean_steps``, which averages
    ``steps`` over the fixating runs only. Counted in elementary birth-death events,
    including null events, matching ``MoranProcess.step``.
    """
    return _elementwise(_fixation_time_scalar, n, r)


def analytic_moran_fc_absorption_time(n, r):
    """Expected steps until the mutant either fixes or dies out, on a complete graph.

    Unconditional: averages over both outcomes. Provided for completeness; the
    simulations condition on fixation, so ``analytic_moran_fc_fixation_time`` is the
    one that matches ``mean_steps``.
    """
    return _elementwise(_absorption_time_scalar, n, r)


def _fixation_time_scalar(n, r):
    phi = _fixation_probs(n, r)
    return _solve_chain(n, r, phi)[0] / phi[0]


def _absorption_time_scalar(n, r):
    return _solve_chain(n, r, np.ones(n - 1))[0]

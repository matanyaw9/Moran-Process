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

from functools import lru_cache

import numpy as np
from scipy.linalg import solve_banded

__all__ = [
    "analytic_moran_fc_fixation_prob",
    "analytic_moran_fc_fixation_time",
    "analytic_moran_fc_absorption_time",
    "analytic_star_fixation_prob",
    "analytic_star_fixation_time",
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


# ---------------------------------------------------------------------------
# The star: the other topology whose symmetry makes it exactly solvable
# ---------------------------------------------------------------------------
#
# One hub (node 0) and m = n-1 leaves, undirected, matching
# PopulationGraph.star_graph. The leaves are interchangeable, so the state
# collapses from 2^n to the pair
#
#     (a, j)   a = hub type (0 wild, 1 mutant),  j = number of mutant leaves
#
# which is 2n states. It does NOT collapse to the mutant count alone the way
# the complete graph does: the hub is the only bridge between leaves, so
# whether it is a mutant is what decides which way the next event can go.
#
# Transitions, read straight off MoranProcess.step (reproducer drawn
# proportional to fitness, victim uniform among ITS neighbours):
#
#   from (1, j):  F = r(1+j) + (m-j)
#       -> (1, j+1)  w.p. (r/F) * (m-j)/m     hub seeds a wild leaf
#       -> (0, j)    w.p. (m-j)/F             a wild leaf overwrites the hub
#   from (0, j):  F = r*j + (1+m-j)
#       -> (0, j-1)  w.p. (1/F) * j/m         hub overwrites a mutant leaf
#       -> (1, j)    w.p. r*j/F               a mutant leaf takes the hub
#
# The leftover probability is a null event (a node copying onto a same-type
# neighbour). It still costs a step, so it stays in the time equations, as on
# the complete graph.
#
# Note what these rates say about why the star amplifies. A leaf reproduces
# into the hub, and the hub reproduces into a leaf; the hub therefore acts
# as an amplifier of whichever type most recently captured it, and a mutant
# leaf captures it at rate r while a wild leaf captures it at rate 1. The
# selective advantage gets applied twice per round trip, which is the origin
# of the classic r -> r^2 asymptotic.

_STAR_ABSORB_OFFSET = 1  # transient index t = k - 1; k = 0 is extinction


def _star_bands(n, r):
    """Transition probabilities of the (hub, mutant-leaf-count) chain.

    Returns the four off-diagonal probability vectors indexed by the transient
    state t = 2j + a - 1, t = 0 .. 2m-1, plus the count of transient states.
    """
    m = n - 1
    j1 = np.arange(0, m, dtype=float)  # states (1, j), j = 0..m-1, at t = 2j
    f1 = r * (1.0 + j1) + (m - j1)
    up1 = (r / f1) * (m - j1) / m  # (1,j) -> (1,j+1),  t -> t+2
    down1 = (m - j1) / f1  # (1,j) -> (0,j),    t -> t-1

    j0 = np.arange(1, m + 1, dtype=float)  # states (0, j), j = 1..m, at t = 2j-1
    f0 = r * j0 + (1.0 + m - j0)
    up0 = r * j0 / f0  # (0,j) -> (1,j),    t -> t+1
    down0 = (1.0 / f0) * (j0 / m)  # (0,j) -> (0,j-1),  t -> t-2
    return up1, down1, up0, down0, 2 * m


def _star_system(n, r):
    """Build (I - P) for the transient states, in solve_banded's (2, 2) layout.

    Also returns b, the one-step probability of landing in the FIXATION state,
    which is the right-hand side of the fixation-probability system.
    """
    up1, down1, up0, down0, size = _star_bands(n, r)

    ab = np.zeros((5, size))
    ab[2, 0::2] = up1 + down1  # diagonal, rows (1, j)
    ab[2, 1::2] = up0 + down0  # diagonal, rows (0, j)

    # solve_banded stores A[i, j] at ab[2 + i - j, j], so each off-diagonal is
    # indexed by its COLUMN. Transient index t: even t = 2j is state (1, j),
    # j = 0..m-1; odd t = 2j-1 is state (0, j), j = 1..m.
    ab[0, 2::2] = -up1[:-1]  # A[t, t+2], t even: -> (1, j+1), columns 2..2m-2
    ab[1, 2::2] = -up0[:-1]  # A[t, t+1], t odd:  -> (1, j),   columns 2..2m-2
    ab[3, 1:-1:2] = -down1[1:]  # A[t, t-1], t even: -> (0, j),   columns 1..2m-3
    ab[4, 1:-1:2] = -down0[1:]  # A[t, t-2], t odd:  -> (0, j-1), columns 1..2m-3

    # The two ways out of the transient set and into FIXATION (1, m): the hub
    # seeding the last wild leaf, and the last wild leaf being the hub itself.
    b = np.zeros(size)
    b[-2] = up1[-1]  # (1, m-1) -> (1, m)
    b[-1] = up0[-1]  # (0, m)   -> (1, m)
    return ab, b


@lru_cache(maxsize=None)
def _star_solve(n, r):
    """Exact (rho, E[steps | fixation]) for one uniformly placed mutant.

    Cached because a results frame holds tens of thousands of rows over a
    handful of distinct (n, r) cells, and because both public functions below
    want both halves of the same pair of solves.
    """
    if n < 3:
        raise ValueError("a star needs a hub and at least two leaves")
    ab, b = _star_system(n, r)
    phi = solve_banded((2, 2), ab, b)  # P(fixation | state)
    u = solve_banded((2, 2), ab, phi)  # E[T * 1{fixation} | state]

    # initialize_random_mutant picks one node uniformly: the hub with
    # probability 1/n, giving state (1, 0) at t = 0, otherwise a leaf, giving
    # state (0, 1) at t = 1.
    w_hub, w_leaf = 1.0 / n, (n - 1.0) / n
    rho = w_hub * phi[0] + w_leaf * phi[1]
    # Conditional time over the mixture is the ratio of the mixed E[T*1_fix]
    # to the mixed P(fix), which is exactly how mean_steps averages steps over
    # whichever runs happened to fixate.
    return rho, (w_hub * u[0] + w_leaf * u[1]) / rho


def analytic_star_fixation_prob(n, r):
    """Fixation probability of one mutant of fitness r on an undirected star of size n.

    Exact for finite n, from the collapsed 2n-state chain: not the familiar
    (1 - r^-2) / (1 - r^-2n) asymptotic, which assumes the hub equilibrates
    between leaf events and is only the n -> infinity limit.

    Matches PopulationGraph.star_graph(n) simulated with MoranProcess, with the
    mutant placed uniformly at random (hub with probability 1/n).
    """
    return _elementwise(lambda nn, rr: _star_solve(nn, rr)[0], n, r)


def analytic_star_fixation_time(n, r):
    """Expected steps to fixation on an undirected star, CONDITIONED on the mutant winning.

    The counterpart of analytic_moran_fc_fixation_time: same units (elementary
    birth-death events, null events included), same conditioning, so it is
    directly comparable to a simulated ``mean_steps``.
    """
    return _elementwise(lambda nn, rr: _star_solve(nn, rr)[1], n, r)

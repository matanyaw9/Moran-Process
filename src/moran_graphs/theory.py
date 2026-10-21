"""Exact Bd fixation probability and mean fixation time for the graph families that have a
closed formula, to check simulations against.

One mutant of fitness r (residents 1) on a uniformly random node of an unweighted graph
without self-loops, as the factories build them. Time is in Moran steps, null steps
included, and conditional on fixation, like steps_mean_given_fixed. A family's functions
return (rho, time); time is NaN where no formula is known yet.
"""

import numpy as np


def _isothermal_chain(t_plus, r):
    """rho and time of a chain on the mutant count m = 1 .. N-1 whose T-/T+ is 1/r at every m,
    from its T+. Traulsen & Hauert 2009 (Stochastic evolutionary game dynamics) eqs. 20 and 38,
    with the products of T-/T+ summed as G_j = 1 + 1/r + ... + r^-j, which also holds at r = 1."""
    G = np.cumsum(float(r) ** -np.arange(len(t_plus) + 1))
    phi = G[:-1] / G[-1]  # fixation probability from m mutants
    return phi[0], np.sum(phi / t_plus * G[-2::-1])


def complete(N, r):
    m = np.arange(1, N)
    # no self-loops: the offspring replaces one of the other N - 1
    return _isothermal_chain(r * m * (N - m) / ((N - 1) * (r * m + N - m)), r)


def cycle(N, r):
    """Undirected or directed alike: the mutants stay one arc, which gains a node only when an
    end of the arc reproduces outward, with probability r / (total fitness) in all."""
    m = np.arange(1, N)
    return _isothermal_chain(r / (r * m + N - m), r)


def star(N, r):
    """rho from Broom & Rychtar 2008 (Proc R Soc A 464:2609), as written in Chalub, arXiv:1404.3944, eq. 2."""
    n = N - 1
    q = (n + r) / (r * (n * r + 1))
    rho = (r / (n + r) + n**2 * r / (n * r + 1)) / (N * (1 + n / (n + r) * np.sum(q ** np.arange(1, n))))
    return rho, np.nan


def single_root_tree(N, r):
    """Every edge points away from the root, so nothing ever replaces it: a mutant fixes iff it
    starts there, whatever r (Lieberman, Hauert & Nowak 2005)."""
    return 1 / N, np.nan


def directed_line(N, r):
    """Single root (node 0); after it, the front advances when the last mutant reproduces,
    probability r / (r m + N - m), and nothing pushes it back. Time: the sum of 1/T+."""
    return 1 / N, N * (N - 1) * (r + 1) / (2 * r)


def directed_star(N, r):
    """Single root (the hub), which reaches a resident leaf with probability
    r (N - m) / ((N - 1)(r m + N - m)); leaves place no offspring. Time: the sum of 1/T+."""
    harmonic = np.sum(1 / np.arange(1, N))
    return 1 / N, (N - 1) * (N * harmonic - (N - 1) * (1 - 1 / r))


KNOWN = {  # (factory, directed) -> (rho, time) as functions of N and r
    ("complete_graph", False): complete,
    ("cycle_graph", False): cycle,
    ("cycle_graph", True): cycle,
    ("star_graph", False): star,
    ("star_graph", True): directed_star,
    ("line_graph", True): directed_line,
    ("mammalian_lung_graph", True): single_root_tree,
    ("asymmetric_lung_graph", True): single_root_tree,
    ("fish_graph", True): single_root_tree,
}


MORAN = r"$\rho = \frac{1 - 1/r}{1 - r^{-N}}$"
FORMULAS = {  # (rho, time) as matplotlib mathtext for figure legends, None where unknown
    complete: (MORAN, r"$T = \frac{N-1}{(1-1/r)(1-r^{-N})} \sum_{l=1}^{N-1} (1-r^{-l})(1-r^{l-N})"
                      r"(\frac{1}{N-l} + \frac{1}{rl})$"),
    cycle: (MORAN, r"$T = \frac{1}{(r-1)(1-r^{-N})} \sum_{l=1}^{N-1} (1-r^{-l})(1-r^{l-N})(rl+N-l)$"),
    star: (r"$\rho = \frac{r/(n+r) + n^2 r/(nr+1)}{N\,(1 + \frac{n}{n+r} \sum_{j=1}^{n-1} q^j)}$, "
           r"$q = \frac{n+r}{r(nr+1)}$, $n = N-1$", None),
    single_root_tree: (r"$\rho = 1/N$", None),
    directed_line: (r"$\rho = 1/N$", r"$T = \frac{N(N-1)(r+1)}{2r}$"),
    directed_star: (r"$\rho = 1/N$", r"$T = (N-1)\,(N H_{N-1} - (N-1)(1-1/r))$, $H_{N-1} = \sum_{k=1}^{N-1} 1/k$"),
}


def fixation(factory, directed, N, r):
    """(rho, time) of a factory's graph with N nodes; NaN for a family without theory."""
    known = KNOWN.get((factory, bool(directed)))
    return known(N, r) if known else (np.nan, np.nan)


def formulas(factory, directed):
    return FORMULAS.get(KNOWN.get((factory, bool(directed))), (None, None))

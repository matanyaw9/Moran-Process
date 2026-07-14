import numpy as np


def analytic_moran_fc_fixation_prob(n, r):
    r"""Analytic Moran fixation probability for a single mutant on a complete graph.

        rho(N, r) = (1 - 1/r) / (1 - 1/r^N)

    Vectorized over ``n``: pass either a scalar or an array of population sizes
    (e.g. a ``np.linspace``) together with a scalar selection coefficient ``r``,
    and the result broadcasts elementwise (the scalar ``1 - 1/r`` spreads over the
    array ``np.pow(r, n)``). At ``r == 1`` the expression is 0/0 -> nan; that
    neutral limit is exactly ``1/N``, which is drawn as a separate reference line.
    """
    if r < 1:
        raise ValueError("r must be >= 1 for this to work")
    if r == 1:
        return 1 / n
    return (1 - 1 / r) / (1 - 1 / np.pow(r, n))


def analytic_moran_fc_fixation_time(n, r):
    """Analytic moran process fixation [(r + 1)/(r - 1)]N log N"""

    if r < 1:
        raise ValueError("r must be >= 1 for this to work")

    if r == 1:
        return np.pow(n - 1, 2)
    return n * np.log(n) * (r + 1) / (r - 1)


def analytic_moran_fc_absorption_time(n, r):
    """
    Analytic Moran absorption time for a single mutant fully connected graph is
    AbsorptionTime(n, r) = [(r + 1)/r]N log N
    """

    if r < 1:
        raise ValueError("r must be > 1 for this to work")
    if r == 1:
        return n * np.log(n)
    return n * np.log(n) * (r + 1) / r

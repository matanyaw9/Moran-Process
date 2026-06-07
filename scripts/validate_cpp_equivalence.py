"""Validate that the C++ Moran core is statistically equivalent to the Python one.

For a set of representative graphs and selection coefficients, run many repeats
with both engines and check that they agree on:

  * fixation probability rho  -> two-proportion z-test (want p > alpha)
  * fixation-time distribution -> two-sample KS test on fixated-run step counts

A *high* p-value is the desired outcome: it means we cannot distinguish the two
engines, consistent with statistical equivalence. A low p-value flags a real
discrepancy (e.g. a sampling bug). Also reports the wall-clock speedup.

Run:  uv run python scripts/validate_cpp_equivalence.py
"""

import time

import numpy as np
from scipy import stats

from moran_process.core.population_graph import PopulationGraph
from moran_process.simulations.moran_simulation_process import MoranProcess
from moran_process.simulations.cpp_moran import CppMoranProcess

N_REPEATS = 20_000
ALPHA = 0.001  # strict: only flag clearly significant divergences


def run_engine(engine_cls, graph_core, r, n_repeats, seed):
    sim = engine_cls(graph_core=graph_core, selection_coefficient=r, seed=seed)
    fixations = np.empty(n_repeats, dtype=bool)
    steps = np.empty(n_repeats, dtype=np.int64)
    for i in range(n_repeats):
        sim.initialize_random_mutant()
        res = sim.run()
        fixations[i] = res["fixation"]
        steps[i] = res["steps"]
    return fixations, steps


def two_proportion_z(k1, n1, k2, n2):
    """Two-sided z-test for equality of two proportions; returns (z, p)."""
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    pval = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, pval


def build_cases():
    """A small zoo spanning topologies and selection regimes."""
    graphs = [
        PopulationGraph.complete_graph(n_nodes=10),
        PopulationGraph.cycle_graph(n_nodes=20),
        PopulationGraph.mammalian_lung_graph(branching_factor=2, depth=4),
    ]
    r_values = [0.5, 1.0, 2.0]  # deleterious, neutral, advantageous
    return graphs, r_values


def main():
    graphs, r_values = build_cases()
    print(f"Validation: {N_REPEATS} repeats/engine, alpha={ALPHA}\n")

    any_fail = False
    py_time_total = 0.0
    cpp_time_total = 0.0

    header = (f"{'graph':<22}{'r':>5}{'rho_py':>9}{'rho_cpp':>9}"
              f"{'p(rho)':>10}{'p(KS)':>10}{'speedup':>9}  verdict")
    print(header)
    print("-" * len(header))

    for g in graphs:
        gc = g.to_simulation_struct()
        for r in r_values:
            t0 = time.perf_counter()
            fx_py, st_py = run_engine(MoranProcess, gc, r, N_REPEATS, seed=12345)
            t_py = time.perf_counter() - t0

            t0 = time.perf_counter()
            fx_cpp, st_cpp = run_engine(CppMoranProcess, gc, r, N_REPEATS, seed=12345)
            t_cpp = time.perf_counter() - t0

            py_time_total += t_py
            cpp_time_total += t_cpp

            k_py, k_cpp = int(fx_py.sum()), int(fx_cpp.sum())
            rho_py, rho_cpp = k_py / N_REPEATS, k_cpp / N_REPEATS
            _, p_rho = two_proportion_z(k_py, N_REPEATS, k_cpp, N_REPEATS)

            # KS on step counts of runs that fixated (the fixation-time dist).
            st_py_fix = st_py[fx_py]
            st_cpp_fix = st_cpp[fx_cpp]
            if len(st_py_fix) > 1 and len(st_cpp_fix) > 1:
                p_ks = stats.ks_2samp(st_py_fix, st_cpp_fix).pvalue
            else:
                p_ks = float("nan")

            ok = (p_rho > ALPHA) and (np.isnan(p_ks) or p_ks > ALPHA)
            any_fail = any_fail or not ok
            speedup = t_py / t_cpp if t_cpp > 0 else float("inf")

            print(f"{g.name:<22}{r:>5}{rho_py:>9.4f}{rho_cpp:>9.4f}"
                  f"{p_rho:>10.3f}{p_ks:>10.3f}{speedup:>8.1f}x  "
                  f"{'OK' if ok else 'FAIL <<<'}")

    print("-" * len(header))
    print(f"\nTotal wall-clock: Python {py_time_total:.2f}s | "
          f"C++ {cpp_time_total:.2f}s | overall speedup "
          f"{py_time_total / cpp_time_total:.1f}x")
    print("\nRESULT:", "ALL STATISTICALLY EQUIVALENT"
          if not any_fail else "DISCREPANCY DETECTED")
    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())

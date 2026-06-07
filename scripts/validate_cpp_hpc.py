"""HPC validation of C++ vs Python Moran engine (statistical equivalence).

Two modes:

  submit   Build a small validation zoo and submit TWO LSF batches with the same
           batch_seed / r-values / n_repeats -- one with --engine python, one with
           --engine cpp. The heavy simulation runs on WEXAC, not the login node.

  compare  After both batches finish, read their per-job Parquet results and test
           agreement per (graph, r) cell:
              * fixation probability rho -> two-proportion z-test (want p > alpha)
              * fixation-time distribution -> two-sample KS test (want p > alpha)
           This step is light and safe to run interactively.

Typical use (from an inode session or the login node -- submit only bsubs):

  uv run python scripts/validate_cpp_hpc.py submit --base-name cpp_validate_01
  # ... wait for both LSF arrays to finish ...
  uv run python scripts/validate_cpp_hpc.py compare --base-name cpp_validate_01

Both batches live under simulation_data/<base>_python and <base>_cpp.
"""

import argparse
import glob
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats

from moran_process.core.population_graph import PopulationGraph
from moran_process.pipeline.process_lab import ProcessLab

ROOT = Path(os.getcwd())
SIMULATION_DATA_DIR = ROOT / "simulation_data"

# Fixed so the two batches (and reruns) share task structure. The two engines
# still diverge per-trajectory (different RNG families) -- this only keeps the
# (graph, r, n_repeats) layout identical so the comparison is apples-to-apples.
VALIDATION_SEED = 20260607

ALPHA = 0.001  # strict: only flag clearly significant divergences


def build_validation_zoo():
    """A small zoo spanning topologies relevant to the thesis."""
    return [
        PopulationGraph.complete_graph(n_nodes=10),
        PopulationGraph.cycle_graph(n_nodes=20),
        PopulationGraph.mammalian_lung_graph(branching_factor=2, depth=4),
        PopulationGraph.avian_graph(n_rods=4, rod_length=7),
        PopulationGraph.fish_graph(n_rods=3, rod_length=3),
    ]


def _submit_one(zoo, r_values, n_repeats, n_jobs, batch_name, engine, queue, memory):
    batch_dir = SIMULATION_DATA_DIR / batch_name
    tmp_dir = batch_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    zoo_path = tmp_dir / "graph_zoo.joblib"
    with open(zoo_path, "wb") as f:
        joblib.dump(zoo, f)
    print(f"[{engine}] serialized {len(zoo)} graphs -> {zoo_path}")

    ProcessLab().submit_jobs(
        zoo_path=zoo_path,
        n_graphs=len(zoo),
        r_values=r_values,
        batch_name=batch_name,
        batch_dir=batch_dir,
        n_repeats=n_repeats,
        n_requested_jobs=n_jobs,
        queue=queue,
        memory=memory,
        batch_seed=VALIDATION_SEED,
        engine=engine,
        description=f"C++/Python equivalence validation ({engine} engine)",
        notes="Paired with the other-engine batch sharing VALIDATION_SEED.",
    )


def cmd_submit(args):
    zoo = build_validation_zoo()
    r_values = args.r_values
    print(f"Validation zoo: {[g.name for g in zoo]}")
    print(f"r_values={r_values}  n_repeats={args.n_repeats}  n_jobs={args.n_jobs}\n")

    for engine in ("python", "cpp"):
        batch_name = f"{args.base_name}_{engine}"
        print("=" * 60)
        print(f"Submitting batch '{batch_name}' (engine={engine})")
        print("=" * 60)
        _submit_one(zoo, r_values, args.n_repeats, args.n_jobs,
                    batch_name, engine, args.queue, args.memory)

    print("\nBoth batches submitted. When they finish, run:")
    print(f"  uv run python scripts/validate_cpp_hpc.py compare "
          f"--base-name {args.base_name}")


def _load_results(batch_name):
    """Read all per-job Parquet results for a batch into one DataFrame."""
    pattern = str(SIMULATION_DATA_DIR / batch_name / "tmp" / "results"
                  / "result_job_*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No result Parquet files for batch '{batch_name}' "
                                f"(looked in {pattern}). Has it finished?")
    df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    print(f"[{batch_name}] loaded {len(df)} rows from {len(files)} job files")
    return df


def two_proportion_z(k1, n1, k2, n2):
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    return z, 2 * (1 - stats.norm.cdf(abs(z)))


def cmd_compare(args):
    df_py = _load_results(f"{args.base_name}_python")
    df_cpp = _load_results(f"{args.base_name}_cpp")

    keys = ["graph_name", "r"]
    header = (f"{'graph':<22}{'r':>6}{'rho_py':>9}{'rho_cpp':>9}"
              f"{'n_py':>8}{'n_cpp':>8}{'p(rho)':>10}{'p(KS)':>10}  verdict")
    print("\n" + header)
    print("-" * len(header))

    any_fail = False
    cells = sorted(set(map(tuple, df_py[keys].drop_duplicates().values))
                   & set(map(tuple, df_cpp[keys].drop_duplicates().values)))

    for gname, r in cells:
        a = df_py[(df_py.graph_name == gname) & (df_py.r == r)]
        b = df_cpp[(df_cpp.graph_name == gname) & (df_cpp.r == r)]
        n_py, n_cpp = len(a), len(b)
        k_py, k_cpp = int(a.fixation.sum()), int(b.fixation.sum())
        rho_py, rho_cpp = k_py / n_py, k_cpp / n_cpp
        _, p_rho = two_proportion_z(k_py, n_py, k_cpp, n_cpp)

        st_py = a.loc[a.fixation, "steps"].to_numpy()
        st_cpp = b.loc[b.fixation, "steps"].to_numpy()
        if len(st_py) > 1 and len(st_cpp) > 1:
            p_ks = stats.ks_2samp(st_py, st_cpp).pvalue
        else:
            p_ks = float("nan")

        ok = (p_rho > ALPHA) and (np.isnan(p_ks) or p_ks > ALPHA)
        any_fail = any_fail or not ok
        print(f"{gname:<22}{r:>6}{rho_py:>9.4f}{rho_cpp:>9.4f}"
              f"{n_py:>8}{n_cpp:>8}{p_rho:>10.3f}{p_ks:>10.3f}  "
              f"{'OK' if ok else 'FAIL <<<'}")

    print("-" * len(header))
    print("\nRESULT:", "ALL STATISTICALLY EQUIVALENT"
          if not any_fail else "DISCREPANCY DETECTED")
    return 1 if any_fail else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sub = sub.add_parser("submit", help="Submit paired python+cpp batches to WEXAC")
    p_sub.add_argument("--base-name", required=True,
                       help="Base batch name; batches are <base>_python and <base>_cpp")
    p_sub.add_argument("--r-values", type=float, nargs="+",
                       default=[0.5, 1.0, 2.0], help="Selection coefficients")
    p_sub.add_argument("--n-repeats", type=int, default=20_000)
    p_sub.add_argument("--n-jobs", type=int, default=20)
    p_sub.add_argument("--queue", default="short")
    p_sub.add_argument("--memory", default="2GB")
    p_sub.set_defaults(func=cmd_submit)

    p_cmp = sub.add_parser("compare", help="Compare finished python+cpp batches")
    p_cmp.add_argument("--base-name", required=True)
    p_cmp.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    return args.func(args) or 0


if __name__ == "__main__":
    raise SystemExit(main())

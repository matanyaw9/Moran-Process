"""Compare two simulation batches and test whether they came from the same distribution.

Intended use: run two batches that are identical in every parameter EXCEPT the
simulation engine (``engine="python"`` vs ``engine="cpp"``) via ``submit_jobs``
from ``design_zoo.ipynb``. Because ``submit_jobs`` fans the work out across many
LSF jobs, even the slow Python engine finishes quickly, so this gives a fast,
large-scale equivalence check of the *full production path* (worker, manifest,
parquet I/O), not just the simulation class.

For every comparison cell ``(wl_hash, r)`` present in both batches it tests:

  * fixation probability rho   -> two-proportion z-test
  * fixation-time distribution -> two-sample KS test on fixated-run step counts

A *high* p-value is the desired outcome: it means the two engines are
indistinguishable, consistent with statistical equivalence.

Multiple-comparison handling. A batch with many graphs x r-values yields many
cells, so a single raw alpha would produce spurious flags even under true
equivalence. The overall verdict therefore uses a Bonferroni-corrected threshold
(alpha / n_cells) to control the family-wise error rate. In addition, the script
KS-tests the whole collection of p-values against Uniform(0, 1): under true
equivalence the p-values should look uniform, so a systematic skew toward zero
(too weak to trip any single Bonferroni threshold) still gets caught.

Run (offline, no compute; safe on the login node):
    uv run python scripts/compare_batches.py <batch_dir_a> <batch_dir_b>

If a batch is missing its raw_results file, it is aggregated automatically from
that batch's tmp/results/ via aggregate_results_no_load.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from moran_process.analysis.analysis_utils import (
    aggregate_results_no_load,
    resolve_results_path,
)


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


def load_batch(batch_dir):
    """Return the per-repeat results DataFrame for a batch, aggregating if needed.

    Prefers an existing raw_results.parquet / raw_results.csv. If neither
    exists, aggregates from batch_dir/tmp/results/ (Parquet preferred over CSV).
    """
    batch_path = Path(batch_dir)
    path = resolve_results_path(batch_path)

    if path is None:
        print(f"[{batch_path.name}] No aggregated file found; aggregating from tmp/results/ ...")
        out = aggregate_results_no_load(str(batch_path))
        if out is None:
            raise FileNotFoundError(
                f"[{batch_path.name}] No result files to aggregate in {batch_path/'tmp'/'results'}."
            )
        path = Path(out)

    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    print(f"[{batch_path.name}] Loaded {len(df):,} rows from {path.name}")
    return df


def cell_summary(df):
    """Collapse per-repeat rows into per-(wl_hash, r) cells.

    Returns a dict keyed by (wl_hash, r) -> {graph_name, n, k, steps_fix}, where
    n is repeat count, k is the number of fixations, and steps_fix is the array
    of step counts for fixated runs (the fixation-time distribution).
    """
    cells = {}
    for (wl_hash, r), grp in df.groupby(["wl_hash", "r"]):
        fix = grp["fixation"].to_numpy(dtype=bool)
        cells[(wl_hash, float(r))] = {
            "graph_name": grp["graph_name"].iloc[0],
            "n": len(grp),
            "k": int(fix.sum()),
            "steps_fix": grp["steps"].to_numpy()[fix],
        }
    return cells


def main():
    parser = argparse.ArgumentParser(
        description="Test whether two simulation batches are statistically equivalent."
    )
    parser.add_argument("batch_a", help="Path to the first batch directory")
    parser.add_argument("batch_b", help="Path to the second batch directory")
    parser.add_argument("--alpha", type=float, default=0.001,
                        help="Family-wise significance level (default: 0.001)")
    args = parser.parse_args()

    cells_a = cell_summary(load_batch(args.batch_a))
    cells_b = cell_summary(load_batch(args.batch_b))

    keys_a, keys_b = set(cells_a), set(cells_b)
    common = sorted(keys_a & keys_b, key=lambda kr: (cells_a[kr]["graph_name"], kr[1]))
    only_a = keys_a - keys_b
    only_b = keys_b - keys_a

    name_a, name_b = Path(args.batch_a).name, Path(args.batch_b).name
    print(f"\nComparing '{name_a}' vs '{name_b}'")
    print(f"{len(common)} common cells | "
          f"{len(only_a)} only in A | {len(only_b)} only in B\n")

    if only_a:
        print(f"WARNING: {len(only_a)} (wl_hash, r) cells only in A "
              f"(e.g. {sorted({cells_a[k]['graph_name'] for k in only_a})[:5]})")
    if only_b:
        print(f"WARNING: {len(only_b)} (wl_hash, r) cells only in B "
              f"(e.g. {sorted({cells_b[k]['graph_name'] for k in only_b})[:5]})")
    if not common:
        print("\nNo overlapping cells to compare. Did the two batches use the same zoo and r_values?")
        return 1

    n_cells = len(common)
    bonferroni = args.alpha / n_cells

    rows = []
    for key in common:
        a, b = cells_a[key], cells_b[key]
        _, r = key

        if a["n"] != b["n"]:
            print(f"NOTE: repeat-count mismatch for {a['graph_name']} r={r}: "
                  f"A={a['n']} B={b['n']} (n_repeats differed between batches)")

        _, p_rho = two_proportion_z(a["k"], a["n"], b["k"], b["n"])

        if len(a["steps_fix"]) > 1 and len(b["steps_fix"]) > 1:
            p_ks = stats.ks_2samp(a["steps_fix"], b["steps_fix"]).pvalue
        else:
            p_ks = float("nan")

        rows.append({
            "graph": a["graph_name"],
            "r": r,
            "rho_a": a["k"] / a["n"],
            "rho_b": b["k"] / b["n"],
            "p_rho": p_rho,
            "p_ks": p_ks,
            "worst": np.nanmin([p_rho, p_ks]),
        })

    rows.sort(key=lambda d: d["worst"])  # worst offenders first

    header = (f"{'graph':<22}{'r':>5}{'rho_a':>9}{'rho_b':>9}"
              f"{'p(rho)':>10}{'p(KS)':>10}  verdict")
    print("\n" + header)
    print("-" * len(header))
    n_flagged = 0
    for d in rows:
        flagged = (d["p_rho"] < bonferroni) or (
            not np.isnan(d["p_ks"]) and d["p_ks"] < bonferroni)
        n_flagged += flagged
        print(f"{d['graph']:<22}{d['r']:>5}{d['rho_a']:>9.4f}{d['rho_b']:>9.4f}"
              f"{d['p_rho']:>10.3g}{d['p_ks']:>10.3g}  "
              f"{'FLAG <<<' if flagged else 'ok'}")
    print("-" * len(header))

    # Uniformity check: under true equivalence the p-values are ~Uniform(0,1).
    all_p = np.array([d["p_rho"] for d in rows]
                     + [d["p_ks"] for d in rows if not np.isnan(d["p_ks"])])
    unif_p = stats.kstest(all_p, "uniform").pvalue

    print(f"\nCells compared:        {n_cells}")
    print(f"Bonferroni threshold:  alpha/{n_cells} = {bonferroni:.2e}")
    print(f"Cells flagged:         {n_flagged}")
    print(f"p-value uniformity:    KS vs Uniform(0,1) p = {unif_p:.3f} "
          f"({'ok' if unif_p > 0.05 else 'SKEWED <<<'})")

    equivalent = (n_flagged == 0) and (unif_p > 0.05)
    print("\nRESULT:", "STATISTICALLY EQUIVALENT" if equivalent
          else "DISCREPANCY DETECTED")
    return 0 if equivalent else 1


if __name__ == "__main__":
    raise SystemExit(main())

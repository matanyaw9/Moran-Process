"""Check the analytic complete-graph baselines against simulated complete graphs.

The baselines in ``analysis_utils.theory`` are exact for the process the simulator
actually runs, but "exact" is a claim about the model, not about the code. This script
tests it against ground truth: any batch whose zoo contains a Complete graph already
holds the answer, so we just compare.

    uv run python scripts/validate_theory.py simulation_data/<batch>

A mismatch means the theory and the simulator have drifted apart. The usual suspects
are the death-pool convention (does a node replace a neighbor, or may it replace
itself?) and the definition of a step.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from moran_process.analysis.analysis_utils.theory import (  # noqa: E402
    analytic_moran_fc_fixation_prob,
    analytic_moran_fc_fixation_time,
)

# Tolerance in standard errors. The simulated values are Monte Carlo estimates, so they
# are expected to sit within a few SE of the exact answer, not to match it digit for
# digit. 4 SE is a ~1-in-16000 false alarm rate per cell.
SIGMA_TOLERANCE = 4.0


def main(batch_dir: Path) -> int:
    stats_path = batch_dir / "graph_statistics.csv"
    if not stats_path.exists():
        print(f"No graph_statistics.csv in {batch_dir}")
        return 2

    df = pd.read_csv(stats_path)
    complete = df[df["category"] == "Complete"]
    if complete.empty:
        print(f"{batch_dir.name} has no Complete graphs to validate against.")
        return 2

    rows, failures = [], 0
    for _, g in complete.sort_values(["n_nodes", "r"]).iterrows():
        n, r, reps = int(g.n_nodes), float(g.r), int(g.n_grouped)

        exact_rho = float(analytic_moran_fc_fixation_prob(n, r))
        exact_time = float(analytic_moran_fc_fixation_time(n, r))

        # Binomial SE for rho; SE of the conditional mean for the time, which only the
        # fixating runs contribute to.
        se_rho = np.sqrt(exact_rho * (1 - exact_rho) / reps)
        se_time = g.std_steps / np.sqrt(max(reps * g.prob_fixation, 1))

        z_rho = (g.prob_fixation - exact_rho) / se_rho
        z_time = (g.mean_steps - exact_time) / se_time
        ok = abs(z_rho) < SIGMA_TOLERANCE and abs(z_time) < SIGMA_TOLERANCE
        failures += not ok

        rows.append(
            {
                "N": n,
                "r": r,
                "sim_rho": round(g.prob_fixation, 4),
                "exact_rho": round(exact_rho, 4),
                "z": round(z_rho, 1),
                "sim_T": round(g.mean_steps, 1),
                "exact_T": round(exact_time, 1),
                "z_T": round(z_time, 1),
                "T_ratio": round(g.mean_steps / exact_time, 3),
                "ok": "yes" if ok else "NO",
            }
        )

    print(f"\n{batch_dir.name}: {len(rows)} complete-graph cells, {SIGMA_TOLERANCE}-sigma tolerance\n")
    print(pd.DataFrame(rows).to_string(index=False))

    if failures:
        print(f"\n{failures} cell(s) disagree with theory beyond tolerance.")
        return 1
    print("\nAll cells agree with theory.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_dir", type=Path, help="simulation_data/<batch>")
    raise SystemExit(main(parser.parse_args().batch_dir))

"""Precompute the raw-data sample the violin and p-value figures need.

Everything in ``experiment_analysis.ipynb`` reads ``graph_statistics.csv`` except two
figures: ``plot_steps_violin`` and ``plot_steps_pvalue_matrix``. Both need *individual*
runs rather than moments, because a kernel density estimate and a Mann-Whitney U test
cannot be built from a mean and a standard deviation. So those two are the only consumers
still scanning the full 7.2e9-row batch, and they are the reason opening the notebook is
slow long after aggregation finished.

What they actually consume is small: fixation rows only, subsampled to a fixed cap per
category (a 50k sample is visually and statistically indistinguishable from the full
distribution). This job materialises that sample once per r value, on a compute node, so
every later figure is a file read of a few MB.

It caches the *data*, not the rendered figure, deliberately: a cached PNG would freeze one
styling and would not help the p-value matrix at all, while a cached sample lets the
notebook restyle, reorder and recolor interactively at no cost, and feeds both figures.

Runs the same way however it is launched --

  * automatically, as the job ProcessLab chains after the aggregation
    (see process_lab.submit_violin_cache_job); or
  * by hand:
      uv run python -m moran_process.pipeline.cache_violin_data --batch-dir <batch>

Imports from the ``io`` submodule directly (not the analysis_utils package root) so this
never pulls in the plotting stack, exactly like aggregate_batch.
"""

import argparse
import logging

from moran_process.analysis.analysis_utils.io import build_fixation_steps_cache

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cache the fixation-steps sample used by the violin / p-value figures."
    )
    parser.add_argument(
        "--batch-dir",
        required=True,
        help="Batch directory containing tmp/results/ and graph_props.csv.",
    )
    parser.add_argument(
        "--r-values",
        nargs="+",
        type=float,
        default=None,
        help="Which r values to cache (default: every r in graph_statistics.csv).",
    )
    parser.add_argument(
        "--max-points-per-category",
        type=int,
        default=50_000,
        help="Subsample cap per category; part of the cache key (default: 50000).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even where a matching cache already exists.",
    )
    args = parser.parse_args()

    written = build_fixation_steps_cache(
        args.batch_dir,
        r_values=args.r_values,
        max_points_per_category=args.max_points_per_category,
        force=args.force,
    )
    log.info("Violin cache complete: %d file(s)", len(written))

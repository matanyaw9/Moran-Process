"""Turn a finished batch into analysis-ready files.

Performs the heavy, one-shot rollup that used to run interactively at the top of
experiment_analysis.ipynb, so the notebook (and the ML step) only ever read the cache:

    tmp/results/raw_results_job_*.parquet + graph_props.csv  ->  graph_statistics.csv

It reads the per-job shards directly as a glob and never concatenates them. Stock polars
indexes rows with a u32 and its Parquet reader rejects any SINGLE file over 2**32-1 rows;
the 100K-reps batch concatenates to 7.2e9 rows, so the fused raw_results.parquet is
unreadable even though the 1000 ~7.2M-row shards it came from are each fine. Scanning the
glob sidesteps the ceiling, needs no 42GB intermediate, and is faster. See
analysis_utils.io.resolve_results_source.

There is nothing LSF-specific here: it just aggregates a batch directory, so it runs the
same way however it is launched --

  * automatically, as the dependent job ProcessLab.submit_jobs chains after the array
    (see submit_aggregation_job, which adds the ``bsub -w`` dependency); or
  * by hand, on an interactive (inode) session or via bsub:
      uv run python -m moran_process.pipeline.aggregate_batch --batch-dir <batch>

Import from the ``io`` submodule directly (not the analysis_utils package root) so this
never pulls in the plotting stack (plotly/matplotlib).
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from moran_process.analysis.analysis_utils.constants import HASH_DTYPES

from moran_process.analysis.analysis_utils.io import (
    build_graph_statistics,
    resolve_results_source,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def run_aggregation(batch_dir, include_order_stats=False):
    """Build graph_statistics.csv for a finished batch, straight from the per-job shards.

    Args:
        batch_dir: the batch directory (holds tmp/results/ and graph_props.csv).
        include_order_stats: if True, also compute the (slow) median/quartile/iqr of
            steps. Off by default -- see build_graph_statistics for the rationale.

    Returns:
        (results_source, graph_statistics_path)
    """
    batch_path = Path(batch_dir)

    results_source = resolve_results_source(batch_path)
    if results_source is None:
        raise SystemExit(
            f"No result files found under {batch_path / 'tmp' / 'results'}"
        )
    log.info("Reading raw results from: %s", results_source)

    graph_props_path = batch_path / "graph_props.csv"
    if not graph_props_path.exists():
        raise SystemExit(
            f"graph_props.csv missing at {graph_props_path}; did register_graphs run?"
        )
    df_graph_props = pd.read_csv(graph_props_path, dtype=HASH_DTYPES)

    graph_statistics_path = batch_path / "graph_statistics.csv"
    log.info(
        "Building per-(graph, r) statistics (order_stats=%s) -> %s",
        include_order_stats,
        graph_statistics_path,
    )
    build_graph_statistics(
        results_source,
        df_graph_props,
        graph_statistics_path,
        include_order_stats=include_order_stats,
    )

    log.info("Aggregation complete: %s", graph_statistics_path)
    return results_source, graph_statistics_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate a finished simulation batch into analysis-ready files."
    )
    parser.add_argument(
        "--batch-dir",
        required=True,
        help="Batch directory containing tmp/results/ and graph_props.csv.",
    )
    parser.add_argument(
        "--order-stats",
        action="store_true",
        help="Also compute the slow median/quartile/iqr of steps (default: off).",
    )
    args = parser.parse_args()
    run_aggregation(args.batch_dir, include_order_stats=args.order_stats)

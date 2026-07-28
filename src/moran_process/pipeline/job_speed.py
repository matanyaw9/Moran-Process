"""Reduce a finished batch's raw shards to per-job totals, for the speed figures.

``batch_speed_report`` draws how long each LSF worker took and how many steps it
simulated. It needs one row per job_id, and it gets there by grouping the raw results and
summing -- which in the notebook meant a full scan of the raw shards (39GB on the
100K-reps batch) every time the cell ran. That was the last job-sized computation left in
experiment_analysis.ipynb after aggregation and the violin cache moved out.

The output is one row per LSF array index:

    job_id,steps,duration

``steps`` and ``duration`` are per-job SUMS, not per-run values. The column names match
what ``batch_speed_report`` expects because it re-groups by job_id and sums again, so
feeding it these pre-summed rows yields identical numbers to feeding it the raw table.
Timing and memory come from the LSF ``.out`` logs, which that function parses itself, so
nothing here needs to touch them.

Deliberately does NOT depend on the aggregation: it reads the raw shards and needs nothing
from graph_statistics.csv, so ProcessLab hangs it off ``ended(array)`` and LSF runs it
concurrently with aggregate_batch (see process_lab.submit_job_speed_job).

Scoped to one batch by construction: job_id is an LSF array index numbered 1..N within a
single submission, so it is only meaningful alongside the batch that produced it.

Runs the same way however it is launched --

  * automatically, as the job ProcessLab chains off the array; or
  * by hand:
      uv run python -m moran_process.pipeline.job_speed --batch-dir <batch>

Imports from the ``io`` submodule directly (not the analysis_utils package root) so this
never pulls in the plotting stack, exactly like aggregate_batch.
"""

import argparse
import logging
from pathlib import Path

from moran_process.analysis.analysis_utils.io import (
    _expand_shards,
    resolve_results_source,
    scan_results,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

JOB_SPEED_FILENAME = "job_speed.csv"


def job_speed_path(batch_dir):
    """Path of a batch's per-job speed totals."""
    return Path(batch_dir) / JOB_SPEED_FILENAME


def run_job_speed(batch_dir, progress_every=200):
    """Sum steps and duration per job_id and write job_speed.csv.

    Aggregated shard by shard rather than in one group_by over the glob. The worker writes
    exactly one shard per array index, so each shard reduces to a single row and peak
    memory is one shard whatever the batch size. Summing is associative, so combining the
    per-shard rows afterwards is exact rather than approximate -- the same argument that
    makes io._aggregate_chunked exact.

    Returns:
        (output_path, n_jobs)
    """
    import polars as pl

    batch_path = Path(batch_dir)

    results_source = resolve_results_source(batch_path)
    if results_source is None:
        raise SystemExit(
            f"No result files found under {batch_path / 'tmp' / 'results'}"
        )

    files = _expand_shards(results_source)
    log.info("Summing per-job totals over %d shard(s) from %s", len(files), results_source)

    partials = []
    for i, path in enumerate(files):
        if i > 0 and i % progress_every == 0:
            log.info("  summed %d/%d shards...", i, len(files))
        lf = scan_results(path)
        names = lf.collect_schema().names()
        if "job_id" not in names:
            raise SystemExit(
                f"{path} has no job_id column; this batch predates per-job result shards "
                f"and has no per-job speed to report."
            )
        aggs = [pl.col("steps").cast(pl.Int64).sum().alias("steps")]
        if "duration" in names:
            aggs.append(pl.col("duration").cast(pl.Float64).sum().alias("duration"))
        partials.append(lf.group_by("job_id").agg(aggs).collect())

    combined = (
        pl.concat(partials)
        .group_by("job_id")
        .agg([pl.col(c).sum().alias(c) for c in partials[0].columns if c != "job_id"])
        .sort("job_id")
    )

    out_path = job_speed_path(batch_path)
    combined.write_csv(out_path)
    log.info("Wrote per-job totals for %d job(s) -> %s", combined.height, out_path)
    return out_path, combined.height


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Summarise per-job simulation totals for a finished batch."
    )
    parser.add_argument(
        "--batch-dir",
        required=True,
        help="Batch directory containing tmp/results/.",
    )
    args = parser.parse_args()
    run_job_speed(args.batch_dir)

"""One entry point for everything that has to happen after a batch's simulations finish.

A finished job array leaves 1000 parquet shards and nothing else. Four jobs turn that into
something ``experiment_analysis.ipynb`` can open instantly, and they form a small DAG:

                    register_graphs --+
                                      +-->  aggregate  --+-->  verify
       simulation array (1..N) -------+                  +-->  violin cache
                                      +-->  job speed

``aggregate`` rolls the raw rows up to one row per (graph, r). ``verify`` answers whether
every requested run actually happened and whether any job failed quietly. ``violin cache``
precomputes the one figure input that still needs individual runs. ``job speed`` sums
steps and duration per worker. Job speed does not read the aggregation, so it hangs off
the array directly and overlaps it; the other two wait.

Two things live here, and the split is the whole point:

  * ``post_batch_status`` INSPECTS. It reads a handful of small files and never submits
    anything, so a notebook can call it on every Run All and print what is ready.
  * ``ensure_post_batch`` SUBMITS. It is the only thing that starts work.

That separation is what keeps the notebook honest. Before it, the same call could be a
20ms file read or a 42GB scan depending on state you could not see, and a fresh batch
turned "open the notebook" into an unannounced hour of compute inside a Jupyter kernel.

Batch kinds
-----------
CURRENT
    A normally simulated batch: batch_info.json plus per-job shards under
    tmp/results/raw_results_job_*. All four steps apply.
COMBINED
    Synthesised by ``combine_batches`` from two or more parents. Its graph_statistics.csv
    is the exact concatenation of the parents' rollups, so aggregation is INHERITED and is
    never re-derived, not even with --force: re-deriving it would mean re-reading both
    parents to reproduce a file we already have exactly. Job speed does not apply either,
    because the linked shards still carry each parent's own job_id numbering. verify and
    the violin cache work normally.
LEGACY
    An older batch whose layout predates the per-job parquet shards (result_job_*.csv, or
    a fused raw_results.csv only), and usually with no batch_info.json. Reported as
    unsupported with the specific reasons, rather than crashing somewhere downstream.
    These already have graph_statistics.csv, so the property figures still work; only the
    post-batch jobs are unavailable.

Usage
-----
    uv run python -m moran_process.pipeline.post_batch --batch-dir <batch>            # status
    uv run python -m moran_process.pipeline.post_batch --batch-dir <batch> --submit
    uv run python -m moran_process.pipeline.post_batch --batch-dir <batch> --submit --force

From a notebook:

    from moran_process.pipeline.post_batch import post_batch_status, print_post_batch_status
    print_post_batch_status(post_batch_status(BATCH_DIR, r_values=R_VALUE_FILTER))
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from moran_process.analysis.analysis_utils.io import (
    PER_JOB_RESULT_STEM,
    fixation_steps_cache_path,
    resolve_results_source,
)

__all__ = [
    "CURRENT",
    "COMBINED",
    "LEGACY",
    "classify_batch",
    "post_batch_status",
    "print_post_batch_status",
    "ensure_post_batch",
]

CURRENT = "CURRENT"
COMBINED = "COMBINED"
LEGACY = "LEGACY"

# Step states. INHERITED is not a weaker DONE: it means the artefact is exact and was
# never computed here, so "rebuild it" is not a thing you can ask for.
DONE = "DONE"
MISSING = "MISSING"
INHERITED = "INHERITED"
NOT_APPLICABLE = "N/A"


def _read_batch_info(batch_path):
    """batch_info.json as a dict, or None if the batch has none."""
    info_path = batch_path / "batch_info.json"
    if not info_path.exists():
        return None
    with open(info_path) as f:
        return json.load(f)


def classify_batch(batch_dir):
    """Return (kind, reasons) for a batch directory.

    ``reasons`` is empty for CURRENT and COMBINED, and lists what disqualified a LEGACY
    batch so the message can say which prerequisite is missing rather than just "no".
    """
    batch_path = Path(batch_dir)
    info = _read_batch_info(batch_path)

    if info is not None and info.get("combined_from"):
        return COMBINED, []

    reasons = []
    if info is None:
        reasons.append("no batch_info.json (predates provenance capture)")

    shards_dir = batch_path / "tmp" / "results"
    has_shards = any(shards_dir.glob(f"{PER_JOB_RESULT_STEM}_*.parquet")) or any(
        shards_dir.glob(f"{PER_JOB_RESULT_STEM}_*.csv")
    )
    if not has_shards:
        reasons.append(
            f"no per-job shards under {shards_dir} "
            f"(expected {PER_JOB_RESULT_STEM}_*.parquet)"
        )

    if not (batch_path / "graph_props.csv").exists():
        reasons.append("no graph_props.csv (register_graphs never ran)")

    return (LEGACY, reasons) if reasons else (CURRENT, [])


def _resolve_r_values(batch_path, r_values):
    """Which r values the violin cache should cover.

    graph_statistics.csv is the authoritative record of what was simulated, so it wins.
    Before aggregation runs we fall back to batch_info's requested r values, which is what
    lets the status table say "cache pending for these r" on a batch that has not
    aggregated yet instead of saying nothing at all.
    """
    if r_values is not None:
        return sorted({float(r) for r in r_values})

    stats_path = batch_path / "graph_statistics.csv"
    if stats_path.exists():
        return sorted(pd.read_csv(stats_path, usecols=["r"])["r"].unique().tolist())

    info = _read_batch_info(batch_path)
    requested = ((info or {}).get("simulation") or {}).get("r_values") or []
    return sorted(float(r) for r in requested)


def post_batch_status(batch_dir, r_values=None, max_points_per_category=50_000):
    """Report which post-batch artefacts exist. Pure inspection: submits nothing.

    Reads only small files (a few CSVs' headers, two JSONs, a directory listing), so it is
    milliseconds regardless of batch size and safe to call on every notebook Run All.

    Args:
        batch_dir: the batch directory.
        r_values: which r values the violin cache must cover to count as ready. Defaults
            to every r in graph_statistics.csv. Narrow it to what you are actually
            plotting (e.g. R_VALUE_FILTER) so a batch is not reported as incomplete over
            an r you will never draw.
        max_points_per_category: the cap the caches must have been built with; part of
            the cache key.

    Returns:
        dict with keys: batch_dir, batch_name, kind, reasons, steps, violin_cache,
        r_values, ready. ``steps`` maps step name to state; ``violin_cache`` maps r to
        state; ``ready`` is True when nothing is MISSING.
    """
    batch_path = Path(batch_dir)
    kind, reasons = classify_batch(batch_path)

    status = {
        "batch_dir": str(batch_path),
        "batch_name": batch_path.name,
        "kind": kind,
        "reasons": reasons,
        "steps": {},
        "violin_cache": {},
        "r_values": [],
        "ready": False,
    }
    if kind == LEGACY:
        return status

    stats_exists = (batch_path / "graph_statistics.csv").exists()
    if kind == COMBINED:
        # Inherited exactly from the parents, so there is nothing to build and nothing to
        # force. If it is somehow absent, combine_batches itself did not finish.
        status["steps"]["aggregate"] = INHERITED if stats_exists else MISSING
    else:
        status["steps"]["aggregate"] = DONE if stats_exists else MISSING

    verification_path = batch_path / "report" / "verification.json"
    status["steps"]["verify"] = DONE if verification_path.exists() else MISSING
    if verification_path.exists():
        with open(verification_path) as f:
            status["verify_overall"] = json.load(f).get("overall")

    if kind == COMBINED:
        status["steps"]["job_speed"] = NOT_APPLICABLE
    else:
        status["steps"]["job_speed"] = (
            DONE if (batch_path / "job_speed.csv").exists() else MISSING
        )

    resolved_r = _resolve_r_values(batch_path, r_values)
    status["r_values"] = resolved_r
    for r in resolved_r:
        path = fixation_steps_cache_path(batch_path, r, max_points_per_category)
        complete = path.exists() and path.with_suffix(".json").exists()
        status["violin_cache"][r] = DONE if complete else MISSING
    status["steps"]["violin_cache"] = (
        DONE
        if resolved_r and all(v == DONE for v in status["violin_cache"].values())
        else MISSING
    )

    status["ready"] = MISSING not in status["steps"].values()
    return status


def print_post_batch_status(status):
    """Print a status table and, when something is missing, the command that fixes it."""
    print(f"Post-batch status: {status['batch_name']}  [{status['kind']}]")

    if status["kind"] == LEGACY:
        print("  This batch's layout predates the post-batch jobs:")
        for reason in status["reasons"]:
            print(f"    - {reason}")
        print(
            "  The post-batch jobs do not run on it. Figures reading graph_statistics.csv\n"
            "  still work; the violin / p-value / speed figures do not."
        )
        return

    for step in ("aggregate", "job_speed", "verify", "violin_cache"):
        state = status["steps"].get(step, MISSING)
        extra = ""
        if step == "verify" and status.get("verify_overall"):
            extra = f"  (overall: {status['verify_overall']})"
        if step == "violin_cache" and status["violin_cache"]:
            per_r = ", ".join(
                f"r={r}:{v}" for r, v in sorted(status["violin_cache"].items())
            )
            extra = f"  [{per_r}]"
        print(f"  {step:<14} {state:<10}{extra}")

    if status["ready"]:
        print("  -> ready: every figure input is a file read.")
        return

    print(
        "  -> not ready. Submit the missing jobs with:\n"
        "       from moran_process.pipeline.post_batch import ensure_post_batch\n"
        f"       ensure_post_batch({status['batch_dir']!r})\n"
        "     or from a shell:\n"
        "       uv run python -m moran_process.pipeline.post_batch "
        f"--batch-dir {status['batch_dir']} --submit"
    )


def ensure_post_batch(
    batch_dir,
    force=False,
    r_values=None,
    max_points_per_category=50_000,
    queue="short",
):
    """Submit whichever post-batch jobs this batch is missing. The only thing that submits.

    With ``force=False`` this fills gaps: each missing step is submitted, and if the
    aggregation is among them its consumers are chained onto it so they PEND rather than
    reading a half-written graph_statistics.csv.

    With ``force=True`` every applicable step is resubmitted and the consumers are chained
    onto the fresh aggregation. Aggregation is still never resubmitted for a COMBINED
    batch: its rollup is the parents' concatenated exactly, so there is no computation to
    repeat. If you really want to re-derive it from the linked shards, call
    ``submit_aggregation_job`` directly.

    To rerun a single step, call its ``submit_*_job`` helper or its CLI. This function is
    deliberately coarse, because the thing it exists to get right is the DAG.

    Args:
        force: resubmit every applicable step rather than only the missing ones.
        r_values: passed to post_batch_status to decide whether the violin cache counts as
            complete. The submitted job always builds every r in graph_statistics.csv.
        queue: LSF queue for all submitted jobs.

    Returns:
        dict of step name -> LSF job id (or None where bsub failed). Steps that were
        already satisfied are absent.
    """
    from .process_lab import (
        submit_aggregation_job,
        submit_job_speed_job,
        submit_verify_job,
        submit_violin_cache_job,
    )

    batch_path = Path(batch_dir)
    status = post_batch_status(
        batch_path, r_values=r_values, max_points_per_category=max_points_per_category
    )

    if status["kind"] == LEGACY:
        raise SystemExit(
            f"{batch_path.name} is a legacy-layout batch; the post-batch jobs do not "
            f"apply to it:\n  - " + "\n  - ".join(status["reasons"])
        )

    if resolve_results_source(batch_path) is None:
        raise SystemExit(
            f"No raw results resolvable under {batch_path / 'tmp' / 'results'}; there is "
            f"nothing for the post-batch jobs to read."
        )

    steps = status["steps"]
    submitted = {}

    # The array has long since left LSF's records by the time this runs standalone, so
    # every job here is submitted with no array dependency and starts immediately. The
    # only ordering that still matters is aggregate -> {verify, violin cache}.
    aggregate_job_id = None
    if steps["aggregate"] != INHERITED and (force or steps["aggregate"] == MISSING):
        aggregate_job_id = submit_aggregation_job(
            batch_dir=str(batch_path), batch_name=batch_path.name, queue=queue
        )
        submitted["aggregate"] = aggregate_job_id

    # If the aggregation is being rebuilt, its consumers' input is about to change, so a
    # verification or a cache built against the old graph_statistics.csv is stale even
    # though it exists. Rerun them too, chained onto the new aggregation.
    rebuilding_input = aggregate_job_id is not None

    if force or rebuilding_input or steps["verify"] == MISSING:
        submitted["verify"] = submit_verify_job(
            str(batch_path),
            batch_path.name,
            aggregate_job_id=aggregate_job_id,
            queue=queue,
        )

    if force or rebuilding_input or steps["violin_cache"] == MISSING:
        submitted["violin_cache"] = submit_violin_cache_job(
            str(batch_path),
            batch_path.name,
            aggregate_job_id=aggregate_job_id,
            queue=queue,
            max_points_per_category=max_points_per_category,
            # Without this the job would skip every cache that already exists, which is
            # exactly the set we are rerunning it to replace.
            force=force or rebuilding_input,
        )

    if steps["job_speed"] != NOT_APPLICABLE and (
        force or steps["job_speed"] == MISSING
    ):
        submitted["job_speed"] = submit_job_speed_job(
            str(batch_path), batch_path.name, queue=queue
        )

    if not submitted:
        print(f"{batch_path.name}: nothing to do, every post-batch step is present.")
    else:
        print(
            f"{batch_path.name}: submitted "
            + ", ".join(f"{k}={v}" for k, v in submitted.items())
        )
    return submitted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Report or submit the post-simulation jobs for a batch."
    )
    parser.add_argument("--batch-dir", required=True, help="Batch directory.")
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit the missing jobs (default: report status only).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="With --submit, resubmit every applicable step, not just the missing ones.",
    )
    parser.add_argument(
        "--r-values",
        nargs="+",
        type=float,
        default=None,
        help="Which r values the violin cache must cover (default: all in graph_statistics.csv).",
    )
    parser.add_argument(
        "--max-points-per-category",
        type=int,
        default=50_000,
        help="Violin subsample cap; part of the cache key (default: 50000).",
    )
    args = parser.parse_args()

    if args.submit:
        ensure_post_batch(
            args.batch_dir,
            force=args.force,
            r_values=args.r_values,
            max_points_per_category=args.max_points_per_category,
        )
    else:
        print_post_batch_status(
            post_batch_status(
                args.batch_dir,
                r_values=args.r_values,
                max_points_per_category=args.max_points_per_category,
            )
        )

"""Verify a finished batch: did every requested run happen, and did any job fail quietly?

A batch is thousands of independent LSF tasks writing thousands of files, so the failure
modes are quiet ones: a worker that died leaves a missing shard, a truncated write leaves
a short one, a graph that never registered leaves result rows with no properties, and none
of that raises anywhere. You would notice weeks later as a hole in a figure.

This job runs the checks that catch those, and writes both a machine-readable
``report/verification.json`` and a human-readable ``report/verification.txt``. Every check is
cheap by construction: it reads graph_props.csv, graph_statistics.csv, batch_info.json and
the parquet *footers* (row counts are metadata, not data), so it never scans the 7.2e9 raw
rows and finishes in seconds regardless of batch size.

Each check reports OK / WARN / FAIL. FAIL means the batch is incomplete or inconsistent and
the analysis will be wrong; WARN means something is worth a look but is often intentional
(a combined batch has no single n_repeats, for instance).

Runs the same way however it is launched --

  * automatically, as the job ProcessLab chains after the aggregation
    (see process_lab.submit_verify_job); or
  * by hand:
      uv run python -m moran_process.pipeline.batch_verify --batch-dir <batch>

Imports from the ``io``/``provenance`` submodules directly (not the analysis_utils package
root) so this never pulls in the plotting stack, exactly like aggregate_batch.
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from moran_process.analysis.analysis_utils.io import (
    PER_JOB_RESULT_STEM,
    resolve_results_source,
)
from moran_process.analysis.analysis_utils.provenance import _bi_get, load_batch_info

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

OK, WARN, FAIL = "OK", "WARN", "FAIL"


class _Checks:
    """Accumulator for check results, so each check is one call and the order is the
    order they were run in."""

    def __init__(self):
        self.results = []

    def add(self, status, name, message, **detail):
        self.results.append(
            {"status": status, "name": name, "message": message, **detail}
        )
        log.info("[%-4s] %s: %s", status, name, message)

    def worst(self):
        for level in (FAIL, WARN):
            if any(c["status"] == level for c in self.results):
                return level
        return OK


def _check_shards(checks, batch_path, info, count_rows=True):
    """Verify the raw shards: presence, contiguous numbering, and non-empty content.

    Job indices are checked against ``hpc.n_requested_jobs`` because LSF numbers an array
    1..N and a dead worker simply never writes its file -- there is no error to catch, only
    a gap. Row counts come from the parquet footer, which is metadata, so counting all
    7.2e9 rows costs one seek per file rather than a read.
    """
    results_dir = batch_path / "tmp" / "results"
    shards = sorted(
        results_dir.glob(f"{PER_JOB_RESULT_STEM}_*.parquet"),
        key=lambda f: int(f.stem.split("_")[-1]),
    )
    if not shards:
        checks.add(
            WARN,
            "raw_shards",
            f"no per-job shards under {results_dir} (aggregated-only batch?)",
            n_shards=0,
        )
        return

    indices = {int(f.stem.split("_")[-1]) for f in shards}
    expected = _bi_get(info, "hpc", "n_requested_jobs")
    if expected:
        missing = sorted(set(range(1, int(expected) + 1)) - indices)
        if missing:
            checks.add(
                FAIL,
                "raw_shards",
                f"{len(missing)} of {expected} job shard(s) missing, e.g. {missing[:10]}",
                n_shards=len(shards),
                n_missing=len(missing),
                missing_sample=missing[:50],
            )
        else:
            checks.add(
                OK,
                "raw_shards",
                f"all {expected} job shard(s) present",
                n_shards=len(shards),
            )
    else:
        checks.add(
            WARN,
            "raw_shards",
            f"{len(shards)} shard(s) present; batch_info has no n_requested_jobs to check against",
            n_shards=len(shards),
        )

    if not count_rows:
        return

    import pyarrow.parquet as pq

    empty, total_rows = [], 0
    for f in shards:
        try:
            n = pq.ParquetFile(str(f)).metadata.num_rows
        except Exception as e:
            # A broken footer means the writer died mid-file; that shard's runs are lost
            # and every downstream scan will raise, so it is a FAIL, not a WARN.
            checks.add(FAIL, "raw_shards_readable", f"{f.name} unreadable: {e}")
            continue
        total_rows += n
        if n == 0:
            empty.append(f.name)
    if empty:
        checks.add(
            FAIL,
            "raw_shard_rows",
            f"{len(empty)} shard(s) hold zero rows, e.g. {empty[:5]}",
            total_rows=total_rows,
            n_empty=len(empty),
        )
    else:
        checks.add(
            OK, "raw_shard_rows", f"{total_rows:,} raw runs total", total_rows=total_rows
        )


def _check_props(checks, df_props):
    """graph_props must be one row per distinct graph: it is the right-hand side of every
    join in the analysis, so a duplicated wl_hash silently multiplies result rows."""
    dup = int(df_props["wl_hash"].duplicated().sum())
    if dup:
        checks.add(FAIL, "graph_props", f"{dup} duplicate wl_hash row(s)", n_duplicate=dup)
    else:
        checks.add(
            OK,
            "graph_props",
            f"{df_props['wl_hash'].nunique():,} distinct graph(s), no duplicates",
            n_graphs=int(df_props["wl_hash"].nunique()),
        )


def _check_stats(checks, df_stats, df_props, info):
    """The per-(graph, r) rollup: key uniqueness, join coverage, NaN, and r coverage."""
    dup = int(df_stats.duplicated(subset=["wl_hash", "r"]).sum())
    if dup:
        checks.add(
            FAIL,
            "stats_keys",
            f"{dup} duplicate (wl_hash, r) row(s)",
            n_duplicate=dup,
        )
    else:
        checks.add(
            OK,
            "stats_keys",
            f"{len(df_stats):,} unique (graph, r) row(s)",
            n_rows=len(df_stats),
        )

    # A result row with no graph_props match keeps n_nodes NaN, which later crashes the
    # analytic complete-graph baseline. Catch it here rather than 200 lines downstream.
    orphans = set(df_stats["wl_hash"]) - set(df_props["wl_hash"])
    if orphans:
        checks.add(
            FAIL,
            "stats_join",
            f"{len(orphans)} simulated graph(s) absent from graph_props (did register_graphs run?)",
            n_orphans=len(orphans),
            sample=sorted(orphans)[:5],
        )
    else:
        checks.add(OK, "stats_join", "every simulated graph has properties")

    nan_cols = {
        c: int(df_stats[c].isna().sum())
        for c in ("prob_fixation", "mean_steps", "n_grouped", "n_nodes", "category")
        if c in df_stats.columns and df_stats[c].isna().any()
    }
    if nan_cols:
        # mean_steps is legitimately NaN when a graph never fixated at that r (there is no
        # steps-to-fixation to average), so it warns; the others indicate a real hole.
        status = WARN if set(nan_cols) <= {"mean_steps"} else FAIL
        checks.add(status, "stats_nan", f"NaN present: {nan_cols}", nan_counts=nan_cols)
    else:
        checks.add(OK, "stats_nan", "no NaN in the key statistic columns")

    r_present = sorted(df_stats["r"].unique().tolist())
    r_expected = _bi_get(info, "simulation", "r_values") or []
    missing_r = [r for r in r_expected if r not in r_present]
    if missing_r:
        checks.add(
            FAIL,
            "r_coverage",
            f"r values submitted but absent from results: {missing_r}",
            r_present=r_present,
            r_expected=list(r_expected),
        )
    else:
        checks.add(OK, "r_coverage", f"r values present: {r_present}", r_present=r_present)

    # Every graph should appear at every r. A graph short of the full set means some of
    # its tasks never completed, which skews any per-r comparison that includes it.
    per_graph = df_stats.groupby("wl_hash")["r"].nunique()
    short = per_graph[per_graph < len(r_present)]
    if len(short):
        checks.add(
            FAIL,
            "graph_r_completeness",
            f"{len(short):,} graph(s) missing at least one r value",
            n_incomplete=int(len(short)),
            sample=short.index[:5].tolist(),
        )
    else:
        checks.add(
            OK, "graph_r_completeness", f"all graphs simulated at all {len(r_present)} r value(s)"
        )


def _check_repeats(checks, df_stats, info):
    """n_grouped is the number of runs behind each (graph, r). Short and over are very
    different problems, so they are graded differently.

    SHORT (n_grouped < n_repeats) is always a failure: runs were submitted and never came
    back, so that cell's rho is built on less data than every other cell.

    OVER, by an exact multiple, is expected and benign. The zoo is a list of graphs while
    the rollup groups by wl_hash, so two isomorphic zoo entries are simulated as two tasks
    and then pool into one row. That cell simply has k x the precision. It is worth
    surfacing (unequal n across graphs surprises people reading a figure) but it is not an
    error, and "fixing" it would mean throwing away good simulations.

    OVER by a non-multiple is a failure, because no amount of pooling produces it.
    """
    if "n_grouped" not in df_stats.columns:
        return
    n_repeats = _bi_get(info, "simulation", "n_repeats")
    lo, hi = int(df_stats["n_grouped"].min()), int(df_stats["n_grouped"].max())
    if n_repeats is None:
        checks.add(
            WARN,
            "n_repeats",
            f"runs per (graph, r) range {lo:,}..{hi:,}; batch_info has no n_repeats "
            f"(expected for a combined batch whose parents disagreed)",
            min=lo,
            max=hi,
        )
        return
    n_repeats = int(n_repeats)
    n = df_stats["n_grouped"]
    short = df_stats[n < n_repeats]
    ragged = df_stats[(n > n_repeats) & (n % n_repeats != 0)]
    pooled = df_stats[(n > n_repeats) & (n % n_repeats == 0)]

    if len(short) or len(ragged):
        parts = []
        if len(short):
            parts.append(f"{len(short):,} cell(s) SHORT of {n_repeats:,} runs (min {lo:,})")
        if len(ragged):
            parts.append(
                f"{len(ragged):,} cell(s) over {n_repeats:,} by a non-multiple (max {hi:,})"
            )
        checks.add(
            FAIL,
            "n_repeats",
            "; ".join(parts),
            expected=n_repeats,
            n_short=int(len(short)),
            n_ragged=int(len(ragged)),
            min=lo,
            max=hi,
        )
    elif len(pooled):
        graphs = pooled["wl_hash"].nunique()
        checks.add(
            WARN,
            "n_repeats",
            f"{graphs} graph(s) pooled {sorted(set(pooled['n_grouped'] // n_repeats))}x "
            f"{n_repeats:,} runs (isomorphic zoo entries merged by wl_hash); all others exact",
            expected=n_repeats,
            n_pooled_cells=int(len(pooled)),
            n_pooled_graphs=int(graphs),
            max=hi,
        )
    else:
        checks.add(
            OK,
            "n_repeats",
            f"every cell has exactly {n_repeats:,} runs",
            expected=n_repeats,
        )


def _category_table(df_stats):
    """Graphs per category per r: the table you actually look at to see what is in a
    batch, and the fastest way to spot a category that silently lost most of its graphs."""
    if "category" not in df_stats.columns:
        return {}
    tbl = (
        df_stats.groupby(["category", "r"])["wl_hash"]
        .nunique()
        .unstack(fill_value=0)
        .sort_index()
    )
    return {str(cat): row.to_dict() for cat, row in tbl.iterrows()}


def _render_text(report):
    """Format the report as fixed-width text. Kept separate from the checks so the JSON
    stays the single source of truth and the text is purely a view of it."""
    lines = [
        "=" * 78,
        f"BATCH QC REPORT  --  {report['batch_name']}",
        f"generated: {report['generated_at']}",
        f"batch dir: {report['batch_dir']}",
        "=" * 78,
        "",
        f"OVERALL: {report['overall']}",
        "",
        "CHECKS",
        "-" * 78,
    ]
    for c in report["checks"]:
        lines.append(f"[{c['status']:<4}] {c['name']:<22} {c['message']}")

    if report["categories"]:
        lines += ["", "GRAPHS PER CATEGORY PER r", "-" * 78]
        r_keys = sorted({r for row in report["categories"].values() for r in row})
        lines.append("  " + "category".ljust(44) + "".join(f"{str(r):>9}" for r in r_keys))
        for cat, row in sorted(report["categories"].items()):
            lines.append(
                "  " + cat[:43].ljust(44) + "".join(f"{row.get(r, 0):>9,}" for r in r_keys)
            )
    lines.append("")
    return "\n".join(lines)


def run_verification(batch_dir, count_rows=True):
    """Answer the two verification questions and write report/verification.{json,txt}.

    The questions are: did every requested run actually happen, and did any job fail
    without saying so. Everything here exists to answer one of those.

    Returns the verification dict. Does not raise on a failing check: a verdict that
    refuses to be written is a verdict you cannot read, and the caller (an LSF job) has
    nothing to do with a non-zero exit anyway. The overall status is in the file, in the
    log, and in what post_batch_status reports.
    """
    batch_path = Path(batch_dir)
    info = load_batch_info(batch_path)
    checks = _Checks()

    props_path = batch_path / "graph_props.csv"
    stats_path = batch_path / "graph_statistics.csv"
    for p in (props_path, stats_path):
        if not p.exists():
            raise SystemExit(
                f"{p} missing; run moran_process.pipeline.aggregate_batch on this batch first."
            )

    df_props = pd.read_csv(props_path)
    df_stats = pd.read_csv(stats_path)

    if resolve_results_source(batch_path) is None:
        checks.add(WARN, "results_source", "no raw results resolvable for this batch")
    _check_shards(checks, batch_path, info, count_rows=count_rows)
    _check_props(checks, df_props)
    _check_stats(checks, df_stats, df_props, info)
    _check_repeats(checks, df_stats, info)

    report = {
        "batch_name": _bi_get(info, "name", default=batch_path.name),
        "batch_dir": str(batch_path.resolve()),
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "overall": checks.worst(),
        "checks": checks.results,
        "categories": _category_table(df_stats),
    }

    report_dir = batch_path / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / "verification.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    text = _render_text(report)
    with open(report_dir / "verification.txt", "w") as f:
        f.write(text)

    print("\n" + text)
    log.info("Verification written to %s (overall: %s)", report_dir, report["overall"])
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify a finished (or combined) simulation batch ran completely."
    )
    parser.add_argument(
        "--batch-dir", required=True, help="Batch directory to check."
    )
    parser.add_argument(
        "--skip-row-counts",
        action="store_true",
        help="Do not read the parquet footers (skips the empty/truncated shard check).",
    )
    args = parser.parse_args()
    run_verification(args.batch_dir, count_rows=not args.skip_row_counts)

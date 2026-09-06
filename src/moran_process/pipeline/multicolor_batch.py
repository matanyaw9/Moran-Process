"""Submit, run and read a multi-color (neutral lineage competition) batch.

Why this is not `worker_lsf` with a flag
----------------------------------------
The two-color pipeline is built around one observable, "did the mutant fix",
and three facts follow from it: the manifest is a (graph x r) cross product,
the result schema is `fixation/steps/censored`, and the rollup keys on
`(wl_hash, r)`. None of the three survives here. Selection is neutral by
construction, so `r` is not a dimension; the observable is *which node's*
lineage won, so a row needs a `winner`; and the natural summary is a per-node
vector, not a per-r scalar. `MultiColorMoranProcess` also does not implement
`initialize_random_mutant`, which is the first thing `worker_lsf` calls.

The other half of the reason is size. The two-color pipeline needs zoo shards,
streamed row-groups, glob-scanned aggregation and a u32 row-count workaround
because it writes 7.2e9 rows. A multi-color batch at 10K trials over 30 graphs
writes 300K rows, about 5 MB. So there is no aggregate job and no builder/reader
split: `load_multicolor` is a plain Parquet read, and `win_table` is a groupby.

Job splitting
-------------
One array index per (graph, trial chunk), *not* an even split of total trials.
Consensus time here is a coalescing random walk, so it is set by how long
lineages take to meet rather than by node count: measured, `line_n50` is 20x
slower per trial than `complete_n50` at identical N (O(N^3) against O(N^2)).
Balancing by trial count, the way `_create_task_list` does, would hand one
worker ten cheap graphs and another ten expensive ones. Giving each graph its
own index lets LSF schedule around the imbalance for free.

Engines
-------
`--engine {cpp,python}`, default cpp, exactly as in `worker_lsf`. The C++ core
(`_cpp/moran_core.cpp`, class `MultiColorCore`) is ~1500x faster than the
reference and statistically equivalent to it, not bit-exact. Two things buy the
speedup: the loop leaves Python, and absorption is detected in O(1) from a live
count of surviving lineages rather than by the reference's O(n)
`np.all(state == state[0])` scan on every step.

CLI (the worker; `submit_multicolor` is called from a notebook or a shell):

    python -m moran_process.pipeline.multicolor_batch \\
        --batch-dir <batch>/tmp --zoo-shard-dir <batch>/tmp/zoo_shards \\
        --manifest-path <batch>/tmp/task_manifest.csv \\
        [--engine cpp|python] [--job-index N]
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

log = logging.getLogger(__name__)

#: Result files are found by glob, so the stem is shared by writer and reader.
PER_JOB_RESULT_STEM = "multicolor_job"

#: Fixed schema for every result file; column order must match the RecordBatch below.
RESULT_SCHEMA = pa.schema(
    [
        pa.field("wl_hash", pa.string()),
        pa.field("graph_name", pa.string()),
        pa.field("trial", pa.int64()),
        # -1 when the run was cut short by max_steps: see MultiColorMoranProcess.run.
        pa.field("winner", pa.int32()),
        pa.field("fixed", pa.bool_()),
        pa.field("steps", pa.int64()),
        pa.field("duration", pa.float64()),
    ]
)


def _fmt_hours(h):
    """Format an hour count in whatever unit makes it readable.

    The C++ engine put a whole 10K-trial zoo under a minute, at which point a
    fixed "%.2f h" prints 0.00 for every row and the preview stops answering the
    only question it is asked.
    """
    if h < 1 / 60:
        return f"{h * 3600:.1f} s"
    if h < 1:
        return f"{h * 60:.1f} min"
    return f"{h:.2f} h"


def _resolve_engine(engine):
    """Return the multi-color simulation class for the requested engine.

    Mirrors `worker_lsf._resolve_engine`: both classes share one interface, so
    everything downstream of this call is identical. 'cpp' is the compiled core
    (statistically equivalent, ~1500x faster); 'python' is the reference.
    """
    if engine == "cpp":
        from moran_process.simulations.cpp_moran_wrapper import (
            CppMultiColorMoranProcess,
        )

        return CppMultiColorMoranProcess
    if engine == "python":
        from moran_process.simulations.multi_color_moran_process import (
            MultiColorMoranProcess,
        )

        return MultiColorMoranProcess
    raise ValueError(f"Unknown engine '{engine}' (expected 'cpp' or 'python').")


# --------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------
def preview(zoo, n_trials, pilot=3, max_steps=1_000_000, seed=0, engine="cpp"):
    """Time a few trials per graph and extrapolate the cost of the full batch.

    Runs `pilot` real trials on every graph, so the estimate reflects this zoo
    rather than a rule of thumb. Consensus time varies by more than an order of
    magnitude between topologies of the same N, which is exactly what a preview
    has to expose before anything is submitted.

    `pilot` is per-graph and must be large enough to time honestly: the C++
    engine runs a trial in tens of microseconds, so 3 repeats measures clock
    noise. Raise it (100+) when previewing with engine='cpp'.

    Returns a DataFrame, one row per graph, and prints it.
    """
    Engine = _resolve_engine(engine)

    rows = []
    for g in zoo:
        sim = Engine(g, max_steps=max_steps, seed=seed)
        out = sim.run_repeats(pilot)
        sec = float(out["duration"].mean())
        rows.append(
            {
                "graph_name": g.name,
                "n_nodes": g.number_of_nodes(),
                "mean_steps": float(out["steps"].mean()),
                "sec_per_trial": sec,
                "hours_for_batch": sec * n_trials / 3600,
                "pilot_censored": int((~out["fixed"]).sum()),
            }
        )

    df = pd.DataFrame(rows).sort_values("hours_for_batch", ascending=False)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(
        f"\n{len(zoo)} graphs x {n_trials:,} trials = {len(zoo) * n_trials:,} runs"
        f"  [{engine} engine]"
        f"\nslowest graph: {_fmt_hours(df['hours_for_batch'].iloc[0])} "
        "(this sets wall clock)"
        f"\ntotal:         {_fmt_hours(df['hours_for_batch'].sum())} of core time"
    )
    if df["pilot_censored"].sum():
        print(
            f"\nWARNING: {int(df['pilot_censored'].sum())} pilot runs hit "
            f"max_steps={max_steps:,}. Those trials have winner=-1 and are "
            "dropped from win counts. Raise max_steps."
        )
    return df


def build_manifest(zoo, n_trials, batch_seed=None, trials_per_job=None):
    """One row per LSF array index: (graph, trial chunk).

    `trials_per_job` splits a single graph's trials across several indices, for
    a topology slow enough to overrun the queue's wall-clock limit on its own.
    Left as None each graph is one index. Chunks of the same graph carry
    disjoint `trial_start` offsets so the `trial` column stays unique per graph.

    Seeds come from one root generator over `batch_seed`, so the whole batch
    replays from that single integer in batch_info.json.
    """
    rng = np.random.default_rng(batch_seed)
    rows = []
    for graph_idx, g in enumerate(zoo):
        chunk = trials_per_job or n_trials
        for start in range(0, n_trials, chunk):
            rows.append(
                {
                    "worker_id": len(rows) + 1,  # 1-based, matches LSB_JOBINDEX
                    "graph_idx": graph_idx,
                    "graph_name": g.name,
                    "wl_hash": g.wl_hash,
                    "n_nodes": g.number_of_nodes(),
                    "trial_start": start,
                    "n_trials": min(chunk, n_trials - start),
                    "seed": int(rng.integers(0, 2**31)),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Submission
# --------------------------------------------------------------------------
def submit_multicolor(
    zoo,
    batch_dir,
    n_trials,
    batch_name=None,
    batch_seed=None,
    trials_per_job=None,
    max_steps=1_000_000,
    queue="short",
    memory="2GB",
    engine="cpp",
    description="",
    notes="",
):
    """Write the shards and manifest, chain register_graphs, bsub the array.

    `zoo` is a list of PopulationGraph. It is converted to GraphCore shards here
    (one graph per index), so workers never load NetworkX.

    Returns {'register': id, 'array': id}; either may be None if bsub failed,
    which callers that chain on them should treat as "runs immediately".
    """
    from moran_process.analysis.analysis_utils.provenance import create_batch_info
    from moran_process.pipeline.process_lab import (
        ProcessLab,
        _parse_lsf_job_id,
        _parse_memory_mb,
        register_graphs_job,
    )

    batch_dir = Path(batch_dir)
    batch_name = batch_name or batch_dir.name
    tmp_dir = batch_dir / "tmp"
    logs_dir = batch_dir / "logs"
    for d in (tmp_dir / "results", logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    # The zoo pickle is what register_graphs reads to write graph_props.csv,
    # which carries the structural properties every result joins to on wl_hash.
    zoo_path = tmp_dir / "graph_zoo.joblib"
    joblib.dump(list(zoo), zoo_path)
    log.info("Zoo written: %s (%d graphs)", zoo_path, len(zoo))

    manifest = build_manifest(zoo, n_trials, batch_seed, trials_per_job)
    shards_dir = tmp_dir / "zoo_shards"
    # Reused verbatim from the two-color path: same shard naming, same
    # local_graph_idx column, so a shard is interchangeable between the workers.
    manifest = ProcessLab._write_zoo_shards(manifest, list(zoo), shards_dir)
    manifest_path = tmp_dir / "task_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    n_jobs = len(manifest)
    log.info("Manifest: %d array indices -> %s", n_jobs, manifest_path)

    register_job_id = register_graphs_job(
        str(zoo_path), batch_name, str(batch_dir), queue=queue
    )

    memory_mb = _parse_memory_mb(memory)
    cmd = [
        "bsub",
        "-q", queue,
        "-J", f"mc_{batch_name}[1-{n_jobs}]",
        "-o", str(logs_dir / "mc_%J_%I.out"),
        "-e", str(logs_dir / "mc_%J_%I.err"),
        "-R", f"rusage[mem={memory_mb}]",
        "-env", "OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, PYTHONPATH=src",
        sys.executable, "-u", "-m", "moran_process.pipeline.multicolor_batch",
        "--batch-dir", str(tmp_dir),
        "--zoo-shard-dir", str(shards_dir),
        "--manifest-path", str(manifest_path),
        "--max-steps", str(max_steps),
        "--engine", engine,
    ]
    bsub_command = " ".join(cmd)
    log.info("Submitting: %s", bsub_command)
    result = subprocess.run(cmd, capture_output=True, text=True)
    array_job_id = _parse_lsf_job_id(result.stdout)
    if result.returncode == 0:
        log.info("Submitted. LSF job id: %s", array_job_id or "unknown")
    else:
        log.error(
            "bsub failed with return code %d: %s",
            result.returncode,
            (result.stderr or "").strip(),
        )

    create_batch_info(
        batch_dir=str(batch_dir),
        name=batch_name,
        description=description,
        notes=notes,
        # r is not a dimension of a neutral batch; recorded as empty rather than
        # faked as [1.0], so a reader cannot mistake this for a selection sweep.
        r_values=[],
        n_repeats=n_trials,
        total_simulations=len(zoo) * n_trials,
        batch_seed=batch_seed,
        # Tagged with the kind of batch as well as the core, because `engine`
        # is the only provenance field a reader sees on a figure subtitle and
        # "cpp" alone would not say this was a neutral multi-color run.
        engine=f"multicolor-{engine}",
        max_steps=max_steps,
        n_graphs=len(zoo),
        graph_types=sorted({g.category for g in zoo}),
        node_sizes=sorted({g.number_of_nodes() for g in zoo}),
        zoo_path=str(zoo_path),
        n_requested_jobs=n_jobs,
        queue=queue,
        memory_mb=memory_mb,
        job_array_name=f"mc_{batch_name}",
        lsf_job_id=array_job_id,
        bsub_command=bsub_command,
    )
    return {"register": register_job_id, "array": array_job_id}


# --------------------------------------------------------------------------
# Worker
# --------------------------------------------------------------------------
def run_worker_slice(
    batch_dir,
    zoo_shard_dir,
    manifest_path,
    worker_index,
    max_steps=1_000_000,
    engine="cpp",
):
    """Run this array index's (graph, trial chunk) rows and write one Parquet file."""
    Engine = _resolve_engine(engine)

    shard_path = os.path.join(zoo_shard_dir, f"zoo_worker_{worker_index}.pkl")
    shard = joblib.load(shard_path)
    manifest = pd.read_csv(manifest_path)
    my_tasks = manifest[manifest["worker_id"] == worker_index]
    if my_tasks.empty:
        log.warning("[Worker %s] no rows in manifest. Exiting.", worker_index)
        return

    save_path = os.path.join(
        batch_dir, "results", f"{PER_JOB_RESULT_STEM}_{worker_index}.parquet"
    )
    with pq.ParquetWriter(save_path, RESULT_SCHEMA) as writer:
        for row in my_tasks.itertuples():
            graph_core = shard[row.local_graph_idx]
            log.info(
                "[Worker %s] %s (N=%d) trials %d..%d seed=%s [%s]",
                worker_index,
                graph_core.name,
                graph_core.n_nodes,
                row.trial_start,
                row.trial_start + row.n_trials - 1,
                row.seed,
                engine,
            )
            sim = Engine(graph_core, max_steps=max_steps, seed=int(row.seed))
            out = sim.run_repeats(int(row.n_trials))

            n_censored = int((~out["fixed"]).sum())
            if n_censored:
                log.warning(
                    "%s: %d/%d trials hit max_steps=%d. Those have winner=-1 and "
                    "are excluded from win counts.",
                    graph_core.name,
                    n_censored,
                    row.n_trials,
                    max_steps,
                )

            trials = np.arange(
                row.trial_start, row.trial_start + row.n_trials, dtype=np.int64
            )
            writer.write_batch(
                pa.RecordBatch.from_arrays(
                    [
                        pa.array([graph_core.wl_hash] * len(trials)),
                        pa.array([graph_core.name] * len(trials)),
                        pa.array(trials),
                        pa.array(out["winner"].astype(np.int32)),
                        pa.array(out["fixed"]),
                        pa.array(out["steps"]),
                        pa.array(out["duration"]),
                    ],
                    schema=RESULT_SCHEMA,
                )
            )
    log.info("[Worker %s] wrote %s", worker_index, save_path)


# --------------------------------------------------------------------------
# Readers
# --------------------------------------------------------------------------
def load_multicolor(batch_dir):
    """Read every result shard of a batch into one per-trial DataFrame.

    Accepts a single batch directory or a list of them; a `batch` column records
    each row's source, so several batches of the same zoo stitch at read time
    the way `load_graph_statistics` does.

    Columns: batch, wl_hash, graph_name, trial, winner, fixed, steps, duration.
    """
    # No HASH_DTYPES dance here: that guards CSV type inference, and Parquet
    # carries wl_hash as a real string type.
    dirs = [batch_dir] if isinstance(batch_dir, (str, Path)) else list(batch_dir)
    frames = []
    for d in dirs:
        d = Path(d)
        files = sorted((d / "tmp" / "results").glob(f"{PER_JOB_RESULT_STEM}_*.parquet"))
        if not files:
            raise FileNotFoundError(
                f"No multicolor results in {d / 'tmp' / 'results'}. The array has "
                "not run, or this is not a multicolor batch."
            )
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        df.insert(0, "batch", d.name)
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    n_censored = int((~out["fixed"]).sum())
    print(
        f"Loaded {len(out):,} trials over {out['wl_hash'].nunique()} graphs "
        f"from {len(dirs)} batch(es)."
    )
    if n_censored:
        print(
            f"WARNING: {n_censored:,} trials ({n_censored / len(out):.2%}) hit "
            "max_steps and carry winner=-1. win_table drops them and reports "
            "the reduced n_trials."
        )
    return out


def win_table(raw, wl_hash=None, n_nodes=None):
    """Per-node win counts, plus steps and time, for one graph.

    `raw` is the frame from `load_multicolor`. Pass `wl_hash` to select a graph,
    or leave it None when the frame already holds exactly one.

    Censored trials (winner == -1) are dropped, and `n_trials` reports what
    remains: a run that never reached consensus has no winner to credit, and
    counting it against node 0 would invent the very positional bias this
    measures. Nodes that never won are present with wins=0, so the returned
    `win_frac` is always safe to hand to `draw_colored_graph`.

    mean_steps / mean_duration are conditional on that node winning, which is
    the question "does a takeover from here take longer?". The graph-level
    figures are in the frame itself.
    """
    df = raw if wl_hash is None else raw[raw["wl_hash"] == wl_hash]
    if df.empty:
        raise ValueError(f"No rows for wl_hash={wl_hash!r}")
    if df["wl_hash"].nunique() > 1:
        raise ValueError(
            f"{df['wl_hash'].nunique()} graphs in the frame; pass wl_hash to pick one."
        )

    won = df[df["fixed"]]
    if n_nodes is None:
        # Nodes that never win leave no row, so the max winner id understates N.
        # Fall back to it only when the caller cannot supply the true N.
        n_nodes = int(won["winner"].max()) + 1

    grouped = won.groupby("winner").agg(
        wins=("winner", "size"),
        mean_steps=("steps", "mean"),
        mean_duration=("duration", "mean"),
    )
    table = grouped.reindex(range(n_nodes))
    table["wins"] = table["wins"].fillna(0).astype(int)
    table.index.name = "node"
    table.insert(1, "win_frac", table["wins"] / len(won))
    table.attrs["n_trials"] = len(won)
    table.attrs["n_censored"] = len(df) - len(won)
    table.attrs["graph_name"] = df["graph_name"].iloc[0]
    return table.reset_index()


def _main():
    parser = argparse.ArgumentParser(description="Multi-color batch worker.")
    parser.add_argument("--batch-dir", required=True)
    parser.add_argument("--zoo-shard-dir", required=True)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--max-steps", type=int, default=1_000_000)
    parser.add_argument("--engine", choices=("cpp", "python"), default="cpp")
    parser.add_argument(
        "--job-index",
        type=int,
        default=None,
        help="Override LSB_JOBINDEX, for local debugging of one slice.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    worker_index = args.job_index or int(os.environ.get("LSB_JOBINDEX", 1))
    run_worker_slice(
        batch_dir=args.batch_dir,
        zoo_shard_dir=args.zoo_shard_dir,
        manifest_path=args.manifest_path,
        worker_index=worker_index,
        max_steps=args.max_steps,
        engine=args.engine,
    )


if __name__ == "__main__":
    _main()

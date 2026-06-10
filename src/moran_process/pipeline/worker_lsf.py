import argparse
import logging
import os
import resource
import sys
import numpy as np
import pandas as pd
import joblib
import pyarrow as pa
import pyarrow.parquet as pq

log = logging.getLogger(__name__)


def _resolve_engine(engine: str):
    """Return the simulation class for the requested engine.

    Both classes share the same interface (initialize_random_mutant/run), so the
    worker loop is identical regardless of choice. 'cpp' is the fast C++ core
    (statistically equivalent); 'python' is the pure-Python reference.
    """
    if engine == "cpp":
        from moran_process.simulations.cpp_moran_wrapper import CppMoranProcess
        return CppMoranProcess
    if engine == "python":
        from moran_process.simulations.moran_process import MoranProcess
        return MoranProcess
    raise ValueError(f"Unknown engine '{engine}' (expected 'cpp' or 'python').")

# Fixed schema for all result Parquet files; column order must match RecordBatch construction below.
_RESULT_SCHEMA = pa.schema([
    pa.field("task_id", pa.int64()),
    pa.field("job_id", pa.int32()),
    pa.field("wl_hash", pa.string()),
    pa.field("graph_name", pa.string()),
    pa.field("r", pa.float64()),
    pa.field("fixation", pa.bool_()),
    pa.field("steps", pa.int64()),
    pa.field("duration", pa.float64()),
])


def load_data(zoo_shard_dir, task_manifest_path, worker_index):
    """Load this worker's GraphCore shard and the shared task manifest.

    The shard is a list[GraphCore] containing only the graphs this worker
    references, already in compact CSR form. No NetworkX objects are loaded.
    """
    shard_path = os.path.join(zoo_shard_dir, f"zoo_worker_{worker_index}.pkl")
    if not os.path.exists(shard_path):
        raise FileNotFoundError(f"Shard not found: {shard_path}")
    if not os.path.exists(task_manifest_path):
        raise FileNotFoundError(f"Manifest not found: {task_manifest_path}")

    with open(shard_path, "rb") as f:
        graph_zoo = joblib.load(f)
    log.info("[Worker %s] Loaded %d graphs from shard.", worker_index, len(graph_zoo))

    manifest_df = pd.read_csv(task_manifest_path)
    log.info("[Worker %s] Loaded manifest (%d total tasks across all workers).",
             worker_index, len(manifest_df))

    return graph_zoo, manifest_df


def _rss_mb() -> int:
    """Current process peak RSS in MB (Linux: ru_maxrss is in KB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024


def run_worker_slice(batch_dir, zoo_shard_dir, manifest_path, worker_index, engine="cpp"):
    """Run simulations for all tasks assigned to this LSF job index.

    1. Load this worker's GraphCore shard and filter manifest to our rows.
    2. Run simulations for each (graph, r) task using the selected engine.
    3. Stream results to a per-job Parquet file (one row-group per task).
    """
    MoranProcess = _resolve_engine(engine)
    log.info("--- Worker %s started | engine=%s | RSS=%d MB ---",
             worker_index, engine, _rss_mb())

    # 1. Load data
    graph_zoo, manifest_df = load_data(zoo_shard_dir, manifest_path, worker_index)

    # 2. Filter to tasks assigned to this worker (worker_id is 1-based, matches LSB_JOBINDEX)
    my_tasks = manifest_df[manifest_df['worker_id'] == worker_index]
    log.info("\n[Worker %s] assigned tasks:\n%s", worker_index, my_tasks.to_string())

    if my_tasks.empty:
        log.warning("[Worker %s] No tasks found for this worker_id. Exiting.", worker_index)
        return

    n_tasks = len(my_tasks)
    total_sims = int(my_tasks['n_repeats'].sum())
    log.info("\n[Worker %s] %d tasks / %d simulations.\n", worker_index, n_tasks, total_sims)

    # 3. Stream results: open one Parquet file, write one row-group per (graph, r) task.
    #    Pre-allocating fixed NumPy arrays per task avoids per-rep dict allocation and
    #    the large pd.DataFrame(buffer) copy at the end.
    os.makedirs(os.path.join(batch_dir, "results"), exist_ok=True)
    # Name must match analysis_utils.PER_JOB_RESULT_STEM (kept as a literal here so
    # the worker doesn't import the heavy analysis module).
    save_path = os.path.join(batch_dir, "results", f"raw_results_job_{worker_index}.parquet")
    total_written = 0

    with pq.ParquetWriter(save_path, _RESULT_SCHEMA) as writer:
        for task_num, row in enumerate(my_tasks.itertuples(), start=1):
            try:
                graph_core = graph_zoo[row.local_graph_idx]
                r_val = row.r_value
                n_repeats = row.n_repeats

                log.info("[Worker %s] Task %d/%d | graph=%s r=%s reps=%s | RSS=%d MB",
                         worker_index, task_num, n_tasks, graph_core.name, r_val,
                         n_repeats, _rss_mb())

                # Seed from manifest: int → reproducible task, NaN → OS entropy.
                task_seed = None if pd.isna(row.seed) else int(row.seed)
                sim = MoranProcess(graph_core=graph_core, selection_coefficient=r_val,
                                   seed=task_seed)

                # Run all repeats inside the engine: one boundary crossing per
                # task, returning ready-made column arrays for the row-group.
                out = sim.run_repeats(n_repeats)
                fixations = out["fixation"]
                steps_arr = out["steps"]
                durations = out["duration"]

                # Flush this task's results as one Parquet row-group
                batch = pa.RecordBatch.from_arrays(
                    [
                        pa.array(np.full(n_repeats, row.task_id, dtype=np.int64)),
                        pa.array(np.full(n_repeats, worker_index, dtype=np.int32)),
                        pa.array([graph_core.wl_hash] * n_repeats),
                        pa.array([graph_core.name] * n_repeats),
                        pa.array(np.full(n_repeats, r_val, dtype=np.float64)),
                        pa.array(fixations),
                        pa.array(steps_arr),
                        pa.array(durations),
                    ],
                    schema=_RESULT_SCHEMA,
                )
                writer.write_batch(batch)
                total_written += n_repeats

                # Completion line: lets you see at a glance which tasks have
                # finished (the start line above only marks what is in-flight).
                n_fixed = int(np.count_nonzero(fixations))
                log.info("[Worker %s] Task %d/%d done | %s r=%s | fixed %d/%d (%.1f%%) | RSS=%d MB",
                         worker_index, task_num, n_tasks, graph_core.name, r_val,
                         n_fixed, n_repeats, 100.0 * n_fixed / n_repeats, _rss_mb())

            except Exception:
                log.exception("[Worker %s] ERROR in task %s", worker_index, row.task_id)
                continue

    if total_written:
        log.info("--- Worker %s done. %d rows -> %s | RSS=%d MB ---",
                 worker_index, total_written, os.path.basename(save_path), _rss_mb())
    else:
        log.warning("--- Worker %s done. No results generated. ---", worker_index)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HPC worker: runs Moran simulations for one LSF array job index."
    )
    parser.add_argument("--zoo-shard-dir", required=True,
                        help="Directory containing per-worker GraphCore shards (zoo_worker_N.pkl)")
    parser.add_argument("--manifest-path", required=True,
                        help="Path to the shared task_manifest.csv")
    parser.add_argument("--batch-dir", required=True,
                        help="Path to the batch tmp/ directory; results written to batch-dir/results/")
    parser.add_argument("--job-index", type=int, default=None,
                        help="Override job index (default: read from $LSB_JOBINDEX)")
    parser.add_argument("--engine", choices=["cpp", "python"], default="cpp",
                        help="Simulation engine: 'cpp' (fast, default) or 'python' (reference)")

    args = parser.parse_args()

    # Configure logging once, at the entry point. stdout -> LSF .out file.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    # Resolve job index: explicit arg takes precedence, then LSF env variable
    job_idx = args.job_index
    if job_idx is None:
        env_idx = os.environ.get("LSB_JOBINDEX")
        if env_idx:
            job_idx = int(env_idx)
        else:
            log.error("No job index found. Pass --job-index or submit via bsub.")
            sys.exit(1)

    run_worker_slice(args.batch_dir, args.zoo_shard_dir, args.manifest_path, job_idx,
                     engine=args.engine)

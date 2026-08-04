import pandas as pd
import time
import os
import glob
import logging
import shutil
import pickle
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import itertools
import joblib

from moran_process.core.population_graph import PopulationGraph
from moran_process.pipeline.worker_lsf import _resolve_engine
from moran_process.analysis.analysis_utils import create_batch_info

log = logging.getLogger(__name__)


def _parse_memory_mb(memory) -> int:
    """Convert a human-readable memory string to MB (integer) for LSF rusage.

    Accepts: "2GB", "2G", "512MB", "512M", or a bare integer string/int (treated as MB).
    Examples: "2GB" -> 2048, "8G" -> 8192, "512MB" -> 512, "2048" -> 2048.
    """
    if isinstance(memory, int):
        return memory
    s = str(memory).strip().upper()
    if s.endswith("GB") or s.endswith("G"):
        factor = 1024
        num = s.rstrip("GB").rstrip("G")
    elif s.endswith("MB") or s.endswith("M"):
        factor = 1
        num = s.rstrip("MB").rstrip("M")
    else:
        factor = 1
        num = s
    return int(float(num) * factor)


def _parse_lsf_job_id(bsub_stdout: str) -> str | None:
    """Extract the numeric job id from bsub's confirmation line.

    bsub prints e.g. ``Job <123456> is submitted to queue <short>.`` We grab the
    digits between the first angle brackets so batch_info can record the LSF id
    that owns this batch's logs and results.
    """
    if not bsub_stdout:
        return None
    import re

    m = re.search(r"Job <(\d+)>", bsub_stdout)
    return m.group(1) if m else None


class ProcessLab:
    """Manages multiple process runs and stores their results"""

    def __init__(self):
        """ """

    def run_comparative_study(
        self,
        graphs_zoo,
        r_values,
        n_repeats=100,
        print_time=True,
        output_path=None,
        engine="cpp",
    ):
        """
        Run comparative study across multiple graphs and selection coefficients.

        :param graphs: List of instantiated PopulationGraph objects
        :param r_values: List of floats (selection coefficients)
        :param n_repeats: Number of repetitions per configuration
        :param print_time: Whether to print timing information for each run
        :param output_path: Optional path to save results CSV. If provided, results will be
                           appended to existing file or create new file. Can be absolute or
                           relative path (e.g., 'simulation_data/results.csv')
        :return: DataFrame with all results
        """
        all_results = []

        # Resolve the engine class once (same helper the HPC worker uses);
        # 'cpp' is the fast C++ core, 'python' the pure-Python reference.
        MoranProcess = _resolve_engine(engine)

        # Total iterations for progress bar
        total_sims = len(graphs_zoo) * len(r_values) * n_repeats

        log.info(
            "--- Starting Study: %d Graphs x %d r-vals x %d = %d repeats (engine=%s) ---",
            len(graphs_zoo),
            len(r_values),
            n_repeats,
            total_sims,
            engine,
        )

        # We can optimize by converting graphs to adjacency lists ONCE
        for graph_obj in graphs_zoo:
            # Convert once per graph; reused across all r values and repeats
            graph_core = graph_obj.to_simulation_struct()

            for r in r_values:
                for _ in range(n_repeats):
                    sim = MoranProcess(graph_core=graph_core, selection_coefficient=r)
                    sim.initialize_random_mutant()
                    raw_result = sim.run()

                    record = {
                        **graph_obj.metadata,
                        "r": r,
                        "fixation": raw_result["fixation"],
                        "steps": raw_result["steps"],
                        "duration": raw_result["duration"],
                    }
                    all_results.append(record)
                    if print_time:
                        seconds = raw_result["duration"]
                        log.info(
                            "Graph: %s, r: %s, Fixation: %s, n_nodes: %d, Steps: %s, Time: %.4fs",
                            graph_obj.name,
                            r,
                            raw_result["fixation"],
                            graph_obj.number_of_nodes(),
                            raw_result["steps"],
                            seconds,
                        )

        log.info("Done.")
        df = pd.DataFrame(all_results)

        # Save to CSV if output_path is provided
        if output_path:
            ProcessLab.save_results(df, output_path)

        return df

    @staticmethod
    def save_results(df, output_path):
        """
        Save results to CSV file, appending to existing file if it exists.

        :param df: DataFrame with results to save
        :param output_path: Path to CSV file
        """
        # Ensure directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Append to existing CSV if it exists, otherwise create new
        if os.path.exists(output_path):
            existing_df = pd.read_csv(output_path)
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            log.info(
                "Appending %d new rows to existing CSV with %d rows",
                len(df),
                len(existing_df),
            )
            combined_df.to_csv(output_path, index=False)
            log.info("Total rows in CSV: %d", len(combined_df))
        else:
            df.to_csv(output_path, index=False)
            log.info("Created new CSV file with %d rows", len(df))

        log.info("Results saved to: %s", output_path)

    # --- HPC SUBMISSION ENGINE ---
    def submit_jobs(
        self,
        zoo_path,
        n_graphs,
        r_values,
        batch_name,
        batch_dir,
        n_repeats=10,
        n_requested_jobs=1,
        queue="short",
        memory="2GB",
        graph_types=None,
        node_sizes=None,
        description="",
        notes="",
        batch_seed=None,
        engine="cpp",
        zoo_config=None,
        post_batch="all",
    ):
        """
        1. Dumps all graphs to 'graphs.pkl'
        2. Creates 'task_manifest.csv' (The Huge Table)
        3. Submits an LSF Job Array where each worker takes a 'chunk' of the table.

        zoo_config: optional dict describing how the zoo was *built* (e.g.
            {'graph_zoo_seed': 42, 'random_graph_config': {...},
             'biological_graphs': [...]}). It is recorded verbatim under the
            'zoo' section of batch_info.json so the batch is reproducible from
            that file alone. main.py assembles it; everything else here is
            captured automatically.

        post_batch: which of the post-simulation jobs to chain.
            'all'       -- aggregate, verify, violin cache, job speed (the default; what
                           every ordinary batch wants).
            'aggregate' -- aggregate only. For callers that submit a batch per iteration
                           and only need prob_fixation / mean_steps back, where the other
                           three are pure scheduling latency: verify, the violin cache and
                           job speed all exist to serve figures and QC on a large one-off
                           batch. ga_search uses this.
            'none'      -- nothing beyond the array. The batch is left raw.

        Returns:
            dict of LSF job ids keyed by step ('register', 'array', 'aggregate', and
            whichever post-batch steps were submitted). Values may be None if a bsub
            failed or its output could not be parsed. Callers that chain on these degrade
            to "runs immediately" rather than PENDing forever on a stale condition.
        """
        if post_batch not in ("all", "aggregate", "none"):
            raise ValueError(
                f"post_batch must be 'all', 'aggregate' or 'none', got {post_batch!r}"
            )
        log.info(
            "Submitting batch '%s' (engine=%s, %d jobs, queue=%s)",
            batch_name,
            engine,
            n_requested_jobs,
            queue,
        )

        # Create subdirs for logs and results
        if os.path.exists(batch_dir):
            log.warning(
                "Batch directory %s already exists. Appending/Overwriting.", batch_name
            )

        tmp_dir = os.path.join(batch_dir, "tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        results_dir = os.path.join(tmp_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        logs_dir = os.path.join(batch_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)

        register_job_id = register_graphs_job(zoo_path, batch_name, batch_dir)

        log.info("--- Preparing Batch %s ---", batch_name)

        # 3. Generate Task Manifest (The Huge Table)
        # We expand the loops into a list of rows
        manifest_path = os.path.join(tmp_dir, "task_manifest.csv")
        manifest_df = ProcessLab._create_task_list(
            n_graphs,
            r_values,
            n_repeats,
            n_requested_jobs,
            output_path=manifest_path,
            batch_seed=batch_seed,
        )

        log.info("Created manifest with %d rows.", len(manifest_df))

        # 4. Load the full zoo once here (login node), convert to per-worker shards.
        # Workers receive a small list[GraphCore] shard (~50 graphs) instead of
        # the full zoo (50k graphs). This is the main RAM fix.
        log.info("Loading zoo from %s ...", zoo_path)
        with open(zoo_path, "rb") as f:
            graph_zoo = joblib.load(f)
        log.info("Zoo loaded: %d graphs.", len(graph_zoo))

        zoo_shards_dir = os.path.join(tmp_dir, "zoo_shards")
        manifest_df = ProcessLab._write_zoo_shards(
            manifest_df, graph_zoo, zoo_shards_dir
        )
        del graph_zoo  # free the full zoo; shards are on disk now

        manifest_df.to_csv(manifest_path, index=False)
        log.info("Manifest updated with local_graph_idx -> %s", manifest_path)

        # 5. Submit LSF job array.
        # Each worker receives --zoo-shard-dir and constructs its own shard path
        # using $LSB_JOBINDEX, so no global zoo path is needed at runtime.
        python_exec = sys.executable
        memory_mb = _parse_memory_mb(memory)

        cmd_job = [
            "bsub",
            "-q",
            queue,
            "-J",
            f"batch_{batch_name}[1-{n_requested_jobs}]",
            "-o",
            os.path.join(logs_dir, "job_%J_%I.out"),
            "-e",
            os.path.join(logs_dir, "job_%J_%I.err"),
            "-R",
            f"rusage[mem={memory_mb}]",
            "-env",
            "OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, PYTHONPATH=src",
        ]

        cmd_process = [
            python_exec,
            "-u",
            "-m",
            "moran_process.pipeline.worker_lsf",
            "--zoo-shard-dir",
            str(zoo_shards_dir),
            "--manifest-path",
            str(manifest_path),
            "--batch-dir",
            str(tmp_dir),
            "--engine",
            str(engine),
        ]
        cmd = cmd_job + cmd_process
        bsub_command = " ".join(cmd)

        # Capture stdout/stderr so we can parse the LSF job id and report real
        # errors (the old code passed no capture flags, so result.stderr was
        # always None).
        log.info("Submitting: %s", bsub_command)
        result = subprocess.run(cmd, capture_output=True, text=True)
        lsf_job_id = _parse_lsf_job_id(result.stdout)
        if result.returncode == 0:
            log.info("Submitted. LSF job id: %s", lsf_job_id or "unknown")
        else:
            log.error(
                "bsub failed with return code %d: %s",
                result.returncode,
                (result.stderr or "").strip(),
            )
        log.info("Batch submitted! Logs: %s | Results: %s", logs_dir, results_dir)

        create_batch_info(
            batch_dir=batch_dir,
            name=batch_name,
            description=description,
            notes=notes,
            r_values=r_values,
            n_repeats=n_repeats,
            total_simulations=n_graphs * len(r_values) * n_repeats,
            batch_seed=batch_seed,
            engine=engine,
            n_graphs=n_graphs,
            graph_types=graph_types,
            node_sizes=node_sizes,
            zoo_path=zoo_path,
            zoo_config=zoo_config,
            n_requested_jobs=n_requested_jobs,
            queue=queue,
            memory_mb=memory_mb,
            job_array_name=f"batch_{batch_name}",
            lsf_job_id=lsf_job_id,
            bsub_command=bsub_command,
        )

        job_ids = {"register": register_job_id, "array": lsf_job_id}

        if post_batch == "none":
            log.info("post_batch='none': no post-simulation jobs chained.")
            return job_ids

        # Chain the post-processing job: it PENDs until the array has ended and
        # register_graphs is done, then builds raw_results.parquet + graph_statistics.csv
        # on a compute node. This is what makes experiment_analysis.ipynb open instantly.
        aggregate_job_id = submit_aggregation_job(
            batch_dir=batch_dir,
            batch_name=batch_name,
            array_job_id=lsf_job_id,
            register_job_id=register_job_id,
            queue=queue,
        )
        job_ids["aggregate"] = aggregate_job_id

        # And chain the rest of the post-batch DAG: verify (did every requested run
        # happen?) and the violin-sample cache (the only figure input that still needs raw
        # rows) PEND on the aggregation; job speed hangs off the array directly and so runs
        # alongside it. By the time you open experiment_analysis.ipynb the batch is
        # verified, aggregated, and every figure input is a file read.
        if post_batch == "all":
            job_ids.update(
                submit_post_batch_jobs(
                    batch_dir=batch_dir,
                    batch_name=batch_name,
                    aggregate_job_id=aggregate_job_id,
                    array_job_id=lsf_job_id,
                    queue=queue,
                )
            )

        return job_ids

    # @staticmethod
    # def _create_task_list(n_graphs, r_values, n_jobs, n_repeats):
    #     """Create CSV task list for job array execution."""
    #     tasks = []
    #     task_id = 0
    #     simulations_per_worker = math.ceil((n_graphs * len(r_values) * n_repeats) / n_jobs)
    #     simulations = simulations_per_worker
    #     for graph_idx in range(n_graphs):
    #         for r in r_values:
    #             repeats = min(simulations, n_repeats)
    #             tasks.append({
    #                 'task_id': task_id,
    #                 'graph_idx': graph_idx,
    #                 'r': r,
    #                 'repeats': min(repeats)
    #             })
    #             task_id += 1
    #             simulations -= repeats

    #     return pd.DataFrame(tasks)

    def _create_task_list(
        n_graphs,
        r_values,
        repeats_per_config,
        num_workers,
        output_path="task_manifest.csv",
        batch_seed=None,
    ):
        """
        Allocates simulations to workers as evenly as possible.
        Splits a single configuration across multiple workers if necessary.

        batch_seed: integer seed for reproducible batches. A per-task seed is derived
                    from a batch-level RNG so the batch can be exactly replayed by
                    storing batch_seed in batch_info.json. None = random (no seeds stored).
        """
        import numpy as np

        # None → seeds from OS entropy; int → deterministic. Either way, seeds are
        # stored in the manifest so any batch can be replayed from its manifest alone.
        task_rng = np.random.default_rng(batch_seed)

        # 1. Generate all unique configurations (Graph X, r Y)
        configs = list(itertools.product(range(n_graphs), r_values))
        num_configs = len(configs)

        # 2. Calculate total work and fair share
        total_sims = num_configs * repeats_per_config
        base_share = total_sims // num_workers
        remainder = total_sims % num_workers

        tasks = []

        # Trackers for our position in the configurations list
        current_config_idx = 0
        # How many repeats of the current config are still waiting to be assigned?
        repeats_left_in_current_config = repeats_per_config
        task_id = 0
        # 3. Assign work to each worker
        for worker_id in range(num_workers):

            # Calculate exactly how many repeats this worker should handle
            # (Distribute the remainder: first few workers get +1 simulation)
            worker_target = base_share + (1 if worker_id < remainder else 0)

            while worker_target > 0 and current_config_idx < num_configs:
                graph_idx, r = configs[current_config_idx]

                # How many can we take from the current config?
                # Either all that are left in this config, or just enough to fill the worker.
                take = min(worker_target, repeats_left_in_current_config)

                # Add the row to our manifest
                tasks.append(
                    {
                        "task_id": task_id,
                        "worker_id": worker_id + 1,
                        "graph_idx": graph_idx,
                        "r_value": r,
                        "n_repeats": take,
                        "seed": int(task_rng.integers(0, 2**31)),
                    }
                )

                # Update counters
                worker_target -= take
                repeats_left_in_current_config -= take

                # If we used up this configuration, move to the next one
                if repeats_left_in_current_config == 0:
                    current_config_idx += 1
                    repeats_left_in_current_config = repeats_per_config

                task_id += 1
        # 4. Create DataFrame and save
        manifest = pd.DataFrame(tasks)
        manifest.to_csv(output_path, index=False)

        log.info(
            "Manifest created! Total Sims: %d. Distributed across %d workers.",
            total_sims,
            num_workers,
        )
        return manifest

    @staticmethod
    def _write_zoo_shards(manifest_df, graph_zoo, shards_dir):
        """Write one GraphCore shard per worker to shards_dir.

        Each shard is a list[GraphCore] containing only the graphs that worker
        needs, converted from PopulationGraph at submission time so workers
        never load NetworkX objects. The manifest is returned with an added
        `local_graph_idx` column (0-based index into the shard) alongside the
        original global `graph_idx` for debugging.
        """
        os.makedirs(shards_dir, exist_ok=True)
        local_idx_map = {}  # (worker_id, global_graph_idx) -> local_graph_idx

        n_workers = manifest_df["worker_id"].nunique()
        log.info("Creating %d zoo shards (GraphCore / CSR format)...", n_workers)

        for worker_id, group in manifest_df.groupby("worker_id"):
            global_idxs = sorted(group["graph_idx"].unique())
            for local_i, global_i in enumerate(global_idxs):
                local_idx_map[(worker_id, global_i)] = local_i

            shard = [graph_zoo[g].to_simulation_struct() for g in global_idxs]
            shard_path = os.path.join(shards_dir, f"zoo_worker_{worker_id}.pkl")
            joblib.dump(shard, shard_path)

            if worker_id % 100 == 0 or worker_id == 1:
                log.info(
                    "  [Shards] %s/%d - %d graphs -> %s",
                    worker_id,
                    n_workers,
                    len(shard),
                    os.path.basename(shard_path),
                )

        manifest_df = manifest_df.copy()
        manifest_df["local_graph_idx"] = [
            local_idx_map[(r.worker_id, r.graph_idx)] for r in manifest_df.itertuples()
        ]

        log.info("All %d shards written to %s", n_workers, shards_dir)
        return manifest_df


def submit_aggregation_job(
    batch_dir,
    batch_name,
    array_job_id=None,
    register_job_id=None,
    queue="short",
    memory="16GB",
    order_stats=False,
):
    """Submit the dependent post-processing job for a finished batch.

    When called during batch submission the numeric job ids are known, so an LSF
    ``-w`` dependency holds this job in PEND until the batch is done -- no polling,
    no login-node compute. The condition is:

        ended(<array>) [&& done(<register>)]

    ``ended`` (not ``done``) on the array means a single crashed worker will not
    strand this job in PEND forever; the aggregator surfaces any missing job
    indices instead. The register dependency guarantees graph_props.csv exists
    before the per-(graph, r) rollup runs. We key on numeric job ids (parsed from
    bsub) rather than names, so reusing a batch name across runs can't collide.

    Called standalone with no ids (``array_job_id`` and ``register_job_id`` both
    None) -- e.g. re-aggregating a batch whose array already finished -- no ``-w``
    flag is added and the job runs immediately. We deliberately do not fall back
    to a name-based ``ended(batch_<name>)`` dependency: once the array has left
    LSF's records that condition is rejected with "No matching job found".

    The job reads the per-job shards as a glob and writes only graph_statistics.csv; it
    never concatenates a raw_results.parquet (see aggregate_batch's module docstring for
    why the fused file is both unnecessary and unreadable past 2**32-1 rows).

    memory defaults to 16GB. With order stats off (the default) the polars group_by is a
    pure streaming reduction and needs far less, but flipping ``order_stats=True`` brings
    back the median/quantile materialization -- the heaviest, most memory-hungry step -- so
    the default leaves headroom for it. Bump it for very large batches if the job is killed.
    """
    # Only wait on jobs we can actually name by numeric id. If neither id is
    # given (e.g. re-aggregating a batch whose array already ended), we add no
    # -w flag at all so the job runs immediately -- a name-based ended(batch_*)
    # dependency would just be rejected with "No matching job found" once the
    # array has left LSF's records.
    return _submit_dependent_job(
        batch_dir=batch_dir,
        step="aggregate",
        batch_name=batch_name,
        module="moran_process.pipeline.aggregate_batch",
        module_args=["--order-stats"] if order_stats else [],
        dependencies=[
            f"ended({array_job_id})" if array_job_id else None,
            f"done({register_job_id})" if register_job_id else None,
        ],
        queue=queue,
        memory=memory,
    )


def _submit_dependent_job(
    batch_dir,
    step,
    batch_name,
    module,
    module_args=(),
    dependencies=(),
    queue="short",
    memory="16GB",
):
    """bsub ``python -m <module> --batch-dir <batch_dir>``, held until ``dependencies``.

    Every post-processing step (aggregate, then the violin cache and the QC report that
    chain off it) is the same submission with a different module and a different wait
    condition, so the bsub construction lives here once. Keeping it in one place is what
    guarantees they all get the same logs dir, the same PYTHONPATH, and the same
    ``sys.executable`` -- the last one matters because the venv python is what has polars.

    Args:
        step: short tag used for the job name (``batch_<name>_<step>``) and the log
            filenames, so a batch's logs directory stays self-describing.
        dependencies: LSF ``-w`` conditions, ANDed. ``None``/empty entries are dropped, and
            if nothing survives no ``-w`` flag is passed at all and the job runs
            immediately. That is the standalone case (re-running a step on a batch whose
            array has already left LSF's records, where a stale condition would be
            rejected outright rather than treated as satisfied).

    Returns:
        The LSF job id as a string, or None if bsub failed or its output could not be
        parsed. Callers chain on this, so a None id degrades to "runs immediately"
        rather than stranding the next job in PEND forever.
    """
    logs_dir = os.path.join(batch_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    dependency = " && ".join(c for c in dependencies if c)

    cmd = [
        "bsub",
        "-q",
        queue,
        "-J",
        f"batch_{batch_name}_{step}",
        *(["-w", dependency] if dependency else []),
        "-o",
        os.path.join(logs_dir, f"job_%J_{step}.out"),
        "-e",
        os.path.join(logs_dir, f"job_%J_{step}.err"),
        "-R",
        f"rusage[mem={_parse_memory_mb(memory)}]",
        "-env",
        "PYTHONPATH=src",
        sys.executable,
        "-u",
        "-m",
        module,
        "--batch-dir",
        str(batch_dir),
        *module_args,
    ]

    log.info(
        "Submitting %s job (depends on: %s): %s",
        step,
        dependency or "nothing (runs immediately)",
        " ".join(cmd),
    )
    result = subprocess.run(cmd, capture_output=True, text=True)
    job_id = _parse_lsf_job_id(result.stdout)
    if result.returncode == 0:
        log.info(
            "%s job submitted. LSF job id: %s (runs when: %s)",
            step,
            job_id or "unknown",
            dependency or "immediately",
        )
    else:
        log.error(
            "%s bsub failed with return code %d: %s",
            step,
            result.returncode,
            (result.stderr or "").strip(),
        )
    return job_id


def submit_violin_cache_job(
    batch_dir,
    batch_name,
    aggregate_job_id=None,
    queue="short",
    memory="8GB",
    max_points_per_category=50_000,
    force=False,
):
    """Chain the violin-sample cache after aggregation.

    Depends on ``done`` (not ``ended``) of the aggregation job, because it reads
    graph_statistics.csv to learn which r values were simulated. If aggregation failed
    there is nothing to cache, and PENDing forever is the honest outcome.

    8GB is roughly 20x the measured peak. The sampler holds one shard plus the reservoir
    (about 17 categories x 50k rows), never the full fixation set, so its memory does not
    grow with batch size. This used to ask for 32GB and still died at it (exit 137) on a
    7.2e9-row batch, because the old implementation collected every fixation row for one r
    before subsampling; see io.compute_fixation_steps_by_category and OPTIMIZATION_NOTES
    section 11a.
    """
    return _submit_dependent_job(
        batch_dir=batch_dir,
        step="violin_cache",
        batch_name=batch_name,
        module="moran_process.pipeline.cache_violin_data",
        module_args=[
            "--max-points-per-category",
            str(max_points_per_category),
            # The only post-batch step that skips work it has already done, so it is the
            # only one that needs telling to redo it. The others always overwrite.
            *(["--force"] if force else []),
        ],
        dependencies=[f"done({aggregate_job_id})" if aggregate_job_id else None],
        queue=queue,
        memory=memory,
    )


def submit_verify_job(
    batch_dir,
    batch_name,
    aggregate_job_id=None,
    queue="short",
    memory="8GB",
):
    """Chain the batch verification after aggregation.

    Reads only the CSVs, batch_info.json and the parquet footers, so it is seconds of work
    and modest memory whatever the batch size. Depends on ``done`` of aggregation because
    graph_statistics.csv is its main input.
    """
    return _submit_dependent_job(
        batch_dir=batch_dir,
        step="verify",
        batch_name=batch_name,
        module="moran_process.pipeline.batch_verify",
        dependencies=[f"done({aggregate_job_id})" if aggregate_job_id else None],
        queue=queue,
        memory=memory,
    )


def submit_job_speed_job(
    batch_dir,
    batch_name,
    array_job_id=None,
    queue="short",
    memory="8GB",
):
    """Reduce the raw shards to per-job step/duration totals, for the speed figures.

    Unlike verify and the violin cache, this does NOT wait on aggregation: it reads the
    raw shards and the LSF logs, and needs nothing from graph_statistics.csv. So it hangs
    off ``ended(array)`` directly and LSF runs it concurrently with the aggregation,
    making its wall-clock cost effectively zero.

    ``ended`` rather than ``done`` for the same reason aggregation uses it: one crashed
    worker should not strand this in PEND, and a batch with a dead worker is exactly the
    batch whose speed numbers you want to look at.
    """
    return _submit_dependent_job(
        batch_dir=batch_dir,
        step="job_speed",
        batch_name=batch_name,
        module="moran_process.pipeline.job_speed",
        dependencies=[f"ended({array_job_id})" if array_job_id else None],
        queue=queue,
        memory=memory,
    )


def submit_post_batch_jobs(
    batch_dir,
    batch_name,
    aggregate_job_id=None,
    array_job_id=None,
    queue="short",
    force=False,
):
    """Submit the post-aggregation jobs: verify, the violin cache, and job speed.

    verify and the violin cache are independent of each other, so both wait on the
    aggregation alone and LSF is free to run them concurrently. Job speed waits on the
    array instead, so it overlaps the aggregation entirely.
    """
    jobs = {
        "verify": submit_verify_job(
            batch_dir, batch_name, aggregate_job_id=aggregate_job_id, queue=queue
        ),
        "violin_cache": submit_violin_cache_job(
            batch_dir,
            batch_name,
            aggregate_job_id=aggregate_job_id,
            queue=queue,
            force=force,
        ),
    }
    jobs["job_speed"] = submit_job_speed_job(
        batch_dir, batch_name, array_job_id=array_job_id, queue=queue
    )
    return jobs


def register_graphs_job(
    graph_zoo_path, batch_name, batch_dir, queue="short", memory="8GB"
):

    log.info("Submitting register_graphs job for batch %s", batch_name)
    logs_dir = os.path.join(batch_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    python_exec = sys.executable
    memory_mb = _parse_memory_mb(memory)

    cmd_job = [
        "bsub",
        "-q",
        queue,
        "-J",
        f"batch_{batch_name}_register_graphs",
        "-o",
        os.path.join(logs_dir, "job_%J_register_graphs.out"),  # Log stdout
        "-e",
        os.path.join(logs_dir, "job_%J_register_graphs.err"),  # Log stderr
        "-R",
        f"rusage[mem={memory_mb}]",
        "-env",
        "PYTHONPATH=src",
    ]

    cmd_process = [
        python_exec,
        "-u",
        "-m",
        "moran_process.core.population_graph",
        "--register",
        "--batch-dir",
        str(batch_dir),
        "--graph-zoo-path",
        str(graph_zoo_path),
    ]
    cmd = cmd_job + cmd_process
    # cmd = cmd_process + ['--job-index', '1']

    log.info("Submitting register_graphs: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    register_job_id = _parse_lsf_job_id(result.stdout)
    if result.returncode == 0:
        log.info("register_graphs submitted. LSF job id: %s", register_job_id or "unknown")
    else:
        log.error(
            "register_graphs bsub failed with return code %d: %s",
            result.returncode,
            (result.stderr or "").strip(),
        )
    return register_job_id

# HPC Workflow: WEXAC (Weizmann)

The WEXAC cluster uses the **LSF** scheduler. All job submission uses `bsub`.
Full docs: https://hpcwiki.weizmann.ac.il/en/home/lsf/basic

---

## Simulation Engine (C++ vs Python)

Simulations default to a compiled C++ core (`_moran_cpp`, built from
`src/moran_process/simulations/_cpp/moran_core.cpp` via pybind11 + scikit-build-core). It is
~300x-1800x faster than and statistically equivalent to the pure-Python
`MoranProcess`.

- `uv sync` compiles the extension automatically (g++ is available on WEXAC).
- After editing the `.cpp`, recompile with `uv sync --reinstall-package moran_process`.
- Select the engine with `engine="cpp"` (default) or `engine="python"` in
  `submit_jobs`/`run_comparative_study`, or `--engine` on the CLI. The chosen
  engine is recorded in `batch_info.json`.

### Validating equivalence (two-batch method)

Equivalence is checked end-to-end through the real pipeline, not just the
simulation class. Submit two batches from `design_zoo.ipynb` that are identical
in every parameter (same `zoo_path`, `r_values`, `n_repeats`, `n_graphs`) except
the engine (one `engine="python"`, one `engine="cpp"`), then compare them:

```bash
uv run python scripts/compare_batches.py \
    simulation_data/<name>_python simulation_data/<name>_cpp
```

The comparator joins the two batches on `(wl_hash, r)` and, per cell, runs a
two-proportion z-test on ρ and a KS test on fixation time. The overall verdict
uses a Bonferroni-corrected threshold (`alpha / n_cells`) to control the
family-wise error rate, plus a KS test of all p-values against Uniform(0,1) to
catch a systematic sub-threshold bias. It auto-aggregates each batch's
`raw_results.parquet` from `tmp/results/` if missing.

**Note:** The heavy work is the two LSF batches; the Python batch is the slow
one, so give it enough `n_requested_jobs` that each worker's slice fits the
queue wall limit. `compare_batches.py` itself is light (it only reads results)
and is safe to run on the login node.

---

## Typical Workflow for a Simulation Batch

### Step 0: Design the Graph Zoo (always first)
Open `notebooks/design_zoo.ipynb`. This is the entry point for every new batch.

1. Set `BATCH_NAME` at the top (e.g. `"2026-05-20_my_study"`).
2. Run the cells for the graph types you want (mammalian, avian, fish, complete, cycle, random, ...).
3. Visualize with `zoo.draw_all()` to confirm the topology.
4. Serialize the zoo with joblib to `../simulation_data/{BATCH_NAME}/tmp/graph_zoo.joblib`.

The saved `graph_zoo.joblib` is the input to all downstream steps.

### Step 1: Submit the Batch
Load the saved zoo and call `ProcessLab.submit_jobs()` (Section 4 of the notebook, or from a script):

```python
from moran_process import GraphZoo, ProcessLab

zoo = GraphZoo.load(f"../simulation_data/{BATCH_NAME}/tmp/graph_zoo.joblib")
lab = ProcessLab()
lab.submit_jobs(
    zoo_path=f"../simulation_data/{BATCH_NAME}/tmp/graph_zoo.joblib",
    r_values=[1.0, 1.1, 1.2, 1.3, 2.0],
    n_repeats=10_000,
    n_requested_jobs=1000,
    n_graphs=len(zoo),
    queue="gsla-cpu",
    batch_dir=f"../simulation_data/{BATCH_NAME}",
    batch_name=BATCH_NAME,
    engine="cpp",  # "cpp" (fast, default) or "python" (reference)
)
```

This call:
1. Submits a short `register_graphs` job that writes graph properties to `<batch_dir>/graph_props.csv`
2. Creates `tmp/task_manifest.csv` (all tasks enumerated, each assigned a `worker_id`)
3. Splits the zoo into per-worker GraphCore shards `tmp/zoo_shards/zoo_worker_*.pkl` (and adds `local_graph_idx` to the manifest)
4. Submits the main job array via `bsub`

### Step 2: Monitor Jobs
```bash
bjobs                      # list your jobs
bjobs -w                   # wide format (see full names)
bjobs -l <job_id>          # detailed info (why pending?)
bpeek -f <job_id>          # follow stdout of a running job
```

### Step 3: Post-simulation jobs

Four independent jobs turn the raw shards into analysis-ready files. `submit_jobs` already chains them onto a fresh batch, so for a normal submission there is nothing to do here: check back later and the batch is ready.

```
             register_graphs --+
                               +--> aggregate --+--> verify
  simulation array (1..N) -----+                +--> violin cache
                               +--> job speed
```

They are four separate `bsub` jobs, not one job with four stages and not an array. Their resource footprints differ (16GB for the rollup, 8GB for the rest), so one job would reserve the maximum for the whole runtime; separate jobs let the independent ones overlap; a crash in one leaves the others' outputs intact; and each stays independently re-runnable without skip-flags. The edges are data dependencies only. `verify` and `violin cache` both read what `aggregate` writes but neither reads the other, which is what lets LSF run them concurrently. `job speed` needs nothing from the aggregation, so it hangs off the array directly and its wall-clock cost is effectively zero.

Dependencies are keyed on **numeric LSF job ids**, not job names, so reusing a batch name cannot collide and a stale name-based condition cannot be rejected after the array has left LSF's records:

```
aggregate       -w "ended(<array>) && done(<register>)"
verify          -w "done(<aggregate>)"
violin_cache    -w "done(<aggregate>)"
job_speed       -w "ended(<array>)"
```

`ended` on the array, not `done`: a single crashed worker must not strand the aggregation in PEND forever. Verify then surfaces the missing shard. `done` on aggregate for its two consumers: if the rollup failed there is nothing to verify or cache, and PENDing is the honest outcome. There is no polling anywhere; LSF holds each job in PEND until its condition is met.

**Checking, and running them by hand.** For a batch whose simulations already finished, or one submitted before this existed:

```bash
# inspect only, submits nothing, safe on the login node
uv run python -m moran_process.pipeline.post_batch --batch-dir simulation_data/<BATCH_NAME>

# submit whatever is missing (nothing to wait for, so everything starts at once)
uv run python -m moran_process.pipeline.post_batch --batch-dir simulation_data/<BATCH_NAME> --submit

# rebuild everything, rechaining the consumers onto the fresh aggregation
uv run python -m moran_process.pipeline.post_batch --batch-dir simulation_data/<BATCH_NAME> --submit --force
```

Each job also has its own CLI if you want to rerun exactly one:
```bash
uv run python -m moran_process.pipeline.aggregate_batch   --batch-dir <batch> [--order-stats]
uv run python -m moran_process.pipeline.batch_verify      --batch-dir <batch> [--skip-row-counts]
uv run python -m moran_process.pipeline.cache_violin_data --batch-dir <batch> [--r-values 1.0 1.1 ...] [--force]
uv run python -m moran_process.pipeline.job_speed         --batch-dir <batch>
```

`batch_verify` is seconds and modest memory at any batch size (it reads Parquet footers, not data), so it is safe on the login node. The other three are not.

**Combined batches.** `combine_batches` unions the parents' CSVs and symlinks their raw shards. Two steps do not apply and are reported as such rather than silently skipped: aggregate is `INHERITED` (concatenating the parents' `graph_statistics.csv` is exact, so there is nothing to recompute) and job speed is `N/A` (the linked shards still carry each parent's own 1..N `job_id` numbering, so summing by `job_id` would add unrelated workers together). Verify and the violin cache run normally.

The shards under a combined batch's `tmp/results/` are **absolute symlinks** into the parents. The parents must stay in place; do not move or delete them.

`aggregate_results_no_load(batch_dir)` still exists and fuses all shards into a single `raw_results.parquet`, but it is no longer part of the normal path: polars indexes rows with a u32 and cannot read a single Parquet file over 2**32-1 rows, and the 100K-reps batch is 7.2e9. Everything scans `tmp/results/*.parquet` as a glob instead.

### Step 4: Analyze
Once Step 3 reports `ready`, open `notebooks/experiment_analysis.ipynb` via `ijup` on a compute node and Run All. Every figure input is a file read; no cell triggers job-sized compute. The readiness cell at the top prints the per-step status table and, if anything is missing, the exact `ensure_post_batch(...)` call to fix it.

---

## Manual Worker Test (Debugging)
Run a single worker slice without bsub (pass `--job-index` explicitly instead of `LSB_JOBINDEX`; run as a module with `PYTHONPATH=src`):
```bash
PYTHONPATH=src uv run python -m moran_process.pipeline.worker_lsf \
    --zoo-shard-dir simulation_data/<BATCH_NAME>/tmp/zoo_shards \
    --manifest-path simulation_data/<BATCH_NAME>/tmp/task_manifest.csv \
    --batch-dir simulation_data/<BATCH_NAME>/tmp \
    --job-index 14 \
    --engine cpp   # or python
```

---

## LSF Quick Reference

### Queues (WEXAC)
| Queue | Max Walltime | Use Case |
|---|---|---|
| `short` | 30 min | Testing, small runs |
| `new-short` | ~12 hours | Standard short jobs |
| `medium` | 48 hours | Production runs |
| `long` | 7+ days | Very long simulations |
| `idle` | Unlimited | Zero priority, can be killed |

Check available queues: `bqueues`

### Common bsub Options
```bash
bsub -q <queue>                   # queue
bsub -n <cores>                   # number of cores
bsub -R "rusage[mem=2048]"        # memory per core (MB)
bsub -R "span[hosts=1]"           # all cores on same node
bsub -W 4:00                      # walltime HH:MM
bsub -J "name[1-N]"               # job array (indices 1 to N)
bsub -o logs/job_%J_%I.out        # stdout (%J=jobID, %I=arrayIndex)
bsub -e logs/job_%J_%I.err        # stderr
```

### Job Control
```bash
bkill <job_id>     # kill one job
bkill 0            # kill ALL your jobs
bstop <job_id>     # pause
bresume <job_id>   # resume
```

### Interactive Session (for debugging on compute node)
```bash
bsub -Is -q new-short -n 2 -W 30 bash
```

### Environment Variables in Workers
- `LSB_JOBINDEX`: the array index (1-based). `worker_lsf.py` reads this automatically.

---

## Key Paths on WEXAC
The project must be uploaded/synced to WEXAC before running. The typical structure mirrors the OneDrive structure. Worker scripts use **relative paths** from the `Moran-Process/` directory.

- Batch output: `Moran-Process/simulation_data/<batch_name>/`
- Graph properties (per batch, no global DB): `Moran-Process/simulation_data/<batch_name>/graph_props.csv`
- Logs: `Moran-Process/simulation_data/<batch_name>/logs/`
- Per-job results: `Moran-Process/simulation_data/<batch_name>/tmp/results/`
- Aggregated results: `Moran-Process/simulation_data/<batch_name>/raw_results.parquet`

---

## Notes & Gotchas
- `uv run` is used instead of `python` on the cluster (manages the venv).
- There is **no global graph database**. Structural properties are written per batch to `<batch>/graph_props.csv`; the join key between results and properties is `wl_hash`.
- Work is distributed via `tmp/task_manifest.csv`: each row carries a `worker_id`, and a worker runs the rows where `worker_id == LSB_JOBINDEX`. Repeats for a single (graph, r) config can be split across several workers.
- If `batch_dir` already exists, jobs append/overwrite (no auto-clean).
- The `memory` argument accepts strings like `"2GB"`/`"512MB"` (parsed to MB by `_parse_memory_mb`); the default is `"2GB"`. Increase for larger graphs.

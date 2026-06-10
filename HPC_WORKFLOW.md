# HPC Workflow — WEXAC (Weizmann)

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
the engine — one `engine="python"`, one `engine="cpp"` — then compare them:

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

### Step 0 — Design the Graph Zoo (always first)
Open `notebooks/design_zoo.ipynb`. This is the entry point for every new batch.

1. Set `BATCH_NAME` at the top (e.g. `"2026-05-20_my_study"`).
2. Run the cells for the graph types you want (mammalian, avian, fish, complete, cycle, random, ...).
3. Visualize with `zoo.draw_all()` to confirm the topology.
4. Serialize the zoo with joblib to `../simulation_data/{BATCH_NAME}/tmp/graph_zoo.joblib`.

The saved `graph_zoo.joblib` is the input to all downstream steps.

### Step 1 — Submit the Batch
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

### Step 2 — Monitor Jobs
```bash
bjobs                      # list your jobs
bjobs -w                   # wide format (see full names)
bjobs -l <job_id>          # detailed info (why pending?)
bpeek -f <job_id>          # follow stdout of a running job
```

### Step 3 — Collect Results
After all jobs finish, `simulation_data/<BATCH_NAME>/tmp/results/` will contain `raw_results_job_1.parquet` through `raw_results_job_N.parquet`. Stream them into one file with the built-in helper (copies files without loading them into memory):
```python
from moran_process.analysis.analysis_utils import aggregate_results_no_load
aggregate_results_no_load("simulation_data/<BATCH_NAME>")  # writes <BATCH_NAME>/raw_results.parquet
# pass delete_temp=True to also remove tmp/ afterward
```

### Step 4 — Analyze
Open notebooks in `analysis/` on the Windows PC (notebooks run locally using the OneDrive-synced data, or via Jupyter on WEXAC).

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
- `LSB_JOBINDEX` — the array index (1-based). `worker_lsf.py` reads this automatically.

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

---
name: moran-hpc
preamble-tier: 1
version: 1.0.0
description: |
  How the moran-process repository runs simulation batches on the WEXAC LSF
  cluster. Covers the design_zoo -> submit_jobs -> job array -> aggregate
  workflow, the task manifest and per-worker zoo shards, worker_lsf and how to
  debug a single slice, the cpp vs python simulation engine and its equivalence
  check, batch directory layout, and pipeline gotchas.
  Use when the user asks about submitting or monitoring a simulation batch,
  ProcessLab.submit_jobs, task_manifest.csv, zoo shards, worker_lsf,
  raw_results parquet, aggregate_results_no_load, graph_props.csv, wl_hash,
  the C++ engine, or debugging a failed batch job.
  For raw LSF syntax (bsub options, queues, GPU flags, bjobs, inode/ijup) see
  the separate `wexac-lsf` skill.
---

# moran-process on WEXAC

This skill covers how *this repository* uses LSF. For generic cluster knowledge
(queue walltimes, `bsub` options, `bjobs`, `inode`/`ijup`, the login-node
watchdog) load the **`wexac-lsf`** skill instead. The narrative version of this
document lives in `HPC_WORKFLOW.md` at the repo root.

---

## Simulation engine (C++ vs Python)

Simulations default to a compiled C++ core (`_moran_cpp`, built from
`src/moran_process/simulations/_cpp/moran_core.cpp` via pybind11 and
scikit-build-core). `CppMoranProcess` is a drop-in replacement for the pure
Python `MoranProcess`, roughly 300x to 1800x faster.

It is statistically equivalent but **not bit-exact**: it uses xoshiro256++
rather than NumPy's PCG64, so per-seed trajectories differ while fixation
probability and fixation-time distributions match within Monte Carlo error.
Sampling uses a two-pool O(1) trick (mutants vs wild-type partition) instead of
NumPy's O(N) cumulative `choice`; the distribution is identical.

```bash
uv sync                                    # also compiles the extension (g++ is on WEXAC)
uv sync --reinstall-package moran_process  # required after editing the .cpp
```

Plain `uv sync` will **not** recompile if nothing else changed, so the
`--reinstall-package` form is the one to use after touching the C++.

Pick the engine with `engine="cpp"` (default) or `engine="python"` in
`submit_jobs` / `run_comparative_study`, or `--engine` on the CLI.
`worker_lsf._resolve_engine()` swaps the class at startup so the run loop is
identical either way, and the choice is recorded in `batch_info.json`.

### Validating equivalence (two-batch method)

Equivalence is checked end to end through the real pipeline, not just the
simulation class. Submit two batches identical in every parameter (`zoo_path`,
`r_values`, `n_repeats`, `n_graphs`) except the engine, then compare:

```bash
uv run python scripts/compare_batches.py \
    simulation_data/<name>_python simulation_data/<name>_cpp
```

The comparator joins the two batches on `(wl_hash, r)` and, per cell, runs a
two-proportion z-test on rho and a KS test on fixation time. The verdict uses a
Bonferroni-corrected threshold (`alpha / n_cells`) plus a KS test of all
p-values against Uniform(0,1) to catch systematic sub-threshold bias. It
auto-aggregates each batch's `raw_results.parquet` from `tmp/results/` if
missing.

The heavy part is the two batches themselves; the Python one is the slow side,
so give it enough `n_requested_jobs` that each worker slice fits the queue
walltime. `compare_batches.py` only reads results and is safe on the login node.

---

## Batch workflow

### Step 0, design the zoo (always first)

Open `notebooks/design_zoo.ipynb`. This is the entry point for every new batch.

1. Set `BATCH_NAME` at the top (e.g. `"2026-05-20_my_study"`).
2. Run the cells for the graph types you want (mammalian, avian, fish,
   complete, cycle, random, ...).
3. Visualize with `zoo.draw_all()` to confirm the topology.
4. Serialize the zoo with joblib to
   `simulation_data/{BATCH_NAME}/tmp/graph_zoo.joblib`.

That file is the input to every downstream step.

### Step 1, submit

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
    engine="cpp",
)
```

That single call does four things:

1. `bsub`s a short `register_graphs` job that writes `<batch_dir>/graph_props.csv`
   (dedup by WL hash)
2. writes `tmp/task_manifest.csv`, enumerating every task with a `worker_id`
3. splits the zoo into per-worker shards `tmp/zoo_shards/zoo_worker_*.pkl` and
   adds `local_graph_idx` to the manifest
4. submits the main job array via `bsub`

There is also a CLI entry point that builds the respiratory plus random zoo and
submits it:

```bash
uv run python -m moran_process.pipeline.main --batch-name <name> [--engine cpp|python]
```

For small local runs, call `ProcessLab.run_comparative_study()` directly instead;
it runs serially in-process and appends to an existing CSV automatically.

### Step 2, monitor

`bjobs`, `bjobs -w`, `bjobs -l <id>`, `bpeek -f <id>`. See the `wexac-lsf` skill
for the full command set and for reading pending or suspended reasons.

### Step 3, collect

Workers write `simulation_data/<BATCH_NAME>/tmp/results/raw_results_job_<N>.parquet`,
one row group per task. Stream them into one file without loading them into
memory:

```python
from moran_process.analysis.analysis_utils import aggregate_results_no_load
aggregate_results_no_load("simulation_data/<BATCH_NAME>")  # writes <BATCH_NAME>/raw_results.parquet
# delete_temp=True also removes tmp/ afterward
```

This copies files rather than concatenating dataframes, so it is safe on the
login node even for large batches.

### Step 4, analyze

Notebooks in `analysis/`, run locally against the synced data or through `ijup`
on WEXAC.

---

## Debugging a single worker slice

The worker is normally invoked as a module and reads `LSB_JOBINDEX` to decide
which manifest rows are its own:

```bash
python -m moran_process.pipeline.worker_lsf \
    --zoo-shard-dir <batch>/tmp/zoo_shards \
    --manifest-path <batch>/tmp/task_manifest.csv \
    --batch-dir <batch>/tmp
```

To run one slice by hand, pass `--job-index` in place of `LSB_JOBINDEX`. Use
`PYTHONPATH=src` and do it inside `inode`, not on the login node:

```bash
PYTHONPATH=src uv run python -m moran_process.pipeline.worker_lsf \
    --zoo-shard-dir simulation_data/<BATCH_NAME>/tmp/zoo_shards \
    --manifest-path simulation_data/<BATCH_NAME>/tmp/task_manifest.csv \
    --batch-dir simulation_data/<BATCH_NAME>/tmp \
    --job-index 14 \
    --engine cpp
```

Each worker loads only its own `zoo_worker_<index>.pkl` shard, never the full
zoo, and processes the manifest rows whose `worker_id` equals its index.

---

## Batch directory layout

| Path | Contents |
|---|---|
| `simulation_data/<batch>/` | batch root |
| `simulation_data/<batch>/batch_info.json` | parameters, including the engine used |
| `simulation_data/<batch>/graph_props.csv` | structural properties, per batch |
| `simulation_data/<batch>/logs/` | LSF stdout/stderr |
| `simulation_data/<batch>/tmp/graph_zoo.joblib` | serialized input zoo |
| `simulation_data/<batch>/tmp/task_manifest.csv` | task to worker assignment |
| `simulation_data/<batch>/tmp/zoo_shards/` | per-worker graph shards |
| `simulation_data/<batch>/tmp/results/` | per-job parquet output |
| `simulation_data/<batch>/raw_results.parquet` | aggregated output |

Worker scripts use paths relative to the project root, and the cluster path sets
`PYTHONPATH=src`.

---

## Gotchas

- Use `uv run` rather than bare `python` on the cluster; it manages the venv.
- There is **no global graph database**. Structural properties are per batch in
  `<batch>/graph_props.csv`, and the join key between results and properties is
  `wl_hash`.
- `PopulationGraph.metadata` returns only `{wl_hash, graph_name}`. That is what
  gets merged into result rows, so do not add expensive fields to it.
- Repeats for a single `(graph, r)` config can be split across several workers,
  so do not assume one worker owns a whole config.
- If `batch_dir` already exists, jobs append or overwrite. There is no auto-clean.
- The `memory` argument to `submit_jobs` takes strings like `"2GB"` or `"512MB"`
  (parsed by `_parse_memory_mb`), defaulting to `"2GB"`. Raise it for large graphs.
- Graph property computation has performance guards: diameter, radius and ASPL
  are skipped for N > 500; betweenness samples with k=50 for N > 100; closeness
  samples manually for N > 200.
- Aggregation and `compare_batches.py` are light enough for the login node.
  Anything that actually simulates is not.
- Graph names follow `{type}_{param1}{val1}_{param2}{val2}`, e.g. `avian_r4_l7`,
  `mammalian_b2_d4`.

---

## Related docs in this repo

- `HPC_WORKFLOW.md` — the same material in narrative form
- `AI_CONTEXT.md` — comprehensive context primer, start here
- `CODE_ARCHITECTURE.md` — full class API and the ML pipeline
- `PROJECT_OVERVIEW.md` — research question and current status

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

MSc Computational Biology thesis (Weizmann Institute of Science, supervised by Tzachi). Research question: do respiratory organ topologies (mammalian/avian/fish lung graphs) act as evolutionary amplifiers or suppressors compared to random graphs of similar size? Key metrics: fixation probability (ρ) and fixation time, varied across selection coefficient r. Python >=3.11, managed with `uv`.

## Commands

```bash
# Install dependencies (also compiles the C++ extension via scikit-build-core)
uv sync

# Rebuild ONLY the C++ extension after editing src/moran_process/simulations/_cpp/*.cpp
# (plain `uv sync` will not recompile if nothing else changed)
uv sync --reinstall-package moran_process

# Build and submit a simulation batch (LSF job array on WEXAC).
# --engine cpp (default) uses the fast C++ core; --engine python the reference.
uv run python -m moran_process.pipeline.main --batch-name <name> [--engine cpp|python]
```

**Note:** The `tests/` directory exists but all tests are currently untrusted (AI-generated, outdated). Do not run them. New tests will be written from scratch.

## Architecture

Code is an installable package under `src/moran_process/`. Run modules with
`uv run python -m moran_process.<subpackage>.<module>` (the cluster path sets `PYTHONPATH=src`).
The simulation pipeline has three layers:

**1. Graph Layer: `core/population_graph.py`**
- `PopulationGraph` wraps a NetworkX graph and computes a Weisfeiler-Lehman hash (`wl_hash`) for deduplication. Construction has NO database side effect.
- Factory classmethods: `complete_graph`, `cycle_graph`, `mammalian_lung_graph`, `avian_graph`, `fish_graph`, `random_connected_graph`.
- Registration is per batch: `PopulationGraph.batch_register(zoo, batch_dir)` writes `<batch_dir>/graph_props.csv` (dedup by WL hash). CLI: `python -m moran_process.core.population_graph --register --batch-dir ... --graph-zoo-path ...`.
- `save()`/`load()` use pickle for HPC serialization.
- `core/graph_zoo.py` defines `GraphZoo`, an ordered collection of graphs (the pipeline often serializes a plain `list[PopulationGraph]` via joblib instead).
- Performance guard: diameter/radius/ASPL are skipped for N > 500; betweenness uses k=50 sampling for N > 100; closeness uses manual sampling for N > 200.

**2. Simulation Layer: `simulations/`**
- `MoranProcess` (`moran_process.py`) implements one Moran process in pure Python: fitness-weighted reproduction, random neighbor replacement. This is the reference implementation.
- `initialize_random_mutant()` then `run()` returns `{fixation, steps, initial_mutants, selection_coeff, duration}`.
- `run(track_history=True)` also returns the mutant-count trajectory.
- `CppMoranProcess` (`cpp_moran_wrapper.py`) is a **drop-in replacement** with the identical interface, delegating the hot loop to the compiled `_moran_cpp` extension (`simulations/_cpp/moran_core.cpp`, built via pybind11 + scikit-build-core).
  - It is **statistically equivalent**, not bit-exact: it uses xoshiro256++ (not NumPy's PCG64), so per-seed trajectories differ but fixation probability (ρ) and fixation-time distributions match within Monte Carlo error. Validated by running two batches that differ only in `--engine` and comparing them with `scripts/compare_batches.py` (per-cell z-test on ρ and KS test on fixation time, Bonferroni-corrected, plus a p-value uniformity check); ~300x-1800x faster than the Python engine.
  - Sampling uses a two-pool O(1) trick (mutants/wild-type partition) instead of NumPy's O(N) cumulative `choice`; distribution is identical.
- Engine selection is via the `--engine {cpp,python}` flag (default `cpp`); `worker_lsf._resolve_engine()` swaps the class at startup so the run loop is identical for both.

**3. Orchestration Layer: `pipeline/process_lab.py`**
- `ProcessLab.run_comparative_study(graphs_zoo, r_values, n_repeats, output_path, engine="cpp")` runs locally and serially; appends to an existing CSV automatically.
- `ProcessLab.submit_jobs(zoo_path, n_graphs, r_values, batch_name, batch_dir, n_repeats, n_requested_jobs, queue, memory, engine="cpp")` is the HPC path: it bsubs a `register_graphs` job (writes `graph_props.csv`), generates `tmp/task_manifest.csv`, then submits an LSF job array that runs the worker as a module. The chosen `engine` is passed to the worker and recorded in `batch_info.json`.

**HPC Worker: `pipeline/worker_lsf.py`**
- Invoked as `python -m moran_process.pipeline.worker_lsf --zoo-shard-dir <batch>/tmp/zoo_shards --manifest-path <m> --batch-dir <batch>/tmp`. Each worker loads only its own `zoo_worker_<LSB_JOBINDEX>.pkl` shard, not the full zoo.
- Reads `LSB_JOBINDEX` and processes the manifest rows whose `worker_id` equals that index.
- Writes per-job results to `<batch_dir>/tmp/results/raw_results_job_<idx>.parquet` (one row-group per task).
- `--engine {cpp,python}` (default `cpp`) selects the simulation engine via `_resolve_engine()`.
- For local debugging: pass `--job-index 1` explicitly.

**4. Post-Simulation Layer: four independent jobs, chained by LSF dependencies**
Everything that turns raw shards into analysis-ready files runs on compute nodes, never in a notebook and never on the login node.

```
             register_graphs --+
                               +--> aggregate --+--> verify
  simulation array (1..N) -----+                +--> violin cache
                               +--> job speed
```

- `pipeline/aggregate_batch.py` (16GB): raw shards + `graph_props.csv` -> `graph_statistics.csv`, one row per `(wl_hash, r)`. Scans the shards as a **glob** and never concatenates them (polars indexes rows with a u32 and rejects a single Parquet file over 2**32-1 rows; the 100K-reps batch is 7.2e9). Every statistic it writes is an additive per-shard partial, so the rollup is bounded memory. `--order-stats` adds a median/quartile pass, which does not decompose and is off by default.
- `pipeline/batch_verify.py` (8GB): answers two questions. Did every requested run happen (every graph at every r, every cell holding its full `n_repeats`)? Did any job fail (every array index produced a non-empty, readable shard)? Reads Parquet **footers** for row counts, so it never scans the data. This is not redundant with aggregate: aggregate does not fail on a missing shard, it rolls up whatever exists, so a preempted worker silently yields a clean-looking `graph_statistics.csv`. Exits 0 even on FAIL by design; the verdict lives in `report/verification.json`.
- `pipeline/cache_violin_data.py` (8GB): raw shards -> `cache/fixation_steps_r{r}_n{cap}.parquet`. Violin and Mann-Whitney plots need individual fixation-step values, which cannot be reconstructed from moments. Bounded by a reservoir; one pass covers every r at once (see `OPTIMIZATION_NOTES.md` section 11).
- `pipeline/job_speed.py` (8GB): raw shards -> `job_speed.csv` (`job_id,steps,duration`, per-job sums). Replaces the last job-sized computation that used to run inside the notebook kernel.

`pipeline/post_batch.py` is the single entry point and splits along an inspect/act line:
- `post_batch_status(batch_dir, r_values=None)` **inspects**: stats a few small files, submits nothing, safe on every Run All.
- `ensure_post_batch(batch_dir, force=False)` **submits**: fills gaps, or with `force=True` rebuilds everything and rechains the consumers.

```bash
uv run python -m moran_process.pipeline.post_batch --batch-dir <batch>            # status
uv run python -m moran_process.pipeline.post_batch --batch-dir <batch> --submit
```

Two batch kinds are classified and reported, so "works on any batch" includes saying no clearly: `CURRENT` (all four steps apply) and `LEGACY` (pre-June batches; none apply, no compat shims were added).

**Spanning several batches: stitch at read time, do not build a combined batch.** Both readers take a single batch directory **or a list of them**:

```python
analysis_df = load_graph_statistics([dir_a, dir_b], r_filter=[1.1])
plot_steps_violin([dir_a, dir_b], df_graph_props, r=1.1, ...)
```

The frames are concatenated in memory and a `batch` column records each row's source. This is exact, not an approximation: every statistic in `graph_statistics.csv` is an additive per-shard partial keyed by `(wl_hash, r)`, so concatenating two rollups equals the rollup of their union, and the violin cache is a per-category reservoir, so joining two samples equals sampling the union. Counts behind the rho annotation are taken before subsampling and therefore sum.

Two things are reported at load time rather than left to surprise you mid-figure: **ragged columns** (the respiratory-only construction params, which a GA graph genuinely does not have, so they are NaN elsewhere) and **`(wl_hash, r)` collisions**. A collision is not an error: stitching two batches of the *same* zoo at different `n_repeats` is a legitimate comparison, so both rows are kept and labelled by `batch`. Facet or color by it, or the same graph is drawn twice. A batch that cannot serve the requested `r` raises, and the message says whether the cache merely has not been built (fixable) or `r` was never simulated there (not fixable).

There is deliberately no on-disk "combined batch". An earlier version built one by unioning the CSVs and symlinking 2000 raw shards into a third directory, which bought only the ability to re-run shard-scanning jobs over the union, and cost a permanent coupling to the parents' locations.

**Builder/reader split.** Every expensive artefact has a builder that only jobs call and a reader that only consumers call, and the reader raises rather than silently building:

| artefact | builder (jobs only) | reader (notebook, streamlit, ML) |
|---|---|---|
| `graph_statistics.csv` | `io.build_graph_statistics` | `io.load_graph_statistics` |
| violin sample | `io.compute_fixation_steps_by_category` | `io.load_fixation_steps_by_category` |

Both used to be single "compute if missing, else load" functions. That shape is right when the compute is milliseconds and wrong once it costs 42GB of I/O: the same call was either a 20ms file read or an unannounced hour of compute inside a Jupyter kernel, depending on state invisible from the call site. Do not reintroduce the fallback.

- ijup and inode are functions defined in .bashrc for interactive jobs

## Data Flow

```
notebooks/design_zoo.ipynb  (or pipeline/main.py)
  -> serialize zoo to simulation_data/<batch>/tmp/graph_zoo.joblib
  -> ProcessLab.submit_jobs()                       # HPC
      -> register_graphs job -> simulation_data/<batch>/graph_props.csv
      -> simulation_data/<batch>/tmp/task_manifest.csv
      -> bsub job array -> worker_lsf
          -> simulation_data/<batch>/tmp/results/raw_results_job_N.parquet
          -> aggregate  -> simulation_data/<batch>/graph_statistics.csv
              -> verify        -> simulation_data/<batch>/report/verification.json
              -> violin cache  -> simulation_data/<batch>/cache/fixation_steps_r*_n*.parquet
          -> job speed  -> simulation_data/<batch>/job_speed.csv
  -> notebooks/experiment_analysis.ipynb            # file reads only, no compute
  -> ProcessLab.run_comparative_study()             # Local alternative
      -> simulation_data/*.csv
```

There is no global graph database. Structural properties live per batch in `graph_props.csv`,
and the join key between results and properties is `wl_hash`.

## Aggregating Batch Results

Use the post-simulation jobs above. `ProcessLab.submit_jobs` already chains them onto a fresh batch, so nothing manual is needed; for a batch whose simulations already finished, run `post_batch --submit`.

`aggregate_results_no_load(batch_dir)` still exists and fuses the per-job shards into a single `raw_results.parquet`, but it is **not** part of the normal path. Every consumer scans `tmp/results/*.parquet` as a glob instead, because a fused file above 2**32-1 rows is unreadable by polars.

## Key Conventions

- Graph names follow the pattern `{type}_{param1}{val1}_{param2}{val2}` (e.g. `avian_r4_l7`, `mammalian_b2_d4`).
- `PopulationGraph.metadata` returns only `{wl_hash, graph_name}`: this is what gets merged into result rows; do not add expensive fields here.
- `pipeline/main.py` builds the respiratory + random zoo and submits via `submit_jobs` (HPC). For local small-scale runs, use `run_comparative_study` directly from `design_zoo.ipynb` (cell 17, commented-out block).
- The VS Code green Run button always executes on the WEXAC **login node**. Use an `inode` terminal session for anything compute-heavy, or `bsub` for real simulations.

## Reference Docs

Read these files when the task requires deeper context:
- `AI_CONTEXT.md`: single comprehensive, up-to-date context primer; start here
- `PROJECT_OVERVIEW.md`: research question, graph types, current status, open tasks
- `CODE_ARCHITECTURE.md`: full class API, ML pipeline, analysis notebooks
- `HPC_WORKFLOW.md`: WEXAC job submission, monitoring, the post-simulation job DAG, debugging
- `OPTIMIZATION_NOTES.md`: memory and speed work, with the measurements behind each decision
- `RESEARCH_BACKGROUND.md`: Moran process math, amplifier/suppressor theory, key papers
- `VSCODE_WEXAC_WORKFLOW.md`: VS Code + WEXAC setup, when to use inode/ijup/bsub

## Rules

- Never use em dashes
- Before applying a non-trivial code change that involves a design decision (imputation strategy, algorithm choice, data filtering), explain the reasoning and the alternatives considered, then wait for confirmation before editing the file.
- If you want to remove some part of the code, that's okay but justify it first and don't do it silently.
- Avoid overkill solutions - keep things simple.
- Don't run heavy tests yourself, this is a login node. If you do, ask for permission before.

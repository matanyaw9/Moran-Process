# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

MSc Computational Biology thesis (Weizmann Institute of Science, supervised by Tzachi). Research question: do respiratory organ topologies (mammalian/avian/fish lung graphs) act as evolutionary amplifiers or suppressors compared to random graphs of similar size? Key metrics: fixation probability (ρ) and fixation time, varied across selection coefficient r. Python >=3.11, managed with `uv`.

## Commands

```bash
# Install dependencies (also compiles the C++ extension via scikit-build-core)
uv sync

# Rebuild ONLY the C++ extension after editing src/moran_process/_cpp/*.cpp
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
- `MoranProcess` (`moran_simulation_process.py`) implements one Moran process in pure Python: fitness-weighted reproduction, random neighbor replacement. This is the reference implementation.
- `initialize_random_mutant()` then `run()` returns `{fixation, steps, initial_mutants, selection_coeff, duration}`.
- `run(track_history=True)` also returns the mutant-count trajectory.
- `CppMoranProcess` (`cpp_moran.py`) is a **drop-in replacement** with the identical interface, delegating the hot loop to the compiled `_moran_cpp` extension (`_cpp/moran_core.cpp`, built via pybind11 + scikit-build-core).
  - It is **statistically equivalent**, not bit-exact: it uses xoshiro256++ (not NumPy's PCG64), so per-seed trajectories differ but fixation probability (ρ) and fixation-time distributions match within Monte Carlo error. Validated by running two batches that differ only in `--engine` and comparing them with `scripts/compare_batches.py` (per-cell z-test on ρ and KS test on fixation time, Bonferroni-corrected, plus a p-value uniformity check); ~300x-1800x faster than the Python engine.
  - Sampling uses a two-pool O(1) trick (mutants/wild-type partition) instead of NumPy's O(N) cumulative `choice`; distribution is identical.
- Engine selection is via the `--engine {cpp,python}` flag (default `cpp`); `worker_wrapper._resolve_engine()` swaps the class at startup so the run loop is identical for both.

**3. Orchestration Layer: `pipeline/process_lab.py`**
- `ProcessLab.run_comparative_study(graphs_zoo, r_values, n_repeats, output_path, engine="cpp")` runs locally and serially; appends to an existing CSV automatically.
- `ProcessLab.submit_jobs(zoo_path, n_graphs, r_values, batch_name, batch_dir, n_repeats, n_requested_jobs, queue, memory, engine="cpp")` is the HPC path: it bsubs a `register_graphs` job (writes `graph_props.csv`), generates `tmp/task_manifest.csv`, then submits an LSF job array that runs the worker as a module. The chosen `engine` is passed to the worker and recorded in `batch_info.json`.

**HPC Worker: `pipeline/worker_wrapper.py`**
- Invoked as `python -m moran_process.pipeline.worker_wrapper --zoo-path <z> --manifest-path <m> --batch-dir <batch>/tmp`.
- Reads `LSB_JOBINDEX` and processes the manifest rows whose `worker_id` equals that index.
- Writes per-job results to `<batch_dir>/tmp/results/result_job_<idx>.parquet` (one row-group per task).
- `--engine {cpp,python}` (default `cpp`) selects the simulation engine via `_resolve_engine()`.
- For local debugging: pass `--job-index 1` explicitly.

## Data Flow

```
notebooks/design_zoo.ipynb  (or pipeline/main.py)
  -> serialize zoo to simulation_data/<batch>/tmp/graph_zoo.joblib
  -> ProcessLab.submit_jobs()                       # HPC
      -> register_graphs job -> simulation_data/<batch>/graph_props.csv
      -> simulation_data/<batch>/tmp/task_manifest.csv
      -> bsub job array -> worker_wrapper
          -> simulation_data/<batch>/tmp/results/result_job_N.csv
  -> ProcessLab.run_comparative_study()             # Local alternative
      -> simulation_data/*.csv
```

There is no global graph database. Structural properties live per batch in `graph_props.csv`,
and the join key between results and properties is `wl_hash`.

## Aggregating Batch Results

After all LSF jobs finish, stream the per-job CSVs into one file with the built-in helper:
```python
from moran_process.analysis.analysis_utils import aggregate_results_no_load
aggregate_results_no_load("simulation_data/<batch>")  # writes <batch>/full_results.csv
```
This copies files without loading them into memory. Pass `delete_temp=True` to remove `tmp/` afterward.

## Key Conventions

- Graph names follow the pattern `{type}_{param1}{val1}_{param2}{val2}` (e.g. `avian_r4_l7`, `mammalian_b2_d4`).
- `PopulationGraph.metadata` returns only `{wl_hash, graph_name}` — this is what gets merged into result rows; do not add expensive fields here.
- `pipeline/main.py` builds the respiratory + random zoo and submits via `submit_jobs` (HPC). For local small-scale runs, use `run_comparative_study` directly from `design_zoo.ipynb` (cell 17, commented-out block).
- The VS Code green Run button always executes on the WEXAC **login node**. Use an `inode` terminal session for anything compute-heavy, or `bsub` for real simulations.

## Reference Docs

Read these files when the task requires deeper context:
- `AI_CONTEXT.md` — single comprehensive, up-to-date context primer; start here
- `PROJECT_OVERVIEW.md` — research question, graph types, current status, open tasks
- `CODE_ARCHITECTURE.md` — full class API, ML pipeline, analysis notebooks
- `HPC_WORKFLOW.md` — WEXAC job submission, monitoring, result aggregation, debugging
- `RESEARCH_BACKGROUND.md` — Moran process math, amplifier/suppressor theory, key papers
- `VSCODE_WEXAC_WORKFLOW.md` — VS Code + WEXAC setup, when to use inode/ijup/bsub

## Rules

- Never use em dashes
- Before applying a non-trivial code change that involves a design decision (imputation strategy, algorithm choice, data filtering), explain the reasoning and the alternatives considered, then wait for confirmation before editing the file.
- If you want to remove some part of the code, that's okay but justify it first and don't do it silently.

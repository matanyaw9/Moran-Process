# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

MSc Computational Biology thesis (Weizmann Institute of Science, supervised by Tzachi). Research question: do respiratory organ topologies (mammalian/avian/fish lung graphs) act as evolutionary amplifiers or suppressors compared to random graphs of similar size? Key metrics: fixation probability (ρ) and fixation time, varied across selection coefficient r. Python >=3.11, managed with `uv`.

## Commands

```bash
# Install dependencies
uv sync

# Build and submit a simulation batch (LSF job array on WEXAC)
uv run python -m moran_process.pipeline.main --batch-name <name>

# Local random-graph path (no HPC)
uv run python -m moran_process.pipeline.run_random_graphs
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

**2. Simulation Layer: `simulations/process_run.py`**
- `ProcessRun` implements one Moran process: fitness-weighted reproduction, random neighbor replacement.
- `initialize_random_mutant()` then `run()` returns `{fixation, steps, initial_mutants, selection_coeff, duration}`.
- `run(track_history=True)` also returns the mutant-count trajectory.

**3. Orchestration Layer: `pipeline/process_lab.py`**
- `ProcessLab.run_comparative_study(graphs_zoo, r_values, n_repeats, output_path)` runs locally and serially; appends to an existing CSV automatically.
- `ProcessLab.submit_jobs(zoo_path, n_graphs, r_values, batch_name, batch_dir, n_repeats, n_requested_jobs, queue, memory)` is the HPC path: it bsubs a `register_graphs` job (writes `graph_props.csv`), generates `tmp/task_manifest.csv`, then submits an LSF job array that runs the worker as a module.

**HPC Worker: `pipeline/worker_wrapper.py`**
- Invoked as `python -m moran_process.pipeline.worker_wrapper --zoo-path <z> --manifest-path <m> --batch-dir <batch>/tmp`.
- Reads `LSB_JOBINDEX` and processes the manifest rows whose `worker_id` equals that index.
- Writes per-job results to `<batch_dir>/tmp/results/result_job_<idx>.csv`.
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
- `pipeline/main.py` builds the respiratory + random zoo and submits via `submit_jobs` (HPC); `pipeline/run_random_graphs.py` is the local random-only path using `run_comparative_study`. Check which execution path is needed before running.
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

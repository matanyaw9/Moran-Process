# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

MSc Computational Biology thesis (Weizmann Institute of Science, supervised by Tzachi). Research question: do respiratory organ topologies (mammalian/avian/fish lung graphs) act as evolutionary amplifiers or suppressors compared to random graphs of similar size? Key metrics: fixation probability (ρ) and fixation time, varied across selection coefficient r. Python 3.13, managed with `uv`.

## Commands

```bash
# Install dependencies
uv sync

# Run respiratory graph experiments (submits LSF job array to WEXAC)
uv run main.py

# Run random graph experiments locally
uv run run_random_graphs.py

# Quick local test (fewer graphs/repeats)
uv run tests/test_run_random_graphs.py

# Run pytest suite
uv run pytest tests/

# Run a single test file
uv run pytest tests/test_csv_append.py
```

## Architecture

The simulation pipeline has three layers:

**1. Graph Layer — `population_graph.py`**
- `PopulationGraph` wraps a NetworkX graph and auto-registers it in `simulation_data/graph_database.csv` on construction (keyed by WL hash to avoid duplicates).
- Factory classmethods: `complete_graph`, `cycle_graph`, `mammalian_lung_graph`, `avian_graph`, `fish_graph`, `random_connected_graph`. Pass `register_in_db=False` to skip the database write.
- `save()`/`load()` use pickle for HPC serialization.
- Performance guard: diameter/radius/ASPL are skipped for N > 500; betweenness uses k=50 sampling for N > 100; closeness uses manual sampling for N > 200.

**2. Simulation Layer — `process_run.py`**
- `ProcessRun` implements one Moran process: fitness-weighted reproduction, random neighbor replacement.
- `initialize_random_mutant()` then `run()` → returns `{fixation, steps, initial_mutants, selection_coeff, duration}`.
- `run(track_history=True)` also returns the mutant-count trajectory.

**3. Orchestration Layer — `process_lab.py`**
- `ProcessLab.run_comparative_study(graphs_zoo, r_values, n_repeats, output_path)` — local serial execution; appends to existing CSV automatically.
- `ProcessLab.submit_jobs(...)` — HPC path: serializes graph zoo to `graphs.pkl`, generates a `task_manifest.csv`, then submits an LSF job array via `bsub`. Each array job runs `worker_wrapper.py`.

**HPC Worker — `worker_wrapper.py`**
- Invoked by `bsub` as `uv run worker_wrapper.py --batch-dir <path> --chunk-size <n>`.
- Reads `LSB_JOBINDEX` from the environment to determine its slice of the manifest.
- Writes per-job results to `<batch_dir>/results/result_job_<idx>.csv`.
- For local debugging: pass `--job-index 1` explicitly.

## Data Flow

```
main.py / run_random_graphs.py
  → ProcessLab.submit_jobs()          # HPC
      → batch_dir/graphs.pkl
      → batch_dir/task_manifest.csv
      → bsub job array → worker_wrapper.py
          → batch_dir/results/result_job_N.csv
  → ProcessLab.run_comparative_study()  # Local
      → simulation_data/*.csv
```

Results CSVs append on re-run. The graph database (`simulation_data/graph_database.csv`) is a class-level singleton in `PopulationGraph` — it is loaded once per process and written after each new graph registration.

## Aggregating Batch Results

After all LSF jobs finish, results are split across individual CSVs. No built-in aggregation method exists yet — do it manually:
```python
import glob, pandas as pd
files = glob.glob('simulation_data/tmp/batch_NAME/results/result_job_*.csv')
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df.to_csv('simulation_data/combined_results.csv', index=False)
```

## Key Conventions

- Graph names follow the pattern `{type}_{param1}{val1}_{param2}{val2}` (e.g. `avian_r4_l7`, `mammalian_b2_d4`).
- `PopulationGraph.metadata` returns only `{wl_hash, graph_name}` — this is what gets merged into result rows; do not add expensive fields here.
- `main.py` currently uses `submit_jobs` (HPC); `run_random_graphs.py` uses `run_comparative_study` (local). Check which execution path is needed before running.
- The VS Code green Run button always executes on the WEXAC **login node**. Use an `inode` terminal session for anything compute-heavy, or `bsub` for real simulations.

## Reference Docs

Read these files when the task requires deeper context:
- `PROJECT_OVERVIEW.md` — research question, graph types, current status, open tasks
- `CODE_ARCHITECTURE.md` — full class API, ML pipeline, analysis notebooks
- `HPC_WORKFLOW.md` — WEXAC job submission, monitoring, result aggregation, debugging
- `RESEARCH_BACKGROUND.md` — Moran process math, amplifier/suppressor theory, key papers
- `VSCODE_WEXAC_WORKFLOW.md` — VS Code + WEXAC setup, when to use inode/ijup/bsub

## Rules

- Never use em dashes


# A Newer Moran Process Project

This project is a rewrite of my Moran Process repo. It's a library where one can create population graphs, run moran simulaiton on them and get statistics about it.

In particular I am interested in population graphs that are shaped like Respiratory organs. Could it be that Respiratory Organs have evolved to supress / slow down the fixation of mutations in a population that lives inside them?

## History

This repo started as the Moran-Process project. In October 2026 the code was rewritten from scratch in a separate repo (`moran-graphs`) and merged back here with its full history.

- The old code is at tag `v1-archive`. To get a runnable copy next to this one: `git worktree add ../moran-v1 v1-archive`.
- Its results (raw simulations, ML models, GA runs, graph zoos) are in `/home/labs/pilpel/matanyaw/moran-v1-data`, outside the repo.
- `rust/exact/` is Hila Friedman's exact solver (moved from `src/moran_process/numeric/`; `git log --follow rust/exact` shows its history).

## Setup

```
uv sync          # on a compute node (`inode`): compiles the Rust engine in rust/engine/
```

`from moran_graphs import PopulationGraph, simulate` gives the graph factories and the engine.
`notebooks/try_simulation.ipynb` shows single runs; `batch.ipynb` writes batch plans and reads their reports.

## Running a batch on WEXAC (LSF)

Batch data goes to `$MORAN_GRAPHS_DATA` (default `/home/labs/pilpel/matanyaw/moran-graphs-data`), never into the repo.

1. **Write a plan** in `batch.ipynb` (`make_plan`, `spec`, `describe`, `save_plan`). It saves `plans/<batch>.json`:
   graph recipes, r values, repeats, seeds and LSF settings.
2. **Submit** from the repo:
   ```
   uv run python -m moran_graphs.hpc submit plans/<batch>.json --dry-run   # builds every graph to check the plan, creates nothing (inode for large plans)
   uv run python -m moran_graphs.hpc submit plans/<batch>.json             # light, fine on the login node
   ```
   This queues a prepare job (builds graphs and the work list), then a worker array and a properties
   job side by side, then a finalize job. If prepare fails, LSF kills every job after it; read
   `<batch>/logs/prepare_*.err`. If only properties fails, the simulations still finish: rerun
   `properties <batch>` (in a job), then `finalize <batch>`.
3. **Watch:** `uv run python -m moran_graphs.hpc status <batch>` (chunks done, LSF states, rough ETA).
4. **Recover:** `uv run python -m moran_graphs.hpc resume <batch>` reruns only missing chunks and finalizes;
   `finalize <batch>` rebuilds the report by hand.
5. **Read:** `<batch>/report/` holds `verification.txt`, `speed.txt`, `results.parquet` (one row per graph and r,
   with graph properties), `histograms.parquet` and `chunk_timing.parquet`; `hpc.load_report(<batch>)` loads them.

Without LSF (on an inode), `submit plans/<batch>.json --local` runs the same chain with local processes.

Workers claim chunks of work from a shared list (an atomic file create per chunk), so jobs that finish early
pick up remaining work and a preempted job loses only its current chunk. Each chunk has its own seed, so results
do not depend on which worker ran it.

## Tests

```
uv run pytest tests         # Python: batch pipeline, seeding, hashing
cd rust/engine && cargo test   # Rust engine: Moran formula, directed graphs, plain vs null-step skipping
```

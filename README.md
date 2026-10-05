
# A Newer Moran Process Project

This project is a rewrite of my Moran Process repo. It's a library where one can create population graphs, run moran simulaiton on them and get statistics about it.

In particular I am interested in population graphs that are shaped like Respiratory organs. Could it be that Respiratory Organs have evolved to supress / slow down the fixation of mutations in a population that lives inside them?

## Setup

```
uv sync          # on a compute node (`inode`): compiles the Rust engine in rust/
```

`from moran_graphs import PopulationGraph, simulate` gives the graph factories and the engine.
`try_simulation.ipynb` shows single runs; `batch.ipynb` writes batch plans and reads their reports.

## Running a batch on WEXAC (LSF)

Batch data goes to `$MORAN_GRAPHS_DATA` (default `/home/labs/pilpel/matanyaw/moran-graphs-data`), never into the repo.

1. **Write a plan** in `batch.ipynb` (`make_plan`, `spec`, `describe`, `save_plan`). It saves `plans/<batch>.json`:
   graph recipes, r values, repeats, seeds and LSF settings.
2. **Submit** from the repo (light, fine on the login node):
   ```
   uv run python -m moran_graphs.hpc submit plans/<batch>.json --dry-run   # check, creates nothing
   uv run python -m moran_graphs.hpc submit plans/<batch>.json
   ```
   This queues a prepare job (builds graphs, properties, work list), a worker array and a finalize job.
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
cd rust && cargo test       # Rust engine: Moran formula, directed graphs, plain vs null-step skipping
```

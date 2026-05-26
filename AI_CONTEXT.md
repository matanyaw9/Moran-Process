# AI Context Prompt: Moran Process on Respiratory Graphs

This file is a single self-contained context primer for AI sessions working on this
repository. It describes the research, the science, the actual code layout, the data
flow, the HPC workflow, and the analysis pipeline. It is written to reflect the code
as it actually exists (the older `.md` docs in this repo are partly out of date; see
"Corrections to older docs" near the end).

House rule for everything you write in this repo: never use em dashes.

---

## 1. What this project is

MSc Computational Biology thesis at the Weizmann Institute of Science.
- Student: Matanya Wiener
- Supervisor: Tzachi
- Language: Python (project targets >=3.11; the active venv is 3.12), managed with `uv`.

Research question: do the topologies of respiratory organs (mammalian lung, avian lung,
fish gill) have special evolutionary properties? Specifically, do they act as amplifiers
or suppressors of natural selection compared to random graphs of similar size?

Biological motivation: population structure (such as lung architecture) can affect how a
pathogen or a cancer mutation spreads. Amplifiers make it easier for an advantageous
mutant to take over; suppressors make it harder.

---

## 2. The science

### Moran process (Birth-Death on a graph)
Two node states: 0 = wild type (fitness 1.0), 1 = mutant (fitness r). Each step:
1. Birth: pick a node to reproduce, with probability proportional to fitness.
2. Death: that reproducer picks a uniformly random neighbor to overwrite.
3. The neighbor takes the reproducer's state.

The run continues until fixation (all nodes mutant) or extinction (no mutants left).
This is the Birth-Death (Bd) rule. There is also a Death-Birth (dB) rule that gives
different fixation probabilities; this project implements Bd only.

### Key metrics
- Fixation probability (rho): fraction of trials where the mutant fixes.
  Well-mixed baseline: rho = (1 - 1/r) / (1 - 1/r^N). Neutral case r=1 gives rho = 1/N.
- Fixation time / steps: number of steps to absorption.
- Selection coefficient r: r=1 neutral, r>1 advantageous, r<1 deleterious.

### Theory anchor
Amplifiers vs suppressors are defined relative to the well-mixed baseline. Any Bd Moran
process on a regular (isothermal) graph gives the same fixation probability as well-mixed.
The star graph is the classic amplifier. Central reference: Nowak et al. 2019, "Population
structure determines the tradeoff between fixation probability and fixation time"
(Nature Communications Biology). The core finding: amplifiers buy higher fixation
probability at the cost of longer fixation time.

### Graph types studied
| Category | Description | Key params |
|---|---|---|
| Complete | Fully connected, well-mixed baseline | n_nodes |
| Cycle | Ring, known suppressor under Bd | n_nodes |
| Mammalian | Balanced binary tree (bronchi). `nx.balanced_tree` | branching_factor, depth |
| Avian | Parallel rods (parabronchi) plus Inlet, Outlet, Circuit loop | n_rods, rod_length |
| Fish | Comb: arch plus filaments plus lamellae | n_rods, rod_length |
| Random | Random connected graph (null model): random spanning tree plus extra edges | n_nodes, n_edges, seed |

The respiratory graphs are compared against random graphs with a similar edge count. The
working size is around N=31 (mammalian b=2 d=4 gives 31 nodes); avian r4 l7 has about 34
edges, so random graphs with roughly 30 to 35 edges are the matched null model. Note:
`extreme_graphs.py` adds two synthetic categories produced by mutation search,
"Accelerator" and "Decelerator".

---

## 3. Repository layout (actual)

Code is an installable package under `src/moran_process/`.

```
moran-process/
├── pyproject.toml            # package name moran_process, hatchling build
├── uv.lock
├── CLAUDE.md                 # project instructions for Claude
├── AI_CONTEXT.md             # this file
├── PROJECT_OVERVIEW.md / CODE_ARCHITECTURE.md / HPC_WORKFLOW.md /
│   RESEARCH_BACKGROUND.md / VSCODE_WEXAC_WORKFLOW.md / WORKFLOW.md   # background docs (partly stale)
├── task_list.md              # running TODO and "what I was doing" log
├── submit_main.sh            # bsub entry point for main.py
├── notebooks/                # analysis and design notebooks (run on a compute node)
├── graph_zoos/               # saved zoos (*.joblib), gitignored
├── simulation_data/          # batch outputs, gitignored; one subdir per batch
└── src/moran_process/
    ├── __init__.py           # re-exports the public API (see below)
    ├── core/
    │   ├── population_graph.py   # PopulationGraph + GRAPH_PROPS + register CLI
    │   └── graph_zoo.py          # GraphZoo collection
    ├── simulations/
    │   └── process_run.py        # ProcessRun (single Moran simulation)
    ├── pipeline/
    │   ├── process_lab.py        # ProcessLab: local study + HPC submission
    │   ├── worker_wrapper.py     # LSF array worker
    │   ├── main.py               # respiratory + random batch builder/submitter
    │   ├── run_random_graphs.py  # random-only runner (legacy/local)
    │   ├── merge_batches.py      # merge two batches into one
    │   └── extreme_graphs.py     # mutation/GA search for extreme graphs
    └── analysis/
        └── analysis_utils.py     # plotting, aggregation, batch_info helpers
```

Public API (from `moran_process/__init__.py`):
`PopulationGraph`, `GraphZoo`, `ProcessRun`, `ProcessLab`, `GRAPH_PROPS`,
`CATEGORY_COLOR_DICT`, `GRAPH_PROPERTY_COLUMNS`, `GRAPH_PROPERTY_DESCRIPTION`.

On WEXAC the package is run with `PYTHONPATH=src` and module syntax
(`python -m moran_process.pipeline.worker_wrapper ...`), as set in `submit_jobs`.

---

## 4. Core classes and APIs

### PopulationGraph (`core/population_graph.py`)
Wraps a `networkx.Graph`. Constructed as
`PopulationGraph(graph, name, category, params=None, labeled_edges=False)`.

Important: construction does NOT write any database. It computes a Weisfeiler-Lehman
hash (`self.wl_hash`) used for deduplication, and optionally labels edges. The
`metadata` property returns only `{wl_hash, graph_name}`; this is what gets merged into
result rows, so keep it cheap.

Factory classmethods:
- `complete_graph(n_nodes)`
- `cycle_graph(n_nodes)`
- `mammalian_lung_graph(branching_factor=2, depth=3)`  (name `mammalian_b{bf}_d{depth}`)
- `avian_graph(n_rods, rod_length, directed=False)`     (name `avian_r{n}_l{len}`)
- `fish_graph(n_rods, rod_length)`                       (name `fish_r{n}_l{len}`)
- `random_connected_graph(n_nodes, n_edges=None, name=None, seed=None)` uses a local
  `np.random.default_rng` and a random spanning tree plus rejection-sampled extra edges.

Other notable members:
- `calculate_graph_properties()` returns the full property dict (see GRAPH_PROPS).
  Performance guards: diameter / radius / average_shortest_path_length only for N<=500;
  betweenness uses k=50 sampling for N>100; closeness uses manual 50-node sampling for N>200.
- `mutate_graph(name=None, seed=None)` removes one edge and re-adds an edge while keeping
  the graph connected (used by the evolutionary search).
- `batch_register(graph_zoo_path, batch_dir)` classmethod: loads a zoo (list of graphs)
  and writes one `graph_props.csv` into `batch_dir`, deduplicated by WL hash. This is the
  real registration path, run as its own LSF job.
- `draw(...)`, `to_adjacency_matrix()`, `save()/load()` (pickle).
- CLI: `python -m moran_process.core.population_graph --register --batch-dir ... --graph-zoo-path ...`

`GRAPH_PROPS` (the property names computed for the database):
n_nodes, n_edges, density, diameter, avg_degree, average_clustering,
average_shortest_path_length, degree_assortativity, avg_betweenness_centrality,
max_degree, min_degree, degree_std, transitivity, radius, avg_degree_centrality,
max_degree_centrality, max_betweenness_centrality, avg_closeness_centrality,
max_closeness_centrality.

### GraphZoo (`core/graph_zoo.py`)
An ordered collection of `PopulationGraph`. Methods: `add(graph)`, `draw_all(cols=3)`,
`save(path)` / `load(path)` (pickle), plus `len`, iteration, and indexing. Note: in
practice the pipeline often serializes a plain `list[PopulationGraph]` with `joblib`
rather than a `GraphZoo` instance, and the loaders accept either.

### ProcessRun (`simulations/process_run.py`)
One Moran simulation on one graph.
`ProcessRun(population_graph, selection_coefficient=1.0, max_steps=1_000_000)`.
- `initialize_random_mutant(n_mutants=1, seed=None)` places mutant(s) at random nodes.
- `step()` does one Bd step (numpy fitness-weighted reproducer choice, random neighbor victim).
- `run(track_history=False)` returns:
  `{fixation: bool, steps: int, initial_mutants: int, selection_coeff: float, duration: float}`,
  plus `history` (mutant-count trajectory) when `track_history=True`.
Adjacency is precomputed as a list of neighbor lists for speed.

### ProcessLab (`pipeline/process_lab.py`)
Batch manager with two modes.

Local mode:
`run_comparative_study(graphs_zoo, r_values, n_repeats=100, print_time=True, output_path=None)`
runs serially, merges `graph.metadata` plus r plus the run result into one row per
simulation, and appends to `output_path` if given (`save_results` handles append).

HPC mode:
`submit_jobs(zoo_path, n_graphs, r_values, batch_name, batch_dir, n_repeats=10,
n_requested_jobs=1, queue="short", memory="2048")`. It:
1. Calls `register_graphs_job(...)` which bsubs a job running the population_graph
   `--register` CLI to write `<batch_dir>/graph_props.csv`.
2. Builds `<batch_dir>/tmp/task_manifest.csv` via `_create_task_list`, which distributes
   total simulations across `n_requested_jobs` workers as evenly as possible. Manifest
   columns: `task_id, worker_id, graph_idx, r_value, n_repeats`. A single config can be
   split across workers.
3. Submits an LSF job array `batch_<name>[1-n_requested_jobs]` that runs
   `python -m moran_process.pipeline.worker_wrapper --zoo-path ... --manifest-path ...
   --batch-dir <batch_dir>/tmp` with `PYTHONPATH=src` and single-threaded BLAS env vars.

### worker_wrapper (`pipeline/worker_wrapper.py`)
Each array task reads its index from `LSB_JOBINDEX` (or `--job-index` locally), selects
the manifest rows where `worker_id == index`, runs the simulations, and writes
`<batch_dir>/tmp/results/result_job_<index>.csv`. Result columns:
`task_id, job_id, wl_hash, graph_name, r, fixation, steps, initial_mutants, duration`.
There is no separate "steps to extinction" column yet (open TODO).

---

## 5. End-to-end pipeline and data flow

The intended workflow (see also `task_list.md` and `WORKFLOW.md`):

1. Design a zoo: `notebooks/design_zoo.ipynb`. Set a `BATCH_NAME`, build the graphs you
   want, visualize, and save a zoo (`*.joblib` / `*.pkl`).
2. Submit a batch: call `ProcessLab.submit_jobs(...)` (from the notebook or `main.py`).
   `main.py` is the canonical builder: it instantiates the respiratory graphs, generates
   many random graphs with deduplicated WL hashes, serializes them to
   `<batch_dir>/tmp/graph_zoo.joblib`, then submits.
3. LSF runs: a `register_graphs` job writes `graph_props.csv`; the worker array writes
   `tmp/results/result_job_*.csv`.
4. Aggregate raw results: `analysis_utils.aggregate_results_no_load(batch_dir)` streams
   all `result_job_*.csv` into `<batch_dir>/full_results.csv` without loading them into
   memory (uses file copy, header-skipping). This replaced the old manual glob+concat.
5. Analyze: `notebooks/experiment_analysis.ipynb` reads `full_results.csv` plus
   `graph_props.csv`, aggregates per graph (fixation probability, mean/std steps, etc.),
   merges in structural properties, and writes `<batch_dir>/graph_statistics.csv`.
6. ML: `notebooks/ml_predictors.ipynb` reads `graph_statistics.csv` and trains models.
7. Extreme graphs: `pipeline/extreme_graphs.py` and `notebooks/extreme_graphs.ipynb`
   use `mutate_graph` to evolve graphs toward extreme predicted fixation time/probability.
8. Merge batches when needed: `pipeline/merge_batches.py` concatenates two batches'
   `full_results.csv` and `graph_props.csv` and merges their zoos (dedup by WL hash).

### Batch directory layout (actual)
A batch lives directly under `simulation_data/<batch_name>/`:
```
simulation_data/<batch_name>/
├── batch_info.json          # written by create_batch_info (name, description, params)
├── graph_props.csv          # structural properties, one row per unique graph
├── full_results.csv         # aggregated raw simulation rows
├── graph_statistics.csv     # per-graph aggregated stats merged with properties (analysis output)
├── logs/                    # bsub stdout/stderr: job_%J_%I.out/.err
└── tmp/
    ├── graph_zoo.joblib     # serialized list of PopulationGraph
    ├── task_manifest.csv    # the full task table
    └── results/             # result_job_1.csv ... result_job_N.csv
```
Note: this differs from the older docs, which described `simulation_data/tmp/batch_<name>/`.
The real structure puts `tmp/` inside each batch dir.

---

## 6. HPC workflow (WEXAC, LSF scheduler)

WEXAC uses LSF; submission is via `bsub`. Wiki: https://hpcwiki.weizmann.ac.il/en/home/lsf/basic

Run a real batch (submits the job array):
```bash
uv run python -m moran_process.pipeline.main --batch-name <name>
# or
bash submit_main.sh
```

Monitor and control:
```bash
bjobs            # list jobs
bjobs -w         # wide (full names)
bjobs -l <id>    # why pending
bpeek -f <id>    # follow stdout
bkill <id>       # kill one; bkill 0 kills all
```

Queues (approximate walltimes): `short` (~30 min), `new-short` (~12 h), `medium` (48 h),
`long` (7+ days), `idle` (unlimited, preemptible), `gsla-cpu` (the lab queue used
interactively). Check with `bqueues`.

Manual single-worker test (no bsub):
```bash
uv run python -m moran_process.pipeline.worker_wrapper \
  --zoo-path simulation_data/<batch>/tmp/graph_zoo.joblib \
  --manifest-path simulation_data/<batch>/tmp/task_manifest.csv \
  --batch-dir simulation_data/<batch>/tmp \
  --job-index 1
```

### Where things run (critical)
VS Code is connected by Remote SSH to the WEXAC login node. The login node is for editing,
git, and bsub submission only. The green Run button and F5 always run on the login node.
Heavy compute must go to a compute node.
- `inode [mem] [walltime] [queue]` opens an interactive compute shell (default 8GB, 4h,
  gsla-cpu). Run scripts/notebooks heavier than about 10 s or 1 GB here.
- `ijup [mem] [walltime] [queue]` starts JupyterLab on a compute node (default 16GB, 4h);
  paste the URL into the VS Code kernel selector. Use this for analysis notebooks.
- Both helpers live in `~/.bashrc` on WEXAC.
Do not connect VS Code itself to the compute node; the session dies when the job expires.

---

## 7. Analysis and ML

### analysis_utils.py (the shared toolbox)
- Constants: `CATEGORY_COLOR_DICT` (per-category colors, including Accelerator/Decelerator),
  `GRAPH_PROPERTY_COLUMNS`, `GRAPH_PROPERTY_DESCRIPTION` (human-readable text per property).
- Aggregation: `aggregate_results_no_load(batch_dir, delete_temp=False, output_file=None)`.
- Batch metadata: `create_batch_info(...)` and `load_batch_info(batch_dir)` read/write
  `batch_info.json`.
- Plotting functions share a figure-cache pattern: each one takes `figures_dir`,
  `force_recompute`, and `batch_name`. If a cached PNG exists and `force_recompute=False`,
  it displays the PNG instead of recomputing; otherwise it renders, saves the PNG, and
  stamps the source batch name. Key plotters:
  - `plot_steps_violin(...)` uses a polars lazy scan to pull only needed columns from a
    large `full_results.csv`.
  - `plot_outcome_vs_property(...)` is the current recommended scatter-vs-property plot
    (auto violins for discrete x, vectorized jitter, correct 1/N neutral line). It
    supersedes the older `plot_hybrid_density(...)` and `plot_property_effect(...)`.
  - `plot_two_property_effect(...)` and `plot_two_property_effect_hexbin(...)` color a
    2D property plane by an outcome.

### ML pipeline (`notebooks/ml_predictors.ipynb`)
Reads `graph_statistics.csv`. Set `BATCH_NAME`, target column(s), and an r filter at the
top. Feature matrix is the structural property columns (median-imputed). Pipeline:
StandardScaler + LinearRegression for standardized coefficients, plus an XGBoost
regressor for nonlinear prediction, plus SHAP for interpretability. Models are saved to
`<batch_dir>/ml_models/{target}_{model_type}.joblib`.

### Dependencies (managed by uv)
networkx, numpy, pandas, polars, pyarrow, scipy, matplotlib, seaborn, scikit-learn,
xgboost, shap, jupyterlab, ipykernel, nbstripout. Dev: pytest. Pillow pinned <11.

---

## 8. Key conventions

- Graph names follow `{type}_{param}{val}_...` e.g. `avian_r4_l7`, `mammalian_b2_d4`,
  `random_n31_e34_s42`.
- Deduplication is by Weisfeiler-Lehman hash (`wl_hash`), which is the join key between
  results and properties.
- `PopulationGraph.metadata` is intentionally minimal (`wl_hash`, `graph_name`). Do not
  add expensive fields to it.
- Result CSVs and the aggregation path are append/stream friendly; expect large files
  (multi-GB batches), so prefer polars lazy scans or streaming over `pd.concat` of
  everything.
- Branching (solo developer): `feature/`, `refactor/`, `experiment/`, `wip/`. WEXAC is the
  source of truth; `git push origin` is backup. See `WORKFLOW.md`.
- `nbstripout` is intended to strip notebook outputs from commits.

---

## 9. Status and open work

Done: all three respiratory topologies; WL-hash dedup; LSF batch pipeline; random-graph
null model; streaming aggregation; per-graph statistics; LR + XGBoost + SHAP; figure
caching; batch_info metadata; evolutionary search for extreme graphs; a scaling study
(see `notebooks/scaling_experiment_analysis.ipynb`, batch `2026-05-20_scaling_study_3`).

Open (from `task_list.md` and PROJECT_OVERVIEW.md):
- Record steps to extinction separately, not only steps to fixation.
- Speed up the simulation (C++ / Cython / Numba on the Moran step, or multiprocessing).
- Multi-color / multi-type Moran (more than two states); likely needs a `Process` ABC.
- GNN approach to predict fixation properties from structure (fixed vs variable N).
- Justify N=31 with a size sweep showing qualitative consistency.
- Add the analytical fixation-probability reference line to plots.
- Engineering: move all plotting behind functions (largely done in analysis_utils), add a
  proper CLI (Typer/Click), add a `ProcessLab.aggregate_batch()` method, fix per-batch log
  naming, address violin-plot FutureWarning.
- Reading list: Uri Alon (network motifs); Kishony 2011 (parallel bacterial evolution).

---

## 10. How to run, quickly

```bash
uv sync                                              # install deps
uv run python -m moran_process.pipeline.main --batch-name <name>   # submit a batch (HPC)
# analysis: open notebooks/ via ijup on a compute node
```

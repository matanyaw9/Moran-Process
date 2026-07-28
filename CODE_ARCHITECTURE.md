# Code Architecture

Code is an installable package under `src/moran_process/`. Run modules with
`uv run python -m moran_process.<subpackage>.<module>` (the cluster sets `PYTHONPATH=src`).
Public API is re-exported from `moran_process/__init__.py`: `PopulationGraph`, `GraphZoo`,
`MoranProcess`, `MultiColorMoranProcess`, `ProcessLab`, `GRAPH_PROPS`, and the analysis constants.

---

## Class: `PopulationGraph` (`core/population_graph.py`)

A wrapper around a `networkx.Graph`. Central object for graph topology and properties.

### Constructor
```python
PopulationGraph(graph: nx.Graph, name: str, category: str, params: dict|None = None, labeled_edges: bool = False)
```
- `name`: e.g., `"mammalian_b2_d4"`, `"avian_r4_l7"`, `"random_n31_e34_s42"`
- `category`: `"Mammalian"`, `"Avian"`, `"Fish"`, `"Complete"`, `"Cycle"`, `"Random"`
- Construction computes `wl_hash` and (optionally) edge labels. It does NOT write any database.

### Factory Methods (use these to create graphs)
```python
PopulationGraph.complete_graph(N=31)
PopulationGraph.cycle_graph(N=31)
PopulationGraph.mammalian_lung_graph(branching_factor=2, depth=4)   # N = sum(2^i for i in 0..depth) = 31 for depth=4
PopulationGraph.avian_graph(n_rods=4, rod_length=7)                  # ~34 edges
PopulationGraph.fish_graph(n_rods=3, rod_length=3)
PopulationGraph.random_connected_graph(n_nodes=31, n_edges=34, seed=42)
```

### Key Attributes
- `self.graph`: networkx Graph object
- `self.n_nodes`: number of nodes (also `number_of_nodes()`)
- `self.wl_hash`: Weisfeiler-Lehman graph hash (str) — used for deduplication
- `self.is_directed`: bool
- `self.name`, `self.category`, `self.params`

### Key Methods
- `calculate_graph_properties()` → dict of all metrics
- `PopulationGraph.batch_register(graph_zoo_path, batch_dir)` (classmethod) → writes a deduplicated `<batch_dir>/graph_props.csv`; there is no global graph database
- `save(filepath)` / `PopulationGraph.load(filepath)` — pickle serialization for HPC
- `draw(ax, filename, descriptive)` — matplotlib visualization
- `to_adjacency_matrix()` → numpy array

### Graph Properties Computed
`GRAPH_PROPS = ['n_nodes', 'n_edges', 'density', 'diameter', 'avg_degree', 'average_clustering', 'average_shortest_path_length', 'degree_assortativity', 'avg_betweenness_centrality', 'max_degree', 'min_degree', 'degree_std', 'transitivity', 'radius', 'avg_degree_centrality', 'max_degree_centrality', 'max_betweenness_centrality', 'avg_closeness_centrality', 'max_closeness_centrality']`

**Computation guards:** Diameter/ASPL only computed for N ≤ 500; betweenness uses k=50 sampling for N > 100; closeness uses manual 50-node sampling for N > 200.

---

## Class: `MoranProcess` (`simulations/moran_process.py`)

Runs a single Moran simulation. Takes a `GraphCore` (compact CSR struct from
`PopulationGraph.to_simulation_struct()`), not a full `PopulationGraph`.

A fast, statistically-equivalent C++ drop-in with the identical interface lives
in `simulations/cpp_moran_wrapper.py` (`CppMoranProcess`, backed by the `_moran_cpp`
pybind11 extension). It is the default engine; select with `--engine {cpp,python}`.

### Constructor
```python
MoranProcess(graph_core: GraphCore, selection_coefficient=1.0, max_steps=1_000_000, seed=None)
```

### Key Methods
```python
sim.initialize_random_mutant(n_mutants=1)  # Place mutant(s) randomly
sim.step()                                  # Single Moran step (birth-death)
result = sim.run(track_history=False)       # Run to fixation/extinction
```

### `run()` Return Dict
```python
{
    "fixation": bool,         # True = mutant fixed, False = extinction
    "steps": int,             # Steps until absorption
    "initial_mutants": int,   # How many mutants at start
    "selection_coeff": float, # r value used
    "duration": float,        # Wall-clock seconds
    # "history": np.array     # Only if track_history=True
}
```
Note: this is the *in-memory* dict. The persisted Parquet schema in `raw_results.parquet`
is `task_id, job_id, wl_hash, graph_name, r, fixation, steps, duration` — it does **not**
carry `initial_mutants` or `selection_coeff` (the r value is stored in the `r` column).

### Moran Step Logic
```
fitness_map = [r if mutant, 1.0 if wild_type]
reproducer = np.random.choice(n_nodes, p=fitness/total_fitness)
victim = random.choice(adj_list[reproducer])
state[victim] = state[reproducer]
```

---

## Class: `ProcessLab` (`process_lab.py`)

Batch experiment manager with two modes: **local** and **HPC (LSF)**.

### Local Mode
```python
lab = ProcessLab()
df = lab.run_comparative_study(
    graphs_zoo,         # list of PopulationGraph
    r_values,           # e.g., [1.0, 1.1, 1.2, 2.0]
    n_repeats=1000,
    output_path='simulation_data/results.csv'  # auto-appends
)
```

### HPC Mode (WEXAC)
```python
lab.submit_jobs(
    zoo_path="simulation_data/my_batch/tmp/graph_zoo.joblib",  # joblib-serialized list[PopulationGraph]
    n_graphs=len(zoo),
    r_values=[1.0, 1.1, 1.2, 2.0],
    batch_name="my_batch",
    batch_dir="simulation_data/my_batch",
    n_repeats=10_000,
    n_requested_jobs=1000,   # number of LSF array jobs
    queue="short",
    memory="2GB",            # parsed to MB by _parse_memory_mb
    engine="cpp",            # "cpp" (default) or "python"
    zoo_config=None,         # optional dict describing how the zoo was built; recorded in batch_info.json
)
```

This creates (one directory per batch, with a nested `tmp/`):
```
simulation_data/<batch_name>/
    graph_props.csv          # structural properties (written by the register_graphs job)
    batch_info.json          # full provenance + run config
    logs/                    # bsub stdout/stderr (job_%J_%I.out / .err)
    tmp/
        graph_zoo.joblib     # joblib-serialized list[PopulationGraph]
        task_manifest.csv    # columns: task_id, worker_id, graph_idx, r_value, n_repeats, seed, local_graph_idx
        zoo_shards/          # zoo_worker_1.pkl ... zoo_worker_N.pkl (per-worker GraphCore shards)
        results/             # raw_results_job_1.parquet ... raw_results_job_N.parquet
```

Then submits: `bsub -q short -J "batch_<name>[1-1000]" ... python -u -m moran_process.pipeline.worker_lsf --zoo-shard-dir <tmp>/zoo_shards --manifest-path <tmp>/task_manifest.csv --batch-dir <tmp> --engine cpp`

### Post-Simulation Jobs
`submit_jobs` chains four dependent jobs after the array. See `HPC_WORKFLOW.md` step 3 for the DAG and the LSF conditions, and `CLAUDE.md` for the design rationale.

```python
from moran_process.pipeline.process_lab import ProcessLab
ProcessLab.submit_post_batch_jobs(
    batch_dir, batch_name,
    aggregate_job_id=None,    # chain the consumers onto this rollup; None = run now
    array_job_id=None,        # gate aggregate and job_speed on this array; None = run now
    queue="short", force=False,
)
```

One rule unifies every launch mode: **if a job id is passed the job PENDs on it; if `None` is passed no `-w` flag is added and the job runs immediately.** That is what lets the same four submitters serve a fresh batch and a finished one without branching.

| module | mem | in | out |
|---|---|---|---|
| `pipeline/aggregate_batch.py` | 16GB | shards, `graph_props.csv` | `graph_statistics.csv` |
| `pipeline/batch_verify.py` | 8GB | props, stats, `batch_info.json`, shard footers | `report/verification.{json,txt}` |
| `pipeline/cache_violin_data.py` | 8GB | shards | `cache/fixation_steps_r{r}_n{cap}.parquet` + JSON sidecar |
| `pipeline/job_speed.py` | 8GB | shards | `job_speed.csv` (`job_id,steps,duration`) |

`pipeline/post_batch.py` is the entry point:
- `classify_batch(batch_dir)` -> `CURRENT` / `LEGACY`.
- `post_batch_status(batch_dir, r_values=None)` -> per-step `DONE` / `MISSING` / `NOT_APPLICABLE`. Inspection only.
- `print_post_batch_status(status)` renders the table the notebook shows.
- `ensure_post_batch(batch_dir, force=False)` submits. The only thing that starts work.

Note `ensure_post_batch` treats "the aggregation is being rebuilt" as a reason to rerun its consumers even when their outputs exist, since those outputs are now stale by construction.

`aggregate_results_no_load(batch_dir)` still fuses per-job shards into one `raw_results.parquet`, but it is off the normal path: a fused file over 2**32-1 rows exceeds polars' u32 row index. Consumers glob `tmp/results/*.parquet`.

---

## Worker Script: `worker_lsf.py`

Run by each LSF array job. Selects the manifest rows where `worker_id == LSB_JOBINDEX`,
loads only its own `zoo_worker_<index>.pkl` shard, runs those tasks, and streams results to
`<batch>/tmp/results/raw_results_job_<index>.parquet` (one row-group per task). Run it as a module.
```bash
# Manual test (local): pass --job-index explicitly instead of LSB_JOBINDEX
PYTHONPATH=src python -m moran_process.pipeline.worker_lsf \
    --zoo-shard-dir simulation_data/<BATCH>/tmp/zoo_shards \
    --manifest-path simulation_data/<BATCH>/tmp/task_manifest.csv \
    --batch-dir simulation_data/<BATCH>/tmp \
    --job-index 1 [--engine cpp]
# On cluster (automatic): LSF sets LSB_JOBINDEX, so --job-index is omitted.
```

---

## Experiment Scripts

### `main.py`
Runs the respiratory graphs (mammalian b2 d4, avian r7 l4, avian r4 l7, fish r3 l3) plus random graphs via HPC submission, serializing the zoo to `<batch>/tmp/graph_zoo.joblib` first.
Current config: `n_nodes=range(29,34)`, `edge_range=5`, `n_random_graphs_per_combination=500`, `r_values=[1.1]`, `n_repeats=10_000`, `n_jobs=1000`, `GRAPH_ZOO_SEED=42`.

---

## Analysis Notebooks (`notebooks/`)

| Notebook | Purpose |
|---|---|
| `experiment_analysis.ipynb` | Main analysis: distributions, per-property plots against fixation probability and time. Reads only finished files |
| `ml_predictors.ipynb` | ML training and evaluation: LR + XGBoost across all targets in one run, SHAP, cross-model summary |
| `extreme_graphs.ipynb` | Analysis of graphs generated to maximise/minimise model predictions |
| `design_zoo.ipynb` | Graph zoo design and inspection |

**Workflow:** the post-simulation jobs produce `graph_statistics.csv`; `experiment_analysis.ipynb` and then `ml_predictors.ipynb` both read it. The notebook no longer builds it. Its first cell after the config prints the post-batch status table and, if anything is missing, the exact `ensure_post_batch(...)` call.

**Builder/reader split.** Every expensive artefact has a builder that only jobs call and a reader that only notebooks, streamlit, and ML call:

| artefact | builder (jobs only) | reader (consumers) |
|---|---|---|
| `graph_statistics.csv` | `io.build_graph_statistics` | `io.load_graph_statistics` |
| violin sample | `io.compute_fixation_steps_by_category` | `io.load_fixation_steps_by_category` |

The readers raise rather than falling back to building, naming the missing file and the CLI that produces it. Both were previously single "compute if missing, else load" functions, which meant the same call was either a 20ms file read or an unannounced hour of compute inside a Jupyter kernel depending on invisible state. `plot_steps_violin` and `plot_steps_pvalue_matrix` take `batch_dir` accordingly, not `results_path` plus `cache_dir`.

Stale notebooks (`notebooks/tmp_*`, `scaling_experiment_analysis.ipynb`) were left alone: they already called the plots with a `results_csv_path=` parameter that had not existed for some time, so they were broken before this change.

---

## Streamlit Dashboard (`streamlit_app.py`)

Interactive browser over a single batch's results (`uv run streamlit run streamlit_app.py`). Pick a batch and an `r`; three pages share the same loaded data:

| Page | Contents |
|---|---|
| Overview | Batch info card and run speed / resource report |
| Fixation time | Steps-to-fixation violins, pairwise Mann-Whitney significance matrix, per-metric histograms |
| Property effects | Outcome vs. a single structural property, and the combined two-property view |

**On-disk figure cache (`cached_figure`).** This is a **streamlit-only** feature. Every plot in the dashboard is served from a cached PNG under `simulation_data/<batch>/figures/`, built with the same `_resolve_figure_path` the `plot_*` functions use to decide where to save. The notebooks no longer read these PNGs: their equivalent cache was removed once every heavy figure input became a job-built file read, so the draw itself is all that remains (7.4s for the heaviest violin, 0.5s for a scatter) and each `plot_*` call now prints its own build time instead. In streamlit the cache stays worth having because a dashboard redraws on every widget interaction, and because it is under explicit user control rather than silent. The cache has no code-version awareness: a figure that gained content from a code change still shows the old PNG until rebuilt. Each plot exposes **Regenerate** (rebuild from raw data, show unsaved) and **Save / Overwrite** (persist the displayed bytes). `cache_key` carries everything that changes the figure (`r`, selected columns, style) so distinct selections map to distinct files. The freshly built PNG bytes are stashed in `session_state`, so Save writes instantly instead of re-running a multi-GB scan.

**Combined two-property view.** X, Y, and Color can each show *either* a structural trait *or* a simulation result; the selectbox groups them with a `Property ·` / `Result ·` label prefix. When X is `n_nodes` and Y is `prob_fixation`, the plot overlays analytic reference lines: the neutral `1/N` baseline (always) and the complete-graph Moran curve `rho(N, r)` (only when the data is a single `r`). Both lines are drawn in the shared tail `_finish_two_property_figure`, so scatter and hexbin styles get them identically.

**Analytic curve (`basic_moran_fixation_prob`, `analysis_utils/plots.py`).** `rho(N, r) = (1 - 1/r) / (1 - 1/r^N)`, vectorized over `N`. At `r == 1` it is 0/0 -> nan; that neutral limit is exactly `1/N`, drawn as the separate baseline.

---

## ML Pipeline (`notebooks/ml_predictors.ipynb`)

1. **Configuration:** set `BATCH_NAME`, `TARGET_COLUMNS` (list), and `R_FILTER` at the top of the notebook.
2. **Feature matrix:** graph property columns from `graph_statistics.csv`, median-imputed for NaNs. If `R_FILTER=None`, `r` is included as a feature.
3. **Train/test split:** computed once on `X`; all targets share the same split (`random_state=42`).
4. **Linear Regression:** `sklearn.pipeline.Pipeline([StandardScaler(), LinearRegression()])` — standardized coefficients give % contribution.
5. **XGBoost:** `xgboost.XGBRegressor` for non-linear prediction.
6. **SHAP:** `shap.LinearExplainer` for LR; `shap.PermutationExplainer` for XGBoost.
7. **Model saving:** saved to `{BATCH_DIR}/ml_models/{target}_{model_type}.joblib`. Loaded dynamically via `MODEL_TYPE_MAP` — adding a new model type requires only updating that dict.
8. **Cross-model summary:** final section loads all saved models and plots predicted vs true in a dynamic grid (rows = targets, cols = model types).

---

## Dependencies

Managed with `uv`. Key packages:
- `networkx` — graph generation and metrics
- `numpy`, `pandas` — data
- `matplotlib`, `seaborn` — visualization
- `scikit-learn` — linear regression pipeline, StandardScaler
- `xgboost` — non-linear model
- `shap` — model interpretability
- `jupyter` — notebooks

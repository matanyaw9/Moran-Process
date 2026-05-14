# Code Architecture

All code lives in `Moran-Process/`. Run with `uv run <script.py>` or activate `.venv`.

---

## Class: `PopulationGraph` (`population_graph.py`)

A wrapper around a `networkx.Graph`. Central object for graph topology and the graph database.

### Constructor
```python
PopulationGraph(graph: nx.Graph, name: str, category: str, params: dict|None, register_in_graph_props=True)
```
- `name`: e.g., `"mammalian_b2_d4"`, `"avian_r4_l7"`, `"random_n31_e34_5"`
- `category`: `"Mammalian"`, `"Avian"`, `"Fish"`, `"Complete"`, `"Cycle"`, `"Random"`
- `register_in_graph_props=True`: auto-registers in `simulation_data/graph_database.csv`

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
- `self.N`: number of nodes
- `self.wl_hash`: Weisfeiler-Lehman graph hash (str) — used for deduplication
- `self.is_directed`: bool
- `self.name`, `self.category`, `self.params`

### Key Methods
- `calculate_graph_properties()` → dict of all metrics
- `get_database()` → DataFrame of all registered graphs
- `save(filepath)` / `PopulationGraph.load(filepath)` — pickle serialization for HPC
- `draw(ax, filename, descriptive)` — matplotlib visualization
- `to_adjacency_matrix()` → numpy array

### Graph Properties Computed
`GRAPH_PROPS = ['n_nodes', 'n_edges', 'density', 'diameter', 'avg_degree', 'average_clustering', 'average_shortest_path_length', 'degree_assortativity', 'avg_betweenness_centrality', 'max_degree', 'min_degree', 'degree_std', 'transitivity', 'radius', 'avg_degree_centrality', 'max_degree_centrality', 'max_betweenness_centrality', 'avg_closeness_centrality', 'max_closeness_centrality']`

**Computation guards:** Diameter/ASPL only computed for N ≤ 500; betweenness uses k=50 sampling for N > 100; closeness uses manual 50-node sampling for N > 200.

---

## Class: `ProcessRun` (`process_run.py`)

Runs a single Moran simulation on a `PopulationGraph`.

### Constructor
```python
ProcessRun(population_graph: PopulationGraph, selection_coefficient=1.0, max_steps=1_000_000)
```

### Key Methods
```python
sim.initialize_random_mutant(n_mutants=1, seed=None)  # Place mutant(s) randomly
sim.step()                                              # Single Moran step (birth-death)
result = sim.run(track_history=False)                  # Run to fixation/extinction
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
    graphs_zoo,
    r_values,
    n_repeats=10_000,
    n_jobs=1000,          # number of LSF array jobs
    queue="short",        # LSF queue
    memory="2048",        # MB per job
    output_dir="simulation_data/tmp",
    batch_name="my_batch"  # or auto datetime
)
```

This creates:
```
simulation_data/tmp/batch_<name>/
    graphs.pkl           # pickled list of PopulationGraph objects
    task_manifest.csv    # columns: task_id, graph_idx, r, repeat
    logs/                # bsub stdout/stderr (job_%J_%I.out)
    results/             # result_job_1.csv ... result_job_N.csv
```

Then submits: `bsub -q short -J "batch_name[1-1000]" ... uv run worker_wrapper.py --batch-dir <path> --chunk-size <N>`

### Aggregating Results After Jobs Complete
The results are individual CSVs in `batch_dir/results/`. Aggregate manually:
```python
import glob, pandas as pd
files = glob.glob('simulation_data/tmp/batch_NAME/results/result_job_*.csv')
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df.to_csv('simulation_data/combined_results.csv', index=False)
```
(No built-in aggregation method yet — this is a known TODO.)

---

## Worker Script: `worker_wrapper.py`

Run by each LSF array job. Gets its chunk via `LSB_JOBINDEX` env var.
```bash
# Manual test (local):
python worker_wrapper.py --batch-dir simulation_data/tmp/batch_NAME --chunk-size 1000 --job-index 1
# On cluster (automatic):
bsub ... uv run worker_wrapper.py --batch-dir ... --chunk-size ...
# LSB_JOBINDEX is set automatically by LSF
```

---

## Experiment Scripts

### `main.py`
Runs the 3 respiratory graphs + random graphs via HPC submission.
Current config: `n_nodes=31`, `edge_counts=range(30,35)`, `n_graphs_per_edge_count=50`, `r_values=[1.0,1.1,1.2,1.3,2]`, `n_repeats=10_000`, `n_jobs=1000`.

### `run_random_graphs.py`
Runs random graphs only, locally (not HPC). Smaller scale.

---

## Analysis Notebooks (`analysis/`)

| Notebook | Purpose |
|---|---|
| `df_analysis.ipynb` | Load graph database + experiment results; fixation probability plots; median steps; histograms |
| `compare_random_vs_respiratory.ipynb` | Comparative analysis: random vs respiratory graphs; evolutionary trade-off maps |
| `experiment_analysis.ipynb` | General experiment analysis |
| `analyse_tests.ipynb` | Quick analysis on test batches |

**Data loading pattern:**
```python
from analysis_utils import setup_analysis_environment, load_all_data
setup_analysis_environment()  # sets working dir to project root
data = load_all_data()
df_all = data['all_experiments']  # combined DataFrame
```

---

## ML Pipeline (implemented in notebooks)

1. **Feature matrix:** Graph properties from `graph_database.csv` joined to experiment results on `wl_hash` or `graph_name`.
2. **Target:** Mean fixation time (steps) per graph per r value.
3. **Linear Regression:** `sklearn.pipeline.Pipeline([StandardScaler(), LinearRegression()])` — standardized coefficients give % contribution to fixation time.
4. **XGBoost:** `xgboost.XGBRegressor` for non-linear prediction.
5. **SHAP:** `shap.PermutationExplainer` on XGBoost model to identify true topological drivers.

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

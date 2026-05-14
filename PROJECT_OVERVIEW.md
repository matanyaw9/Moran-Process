# Evolutionary Graph Theory Project — Master Context

## What This Project Is

**MSc thesis in Computational Biology** at the Weizmann Institute of Science.
**Student:** Matanya Wiener (matanya.wiener@weizmann.ac.il)
**Supervisor:** Tzachi

**Research Question:** Do the topologies of respiratory organs (mammalian, avian, fish lungs) have special evolutionary properties — specifically, do they act as amplifiers or suppressors of natural selection compared to random graphs of similar size?

**Biological Motivation:** Understanding how population structure (e.g., lung architecture) affects the spread of pathogens or the fixation of cancer mutations. If respiratory graphs are amplifiers, they make it easier for advantageous pathogens/cancer cells to take over; suppressors make it harder.

---

## The Core Science

### Moran Process (Birth-Death)
The simulation model. At each step:
1. Select a node to reproduce, weighted by fitness `r` (mutant) vs `1.0` (wild type).
2. Select a random neighbor of the reproducer to be replaced ("die").
3. The neighbor takes the reproducer's type.

This continues until **fixation** (mutant takes over all N nodes) or **extinction** (mutant disappears).

### Key Metrics
- **Fixation Probability (ρ):** Fraction of trials where the mutant fixes. Theoretical baseline for well-mixed (complete graph): `ρ = (1 - 1/r) / (1 - 1/r^N)`.
- **Fixation Time:** Number of steps until absorption (fixation or extinction).
- **Selection coefficient r:** Fitness of mutant relative to wild type. `r=1` neutral, `r>1` advantageous, `r<1` deleterious.

### Key Literature Finding
From Nowak et al. (Nature Comms Bio, 2019): All known **amplifiers** (structures increasing fixation probability) achieve this at the cost of **longer fixation times**. There is a fundamental trade-off. Accelerators decrease fixation time but typically at the cost of lower fixation probability.

### Graph Types Studied
| Category | Description | Key Parameters |
|---|---|---|
| Complete | Fully connected (well-mixed baseline) | N nodes |
| Cycle | Ring graph | N nodes |
| Mammalian | Balanced binary tree (airways/bronchi) | branching_factor, depth |
| Avian | Parallel rods (parabronchi) with inlet/outlet/circuit loop | n_rods, rod_length |
| Fish | Comb structure (gill arch + lamellae) | n_rods, rod_length |
| Random | Random connected graph (null model) | n_nodes, n_edges |

The respiratory graphs are compared against random graphs with a **similar number of edges** (especially ~34 edges, like the avian model).

---

## Working Environment

| Location | Purpose |
|---|---|
| Windows PC (office) | Code development, analysis notebooks, submitting jobs |
| WEXAC HPC (Linux) | Running batch simulations via LSF job arrays |
| OneDrive sync | Syncs code between office PC and potentially other machines |

**The code lives on the Windows PC and is synced/uploaded to WEXAC for job submission.**

---

## Directory Structure

```
Evolutionary_Graph_Theory_Project/
├── Moran-Process/                  # Main code directory
│   ├── population_graph.py         # Graph definitions + database
│   ├── process_run.py              # Single simulation engine
│   ├── process_lab.py              # Batch runner + HPC submission
│   ├── worker_wrapper.py           # HPC worker script (run by bsub)
│   ├── main.py                     # Respiratory graph experiment runner
│   ├── run_random_graphs.py        # Random graph experiment runner
│   ├── gemini_script.py            # Early prototype (fractal lung, different approach)
│   ├── analysis/
│   │   ├── df_analysis.ipynb
│   │   ├── compare_random_vs_respiratory.ipynb
│   │   ├── experiment_analysis.ipynb
│   │   ├── analyse_tests.ipynb
│   │   └── analysis_utils.py
│   ├── tests/
│   │   ├── run_random_graphs_test.py
│   │   ├── test_csv_append.py
│   │   ├── example_usage.py
│   │   └── testing_scripts.ipynb
│   └── simulation_data/            # CSVs of results and graph database
│       ├── graph_database.csv
│       ├── respiratory_runs.csv
│       └── random_graphs_*.csv
└── TasksVault/                     # Obsidian notes vault
    ├── Context Prompt.md           # Short context summary
    ├── PROJECT_OVERVIEW.md         # THIS FILE
    ├── CODE_ARCHITECTURE.md        # Detailed code docs
    ├── HPC_WORKFLOW.md             # WEXAC/LSF how-to
    ├── RESEARCH_BACKGROUND.md      # Biology + math background
    ├── WEXAC Cheat Sheet.md        # LSF commands reference
    ├── Weekly Road Map.md          # Weekly task tracking
    └── Project Road Map.md         # High-level roadmap
```

---

## Current Status (as of ~Week 8, early 2026)

### Completed
- All three respiratory graph topologies implemented (mammalian, avian, fish)
- Graph database with WL hashing for isomorphism detection and deduplication
- HPC batch submission pipeline (LSF job arrays via `bsub`)
- Random graph generation for comparison (null model)
- Linear regression with standardized coefficients for feature importance
- XGBoost regressor for fixation time prediction
- SHAP (PermutationExplainer) for model interpretability
- Simulated annealing to generate graphs that extremize specific topological properties
- Comparative analysis notebooks

### In Progress / Next Steps
- [ ] Count steps to extinction (not only steps to fixation)
- [ ] Make analysis pipeline work for large batches (>10 GB)
- [ ] Make simulation faster (C++? Multiprocessing?)
- [ ] Multi-color/multi-type simulation (more than 2 types)
- [ ] Explore GNN approach (Yael's suggestion: train on fixed-size or variable-size graphs)
- [ ] Justify/minimize graph size (is N=31 meaningful? Can we show results are consistent across sizes?)
- [ ] Add fixation probability formula to plot axes
- [ ] Read Uri Alon's paper on network motifs
- [ ] Read Roy Kishony's paper on parallel bacterial evolution

---

## Key Design Decisions

1. **WL Graph Hashing:** Each graph gets a Weisfeiler-Lehman hash on creation. This detects topologically equivalent (isomorphic) graphs and avoids re-registering them in the database.
2. **Graph Database:** `simulation_data/graph_database.csv` stores all graph topologies with their computed properties. New graphs auto-register on instantiation (unless `register_in_db=False`).
3. **HPC Parallelism:** Jobs are submitted as LSF job arrays. A `task_manifest.csv` enumerates all (graph, r, repeat) combinations; each worker processes a contiguous chunk indexed by `LSB_JOBINDEX`.
4. **Serialization:** Graphs are pickled to `graphs.pkl` in the batch directory; workers unpickle and run their slice.
5. **Metric cutoffs:** Expensive graph metrics (diameter, ASPL, betweenness) are only computed for N ≤ 100–500 to avoid freezing.

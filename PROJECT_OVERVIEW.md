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
moran-process/
├── pyproject.toml                  # package moran_process (hatchling), deps via uv
├── uv.lock
├── AI_CONTEXT.md                   # comprehensive up-to-date context primer
├── CLAUDE.md                       # project instructions for Claude
├── PROJECT_OVERVIEW.md             # THIS FILE
├── CODE_ARCHITECTURE.md            # class API and pipeline
├── HPC_WORKFLOW.md                 # WEXAC/LSF how-to
├── RESEARCH_BACKGROUND.md          # biology + math background
├── VSCODE_WEXAC_WORKFLOW.md        # where code runs (login vs compute node)
├── WORKFLOW.md                     # solo-dev branching/run habits
├── task_list.md                    # running TODO + work log
├── submit_main.sh                  # bsub entry point for main.py
├── notebooks/                      # design + analysis notebooks (run on compute node)
├── graph_zoos/                     # saved zoos (*.joblib), gitignored
├── simulation_data/                # batch outputs, gitignored; one subdir per batch
├── tests/                          # AI-generated, untrusted, do not run
└── src/moran_process/
    ├── __init__.py                 # public API re-exports
    ├── core/
    │   ├── population_graph.py          # PopulationGraph + GRAPH_PROPS + register CLI
    │   ├── graph_core.py                # GraphCore: compact CSR struct for the hot loop
    │   └── graph_zoo.py                 # GraphZoo collection
    ├── simulations/
    │   ├── simulation_process.py        # SimulationProcess ABC
    │   ├── moran_process.py             # MoranProcess (pure-Python reference)
    │   ├── cpp_moran_wrapper.py         # CppMoranProcess (fast C++ drop-in, default)
    │   ├── multi_color_moran_process.py # MultiColorMoranProcess
    │   └── _cpp/moran_core.cpp          # pybind11 extension -> _moran_cpp
    ├── pipeline/
    │   ├── process_lab.py               # local study + HPC submission
    │   ├── worker_lsf.py                # LSF array worker
    │   ├── main.py                      # respiratory + random batch builder
    │   ├── post_batch.py                # post-simulation status + submission
    │   ├── aggregate_batch.py           # post-sim job 1: shards -> graph_statistics.csv
    │   ├── batch_verify.py              # post-sim job 2: completeness + failure verdict
    │   ├── cache_violin_data.py         # post-sim job 3: bounded fixation-step sample
    │   ├── job_speed.py                 # post-sim job 4: per-job steps/duration
    │   ├── combine_batches.py           # union two batches into one
    │   └── extreme_graphs.py            # mutation/GA search for extreme graphs
    └── analysis/
        ├── analysis_utils/              # package: io, plots, colors, theory, provenance
        └── batch_speed_report.py        # engine/worker speed-comparison report
```

A single batch lives at `simulation_data/<batch_name>/` and contains `graph_props.csv`,
`graph_statistics.csv`, `job_speed.csv`, `batch_info.json`, `report/`, `cache/`,
`figures/`, `logs/`, and a `tmp/` holding `graph_zoo.joblib`, `task_manifest.csv`,
`zoo_shards/zoo_worker_*.pkl`, and `results/raw_results_job_*.parquet`. Everything above
`tmp/` is produced by the post-simulation jobs and is what the analysis notebooks read.

---

## Current Status (mid 2026)

### Completed
- All three respiratory graph topologies (mammalian, avian, fish)
- WL hashing for isomorphism detection and deduplication
- HPC batch submission pipeline (LSF job arrays via `bsub`)
- Random graph generation for comparison (null model)
- Streaming aggregation of per-job results into `raw_results.parquet`
- Per-graph statistics (`graph_statistics.csv`) merged with structural properties
- Linear regression with standardized coefficients for feature importance
- XGBoost regressor for fixation time prediction, with SHAP interpretability
- Evolutionary/mutation search for graphs that extremize fixation time/probability
- Figure caching and per-batch metadata (`batch_info.json`)
- A scaling study across N (see `notebooks/scaling_experiment_analysis.ipynb`)
- Per-worker zoo shards + CSR `GraphCore` struct (eliminates NetworkX from workers; ~10x RAM reduction)
- Parquet output with streaming results (constant-memory workers; ~5x smaller files vs CSV)
- `SimulationProcess` ABC with `MoranProcess` and `MultiColorMoranProcess` subclasses
- Incremental mutant-count tracking; unified per-instance RNG with optional batch seed for reproducibility
- Polars streaming aggregation in `analysis_utils.io.build_graph_statistics`
- C++ simulation engine (`CppMoranProcess`), statistically validated against the Python
  reference; roughly 300x to 1800x faster
- Batch combination (`combine_batches.py`): union two batches into one analysable batch
- Post-simulation jobs: aggregate, verify, violin cache, and job speed, chained
  automatically after the array so `experiment_analysis.ipynb` runs on any batch with no
  compute in the kernel and none on the login node

### In Progress / Next Steps
- [ ] Record steps to extinction separately, not only steps to fixation
- [x] Make the simulation faster -- C++ engine via pybind11, default since June 2026
- [ ] Write real tests; the existing `tests/` are AI-generated, outdated, and untrusted
- [x] Multi-color/multi-type simulation -- `MultiColorMoranProcess` subclasses `SimulationProcess` ABC
- [ ] Explore a GNN approach (fixed-size vs variable-size graphs)
- [ ] Justify N=31 with a size sweep showing qualitative consistency
- [ ] Add the analytical fixation-probability reference line to plots
- [ ] Read Uri Alon (network motifs) and Kishony 2011 (parallel bacterial evolution)

---

## Key Design Decisions

1. **WL Graph Hashing:** Each graph gets a Weisfeiler-Lehman hash on creation. This detects topologically equivalent (isomorphic) graphs and is the dedup/join key (`wl_hash`).
2. **Per-batch properties:** There is no global graph database. `PopulationGraph.batch_register` computes properties once per unique graph and writes them to `<batch>/graph_props.csv`. Construction itself has no database side effect.
3. **HPC Parallelism:** Jobs are submitted as LSF job arrays. A `task_manifest.csv` enumerates the work and assigns each row a `worker_id`; each worker processes the rows whose `worker_id` matches its `LSB_JOBINDEX`.
4. **Serialization:** The graph zoo is serialized with joblib to `<batch>/tmp/graph_zoo.joblib`; workers load it and run their assigned rows.
5. **Metric cutoffs:** Expensive metrics (diameter, ASPL, betweenness, closeness) are only computed below N thresholds (roughly 100 to 500) to avoid freezing.
6. **Builder/reader split:** Every expensive artefact has a builder that only compute jobs call and a reader that only notebooks call, and the reader raises rather than silently building. The two used to be single "compute if missing, else load" functions, which meant one call site was either a 20ms file read or an unannounced hour of compute inside a Jupyter kernel depending on invisible state. All heavy lifting belongs in the post-simulation jobs, on compute nodes.
7. **Shards are never fused:** Consumers scan `tmp/results/*.parquet` as a glob. Polars indexes rows with a u32 and cannot read a single Parquet file over 2**32-1 rows, and the 100K-reps batch is 7.2e9. Every statistic the rollup writes is an additive per-shard partial, so it is a bounded-memory reduction; order statistics do not decompose and are opt-in.

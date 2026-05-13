# Research Background

## Biological Motivation

This project asks: **does the physical structure of a respiratory organ affect how quickly a pathogen or cancer mutation can take over?**

Respiratory organs evolved under very different constraints in different vertebrates:
- **Mammalian lungs:** Tree-like bifurcating bronchi (fractal-like, dead-end alveoli)
- **Avian lungs:** Parallel parabronchi with through-flow (continuous unidirectional flow)
- **Fish gills:** Comb-like lamellae attached to gill arches (counter-current exchange)

These structures are modeled as graphs, and the Moran process is run on them to measure how topology shapes evolutionary dynamics.

---

## Mathematical Framework

### Moran Process (Birth-Death on Graphs)
At each discrete time step:
1. **Birth:** Choose a node to reproduce, with probability proportional to fitness. Mutant fitness = `r`, wild-type fitness = `1`.
2. **Death:** The reproducer picks a random **neighbor** to replace.
3. The replaced node takes the reproducer's type.

This is the **Birth-Death (Bd)** Moran process on graphs. Note: there is also **Death-Birth (dB)** — random death first, then neighbors compete to reproduce. The two give **different fixation probabilities** on the same graph.

### Fixation Probability
For a well-mixed population of size N with a single mutant of fitness r:

$$\rho_{WM} = \frac{1 - 1/r}{1 - 1/r^N}$$

For neutral mutation (r=1): ρ = 1/N.

For graph-structured populations, ρ depends on topology. A **Bd Moran process on any regular graph** gives the same fixation probability as the well-mixed case — regular graphs are "isothermal" in the Bd rule.

### Amplifiers vs Suppressors
- **Amplifier:** ρ_graph > ρ_WM for all r > 1. Makes it easier for advantageous mutants to fix.
- **Suppressor:** ρ_graph < ρ_WM for all r > 1. Makes it harder.
- **Star graph:** Classic amplifier under Bd rule. ρ_star → 1 for large N and r > 1.

### The Amplifier Trade-off (Nowak et al. 2019)
Key result: Any population structure that amplifies selection (amplifier) must also **slow down** fixation. Specifically, the absorption time on any amplifier is asymptotically at least as large as the well-mixed case. There is a fundamental trade-off between fixation probability and fixation time.

---

## Graph Topologies in Code

### Mammalian Lung (`mammalian_lung_graph`)
```
PopulationGraph.mammalian_lung_graph(branching_factor=2, depth=4)
```
- Implemented as `nx.balanced_tree(branching_factor, depth)`
- For b=2, d=4: N = 2^5 - 1 = 31 nodes
- Custom position layout for visualization (hierarchical top-down)

### Avian Lung (`avian_graph`)
```
PopulationGraph.avian_graph(n_rods=4, rod_length=7)
```
- `n_rods` parallel "parabronchi" (linear paths)
- An "Inlet" node connects to all rod starts, "Outlet" to all rod ends
- A "Circuit" node connects Outlet back to Inlet (models recurrent airflow)
- Total nodes: n_rods × rod_length + 3 (Inlet, Outlet, Circuit)
- Can be directed (`directed=True`) for true unidirectional flow

### Fish Gill (`fish_graph`)
```
PopulationGraph.fish_graph(n_rods=3, rod_length=3)
```
- A vertical "main arch" of n_rods nodes
- Each arch node has a horizontal "filament" of rod_length nodes (comb tooth)
- Each filament node connects to upper and lower "lamella" nodes (3-leaf structure)
- Models the gill arch + primary filaments + secondary lamellae

---

## Comparison Strategy

The respiratory graphs are compared against:
1. **Complete graph** (well-mixed baseline)
2. **Cycle graph** (chain-like, known suppressor under Bd)
3. **Random connected graphs** with similar `n_edges` (null model)

Random graphs are generated with a random spanning tree + additional random edges.

Key controlled variable: **edge count**. The avian model (`n_rods=4, rod_length=7`) has ~34 edges, so random graphs with 30–35 edges are the null model.

---

## Graph Metrics Used as Features

For ML models predicting fixation time from topology:
- `density` — edges / max_possible_edges
- `diameter` — longest shortest path
- `avg_degree`, `max_degree`, `min_degree`, `degree_std`
- `average_clustering` — local clustering coefficient
- `average_shortest_path_length`
- `degree_assortativity` — do high-degree nodes connect to high-degree nodes?
- `avg_betweenness_centrality`, `max_betweenness_centrality`
- `avg_closeness_centrality`, `max_closeness_centrality`
- `transitivity` — global clustering coefficient
- `radius`

---

## Key Papers

1. **Nowak et al. (2019)** — "Population structure determines the tradeoff between fixation probability and fixation time." *Nature Communications Biology.* https://www.nature.com/articles/s42003-019-0373-y — **Central reference**

2. **Kishony lab (2011)** — "Parallel bacterial evolution within multiple patients identifies candidate pathogenicity genes." *Nature Genetics.* — On to-read list.

3. **Uri Alon** — Network motifs paper. On to-read list.

---

## Open Biological Questions

- Are respiratory graphs amplifiers or suppressors? (Main thesis question)
- Which topological feature best predicts fixation time? (SHAP analysis)
- Does the answer change with r value?
- Can we construct a graph that maximizes suppression while remaining "breathable"? (Simulated annealing angle)
- What is the "breathability" constraint? Not all graphs are valid respiratory models.
- Is a GNN approach viable for predicting fixation properties from structure?

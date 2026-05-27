# Optimization Notes for the Moran-Process Pipeline

Audience: a future AI/engineering session tasked with fixing the memory and
performance issues that made batch `2026-05-26_scaling_study_4` fail.

Each section starts with **What** (the change), **Where** (file/lines),
**Why** (the reasoning) and **Expected impact** so you can prioritise.

---

## 0. TL;DR — Does N=100 / 80% density justify >16 GB? No.

A single 100-node, 80%-density graph stored as CSR uses ~32 KB; as a Python
list-of-lists adjacency, ~70 KB. The Moran-process state array is 800 bytes.
One running simulation, including transient `np.where` allocations, fits
comfortably in <5 MB. The entire 50,030-graph zoo at CSR resolution would be
~250 MB.

The 16 GB OOM is **not caused by the simulation's working set**. It is caused
by four compounding pipeline problems:

1. The whole zoo (1.78 GB on disk, 3-5 GB in NetworkX form after unpickling) is
   broadcast to every one of 1000 workers, even though each worker only needs
   ~50 graphs.
2. A fresh `MoranProcess` object — and a fresh Python list-of-lists `adj_list`
   — is constructed inside the inner `for rep in range(n_repeats)` loop, so
   500,300 short-lived allocations per worker churn the small-object allocator.
3. `results_buffer` is a Python list of 500,300 dicts kept in memory until the
   very end of the worker (it should be streamed to disk).
4. `MoranProcess.step()` allocates two fresh NumPy arrays (`fitness`, `probs`)
   every single Moran step and calls `np.random.choice(p=probs)`, which
   internally renormalises and copies. With r=0.1 most sims die in ~30 steps,
   so this isn't the dominant cost in this particular failing batch, but it
   *will* dominate later batches with r near 1.

Confirmation from `simulation_data/2026-05-26_scaling_study_4/logs`:
- `TERM_MEMLIMIT` in 578 logs already (and many more pending).
- `Max Memory: 16384 MB` in every failure.
- `Average Memory: ~5-9 GB` — workers cross from baseline ~5 GB to the 16 GB
  cap within ~2 minutes. Baseline ~5 GB is the deserialised zoo.
- Zero result CSVs produced (`tmp/results/` is empty).

---

## 1. Memory: stop shipping the whole zoo to every worker  *(HIGHEST IMPACT)*

**Where.** `src/moran_process/pipeline/process_lab.py` (`submit_jobs`,
`_create_task_list`) and `src/moran_process/pipeline/worker_wrapper.py`
(`load_data`).

**What.** The task manifest already partitions tasks by `worker_id` and each
worker only touches a tiny contiguous range of `graph_idx` (worker 1 touches
graph_idx 0..50 — i.e. 51 of the 50,030 graphs). So:

1. After creating `task_manifest.csv`, compute the set of `graph_idx` each
   worker will actually need.
2. Write **per-worker zoo shards** to `tmp/zoo_shards/zoo_worker_{i}.pkl` — each
   shard holds only the graphs that worker `i` references, re-indexed locally.
3. Rewrite each worker's manifest rows to use the local index. Drop the global
   `--zoo-path` argument and pass `--zoo-shard-path` constructed from
   `LSB_JOBINDEX` (e.g. `tmp/zoo_shards/zoo_worker_${LSB_JOBINDEX}.pkl`).

For the failing batch: each shard is ~50 graphs at ~30 KB CSR each = ~1.5 MB.
Even with NetworkX overhead, ~30 MB. Worker startup time also drops because
joblib no longer has to inflate a 1.78 GB file.

**Why.** Worker 1 currently loads 50,030 graphs but uses 51. That is a 1000x
waste. The shards approach is trivially correct: tasks per worker are already
known at submit time.

**Expected impact.** Baseline RAM drops from ~5 GB to <100 MB. Loading time
drops from ~30 s to <1 s. This single change is probably enough to fit the
current batch in 4 GB instead of 16 GB.

---

## 2. Memory: convert graphs to a compact CSR form before pickling

**Where.** `src/moran_process/core/population_graph.py` (add a method
`to_simulation_struct()`) and `src/moran_process/simulations/`
(make `SimulationProcess` accept the struct directly).

**What.** A worker needs only:
- `n_nodes` (int)
- `nbrs` (int32 array of concatenated neighbour lists, length 2E)
- `offsets` (int32 array, length N+1)
- `wl_hash` (str)
- `name` (str)

Define a small struct/namedtuple/dataclass — call it `GraphCore` — holding
exactly those fields. Convert each `PopulationGraph` to a `GraphCore` once
during zoo-shard creation. The shards then pickle a list of `GraphCore`
objects, not full `PopulationGraph`/NetworkX objects.

Pickled size goes from ~30 KB per 100-node 80%-density graph (NetworkX) to
~16 KB (CSR int32). Deserialised resident memory drops by another factor of
~10x because we avoid all the Python dicts that NetworkX uses for `.adj`,
`.nodes`, and per-edge data.

**Why.** NetworkX adjacency is a `dict[node, dict[node, dict[attr, val]]]`. For
a 100-node, 80%-density graph that is 100 outer-dict entries with ~80 inner
dicts each — about 8000 Python dicts per graph, each with hash table overhead.
Numpy arrays are dense, contiguous bytes. Bonus: the inner loop in
`MoranProcess.step()` can use the CSR arrays directly with integer indexing,
which is also faster than iterating Python lists.

**Expected impact.** ~3-5x further reduction in worker baseline RAM, and a
nice speed-up in `step()` because neighbour lookup becomes
`nbrs[offsets[i]:offsets[i+1]]` (NumPy slice, no Python list construction).

---

## 3. Memory: stream results to disk; do not buffer 500k dicts

**Where.** `src/moran_process/pipeline/worker_wrapper.py` (`run_worker_slice`).

**What.** Replace the `results_buffer = []` + `.append(record)` pattern with:
- pre-allocate fixed-width NumPy arrays for `steps`, `initial_mutants`,
  `fixation`, `duration`, etc., sized to `my_tasks['n_repeats'].sum()`;
- after each task (one (graph_idx, r) pair) is done, flush the rows for that
  task to a single open CSV/Parquet writer and clear the buffer.

Better: write Parquet via `pyarrow.parquet.ParquetWriter` row-group by row-group.
Parquet is ~5x smaller than CSV for this data and reads ~20x faster in pandas.

**Why.** With 500,300 results per worker, the list of dicts costs ~150-250 MB
of resident RAM by the end of the run, and most of that is Python overhead
(each dict has ~10 string keys, each string interned but each value boxed).
Streaming makes worker RAM constant in the number of results.

**Expected impact.** Removes the ~200 MB growth tail. Also removes the giant
`pd.DataFrame(results_buffer)` allocation at the end of each worker, which is
itself a momentary 2x copy.

---

## 4. Memory + Speed: stop rebuilding `MoranProcess` and `adj_list` per repeat

**Where.** `src/moran_process/pipeline/worker_wrapper.py` lines 75-77 and
`src/moran_process/simulations/simulation_process.py` lines 20-23.

**What.** Today the worker does:

```python
for rep in range(n_repeats):
    sim = MoranProcess(population_graph=target_graph, selection_coefficient=r_val)
    sim.initialize_random_mutant()
    raw_result = sim.run()
```

`MoranProcess.__init__` (via `SimulationProcess.__init__`) recomputes
`self.adj_list = [list(self.pop_graph.graph.neighbors(n)) for n in range(...)]`
every single call. For 10,000 reps on a 100-node graph that's 10,000 redundant
list-of-list constructions, each producing ~100 small Python lists.

Refactor `MoranProcess`/`SimulationProcess` so the per-graph setup (adj_list /
CSR arrays) lives in a `prepare(graph)` method called **once** per
`(graph_idx, r_value)` task, and the per-rep code is a thin `reset()` +
`run()`. Or, equivalently, take `adj_list`/`offsets+nbrs` as constructor args
that are computed in the worker once per task.

**Why.** Each `MoranProcess.__init__` allocates ~50-200 KB of Python list
objects that immediately become garbage when the next iteration overwrites
`sim`. CPython's GC keeps freed memory in arenas that can fragment when the
allocation pattern is many-small-objects-of-similar-size. Over 500k iterations
this is a measurable contributor to RSS growth.

**Expected impact.** Both speed (~10-20% reduction in per-rep overhead) and
memory (removes one of the main churn sources).

---

## 5. Speed: rewrite the Moran inner loop  *(big lever for future r-near-1 batches)*

**Where.** `src/moran_process/simulations/moran_simulation_process.py`.

The current `step()` is:

```python
fitness = np.where(self.state == 1, self.r, 1.0)   # allocates length-N array
probs = fitness / fitness.sum()                    # allocates another length-N array
reproducer = np.random.choice(self.n_nodes, p=probs)
neighbors = self.adj_list[reproducer]
if neighbors:
    victim = random.choice(neighbors)
    self.state[victim] = self.state[reproducer]
```

And `run()` calls `int(np.sum(self.state))` every loop iteration to check
fixation/extinction.

### 5a. Track mutant count incrementally (always)

Maintain `self.mutant_count` and update it whenever `state[victim]` actually
changes type. Termination check becomes two integer comparisons. The current
`int(np.sum(...))` on every iteration is O(N) and is by far the easiest
correctness-preserving win.

### 5b. Sample reproducer in O(1) using the two-fitness structure

There are only two fitness values. Let `M = mutant_count`, `W = N - M`. The
probability of picking a mutant as reproducer is
`p_mut = r*M / (r*M + W)`. Then:
- draw `u ~ Uniform(0,1)`; if `u < p_mut` reproducer is a uniformly-chosen
  mutant, else a uniformly-chosen wild type;
- to pick within the chosen class in O(1), maintain two index arrays
  `mutants` and `wilds` and `pos_in_mutants[i]` (so swap-removes are O(1)).

This drops the per-step cost from O(N) (for `np.where` + `np.random.choice`
with `p=`) to O(1) and removes per-step allocations entirely.

### 5c. JIT the inner loop with Numba (preferred to C++)

Wrap the rewritten `step`/`run` in a Numba `@njit` function operating on
the CSR `(offsets, nbrs)` arrays and the integer `state` and the index arrays
from 5b. Numba compiles to native code; expected speedup over the current
Python+NumPy loop is 30-100x, comparable to C++/pybind11 but with vastly
lower build complexity (no setuptools, no manifest, single source of truth in
Python).

Only consider C++ if (a) you've done 5a+5b+5c and measured, and (b) the
simulation is still the bottleneck. For this project I expect Numba to be
sufficient; the 50,030-graph batch should run in <1 hour per worker after
this.

**Why not C++.** A C++ kernel via pybind11 buys maybe a further 2x over a
well-written Numba kernel for this style of code (irregular indexed
memory access on small graphs). The cost is: a build system, a wheel /
extension to keep in sync with the Python code, harder debugging on WEXAC,
and a barrier to anyone else (including future-you) modifying the kernel.
If you do go to C++ later, the right entry point is to keep the Numba
kernel as the reference implementation and only port hot loops.

### 5d. RNG hygiene

Today the code mixes `numpy.random.choice` (uses the global legacy RNG),
Python's `random.choice`, and `random.sample`. Three RNG state machines for
no good reason. Standardise on a single `np.random.Generator` per
`MoranProcess` instance (seeded explicitly). This is faster than the legacy
API (the new Generator is ~5-10% faster on common ops) and makes runs
reproducible.

---

## 6. Speed: avoid recomputing the WL hash and expensive graph metrics

**Where.** `src/moran_process/core/population_graph.py`.

- `PopulationGraph.__init__` calls `nx.weisfeiler_lehman_graph_hash(self.graph)`
  unconditionally. For a 50k-graph zoo where many graphs are programmatically
  constructed and you don't actually need the hash until registration, this is
  wasted work. Make it lazy: compute on first access via `@cached_property`.
- `calculate_graph_properties()` runs full Brandes betweenness for N<=100 — for
  N=100 with 80% density (~4000 edges) that is ~400k ops per graph, times the
  number of unique graphs (50k). Either cap the sample size further (e.g.
  always sample k=50 regardless of N), or run registration as a separate
  parallel job array rather than a single bsub.
- `mutate_graph` constructs a new PopulationGraph via `cls(G, ...)`, which
  computes a fresh WL hash even though we know nothing about the graph yet.
  With laziness from above this becomes free.

**Why.** The `register_graphs_job` step is single-process and currently runs
all 50k graphs sequentially with `calculate_graph_properties` doing nontrivial
NetworkX work. Even at 10 ms/graph that's 8-10 minutes. Lazy hashing alone
will shave a noticeable chunk off zoo construction.

**Expected impact.** Faster batch setup; not a memory win for workers.

---

## 7. Pipeline correctness / hygiene

A handful of things that aren't bugs today but will bite later.

- **`worker_wrapper.py` swallows exceptions silently** (line 101-104):
  `except Exception as e: print(...); continue`. For an HPC run this means a
  worker can produce a CSV that's missing rows for a task it failed on, and
  the failure is buried in stdout. Re-raise after logging, or write the failed
  task ids to a sidecar file the orchestrator can pick up.
- **`load_data`** reads the manifest CSV in every worker. The manifest is only
  1.3 MB so this is not a problem today, but if you stop sharding by writing
  per-worker manifests (recommended), this becomes free.
- **`process_lab.py:184`** writes the manifest via `to_csv` then submits. If
  the bsub fails the partial manifest is left on disk. Wrap in a try/except
  and clean up, or write to `manifest.csv.tmp` and rename.
- **`process_lab.py` `_create_task_list`** allocates one row per (worker_id,
  graph_idx) — for 1000 workers and 50k graphs the manifest can have ~50k
  rows. That's fine, but if you scale further consider Parquet for the
  manifest too.
- **Module-level filterwarning in `population_graph.py`:12** filters NetworkX's
  hash-instability warning globally; that's a footgun in case the WL hashing
  ever changes. Localise it to the `__init__` that actually calls the hash.
- **`worker_wrapper.py` accepts `--batch-dir` but the code expects it to point
  to `<batch>/tmp`** (it writes to `os.path.join(batch_dir, "results")`). This
  is implied by `submit_jobs` passing `tmp_dir` but not documented in the
  module docstring or `CLAUDE.md`. Fix the variable name or the docstring.

---

## 8. What I would do, in order, if I had to fix this today

1. **Per-worker zoo shards** (section 1). One afternoon's work; immediately
   fixes the OOM for the current batch.
2. **Stream results to Parquet** (section 3). Half a day; turns worker RAM
   into a constant.
3. **Hoist `adj_list` construction out of the inner loop** (section 4). One
   hour; nice constant-factor speedup.
4. **Incremental mutant count + drop `np.sum` per step** (section 5a). One
   hour; speeds up r-near-1 batches by ~5x.
5. **CSR adjacency in `GraphCore`** (section 2). Half a day; further memory
   reduction and sets up section 6.
6. **Numba JIT of `step`/`run`** (section 5c). One day; speeds up the long-tail
   sims by 30-100x.
7. **(Optional, only if step 6 still isn't fast enough.)** C++ via pybind11.
   Don't start here.

After step 1 alone, the current batch should fit in <4 GB and run to
completion. Steps 1+3 together should let you go back to ~2 GB per job, freeing
LSF slots for more parallelism. Steps 5+6 are needed for the *next* design
study, not this one.

---

## 9. Cross-cutting concerns

- **Workers report no per-task progress.** Add a once-every-N-tasks log line
  ("processed task X of Y, peak RSS = Z MB") so when a worker dies you can see
  *where* it died, not just that it died. Use `resource.getrusage(RUSAGE_SELF)`
  for RSS — no extra dependency.
- **Local debugging path.** The current `--job-index` flag is the right idea
  but the inner loop's `try/except Exception` will hide local crashes too.
  Add a `--strict` flag that re-raises.
- **`uv run` paths.** The submit script bakes
  `OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1` into the bsub
  `-env` line. Good — confirms you've already addressed BLAS oversubscription.
  Keep this when moving to Numba.
- **Aggregation step.** Once you move to Parquet, `aggregate_results_no_load`
  becomes a one-liner with `pyarrow.dataset.dataset(...).to_table()`. Worth
  doing in the same PR as section 3.

---

## 10. A note on the C++ vs Numba decision

You mentioned wanting to rewrite the heavy-lifting part in C++. Here is the
honest tradeoff:

- **C++ via pybind11**: ultimate per-step performance (~30-50 ns/step on the
  L1-resident workload), full control over memory layout, well-understood
  build patterns. Build complexity: a `CMakeLists.txt`, a wheel, a
  `setup.py`/`pyproject.toml` change, debug-vs-release builds on WEXAC nodes
  (which often have older glibc). You also lose the ability for a future
  collaborator to read the simulation code without knowing C++.
- **Numba `@njit`**: ~50-100 ns/step on the same workload, identical
  algorithmic structure, single-language codebase, no build system. The
  warm-up compile is ~1 s per worker (negligible vs the 30 s of joblib load
  you have today). Numba's small-graph perf is comparable to hand-written C
  for this style of loop with integer indexing.
- **Numba pitfalls**: no Python objects in the JIT'd region, so you must pass
  arrays only (which you should be doing anyway after section 2). Some
  NumPy random ops have different JIT support — use `numpy.random.Generator`
  inside `@njit` carefully or pass pre-generated random batches.

My recommendation: **do Numba first**. If after sections 1-6 you measure the
inner loop and it's still the bottleneck for the production workload you care
about, *then* rewrite the kernel in C++. The Numba version is also a useful
ground-truth reference for any C++ port.

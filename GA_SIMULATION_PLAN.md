# Plan: simulation-driven genetic algorithm

Branch `feature/GA-sim`. Replace the ML-predicted fitness in the evolutionary search with
the **measured** fitness from a real simulation batch, so that "these are the extreme
graphs" becomes an objective statement about the Moran process rather than a statement
about what a regressor believes.

Each generation is its own simulation batch. All of them live under one dedicated
directory.

---

## 1. What changes, and why

`notebooks/extreme_graphs.ipynb` runs a (mu + lambda) evolutionary strategy whose fitness
is `model.predict(graph_properties)`, using the residual predictors in
`ml_models_residual/`. Evaluating 110 candidates costs milliseconds, so the whole search
fits in a notebook cell.

The new search evaluates candidates by simulating them. That single substitution changes
every cost in the system:

| | ML-predicted fitness | measured fitness |
|---|---|---|
| cost of evaluating a generation | ~1 s (CPU) | ~28 core-min, ~2 min wall clock (LSF) |
| where the loop can live | a notebook cell | a long-running job |
| what a generation produces | a float per graph | a full batch directory |
| dominant cost | nothing | LSF scheduling latency |
| number of runs | 8 (4 models x 2 objectives) | 4 (2 metrics x 2 objectives) |

The last row is the cleanest simplification: the LR-vs-XGBOOST axis was a modelling
artefact. Simulation has no such variants, so the run matrix halves.

---

## 2. Measurements this plan is built on

Everything below was measured on this repository's own batches, not estimated.

**Per-simulation cost** (`2026_07_28-respiratory-vs-random-10K-reps-3/job_speed.csv`,
4.8e8 simulations): **13.9 us/sim**, 6.6e7 steps/s/core, 917 steps/sim.

**Signal versus noise** at exactly the GA's regime (N=31, E=34, r=1.1, 501 random graphs,
10K repeats, from that batch's `graph_statistics.csv`):

| metric | between-graph SD (signal) | per-graph SEM at 10K (noise) | S/N |
|---|---|---|---|
| `mean_steps` | 866 (range 3091 - 8509) | 87 | 9.9 |
| `prob_fixation` | 0.00425 (range 0.101 - 0.128) | 0.00319 | **1.3** |

This is the finding that sets `n_repeats`. **At 10K repeats, selecting on
`prob_fixation` would be selecting noise**: ranking 220 candidates and keeping the top 20
would return mostly lucky draws, and the resulting fitness trajectory would be the GA
climbing its own sampling error. `n = p(1-p)/SEM^2` with p ~= 0.11 gives ~540K repeats to
reach S/N ~ 10. `n_repeats = 500_000` is used for all four runs.

**LSF latency** (`2026_07_20-extreme_ocmbined_100K-2/logs`, timestamps): an array element
took ~80 s from submit to done while performing ~1 s of simulation. Fixed cost per array
element is therefore tens of seconds, which is why the array is auto-sized to give each
worker real work rather than being pinned at 1000.

**Storage** (same batch, `tmp/results`): 8.41 bytes/row. A generation of 220 graphs x 500K
repeats is 1.1e8 rows = ~0.93 GB. Retaining all of it would be ~370 GB across 4 runs x 100
generations, hence the retention policy in section 5.

---

## 3. Decisions

| decision | choice |
|---|---|
| loop location | new module, one `bsub`'d long-running driver job per GA run |
| batch per generation | `register_graphs` + array + chained `aggregate`; no verify / violin / job_speed |
| array size | auto-sized from 13.9 us/sim, targeting ~30 s of work per worker |
| fitness | 4 independent runs: `{mean_steps, prob_fixation}` x `{maximize, minimize}` |
| `n_repeats` | 500,000 for every run |
| `r_values` | `[1.1]` throughout, including the final populations |
| layout | `simulation_data/ga_runs/<run>/generations/gen_NNN/` as standard batch dirs |
| raw shards | deleted after successful aggregation |
| elite scoring | re-simulated fresh every generation |
| population | 20 elites x 10 children = 220 candidates/generation |
| generations | fixed 100, no early stopping |
| initial population | 20 random (31, 34) graphs, one seed, shared by all 4 runs |
| reference graphs | none |
| incomplete generation | strict check, automatic resubmit with a warning, cap 2 retries |
| notebooks | delete the sketch, add a thin launcher/analysis notebook, leave the ML one alone |

Two of these deserve their reasoning recorded.

**Elites are re-simulated every generation.** In a (mu + lambda) ES with a noisy fitness,
an elite that carries its old score forward is never re-tested, so a graph that got a
lucky-high estimate sits at the top of the ranking permanently and children (freshly
measured, hence unlucky on average by comparison) can never displace it. After 100
generations the population would be a museum of lucky draws. Re-simulating the 20 elites
alongside the 200 children costs 9% more compute and makes each generation's ranking a
fair comparison of independent estimates. Pooled estimates remain computable afterwards,
because every generation's `graph_statistics.csv` is kept.

**Raw shards are deleted after aggregation.** `populations/gen_NNN.pkl` is retained, so any
generation's raw data is reproducible in ~2 minutes by resubmitting that population.
Deleted shards are reproducible, not lost. This takes the experiment from ~375 GB to well
under 1 GB.

**Why N=31, E=34.** These are exactly `avian_r4_l7`. Because `mutate_graph` removes one
edge and adds one, both counts are invariant, so every graph the GA ever sees is
size-matched to the avian lung graph. The experiment therefore asks: *among all connected
graphs with the avian graph's node and edge count, what are the extremes, and where does
the real avian topology sit among them?*

A consequence worth noting: with N and E fixed, the complete-graph baselines are constants
within a run, so `rho - rho_c` and `log(T / T_c)` are monotone transforms of `rho` and
`T`. Residual versus raw changes the axis labels, not the argsort. Selection uses the raw
measured values; the figures display residuals where the old ones did.

---

## 4. Directory layout

```
simulation_data/ga_runs/
  2026_07_29-max-mean_steps/
    ga_config.json              # every hyperparameter, seed, git provenance
    ga_state.json               # checkpoint + live progress (see section 7)
    ga_history.csv              # one row per (generation, candidate)
    populations/
      gen_000.pkl ... gen_099.pkl
    generations/
      gen_000/                  # a standard batch directory
        batch_info.json
        graph_props.csv
        graph_statistics.csv
        logs/
        tmp/                    # results/ deleted after aggregation
      gen_001/ ...
    final_population.pkl
    figures/
  2026_07_29-min-mean_steps/
  2026_07_29-max-prob_fixation/
  2026_07_29-min-prob_fixation/
```

Each `gen_NNN/` is a standard batch directory, so every existing reader works on it
unchanged: `load_graph_statistics` (including its list-of-batches stitching),
`post_batch_status`, and the streamlit app.

`ga_history.csv` columns:

```
generation, wl_hash, graph_name, parent_wl_hash, prob_fixation, mean_steps,
std_steps, n_grouped, fitness, rank, survived, is_new
```

This replaces the old in-memory `history` dict, which held only the mean predicted fitness
of survivors and so could never answer which graph won or when a lineage appeared.

---

## 5. The driver

`src/moran_process/pipeline/ga_search.py`.

```
python -m moran_process.pipeline.ga_search \
    --run-dir simulation_data/ga_runs/2026_07_29-max-mean_steps \
    --metric mean_steps --objective maximize \
    [--generations 100] [--pop-size 20] [--n-children 10] \
    [--n-repeats 500000] [--queue gsla-cpu] [--resume] [--force]
```

Launched with a single `bsub` onto a long queue with ~2 GB, since the driver itself uses
almost no CPU: it spends its life waiting on LSF.

Per generation:

1. **Reproduce.** For each of the 20 elites, generate 10 mutants via `mutate_graph`,
   rejecting any whose `wl_hash` is already in the run's seen-set (up to 10 attempts, as
   the current notebook does). Candidates = 20 elites + 200 children.
2. **Serialize.** Write `populations/gen_NNN.pkl` and the generation's
   `tmp/graph_zoo.joblib`.
3. **Submit.** `register_graphs` job and the simulation array (concurrent), then the
   `aggregate` job with `-w ended(array) && done(register)`. Three bsubs. `register` costs
   nothing serially since only `aggregate` waits on it.
4. **Wait.** Poll `bjobs` on the aggregate job id until `DONE`/`EXIT`, then read
   `graph_statistics.csv`. Polling the file directly would risk reading it mid-write, since
   `build_graph_statistics` writes with a plain `to_csv`. A wall-clock timeout covers an
   aggregate job that dies without a state transition.
5. **Verify.** Assert every candidate has `n_grouped == n_repeats`. If any is short, log a
   `WARNING`, record it in `ga_state.json`, and resubmit the whole generation (cap 2
   retries, then checkpoint and exit non-zero). A graph measured with 5K instead of 500K
   repeats has ~10x the noise and can win selection on luck alone, so this check is
   load-bearing, not hygiene.
6. **Select.** `argsort` on the chosen metric in the chosen direction, keep the top 20.
7. **Record.** Append to `ga_history.csv`, rewrite `ga_state.json`, delete
   `generations/gen_NNN/tmp/results/`.

Array size:

```python
TARGET_SECONDS_PER_WORKER = 30
US_PER_SIM = 13.9e-6          # measured, section 2
est = n_graphs * n_repeats * len(r_values) * US_PER_SIM
n_jobs = clip(ceil(est / TARGET_SECONDS_PER_WORKER), 1, 200)
# 220 graphs x 500K x 1 r -> est 1529 s -> n_jobs = 51
```

Parallelism helps only until per-worker work drops below per-worker startup; past that,
more workers means more wall clock, not less. The constant is the measured one, so if
`n_repeats` or N changes later the sizing follows without retuning.

**Superseded by section 13.2:** `US_PER_SIM` is no longer a constant. It is re-estimated
from each generation's own results, because a search that maximizes fixation time makes its
own simulations 9x more expensive as it succeeds.

**Resume.** `ga_state.json` is rewritten after every generation (generation index, elite WL
hashes, RNG state, retry warnings); the seen-topology set lives beside it in
`seen_hashes.txt`, since it grows to ~20k hashes and would bury the dozen fields worth
reading. An existing `ga_state.json` is **resumed from automatically** -- there is no
`--resume` flag, because `medium`/`long` are preemptable and a preempted job is requeued
from the beginning, so resuming is the normal path rather than an exceptional one.
`--force` is the only way to start over. The RNG state survives the JSON round-trip
exactly (verified), so a resumed run continues the same mutation sequence.

**Launching.** `ga_search.submit_driver(...)` bsubs one driver; `submit_all_runs(...)`
submits the 2x2 matrix sharing one initial-population seed. The driver goes to
`gsla-cpu` with a 12 h walltime, 4 GB and one slot (see section 13.1: it was 2 GB before
the rollup moved into it); the per-generation arrays go to
`gsla-cpu` too. These are separate decisions that happen to land on the same queue: the
driver is one long idle process, the arrays are many short compute jobs. `gsla-cpu` is the
right home for the driver because `bqueues -l` reports it PREEMPTIVE with a 45000-minute
limit, while `short` (1440 min), `medium` (4320 min) and `long` (10080 min) are all
PREEMPTABLE. A preempted driver does resume, but it discards the generation it was in the
middle of.

---

## 6. Change to `ProcessLab.submit_jobs`

`submit_jobs` currently hard-codes the full six-bsub chain and returns `None`. The driver
needs neither the full chain nor an opaque return. Two small, backwards-compatible
changes:

- Add `post_batch="all"`, accepting `"all"` (current behaviour, the default) or
  `"aggregate"` (register + array + aggregate only). Nothing else changes for existing
  callers.
- Return a dict of job ids (`register`, `array`, `aggregate`, ...). No current caller uses
  the return value (checked: `main.py`, `design_zoo.ipynb`, `design_zoo_toy.ipynb`,
  `design_zoo_scaling_study.ipynb`), so this is additive.

The driver then calls `submit_jobs(..., post_batch="aggregate")` and waits on the returned
aggregate id. No simulation code is duplicated: generations go through the same
`worker_lsf` array as every other batch in the project.

---

## 7. Progress reporting

The driver runs detached on a compute node, so progress has to be readable from outside
the process. Three layers, all fed from the same state:

**In the driver's LSF log.** One line per generation, written as a bar so a `tail -f` of
`logs/ga_driver_*.out` shows position at a glance:

```
[max mean_steps] gen  37/100 |##########------------| 37%  best=8842  median=8103  new_elites=4  eta 2h03m
```

**In `ga_state.json`.** Machine-readable, updated after every generation, so a notebook or
streamlit can render progress for a run it did not launch:

```json
{"generation": 37, "generations": 100, "pct": 37.0,
 "started_at": "...", "last_generation_at": "...", "eta_seconds": 7380,
 "best_fitness": 8842.1, "median_fitness": 8103.4,
 "status": "running", "warnings": []}
```

**In the notebook.** `ga_progress(run_dirs)` renders one bar per run, so a single cell
shows all four:

```
max  mean_steps     |##########------------| 37/100   eta 2h03m
min  mean_steps     |############----------| 42/100   eta 1h51m
max  prob_fixation  |#########-------------| 35/100   eta 2h11m   1 warning
min  prob_fixation  |##########------------| 38/100   eta 1h58m
```

The bar is hand-rolled rather than `tqdm`, because it carries custom fields (best, median,
new_elites, eta) and must emit one whole line per generation: a carriage-return animation
renders as garbage in an LSF `.out` file. Logging is configured onto **stdout**, not the
default stderr, so the bar lands in the `.out` file you would actually tail.

---

## 8. Figures

The same figures the ML notebook produced, on the same axes, with measured values in place
of predictions.

**Fitness trajectory** (`plot_ga_history`, adapted from `plot_multi_model_history`). The
old figure plotted the primary model's prediction plus the other three models' predictions
on twin physical axes. The new one plots the optimized metric plus the *other* measured
metric for the same survivors, which maps onto the identical two-axis layout: left axis
mean fixation time, right axis fixation probability, complete-graph baselines drawn as the
residual origin via `analytic_moran_fc_fixation_prob` / `analytic_moran_fc_fixation_time`
at (N=31, r=1.1). Kept from the original: the solid-for-time / dashed-for-probability
linestyle convention, the thicker line for the optimized metric, the baseline annotations,
and the sorted unified legend below the axes.

What is added, because measurement makes it available and prediction did not: a shaded
band for the spread across the 20 survivors, and error bars at the SEM of the measurement,
so the figure shows directly that the trajectory exceeds the noise floor.

**Winners gallery.** `graph.draw(title=...)` per category, as in the ML notebook's final
cell, driven off `final_population.pkl` and grouped by run.

**Cross-run comparison.** All four trajectories on one pair of axes, which the ML notebook
could not do because its eight runs were not commensurable.

**Where the avian graph sits.** The final populations against the distribution of random
(31, 34) graphs already measured in `2026_07_28-respiratory-vs-random-10K-reps-3`, with
`avian_r4_l7` marked. This is the figure the whole experiment exists to produce.

**ML versus simulation** (`plot_ml_vs_simulation`). The ML-driven winners in
`2026_07_15-.../extreme_graph_zoo/extreme_graphs.pkl` were measured at 100K repeats in
`2026_07_20-extreme_ocmbined_100K-2`, and are all (31, 34) at r=1.1, so the two sets of
winners are exactly comparable. Both panels (mean fixation time, fixation probability) show
one row per winner group: a bar at the group mean and a marker per individual winner, so
the figure answers how far each group got *and* how tightly its winners agree. A group
whose markers scatter across the axis never converged, which a chart of means alone would
hide. Bars grow from the complete-graph baseline rather than from zero, which is the
project's usual residual origin and makes bar direction read as amplifier versus
suppressor; anchoring at zero spends 60% of the probability axis on empty space below 0.08
and leaves every group looking the same length. Colors come from `CATEGORY_COLOR_DICT` via
`generate_robust_color_dict`, so a simulation-driven category shares its color with the
ML-driven category it corresponds to. This is the only figure in the project where the
residual predictors can be checked against ground truth.

**The two metrics jointly** (`plot_ml_vs_simulation_scatter`). The same winners as a
scatter of fixation probability against fixation time, over the random (31, 34) cloud, with
the complete graph as a crosshair splitting the plane into the four amplifier / suppressor
quadrants. The bar figure above plots each metric marginally and so cannot show how the two
move together, which is the actual research question. Fixation time is on a log axis
because the winners span 2.4K to 35K steps and the fast groups otherwise collapse into the
left margin.

---

## 9. Files

**Add**
- `src/moran_process/pipeline/ga_search.py` - the driver, plus `submit_driver` /
  `submit_all_runs`.
- `src/moran_process/analysis/analysis_utils/ga_io.py` - readers for `ga_history.csv` /
  `ga_state.json`, and `ga_progress`. Reader-only, following the builder/reader split;
  needs only pandas, so progress can be checked without importing the plotting stack.
- `src/moran_process/analysis/analysis_utils/ga_plots.py` - the GA figures. A separate
  module from `plots.py` for the reason `plots.py` was split out of the original
  monolith: it is already ~1600 lines, and these figures read a different artefact.
- `notebooks/ga_simulation.ipynb` - launch the four drivers, render progress, produce the
  figures.

**Change**
- `src/moran_process/pipeline/process_lab.py` - `post_batch` parameter, return job ids.
- `src/moran_process/analysis/analysis_utils/colors.py` - four `CATEGORY_COLOR_DICT`
  entries for the simulation-driven GA categories (`maximize mean_steps` etc.). Each takes
  the same hue as its ML-driven counterpart -- probability in blues, time in reds -- so
  the two searches share a colour in the comparison figure. Without them they would fall
  through to auto-generated husl.
- `CLAUDE.md` - the GA run layout and the `ga_runs/` convention.

**Delete**
- `notebooks/extreme_graphs-simulation.ipynb` - the sketch. Its loop cell called
  `submit_jobs` and then still called `model.predict` on a `props_df` that no longer
  existed; the two halves were never reconciled, and the loop is moving to a module
  anyway.

**Untouched**
- `notebooks/extreme_graphs.ipynb` - the ML-driven GA, kept for comparison.
- `src/moran_process/pipeline/extreme_graphs.py` - kept by decision. Nothing imports it and
  `ga_search.py` does not build on it.

---

## 10. Implementation order

1. `submit_jobs`: `post_batch` parameter and job-id return. Verify against a 3-graph,
   1000-repeat toy batch.
2. `ga_search.py` with `--generations 2 --pop-size 3 --n-children 2 --n-repeats 1000`.
   Confirms the whole loop (submit, wait, verify, select, checkpoint, prune) for ~4
   minutes of cluster time.
3. `--resume` against that toy run, killed mid-flight.
4. `ga_io.py` + progress rendering.
5. Full-scale launch: four drivers, 100 generations, ~3.5 h.
6. Figures.

Steps 1-4 cost minutes of cluster time. Nothing at full scale runs until the toy run
completes and resumes cleanly.

---

## 11. Risks

**The `prob_fixation` runs remain the statistically hard ones.** 500K repeats buys S/N ~ 10
against the between-graph spread *of random graphs*. As the GA pushes into the tail that
spread should widen, which helps, but the first few generations are the noisiest and the
most likely to fix a lucky lineage. If the two `prob_fixation` trajectories look like
noise, the diagnosis is `n_repeats`, and `ga_history.csv` retains the per-candidate SEM
needed to check that directly.

**Latency, not compute, sets the wall clock.** ~2 min per generation is mostly LSF
scheduling. A congested queue stretches the run without any way to speed it up from this
side. Four concurrent drivers is deliberate: it makes wall clock one run's, not four.

**`mean_steps` is conditional on fixation.** `_steps_success_expr()` nulls out `steps` on
extinct runs. Maximizing it is unambiguous, but the *minimize* `prob_fixation` run drives
toward graphs that rarely fixate, whose `mean_steps` is then estimated from few surviving
runs. That only affects the secondary curve on that run's figure, not its selection, but
the figure should show the SEM so it is not read as signal.

---

## 12. FIXED: `ga_progress` rendered nothing when it was given nothing

Observed on the first real launch (LSF jobs 490910/490912/490913/490914, 50 generations,
pop 10 x 10 children, 1000 repeats). Section 2 of `notebooks/ga_simulation.ipynb` produced
**no output at all** while all four runs were in fact healthy and at generation 34/50.

**What actually happened.** The notebook computes its run list by globbing rather than from
what it launched:

```python
RUNS = sorted(GA_RUNS_DIR.glob(f"{PREFIX}-*"))
ga_io.ga_progress(RUNS)
```

`PREFIX` had been edited to `2026_07_29-toy-run` at some point after the launch cell ran,
but `submit_all_runs` had already created the directories from the earlier value, so they
are named `2026_07_29-<objective>-<metric>`. The glob `2026_07_29-toy-run-*` matches none
of them, `RUNS` is `[]`, and `ga_progress` loops over an empty list and returns silently.
Verified directly: the same call on `glob("2026_07_29-*")` prints all four bars correctly,
so the bar itself, `_bar`, and `ga_state.json` are all fine.

**Why this is a code defect and not just a typo.** `ga_progress` already refuses to be
silent about a run that has not checkpointed yet, printing `not started` for it. The intent
was clearly never to render nothing. That intent just does not cover the zero-run case, and
the zero-run case is the one that matters most, because it is indistinguishable from
"submitted, nothing has finished yet" -- which is a legitimate state you would wait through.
The user waits on a working run that they simply are not looking at.

There is a second-order cause worth fixing at the same time: `submit_all_runs` **returns a
dict keyed by run directory name**, and the notebook throws it away and re-derives the same
list from `PREFIX`. So the launch and the progress cell are coupled only through a string
that either cell can be edited to change independently. That is what allowed them to drift.

**Fix, when the current run finishes:**

1. `ga_io.ga_progress` prints an explicit line when the run list is empty, naming the
   directory it looked in, so "nothing matched" is never rendered as "nothing to say".
   Same treatment for `load_ga_runs` returning nothing.
2. The notebook's section 2 derives `RUNS` from the launch cell's return value
   (`RUNS = [GA_RUNS_DIR / name for name in jobs]`), falling back to the glob only when
   `jobs` is not defined in the kernel. That removes the drift entirely for a launch and
   read in one session, and keeps the glob working for reading a run launched earlier.
3. Consider having `load_ga_runs(GA_RUNS_DIR)` be the documented way to list runs, since it
   filters on the presence of `ga_config.json` rather than on a name prefix, and so cannot
   miss a run because its name was typed differently.

Nothing here affects a running search: all three are read-path changes.

**All three landed on 2026-08-04.** `ga_progress` prints an explicit line naming the
likely cause when the run list is empty; the notebook derives `RUNS` from the launch
cell's `jobs` when this kernel launched, falling back to `load_ga_runs` otherwise; and
`load_ga_runs` gained a checked `prefix=` that **raises and lists what does exist**
rather than returning an empty list, since every caller reads empty as "no progress".

The general shape is worth naming, because it recurred: an operation that succeeds
*vacuously* is indistinguishable from one that succeeds *meaningfully*. Iterating an
empty list completes without error, and posting to an unsubscribed ntfy topic returns
HTTP 200 (see 13.5). In both the code was correct and the only defence was to make
emptiness announce itself.

---

## 13. Wall-clock work, seed determinism, replicates, notifications

Five changes, driven by measuring a real generation rather than reasoning about it.
`gen_001` of `2026_07_28-long-100-gen-run-minimize-mean_steps` (220 graphs, 100K repeats,
11 workers) took 202 s end to end, and it went:

| stage | seconds | share |
|---|---|---|
| array queue wait | 70 | 35% |
| array run (the actual simulation) | 50 | 25% |
| aggregate queue wait | 51 | 25% |
| aggregate run (7.2 s CPU, 513 MB peak) | 19 | 9% |
| driver poll detection | 15 | 7% |
| register_graphs | 0, it runs concurrently | 0% |

**Simulation was 13% of a generation. 60% was LSF dispatch latency, paid twice.**

### 13.1 The rollup moved into the driver

`_submit_generation` now passes `post_batch="none"` and `_run_generation` calls
`aggregate_batch.run_aggregation` itself, having waited on the array and the register job.

The chained aggregate job was correct for a real batch (7.2e9 rows, genuinely needs 16 GB)
and wrong for a GA generation (2.2e7 rows). Handing 19 s of work to another machine cost
66 s of waiting, once per generation, while the driver's own slot sat in a sleep loop. The
driver's reservation went 2 GB to 4 GB to cover it, measured against the 513 MB the job
actually peaked at.

Two knock-on changes were required:

- `_job_state` now collapses an array's per-index rows to one state, pessimistically: any
  index still PEND or RUN means the job is that. It previously read row 1 only, which was
  harmless while the only thing waited on was a single aggregate job and would have called
  an 11-index array finished the moment index 1 exited.
- `POLL_SECONDS` 20 to 5. At 100 generations a 20 s poll spends ~17 minutes per run
  noticing work that had already finished.

Not changed, because measurement said not to: **`register_graphs` stays a separate job.**
It is submitted with no dependency and finished 111 s before the array did, so it costs no
wall clock. An earlier draft of this plan claimed it was in the critical path; it is not.

### 13.2 Array sizing follows the search

A fixed `SECONDS_PER_SIM` is wrong here because the search changes it. Over the 100-
generation run, `prob_fixation * mean_steps` (the fraction of runs that fixate times how
long they take, a proxy for average steps per simulation) grew 652 to 6101, a factor of
9.4, as `maximize mean_steps` did its job. Sized off the constant, the late generations
were the slow ones.

`_estimate_seconds_per_sim` now re-derives the cost from each generation's own results and
`ga_state.json` checkpoints it, so a resumed run does not revert to generation-0 sizing.
`STEPS_PER_SECOND = 4.9e7` is calibrated from one direct measurement (gen 1: proxy 652
steps/sim against 13.4 us/sim measured, being 2e6 sims in 26.84 s of CPU). It is floored at
the old constant, since the proxy ignores steps burned by runs that go extinct and can
therefore only understate the cost. Effect: 11 workers at gen 1, 92 at gen 99.

### 13.3 Runs are now reproducible from their seed

Two halves, only one of which was deterministic before.

**Measurement.** `_submit_generation` passes `batch_seed = seed * 100003 + generation` to
`submit_jobs`. It previously passed nothing, so `_create_task_list` seeded from OS entropy
and every run drew fresh per-task seeds. Deriving it from the generation number rather than
from a running stream matters: a resubmitted short generation would otherwise consume a
different amount of the stream and desynchronise everything after it.

**Selection.** `_select` sorts on `(metric, wl_hash)`, not on `metric` alone. Exact ties are
routine rather than exotic: `prob_fixation` is k/n_repeats, an integer over a fixed
denominator, so equal k gives bit-identical floats. Measured on the 100-generation run, 27
of 220 rows per generation (12%) shared a value with another row, and **in 7 of 99
generations the tie landed exactly on the elite cutoff**. Sorted on the metric alone, that
decision came from pandas row order, which came from shard glob order: `gen_007` yields two
different elite sets across eight shuffles of its own rows, and one differing elite breeds
different children forever after, so a single flip forks the trajectory. With the hash as
secondary key it yields one.

A hash is the right tiebreak precisely because it is a hash: uncorrelated with fitness so
it plays no favourites, fixed so it costs no RNG state. A coin flip's fairness with a
rule's determinism, which is why `rng.choice` was considered and dropped.

### 13.4 Replicates

`submit_all_runs(..., replicates=k)` repeats the whole 2x2 matrix, replicate *k* seeded at
`seed + k*1000`, which reseeds the initial population, the mutation stream and the
simulation seeds together. Directories gain a `-repN` suffix only when `k > 1`, so a
single-replicate launch keeps the names every existing figure expects.

Varying everything is deliberate. Holding the starting population fixed and varying only
mutation answers a narrower question (how path-dependent is the search from this one
start); the question being asked is whether the whole procedure, run again from nothing,
arrives somewhere similar.

Load: 4 runs per replicate, each one driver slot plus up to `MAX_WORKERS`. Three replicates
is 12 concurrent runs, which will meet the group's 680-slot limit on `gsla-cpu`
(`blimits -w -a -q gsla-cpu`) and queue. That costs wall clock and nothing else.

Read side: `ga_io.final_elite_properties` joins the final elites to their generation's
`graph_props.csv`, and `ga_plots.plot_replicate_agreement` draws the two cheap layers (same
fitness, same structural fingerprint z-scored across the graphs shown). Overlap of
`wl_hash` is deliberately **not** reported: a run visits ~20k topologies out of an
astronomical space, so it is zero between independent runs and always will be, and would
only ever say "the runs disagree completely", which is a fact about the size of the space.

### 13.5 Notifications

**One message per launch, not per run.** `submit_all_runs` submits its drivers with
`notify_on_finish=False` and then bsubs a tiny watcher job holding
`-w "ended(d1) && ended(d2) && ..."` on every driver, which runs `--summarize` and sends a
single message once they have all ended. With replicates that means one message for the
whole 12-run matrix.

`ended()` rather than `done()` is deliberate: `done()` requires success, so one dead driver
would leave the watcher PEND forever and the single message you were waiting for is the one
you never get. `ended()` fires either way and `summarize_runs` reads the state files to
report which runs actually made it, escalating to priority `high` with a
`GA finished with N PROBLEM(S)` title when any did not.

An LSF dependency rather than "the last driver notices it is last", because the drivers are
independent jobs with no view of each other and that check is a race: two finishing in the
same second both see an unfinished sibling and neither sends.

Two exceptions still notify per run. **Failures** fire immediately at high priority, since a
run that dies at hour four is worth interrupting for and folding it into the summary would
mean hearing about it only once its siblings also finished. And a bare `submit_driver` call
outside `submit_all_runs` still announces itself.

Durations in the summary come from `_run_span` (started_at to last_generation_at) rather
than `_elapsed_since_launch` (started_at to now). The latter is right inside the driver,
where "now" is the moment the run ended, and wrong in a watcher that runs afterwards: it
would charge the run for the watcher's own queue wait.

**Launch ping.** `submit_all_runs` also sends one message at submit time. This exists
because the failure mode is silent by construction: posting to a valid but *unsubscribed*
topic returns HTTP 200, so a wrong topic is byte-identical to a working one from the
sender's side and no return-value check can catch it. This was not hypothetical, it
happened: a topic differing by one `-ntfy` segment swallowed a whole smoke test's
notifications, all four of which were sitting unread on the wrong topic. The only test of a
fire-and-forget channel is to fire early and look at the receiving end.

`ga_search.notify` POSTs to an ntfy.sh topic on completion and on failure. stdlib `urllib`,
no dependency; verified reachable from compute node `hgn20`, and a POST returns HTTP 200
with a server-generated message id. The topic comes from `NTFY_TOPIC` in the environment
and is forwarded to the driver by `submit_driver`, so it never lands in the repo or in
`bjobs -l` output. It should be unguessable, since the topic name is the only access
control ntfy.sh has. Every failure is swallowed: a notification is a courtesy and must not
take down the run it was announcing.

**Setting it.** `NTFY_TOPIC` is read, never written, so it has to be exported by the shell
that submits. Generate it rather than inventing it:

```bash
python -c "import secrets,string; print('moran-ga-'+''.join(secrets.choice(string.ascii_letters+string.digits) for _ in range(24)))"
echo "export NTFY_TOPIC='<paste it here>'" >> ~/.bashrc
```

**Letters, digits, `-` and `_` only.** Punctuation is not a way to make a topic stronger and
breaks it two ways. A `#` is a URL fragment, so `topic#secret` requests `/topic` and the
"secret" half never reaches the server; percent-encoding the rest gives a 404, since ntfy
topics match `[-_A-Za-z0-9]{1,64}`. Characters like `(`, `$` and `!` also make the
`~/.bashrc` line a **bash syntax error** unless the value is single-quoted, which breaks
every new shell. Unguessability comes from length: 24 alphanumerics is 143 bits.

`submit_driver` validates the topic against that pattern at submit time and says so once if
it fails, because `notify` swallows its own errors by design and an invalid topic would
otherwise surface as a 404 buried in a driver log hours later.

The trap this creates is a Jupyter kernel, which inherits the environment of the shell that
launched the server. Exporting after `ijup` is already running leaves the variable empty
until the server is restarted, and the driver would then be submitted with an empty topic
and silently never notify. `submit_driver` therefore prints a one-time note when the topic
is missing, rather than letting five hours pass before the absence is noticed.

**Duration.** Both messages carry elapsed time, measured from `started_at` in
`ga_state.json` rather than from a process-local `monotonic` baseline. The baseline resets
on resume, so a preempted run would otherwise report only the time since its last requeue.
What is reported is wall clock from launch, including any interval spent dead in the queue,
because that is the number that answers "how long from asking for it to being able to look
at it". When a resume makes the two clocks differ by more than two minutes both are shown.
The completion message adds per-generation average; the failure message adds which
generation it died at.

```
GA finished: 2026_08_02-run-maximize-mean_steps
maximize mean_steps over 100 generations in 5h30m
3m18s per generation
best=41521.3  median=40880.3
21877 topologies evaluated

GA FAILED: 2026_08_02-run-maximize-mean_steps
maximize mean_steps
died after 2h02m at generation 37/100
SystemExit: Generation 36 still incomplete after 3 attempts.
```

### 13.6 Selection efficiency, and what it revealed

`ga_plots.plot_selection_efficiency` plots, per generation,

    rho = 1 / sqrt(1 + (SEM / SD_between)^2)

the correlation between measured and true fitness, to which the per-generation response to
selection is proportional. It needs no new simulation: `ga_history.csv` already carries
both `_sem` columns. It is computed over all candidates rather than survivors, because the
candidate pool is what selection ranked.

This replaces the raw signal-to-noise framing in section 3, which was too strict. At 1000
repeats rho is 0.39, so that search retains 39% of the ideal response rather than none,
which is why a 1000-repeat run produced usable results.

Measured on the 100-generation run at 100K repeats:

| run | gen 1 | gen 30 | gen 99 |
|---|---|---|---|
| maximize prob_fixation | 0.966 | 0.848 | 0.910 |
| maximize mean_steps | 1.000 | 1.000 | **0.877** |

`prob_fixation` holds around 0.85 throughout, so 100K was the right choice. `mean_steps` is
the surprise and it runs opposite to intuition: it sits at 1.000 for 60 generations and
then collapses, because SEM for a scale metric is proportional to the mean. Between-graph
SD fell 12700 to 504 as the elites converged while SEM rose 30 to 276 as the GA drove the
mean up eightfold. **A run that succeeds at maximizing a scale metric inflates its own
noise floor in step with its own signal.** That is a property of the metric, not of this
implementation, and it argues for raising `n_repeats` late in a run rather than at the
start.

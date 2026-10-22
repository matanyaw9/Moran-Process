
# Brief: lung-selection

`lung-selection` is the rewrite of an MSc thesis codebase (Weizmann, Computational
Biology). You both write and review code here. When I ask for a review, point to
`file:line`, explain the problem, and suggest the fix; do not rewrite unless I ask.
When I ask for code, follow "Writing code" below.

## Research question
Do respiratory topologies (mammalian lungs, including asymmetric and mouse variants;
avian lungs; fish gills) act as evolutionary amplifiers or suppressors of selection,
compared to random graphs of the same size? Metrics: fixation probability (rho) and
fixation time, as a function of the selection coefficient r, under Birth-death (Bd) and
death-Birth (dB) updating. Paper draft due mid-November 2026.

## Layout (the whole package; keep it this small)
```
src/moran_graphs/
  PopulationGraph.py  # Graph dataclass + every factory (lungs, avian, fish, random, ...)
  properties.py       # structural metrics as pure functions of a Graph
  engine.py           # simulate() and (later) exact(); same output columns
  plan.py             # batch plans: JSON recipes, seeding, zoo building, dedup
  hpc.py              # one LSF array submitter + aggregation
  plots.py            # (planned) one function per paper figure, takes a tidy DataFrame
rust/engine/          # current Moran simulation engine (pyo3, built into moran_graphs._engine)
rust/exact/           # Hila's exact solver for small N: standalone binary, JSON in, JSON out
```
Possible later ports, only when I ask: the C++ Moran simulation (pybind11, tag
`v1-archive`) as a faster `simulate()` backend, and `rust/exact/` behind `exact()`.
Until then, `simulate()` runs on `rust/engine/`.
Data lives outside the repo (path from config), never inside it.

## Core design decisions (flag code that breaks these)
- **Graph = weights + node attributes + recipe, nothing computed.** W is a
  `scipy.sparse.csr_matrix`, row-normalized when the engine uses it. Unweighted means all
  weights equal to 1; directed means W is not symmetric (no `directed` flag).
  `node_attrs: dict[str, np.ndarray]` for roles such as trachea/bronchus/alveolus.
  `recipe = (factory_name, kwargs)` uses the factory's own kwarg names, so any graph
  can be rebuilt from its recipe.
- **Identity is a hash** (Weisfeiler-Lehman, weight-aware). Properties, results and plans
  all join on `(hash, r)`.
- **Properties are computed separately**, never at construction, and stored as a table.
- **One measurement interface**: `simulate(graph, r, n_repeats, update_rule, seed)` and
  `exact(graph, r, update_rule)` return the same columns. The engines are swappable.
- **Batch plans are recipes, not graphs**: factory + kwargs + count, r values, n_repeats,
  `zoo_seed`, `batch_seed`. Spec i's RNG stream is `SeedSequence(zoo_seed).spawn(n)[i]`,
  so appending a spec never changes earlier graphs. Dedup by hash; seedable specs redraw
  on collision.
- **Builders vs readers**: anything expensive is built by a job and only read by
  notebooks. A reader raises if its file is missing; it never computes silently.

## Correctness checklist (highest priority in review)
- **Statistics**: rho and time must match the archived v1 engine within Monte Carlo
  error on the validation graphs (complete, cycle, star, one lung, one random).
  The exact solver must match simulation on the same graphs, including directed ones.
- **Censoring**: a run that hits `max_steps` is recorded as `censored=True`, never as
  an extinction. Mean time is conditional on fixation; say so wherever it is computed.
- **Seeding**: every random draw traces back to a seed recorded in the output.
- **Update rules**: Bd = pick reproducer by fitness, then neighbor by W row.
  dB = pick dying node uniformly, then parent among in-neighbors by fitness x weight.
  Check that the direction of W is used consistently in both.
- **Out-degree 0** (directed graphs): a wasted step, not an error, in every engine.

## Known infrastructure lessons (WEXAC / LSF)
- The login node kills CPU-heavy processes (exit 144). Nothing heavy runs outside bsub/inode.
- `/tmp` is node-local: bsub scripts and `-o/-e` paths go under /home/labs.
- SSUSP is not a terminal state; short-queue jobs get preempted and restart from scratch.
- Pipe bsub scripts on stdin; command strings get re-evaluated and mangled.
- Never fuse result shards into one Parquet file (polars u32 row limit); scan as a glob.

## Style rules
- Keep it simple. Flag over-engineering: unneeded classes, config layers, compatibility
  shims, abstractions with one caller. No legacy support for v1 batches.
- No new files or modules without a stated reason.
- Match the surrounding code; comments explain *why*, briefly.
- Never use em dashes.
- Before suggesting a change that involves a design decision (algorithm, filtering,
  statistics), explain the alternatives and wait for my answer.
- Do not run heavy computations. Ask before running anything beyond a quick unit test.

## Writing code
Simple is best. The simplest code that is correct and fast wins.
- Clear names that say what a thing is (`fixation_prob`, not `fp` or `res`). No
  abbreviations except standard ones (`W`, `r`, `N`, `rng`).
- Short, flat functions. Plain functions over classes; a class only when it holds state.
- Comments only where the code cannot say it: a non-obvious *why*, a formula's source.
  No docstrings that restate the signature, no section banners, no commented-out code.
- Fast by default: vectorize with numpy/scipy, keep hot loops in the compiled engine,
  no needless copies or Python loops over nodes or repeats.
- No defensive checks for cases that cannot happen, no unused parameters, no
  "for future use" hooks. Fail loudly on bad input instead of guessing.
- Smallest diff that does the job. Touch only what the task needs.

## Out of scope unless I ask
GA search and ML predictors (archived in v1 at tag `v1-archive`, figures pinned to
commits there). Null-step skipping (Chatterjee et al.) is a possible later engine, not now.

## Review output format
1. Correctness bugs (with a concrete failing input), most severe first.
2. Violations of the design decisions above.
3. Simplifications.
Keep each item to a few lines. If nothing is wrong, say so.

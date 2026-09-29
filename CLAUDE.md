
# Reviewer brief: lung-selection

You are reviewing code in `lung-selection`, the rewrite of an MSc thesis codebase
(Weizmann, Computational Biology). I write the code myself. Your job is to REVIEW, not
to write it: explain problems, point to `file:line`, suggest the fix in words or a short
snippet, and let me apply it. Do not rewrite whole functions unless I ask.

## Research question
Do respiratory topologies (mammalian lungs, including asymmetric and mouse variants;
avian lungs; fish gills) act as evolutionary amplifiers or suppressors of selection,
compared to random graphs of the same size? Metrics: fixation probability (rho) and
fixation time, as a function of the selection coefficient r, under Birth-death (Bd) and
death-Birth (dB) updating. Paper draft due mid-November 2026.

## Layout (the whole package; keep it this small)
```
src/lungsel/
  graph.py       # Graph dataclass + every factory (lungs, avian, fish, random, ...)
  properties.py  # structural metrics as pure functions of a Graph
  engine.py      # simulate() -> C++ core, exact() -> Rust solver; same output columns
  plan.py        # batch plans: JSON recipes, seeding, zoo building, dedup
  hpc.py         # one LSF array submitter + aggregation
  plots.py       # one function per paper figure, each takes a tidy DataFrame
  _cpp/          # Moran hot loop (pybind11)
  _rust/         # exact solver for small N
```
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

## Out of scope unless I ask
GA search and ML predictors (archived in v1 at tag `v1-archive`, figures pinned to
commits there). Null-step skipping (Chatterjee et al.) is a possible later engine, not now.

## Review output format
1. Correctness bugs (with a concrete failing input), most severe first.
2. Violations of the design decisions above.
3. Simplifications.
Keep each item to a few lines. If nothing is wrong, say so.

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

## Out of scope unless I ask
GA search and ML predictors (archived in v1 at tag `v1-archive`, figures pinned to
commits there). Null-step skipping (Chatterjee et al.) is a possible later engine, not now.

## Review output format
1. Correctness bugs (with a concrete failing input), most severe first.
2. Violations of the design decisions above.
3. Simplifications.
Keep each item to a few lines. If nothing is wrong, say so.

★ Insight ─────────────────────────────────────
- The brief states what the code must preserve (design decisions, the correctness checklist) and says very little about how it is implemented. That keeps it valid while the code changes. The old CLAUDE.md described implementations, so every refactor made it stale.
- The "REVIEW, not write" framing at the top is doing the real work for your fluency goal. Without it, a reviewing agent tends to hand back rewritten code, and you're back to not knowing your own repo.
- The one line with weight: the dB definition mentions in-neighbors. On a directed W, "who can replace me" and "whom can I replace" differ, and that is exactly the kind of bug that stays silent on undirected graphs and only shows up on directed ones.
  ─────────────────────────────────────────────────

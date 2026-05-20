# Moran Process — Task List

Last updated: 2026-05-19 

What I was doing 2026-05-19:
  1. added graph-zoo class and graph_design.ipynb files for quick graph zoo designing. 
  2. Cleaned and changed the analysis files, now they work on old batches and new
  3. small edits in the ml section. nothing big. 

## 0. Pre-flight (do these first, this week)

- [x] **Decide on the reorg merge.** `organize-files` holds the cleaner `src/moran_process/` layout from 2026-03. Master has moved on only to add `*.md` context docs (commit `98c7f2b`). Plan:
  - [x] Merge `master` into `organize-files` (brings the new `.md` docs forward; small, low conflict).
  - [x] AI-generated tests are untrusted and will be ignored. New tests to be written from scratch later.
    - [x] Check out `organize-files`. Smoke-test `main.py` with a tiny `n_repeats` value, not a real submission.
    - [x] If working: merge `organize-files` into `master`.
- [x] **Decide what to do with `polars_branch`.** It sits on top of `organize-files` and adds (a) polars in `an
alysis_utils.py`, (b) a `cli.py`, (c) a `merge_batches` rewrite. After the reorg merge lands, rebase this onto `master` and keep it as the next feature branch. - Decided to abandone this branch
- [x] **Delete clutter:**
  - [x] `git branch -D feature/extreme_graphs` — every commit is already in master (merge base = its own HEAD).
  - [x] `git branch -D feature/add-graphs-to-batch` — same (merge base = its own HEAD).
  - [x] `git push origin --delete feature/add-graphs-to-batch docs/context-files feature/add_positions` once you've confirmed nothing in them is missing from master. `docs/context-files` is an earlier copy of the docs already in `master`; `feature/add_positions` is from 2025-12 and looks superseded — open it and double-check before deleting.
- [x] **Stash or commit the working-tree noise:** `ORGANIZATION.md` is `D` (you deleted it, makes sense — it overlaps with `PROJECT_OVERVIEW.md`), and `analysis/analyse_tests.ipynb` has a small modification. Commit the deletion (with a short message) and either commit or revert the notebook so the tree is clean before switching branches.

## 1. Research / science TODOs (from PROJECT_OVERVIEW.md)

- [ ] Count steps to **extinction**, not just steps to fixation (currently both end the run; only fixation time is recorded distinctly).
- [x] Make analysis pipeline scale to >10 GB batches (the notebooks `pd.concat` everything in memory today).
- [ ] Speed up the simulation. Options: C++ rewrite of the Moran step; Cython/Numba on `process_run.step()`; multiprocessing for `n_repeats`. Pick one and prototype.
- [ ] Multi-color / multi-type Moran (more than 2 species). Implies abstracting `ProcessRun` (your task list already flagged this).
- [ ] Explore a GNN approach (Yael's suggestion). Decide first: train on fixed-size N=31 graphs, or variable-size?
- [ ] Justify N=31 as the simulation size. Run a small sweep over N to show results are qualitatively consistent.
- [ ] Add the analytical fixation-probability formula as a reference line on plots.
- [ ] Read papers on the to-read list:
  - [ ] Uri Alon — network motifs.
  - [ ] Roy Kishony (2011) — parallel bacterial evolution.

## 2. Engineering / refactor TODOs
- [x] Create a graph zoo interface.
- [ ] Make analysis able to work with large datasets. Maybe switch to polars.
- [ ] Treat Future Warning in violin plots.
- [ ] Instead of calculating every time the figures again in a notebook, make a script that makes it as a job maybe and saves the figures. 
- [ ] Make `Process` an abstract base class so multi-color / variant rules can subclass it.
- [ ] Move analysis figures behind a function: notebooks should just import and call, not redefine plotting code per notebook.
- [ ] Add a proper CLI (Typer or Click).
- [ ] Add a built-in `aggregate_batch()` method on `ProcessLab` so we stop hand-rolling the `glob + pd.concat` snippet.
- [ ] fix the way logs are made. All logs associated to a batch should be in it's directroy with a meaningful name.




## 3. Workflow / habit TODOs

- Add a one-line "what I was doing" note to the top of this file at the end of each work session, so the next return is easier.
- Stop committing notebook output diffs — install `nbstripout` (`uv add --dev nbstripout && nbstripout --install`) so checkpoints stop polluting commits.

---

## Desired workflow (kept from previous version)

1. Create a graph zoo
2. Run a simulation on the zoo
3. Analyze the simulation results
4. Train ML models on the results
5. Use simulated annealing / EA to extremize fixation time/probability

## Target file layout (already realised on `organize-files` / `polars_branch`)

```
moran-process/
├── pyproject.toml
├── uv.lock
├── README.md
├── task_list.md
├── submit_main.sh
├── tests/
├── notebooks/
├── simulation_data/      # gitignored
└── src/moran_process/
    ├── __init__.py
    ├── cli.py
    ├── core/             # population_graph.py
    ├── simulations/      # process_run.py
    ├── pipeline/         # process_lab.py, worker_wrapper.py, main.py
    └── analysis/         # analysis_utils.py
```
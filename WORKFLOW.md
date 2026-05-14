# My Working Workflow

A personal cheat sheet for how to split work, branch, run, and merge on this project.
This is for the **single-developer** case (just me), so the rules are simple on purpose.

## Where I work

| Place | What for |
|---|---|
| WEXAC (this machine) | The canonical copy. All real work — coding, branching, running, analysis — happens here. |
| Office Windows PC | Only secondary editing if I'm away from the cluster; sync via git, not OneDrive. |
| `git push origin` | Backup + a way to see what's where from the web. Not a collaboration channel. |

**Rule:** WEXAC is source of truth. If something exists locally on WEXAC but not on `origin`, that's normal — push when a chunk of work is "done enough to back up."

## Compute placement (from VSCODE_WEXAC_WORKFLOW.md, in one line)

- Editing / git / `bsub` submission → **login node** (fine).
- Running scripts / notebooks heavier than ~10 s or ~1 GB → **`inode`** or **`ijup`** terminal.
- Real simulations → **`bsub`** via `main.py` / `submit_main.sh`. Never run a full sweep from the login node.

## Branching rules

### When to make a new branch

Make a branch whenever the change is bigger than a small fix and might leave the code in a half-working state for more than a day. Specifically:

- **Yes, branch:** new feature (multi-color Moran, GNN), a refactor (`Process` ABC, src/ reorg), an experiment-only change (new graph type, new analysis pipeline), anything I might want to throw away.
- **No, don't branch:** typo, README/doc edit, single-file bugfix, regenerating a figure I already plot. Commit straight to `master`.

If in doubt: branch. Branches on a solo project are free.

### Naming

- `feature/<short-kebab-name>` for new functionality (`feature/multi-color`, `feature/cpp-step`).
- `refactor/<name>` for "same behaviour, different shape" (`refactor/process-abc`).
- `experiment/<name>` for code that runs simulations but isn't meant to be merged (`experiment/N-sweep`). Don't be afraid to delete these after the results are extracted.
- `wip/<name>` for scratch — assume it gets thrown away.

### Lifetime

A branch should live **at most a couple of weeks**. If it lives longer:

- It's almost certainly drifted from `master` (this just happened: `organize-files` sat for 2 months while `master` got new docs).
- Rebase or merge `master` into it weekly so the eventual merge-back is small.
- If a branch keeps not landing, ask: is it actually wanted? If not, delete it.

### Commit style

- Commit message: imperative mood, short subject line, then optional body. Examples: `add fish_graph factory`, `fix off-by-one in worker chunking`.
- Avoid `wip` as the only message. If it's truly a checkpoint, use `wip: <what>` so future-me has a hint. (My git log has too many bare `wip` entries already.)
- Don't commit notebook output diffs. Run `nbstripout --install` once and let it handle this.
- Don't commit large binaries to git — `simulation_data/`, `graph_zoos/*.joblib`, log files. They belong in `.gitignore`.

## The default loop

```
1. Pull master           git checkout master && git pull
2. Branch                git checkout -b feature/<name>
3. Work                  edit, run, test
4. Commit often          small, named commits
5. Periodically merge in master   (avoid drift)
6. When done:
   a. Run tests          uv run pytest tests/
   b. Smoke-test the script you touched
   c. Merge to master    git checkout master && git merge --no-ff feature/<name>
   d. Push               git push
   e. Delete branch      git branch -d feature/<name>
```

Use `--no-ff` on the final merge so the branch shows up as a unit in `git log --graph` — easier to see what I did when I come back in 3 months.

## When to merge vs rebase

Solo project, so the distinction barely matters. Rule of thumb:

- **Merge** when bringing `master` into a long-lived branch (preserves history honestly).
- **Rebase** when cleaning up local commits on a short-lived branch before merging back to `master` (`git rebase -i master` to squash the `wip`/`fix typo` noise).
- Never rebase commits that are already pushed and might be checked out somewhere else. (Edge case for a solo project, but the habit matters.)

## Running experiments

Experiments aren't code changes — they're runs. Treat them differently:

- A simulation **run** lives in `simulation_data/tmp/batch_<name>/`. The batch name is the experiment ID.
- Code that *defines* the experiment (`main.py`'s config block, an `experiment/<name>` branch) should be committed before launching the `bsub` array. Tag the commit if it matters: `git tag exp/<batch-name>`.
- If I'm tweaking `main.py` to try different configs, do it on an `experiment/<name>` branch, not `master`. Don't push experiment-branch commits unless I want the config preserved.

## Hygiene: returning after a long break

Steps for any time I come back to this project after >1 month away:

1. `git status`, `git branch -a`, `git log --oneline -10` to see where I left off.
2. Read `task_list.md` (top of file, where I left the "what I was doing" note).
3. Look for branches with no recent commits — if they're fully merged into master, delete them. (`git branch --merged master`.)
4. `uv sync` to make sure the venv matches the lockfile.
5. `uv run pytest tests/` to confirm nothing rotted.
6. Update the date at the top of `task_list.md`.

## Files that should be in `.gitignore` but might not be

- `simulation_data/` (or at least `simulation_data/tmp/` and the big result CSVs)
- `graph_zoos/*.joblib`
- `ml_models/*.joblib`
- `logs/`
- `__pycache__/`, `.venv/`, `.pytest_cache/`
- `*.ipynb_checkpoints`

Check before next commit.
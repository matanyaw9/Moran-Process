# HPC Workflow — WEXAC (Weizmann)

The WEXAC cluster uses the **LSF** scheduler. All job submission uses `bsub`.
Full docs: https://hpcwiki.weizmann.ac.il/en/home/lsf/basic

---

## Typical Workflow for a Simulation Batch

### Step 0 — Design the Graph Zoo (always first)
Open `notebooks/design_zoo.ipynb`. This is the entry point for every new batch.

1. Set `BATCH_NAME` at the top (e.g. `"2026-05-20_my_study"`).
2. Run the cells for the graph types you want (mammalian, avian, fish, complete, cycle, random, ...).
3. Visualize with `zoo.draw_all()` to confirm the topology.
4. Save: `zoo.save(f"../simulation_data/{BATCH_NAME}/zoo.pkl")`.

The saved `zoo.pkl` is the input to all downstream steps.

### Step 1 — Submit the Batch
Load the saved zoo and call `ProcessLab.submit_jobs()` (Section 4 of the notebook, or from a script):

```python
from moran_process import GraphZoo, ProcessLab

zoo = GraphZoo.load(f"../simulation_data/{BATCH_NAME}/zoo.pkl")
lab = ProcessLab()
lab.submit_jobs(
    zoo_path=f"../simulation_data/{BATCH_NAME}/zoo.pkl",
    r_values=[1.0, 1.1, 1.2, 1.3, 2.0],
    n_repeats=10_000,
    n_requested_jobs=1000,
    n_graphs=len(zoo),
    queue="gsla-cpu",
    batch_dir=f"../simulation_data/{BATCH_NAME}",
    batch_name=BATCH_NAME,
)
```

This call:
1. Submits a short `register_graphs` job that writes graph properties to `<batch_dir>/graph_props.csv`
2. Creates `task_manifest.csv` (all tasks enumerated)
3. Submits the main job array via `bsub`

### Step 2 — Monitor Jobs
```bash
bjobs                      # list your jobs
bjobs -w                   # wide format (see full names)
bjobs -l <job_id>          # detailed info (why pending?)
bpeek -f <job_id>          # follow stdout of a running job
```

### Step 3 — Collect Results
After all jobs finish, `simulation_data/tmp/batch_NAME/results/` will contain `result_job_1.csv` through `result_job_N.csv`. Aggregate:
```python
import glob, pandas as pd
files = glob.glob('simulation_data/tmp/batch_NAME/results/result_job_*.csv')
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df.to_csv('simulation_data/combined_results.csv', index=False)
```

### Step 4 — Analyze
Open notebooks in `analysis/` on the Windows PC (notebooks run locally using the OneDrive-synced data, or via Jupyter on WEXAC).

---

## Manual Worker Test (Debugging)
Run a single worker chunk without bsub:
```bash
python worker_wrapper.py --batch-dir simulation_data/tmp/batch_NAME --chunk-size 2 --job-index 14
# OR with uv:
uv run worker_wrapper.py --batch-dir simulation_data/tmp/test-batch --chunk-size 2 --job-index 14
```

---

## LSF Quick Reference

### Queues (WEXAC)
| Queue | Max Walltime | Use Case |
|---|---|---|
| `short` | 30 min | Testing, small runs |
| `new-short` | ~12 hours | Standard short jobs |
| `medium` | 48 hours | Production runs |
| `long` | 7+ days | Very long simulations |
| `idle` | Unlimited | Zero priority, can be killed |

Check available queues: `bqueues`

### Common bsub Options
```bash
bsub -q <queue>                   # queue
bsub -n <cores>                   # number of cores
bsub -R "rusage[mem=2048]"        # memory per core (MB)
bsub -R "span[hosts=1]"           # all cores on same node
bsub -W 4:00                      # walltime HH:MM
bsub -J "name[1-N]"               # job array (indices 1 to N)
bsub -o logs/job_%J_%I.out        # stdout (%J=jobID, %I=arrayIndex)
bsub -e logs/job_%J_%I.err        # stderr
```

### Job Control
```bash
bkill <job_id>     # kill one job
bkill 0            # kill ALL your jobs
bstop <job_id>     # pause
bresume <job_id>   # resume
```

### Interactive Session (for debugging on compute node)
```bash
bsub -Is -q new-short -n 2 -W 30 bash
```

### Environment Variables in Workers
- `LSB_JOBINDEX` — the array index (1-based). `worker_wrapper.py` reads this automatically.

---

## Key Paths on WEXAC
The project must be uploaded/synced to WEXAC before running. The typical structure mirrors the OneDrive structure. Worker scripts use **relative paths** from the `Moran-Process/` directory.

- Batch output: `Moran-Process/simulation_data/tmp/batch_<name>/`
- Graph database: `Moran-Process/simulation_data/graph_database.csv`
- Logs: `Moran-Process/simulation_data/tmp/batch_<name>/logs/`
- Results: `Moran-Process/simulation_data/tmp/batch_<name>/results/`

---

## Notes & Gotchas
- `uv run` is used instead of `python` on the cluster (manages the venv).
- The graph database path is **hardcoded** as `simulation_data/graph_database.csv` — scripts must be run from `Moran-Process/`.
- Each worker job processes `chunk_size = ceil(total_tasks / n_jobs)` tasks. The last job may have fewer.
- If `batch_dir` already exists, jobs append/overwrite (no auto-clean).
- Memory request of `2048` MB (`-R "rusage[mem=2048]"`) is the current default. Increase for larger graphs.

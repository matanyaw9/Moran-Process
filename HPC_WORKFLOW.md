# HPC Workflow — WEXAC (Weizmann)

The WEXAC cluster uses the **LSF** scheduler. All job submission uses `bsub`.
Full docs: https://hpcwiki.weizmann.ac.il/en/home/lsf/basic

---

## Typical Workflow for a Simulation Batch

### Step 1 — Prepare the Batch (on Windows PC or WEXAC login node)
Run `main.py` (or a modified version). This calls `ProcessLab.submit_jobs()`, which:
1. Pickles the graph list to `simulation_data/tmp/batch_NAME/graphs.pkl`
2. Creates `task_manifest.csv` (all tasks enumerated)
3. Calls `bsub` to submit the job array

```python
lab = ProcessLab()
lab.submit_jobs(
    graph_zoo, r_values,
    n_repeats=10_000, n_jobs=1000,
    queue="short", memory="2048",
    batch_name="my_experiment"
)
```

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

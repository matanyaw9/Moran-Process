# VS Code + WEXAC Workflow

## The Core Rule

> If a script takes more than a few seconds or uses significant memory — don't run it on the login node.

VS Code is connected via Remote SSH to the **login node**. The login node is for editing, git, and job submission only. Heavy compute must go to a compute node via `bsub`.

---

## Where Things Actually Run

| Action | Runs On |
|---|---|
| Editing a file in VS Code | Login node (fine) |
| Green ▶ Run button / F5 | Login node — **be careful** |
| `bsub` command in terminal | Submits to compute node (fine) |
| `inode` → then run in that terminal | Compute node ✓ |
| `ijup` → paste URL into VS Code kernel | Compute node ✓ |

**The green button always opens a fresh terminal on the login node**, even if you have an `inode` session open in another terminal. There is no way to make it use the compute node without reconnecting VS Code entirely (not worth it).

---

## The Three Tools

### 1. `bsub` — Batch jobs (simulations)
For actual experiment runs. Fire and forget.
```bash
# From main.py / process_lab.py — this is the normal simulation workflow
python main.py   # calls lab.submit_jobs() internally
```

### 2. `inode` — Interactive compute shell
For debugging scripts, testing code that would be too heavy for the login node.
```bash
inode              # 8GB RAM, 4h, gsla-cpu queue
inode 16GB         # more memory
inode 4GB 1:00     # short debugging session

# Once inside: run a single worker slice (PYTHONPATH=src; run as a module)
cd ~/Moran-Process
PYTHONPATH=src uv run python -m moran_process.pipeline.worker_lsf \
    --zoo-shard-dir simulation_data/<BATCH_NAME>/tmp/zoo_shards \
    --manifest-path simulation_data/<BATCH_NAME>/tmp/task_manifest.csv \
    --batch-dir simulation_data/<BATCH_NAME>/tmp \
    --job-index 1
```
Edit files in VS Code as usual — login node and compute nodes share the same filesystem, so edits are visible instantly.

### 3. `ijup` — JupyterLab on a compute node (notebooks)
For running analysis notebooks that need real memory/CPU.
```bash
ijup               # 16GB RAM, 4h
ijup 32GB 8:00     # bigger session
```
When the URL appears, copy it → VS Code → bottom-right kernel selector → **"Select Another Kernel"** → **"Existing Jupyter Server"** → paste URL.

---

## Decision Flowchart

```
What are you doing?
│
├── Editing code / git / bsub commands
│   └── Just use VS Code normally on the login node ✓
│
├── Running a Jupyter notebook
│   └── ijup → paste URL into VS Code kernel selector ✓
│
├── Running a script interactively (debugging, testing)
│   ├── Will it finish in < 10s and use < 1GB RAM?
│   │   └── Green button or terminal on login node is fine ✓
│   └── Heavier than that?
│       └── inode → run from that terminal ✓
│
└── Running a real simulation
    └── bsub (via main.py / process_lab.py) ✓
```

---

## `.bashrc` Functions

Both functions live in `~/.bashrc` on WEXAC.

```bash
inode [mem] [walltime] [queue]   # default: 8GB, 4:00, gsla-cpu
ijup  [mem] [walltime] [queue]   # default: 16GB, 4:00, gsla-cpu
```

---

## Should You Connect VS Code to the Compute Node?

**No.** When the interactive job expires, VS Code loses the connection mid-session. Not worth it. The `inode` terminal approach gives you compute node access for running scripts, and `ijup` handles notebooks — VS Code itself stays on the login node throughout.

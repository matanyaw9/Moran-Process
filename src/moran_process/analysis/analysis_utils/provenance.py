"""
Batch metadata and run provenance: read/write ``batch_info.json`` and snapshot
the git/host/python state a batch was launched from.

Deliberately matplotlib-free (stdlib only) so the submission path
(``process_lab.create_batch_info``) does not pull in the plotting stack. The
title-card figure built from this metadata lives in ``plots.plot_batch_info_card``.
"""

import sys
import socket
import subprocess
import json
from pathlib import Path
from datetime import datetime

__all__ = [
    "load_batch_info",
    "capture_provenance",
    "create_batch_info",
]


def load_batch_info(batch_dir) -> dict:
    """Read batch_info.json from batch_dir; returns name-only fallback if not found."""
    path = Path(batch_dir) / "batch_info.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"name": Path(batch_dir).name, "description": ""}


def _git(args) -> str | None:
    """Run a read-only git command and return stripped stdout, or None on failure
    (not a repo, git missing, detached/odd state). Never raises, so a provenance
    hiccup can't abort a batch submission."""
    try:
        out = subprocess.check_output(
            ["git", *args], stderr=subprocess.DEVNULL, text=True
        )
        return out.strip()
    except Exception:
        return None


def capture_provenance() -> dict:
    """Snapshot the environment a batch was launched from.

    Every field is read automatically -- the user types nothing. ``git_dirty``
    flags whether there were uncommitted changes at submit time, so you can tell
    whether ``git_commit`` fully describes the code that ran.
    """
    status = _git(["status", "--porcelain"])
    return {
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": None if status is None else bool(status.strip()),
        "command": " ".join(sys.argv),
        "python": sys.version.split()[0],
        "hostname": socket.gethostname(),
    }


def create_batch_info(
    batch_dir,
    name,
    description="",
    notes="",
    # simulation parameters
    r_values=None,
    n_repeats=None,
    total_simulations=None,
    batch_seed=None,
    engine=None,
    max_steps=None,
    # zoo description
    n_graphs=None,
    graph_types=None,
    node_sizes=None,
    zoo_path=None,
    zoo_config=None,
    # HPC submission
    n_requested_jobs=None,
    queue=None,
    memory_mb=None,
    job_array_name=None,
    lsf_job_id=None,
    bsub_command=None,
) -> dict:
    """Write a nested, fully-provenanced batch_info.json. Overwrites any existing file.

    The intent is that this file alone documents how the batch was created and
    run. Only ``description`` and ``notes`` are author-supplied; everything else
    is captured from the values submit_jobs already holds plus auto-read
    provenance (git/host/python/command).

    Sections:
        provenance: git commit/branch/dirty, launch command, python, hostname
        zoo:        what was simulated on -- graph counts, types, sizes, and the
                    creation recipe (seed + random-graph config + biological specs)
                    threaded in via ``zoo_config``
        simulation: r values, repeats, total sims, batch seed, engine
        hpc:        job count, queue, memory, LSF job array name + parsed job id
    """
    zoo_config = zoo_config or {}
    info = {
        "name": name,
        "description": description,
        "notes": notes,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "provenance": capture_provenance(),
        "zoo": {
            "n_graphs": n_graphs,
            "graph_types": graph_types or [],
            "node_sizes": node_sizes or [],
            "zoo_path": str(zoo_path) if zoo_path is not None else None,
            **zoo_config,
        },
        "simulation": {
            "r_values": r_values or [],
            "n_repeats": n_repeats,
            "total_simulations": total_simulations,
            "batch_seed": batch_seed,
            "engine": engine,
            "max_steps": max_steps,
        },
        "hpc": {
            "n_requested_jobs": n_requested_jobs,
            "queue": queue,
            "memory_mb": memory_mb,
            "job_array_name": job_array_name,
            "lsf_job_id": lsf_job_id,
            "bsub_command": bsub_command,
        },
    }
    path = Path(batch_dir) / "batch_info.json"
    with open(path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"[batch_info] Written: {path}")
    return info


def _bi_get(batch_info, *path, default=None):
    """Fetch a field from a (possibly nested) batch_info dict.

    Tries the nested path first (e.g. ('simulation', 'r_values')), then falls
    back to the last key at the top level so legacy flat batch_info.json files
    written before the restructure still resolve.
    """
    node = batch_info
    for key in path:
        if isinstance(node, dict) and key in node:
            node = node[key]
        else:
            node = None
            break
    if node is not None:
        return node
    return batch_info.get(path[-1], default)

"""Running a batch plan on WEXAC LSF.

    python -m moran_graphs.hpc submit plan.json [--dry-run | --local]
    python -m moran_graphs.hpc status   <batch>
    python -m moran_graphs.hpc properties <batch>
    python -m moran_graphs.hpc finalize <batch>
    python -m moran_graphs.hpc resume   <batch> [--n-workers K] [--local]

submit (login node, light) -> prepare job -> worker array + properties job -> finalize job.
The properties job runs alongside the workers, so slow metrics (betweenness) do not hold
up the simulations. If prepare fails, LSF kills every job waiting on it (bsub -ti).
--dry-run builds every graph to catch plan errors before anything is queued; for a
large plan that takes long enough to need an inode.

Workers pull chunks from one work list. A chunk is claimed by atomically creating
claims/<chunk_id> (O_EXCL: of two workers, exactly one succeeds), so every chunk runs
once and the batch ends about one chunk after the first worker runs out of work.
Each chunk has its own seed, so who runs it does not change its numbers.

Batch directory (<batch> is a name under MORAN_GRAPHS_DATA, or a path):
    plan.json, submissions.json   the plan and the LSF job ids of each submission
    graphs.npz, graphs.parquet    every graph as CSR, and its recipe and N
    chunks.parquet                the work list, written last by prepare
    properties.parquet            structural metrics per graph, from the properties job
    claims/, journals/            who claimed what (a journal lets a requeued job redo its own)
    results/<chunk_id>.parquet    summaries + timing of a finished chunk
    runs/<chunk_id>.parquet       every run, only if plan.sim.keep_runs
    report/                       results, histograms, chunk_timing, verification, speed
    logs/                         LSF stdout/stderr
"""

import argparse
import itertools
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from moran_graphs.engine import CENSORED, HIST_BINS, WILD_TYPE, simulate, summarize
from moran_graphs.plan import build_zoo, describe, load_plan, make_chunks, save_plan
from moran_graphs.properties import graph_from_W, properties

DATA_ROOT = Path(os.environ.get("MORAN_GRAPHS_DATA", "/home/labs/pilpel/matanyaw/moran-graphs-data"))
ALIVE = {"PEND", "RUN", "PSUSP", "USUSP", "SSUSP", "WAIT"}  # SSUSP is preempted, not finished


def batch_path(batch) -> Path:
    path = Path(batch)
    return path if path.is_absolute() or path.exists() else DATA_ROOT / batch


def _write_parquet(df, path):
    """Write then rename, so a killed job never leaves a half-written file. The temp name
    starts with '.', which pyarrow skips when reading a directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _log(msg):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


# --- graphs as concatenated CSR ---------------------------------------------------


def save_csr(path, matrices):
    n = np.array([W.shape[0] for W in matrices])
    np.savez(
        path,
        n=n,
        indptr=np.concatenate([W.indptr for W in matrices]),
        indices=np.concatenate([W.indices for W in matrices]),
        weights=np.concatenate([W.data for W in matrices]).astype(np.float64),
        ptr_start=np.concatenate([[0], np.cumsum(n + 1)]),
        edge_start=np.concatenate([[0], np.cumsum([W.nnz for W in matrices])]),
    )


def read_csr(z, i):
    n, p0, e0 = int(z["n"][i]), z["ptr_start"][i], z["edge_start"][i]
    e1 = z["edge_start"][i + 1]
    return sp.csr_array(
        (z["weights"][e0:e1], z["indices"][e0:e1], z["indptr"][p0 : p0 + n + 1]), shape=(n, n)
    )


# --- LSF -----------------------------------------------------------------------


def _bsub(batch, label, args, queue, memory_mb, array=None, wait=None):
    """Submits `python -m moran_graphs.hpc <args>` with the script piped on stdin (a
    command string would be re-evaluated by the shell). Returns the numeric job id."""
    name = f"{batch.name}_{label}" + (f"[1-{array}]" if array else "")
    lines = [
        "#!/bin/bash",
        f"#BSUB -J {name}",
        f"#BSUB -q {queue}",
        f'#BSUB -R "rusage[mem={memory_mb}]"',
        f"#BSUB -o {batch}/logs/{label}_%J_%I.out",
        f"#BSUB -e {batch}/logs/{label}_%J_%I.err",
        # -ti kills the job as soon as `wait` can never hold (e.g. prepare failed), instead of
        # leaving it pending forever.
        *([f'#BSUB -w "{wait}"', "#BSUB -ti"] if wait else []),
        # The venv python directly, not `uv run`: a thousand jobs must not each sync the env.
        f"exec {sys.executable} -u -m moran_graphs.hpc {args}",
    ]
    out = subprocess.run(["bsub"], input="\n".join(lines) + "\n", text=True, capture_output=True)
    match = re.search(r"Job <(\d+)>", out.stdout)
    if out.returncode or not match:
        raise RuntimeError(f"bsub failed for {name}: {out.stdout} {out.stderr}")
    _log(f"submitted {name}: job {match.group(1)}")
    return match.group(1)


def _record_submission(batch, **ids):
    path = batch / "submissions.json"
    subs = json.loads(path.read_text()) if path.exists() else []
    subs.append({"time": time.time(), **ids})
    path.write_text(json.dumps(subs, indent=2) + "\n")


def _submit_workers(batch, plan, n_workers, prepare=None, properties=None):
    lsf = plan["lsf"]
    workers = _bsub(batch, "worker", f"worker {batch} --n-workers {n_workers}", lsf["queue"],
                    lsf["memory_mb"], array=n_workers, wait=f"done({prepare})" if prepare else None)
    # ended, not done: one failed worker must not block it. done(properties) too, which needs
    # prepare done, because workers killed after a failed prepare also count as ended.
    wait = f"ended({workers})" + (f" && done({properties})" if properties else "")
    final = _bsub(batch, "finalize", f"finalize {batch}", lsf["queue"], lsf["big_memory_mb"], wait=wait)
    return workers, final


def _alive_jobs():
    """'jobid index' of every unfinished LSF job of this user (index 0 for non-arrays)."""
    out = subprocess.run(["bjobs", "-noheader", "-o", "jobid jobindex stat"], text=True, capture_output=True)
    return {f"{j} {i}" for j, i, stat in (line.split() for line in out.stdout.splitlines()) if stat in ALIVE}


def _run_local(batch, n_workers, prepare_first=False):
    """The LSF chain as local processes, for testing on an inode."""
    py = [sys.executable, "-u", "-m", "moran_graphs.hpc"]
    if prepare_first:
        subprocess.run([*py, "prepare", str(batch)], check=True)
        subprocess.run([*py, "properties", str(batch)], check=True)
    job = f"local{int(time.time())}"
    procs = [
        subprocess.Popen([*py, "worker", str(batch), "--n-workers", str(n_workers)],
                         env={**os.environ, "LSB_JOBID": job, "LSB_JOBINDEX": str(i)})
        for i in range(1, n_workers + 1)
    ]
    for p in procs:
        p.wait()
    return finalize(batch)


# --- commands ------------------------------------------------------------------


def submit(plan_path, dry_run=False, local=False):
    plan = load_plan(plan_path)
    batch = batch_path(plan["batch_name"])
    if dry_run:
        # The full check: building the zoo is what raises plan errors, such as too few
        # distinct random graphs. A real submit leaves the building to the prepare job.
        describe(plan)
        print(f"  {len(build_zoo(plan))} distinct graphs")
        print(f"[dry run] would create {batch}")
        return
    if (batch / "plan.json").exists():
        raise FileExistsError(f"{batch} already holds a batch: choose a new batch_name, or resume it")
    (batch / "logs").mkdir(parents=True, exist_ok=True)
    save_plan(plan, batch / "plan.json")
    if local:
        return _run_local(batch, plan["lsf"]["n_workers"], prepare_first=True)
    lsf = plan["lsf"]
    prep = _bsub(batch, "prepare", f"prepare {batch}", lsf["queue"], lsf["big_memory_mb"])
    props = _bsub(batch, "properties", f"properties {batch}", lsf["queue"], lsf["big_memory_mb"],
                  wait=f"done({prep})")
    workers, final = _submit_workers(batch, plan, lsf["n_workers"], prepare=prep, properties=props)
    _record_submission(batch, prepare=prep, properties=props, workers=workers, finalize=final,
                       n_workers=lsf["n_workers"])


def prepare(batch):
    """Builds the graphs and the work list. Properties come later, from compute_properties."""
    batch = batch_path(batch)
    plan = load_plan(batch / "plan.json")
    t0 = time.time()
    zoo = build_zoo(plan)
    _log(f"built {len(zoo)} graphs in {time.time() - t0:.1f}s")
    graphs = pd.DataFrame([{
        "graph_idx": idx, "hash": rec["hash"], "name": rec["graph"].name, "category": rec["graph"].category,
        "factory": plan["specs"][rec["spec_idx"]]["factory"], "spec_idx": rec["spec_idx"],
        "kwargs": json.dumps(rec["kwargs"]), "n_nodes": rec["W"].shape[0],  # make_chunks costs by N
    } for idx, rec in enumerate(zoo)])
    save_csr(batch / "graphs.npz", [rec["W"] for rec in zoo])
    _write_parquet(graphs, batch / "graphs.parquet")
    chunks = make_chunks(plan, graphs)
    _write_parquet(chunks, batch / "chunks.parquet")  # last: its existence means "prepared"
    _log(f"{len(chunks)} chunks for {plan['lsf']['n_workers']} workers, done in {time.time() - t0:.1f}s")


def compute_properties(batch):
    """Structural metrics of every graph in graphs.npz, keyed by hash."""
    batch = batch_path(batch)
    graphs = pd.read_parquet(batch / "graphs.parquet", columns=["graph_idx", "hash"])
    z = np.load(batch / "graphs.npz")
    t0 = time.time()
    rows = []
    for idx, h in zip(graphs["graph_idx"], graphs["hash"]):
        rows.append({"hash": h, **properties(graph_from_W(read_csr(z, idx)))})
        if (idx + 1) % 500 == 0:
            _log(f"properties of {idx + 1}/{len(graphs)} graphs")
    _write_parquet(pd.DataFrame(rows), batch / "properties.parquet")
    _log(f"properties of {len(graphs)} graphs in {time.time() - t0:.1f}s")


def worker(batch, n_workers, index=None):
    batch = batch_path(batch)
    index = index or int(os.environ["LSB_JOBINDEX"])
    job = os.environ.get("LSB_JOBID", "manual")
    me = f"{job} {index}"
    plan = load_plan(batch / "plan.json")
    sim = plan["sim"]
    chunks = pd.read_parquet(batch / "chunks.parquet")
    col = {k: chunks[k].to_numpy() for k in chunks}  # per-column arrays keep the uint64 seeds exact
    z = np.load(batch / "graphs.npz")
    claims, results = batch / "claims", batch / "results"
    claims.mkdir(exist_ok=True)
    (batch / "journals").mkdir(exist_ok=True)
    journal = batch / "journals" / f"{job}.{index}"
    graph = {"idx": None, "W": None}  # chunks of one graph are contiguous, so one is enough

    def claim(cid):
        try:
            fd = os.open(claims / str(cid), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        os.write(fd, f"{me} {socket.gethostname()} {time.time()}\n".encode())
        os.close(fd)
        with open(journal, "a") as f:
            f.write(f"{cid}\n")
        return True

    def run(cid):
        gi, r, n_runs = int(col["graph_idx"][cid]), float(col["r"][cid]), int(col["n_runs"][cid])
        if graph["idx"] != gi:
            graph.update(idx=gi, W=read_csr(z, gi))
        t0, cpu0 = time.time(), time.process_time()
        res = simulate(graph["W"], [1], [r], n_runs, sim["max_steps"], int(col["seed"][cid]),
                       sim["update_rule"], bool(col["skip_null_steps"][cid]))
        t1, cpu1 = time.time(), time.process_time()
        if sim["keep_runs"]:
            runs = pd.DataFrame({k: res[k] for k in ("winner", "steps", "n_changes")})
            _write_parquet(runs, batch / "runs" / f"{cid}.parquet")
        summary = summarize(res).assign(
            chunk_id=cid, graph_idx=gi, hash=col["hash"][cid], r=r, job=job, index=index,
            host=socket.gethostname(), start=t0, end=t1, cpu=cpu1 - cpu0,
        )
        _write_parquet(summary, results / f"{cid}.parquet")  # its existence means "done"
        _log(f"chunk {cid}: graph {gi} r={r} {n_runs} runs in {t1 - t0:.1f}s")

    # A requeued job (preemption) comes back with the same id and index: finish what it claimed.
    if journal.exists():
        for cid in map(int, journal.read_text().split()):
            if not (results / f"{cid}.parquet").exists():
                run(cid)

    n = len(chunks)
    pos = (index - 1) * n // n_workers  # staggered starts: each worker stays on its own graphs
    done = 0
    while True:
        claimed = set(os.listdir(claims))  # one listing per chunk instead of a failed open per id
        free = next((c for c in itertools.chain(range(pos, n), range(pos)) if str(c) not in claimed), None)
        if free is None:
            break
        if claim(free):
            run(free)
            done += 1
            pos = free + 1
    _log(f"worker {me}: no chunks left, ran {done}")


def finalize(batch):
    """Verifies the batch and writes the report. Returns 1 if a check FAILed."""
    batch = batch_path(batch)
    plan = load_plan(batch / "plan.json")
    sim = plan["sim"]
    chunks = pd.read_parquet(batch / "chunks.parquet")
    graphs = pd.read_parquet(batch / "graphs.parquet")
    props = pd.read_parquet(batch / "properties.parquet").drop(columns="n_nodes")  # graphs has it
    graphs = graphs.merge(props, on="hash", validate="one_to_one")
    report = batch / "report"
    report.mkdir(exist_ok=True)
    have = sorted(int(p.stem) for p in (batch / "results").glob("*.parquet"))
    res = pd.read_parquet(batch / "results") if have else pd.DataFrame(columns=["chunk_id"])
    checks = []

    def check(ok, name, good, bad, level="FAIL"):
        checks.append({"status": "OK" if ok else level, "check": name, "message": good if ok else bad})

    missing = sorted(set(chunks.chunk_id) - set(have))
    claimed = {int(p.name) for p in (batch / "claims").glob("*")} if (batch / "claims").exists() else set()
    orphans = [c for c in missing if c in claimed]
    check(not missing, "chunks", f"all {len(chunks)} chunks have results",
          f"{len(missing)} of {len(chunks)} chunks have no result ({len(orphans)} claimed: running, or "
          f"their job died). Run `status`, then `resume` once no worker is alive. First: {missing[:10]}")
    planned = sum(s["count"] for s in plan["specs"])
    check(len(graphs) == planned, "graphs", f"{planned} graphs built",
          f"{len(graphs)} graphs built, plan asked for {planned} (duplicates dropped)", level="WARN")
    tmp = list((batch / "results").glob(".*.tmp"))
    check(not tmp, "partial_writes", "no partial result files",
          f"{len(tmp)} partial result files, from jobs killed while writing", level="WARN")

    if have:
        per_cell = res.groupby(["graph_idx", "r"])["count"].sum()
        wrong = per_cell[per_cell != sim["n_repeats"]]
        check(wrong.empty, "repeats", f"every (graph, r) has {sim['n_repeats']:,} runs",
              f"{len(wrong)} (graph, r) cells do not have {sim['n_repeats']:,} runs")
        check(res["hash"].isin(graphs["hash"]).all(), "hashes", "every result joins graphs.parquet",
              "results with a hash missing from graphs.parquet")
        n_cens = int(res.loc[res.winner == CENSORED, "count"].sum())
        check(n_cens == 0, "censored", "no run hit max_steps",
              f"{n_cens:,} runs hit max_steps={sim['max_steps']:,} (censored, not extinct): "
              "mean times of those cells are biased low", level="WARN")
        cells, hists = _aggregate(res, graphs)
        _write_parquet(cells, report / "results.parquet")
        _write_parquet(hists, report / "histograms.parquet")
        timing = res.drop_duplicates("chunk_id")[["chunk_id", "graph_idx", "r", "job", "index", "host",
                                                   "start", "end", "cpu"]]
        steps = res.assign(steps=res["count"] * res["steps_mean"], changes=res["count"] * res["changes_mean"])
        timing = timing.merge(steps.groupby("chunk_id")[["steps", "changes", "count"]].sum().reset_index(),
                              on="chunk_id")
        timing = timing.merge(graphs[["graph_idx", "factory", "n_nodes"]], on="graph_idx")
        _write_parquet(timing, report / "chunk_timing.parquet")
        _speed_report(batch, timing)

    worst = "FAIL" if any(c["status"] == "FAIL" for c in checks) else (
        "WARN" if any(c["status"] == "WARN" for c in checks) else "OK")
    (report / "verification.json").write_text(json.dumps({"overall": worst, "checks": checks}, indent=2))
    text = [f"VERIFICATION {batch.name}: {worst}"] + [f"[{c['status']:<4}] {c['check']:<15} {c['message']}"
                                                     for c in checks]
    (report / "verification.txt").write_text("\n".join(text) + "\n")
    print("\n".join(text), flush=True)
    return int(worst == "FAIL")


def _aggregate(res, graphs):
    """One row per (graph, r): rho with a Wilson 95% interval, and mean/sd of steps and
    n_changes conditional on the outcome, merged across chunks exactly (Chan et al.)."""
    key = ["graph_idx", "r", "winner"]
    g = res.assign(s_sum=res["count"] * res["steps_mean"], c_sum=res["count"] * res["changes_mean"])
    tot = g.groupby(key).agg(count=("count", "sum"), s_sum=("s_sum", "sum"), c_sum=("c_sum", "sum"),
                             steps_min=("steps_min", "min"), steps_max=("steps_max", "max"))
    tot["steps_mean"] = tot["s_sum"] / tot["count"]
    tot["changes_mean"] = tot["c_sum"] / tot["count"]
    g = g.join(tot[["steps_mean", "changes_mean"]], on=key, rsuffix="_all")
    g["s_m2"] = g["steps_m2"] + g["count"] * (g["steps_mean"] - g["steps_mean_all"]) ** 2
    g["c_m2"] = g["changes_m2"] + g["count"] * (g["changes_mean"] - g["changes_mean_all"]) ** 2
    m2 = g.groupby(key)[["s_m2", "c_m2"]].sum()
    tot["steps_sd"] = np.sqrt(m2["s_m2"] / (tot["count"] - 1).clip(lower=1))
    tot["changes_sd"] = np.sqrt(m2["c_m2"] / (tot["count"] - 1).clip(lower=1))

    codes, groups = pd.factorize(pd.MultiIndex.from_frame(res[key]))
    hist = np.zeros((len(groups), HIST_BINS), dtype=np.uint64)
    np.add.at(hist, codes, np.vstack(res["steps_hist"].to_numpy()).astype(np.uint64))
    hists = pd.DataFrame(list(groups), columns=key).assign(steps_hist=list(hist))

    wide = tot.reset_index()
    cells = wide.groupby(["graph_idx", "r"])["count"].sum().rename("n").reset_index()
    for label, code in (("fixed", 0), ("extinct", WILD_TYPE), ("censored", CENSORED)):
        part = wide[wide.winner == code].set_index(["graph_idx", "r"])
        cells = cells.join(part["count"].rename(f"n_{label}"), on=["graph_idx", "r"])
        if label != "censored":
            for col in ("steps_mean", "steps_sd", "changes_mean", "changes_sd"):
                cells = cells.join(part[col].rename(f"{col}_given_{label}"), on=["graph_idx", "r"])
    for col in ("n_fixed", "n_extinct", "n_censored"):
        cells[col] = cells[col].fillna(0).astype(np.int64)
    n, z = cells["n"], 1.96
    p = cells["n_fixed"] / n
    cells["rho"] = p  # over all runs; censored runs count as not fixed, see n_censored
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z / (1 + z**2 / n) * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    cells["rho_lo"], cells["rho_hi"] = centre - half, centre + half
    cells["steps_se_given_fixed"] = cells["steps_sd_given_fixed"] / np.sqrt(cells["n_fixed"])
    return cells.merge(graphs, on="graph_idx"), hists


def _speed_report(batch, t):
    subs = batch / "submissions.json"
    submitted = json.loads(subs.read_text())[0]["time"] if subs.exists() else None
    t = t.assign(wall=t["end"] - t["start"])
    per_worker = t.groupby(["job", "index"]).agg(first=("start", "min"), last=("end", "max"),
                                                 busy=("wall", "sum"), chunks=("chunk_id", "count"))
    makespan = t["end"].max() - t["start"].min()
    speed = {
        "workers": len(per_worker),
        "chunks": len(t),
        "queue_wait_s": None if submitted is None else t["start"].min() - submitted,
        "makespan_s": makespan,
        "cpu_hours": t["cpu"].sum() / 3600,
        "utilization": per_worker["busy"].sum() / (len(per_worker) * makespan) if makespan else None,
        "tail_s": t["end"].max() - per_worker["last"].min(),
        "chunk_wall_s": t["wall"].quantile([0.1, 0.5, 0.9, 1.0]).round(2).tolist(),
        "steps_per_cpu_s": t["steps"].sum() / t["cpu"].sum(),
        "steps_per_cpu_s_by_factory": (t.groupby("factory")["steps"].sum()
                                       / t.groupby("factory")["cpu"].sum()).round(0).to_dict(),
    }
    (batch / "report" / "speed.json").write_text(json.dumps(speed, indent=2) + "\n")
    lines = [
        f"SPEED {batch.name}",
        f"  {speed['workers']} workers ran {speed['chunks']} chunks, {speed['cpu_hours']:.2f} CPU hours",
        f"  makespan {makespan / 3600:.2f} h, utilization {speed['utilization']:.0%}, "
        f"tail {speed['tail_s'] / 60:.1f} min (last finish after the first worker went idle)",
        f"  chunk wall time p10/p50/p90/max: {speed['chunk_wall_s']} s",
        f"  {speed['steps_per_cpu_s'] / 1e6:.0f}M Moran steps per CPU second",
    ]
    if submitted is not None:
        lines.insert(2, f"  queue wait before the first chunk: {speed['queue_wait_s'] / 60:.1f} min")
    (batch / "report" / "speed.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def status(batch):
    batch = batch_path(batch)
    if not (batch / "chunks.parquet").exists():
        print(f"{batch.name}: not prepared yet")
    else:
        n = len(pd.read_parquet(batch / "chunks.parquet", columns=["chunk_id"]))
        done = list((batch / "results").glob("*.parquet")) if (batch / "results").exists() else []
        claimed = len(os.listdir(batch / "claims")) if (batch / "claims").exists() else 0
        print(f"{batch.name}: {len(done)}/{n} chunks done, {claimed - len(done)} running or orphaned, "
              f"{n - claimed} not started")
        if done:
            times = [p.stat().st_mtime for p in done]
            rate = len(done) / max(time.time() - min(times), 1)
            if len(done) < n:
                print(f"  rough ETA {(n - len(done)) / rate / 3600:.2f} h at the current rate")
    subs = batch / "submissions.json"
    if subs.exists():
        ids = [str(v) for s in json.loads(subs.read_text()) for k, v in s.items() if k != "time" and k != "n_workers"]
        out = subprocess.run(["bjobs", "-noheader", "-o", "jobid stat", *ids], text=True, capture_output=True)
        states = pd.Series([line.split()[1] for line in out.stdout.splitlines() if len(line.split()) == 2])
        print("  LSF:", states.value_counts().to_dict() if len(states) else "no jobs known to bjobs")


def resume(batch, n_workers=None, local=False):
    """Releases claims held by jobs that are no longer alive and reruns what is missing."""
    batch = batch_path(batch)
    plan = load_plan(batch / "plan.json")
    n = len(pd.read_parquet(batch / "chunks.parquet", columns=["chunk_id"]))
    done = {p.stem for p in (batch / "results").glob("*.parquet")}
    alive = set() if local else _alive_jobs()
    released = 0
    for path in (batch / "claims").glob("*"):
        owner = " ".join(path.read_text().split()[:2])
        if path.name not in done and owner not in alive:
            path.unlink()
            released += 1
    missing = n - len(done)
    print(f"{batch.name}: {missing} chunks missing, released {released} dead claims")
    if not missing:
        return finalize(batch)
    k = min(n_workers or plan["lsf"]["n_workers"], missing)
    if local:
        return _run_local(batch, k)
    workers, final = _submit_workers(batch, plan, k)
    _record_submission(batch, workers=workers, finalize=final, n_workers=k)


def load_report(batch):
    """The finished report of a batch. Raises if finalize has not written it yet."""
    report = batch_path(batch) / "report"
    if not (report / "verification.json").exists():
        raise FileNotFoundError(f"no report in {report}: run `python -m moran_graphs.hpc finalize {batch}`")
    return {
        "verification": json.loads((report / "verification.json").read_text()),
        "speed": json.loads((report / "speed.json").read_text()),
        "results": pd.read_parquet(report / "results.parquet"),
        "histograms": pd.read_parquet(report / "histograms.parquet"),
        "chunk_timing": pd.read_parquet(report / "chunk_timing.parquet"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("submit")
    p.add_argument("plan")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--local", action="store_true", help="run the whole chain here, no LSF")
    for cmd in ("prepare", "properties", "finalize", "status"):
        sub.add_parser(cmd).add_argument("batch")
    p = sub.add_parser("worker")
    p.add_argument("batch")
    p.add_argument("--n-workers", type=int, required=True)
    p.add_argument("--index", type=int, help="default: $LSB_JOBINDEX")
    p = sub.add_parser("resume")
    p.add_argument("batch")
    p.add_argument("--n-workers", type=int)
    p.add_argument("--local", action="store_true")
    a = parser.parse_args()

    if a.cmd == "submit":
        sys.exit(submit(a.plan, a.dry_run, a.local) or 0)
    if a.cmd == "prepare":
        prepare(a.batch)
    elif a.cmd == "properties":
        compute_properties(a.batch)
    elif a.cmd == "worker":
        worker(a.batch, a.n_workers, a.index)
    elif a.cmd == "finalize":
        sys.exit(finalize(a.batch))
    elif a.cmd == "status":
        status(a.batch)
    elif a.cmd == "resume":
        sys.exit(resume(a.batch, a.n_workers, a.local) or 0)


if __name__ == "__main__":
    main()

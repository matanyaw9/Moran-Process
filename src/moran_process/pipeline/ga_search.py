"""Evolutionary search whose fitness is MEASURED, not predicted.

``notebooks/extreme_graphs.ipynb`` runs the same (mu + lambda) strategy against the
residual ML predictors in ``ml_models_residual/``: evaluating a generation is a
``model.predict`` call, so the whole search fits in a notebook cell. Here a generation is
evaluated by actually simulating it, which means one real batch per generation and a loop
that runs for hours. The consequences:

  * The loop lives in a job, not a kernel. This module is submitted once per GA run with
    a single ``bsub`` and then spends its life waiting on LSF, using almost no CPU itself.
  * ``medium``/``long`` are preemptable and a preempted job restarts from the beginning,
    so resuming is the normal path rather than an exceptional one: if ``ga_state.json``
    exists the driver picks up at the next generation automatically. ``--force`` is the
    only way to start over.
  * Nothing is chained after each generation's array (``post_batch="none"``). verify, the
    violin cache and job speed all exist to serve figures and QC on a large one-off batch;
    on a 220-graph generation they are pure scheduling latency. The rollup is not chained
    either: the driver runs it inline, because handing 19 s of work to another machine
    cost 66 s of queue wait once per generation while this job's own slot sat idle. See
    ``_aggregate_inline`` and ``post_batch=`` in ``ProcessLab.submit_jobs``.

Selection reads ``prob_fixation`` and ``mean_steps`` straight out of the generation's
``graph_statistics.csv``, which is keyed on ``(wl_hash, r)`` -- the same ``wl_hash`` the GA
already uses to deduplicate candidates, so the join is free. They are combined into a
single score by the run's search direction ``theta`` (see below), which is the only thing
that distinguishes one run from another.

Run layout is ``<ga_runs_dir>/<prefix>-theta<NNN>/rep<K>/`` (see ``run_dir_for``), and the
reason each piece exists is recorded next to the piece: every constant below carries the
batch it was measured on, and CLAUDE.md section 5 summarises the design.

    python -m moran_process.pipeline.ga_search \
        --run-dir simulation_data/ga_runs/2026_08_19-phase1-theta315 --theta 315
"""

import argparse
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from moran_process.analysis.analysis_utils.constants import HASH_DTYPES

from moran_process.analysis.analysis_utils.theory import (
    analytic_moran_fc_fixation_prob,
    analytic_moran_fc_fixation_time,
)
from moran_process.core.population_graph import PopulationGraph
from moran_process.pipeline.process_lab import ProcessLab

log = logging.getLogger(__name__)

# --- Search space -----------------------------------------------------------------
# Exactly avian_r4_l7's node and edge count. mutate_graph removes one edge and adds one,
# so both are invariant under mutation and every graph the search ever sees is
# size-matched to the avian lung graph.
N_NODES = 31
N_EDGES = 34
R_VALUE = 1.1

# --- Direction ---------------------------------------------------------------------
# A run is defined by ONE number: the angle of the direction it searches in, in the
# standardized (probability, time) plane.
#
#     w_prob = cos(theta),  w_time = sin(theta)
#
# so theta=0 seeks high fixation probability, theta=90 long fixation time, theta=180 low
# probability, theta=270 short time, and the diagonals are the four combinations. Every
# run is therefore a MAXIMIZATION, and there is exactly one place direction is written
# down. The predecessor of this was a `--metric {mean_steps, prob_fixation, weighted}` x
# `--objective {maximize, minimize}` matrix plus a pair of free weights, which encoded
# direction in three redundant places; they disagreed as soon as `--objective minimize`
# met a weighted run, and both of that launch's weighted runs came out labelled with the
# same corner. The four single-metric runs of that matrix are the axis-aligned thetas
# here, so nothing is lost.
#
# Weights are UNIT length rather than the old (+1, -1), which makes the objective the
# projection of a graph onto the search direction: the score is literally "how many
# random-graph SDs out along theta", the support function whose maximum over a set traces
# that set's convex hull. Scores are therefore 1/sqrt(2) of the pre-theta diagonal runs'.
# Selection is unaffected, being scale-invariant.
CORNER_THETAS = (45, 135, 225, 315)

# --- The combined objective -------------------------------------------------------
# mean_steps runs 2400-41000 and prob_fixation 0.09-0.16, so they cannot be weighted
# against each other directly. Each is therefore expressed as a residual from the
# complete graph (the project's usual amplifier/suppressor zero) and divided by the
# spread of that residual among RANDOM graphs of the same size:
#
#   weighted = w_prob * (rho - rho_c)/SD_PROB  +  w_time * log(T / T_c)/SD_LOG_TIME
#
# Both terms are then "standard deviations away from a typical random (31, 34) graph",
# so w = (+1, -1) means one SD of probability gain is worth one SD of time reduction --
# a defensible default rather than an arbitrary knob.
#
# Measured on 501 random (31, 34) graphs at r=1.1 in
# 2026_07_28-respiratory-vs-random-10K-reps-3. Normalizing by the random-graph spread
# rather than by the current population's is what keeps the objective fixed: a
# population-relative scale would rescale itself every generation as the population
# converges, so the quantity being optimized would drift and generation 1 and generation
# 100 would no longer be on the same axis.
SD_PROB_RESIDUAL = 0.00426
SD_LOG_TIME_RESIDUAL = 0.1586

# The residual origin. Constants because N and r are fixed for the whole search.
RHO_COMPLETE = float(analytic_moran_fc_fixation_prob(N_NODES, R_VALUE))
T_COMPLETE = float(analytic_moran_fc_fixation_time(N_NODES, R_VALUE))

# --- Array sizing -----------------------------------------------------------------
# Measured on 2026_07_28-respiratory-vs-random-10K-reps-3/job_speed.csv over 4.8e8
# simulations: 13.9 us/sim. Parallelism helps only until per-worker work drops below
# per-worker startup (tens of seconds of LSF dispatch + python import + shard load); past
# that, more workers means MORE wall clock, not less. So size the array by work rather
# than pinning it, and it follows n_repeats and N without retuning.
SECONDS_PER_SIM = 13.9e-6
TARGET_SECONDS_PER_WORKER = 30
MAX_WORKERS = 200

# A fixed cost per simulation is wrong for this search, because the search changes it. A
# simulation costs time proportional to the number of steps it takes, and on the 100-
# generation run of 2026_07_28 the average steps per simulation grew 9.4x (652 -> 6101)
# as `maximize mean_steps` did its job. Sized off the constant alone, the last generations
# ran ~9x more work per worker than the first and became the slow ones.
#
# graph_statistics.csv does not record steps per simulation directly (mean_steps is
# conditional on fixation, and extinct runs are nulled out), so the proxy used here is
# prob_fixation * mean_steps: the fixating fraction of runs times their length. It ignores
# the steps burned by runs that go extinct, which makes it an underestimate, and the
# constant below absorbs that bias because it is calibrated against the same proxy.
#
# Calibration point: gen 1 of that run had proxy = 652 steps/sim and workers measurably
# spent 13.4 us/sim (2e6 sims in 26.84 s of CPU), giving 4.9e7. Being a single point it is
# good to maybe 30%, which is ample: the decision it feeds is an integer worker count that
# only has to land in the right order of magnitude.
STEPS_PER_SECOND = 4.9e7

# A generation that comes back short is not a cosmetic problem: a graph measured with 5K
# instead of 500K repeats has ~10x the noise and can win selection on luck alone.
MAX_GENERATION_RETRIES = 2

# 5 rather than 20: this is dead time added to every generation, and at 100 generations a
# 20-second poll spends ~17 minutes per run waiting to notice work that already finished.
# bjobs is cheap and the driver has nothing else to do.
POLL_SECONDS = 5
DEFAULT_GENERATION_TIMEOUT_S = 2 * 60 * 60

TERMINAL_STATES = {"DONE", "EXIT", "GONE"}

# A hung bjobs must not hang the driver: without this the whole search blocks on one
# unlucky LSF call, and the generation timeout never gets a chance to fire.
BJOBS_TIMEOUT_S = 60
# GONE is inferred from an absence rather than reported, so it is confirmed across this
# many consecutive polls before it is believed. One flaky reply used to be enough to
# declare a healthy 220-job array finished.
GONE_CONFIRMATIONS = 3
# How long to let a killed job actually leave the queue before its directory is deleted.
# bkill signals, it does not stop the process on the spot.
KILL_GRACE_S = 120

# What ntfy.sh accepts as a topic. Anything else 404s, and a '#' does not even reach the
# server: it is a URL fragment, so 'topic#secret' requests '/topic'.
_VALID_TOPIC = re.compile(r"[-_A-Za-z0-9]{1,64}")


# ======================================================================================
# LSF
# ======================================================================================


def _job_state(job_id):
    """Current LSF state of ``job_id``: DONE, EXIT, RUN, PEND, ... or GONE.

    Collapsed to one word across a job array's indices, because bjobs prints one row per
    index and an array therefore has many states at once. The collapse is deliberately
    pessimistic: while any index is still PEND or RUN the whole job reports that state,
    and only once every index is terminal does this report a terminal one. Reading the
    first row alone would call an 11-index array finished the moment index 1 exited, which
    was harmless while the only thing waited on was the single aggregate job and is not
    now that the driver waits on the array itself.

    ``GONE`` means bjobs no longer has a record (jobs age out of its history), which is
    terminal from our point of view: whatever the job was going to do, it has done.
    Callers then fall back to inspecting the artefact itself.

    ``UNKNOWN`` means bjobs told us nothing, and is deliberately NOT terminal. These used
    to be the same answer, and conflating them was a live corruption path: one flaky
    bjobs reply read as "the array finished", the shard wait then timed out on a
    directory the array was still filling, and the retry deleted it under 220 running
    jobs. An absence of information is not evidence of completion.
    """
    try:
        result = subprocess.run(
            ["bjobs", "-a", "-o", "stat", str(job_id)],
            capture_output=True,
            text=True,
            timeout=BJOBS_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError):
        # bjobs missing from PATH, or it hung past BJOBS_TIMEOUT_S.
        return "UNKNOWN"

    # Output is a STAT header followed by one row per index.
    lines = [ln.strip() for ln in result.stdout.splitlines() if ln.strip()]
    if len(lines) >= 2:
        states = {line.split()[0] for line in lines[1:]}
        unfinished = states - TERMINAL_STATES
        if unfinished:
            return sorted(unfinished)[0]
        # Every index is terminal. One EXIT makes the whole array's result suspect, and
        # the completeness check downstream decides whether it is actually unusable.
        return "EXIT" if "EXIT" in states else "DONE"

    # No rows. LSF says "Job <id> is not found" on stderr once a record has aged out,
    # and that phrasing is the only thing here that actually means the job is over.
    # Every other failure (daemon not responding, a truncated reply, a transient error)
    # says nothing about the job and must not be read as completion.
    if "not found" in result.stderr.lower():
        return "GONE"
    log.warning(
        "bjobs gave no usable state for job %s (rc=%s): %s",
        job_id,
        result.returncode,
        result.stderr.strip()[:200] or "<no stderr>",
    )
    return "UNKNOWN"


def _wait_for_job(job_id, timeout_s, label=""):
    """Block until ``job_id`` reaches a terminal state. Returns that state.

    Polling bjobs rather than watching for graph_statistics.csv to appear is deliberate:
    build_graph_statistics writes with a plain ``to_csv``, so a file that exists may still
    be half-written. Waiting for the job to leave the queue removes that race.

    DONE and EXIT are reported by LSF and believed on sight. GONE is inferred from the
    job's absence from bjobs, so it has to survive GONE_CONFIRMATIONS consecutive polls
    before it ends the wait: a record that has genuinely aged out stays absent, while a
    momentary LSF failure does not.
    """
    deadline = time.monotonic() + timeout_s
    gone_readings = 0
    while True:
        state = _job_state(job_id)
        gone_readings = gone_readings + 1 if state == "GONE" else 0
        settled = state in TERMINAL_STATES and (
            state != "GONE" or gone_readings >= GONE_CONFIRMATIONS
        )
        if settled:
            return state
        if time.monotonic() > deadline:
            log.warning(
                "Timed out after %.0f min waiting for %s job %s (last state: %s)",
                timeout_s / 60,
                label,
                job_id,
                state,
            )
            return "TIMEOUT"
        time.sleep(POLL_SECONDS)


def _bkill_and_settle(job_ids, label="", grace_s=KILL_GRACE_S):
    """Kill ``job_ids`` and wait for them to actually leave the queue.

    Called before a generation directory is deleted. The driver only ever deletes a
    directory because it stopped trusting the generation, and "stopped trusting" includes
    the case where the jobs are alive and healthy and the driver's view of them was
    wrong. So they are killed rather than assumed dead: deleting under a live array would
    leave it writing shards into the path a second array is also writing to, and the
    completeness check counts rows, so the mixture can pass as a healthy generation.

    Best effort by design. bkill on an already-finished job is a harmless no-op that
    complains on stderr, so the return code is not acted on. The settle wait matters more
    than the kill: bkill signals, it does not stop the process on the spot.
    """
    targets = [str(j) for j in job_ids if j]
    if not targets:
        return
    log.warning("Killing %s job(s) %s before reusing their directory.", label, ", ".join(targets))
    try:
        subprocess.run(
            ["bkill"] + targets,
            capture_output=True,
            text=True,
            timeout=BJOBS_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError):
        log.warning("bkill on %s failed to run; waiting for the jobs anyway.", ", ".join(targets))

    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        # UNKNOWN is not terminal, so an LSF outage here spends the grace period rather
        # than concluding the jobs are dead. That is the safe direction: the cost is a
        # two-minute pause, the alternative is deleting under a live array.
        if all(_job_state(j) in TERMINAL_STATES for j in targets):
            return
        time.sleep(POLL_SECONDS)
    log.warning(
        "Job(s) %s still not terminal %ds after bkill; deleting anyway.",
        ", ".join(targets),
        grace_s,
    )


def _pending_path(run_dir):
    return Path(run_dir) / "pending_jobs.json"


def _record_pending(run_dir, generation, job_ids):
    """Note which jobs are live right now, so a *replacement* driver can kill them.

    The in-loop retry remembers its own submissions in a local, but a preempted driver
    does not get to hand anything over: LSF requeues the job from the beginning, and the
    new process restarts at this generation with no idea that the previous one left a
    220-index array running. It would then delete the directory that array is filling.

    Its own file rather than a key in ga_state.json, because this is scratch bookkeeping
    with a lifetime of one generation, and ga_state.json is the run's readable summary
    with several consumers in analysis_utils. Nothing outside this module reads it.
    """
    _pending_path(run_dir).write_text(
        json.dumps(
            {
                "generation": generation,
                "array": job_ids.get("array"),
                "register": job_ids.get("register"),
            },
            indent=2,
        )
    )


def _clear_pending(run_dir):
    """Drop the record once the generation is banked and its jobs are finished."""
    _pending_path(run_dir).unlink(missing_ok=True)


def _load_pending(run_dir, generation):
    """Jobs a previous driver invocation left running at ``generation``, if any.

    A record for a different generation is ignored rather than acted on: the hazard being
    closed is deleting *this* generation's directory, and a stale record is not evidence
    about it. A corrupt or unreadable file is treated the same way, since this is a
    best-effort safety net and must never be the thing that stops a run.
    """
    path = _pending_path(run_dir)
    if not path.exists():
        return {}
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError):
        log.warning("Unreadable %s; ignoring it.", path)
        return {}
    if record.get("generation") != generation:
        return {}
    return record


def _run_dirs_arg(run_dirs):
    """Accept a single run directory or a list of them, uniformly."""
    if isinstance(run_dirs, (str, Path)):
        return [Path(run_dirs)]
    return [Path(d) for d in run_dirs]


def _format_duration(seconds):
    """'3h42m', or '7m12s' under an hour. Compact beats precise on a phone screen."""
    seconds = max(0, int(seconds))
    if seconds >= 3600:
        return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"
    return f"{seconds // 60}m{seconds % 60:02d}s"


def _elapsed_since_launch(state):
    """Seconds since the run was first submitted, or None if it never checkpointed.

    Read from ``started_at`` in the state rather than from a process-local ``monotonic``
    baseline, because that baseline resets on resume: a preempted run would otherwise
    report only the time since its last requeue. This is wall clock from launch, which
    includes any interval the job spent dead in the queue, and that is the number worth
    reporting -- it is the time between asking for the run and being able to look at it.
    """
    started_at = (state or {}).get("started_at")
    if not started_at:
        return None
    try:
        return (datetime.now() - datetime.fromisoformat(started_at)).total_seconds()
    except ValueError:
        return None


def notify(topic, title, message, priority="default"):
    """Push a line to an ntfy topic. Never raises.

    A GA run is hours long and detached, so the useful moment to hear about it is the
    moment it ends, which is exactly when nobody is watching the log. ntfy.sh is a plain
    HTTP POST with no account and no client library, and compute nodes can reach it
    (verified from hgn20), so this is a urllib call rather than a dependency.

    Every failure is swallowed on purpose. A notification is a courtesy; a DNS hiccup at
    hour five must not take down the run that was trying to announce its own success.

    The topic name is the only secret: anyone who knows it can read and post to it, so it
    comes from the environment (NTFY_TOPIC) rather than living in the repo, and it should
    be unguessable rather than memorable.
    """
    if not topic:
        return
    try:
        request = urllib.request.Request(
            f"https://ntfy.sh/{topic}",
            data=message.encode("utf-8"),
            headers={"Title": title, "Priority": priority},
            method="POST",
        )
        urllib.request.urlopen(request, timeout=10).close()
    except Exception as error:  # noqa: BLE001 - see docstring
        log.warning("ntfy notification failed (ignored): %s", error)


def _run_span(state):
    """Seconds from a run's launch to its last completed generation, or None.

    Distinct from _elapsed_since_launch, which measures to *now*. That is right inside the
    driver, where "now" is the moment the run ended, and wrong in the summary job, which
    runs later: it would charge the run for the summary's own queue wait, and for any time
    the log sat around before anyone looked. Both endpoints come from the state file, so
    this is the same number whenever it is computed.
    """
    started_at, ended_at = (state or {}).get("started_at"), (state or {}).get(
        "last_generation_at"
    )
    if not started_at or not ended_at:
        return None
    try:
        return (
            datetime.fromisoformat(ended_at) - datetime.fromisoformat(started_at)
        ).total_seconds()
    except ValueError:
        return None


def summarize_runs(run_dirs):
    """(title, body, priority) for one message covering a whole launch.

    Reads each run's ga_state.json and nothing else, so it works whether a run finished,
    died, or never started. A run with no state file never completed a generation, which
    is a distinct outcome from failing and is reported as such.

    Runs are listed individually up to a point and then collapsed to one line per
    objective showing the best across replicates, because a 12-run replicate launch turns
    into a wall of text on a phone otherwise.
    """
    rows = []
    for run_dir in _run_dirs_arg(run_dirs):
        state_path = Path(run_dir) / "ga_state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        rows.append({
            "name": Path(run_dir).name,
            # A run that never wrote a state file has no theta to report, so it is
            # identified by directory instead. Labelling it '?' would make several such
            # runs collapse into one indistinguishable line.
            "category": state.get("category") or Path(run_dir).name,
            "status": state.get("status", "never started"),
            "best": state.get("best_fitness"),
            "generation": state.get("generation"),
            "generations": state.get("generations"),
            "elapsed": _run_span(state),
        })

    finished = [r for r in rows if r["status"] == "finished"]
    # The longest run is the one that decided when you could look at the results, so that
    # is the number reported rather than a sum or a mean.
    longest = max((r["elapsed"] or 0 for r in rows), default=0)

    if len(rows) <= 6:
        lines = [
            f"{r['category']}: "
            + (f"best={r['best']:.6g}" if r["best"] is not None else "no result")
            + ("" if r["status"] == "finished" else f"  [{r['status']}]")
            for r in rows
        ]
    else:
        lines = []
        for category in sorted({r["category"] for r in rows}):
            group = [r for r in rows if r["category"] == category]
            scored = [r["best"] for r in group if r["best"] is not None]
            done = sum(r["status"] == "finished" for r in group)
            best = (
                f"best={max(scored):.6g}..{min(scored):.6g}"
                if len(scored) > 1
                else (f"best={scored[0]:.6g}" if scored else "no result")
            )
            lines.append(f"{category}: {best}  ({done}/{len(group)} ok)")

    incomplete = [r for r in rows if r["status"] != "finished"]
    title = f"GA complete: {len(finished)}/{len(rows)} runs"
    if incomplete:
        title = f"GA finished with {len(incomplete)} PROBLEM(S): {len(finished)}/{len(rows)} ok"
    body = f"all runs done in {_format_duration(longest)}\n" + "\n".join(lines)
    if incomplete:
        body += "\n\nnot finished:\n" + "\n".join(
            f"  {r['name']}: {r['status']}"
            + (
                f" at generation {r['generation'] + 1}/{r['generations']}"
                if r["generation"] is not None
                else ""
            )
            for r in incomplete
        )
    return title, body, ("high" if incomplete else "default")


def _size_array(n_graphs, n_repeats, n_r_values, seconds_per_sim=SECONDS_PER_SIM):
    """Number of array workers giving each roughly TARGET_SECONDS_PER_WORKER of work."""
    estimated_seconds = n_graphs * n_repeats * n_r_values * seconds_per_sim
    n_jobs = math.ceil(estimated_seconds / TARGET_SECONDS_PER_WORKER)
    return int(np.clip(n_jobs, 1, MAX_WORKERS))


# Everything a generation costs that is not the simulation itself: the register job, LSF
# dispatch of the array, python import and shard load in each worker, and the driver's own
# rollup. Measured on 2026_08_13-smoke-*, where a 30-graph generation at 1e6 repeats was
# sized for 30 s of work per worker and took 101 s and 106 s wall clock end to end, plus
# 7 s and 9 s of rollup. It is a constant, not a rate: it is queue and startup latency, so
# it does not shrink when the generation does. At these small sizes it IS the wall clock.
GENERATION_OVERHEAD_S = 78


def preview_launch(
    generations,
    pop_size,
    n_children,
    n_repeats,
    n_runs=1,
    seconds_per_sim=SECONDS_PER_SIM,
    show=True,
):
    """What a launch will cost, before it is launched. Returns the numbers as a dict.

    Sized through ``_size_array`` itself rather than through a copy of its arithmetic, so
    the shard count printed here is the shard count the driver will actually request.

    Two things it deliberately does not pretend to know:

    * ``seconds_per_sim`` is the generation-0 cost. The search changes it -- on the 100
      generation run, maximizing mean_steps grew the steps per simulation 9.4x -- so any
      run with a positive time weight will end up costing more than this says. The driver
      re-estimates it every generation and resizes, so the effect lands on core-hours,
      not on wall clock.
    * Wall clock assumes the arrays start promptly. On a busy queue that is optimistic,
      and it is the only term here that a preemption changes.
    """
    per_generation = [pop_size] + [pop_size * (1 + n_children)] * (generations - 1)
    workers = [_size_array(n, n_repeats, 1, seconds_per_sim) for n in per_generation]
    sims_per_run = sum(per_generation) * n_repeats

    seconds_per_generation = TARGET_SECONDS_PER_WORKER + GENERATION_OVERHEAD_S
    wall_clock_s = generations * seconds_per_generation
    result = {
        "n_runs": n_runs,
        "candidates_gen0": per_generation[0],
        "candidates_later": per_generation[-1],
        "workers_gen0": workers[0],
        "workers_later": workers[-1],
        "graphs_evaluated_per_run": sum(per_generation),
        "sims_per_run": sims_per_run,
        "sims_total": sims_per_run * n_runs,
        # The simulation work itself, which is what actually consumes the group's CPU
        # allocation. The driver slots are counted separately because they are held for the
        # whole wall clock while doing nothing, and that is a different kind of cost.
        "core_hours": sims_per_run * n_runs * seconds_per_sim / 3600,
        "wall_clock_hours": wall_clock_s / 3600,
        "driver_slot_hours": n_runs * wall_clock_s / 3600,
        "peak_concurrent_workers": max(workers) * n_runs,
    }
    if show:
        print(
            f"{n_runs} run(s) x {generations} generations\n"
            f"  candidates:  {result['candidates_gen0']} in gen 0, "
            f"{result['candidates_later']} after  ->  "
            f"{result['workers_gen0']} / {result['workers_later']} array workers\n"
            f"  graphs evaluated: {result['graphs_evaluated_per_run']:,} per run, "
            f"{result['graphs_evaluated_per_run'] * n_runs:,} total\n"
            f"  simulations:      {result['sims_total']:,.0f} total\n"
            f"  simulation cost:  {result['core_hours']:,.1f} core-hours "
            f"(at {seconds_per_sim * 1e6:.1f} us/sim; a time-maximizing run will exceed this)\n"
            f"  wall clock:       ~{result['wall_clock_hours']:.2f} h per run, runs are "
            f"concurrent\n"
            f"  driver slots:     {result['driver_slot_hours']:,.1f} slot-hours idle "
            f"({n_runs} slots x {result['wall_clock_hours']:.2f} h)\n"
            f"  peak concurrency: {result['peak_concurrent_workers']} array workers "
            f"if every run is in a generation at once"
        )
    return result


def _estimate_seconds_per_sim(stats):
    """Per-simulation cost implied by a finished generation, for sizing the next one.

    Returns None when ``stats`` cannot support an estimate, so the caller keeps whatever
    it was using. Never returns less than SECONDS_PER_SIM: the proxy ignores the steps
    spent by runs that go extinct, so it can only understate the true cost, and
    understating it is the failure mode that hurts (too few workers, a slow generation).
    """
    if stats is None or stats.empty:
        return None
    steps_per_sim = float((stats["prob_fixation"] * stats["mean_steps"]).mean())
    if not np.isfinite(steps_per_sim) or steps_per_sim <= 0:
        return None
    return max(steps_per_sim / STEPS_PER_SECOND, SECONDS_PER_SIM)


# ======================================================================================
# Population
# ======================================================================================


def build_initial_population(pop_size, seed, category):
    """``pop_size`` distinct random connected (N_NODES, N_EDGES) graphs.

    Shared verbatim by all four runs (same seed), so any divergence between them is
    attributable to the objective alone rather than to their starting points.
    """
    rng = np.random.default_rng(seed)
    population, seen = [], set()
    while len(population) < pop_size:
        graph = PopulationGraph.random_connected_graph(
            N_NODES,
            N_EDGES,
            name=f"seed{len(population):02d}",
            seed=int(rng.integers(0, 2**32)),
        )
        if graph.wl_hash in seen:
            continue
        graph.category = category
        seen.add(graph.wl_hash)
        population.append(graph)
    return population


def _reproduce(elites, n_children, seen, rng, generation, max_attempts=10):
    """One mutated child per (elite, k), skipping topologies already seen in this run.

    Names carry the lineage root token forward (``seed03_g017_004``), so a graph's
    ancestry back to its founding random graph is readable off the name alone.
    """
    children, parents = [], {}
    for elite in elites:
        root = elite.name.split("_")[0]
        for k in range(n_children):
            for _ in range(max_attempts):
                child = elite.mutate_graph(
                    seed=int(rng.integers(0, 2**32)),
                    name=f"{root}_g{generation:03d}_{k:03d}",
                )
                if child.wl_hash not in seen:
                    seen.add(child.wl_hash)
                    children.append(child)
                    parents[child.wl_hash] = elite.wl_hash
                    break
    return children, parents


# ======================================================================================
# One generation
# ======================================================================================


def _submit_generation(
    candidates,
    run_dir,
    gen_dir,
    batch_name,
    n_repeats,
    queue,
    memory,
    engine,
    batch_seed,
    seconds_per_sim,
):
    """Write the population and submit register + array. Returns the job ids.

    ``post_batch="none"``: the driver runs the rollup itself (see _run_generation). The
    register job is still a job, because it is submitted with no dependency and finishes
    long before the array does, so it costs no wall clock at all.

    gen_dir is deliberately NOT created here: submit_jobs creates it, and warns if it
    already exists. Pre-creating it would fire that warning on every generation.
    """
    zoo_path = Path(run_dir) / "populations" / f"{gen_dir.name}.pkl"
    zoo_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(candidates, zoo_path)

    n_jobs = _size_array(len(candidates), n_repeats, 1, seconds_per_sim)
    log.info(
        "Submitting %s: %d graphs x %d repeats -> %d workers (%.1f us/sim)",
        batch_name,
        len(candidates),
        n_repeats,
        n_jobs,
        seconds_per_sim * 1e6,
    )
    job_ids = ProcessLab().submit_jobs(
        zoo_path=str(zoo_path),
        n_graphs=len(candidates),
        r_values=[R_VALUE],
        batch_name=batch_name,
        batch_dir=str(gen_dir),
        n_repeats=n_repeats,
        n_requested_jobs=n_jobs,
        queue=queue,
        memory=memory,
        engine=engine,
        post_batch="none",
        batch_seed=batch_seed,
    )
    # Carried out so the caller knows how many shards to wait for; it is the array
    # width, and nothing downstream can otherwise tell "not written yet" from
    # "this generation only needed 11 workers".
    job_ids["n_shards"] = n_jobs
    return job_ids


def _wait_for_shards(gen_dir, n_expected, timeout_s=600, poll_s=5):
    """Block until every shard is present AND complete. Returns None or a complaint.

    LSF reporting an array index DONE means the process exited, not that its output has
    landed on the shared filesystem. The chained aggregate job never had to care: it sat
    in the queue for ~51 s first, which was accidentally long enough for the writes to
    flush. Aggregating in the driver removed that buffer and reads the instant the last
    index reports terminal -- so polars memory-mapped a Parquet file that was still being
    written and segfaulted, taking the whole driver with it. Four of eight corner runs
    died this way, and the same files read back perfectly at rest minutes later.

    Completeness is checked by parsing each footer rather than by counting files or
    comparing sizes, because Parquet writes its footer LAST: a file whose footer parses is
    necessarily whole, and a half-written one raises rather than reading short.

    A crash here cannot be caught in Python (SIGSEGV is not an exception), so this has to
    be prevention rather than recovery.
    """
    import polars as pl

    results = Path(gen_dir) / "tmp" / "results"
    deadline = time.monotonic() + timeout_s
    while True:
        shards = sorted(results.glob("raw_results_job_*.parquet"))
        if len(shards) >= n_expected:
            unreadable = []
            for shard in shards:
                try:
                    pl.read_parquet_schema(shard)
                except Exception:  # noqa: BLE001 - still being written
                    unreadable.append(shard.name)
            if not unreadable:
                return None
        else:
            unreadable = [f"only {len(shards)}/{n_expected} shards present"]

        if time.monotonic() > deadline:
            return (
                f"shards never became readable within {timeout_s}s: "
                f"{', '.join(unreadable[:3])}"
            )
        time.sleep(poll_s)


def _aggregate_inline(gen_dir):
    """Roll the generation's shards up into graph_statistics.csv, here in the driver.

    Returns None on success, or a complaint string.

    This used to be a chained LSF job. On a real batch that is right: aggregation reads
    7.2e9 rows and needs its own 16GB. A GA generation is 2.2e7 rows, and the job was
    measured at 7.2s of CPU, 19s of run time and 513MB peak -- to obtain which the driver
    waited 51s in the queue and then up to another poll interval to notice it had
    finished. Handing 19 seconds of work to another machine cost 66 seconds of waiting,
    repeated once per generation, while the driver's own slot sat idle in a sleep loop.

    aggregate_batch is imported here rather than at module scope because importing it runs
    its module-level logging.basicConfig, which would win the race against this module's
    own basicConfig in main() and silently redirect the progress bar off stdout.
    """
    from moran_process.pipeline.aggregate_batch import run_aggregation

    try:
        run_aggregation(str(gen_dir))
    except (Exception, SystemExit) as error:  # noqa: BLE001 - a retryable complaint
        # SystemExit explicitly, and this is not defensive padding: run_aggregation
        # signals its two expected failures (no shards, no graph_props.csv) by raising
        # SystemExit, which derives from BaseException and so slips straight through
        # `except Exception`. That turned the single most retryable condition in the whole
        # driver into a fatal one, and killed four runs mid-flight the first time the
        # cluster was busy enough for a register job to be preempted.
        return f"inline aggregation failed: {type(error).__name__}: {error}"
    return None


def _read_generation_stats(gen_dir, candidates, n_repeats):
    """Read graph_statistics.csv and check every candidate got its full n_repeats.

    Returns (stats, complaint). ``complaint`` is None when the generation is usable, and
    otherwise says what is wrong -- a missing file, absent graphs, or short cells. The
    caller resubmits on a complaint rather than selecting on damaged data.
    """
    stats_path = gen_dir / "graph_statistics.csv"
    if not stats_path.exists():
        return None, f"graph_statistics.csv missing at {stats_path}"

    stats = pd.read_csv(stats_path, dtype=HASH_DTYPES)
    stats = stats[np.isclose(stats["r"], R_VALUE)]

    wanted = {g.wl_hash for g in candidates}
    missing = wanted - set(stats["wl_hash"])
    if missing:
        return None, f"{len(missing)}/{len(wanted)} candidates absent from the rollup"

    stats = stats[stats["wl_hash"].isin(wanted)].drop_duplicates("wl_hash")
    short = stats[stats["n_grouped"] < n_repeats]
    if not short.empty:
        return None, (
            f"{len(short)}/{len(stats)} cells short of {n_repeats} repeats "
            f"(min {int(short['n_grouped'].min())})"
        )
    return stats, None


def _add_sem(stats):
    """Standard error of each metric, so a trajectory can be read against its noise floor.

    mean_steps is conditional on fixation (io._steps_success_expr nulls out extinct runs),
    so its SEM divides by the number of runs that actually fixated, not by n_grouped.
    """
    n_fixated = (stats["prob_fixation"] * stats["n_grouped"]).clip(lower=1)
    p = stats["prob_fixation"]
    return stats.assign(
        mean_steps_sem=stats["std_steps"] / np.sqrt(n_fixated),
        prob_fixation_sem=np.sqrt(p * (1 - p) / stats["n_grouped"]),
    )


def normalize_theta(theta_deg):
    """Fold an angle into [0, 360). ``-45`` and ``315`` name the same direction.

    Canonicalized rather than taken as given, because the angle is part of a run's
    directory name and its category label. Two launches that both mean "down and to the
    right" must land in the same-named place, or a phase-2 replicate set at one theta
    silently splits into two groups that no figure will ever put together.
    """
    return float(np.mod(theta_deg, 360.0))


def theta_weights(theta_deg):
    """``(w_prob, w_time)`` for a direction: the unit vector at ``theta_deg``.

    Rounded to kill the 6e-17 that ``cos(90 degrees)`` returns in floating point. That
    residue is harmless arithmetically but not cosmetically: it reaches ga_config.json and
    the bsub command line, where an axis-aligned run reads as 6.1e-17 instead of 0. The
    ``+ 0.0`` then turns the resulting -0.0 back into 0.0, which is the same number but
    not the same eight characters in a JSON file someone has to read.
    """
    radians = np.deg2rad(normalize_theta(theta_deg))
    return (
        round(float(np.cos(radians)), 12) + 0.0,
        round(float(np.sin(radians)), 12) + 0.0,
    )


def theta_category(theta_deg):
    """The canonical label for a direction: ``theta=315``.

    Short and sortable, because it has to work as a legend entry for a dozen directions at
    once. The qualitative reading lives in ``quadrant_name``, which is used where there is
    room for words.
    """
    return f"theta={normalize_theta(theta_deg):03.0f}"


def quadrant_name(theta_deg):
    """The words for a direction, e.g. ``high_prob, low_time``.

    For launch printouts, figure annotations and anywhere a reader needs to know what a
    theta means without doing trigonometry. Axis-aligned directions name only the axis
    they move along, since calling theta=0 "high_prob any_time" implies a time preference
    it does not have.
    """
    w_prob, w_time = theta_weights(theta_deg)
    # Tolerance rather than == 0, so a theta of 89.9999 still reads as a pure time run.
    prob = "" if abs(w_prob) < 1e-9 else ("high_prob" if w_prob > 0 else "low_prob")
    # Negative time weight means shorter fixation scores higher.
    time = "" if abs(w_time) < 1e-9 else ("high_time" if w_time > 0 else "low_time")
    return ", ".join(part for part in (prob, time) if part)


def weighted_score(prob_fixation, mean_steps, w_prob, w_time):
    """The combined objective evaluated on any (rho, T), not just on a generation's stats.

    Public and column-free so a reader can score things the search never produced: the
    random-graph cloud, the respiratory graphs, an old run under a different theta. That
    is what makes "did the winner land where the objective says it should" checkable,
    since the check is 'no other graph in the cloud scores higher', and the cloud has no
    weighted column of its own. Accepts scalars or arrays.

    ``_add_weighted`` is the in-pipeline caller, so the score the driver selects on and
    the score a figure draws cannot drift apart.
    """
    return (
        w_prob * (prob_fixation - RHO_COMPLETE) / SD_PROB_RESIDUAL
        + w_time * np.log(mean_steps / T_COMPLETE) / SD_LOG_TIME_RESIDUAL
    )


def _add_weighted(stats, w_prob, w_time):
    """Add the combined objective and its standard error.

    The SEM is propagated rather than left out, so plot_selection_efficiency keeps working
    on weighted runs: without it there would be no noise floor to judge the trajectory
    against, which is the whole reason those error bars exist. Both terms are linear in
    their residual, and log(T/T_c) has SEM = SEM_T / T by the delta method, so

        SEM_weighted = sqrt( (w_prob/SD_PROB * SEM_rho)^2
                           + (w_time/SD_LOG_TIME * SEM_T / T)^2 )

    The two measurements are independent draws from the same simulations, and are treated
    as uncorrelated here. They are not exactly: a graph that fixates more often also
    contributes more fixation-time samples. The residual correlation is second order next
    to the weights themselves, and this SEM is used for display, never for selection.
    """
    score = weighted_score(stats["prob_fixation"], stats["mean_steps"], w_prob, w_time)
    sem = np.sqrt(
        (w_prob / SD_PROB_RESIDUAL * stats["prob_fixation_sem"]) ** 2
        + (w_time / SD_LOG_TIME_RESIDUAL * stats["mean_steps_sem"] / stats["mean_steps"])
        ** 2
    )
    return stats.assign(weighted=score, weighted_sem=sem)


def _run_generation(state, cfg, candidates, generation, seconds_per_sim):
    """Submit, wait, aggregate, verify, and resubmit on failure. Returns verified stats."""
    run_dir = Path(cfg["run_dir"])
    gen_dir = run_dir / "generations" / f"gen_{generation:03d}"

    # Seeded from disk, not empty: on the first attempt these are the jobs a preempted
    # predecessor left behind at this same generation. See _record_pending.
    prior_job_ids = _load_pending(run_dir, generation)
    for attempt in range(MAX_GENERATION_RETRIES + 1):
        # Always start from an empty directory, not only on a retry. A driver killed
        # mid-generation (preemption) restarts at this same generation, and any shards
        # its first attempt left behind would be aggregated together with the new ones --
        # inflating n_grouped past n_repeats, so the completeness check below would pass
        # on doubled data.
        if gen_dir.exists():
            # ...but kill the previous attempt first. A retry does not imply the previous
            # jobs are dead: the loop also lands here after a timeout, and a timeout can
            # mean the driver misread a perfectly healthy array. See _bkill_and_settle.
            _bkill_and_settle(
                [prior_job_ids.get("array"), prior_job_ids.get("register")],
                label=f"gen {generation} attempt {attempt}",
            )
            shutil.rmtree(gen_dir)

        job_ids = _submit_generation(
            candidates,
            run_dir,
            gen_dir,
            batch_name=f"{run_label(run_dir)}_gen_{generation:03d}",
            n_repeats=cfg["n_repeats"],
            queue=cfg["queue"],
            memory=cfg["memory"],
            engine=cfg["engine"],
            # Derived from the generation number rather than drawn from a running stream,
            # so a resubmitted generation cannot desynchronise every generation after it.
            batch_seed=cfg["seed"] * 100003 + generation,
            seconds_per_sim=seconds_per_sim,
        )
        # Recorded before anything can go wrong, so the next attempt knows what to kill --
        # in this process via the local, and in a requeued one via the file.
        prior_job_ids = job_ids
        _record_pending(run_dir, generation, job_ids)
        if job_ids.get("array") is None:
            # Retrying here would rmtree the directory the array is actively writing to.
            raise SystemExit(
                f"Generation {generation}: simulation array was not submitted (bsub "
                f"failed). Register job {job_ids.get('register')} may still be running."
            )
        # Wait on both: the array is the long pole, and the register job's graph_props.csv
        # is an input to the rollup. Register is submitted with no dependency and finishes
        # while the array is still running, so the second wait is normally instant.
        final_state = _wait_for_job(
            job_ids.get("array"),
            timeout_s=cfg["generation_timeout_s"],
            label=f"gen {generation} array",
        )
        register_state = "DONE"
        if job_ids.get("register") is not None:
            register_state = _wait_for_job(
                job_ids["register"],
                timeout_s=cfg["generation_timeout_s"],
                label=f"gen {generation} register",
            )

        # The register job's verdict is acted on, not merely awaited. It writes the
        # graph_props.csv the rollup joins against, so an EXIT here guarantees the
        # aggregation below fails; saying so plainly beats letting it surface as a
        # confusing "graph_props.csv missing" three lines later.
        # Shards first: an unreadable one segfaults polars, which no except clause
        # can catch, so it has to be prevented rather than retried.
        # Budgeted off the same knob as the array wait, not the 600 s default. Under a
        # saturated queue the last array index can still be PENDING when the rest have
        # finished, and a fixed 600 s then throws away a generation that was 101/102
        # complete and re-runs all 102 -- which is how a 3 h launch became a 10 h one.
        complaint = _wait_for_shards(
            gen_dir,
            job_ids.get("n_shards", 1),
            timeout_s=cfg["generation_timeout_s"],
        )
        if complaint is not None:
            pass
        elif register_state == "EXIT":
            complaint = (
                f"register_graphs job {job_ids['register']} exited; graph_props.csv was "
                f"never written"
            )
        else:
            complaint = _aggregate_inline(gen_dir)
        if complaint is None:
            stats, complaint = _read_generation_stats(
                gen_dir, candidates, cfg["n_repeats"]
            )
            if stats is not None:
                stats = _add_sem(stats)
                stats = _add_weighted(stats, *theta_weights(cfg["theta"]))
                # The generation is banked and its jobs are terminal, so there is nothing
                # left for a successor to kill. Leaving the record would make the next
                # driver bkill job ids that have since been recycled by LSF.
                _clear_pending(run_dir)
                return stats, gen_dir

        warning = (
            f"generation {generation} attempt {attempt + 1} unusable "
            f"(array ended {final_state}): {complaint}"
        )
        log.warning("%s", warning)
        state.setdefault("warnings", []).append(
            {"generation": generation, "attempt": attempt + 1, "detail": complaint}
        )
        if attempt < MAX_GENERATION_RETRIES:
            log.warning("Resubmitting generation %d.", generation)

    raise SystemExit(
        f"Generation {generation} still incomplete after {MAX_GENERATION_RETRIES + 1} "
        f"attempts. State checkpointed; fix the cause and rerun to resume."
    )


def _prune_raw(gen_dir):
    """Drop the raw shards once the rollup is verified.

    populations/gen_NNN.pkl is kept, so any generation is replayable in ~2 minutes by
    resubmitting that population: these shards are reproducible, not lost. Keeping them
    would cost ~370GB across four runs of 100 generations. task_manifest.csv stays, since
    it holds the per-task seeds.
    """
    for sub in ("results", "zoo_shards"):
        target = gen_dir / "tmp" / sub
        if target.exists():
            shutil.rmtree(target)


# ======================================================================================
# Selection and bookkeeping
# ======================================================================================


def _select(stats, candidates, pop_size):
    """Top ``pop_size`` candidates by the weighted objective, as PopulationGraph objects.

    Always a maximization, and always on the same column: direction is carried entirely by
    the sign of the weights, so there is no objective flag to get out of step with them.

    wl_hash is the tiebreaker, and it is load-bearing rather than decorative. Exact ties
    are routine, not freak events: prob_fixation is k/n_repeats, an integer over a fixed
    denominator, so two graphs with the same k give bit-identical floats. On the 100-
    generation run of 2026_07_28, 27 of 220 rows per generation (12%) shared a value with
    another row, and in 7 of 99 generations the tie landed exactly on the elite cutoff,
    where it decides which graph survives.

    Sorted on the metric alone, that decision falls out of the row order pandas happens to
    have, which comes from the order the shards were globbed -- so a rerun with identical
    seeds can keep a different graph, and since a different elite breeds different children
    forever after, one flip forks the whole trajectory.

    A hash is the right tiebreak precisely because it is a hash: uncorrelated with fitness,
    so it plays no favourites, yet fixed, so it costs no RNG state and reproduces exactly.
    That is a coin flip's fairness with a rule's determinism.
    """
    ranked = stats.sort_values(["weighted", "wl_hash"], ascending=[False, True])
    ranked = ranked.reset_index(drop=True)
    by_hash = {g.wl_hash: g for g in candidates}
    elites = [by_hash[h] for h in ranked["wl_hash"].head(pop_size)]
    return elites, ranked


def _append_history(run_dir, generation, ranked, pop_size, parents, elite_hashes):
    """One row per (generation, candidate): the full lineage, recoverable after the run.

    The ML notebook kept only the mean predicted fitness of survivors in memory, so it
    could never answer which graph won or when a lineage appeared. This can.
    """
    rows = ranked.assign(
        generation=generation,
        rank=np.arange(1, len(ranked) + 1),
        survived=np.arange(len(ranked)) < pop_size,
        is_new=~ranked["wl_hash"].isin(elite_hashes),
        parent_wl_hash=ranked["wl_hash"].map(parents),
    )
    # No separate 'fitness' column any more: it used to be a copy of whichever metric the
    # run selected on, which was the only way to compare a mean_steps run against a
    # prob_fixation one. Every run now selects on 'weighted', so the copy carried no
    # information and gave two names to one number.
    columns = [
        "generation", "wl_hash", "graph_name", "parent_wl_hash",
        "prob_fixation", "prob_fixation_sem", "mean_steps", "mean_steps_sem",
        "std_steps", "n_grouped", "weighted", "weighted_sem",
        "rank", "survived", "is_new",
    ]
    path = Path(run_dir) / "ga_history.csv"
    rows[columns].to_csv(path, mode="a", header=not path.exists(), index=False)


def _progress_line(cfg, generation, ranked, n_new_elites, elapsed, start_gen=0):
    """A one-line bar for the driver's LSF log.

    The driver runs detached, so this is how you see where the search is from a
    ``tail -f``. Written as a whole line per generation rather than a carriage-return
    animation, which renders as garbage in an LSF .out file.

    ``elapsed`` is measured from *this* driver invocation, so the per-generation rate
    divides by the generations this invocation ran, not by the progress bar's ``done``.
    A run resumed at generation 60 has done 61 but may have run only one of them here,
    and dividing by 61 would report an ETA about sixty times too short.
    """
    total = cfg["generations"]
    done = generation + 1
    ran_here = generation - start_gen + 1
    filled = min(22, max(0, int(22 * done / total)))
    best = ranked["weighted"].iloc[0]
    median = ranked["weighted"].median()
    remaining = (total - done) * (elapsed / ran_here) if ran_here > 0 else 0
    return (
        f"[{theta_category(cfg['theta'])} {quadrant_name(cfg['theta'])}] "
        f"gen {done:3d}/{total} "
        f"|{'#' * filled}{'-' * (22 - filled)}| {100 * done / total:3.0f}%  "
        f"best={best:.4g}  median={median:.4g}  new_elites={n_new_elites}  "
        f"eta {int(remaining // 3600)}h{int(remaining % 3600 // 60):02d}m"
    )


def _write_state(run_dir, state, seen):
    """Rewrite ga_state.json and the seen-topology set.

    JSON rather than a pickled object so a half-finished run stays inspectable with
    ``cat`` while it is still executing. The seen-set lives in its own file because it
    grows to ~20k hashes over a full run, which would bury the dozen fields you actually
    want to read.
    """
    (Path(run_dir) / "ga_state.json").write_text(json.dumps(state, indent=2))
    (Path(run_dir) / "seen_hashes.txt").write_text("\n".join(sorted(seen)))


# ======================================================================================
# Driver
# ======================================================================================


def run_search(cfg):
    run_dir = Path(cfg["run_dir"])
    theta = normalize_theta(cfg["theta"])
    w_prob, w_time = theta_weights(theta)
    category = theta_category(theta)
    state_path = run_dir / "ga_state.json"

    if state_path.exists():
        # The normal path after an LSF preemption, which requeues the job but restarts it
        # from the beginning.
        state = json.loads(state_path.read_text())
        start_gen = state["generation"] + 1
        elites = joblib.load(run_dir / "populations" / f"gen_{state['generation']:03d}.pkl")
        # Restored in elite_hashes order, not in the pickle's candidate order. The state
        # records the hashes straight off _select, so that list IS the fitness ranking,
        # and a plain filter would silently discard it. Normally the next generation's
        # _select re-ranks and the difference washes out, but it does not wash out when
        # the loop body never runs -- relaunching an already-finished run would rewrite
        # final_population.pkl unranked, and everything downstream that takes
        # population[:3] as "the best three" would quietly be showing three arbitrary ones.
        by_hash = {g.wl_hash: g for g in elites}
        elites = [by_hash[h] for h in state["elite_hashes"] if h in by_hash]
        seen = set((run_dir / "seen_hashes.txt").read_text().split())
        rng = np.random.default_rng()
        rng.bit_generator.state = state["rng_state"]
        # Extending a finished run (--generations 200 on a run that did 100) is a normal
        # thing to do, so the target has to follow the new argument rather than the one
        # the run was originally launched with -- otherwise every reader of ga_state.json
        # reports progress against a stale denominator.
        state["generations"] = cfg["generations"]
        log.info(
            "Resuming %s at generation %d of %d.",
            run_label(run_dir), start_gen, cfg["generations"],
        )
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        # ntfy_topic is dropped rather than serialised: anyone holding the topic can push
        # notifications to the phone, and ga_config.json is a shared-filesystem file. It
        # is not needed here anyway, since it is re-read from the environment on resume
        # rather than loaded back from this file.
        (run_dir / "ga_config.json").write_text(
            json.dumps({k: v for k, v in cfg.items() if k != "ntfy_topic"}, indent=2)
        )
        rng = np.random.default_rng(cfg["seed"])
        elites = build_initial_population(cfg["pop_size"], cfg["seed"], category)
        seen = {g.wl_hash for g in elites}
        start_gen = 0
        state = {
            "run": run_label(run_dir),
            "theta": theta,
            "weight_prob": w_prob,
            "weight_time": w_time,
            # Recorded rather than re-derived, so every reader labels a run the same way
            # without importing this module to recompute it.
            "category": category,
            "quadrant": quadrant_name(theta),
            "generations": cfg["generations"],
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "warnings": [],
        }
        log.info(
            "Starting %s: theta=%g (%s), w=(%+.3f, %+.3f), %d generations.",
            run_label(run_dir), theta, quadrant_name(theta), w_prob, w_time,
            cfg["generations"],
        )

    started = time.monotonic()
    # Carried across generations and re-estimated from each one's measured results. On a
    # resume it starts from the checkpointed value, so a preempted run does not go back to
    # sizing its arrays as though it were at generation 0.
    seconds_per_sim = state.get("seconds_per_sim") or SECONDS_PER_SIM

    for generation in range(start_gen, cfg["generations"]):
        elite_hashes = {g.wl_hash for g in elites}

        if generation == 0:
            # Generation 0 measures the initial population itself, so the trajectory has a
            # real starting point rather than one implied by the first round of children.
            candidates, parents = list(elites), {}
        else:
            children, parents = _reproduce(
                elites, cfg["n_children"], seen, rng, generation
            )
            # Elites are re-simulated alongside their children. Carrying an elite's old
            # score forward would mean it is never re-tested, so a lucky-high estimate
            # would sit at the top of the ranking permanently while freshly-measured
            # children could never displace it.
            candidates = list(elites) + children

        stats, gen_dir = _run_generation(
            state, cfg, candidates, generation, seconds_per_sim
        )
        seconds_per_sim = _estimate_seconds_per_sim(stats) or seconds_per_sim
        elites, ranked = _select(stats, candidates, cfg["pop_size"])
        n_new_elites = int(
            (~ranked["wl_hash"].head(cfg["pop_size"]).isin(elite_hashes)).sum()
        )

        _append_history(
            run_dir, generation, ranked, cfg["pop_size"], parents, elite_hashes
        )
        _prune_raw(gen_dir)

        elapsed = time.monotonic() - started
        state.update(
            generation=generation,
            pct=round(100 * (generation + 1) / cfg["generations"], 1),
            elite_hashes=[g.wl_hash for g in elites],
            n_seen=len(seen),
            rng_state=rng.bit_generator.state,
            best_fitness=float(ranked["weighted"].iloc[0]),
            median_fitness=float(ranked["weighted"].median()),
            n_new_elites=n_new_elites,
            seconds_per_sim=seconds_per_sim,
            last_generation_at=datetime.now().isoformat(timespec="seconds"),
            eta_seconds=int((cfg["generations"] - generation - 1) * elapsed
                            / (generation - start_gen + 1)),
            status="running",
        )
        _write_state(run_dir, state, seen)
        log.info(
            "%s",
            _progress_line(cfg, generation, ranked, n_new_elites, elapsed, start_gen),
        )

    joblib.dump(elites, run_dir / "final_population.pkl")
    state["status"] = "finished"
    _write_state(run_dir, state, seen)
    log.info("Done. Final population -> %s", run_dir / "final_population.pkl")

    # Two clocks, because on a resumed run they differ and both are worth knowing: total
    # is what you waited from launch, compute is what this attempt actually spent. On a
    # run that was never preempted they are the same and only one is printed.
    total = _elapsed_since_launch(state)
    this_attempt = time.monotonic() - started
    generations_run = cfg["generations"] - start_gen
    timing = f"in {_format_duration(total if total else this_attempt)}"
    if total and total - this_attempt > 120:
        timing += f" ({_format_duration(this_attempt)} of it computing, after a resume)"
    per_generation = this_attempt / max(generations_run, 1)

    # Suppressed by submit_theta_runs, which sends one summary for the whole launch
    # instead. A lone submit_driver still announces itself.
    notify(
        cfg.get("ntfy_topic") if cfg.get("notify_on_finish", True) else "",
        f"GA finished: {run_label(run_dir)}",
        f"theta={theta:g} ({quadrant_name(theta)}) over {cfg['generations']} "
        f"generations {timing}\n"
        f"{_format_duration(per_generation)} per generation\n"
        f"best={state['best_fitness']:.6g}  median={state['median_fitness']:.6g}\n"
        f"{state['n_seen']} topologies evaluated",
    )


def submit_driver(
    run_dir,
    theta,
    generations=100,
    pop_size=20,
    n_children=10,
    n_repeats=500_000,
    seed=42,
    queue="gsla-cpu",
    driver_queue="gsla-cpu",
    driver_walltime="12:00",
    notify_on_finish=True,
    extra_args=(),
):
    """bsub one driver job. Returns its LSF job id.

    The driver holds the GA state, waits on LSF, and rolls each generation up itself, so
    it asks for one slot and 4GB. It was 2GB while aggregation was a separate 16GB job;
    that job was measured at 513MB peak on a generation, and 4GB leaves room for it plus
    the driver's own pandas footprint on a slot that is otherwise idle anyway.
    ``driver_queue`` needs a walltime that outlasts the whole
    run (~3.5h at the default settings), and ideally is not preemptable: a preempted
    driver resumes from ga_state.json, but the generation it was mid-way through gets
    thrown away and resimulated. gsla-cpu is PREEMPTIVE (it preempts, it is not preempted)
    with a 45000-minute limit, which is why it is the default here; short/medium/long are
    all PREEMPTABLE. The cost is that a nearly-idle slot sits in the group allocation for
    the length of the run.

    ``queue`` is what the per-generation simulation arrays are submitted to, which is a
    different decision: those are many short jobs and want the high-priority queue.
    """
    run_dir = Path(run_dir)
    logs_dir = run_dir.parent / "_driver_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Said once, at submit time, because the alternative is finding out five hours later
    # that no notification is coming. The trap this catches is a Jupyter kernel: it
    # inherits the environment of the shell that started it, so exporting NTFY_TOPIC in
    # .bashrc after the server was launched leaves this empty until the server restarts.
    topic = os.environ.get("NTFY_TOPIC", "")
    complaint = None
    if not topic:
        complaint = (
            "NTFY_TOPIC is not set in this environment, so no notification will be sent "
            "when the run ends. Set it in ~/.bashrc and restart the Jupyter server (or "
            "the kernel's shell) for it to be picked up."
        )
    elif not _VALID_TOPIC.fullmatch(topic):
        # Checked here rather than left to the POST, because notify() swallows its errors
        # by design: an invalid topic would otherwise surface as a 404 buried in a driver
        # log, hours later, on the run that was supposed to be announcing itself.
        complaint = (
            f"NTFY_TOPIC={topic!r} is not a usable ntfy topic. Topics are limited to "
            "letters, digits, '-' and '_' (max 64): '#' truncates the URL because "
            "everything after it is a fragment the server never sees, and percent-encoding "
            "the punctuation gives a 404. Unguessability has to come from length, not from "
            "symbols. No notification will be sent."
        )
    if complaint and not getattr(submit_driver, "_warned_topic", False):
        print(f"note: {complaint}")
        submit_driver._warned_topic = True
    if complaint:
        topic = ""

    label = run_label(run_dir)
    cmd = [
        "bsub",
        "-q", driver_queue,
        "-J", f"ga_{label}",
        "-W", driver_walltime,
        "-R", "rusage[mem=4096]",
        "-o", str(logs_dir / f"{label}_%J.out"),
        "-e", str(logs_dir / f"{label}_%J.err"),
        # NTFY_TOPIC is forwarded from the submitting shell, so the driver can notify
        # without the topic ever being written into the repo or a job's command line.
        # Thread limits, and a requeue-on-crash. The driver is a 1-slot job that spends
        # its life asleep, so LSF packs many of them onto one node -- 7 of the 8 corner
        # drivers landed on cn773. Each then periodically turns into a polars scan of the
        # generation's shards, and polars sizes its thread pool from the machine's core
        # count (168 here), not from the slot it was given. Seven of those at once
        # segfaulted four drivers mid-run with 681MB of a 4GB reservation in use, so it
        # was contention, not memory. This is the cost of aggregating in the driver
        # instead of in its own job; one thread each is ample for 19s of work.
        "-env",
        f"PYTHONPATH=src, NTFY_TOPIC={topic}, POLARS_MAX_THREADS=1, "
        f"OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1",
        # Requeue rather than die if it happens anyway. A driver resumes from
        # ga_state.json by design, so a crashed one losing its current generation and
        # continuing is strictly better than sitting dead until someone notices -- which
        # is what cost this launch a night.
        "-Q", "139",
        sys.executable, "-u", "-m", "moran_process.pipeline.ga_search",
        "--run-dir", str(run_dir),
        "--theta", str(normalize_theta(theta)),
        "--generations", str(generations),
        "--pop-size", str(pop_size),
        "--n-children", str(n_children),
        "--n-repeats", str(n_repeats),
        "--seed", str(seed),
        "--queue", queue,
        *([] if notify_on_finish else ["--no-finish-notify"]),
        *extra_args,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    job_id = None
    if result.returncode == 0:
        match = re.search(r"Job <(\d+)>", result.stdout)
        job_id = match.group(1) if match else None
        print(f"{label:34s} submitted as LSF job {job_id or 'unknown'}")
    else:
        print(f"{label:34s} bsub FAILED: {(result.stderr or '').strip()}")
    return job_id


def run_dir_for(ga_runs_dir, prefix, theta, replicate=0):
    """Where a (prefix, theta, replicate) run lives: ``<prefix>-theta<NNN>/rep<K>/``.

    ``theta315`` rather than ``theta-45``: the angle is normalized first, which keeps the
    name free of sign characters and, more importantly, makes two launches that mean the
    same direction land under the same theta directory.

    Nested rather than a flat ``<prefix>-theta315-rep0`` name (the layout before
    2026-08-24) so every replicate of one direction lives together and is one glob away:
    ``(ga_runs_dir / f"{prefix}-theta315").glob("rep*")`` finds them all, where the flat
    layout needed a prefix match plus a regex to avoid also matching theta315's own
    directory when replicates == 1. ``replicate`` always has a value now -- the old
    ``None`` meant "no -repN suffix" for a single-replicate launch, which was the second
    representation of "how many replicates" alongside the ``replicates`` argument itself,
    and the two disagreeing is exactly the class of bug the theta refactor (2026-08-19)
    was meant to eliminate.
    """
    theta_dir = f"{prefix}-theta{normalize_theta(theta):03.0f}"
    return Path(ga_runs_dir) / theta_dir / f"rep{replicate}"


def run_label(run_dir):
    """Flat identity string for a run directory: ``.../PREFIX-theta315/rep0`` ->
    ``PREFIX-theta315-rep0``.

    Reconstructs the pre-nesting flat name, which is what every place that treats a run as
    an opaque unique string -- the ``run`` column in ``ga_history.csv``, LSF job names,
    driver log filenames, ntfy messages -- used before the directory layout changed.
    Nesting was a filesystem-layout decision; it does not have to also change every
    identifier derived from a run directory's name, so this is the one place that
    translates between the two.
    """
    run_dir = Path(run_dir)
    return f"{run_dir.parent.name}-{run_dir.name}"


def submit_theta_runs(
    ga_runs_dir, prefix, thetas, replicates=1, seed=42, **kwargs
):
    """Submit one driver per direction per replicate. The only launcher.

    ``thetas`` is any iterable of angles in degrees -- ``CORNER_THETAS`` for the four
    diagonals, ``np.linspace(0, 360, K, endpoint=False)`` to sweep a circle, or a single
    ``[45]``. Duplicates are rejected after normalization rather than silently collapsed,
    since ``[45, 405]`` almost certainly means a mistake and would otherwise produce one
    run where two were asked for.

    Replicates are the phase-2 mechanism for "did independent searches in the SAME
    direction converge on the same topology": replicate k gets ``seed + k*1000``, which
    reseeds the initial population, the mutation stream and the simulation seeds together,
    so the runs share nothing but their objective.
    """
    thetas = [normalize_theta(t) for t in thetas]
    duplicates = {t for t in thetas if thetas.count(t) > 1}
    if duplicates:
        raise ValueError(
            f"repeated direction(s) {sorted(duplicates)} after normalizing to [0, 360). "
            f"Angles that differ by a multiple of 360 are the same direction and would "
            f"collide in one run directory. For independent repeats of one direction, "
            f"use replicates=."
        )

    jobs, run_dirs = {}, []
    for replicate in range(replicates):
        for theta in thetas:
            run_dir = run_dir_for(ga_runs_dir, prefix, theta, replicate)
            run_dirs.append(run_dir)
            jobs[run_label(run_dir)] = submit_driver(
                run_dir,
                theta=theta,
                seed=seed + replicate * 1000,
                # One message for the launch, not one per run. Failures still notify
                # individually and immediately: a run that dies at hour four is worth
                # interrupting for, and folding it into the summary would mean hearing
                # about it only once its siblings also finished.
                notify_on_finish=False,
                **kwargs,
            )
    submit_summary_job(run_dirs, list(jobs.values()), prefix)

    # One ping at launch, so the notification path is proved now rather than assumed for
    # the next several hours. The failure it exists to catch is silent by construction:
    # posting to a valid but unsubscribed topic succeeds, so a typo in NTFY_TOPIC looks
    # exactly like a working setup until the run ends and nothing arrives.
    notify(
        os.environ.get("NTFY_TOPIC", ""),
        f"GA launched: {prefix}",
        f"{len(jobs)} runs over {len(thetas)} direction(s) x {replicates} replicate(s)\n"
        + "\n".join(f"theta={t:g}: {quadrant_name(t)}" for t in thetas)
        + "\nOne summary message when they have ALL finished.",
    )
    return jobs


def submit_summary_job(run_dirs, job_ids, prefix, queue="short", walltime="0:10"):
    """bsub a tiny job that waits on every driver and sends ONE summary notification.

    An LSF dependency rather than polling, and rather than having the last driver notice
    it is last. Each driver is an independent job with no view of its siblings, so
    "am I the last one?" would be a race: two finishing in the same second both see one
    unfinished sibling and neither sends, or both do.

    ``ended()`` not ``done()``, deliberately. ``done()`` requires success, so a driver that
    dies would leave this job PEND forever and the one message you were waiting for is the
    one you never get. ``ended()`` fires either way, and summarize_runs reads the state
    files to report which runs actually made it.

    Returns the watcher's job id, or None if it could not be submitted (in which case the
    per-run notifications are the fallback, so nothing is silently lost).
    """
    job_ids = [j for j in job_ids if j]
    if not job_ids:
        return None
    dependency = " && ".join(f"ended({job_id})" for job_id in job_ids)
    logs_dir = Path(run_dirs[0]).parent / "_driver_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "bsub",
        "-q", queue,
        "-J", f"ga_summary_{prefix}",
        "-w", dependency,
        "-W", walltime,
        "-R", "rusage[mem=1024]",
        "-o", str(logs_dir / f"summary_{prefix}_%J.out"),
        "-e", str(logs_dir / f"summary_{prefix}_%J.err"),
        "-env", f"PYTHONPATH=src, NTFY_TOPIC={os.environ.get('NTFY_TOPIC', '')}",
        sys.executable, "-u", "-m", "moran_process.pipeline.ga_search",
        "--summarize", *[str(d) for d in run_dirs],
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"summary job bsub FAILED: {(result.stderr or '').strip()}")
        return None
    match = re.search(r"Job <(\d+)>", result.stdout)
    job_id = match.group(1) if match else None
    print(f"summary notification job {job_id or 'unknown'} (waits for all {len(job_ids)} runs)")
    return job_id


def _main_summarize(argv):
    """``--summarize <run_dir> ...``: send one message covering a whole launch.

    Its own tiny entry point rather than a mode of the driver parser, because it shares
    none of the driver's arguments: it takes a list of directories and no run parameters.
    This is what submit_summary_job bsubs behind an ``ended()`` dependency on every driver.
    """
    parser = argparse.ArgumentParser(description="Summarize a finished GA launch.")
    parser.add_argument("--summarize", nargs="+", required=True, metavar="RUN_DIR")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S", stream=sys.stdout,
    )
    title, body, priority = summarize_runs(args.summarize)
    # Printed as well as pushed, so the summary survives in the job log even if the
    # notification does not go through.
    print(f"{title}\n{body}")
    notify(os.environ.get("NTFY_TOPIC", ""), title, body, priority=priority)


def main():
    if "--summarize" in sys.argv:
        return _main_summarize(sys.argv[1:])

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--theta", type=float, required=True,
        help="Search direction in degrees: w_prob=cos(theta), w_time=sin(theta). "
        "0 seeks high fixation probability, 90 long fixation time, 180 low probability, "
        "270 short time. Normalized to [0, 360), so -45 and 315 are the same run.",
    )
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--pop-size", type=int, default=20)
    parser.add_argument("--n-children", type=int, default=10)
    parser.add_argument("--n-repeats", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--queue", default="gsla-cpu")
    parser.add_argument(
        "--no-finish-notify",
        dest="notify_on_finish",
        action="store_false",
        help="Do not notify when THIS run finishes. Set by submit_theta_runs, which "
        "sends a single summary once every run in the launch has ended. Failure "
        "notifications are unaffected.",
    )
    parser.add_argument("--memory", default="1GB")
    parser.add_argument("--engine", choices=["cpp", "python"], default="cpp")
    parser.add_argument(
        "--generation-timeout-s", type=int, default=DEFAULT_GENERATION_TIMEOUT_S
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Wipe an existing run directory and start over. Without it, an existing "
        "ga_state.json is resumed from (which is what makes an LSF preemption harmless).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        # stdout, not the default stderr, so the per-generation progress bar lands in the
        # job's .out file -- which is what you tail -f.
        stream=sys.stdout,
    )

    run_dir = Path(args.run_dir)
    if args.force and run_dir.exists():
        log.warning("--force: removing %s", run_dir)
        shutil.rmtree(run_dir)

    cfg = vars(args)
    cfg.pop("force")
    cfg["r_values"] = [R_VALUE]
    # Resolved here rather than in the bsub command line so the topic never appears in
    # `bjobs -l` output or in a saved ga_config.json.
    cfg["ntfy_topic"] = os.environ.get("NTFY_TOPIC", "")

    try:
        run_search(cfg)
    except BaseException as error:
        # Includes SystemExit (how a generation that stayed broken gives up) and the
        # SIGTERM-driven exceptions of an LSF kill. A run that dies at hour four is
        # exactly the one worth being told about, so notify before re-raising.
        #
        # State is re-read from disk rather than passed out of run_search, because the
        # interesting failures are the ones where run_search did not return anything at
        # all. It is the last checkpoint, so it says how far the run got before dying.
        state = json.loads((run_dir / "ga_state.json").read_text()) if (
            run_dir / "ga_state.json"
        ).exists() else {}
        elapsed = _elapsed_since_launch(state)
        reached = state.get("generation")
        # theta, not args.objective/args.metric. Those attributes were removed when a run
        # became a single angle, and this handler kept referencing them -- so the crash
        # reporter crashed with AttributeError, swallowing both the real error and the
        # notification that was supposed to announce it. It went unnoticed because this is
        # the one path that only executes when something has ALREADY gone wrong: five
        # drivers died overnight and the only trace was a bare exit code 1.
        notify(
            cfg["ntfy_topic"],
            f"GA FAILED: {run_label(run_dir)}",
            f"theta={args.theta:g} ({quadrant_name(args.theta)})\n"
            + (f"died after {_format_duration(elapsed)}" if elapsed else "died")
            + (
                f" at generation {reached + 1}/{args.generations}\n"
                if reached is not None
                else " before its first generation finished\n"
            )
            + f"{type(error).__name__}: {error}",
            priority="high",
        )
        raise


if __name__ == "__main__":
    main()

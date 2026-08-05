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
  * Only ``aggregate`` is chained after each generation's array. verify, the violin cache
    and job speed all exist to serve figures and QC on a large one-off batch; on a
    220-graph generation they are pure scheduling latency. See ``post_batch=`` in
    ``ProcessLab.submit_jobs``.

Selection reads ``prob_fixation`` and ``mean_steps`` straight out of the generation's
``graph_statistics.csv``, which is keyed on ``(wl_hash, r)`` -- the same ``wl_hash`` the GA
already uses to deduplicate candidates, so the join is free.

Run layout, and the reason each piece exists, is documented in ``GA_SIMULATION_PLAN.md``.

    python -m moran_process.pipeline.ga_search \
        --run-dir simulation_data/ga_runs/2026_07_29-max-mean_steps \
        --metric mean_steps --objective maximize
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

METRICS = ("mean_steps", "prob_fixation", "weighted")
OBJECTIVES = ("maximize", "minimize")

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
    """
    result = subprocess.run(
        ["bjobs", "-a", "-o", "stat", str(job_id)],
        capture_output=True,
        text=True,
    )
    lines = [ln.strip() for ln in result.stdout.splitlines() if ln.strip()]
    # Output is a STAT header followed by one row per index. Anything else (job not found,
    # bjobs unavailable) means we have no state to act on.
    if len(lines) < 2:
        return "GONE"
    states = {line.split()[0] for line in lines[1:]}
    unfinished = states - TERMINAL_STATES
    if unfinished:
        return sorted(unfinished)[0]
    # Every index is terminal. One EXIT makes the whole array's result suspect, and the
    # completeness check downstream is what decides whether it is actually unusable.
    return "EXIT" if "EXIT" in states else "DONE"


def _wait_for_job(job_id, timeout_s, label=""):
    """Block until ``job_id`` reaches a terminal state. Returns that state.

    Polling bjobs rather than watching for graph_statistics.csv to appear is deliberate:
    build_graph_statistics writes with a plain ``to_csv``, so a file that exists may still
    be half-written. Waiting for the job to leave the queue removes that race.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        state = _job_state(job_id)
        if state in TERMINAL_STATES:
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
            # A run that never wrote a state file has no objective/metric to report, so it
            # is identified by directory instead. Labelling it '? ?' would make several
            # such runs collapse into one indistinguishable line.
            "category": (
                f"{state['objective']} {state['metric']}"
                if state.get("objective") and state.get("metric")
                else Path(run_dir).name
            ),
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

    stats = pd.read_csv(stats_path)
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


def weighted_category(w_prob, w_time):
    """A readable name for a weighted run, e.g. ``high_prob low_time``.

    The direction lives in the signs of the weights, not in ``--objective``, so the
    category has to say which corner is being chased or two runs with opposite weights
    would both be called "maximize weighted" and collide in every figure's color map.
    Magnitudes are appended only when they are not 1:1, which keeps the common case short.
    """
    prob = {1: "high_prob", -1: "low_prob"}.get(int(np.sign(w_prob)), "any_prob")
    # Negative time weight means shorter fixation scores higher.
    time = {-1: "low_time", 1: "high_time"}.get(int(np.sign(w_time)), "any_time")
    label = f"{prob} {time}"
    if (abs(w_prob), abs(w_time)) != (1.0, 1.0):
        label += f" ({abs(w_prob):g}:{abs(w_time):g})"
    return label


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
    prob_term = w_prob * (stats["prob_fixation"] - RHO_COMPLETE) / SD_PROB_RESIDUAL
    time_term = w_time * np.log(stats["mean_steps"] / T_COMPLETE) / SD_LOG_TIME_RESIDUAL
    sem = np.sqrt(
        (w_prob / SD_PROB_RESIDUAL * stats["prob_fixation_sem"]) ** 2
        + (w_time / SD_LOG_TIME_RESIDUAL * stats["mean_steps_sem"] / stats["mean_steps"])
        ** 2
    )
    return stats.assign(weighted=prob_term + time_term, weighted_sem=sem)


def _run_generation(state, cfg, candidates, generation, seconds_per_sim):
    """Submit, wait, aggregate, verify, and resubmit on failure. Returns verified stats."""
    run_dir = Path(cfg["run_dir"])
    gen_dir = run_dir / "generations" / f"gen_{generation:03d}"

    for attempt in range(MAX_GENERATION_RETRIES + 1):
        # Always start from an empty directory, not only on a retry. A driver killed
        # mid-generation (preemption) restarts at this same generation, and any shards
        # its first attempt left behind would be aggregated together with the new ones --
        # inflating n_grouped past n_repeats, so the completeness check below would pass
        # on doubled data.
        if gen_dir.exists():
            shutil.rmtree(gen_dir)

        job_ids = _submit_generation(
            candidates,
            run_dir,
            gen_dir,
            batch_name=f"{run_dir.name}_gen_{generation:03d}",
            n_repeats=cfg["n_repeats"],
            queue=cfg["queue"],
            memory=cfg["memory"],
            engine=cfg["engine"],
            # Derived from the generation number rather than drawn from a running stream,
            # so a resubmitted generation cannot desynchronise every generation after it.
            batch_seed=cfg["seed"] * 100003 + generation,
            seconds_per_sim=seconds_per_sim,
        )
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
        complaint = _wait_for_shards(gen_dir, job_ids.get("n_shards", 1))
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
                if cfg["metric"] == "weighted":
                    stats = _add_weighted(
                        stats, cfg["weight_prob"], cfg["weight_time"]
                    )
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


def _select(stats, candidates, metric, objective, pop_size):
    """Top ``pop_size`` candidates by ``metric``, as PopulationGraph objects.

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
    ranked = stats.sort_values(
        [metric, "wl_hash"], ascending=[objective == "minimize", True]
    )
    ranked = ranked.reset_index(drop=True)
    by_hash = {g.wl_hash: g for g in candidates}
    elites = [by_hash[h] for h in ranked["wl_hash"].head(pop_size)]
    return elites, ranked


def _append_history(run_dir, generation, ranked, metric, pop_size, parents, elite_hashes):
    """One row per (generation, candidate): the full lineage, recoverable after the run.

    The ML notebook kept only the mean predicted fitness of survivors in memory, so it
    could never answer which graph won or when a lineage appeared. This can.
    """
    rows = ranked.assign(
        generation=generation,
        fitness=ranked[metric],
        rank=np.arange(1, len(ranked) + 1),
        survived=np.arange(len(ranked)) < pop_size,
        is_new=~ranked["wl_hash"].isin(elite_hashes),
        parent_wl_hash=ranked["wl_hash"].map(parents),
    )
    columns = [
        "generation", "wl_hash", "graph_name", "parent_wl_hash",
        "prob_fixation", "prob_fixation_sem", "mean_steps", "mean_steps_sem",
        "std_steps", "n_grouped", "fitness", "rank", "survived", "is_new",
    ]
    # Present only on weighted runs; the readers key off the metric name, so an
    # absent column is never silently read as zero.
    columns += [c for c in ("weighted", "weighted_sem") if c in ranked]
    path = Path(run_dir) / "ga_history.csv"
    rows[columns].to_csv(path, mode="a", header=not path.exists(), index=False)


def _progress_line(cfg, generation, ranked, metric, n_new_elites, elapsed):
    """A one-line bar for the driver's LSF log.

    The driver runs detached, so this is how you see where the search is from a
    ``tail -f``. Written as a whole line per generation rather than a carriage-return
    animation, which renders as garbage in an LSF .out file.
    """
    total = cfg["generations"]
    done = generation + 1
    filled = min(22, max(0, int(22 * done / total)))
    best = ranked[metric].iloc[0]
    median = ranked[metric].median()
    remaining = (total - done) * (elapsed / done) if done else 0
    return (
        f"[{cfg['objective']} {metric}] gen {done:3d}/{total} "
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
    metric, objective = cfg["metric"], cfg["objective"]
    category = (
        weighted_category(cfg["weight_prob"], cfg["weight_time"])
        if metric == "weighted"
        else f"{objective} {metric}"
    )
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
            run_dir.name, start_gen, cfg["generations"],
        )
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "ga_config.json").write_text(json.dumps(cfg, indent=2))
        rng = np.random.default_rng(cfg["seed"])
        elites = build_initial_population(cfg["pop_size"], cfg["seed"], category)
        seen = {g.wl_hash for g in elites}
        start_gen = 0
        state = {
            "run": run_dir.name,
            "metric": metric,
            "objective": objective,
            # Recorded rather than re-derived: for a weighted run it names the corner
            # and cannot be reconstructed from objective + metric alone.
            "category": category,
            "generations": cfg["generations"],
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "warnings": [],
        }
        log.info("Starting %s: %s %s, %d generations.", run_dir.name, objective, metric,
                 cfg["generations"])

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
        elites, ranked = _select(stats, candidates, metric, objective, cfg["pop_size"])
        n_new_elites = int(
            (~ranked["wl_hash"].head(cfg["pop_size"]).isin(elite_hashes)).sum()
        )

        _append_history(
            run_dir, generation, ranked, metric, cfg["pop_size"], parents, elite_hashes
        )
        _prune_raw(gen_dir)

        elapsed = time.monotonic() - started
        state.update(
            generation=generation,
            pct=round(100 * (generation + 1) / cfg["generations"], 1),
            elite_hashes=[g.wl_hash for g in elites],
            n_seen=len(seen),
            rng_state=rng.bit_generator.state,
            best_fitness=float(ranked[metric].iloc[0]),
            median_fitness=float(ranked[metric].median()),
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
            _progress_line(cfg, generation, ranked, metric, n_new_elites, elapsed),
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

    # Suppressed by submit_all_runs, which sends one summary for the whole launch
    # instead. A lone submit_driver still announces itself.
    notify(
        cfg.get("ntfy_topic") if cfg.get("notify_on_finish", True) else "",
        f"GA finished: {run_dir.name}",
        f"{objective} {metric} over {cfg['generations']} generations {timing}\n"
        f"{_format_duration(per_generation)} per generation\n"
        f"best={state['best_fitness']:.6g}  median={state['median_fitness']:.6g}\n"
        f"{state['n_seen']} topologies evaluated",
    )


def submit_driver(
    run_dir,
    metric,
    objective,
    generations=100,
    pop_size=20,
    n_children=10,
    n_repeats=500_000,
    seed=42,
    queue="gsla-cpu",
    driver_queue="gsla-cpu",
    driver_walltime="12:00",
    notify_on_finish=True,
    weight_prob=1.0,
    weight_time=-1.0,
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

    cmd = [
        "bsub",
        "-q", driver_queue,
        "-J", f"ga_{run_dir.name}",
        "-W", driver_walltime,
        "-R", "rusage[mem=4096]",
        "-o", str(logs_dir / f"{run_dir.name}_%J.out"),
        "-e", str(logs_dir / f"{run_dir.name}_%J.err"),
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
        "--metric", metric,
        "--objective", objective,
        "--generations", str(generations),
        "--pop-size", str(pop_size),
        "--n-children", str(n_children),
        "--n-repeats", str(n_repeats),
        "--seed", str(seed),
        "--queue", queue,
        *(
            ["--weight-prob", str(weight_prob), "--weight-time", str(weight_time)]
            if metric == "weighted"
            else []
        ),
        *([] if notify_on_finish else ["--no-finish-notify"]),
        *extra_args,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    job_id = None
    if result.returncode == 0:
        match = re.search(r"Job <(\d+)>", result.stdout)
        job_id = match.group(1) if match else None
        print(f"{run_dir.name:34s} submitted as LSF job {job_id or 'unknown'}")
    else:
        print(f"{run_dir.name:34s} bsub FAILED: {(result.stderr or '').strip()}")
    return job_id


#: The two corners the single-objective runs cannot reach. The first is the interesting
#: one: among random (31, 34) graphs the two metrics are POSITIVELY correlated (+0.35),
#: so "fixes more often AND finishes sooner" is asking the search to break the natural
#: trend, and none of the four single-objective runs got near it -- the high-probability
#: runs all sat at long times, and the fast run gave up its probability advantage.
CORNERS = {
    "high_prob-low_time": (1.0, -1.0),
    "low_prob-high_time": (-1.0, 1.0),
}


def submit_corner_runs(
    ga_runs_dir, prefix, corners=None, replicates=1, seed=42, **kwargs
):
    """Submit weighted-objective runs, one per corner per replicate.

    ``corners`` maps a name to ``(weight_prob, weight_time)``; it defaults to CORNERS.
    Everything else works exactly as submit_all_runs, including the single summary
    notification once every run has ended.

    Direction lives in the weights, not in ``--objective``, so every one of these is a
    maximization: w = (+1, -1) maximizes "probability gain minus time cost" in units of
    random-graph SDs.
    """
    corners = dict(corners or CORNERS)
    jobs, run_dirs = {}, []
    for replicate in range(replicates):
        suffix = f"-rep{replicate}" if replicates > 1 else ""
        for name, (w_prob, w_time) in corners.items():
            run_dir = Path(ga_runs_dir) / f"{prefix}-{name}{suffix}"
            run_dirs.append(run_dir)
            jobs[run_dir.name] = submit_driver(
                run_dir,
                metric="weighted",
                objective="maximize",
                seed=seed + replicate * 1000,
                weight_prob=w_prob,
                weight_time=w_time,
                notify_on_finish=False,
                **kwargs,
            )
    submit_summary_job(run_dirs, list(jobs.values()), prefix)
    notify(
        os.environ.get("NTFY_TOPIC", ""),
        f"GA launched: {prefix}",
        f"{len(jobs)} weighted-objective runs submitted\n"
        + "\n".join(f"{n}: w=({p:+g}, {t:+g})" for n, (p, t) in corners.items())
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


def submit_all_runs(ga_runs_dir, prefix, replicates=1, seed=42, **kwargs):
    """Submit the 2x2 run matrix, optionally as ``replicates`` independent repeats.

    Within one replicate the four runs share a seed, and that is what makes them
    comparable: they start from the identical 20 random graphs, so any divergence between
    them is attributable to the objective rather than to where they started.

    Across replicates the seed changes, and it changes for everything at once. Replicate
    k gets ``seed + k * 1000``, which reseeds the initial population, the mutation stream
    and the per-task simulation seeds together. The alternative -- holding the starting
    population fixed and varying only mutation -- answers a narrower question (how
    path-dependent is the search from this one starting point). Varying both answers the
    question actually being asked: run the whole procedure again from scratch and does it
    arrive somewhere similar.

    Directories are suffixed ``-rep0``, ``-rep1``, ... only when replicates > 1, so a
    single-replicate launch keeps the names every existing figure and notebook expects.

    Note the load: 4 runs per replicate, each holding one driver slot plus an array of up
    to MAX_WORKERS. Three replicates is 12 concurrent runs, which will bump into the
    group's slot limit (``blimits -w -a -q gsla-cpu``, 680 for molgen) and simply queue.
    That costs wall clock but nothing else.
    """
    jobs, run_dirs = {}, []
    for replicate in range(replicates):
        suffix = f"-rep{replicate}" if replicates > 1 else ""
        for metric in METRICS:
            for objective in OBJECTIVES:
                run_dir = (
                    Path(ga_runs_dir) / f"{prefix}-{objective}-{metric}{suffix}"
                )
                run_dirs.append(run_dir)
                jobs[run_dir.name] = submit_driver(
                    run_dir,
                    metric,
                    objective,
                    seed=seed + replicate * 1000,
                    # One message for the launch, not one per run. Failures still notify
                    # individually and immediately: a run that dies at hour four is worth
                    # interrupting for, and waiting to fold it into the summary would mean
                    # hearing about it only once its siblings also finished.
                    notify_on_finish=False,
                    **kwargs,
                )
    submit_summary_job(run_dirs, list(jobs.values()), prefix)

    # One ping at launch, so the notification path is proved now rather than assumed for
    # the next several hours. The failure it exists to catch is silent by construction:
    # posting to a valid but unsubscribed topic succeeds, so a typo in NTFY_TOPIC looks
    # exactly like a working setup until the run ends and nothing arrives. If this message
    # does not appear on your phone, neither will the summary.
    notify(
        os.environ.get("NTFY_TOPIC", ""),
        f"GA launched: {prefix}",
        f"{len(jobs)} runs submitted"
        + (f" ({replicates} replicates)" if replicates > 1 else "")
        + "\nOne summary message when they have ALL finished.\n"
        + "If this is the only message you ever get, check NTFY_TOPIC.",
    )
    return jobs


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
    parser.add_argument("--metric", required=True, choices=METRICS)
    parser.add_argument("--objective", required=True, choices=OBJECTIVES)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--pop-size", type=int, default=20)
    parser.add_argument("--n-children", type=int, default=10)
    parser.add_argument("--n-repeats", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--queue", default="gsla-cpu")
    parser.add_argument(
        "--weight-prob", type=float, default=1.0,
        help="Weight on (rho - rho_c)/SD_PROB_RESIDUAL. Only used by "
        "--metric weighted. Positive seeks high fixation probability.",
    )
    parser.add_argument(
        "--weight-time", type=float, default=-1.0,
        help="Weight on log(T/T_c)/SD_LOG_TIME_RESIDUAL. Only used by "
        "--metric weighted. NEGATIVE seeks short fixation time.",
    )
    parser.add_argument(
        "--no-finish-notify",
        dest="notify_on_finish",
        action="store_false",
        help="Do not notify when THIS run finishes. Set by submit_all_runs, which "
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
        notify(
            cfg["ntfy_topic"],
            f"GA FAILED: {run_dir.name}",
            f"{args.objective} {args.metric}\n"
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

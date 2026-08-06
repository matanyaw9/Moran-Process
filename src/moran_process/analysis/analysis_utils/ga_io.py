"""Readers for a simulation-driven GA run (``pipeline.ga_search``).

Reader-only, following the builder/reader split the rest of this package uses: nothing
here computes or submits anything. ``ga_search`` writes ``ga_history.csv`` and
``ga_state.json``; a notebook, streamlit, or a terminal reads them through this module,
including while the run is still executing.

Needs only pandas, so progress can be checked without importing the plotting stack.
"""

import json
from pathlib import Path

import pandas as pd

from .constants import HASH_DTYPES

__all__ = [
    "load_ga_history",
    "load_ga_state",
    "ga_progress",
    "load_ga_runs",
    "final_population_stats",
    "load_elite_population",
    "final_elite_properties",
]


def _run_dirs(run_dirs):
    """Accept a single run directory or a list of them, uniformly."""
    if isinstance(run_dirs, (str, Path)):
        return [Path(run_dirs)]
    return [Path(d) for d in run_dirs]


def load_ga_state(run_dir):
    """The run's live state: generation, pct, best/median fitness, eta, warnings.

    Returns None if the run has not written a checkpoint yet (submitted but its first
    generation has not finished), which is a normal state rather than an error.
    """
    path = Path(run_dir) / "ga_state.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def load_ga_history(run_dirs, survivors_only=False):
    """Per-(generation, candidate) history for one or more runs, with a ``run`` column.

    Every candidate ever evaluated is present, not just survivors, so a lineage can be
    traced back through ``parent_wl_hash`` and a rejected branch is still visible. Pass
    ``survivors_only=True`` for the elite trajectory alone.

    Raises rather than returning empty if a run has no history: an absent ga_history.csv
    means the run has not completed its first generation, and silently returning nothing
    would look identical to a run that found nothing.
    """
    frames = []
    for run_dir in _run_dirs(run_dirs):
        path = run_dir / "ga_history.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"No ga_history.csv in {run_dir}. The run has not finished its first "
                f"generation yet (check ga_state.json), or --run-dir was wrong."
            )
        frame = pd.read_csv(path, dtype=HASH_DTYPES)
        frame["run"] = run_dir.name
        state = load_ga_state(run_dir)
        frame["metric"] = state["metric"] if state else None
        frame["objective"] = state["objective"] if state else None
        # Attached here, once, rather than re-derived by each figure as
        # objective + " " + metric. That derivation is right for a single-metric run and
        # wrong for a weighted one, where it yields "maximize weighted" for every corner
        # alike -- so two runs chasing opposite corners would share a color and a legend
        # entry, and the figure contrasting them would show them as the same thing.
        # ga_search records the real category in the state; the fallback covers runs
        # written before it did.
        frame["category"] = (state or {}).get("category") or (
            f"{frame['objective'].iloc[0]} {frame['metric'].iloc[0]}"
            if state
            else None
        )
        frames.append(frame)

    history = pd.concat(frames, ignore_index=True)
    if survivors_only:
        history = history[history["survived"]]
    return history


def _bar(done, total, width=22):
    # Clamped, so a run resumed past its recorded target cannot draw past the end of the
    # bar (or, with a negative remainder, past the end of the line).
    filled = min(width, max(0, int(width * done / total))) if total else 0
    return f"|{'#' * filled}{'-' * (width - filled)}|"


def ga_progress(run_dirs, width=22):
    """Print one progress bar per run. Safe to call while the runs are still going.

    The driver runs detached on a compute node, so this is how a notebook shows where the
    search is. It reads only ga_state.json, so it costs nothing and never blocks.

    Says so explicitly when handed nothing. A bare loop over an empty list prints nothing
    and returns cleanly, which makes "your prefix matched no directories" and "your runs
    have made no progress" byte-identical outcomes: silence. That is not hypothetical --
    editing PREFIX after launching produced exactly this, and the conclusion drawn was
    that the progress bar was broken rather than that it was reporting faithfully on an
    empty list.
    """
    run_dirs = _run_dirs(run_dirs)
    if not run_dirs:
        print(
            "No runs to report on: the list passed in is empty. Nothing is wrong with "
            "the runs, there simply are none here -- most likely PREFIX no longer "
            "matches the directories that were launched. Try load_ga_runs(GA_RUNS_DIR), "
            "which finds runs by the presence of ga_config.json rather than by name."
        )
        return

    for run_dir in run_dirs:
        state = load_ga_state(run_dir)
        if state is None:
            print(f"{run_dir.name:34s} {_bar(0, 1, width)}  not started")
            continue

        done = state["generation"] + 1
        total = state["generations"]
        eta = state.get("eta_seconds", 0)
        note = ""
        if state.get("status") == "finished":
            note = "  finished"
        else:
            note = f"  eta {eta // 3600}h{eta % 3600 // 60:02d}m"
        if state.get("warnings"):
            note += f"   {len(state['warnings'])} warning(s)"

        print(
            f"{run_dir.name:34s} {_bar(done, total, width)} {done:3d}/{total}"
            f"  best={state.get('best_fitness', float('nan')):.4g}{note}"
        )


def load_ga_runs(ga_runs_dir, prefix=None):
    """Every run directory under ``ga_runs_dir``, sorted by name.

    Identifies runs by the presence of ``ga_config.json``, not by name, so it cannot miss
    one because the prefix was typed differently. ``prefix`` narrows the result to a single
    launch and is checked: asking for a prefix that matches nothing raises rather than
    returning an empty list, because every caller of this treats an empty result as "no
    progress yet" and would report that instead of "you asked for the wrong thing".
    """
    runs = sorted(
        p for p in Path(ga_runs_dir).iterdir() if (p / "ga_config.json").exists()
    )
    if prefix is None:
        return runs

    matched = [p for p in runs if p.name.startswith(prefix)]
    if not matched:
        available = "\n  ".join(p.name for p in runs) or "(none)"
        raise FileNotFoundError(
            f"No run under {ga_runs_dir} starts with {prefix!r}. Runs found there:\n"
            f"  {available}"
        )
    return matched


def final_population_stats(run_dirs):
    """The last generation's surviving elites, one row per graph, across runs.

    This is the answer to "which graphs are the best": measured values, not predictions.

    "Last" means each run's own latest generation, which is not the same thing as its
    final one while a run is still going. Comparing a run stopped at generation 50 against
    one that reached 99 understates the first, and nothing about the resulting frame or
    the figures built on it would look wrong -- so a warning is printed rather than left
    for the reader to notice. Deliberately not an error: watching the comparison take
    shape mid-flight is a legitimate thing to do.
    """
    history = load_ga_history(run_dirs, survivors_only=True)
    last = history.groupby("run")["generation"].transform("max")
    result = history[history["generation"] == last].reset_index(drop=True)

    reached = result.groupby("run")["generation"].max()
    if reached.nunique() > 1:
        behind = reached[reached < reached.max()]
        print(
            f"WARNING: these runs have not reached generation {reached.max()} yet, so "
            f"they are being compared before they finished evolving:\n"
            + "\n".join(f"  {name}: generation {gen}" for name, gen in behind.items())
            + "\nCheck ga_progress(); rerun once every run is finished."
        )
    return result


def load_elite_population(run_dir):
    """The run's current best graphs, ranked, whether or not it has finished.

    ``final_population.pkl`` is only written when a run completes, so reading it directly
    raises FileNotFoundError on every run still in flight -- and on any run whose driver
    died, which is exactly when you most want to look at what it found. This falls back to
    reconstructing the elites from the last checkpoint: ``ga_state.json`` records the elite
    hashes in ranked order, and ``populations/gen_NNN.pkl`` holds the graph objects.

    Returns a list of PopulationGraph, best first. Raises only when the run has not
    completed a single generation, since then there is genuinely nothing to show.
    """
    import joblib

    run_dir = Path(run_dir)
    final = run_dir / "final_population.pkl"
    if final.exists():
        return joblib.load(final)

    state = load_ga_state(run_dir)
    if state is None or "elite_hashes" not in state:
        raise FileNotFoundError(
            f"{run_dir.name} has no final_population.pkl and no usable ga_state.json, so "
            f"it has not finished a generation yet. Check ga_progress()."
        )
    population = joblib.load(
        run_dir / "populations" / f"gen_{state['generation']:03d}.pkl"
    )
    by_hash = {g.wl_hash: g for g in population}
    # elite_hashes is written straight off the selection sort, so it IS the ranking; a
    # plain filter over the pickle would silently return them in candidate order.
    return [by_hash[h] for h in state["elite_hashes"] if h in by_hash]


def final_elite_properties(run_dirs):
    """Final elites with their measured fitness AND their structural properties.

    ``final_population_stats`` answers "how good is it". This answers "what is it": the
    same graphs joined to the ``graph_props.csv`` of the generation that produced them, so
    degree distribution, clustering, assortativity and the rest come along.

    That join is what makes replicates comparable. Two independent runs will essentially
    never rediscover the same graph -- a run visits ~20k topologies out of an astronomical
    space, so wl_hash overlap between replicates is expected to be zero and proves nothing
    either way. Whether they converged on the same *kind* of graph is a question about
    these columns.

    A ``replicate`` column is parsed off the ``-repN`` directory suffix that
    ``ga_search.submit_all_runs`` writes, and is None for a single-replicate run.
    """
    elites = final_population_stats(run_dirs)
    by_run = {d.name: d for d in _run_dirs(run_dirs)}

    frames = []
    for run, group in elites.groupby("run"):
        generation = int(group["generation"].iloc[0])
        props_path = (
            by_run[run] / "generations" / f"gen_{generation:03d}" / "graph_props.csv"
        )
        if not props_path.exists():
            raise FileNotFoundError(
                f"No graph_props.csv for {run} generation {generation} at {props_path}. "
                f"The generation directory is pruned of raw shards but graph_props.csv is "
                f"kept, so this means the generation never completed."
            )
        props = pd.read_csv(props_path, dtype=HASH_DTYPES).drop_duplicates(
            "wl_hash"
        )
        # 'category' and 'seed' exist on both sides and mean different things there, so
        # the props copies are dropped rather than suffixed into ambiguity.
        props = props.drop(columns=["category", "seed", "graph_name"], errors="ignore")
        frames.append(group.merge(props, on="wl_hash", how="left"))

    merged = pd.concat(frames, ignore_index=True)
    suffix = merged["run"].str.extract(r"-rep(\d+)$")[0]
    merged["replicate"] = pd.to_numeric(suffix, errors="coerce").astype("Int64")
    # Already carried by load_ga_history, which reads it from the run's state; the
    # objective+metric form would flatten every weighted run to "maximize weighted".
    return merged

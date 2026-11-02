"""Batch plans: JSON recipes for a simulation batch, and turning one into graphs and chunks.

A plan lists how to build graphs (factory, kwargs, count) plus the simulation knobs. It
never holds a graph, so it is a few KB and the batch can be rebuilt from it exactly.
"""

import inspect
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from moran_graphs.PopulationGraph import PopulationGraph, graph_size
from moran_graphs.properties import wl_hash

MAX_SEED_REDRAWS = 100
NOT_FACTORIES = {"sample_random_graphs"}  # a classmethod, but it returns a list
STEPS_PER_SECOND = 9e7  # plain Bd on a lung, for rough cost estimates only
Z95 = 1.96
HALF_WIDTH_TARGET = 0.02  # 95% relative CI half-width that stage two sizes n for


def factories():
    return sorted(
        name
        for name, member in vars(PopulationGraph).items()
        if isinstance(member, classmethod) and not name.startswith("_") and name not in NOT_FACTORIES
    )


def _factory(name):
    if name not in factories():
        raise ValueError(f"unknown factory {name!r}, available: {', '.join(factories())}")
    return getattr(PopulationGraph, name)


def _seedable(fn):
    return "seed" in inspect.signature(fn).parameters


def spec(factory, count=1, skip_null_steps=False, random_matches=0, up_to=False, **kwargs):
    """One plan line: `count` graphs from PopulationGraph.<factory>(**kwargs).

    random_matches > 0 asks for that many random graphs with the same N and E (see
    _random_match_specs). up_to makes count a maximum, for sizes with fewer distinct graphs.
    """
    fn = _factory(factory)
    if count < 1:
        raise ValueError(f"count must be >= 1, got {count}")
    if random_matches and factory == "random_graph":
        raise ValueError("random_matches on a random_graph spec: raise its count instead")
    if count > 1 and not (_seedable(fn) and "seed" not in kwargs):
        raise ValueError(f"{factory}({kwargs}) is deterministic, count={count} would repeat one graph")
    return {"factory": factory, "count": int(count), "skip_null_steps": bool(skip_null_steps),
            "random_matches": int(random_matches), "up_to": bool(up_to), "kwargs": kwargs}


def make_plan(
    batch_name,
    specs,
    r_values,
    n_repeats,
    *,
    zoo_seed,
    batch_seed,
    max_steps=10**10,
    n_workers=100,
    queue="gsla-cpu",
    memory_mb=2000,
    big_memory_mb=8000,
    keep_runs=False,
    chunk_target_steps=1e10,
    update_rule="Bd",
    description="",
):
    """A plan dict. zoo_seed draws the random graphs, batch_seed the simulations.

    memory_mb is per worker; big_memory_mb is for the prepare and finalize jobs.
    keep_runs also stores every run (winner, steps, n_changes), not only summaries.
    chunk_target_steps sizes the chunks a worker claims (see make_chunks).
    Specs with random_matches get their random graphs appended (see _random_match_specs).
    """
    specs = list(specs)
    if not specs or not all(isinstance(s, dict) and "factory" in s for s in specs):
        raise TypeError("specs must be a non-empty list of spec(...) dicts")
    specs += _random_match_specs(specs)
    r_values = [float(r) for r in r_values]
    if not r_values or min(r_values) <= 0:
        raise ValueError("r_values must be non-empty and positive")
    for label, seed in (("zoo_seed", zoo_seed), ("batch_seed", batch_seed)):
        if not isinstance(seed, (int, np.integer)):
            raise TypeError(f"{label} must be an int, every random draw needs a recorded seed")
    if not 1 <= n_repeats < 2**32:
        raise ValueError("n_repeats must be in [1, 2**32)")
    return {
        "batch_name": batch_name,
        "description": description,
        "zoo_seed": int(zoo_seed),
        "batch_seed": int(batch_seed),
        "sim": {
            "update_rule": update_rule,
            "r_values": r_values,
            "n_repeats": int(n_repeats),
            "max_steps": int(max_steps),
            "keep_runs": bool(keep_runs),
            "chunk_target_steps": float(chunk_target_steps),
        },
        "lsf": {
            "n_workers": int(n_workers),
            "queue": queue,
            "memory_mb": int(memory_mb),
            "big_memory_mb": int(big_memory_mb),
        },
        "specs": specs,
    }


def stage_two_plan(pilot_plan, pilot_results, batch_name, batch_seed, target=HALF_WIDTH_TARGET,
                   random_target=HALF_WIDTH_TARGET):
    """The pilot's graphs again (same specs and zoo_seed) with a fresh batch_seed, and
    n_repeats per (hash, r) sized from the pilot: a 95% relative CI half-width of at most
    `target` on both fixation_prob and mean_fixation_time (`random_target` for random
    graphs: lower it to make the reference band less noisy). The pilot only sizes n; its
    runs are not pooled.

    pilot_results is the pilot's report results table. Normal approximation, with p the
    fixation probability and CV the coefficient of variation of the fixation time:
    half-width / p = 1.96 sqrt((1 - p) / (n p)), and half-width / mean time =
    1.96 CV / sqrt(n p), since the mean time is over the n p fixed runs only.
    """
    if batch_seed == pilot_plan["batch_seed"]:
        raise ValueError("batch_seed equals the pilot's: the runs would not be fresh")
    if (pilot_results["n_fixed"] == 0).any():
        raise ValueError("pilot cells with no fixation cannot size n: rerun the pilot with more repeats")
    p = pilot_results["fixation_prob"]
    cv = pilot_results["fixation_time_sd"] / pilot_results["mean_fixation_time"]
    eps = np.where(pilot_results["factory"] == "random_graph", random_target, target)
    n = (Z95 / eps) ** 2 * np.maximum(1 - p, cv**2) / p
    n = np.ceil(n / 1000).astype(np.int64) * 1000
    plan = json.loads(json.dumps(pilot_plan))  # deep copy
    plan["batch_name"], plan["batch_seed"] = batch_name, int(batch_seed)
    plan["pilot"] = pilot_plan["batch_name"]  # plots.pilot_agreement compares the two
    plan["description"] = (f"stage two of {pilot_plan['batch_name']}: n per cell for a 95% relative half-width"
                           f" of {target:.1%} ({random_target:.1%} for random graphs)")
    plan["sim"]["n_repeats"] = [[h, float(r), int(k)] for h, r, k in zip(pilot_results["hash"], pilot_results["r"], n)]
    hours = (pilot_results["n_nodes"].astype(float) ** 2 * n).sum() / STEPS_PER_SECOND / 3600
    print(f"{len(n):,} cells, {n.sum():,} runs (n per cell {n.min():,} to {n.max():,}), rough cost"
          f" {hours:,.1f} CPU hours (N^2 steps per run, can be off 10x)")
    return plan


def repeats_per_cell(sim, graphs):
    """graphs (graph_idx, hash) x r_values with each cell's n_repeats: the plan's one number,
    or its [hash, r, n] table (a stage-two plan)."""
    cells = graphs[["graph_idx", "hash"]].merge(pd.DataFrame({"r": sim["r_values"]}), how="cross")
    if isinstance(sim["n_repeats"], int):
        return cells.assign(n_repeats=sim["n_repeats"])
    table = pd.DataFrame(sim["n_repeats"], columns=["hash", "r", "n_repeats"])
    cells = cells.merge(table, on=["hash", "r"], how="left")
    if cells["n_repeats"].isna().any():
        raise ValueError(f"{cells['n_repeats'].isna().sum()} (graph, r) cells have no n_repeats in the plan's table")
    return cells.astype({"n_repeats": np.int64})


def repeats_label(sim):
    reps = sim["n_repeats"]
    if isinstance(reps, int):
        return f"{reps:,} repeats"
    n = [k for _, _, k in reps]
    return f"{min(n):,} to {max(n):,} repeats per cell ({sum(n):,} runs)"


def save_plan(plan, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2) + "\n")
    return path


def load_plan(path):
    plan = json.loads(Path(path).read_text())
    for s in plan["specs"]:
        _factory(s["factory"])
    return plan


def describe(plan):
    """Prints what the plan will run and a rough cost."""
    sim, lsf = plan["sim"], plan["lsf"]
    n_r, reps = len(sim["r_values"]), sim["n_repeats"]
    print(f"Batch plan: {plan['batch_name']}  {plan['description']}")
    print(f"  zoo_seed={plan['zoo_seed']}  batch_seed={plan['batch_seed']}")
    if not isinstance(reps, int):
        # the table is keyed by graph hash, which only building the zoo gives; stage_two_plan prints the cost
        print(f"  {len(plan['specs'])} specs, {repeats_label(sim)}")
        return {"n_graphs": None, "runs": sum(k for _, _, k in reps), "cpu_hours": None}
    n_graphs = steps = 0
    for i, s in enumerate(plan["specs"]):
        n, _ = graph_size(s["factory"], **s["kwargs"])
        n_graphs += s["count"]
        steps += s["count"] * n_r * reps * n**2
        kw = ", ".join(f"{k}={v!r}" for k, v in s["kwargs"].items())
        skip = "  skip_null_steps" if s["skip_null_steps"] else ""
        up_to = " up to" if s["up_to"] else ""
        print(f"  [{i:>3}] {s['factory']}({kw}) x{s['count']}{up_to}  N={n}{skip}")
    runs = n_graphs * n_r * reps
    hours = steps / STEPS_PER_SECOND / 3600
    print(f"  {n_graphs:,} graphs x {n_r} r values x {reps:,} repeats = {runs:,} runs")
    print(f"  rough cost (N^2 steps per run, can be off 10x): {hours:,.1f} CPU hours,"
          f" ~{hours / lsf['n_workers']:,.2f} h on {lsf['n_workers']} workers ({lsf['queue']})")
    return {"n_graphs": n_graphs, "runs": runs, "cpu_hours": hours}


def _random_match_specs(specs):
    """One up_to random_graph spec per (N, E) that specs with random_matches ask for.

    Specs of the same size share one cloud of the largest random_matches, and random_graph
    specs already in the plan count toward it. Sizes keep first-appearance order, so a new
    size does not reorder the others. The specs come after the given ones, so they never
    change those graphs' seeds. A cloud skips null steps if any spec of its size does:
    skipping leaves the output distribution unchanged, so it only saves time.
    """
    cloud, skip = {}, {}
    for s in specs:
        if s["random_matches"]:
            if s["kwargs"].get("directed"):
                raise ValueError(f"{s['factory']}({s['kwargs']}) is directed, random_graph has no directed match")
            size = graph_size(s["factory"], **s["kwargs"])
            cloud[size] = max(cloud.get(size, 0), s["random_matches"])
            skip[size] = skip.get(size, False) or s["skip_null_steps"]
    for s in specs:
        size = (s["kwargs"].get("n_nodes"), s["kwargs"].get("n_edges"))
        if s["factory"] == "random_graph" and size in cloud:
            cloud[size] -= s["count"]
    return [spec("random_graph", count=k, up_to=True, skip_null_steps=skip[(n, e)], n_nodes=n, n_edges=e)
            for (n, e), k in cloud.items() if k > 0]


def build_zoo(plan):
    """The plan's graphs as records {graph, spec_idx, kwargs, hash}, deduplicated by hash.

    Spec i draws its graph seeds from the i-th child of zoo_seed, so appending a spec
    never changes earlier graphs. A seedable spec redraws on a hash collision and raises
    if it cannot find `count` distinct graphs, unless it is up_to: then that size has no
    graphs left that are not in the zoo, and it stops. A deterministic spec that collides
    is a duplicate line and is dropped.

    Graphs made as a series (a spec with count > 1) get an index in their name,
    random_n31_e40_i0, random_n31_e40_i1, ..., numbered across the whole zoo so two
    specs with the same parameters do not repeat names.
    """
    rngs = np.random.default_rng(plan["zoo_seed"]).spawn(len(plan["specs"]))
    zoo, seen, series_index = [], set(), Counter()
    for spec_idx, (s, rng) in enumerate(zip(plan["specs"], rngs)):
        fn = _factory(s["factory"])
        seedable = _seedable(fn) and "seed" not in s["kwargs"]
        for n_drawn in range(s["count"]):
            for _ in range(MAX_SEED_REDRAWS if seedable else 1):
                kwargs = dict(s["kwargs"])
                if seedable:
                    kwargs["seed"] = int(rng.integers(2**32))
                graph = fn(**kwargs)
                W = graph.csr()
                h = wl_hash(W)
                if h not in seen:
                    seen.add(h)
                    if s["count"] > 1:
                        base = graph.name
                        graph.name = f"{base}_i{series_index[base]}"
                        series_index[base] += 1
                    zoo.append({"graph": graph, "W": W, "spec_idx": spec_idx, "kwargs": kwargs, "hash": h})
                    break
            else:
                if s["up_to"]:
                    print(f"spec[{spec_idx}] {graph.name}: drew {n_drawn} of {s['count']},"
                          f" no new graph of this size after {MAX_SEED_REDRAWS} redraws", flush=True)
                    break
                if seedable:
                    raise ValueError(
                        f"spec[{spec_idx}] {s['factory']}({s['kwargs']}): only {n_drawn} of {s['count']} distinct"
                        f" graphs after {MAX_SEED_REDRAWS} redraws, lower count"
                    )
                print(f"spec[{spec_idx}] {s['factory']}: duplicate graph dropped", flush=True)
    return zoo


def make_chunks(plan, graphs):
    """Splits every (graph, r) cell into chunks of a few minutes, the unit a worker claims.

    graphs needs graph_idx, hash, n_nodes, spec_idx. Cost is estimated as N^2 steps per
    run. A chunk aims at chunk_target_steps, but small enough that there are at least 4
    chunks per worker. Chunks of a graph are contiguous and large graphs come first, so
    workers start on the slowest work and the short chunks fill the end.
    """
    sim = plan["sim"]
    skip = {i: s["skip_null_steps"] for i, s in enumerate(plan["specs"])}
    cells = repeats_per_cell(sim, graphs).merge(graphs[["graph_idx", "n_nodes", "spec_idx"]], on="graph_idx")
    cells = cells.sort_values(["n_nodes", "graph_idx"], ascending=[False, True], kind="stable")  # r order kept
    total = (cells["n_nodes"].astype(float) ** 2 * cells["n_repeats"]).sum()
    target = min(sim["chunk_target_steps"], total / (4 * plan["lsf"]["n_workers"]))

    rows = []
    for c in cells.itertuples():
        runs = int(np.clip(target // c.n_nodes**2, 1, c.n_repeats))
        k = -(-c.n_repeats // runs)  # chunks per cell, sizes differing by at most one run
        base, extra = divmod(c.n_repeats, k)
        for n_runs in [base + 1] * extra + [base] * (k - extra):
            rows.append((c.graph_idx, c.hash, c.r, n_runs, skip[c.spec_idx]))
    chunks = pd.DataFrame(rows, columns=["graph_idx", "hash", "r", "n_runs", "skip_null_steps"])
    seeds = np.random.SeedSequence(plan["batch_seed"]).spawn(len(chunks))
    chunks["seed"] = np.array([s.generate_state(1, np.uint64)[0] for s in seeds], dtype=np.uint64)
    chunks.insert(0, "chunk_id", np.arange(len(chunks)))
    return chunks

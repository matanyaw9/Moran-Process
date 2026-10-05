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

from moran_graphs.PopulationGraph import PopulationGraph
from moran_graphs.properties import wl_hash

MAX_SEED_REDRAWS = 100
NOT_FACTORIES = {"sample_random_graphs"}  # a classmethod, but it returns a list
STEPS_PER_SECOND = 9e7  # plain Bd on a lung, for rough cost estimates only


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


def spec(factory, count=1, skip_null_steps=False, **kwargs):
    """One plan line: `count` graphs from PopulationGraph.<factory>(**kwargs)."""
    fn = _factory(factory)
    if count < 1:
        raise ValueError(f"count must be >= 1, got {count}")
    if count > 1 and not (_seedable(fn) and "seed" not in kwargs):
        raise ValueError(f"{factory}({kwargs}) is deterministic, count={count} would repeat one graph")
    return {"factory": factory, "count": int(count), "skip_null_steps": bool(skip_null_steps), "kwargs": kwargs}


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
    """
    specs = list(specs)
    if not specs or not all(isinstance(s, dict) and "factory" in s for s in specs):
        raise TypeError("specs must be a non-empty list of spec(...) dicts")
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
    """Prints what the plan will run and a rough cost. Builds one graph per spec for N."""
    sim, lsf = plan["sim"], plan["lsf"]
    n_r, reps = len(sim["r_values"]), sim["n_repeats"]
    print(f"Batch plan: {plan['batch_name']}  {plan['description']}")
    print(f"  zoo_seed={plan['zoo_seed']}  batch_seed={plan['batch_seed']}")
    n_graphs = steps = 0
    for i, s in enumerate(plan["specs"]):
        n = _example(s).graph.number_of_nodes()
        n_graphs += s["count"]
        steps += s["count"] * n_r * reps * n**2
        kw = ", ".join(f"{k}={v!r}" for k, v in s["kwargs"].items())
        skip = "  skip_null_steps" if s["skip_null_steps"] else ""
        print(f"  [{i:>3}] {s['factory']}({kw}) x{s['count']}  N={n}{skip}")
    runs = n_graphs * n_r * reps
    hours = steps / STEPS_PER_SECOND / 3600
    print(f"  {n_graphs:,} graphs x {n_r} r values x {reps:,} repeats = {runs:,} runs")
    print(f"  rough cost (N^2 steps per run, can be off 10x): {hours:,.1f} CPU hours,"
          f" ~{hours / lsf['n_workers']:,.2f} h on {lsf['n_workers']} workers ({lsf['queue']})")
    return {"n_graphs": n_graphs, "runs": runs, "cpu_hours": hours}


def _example(s):
    fn = _factory(s["factory"])
    kwargs = dict(s["kwargs"])
    if _seedable(fn) and "seed" not in kwargs:
        kwargs["seed"] = 0
    return fn(**kwargs)


def build_zoo(plan):
    """The plan's graphs as records {graph, spec_idx, kwargs, hash}, deduplicated by hash.

    Spec i draws its graph seeds from the i-th child of zoo_seed, so appending a spec
    never changes earlier graphs. A seedable spec redraws on a hash collision; a
    deterministic one that collides is a duplicate line and is dropped.

    Graphs made as a series (a spec with count > 1) get an index in their name,
    random_n31_e40_i0, random_n31_e40_i1, ..., numbered across the whole zoo so two
    specs with the same parameters do not repeat names.
    """
    rngs = np.random.default_rng(plan["zoo_seed"]).spawn(len(plan["specs"]))
    zoo, seen, series_index = [], set(), Counter()
    for spec_idx, (s, rng) in enumerate(zip(plan["specs"], rngs)):
        fn = _factory(s["factory"])
        seedable = _seedable(fn) and "seed" not in s["kwargs"]
        for _ in range(s["count"]):
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
    reps = sim["n_repeats"]
    skip = {i: s["skip_null_steps"] for i, s in enumerate(plan["specs"])}
    graphs = graphs.sort_values(["n_nodes", "graph_idx"], ascending=[False, True])
    total = (graphs["n_nodes"].astype(float) ** 2).sum() * reps * len(sim["r_values"])
    target = min(sim["chunk_target_steps"], total / (4 * plan["lsf"]["n_workers"]))

    rows = []
    for g in graphs.itertuples():
        runs = int(np.clip(target // g.n_nodes**2, 1, reps))
        k = -(-reps // runs)  # chunks per cell, sizes differing by at most one run
        base, extra = divmod(reps, k)
        sizes = [base + 1] * extra + [base] * (k - extra)
        for r in sim["r_values"]:
            for n_runs in sizes:
                rows.append((g.graph_idx, g.hash, r, n_runs, skip[g.spec_idx]))
    chunks = pd.DataFrame(rows, columns=["graph_idx", "hash", "r", "n_runs", "skip_null_steps"])
    seeds = np.random.SeedSequence(plan["batch_seed"]).spawn(len(chunks))
    chunks["seed"] = np.array([s.generate_state(1, np.uint64)[0] for s in seeds], dtype=np.uint64)
    chunks.insert(0, "chunk_id", np.arange(len(chunks)))
    return chunks

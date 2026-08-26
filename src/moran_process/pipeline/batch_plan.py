"""Batch plans: a declarative, replayable recipe for a simulation batch.

A plan is a small JSON document listing *how to build* a zoo (factory names,
kwargs, counts, one root seed) plus the simulation knobs. It never contains a
graph. That buys three things:

1. The notebook that writes a plan holds no NetworkX objects, so designing a
   50k-graph batch costs a few KB of kernel memory instead of gigabytes.
2. A batch is replayable. ``build_zoo(load_plan(p))`` twice yields the same
   wl_hashes, so the plan stored beside the results is the recipe that made
   them, not a description of them.
3. ``plan_size`` answers "how many graphs, how many simulations" without
   building anything, so a batch can be costed before it is submitted.

The plan replaces the older ``zoo_config`` dict that ``main.py`` assembled by
hand from ``PopulationGraph.params``. That could not work: no factory records
``directed`` in ``params``, and two of them use key names that differ from
their own kwargs, so it described a batch without being able to rebuild one.

Layering: plans sit *above* ProcessLab. ``submit_from_plan`` builds the zoo,
serializes it, and hands the path to ``ProcessLab.submit_jobs`` unchanged.
Nothing downstream of submission knows plans exist.
"""

import argparse
import inspect
import json
import logging
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np

from moran_process.core.population_graph import PopulationGraph

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1

# How many times to redraw a seed when a random graph collides with one already
# in the zoo. Collisions are near-impossible at the sizes we run; this is a
# guard against a spec whose (n_nodes, n_edges) admits very few distinct graphs
# (e.g. a 5-node tree), where asking for 500 of them cannot be satisfied.
MAX_SEED_REDRAWS = 100


# --- Factory resolution -------------------------------------------------


def _factory_names():
    """Public PopulationGraph classmethods that build a graph from scratch."""
    names = []
    for name, member in vars(PopulationGraph).items():
        if name.startswith("_") or not isinstance(member, classmethod):
            continue
        if name in ("batch_register",):  # a classmethod, but not a factory
            continue
        names.append(name)
    return sorted(names)


def _resolve_factory(name):
    if name not in _factory_names():
        raise ValueError(
            f"Unknown graph factory {name!r}. Available: {', '.join(_factory_names())}"
        )
    return getattr(PopulationGraph, name)


def _takes_seed(factory):
    return "seed" in inspect.signature(factory).parameters


# --- Building a plan ----------------------------------------------------


def spec(factory, count=1, **kwargs):
    """One line of a plan: build ``count`` graphs with ``factory(**kwargs)``.

    ``factory`` is validated here rather than at build time, so a typo fails in
    the notebook cell that wrote it instead of inside an LSF job three hours
    later. kwargs are NOT validated against the signature: a factory may grow a
    parameter, and a plan written before that is still valid.
    """
    fn = _resolve_factory(factory)
    if count < 1:
        raise ValueError(f"count must be >= 1, got {count}")
    if count > 1 and not _takes_seed(fn) and "seed" not in kwargs:
        raise ValueError(
            f"{factory} is deterministic (no seed parameter), so count={count} "
            f"would produce {count} identical graphs. Use count=1."
        )
    return {"factory": factory, "count": int(count), "kwargs": dict(kwargs)}


def make_plan(
    batch_name,
    specs,
    r_values,
    n_repeats,
    n_jobs,
    zoo_seed=42,
    batch_seed=42,
    engine="cpp",
    queue="gsla-cpu",
    memory="1GB",
    max_steps=1_000_000,
    post_batch="all",
    description="",
    notes="",
):
    """Assemble a plan dict. See module docstring for what a plan is.

    zoo_seed  -- drives graph *topology* generation (which random graphs exist).
    batch_seed -- drives the *simulation* RNG (which trajectories are drawn).
    They are independent on purpose: re-running the same zoo with a different
    batch_seed is a fresh Monte Carlo sample of the same population structures.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "batch_name": batch_name,
        "description": description,
        "notes": notes,
        "zoo_seed": zoo_seed,
        "specs": list(specs),
        "sim": {
            "r_values": list(r_values),
            "n_repeats": int(n_repeats),
            "n_jobs": int(n_jobs),
            "batch_seed": batch_seed,
            "engine": engine,
            "queue": queue,
            "memory": memory,
            "max_steps": int(max_steps),
            "post_batch": post_batch,
        },
    }


def save_plan(plan, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2) + "\n")
    print(f"Plan saved to {path}")
    return path


def load_plan(path):
    plan = json.loads(Path(path).read_text())
    version = plan.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"Plan schema_version {version!r} != {SCHEMA_VERSION}; this plan was "
            f"written by a different version of batch_plan.py."
        )
    for s in plan["specs"]:
        _resolve_factory(s["factory"])
    return plan


# --- Costing ------------------------------------------------------------


def plan_size(plan):
    """Graph and simulation counts, without building a single graph.

    n_graphs is the *planned* count. build_zoo may return fewer if a spec asks
    for more distinct graphs than its (n_nodes, n_edges) admits; it says so
    when that happens.
    """
    n_graphs = sum(s["count"] for s in plan["specs"])
    sim = plan["sim"]
    total = n_graphs * len(sim["r_values"]) * sim["n_repeats"]
    return {
        "n_graphs": n_graphs,
        "n_r_values": len(sim["r_values"]),
        "n_repeats": sim["n_repeats"],
        "total_simulations": total,
        "n_jobs": sim["n_jobs"],
        "sims_per_job": total / sim["n_jobs"] if sim["n_jobs"] else 0,
    }


def describe(plan):
    """Human-readable summary. Prints; returns the size dict."""
    size = plan_size(plan)
    print(f"Batch plan: {plan['batch_name']}")
    if plan.get("description"):
        print(f"  {plan['description']}")
    print(f"  zoo_seed={plan['zoo_seed']}  batch_seed={plan['sim']['batch_seed']}")
    print(f"  {len(plan['specs'])} specs -> {size['n_graphs']:,} graphs")
    for i, s in enumerate(plan["specs"]):
        kw = ", ".join(f"{k}={v!r}" for k, v in s["kwargs"].items())
        mult = f" x{s['count']}" if s["count"] > 1 else ""
        print(f"    [{i:>3}] {s['factory']}({kw}){mult}")
    print(
        f"  r_values={plan['sim']['r_values']}  n_repeats={size['n_repeats']:,}"
        f"  -> {size['total_simulations']:,} simulations"
    )
    print(
        f"  {size['n_jobs']} jobs  ->  {size['sims_per_job']:,.0f} simulations/job"
        f"  (engine={plan['sim']['engine']}, queue={plan['sim']['queue']})"
    )
    return size


# --- Building the zoo ---------------------------------------------------


def build_zoo(plan, progress_every=500):
    """Materialize the plan into a list[PopulationGraph].

    Seeding: one root generator from ``zoo_seed``, spawned once per spec, so
    spec i's stream is a function of (zoo_seed, i) alone. Appending a spec
    leaves every earlier spec's graphs bit-identical; inserting one in the
    middle shifts everything after it.

    Dedup is by wl_hash across the whole zoo. A seedable spec redraws on a
    collision (so ``count`` stays honest); a deterministic spec that collides
    is a duplicate line in the plan, and is dropped with a warning.
    """
    specs = plan["specs"]
    root = np.random.default_rng(plan["zoo_seed"])
    if plan["zoo_seed"] is None:
        log.warning("zoo_seed is None: this zoo is NOT reproducible.")
    children = root.spawn(len(specs))

    zoo = []
    seen = set()
    n_dropped = 0
    t0 = time.perf_counter()

    for spec_idx, (s, rng) in enumerate(zip(specs, children)):
        factory = _resolve_factory(s["factory"])
        kwargs = dict(s["kwargs"])
        seedable = _takes_seed(factory) and "seed" not in kwargs

        for _ in range(s["count"]):
            graph = None
            for _attempt in range(MAX_SEED_REDRAWS if seedable else 1):
                if seedable:
                    kwargs["seed"] = int(rng.integers(0, 2**32))
                candidate = factory(**kwargs)
                if candidate.wl_hash not in seen:
                    graph = candidate
                    break
            if graph is None:
                n_dropped += 1
                log.warning(
                    "spec[%d] %s: duplicate wl_hash after %s, dropped.",
                    spec_idx,
                    s["factory"],
                    f"{MAX_SEED_REDRAWS} redraws" if seedable else "1 attempt",
                )
                continue
            seen.add(graph.wl_hash)
            zoo.append(graph)

            if progress_every and len(zoo) % progress_every == 0:
                log.info("  built %d graphs (%.1fs)", len(zoo), time.perf_counter() - t0)

    planned = plan_size(plan)["n_graphs"]
    log.info(
        "Zoo built: %d graphs in %.1fs (planned %d, dropped %d duplicates).",
        len(zoo),
        time.perf_counter() - t0,
        planned,
        n_dropped,
    )
    if n_dropped:
        log.warning(
            "%d graphs could not be made distinct. The batch will run %d graphs, "
            "not the %d the plan asked for.",
            n_dropped,
            len(zoo),
            planned,
        )
    return zoo


def verify_reproducible(plan):
    """Build the zoo twice and assert the wl_hashes match.

    Cheap on a small plan, expensive on a large one. This is what turns
    "replayable" into a checked fact; call it once on a new plan shape, not on
    every submission.
    """
    a = [g.wl_hash for g in build_zoo(plan, progress_every=0)]
    b = [g.wl_hash for g in build_zoo(plan, progress_every=0)]
    if a != b:
        raise AssertionError("build_zoo is not deterministic for this plan.")
    print(f"Reproducible: {len(a)} graphs, identical wl_hashes across two builds.")
    return True


# --- Submission ---------------------------------------------------------


def project_root():
    for p in [Path.cwd(), *Path.cwd().parents]:
        if (p / "pyproject.toml").exists():
            return p
    raise RuntimeError("Could not find project root (no pyproject.toml above cwd).")


def submit_from_plan(plan, batch_dir=None, dry_run=False):
    """Build the plan's zoo, serialize it, and submit the batch.

    Deliberately NOT called from a notebook kernel: this is the step that holds
    the whole zoo in RAM. Run it from an ``inode`` session or a bsub'd job.
    See the module __main__ below.
    """
    # Imported here, not at module scope: writing a plan should not drag in
    # ProcessLab (and pandas, and LSF assumptions) in a notebook that only
    # writes JSON.
    from moran_process.pipeline.process_lab import ProcessLab

    if isinstance(plan, (str, Path)):
        plan = load_plan(plan)

    describe(plan)

    batch_dir = Path(batch_dir or (project_root() / "simulation_data" / plan["batch_name"]))

    if dry_run:
        print(f"\n[dry-run] Would build into {batch_dir}. Nothing created, nothing submitted.")
        return None

    batch_dir.mkdir(parents=True, exist_ok=True)

    # The plan travels with the data it produced.
    save_plan(plan, batch_dir / "batch_plan.json")

    zoo = build_zoo(plan)

    tmp_dir = batch_dir / "tmp"
    tmp_dir.mkdir(exist_ok=True)
    zoo_path = tmp_dir / "graph_zoo.joblib"
    joblib.dump(zoo, zoo_path)
    log.info("Serialized %d graphs to %s", len(zoo), zoo_path)

    # Cheap here (the zoo is already in hand); this is what cell 20 of the old
    # notebook reloaded the entire zoo from disk to compute.
    graph_types = sorted({g.category for g in zoo})
    node_sizes = sorted({g.n_nodes for g in zoo})
    n_graphs = len(zoo)
    del zoo

    sim = plan["sim"]
    return ProcessLab().submit_jobs(
        zoo_path=str(zoo_path),
        n_graphs=n_graphs,
        r_values=sim["r_values"],
        batch_name=plan["batch_name"],
        batch_dir=str(batch_dir),
        n_repeats=sim["n_repeats"],
        n_requested_jobs=sim["n_jobs"],
        queue=sim["queue"],
        memory=sim["memory"],
        graph_types=graph_types,
        node_sizes=node_sizes,
        description=plan.get("description", ""),
        notes=plan.get("notes", ""),
        batch_seed=sim["batch_seed"],
        engine=sim["engine"],
        max_steps=sim["max_steps"],
        zoo_config=plan,
        post_batch=sim["post_batch"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--plan", required=True, help="Path to a batch plan JSON file")
    parser.add_argument("--batch-dir", default=None, help="Override the batch directory")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan and its cost, build nothing, submit nothing",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    t0 = time.perf_counter()
    submit_from_plan(args.plan, batch_dir=args.batch_dir, dry_run=args.dry_run)
    log.info("Done in %.1fs", time.perf_counter() - t0)


if __name__ == "__main__":
    main()

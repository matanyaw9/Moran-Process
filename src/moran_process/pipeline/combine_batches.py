"""
Combine several finished batches into one analysable batch directory.

Motivation: the GA in ``notebooks/extreme_graphs.ipynb`` produces topologies that were
designed *against* a source batch's ML models, so the two only mean something side by
side. Rather than re-simulating 12k graphs, this synthesises a batch directory out of
the parents' finished outputs.

A "batch" here is not a database record, it is a directory with a known layout:

    <batch>/batch_info.json          metadata + provenance
    <batch>/graph_props.csv          one row per graph (structural properties)
    <batch>/graph_statistics.csv     one row per (graph, r) (fixation statistics)
    <batch>/tmp/results/*.parquet    one row per simulated run

so a combined batch is built by unioning the first three and symlinking the fourth.
Every consumer (``experiment_analysis.ipynb``, ``build_graph_statistics``,
``resolve_results_source``) then works on it unchanged.

Why concatenating graph_statistics.csv is EXACT rather than an approximation: every
statistic in it decomposes into additive per-shard partials (see ``io._aggregate_chunked``),
and the rollup groups by (wl_hash, r). So as long as no (wl_hash, r) key appears in two
parents, the union of the parents' rollups is bit-identical to re-aggregating the union of
their raw rows. That precondition is checked, not assumed.

CLI:
    python -m moran_process.pipeline.combine_batches \
        --sources simulation_data/<a> simulation_data/<b> \
        --dest simulation_data/<combined> \
        --description "..."
"""

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from ..analysis.analysis_utils.provenance import capture_provenance, load_batch_info

__all__ = ["combine_batches"]

PROPS_FILE = "graph_props.csv"
STATS_FILE = "graph_statistics.csv"
SHARD_GLOB = "raw_results_job_*.parquet"


def _read_parents(source_dirs):
    """Load each parent's props + stats, failing loudly on a parent that isn't finished."""
    parents = []
    for d in source_dirs:
        d = Path(d)
        props_path, stats_path = d / PROPS_FILE, d / STATS_FILE
        for p in (props_path, stats_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"{d.name} is not a finished batch: missing {p.name}. "
                    f"Run moran_process.pipeline.aggregate_batch on it first."
                )
        parents.append(
            {
                "dir": d,
                "info": load_batch_info(d),
                "props": pd.read_csv(props_path),
                "stats": pd.read_csv(stats_path),
            }
        )
    return parents


def _concat_props(parents):
    """Union the parents' graph_props, keyed by wl_hash.

    Parents may carry different columns -- the respiratory zoo has branching/depth/n_rods,
    the GA zoo does not -- so this is a column union with NaN fill, which is the honest
    representation: those parameters genuinely do not exist for a GA-evolved graph.

    A wl_hash present in two parents is the SAME graph (that is what the WL hash means),
    so the duplicate row is dropped rather than being an error.
    """
    props = pd.concat([p["props"] for p in parents], ignore_index=True, sort=False)
    n_before = len(props)
    props = props.drop_duplicates(subset="wl_hash", keep="first")
    if len(props) < n_before:
        print(
            f"[combine] graph_props: dropped {n_before - len(props)} duplicate wl_hash row(s) "
            f"(same graph registered in more than one parent)."
        )
    return props


def _concat_stats(parents):
    """Union the parents' graph_statistics, keyed by (wl_hash, r).

    A key appearing in two parents means the same graph was simulated at the same r in
    both, and the two rollups would have to be POOLED (weighted by n_grouped, and std
    cannot be pooled at all without the sum of squares, which the CSV does not keep).
    Rather than silently keeping one and discarding the other's simulations, this refuses.
    """
    stats = pd.concat([p["stats"] for p in parents], ignore_index=True, sort=False)
    dup = stats.duplicated(subset=["wl_hash", "r"], keep=False)
    if dup.any():
        offenders = stats.loc[dup, ["wl_hash", "r"]].drop_duplicates()
        raise ValueError(
            f"{len(offenders)} (wl_hash, r) key(s) appear in more than one parent, e.g.\n"
            f"{offenders.head().to_string(index=False)}\n"
            "Their rollups would have to be pooled, which std_steps does not permit from "
            "the CSV alone. Re-aggregate the union from the raw shards instead."
        )
    return stats


def _link_shards(parents, dest):
    """Symlink every parent's raw shards into dest/tmp/results, renumbered to stay unique.

    The shards are never copied: the source batch alone is 42GB. Renumbering is required
    because each parent numbers its own jobs 1..N, and ``io._expand_shards`` sorts shards
    by the trailing integer, so colliding names would overwrite each other.

    Note the ``job_id`` COLUMN inside the files is not rewritten, so it stays 1..N per
    parent. Nothing in the analysis groups by job_id except the speed report, where the
    consequence is only that two parents' job 7 are reported as one job.
    """
    results_dir = dest / "tmp" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    offset = 0
    for p in parents:
        shards = sorted(
            (p["dir"] / "tmp" / "results").glob(SHARD_GLOB),
            key=lambda f: int(f.stem.split("_")[-1]),
        )
        if not shards:
            print(f"[combine] {p['dir'].name}: no raw shards found, skipping link step.")
            continue
        for i, shard in enumerate(shards, start=1):
            link = results_dir / f"raw_results_job_{offset + i}.parquet"
            link.unlink(missing_ok=True)
            # Absolute target: the combined batch may later be moved, and an absolute
            # link survives that while a relative one would not.
            link.symlink_to(shard.resolve())
        print(
            f"[combine] linked {len(shards)} shard(s) from {p['dir'].name} "
            f"as job_{offset + 1}..job_{offset + len(shards)}"
        )
        offset += len(shards)


def _combined_batch_info(parents, dest, name, description, notes):
    """Build a batch_info.json describing the union, keeping each parent's own info nested.

    The zoo/simulation/hpc sections are filled with union/sum values so
    ``plot_batch_info_card`` renders a meaningful card; ``combined_from`` is the field that
    actually documents provenance, since the real submission details live in the parents.
    """

    def _bi(p, *path, default=None):
        node = p["info"]
        for key in path:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node if node is not None else default

    props = _concat_props(parents)
    r_values = sorted({r for p in parents for r in _bi(p, "simulation", "r_values", default=[])})
    engines = {_bi(p, "simulation", "engine") for p in parents}
    n_repeats = {_bi(p, "simulation", "n_repeats") for p in parents}

    info = {
        "name": name,
        "description": description,
        "notes": notes,
        "created_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "provenance": capture_provenance(),
        "combined_from": [
            {
                "name": _bi(p, "name", default=p["dir"].name),
                "dir": str(p["dir"].resolve()),
                "n_graphs": int(p["props"]["wl_hash"].nunique()),
                "n_stats_rows": int(len(p["stats"])),
                "description": _bi(p, "description", default=""),
            }
            for p in parents
        ],
        "zoo": {
            "n_graphs": int(props["wl_hash"].nunique()),
            "graph_types": sorted(props["category"].dropna().unique().tolist()),
            "node_sizes": sorted(int(n) for n in props["n_nodes"].dropna().unique()),
            "zoo_path": None,
        },
        "simulation": {
            "r_values": r_values,
            # Only meaningful when the parents agree; otherwise it varies per parent and
            # the honest answer is None rather than an arbitrary pick.
            "n_repeats": n_repeats.pop() if len(n_repeats) == 1 else None,
            "total_simulations": sum(
                int(_bi(p, "simulation", "total_simulations", default=0) or 0) for p in parents
            ),
            "batch_seed": None,
            "engine": engines.pop() if len(engines) == 1 else "mixed",
        },
        "hpc": {
            "n_requested_jobs": sum(
                int(_bi(p, "hpc", "n_requested_jobs", default=0) or 0) for p in parents
            ),
            "queue": None,
            "memory_mb": None,
            "job_array_name": None,
            "lsf_job_id": "combined (see combined_from)",
            "bsub_command": None,
        },
    }
    with open(dest / "batch_info.json", "w") as f:
        json.dump(info, f, indent=2)
    return info


def combine_batches(
    source_dirs,
    dest_dir,
    name=None,
    description="",
    notes="",
    link_raw=True,
    overwrite=False,
    submit_post_jobs=True,
    queue="short",
):
    """Synthesise a combined batch directory from finished parent batches.

    Args:
        source_dirs: iterable of parent batch directories (each needs graph_props.csv and
            graph_statistics.csv, i.e. must already have been aggregated).
        dest_dir: directory to create. Must not exist unless overwrite=True.
        name: batch name recorded in batch_info.json; defaults to the destination folder name.
        description, notes: free text carried into batch_info.json.
        link_raw: symlink the parents' raw shards into dest/tmp/results so raw-data
            consumers (the violin plot, the speed report) keep working. Costs no disk.
        overwrite: replace dest_dir if it already exists.
        submit_post_jobs: bsub the same two post-processing jobs a simulated batch gets --
            the QC report and the violin-sample cache. There is no aggregation job to wait
            for here (the rollup was inherited from the parents), so both run immediately.
            The violin cache is the reason this defaults on: it is the one step that still
            scans the raw shards, and it must not run on the login node.
        queue: LSF queue for those jobs.

    Returns:
        Path to the combined batch directory.
    """
    parents = _read_parents(source_dirs)
    dest = Path(dest_dir)

    if dest.exists():
        if not overwrite:
            raise FileExistsError(f"{dest} already exists; pass overwrite=True to replace it.")
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    props = _concat_props(parents)
    stats = _concat_stats(parents)

    props.to_csv(dest / PROPS_FILE, index=False)
    stats.to_csv(dest / STATS_FILE, index=False)
    print(
        f"[combine] {PROPS_FILE}: {len(props):,} graphs x {props.shape[1]} cols\n"
        f"[combine] {STATS_FILE}: {len(stats):,} (graph, r) rows x {stats.shape[1]} cols"
    )

    if link_raw:
        _link_shards(parents, dest)

    info = _combined_batch_info(parents, dest, name or dest.name, description, notes)
    print(f"[combine] combined batch ready: {dest}")

    if submit_post_jobs:
        # Imported here, not at module scope: process_lab pulls in the graph and worker
        # stack, and combining batches is otherwise pure pandas + symlinks that should
        # stay importable anywhere.
        from .process_lab import submit_post_batch_jobs

        submit_post_batch_jobs(
            batch_dir=str(dest),
            batch_name=info["name"],
            aggregate_job_id=None,  # nothing to wait for: the rollup came from the parents
            queue=queue,
        )

    return dest


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--sources", nargs="+", required=True, help="parent batch directories")
    parser.add_argument("--dest", required=True, help="combined batch directory to create")
    parser.add_argument("--name", default=None)
    parser.add_argument("--description", default="")
    parser.add_argument("--notes", default="")
    parser.add_argument(
        "--no-link-raw",
        action="store_true",
        help="do not symlink the parents' raw shards (aggregated data only)",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no-post-jobs",
        action="store_true",
        help="do not bsub the QC report and violin-cache jobs for the combined batch",
    )
    parser.add_argument("--queue", default="short", help="LSF queue for the post jobs")
    args = parser.parse_args()

    combine_batches(
        args.sources,
        args.dest,
        name=args.name,
        description=args.description,
        notes=args.notes,
        link_raw=not args.no_link_raw,
        overwrite=args.overwrite,
        submit_post_jobs=not args.no_post_jobs,
        queue=args.queue,
    )


if __name__ == "__main__":
    main()

"""
Results IO: locate a batch's aggregated results, stream per-job files into one,
and roll raw results up to per-(graph, r) fixation statistics.

Heavy readers (polars, pyarrow) are imported lazily inside the functions that
need them, so importing this module stays cheap.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .theory import (
    analytic_moran_fc_fixation_prob,
    analytic_moran_fc_fixation_time,
)

__all__ = [
    "RAW_RESULTS_STEM",
    "PER_JOB_RESULT_STEM",
    "ANALYTIC_REFERENCE_COLUMNS",
    "resolve_results_path",
    "resolve_results_source",
    "scan_results",
    "aggregate_results_no_load",
    "add_analytic_reference_columns",
    "build_graph_statistics",
    "load_fixation_steps_by_category",
    "build_fixation_steps_cache",
    "fixation_steps_cache_path",
]


# Aggregated per-repeat results filename. "raw_results" = the raw, one-row-per-run
# table, as opposed to the aggregated graph_statistics.csv. (Historically named
# "full_results"; existing batches were migrated on disk to this name.)
RAW_RESULTS_STEM = "raw_results"

# Per-job temp files the worker writes into tmp/results/, one per LSF array index.
PER_JOB_RESULT_STEM = "raw_results_job"


def resolve_results_path(batch_dir):
    """Return the aggregated raw-results file for a batch, or None if absent.

    Results may be Parquet or CSV depending on when the batch ran, so this picks
    the existing one (Parquet preferred).
    """
    batch_path = Path(batch_dir)
    for ext in (".parquet", ".csv"):
        candidate = batch_path / f"{RAW_RESULTS_STEM}{ext}"
        if candidate.exists():
            return candidate
    return None


def resolve_results_source(batch_dir):
    """Return the best scannable source of raw results for a batch, or None if absent.

    Prefers the per-job shards as a glob (tmp/results/raw_results_job_*.parquet) over the
    concatenated raw_results.parquet, because stock polars indexes rows with a u32 and its
    Parquet reader refuses ANY SINGLE FILE holding more than 2**32-1 rows. The 100K-reps
    batch concatenates to 7.2e9 rows, so the fused file is unreadable while the 1000
    ~7.2M-row shards it was built from are each fine. The limit is per file, and polars
    treats a glob as one logical LazyFrame, so scanning the shards gives identical results
    with no size ceiling -- and no concatenation step at all.

    Falls back to the fused file for older batches that have one but no shards.
    """
    batch_path = Path(batch_dir)
    shards_dir = batch_path / "tmp" / "results"
    for ext in (".parquet", ".csv"):
        if any(shards_dir.glob(f"{PER_JOB_RESULT_STEM}_*{ext}")):
            return str(shards_dir / f"{PER_JOB_RESULT_STEM}_*{ext}")
    return resolve_results_path(batch_dir)


def scan_results(source):
    """Lazily scan raw results from a single file OR a glob of per-job shards.

    ``source`` may be a path to one raw_results.parquet/.csv or a glob pattern such as
    ``.../raw_results_job_*.parquet`` (see resolve_results_source for why the glob is
    preferred). polars accepts globs natively, so both forms yield one LazyFrame.
    """
    import polars as pl

    s = str(source)
    return pl.scan_csv(s) if s.endswith(".csv") else pl.scan_parquet(s)


def aggregate_results_no_load(batch_dir, delete_temp=False, output_file=None):
    """Concatenate per-job result files from batch_dir/tmp/results/ without loading all rows.

    Detects the output format automatically: Parquet files (raw_results_job_*.parquet)
    take priority over CSV files (raw_results_job_*.csv). The output format matches the input.

    Args:
        batch_dir: path to the batch directory containing tmp/results/raw_results_job_*
        delete_temp: if True, removes batch_dir/tmp/ after successful aggregation
        output_file: destination path; defaults to batch_dir/raw_results.parquet (or .csv)

    Returns:
        Path to the output file, or None if no result files were found.
    """
    import pyarrow.parquet as pq

    batch_path = Path(batch_dir)
    tmp_results_path = batch_path / "tmp" / "results"

    # Detect format: Parquet takes priority over CSV
    parquet_files = sorted(
        tmp_results_path.glob(f"{PER_JOB_RESULT_STEM}_*.parquet"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    csv_files = sorted(
        tmp_results_path.glob(f"{PER_JOB_RESULT_STEM}_*.csv"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )

    # --- Parquet path ---
    if parquet_files:
        if not output_file:
            output_file = batch_path / f"{RAW_RESULTS_STEM}.parquet"
        else:
            output_file = Path(output_file)

        if output_file.exists():
            print(f"File {output_file} already exists!")
            return output_file

        print(f"Found {len(parquet_files)} Parquet files. Aggregating...")
        try:
            schema = pq.read_schema(str(parquet_files[0]))
            with pq.ParquetWriter(str(output_file), schema) as writer:
                for i, fpath in enumerate(parquet_files):
                    if i > 0 and i % 100 == 0:
                        print(f"  Processed {i}/{len(parquet_files)} files...")
                    pf = pq.ParquetFile(str(fpath))
                    for batch in pf.iter_batches():
                        writer.write_batch(batch)
            print(f"Master Parquet saved at: {output_file}")
        except Exception as e:
            print(f"Error during Parquet aggregation: {e}")
            if output_file.exists():
                output_file.unlink()
            raise

        if delete_temp:
            tmp_dir = batch_path / "tmp"
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
                print(f"Deleted temporary directory: {tmp_dir}")
        return output_file

    # --- Legacy CSV path ---
    if not csv_files:
        print(f"No result files found in {tmp_results_path}")
        return None

    if not output_file:
        output_file = batch_path / f"{RAW_RESULTS_STEM}.csv"
    else:
        output_file = Path(output_file)

    if output_file.exists():
        print(f"File {output_file} already exists!")
        return output_file

    print(f"Found {len(csv_files)} CSV files. Aggregating...")
    try:
        with open(output_file, "w", encoding="utf-8") as outfile:
            for i, fpath in enumerate(csv_files):
                if i > 0 and i % 100 == 0:
                    print(f"  Processed {i}/{len(csv_files)} files...")
                with open(fpath, "r", encoding="utf-8") as infile:
                    if i == 0:
                        shutil.copyfileobj(infile, outfile)
                    else:
                        next(infile)
                        shutil.copyfileobj(infile, outfile)
        print(f"Master CSV saved at: {output_file}")
    except Exception as e:
        print(f"Error during CSV aggregation: {e}")
        if output_file.exists():
            output_file.unlink()
        raise

    if delete_temp:
        tmp_dir = batch_path / "tmp"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
            print(f"Deleted temporary directory: {tmp_dir}")
    return output_file


ANALYTIC_REFERENCE_COLUMNS = [
    "fc_prob_fixation",
    "fc_fixation_time",
    "delta_prob_fixation",
    "log_ratio_mean_steps",
]


def add_analytic_reference_columns(df, n_col="n_nodes", r_col="r"):
    """Add the complete-graph baseline for each (n, r), plus the residuals against it.

    The baseline answers "what would a fully connected graph of this size have done at
    this r". The residual therefore isolates what the TOPOLOGY contributes once size
    and fitness are accounted for: positive means amplifier, negative means suppressor.

    The two residuals are shaped differently on purpose:

    - ``delta_prob_fixation`` is a plain difference. rho is a bounded probability, so a
      difference is both interpretable and well conditioned.
    - ``log_ratio_mean_steps`` is a difference of logs, i.e. log(T_graph / T_complete).
      Fixation time spans about four orders of magnitude across the zoo, so a raw
      difference would be dominated by the slow, sparse, large-n graphs. This is the
      same reason ``mean_steps`` was already modelled in log space; subtracting the log
      baseline yields a logged target and a residual in one step, so it needs no
      further logging downstream.

    Returns a copy, so it is safe to call twice.
    """
    out = df.copy()
    n, r = out[n_col].to_numpy(), out[r_col].to_numpy()

    out["fc_prob_fixation"] = analytic_moran_fc_fixation_prob(n, r)
    out["fc_fixation_time"] = analytic_moran_fc_fixation_time(n, r)
    out["delta_prob_fixation"] = out["prob_fixation"] - out["fc_prob_fixation"]
    out["log_ratio_mean_steps"] = np.log(out["mean_steps"]) - np.log(
        out["fc_fixation_time"]
    )
    return out


# wl_hash is the graph identity (graph_name is just a per-graph label carried in
# graph_props). Grouping by wl_hash alone means isomorphic graphs that were simulated
# under different names -- e.g. a GA rediscovering the same topology across generations --
# pool their runs into one row instead of splitting, and the canonical graph_name is
# reattached from graph_props at merge time.
GROUP_KEYS = ["wl_hash", "r"]

# Partial sums carried from each shard into the reduce step. Every one is additive, which
# is what makes the map-reduce exact (see _aggregate_chunked).
_PARTIAL_COLS = ["_fix_sum", "_n_total", "_steps_sum", "_steps_n", "_steps_sq_sum"]


def _expand_shards(results_source):
    """Return the list of files a results source refers to (glob -> sorted file list)."""
    s = str(results_source)
    if "*" not in s:
        return [Path(s)]
    p = Path(s)
    files = sorted(
        p.parent.glob(p.name),
        key=lambda f: int(f.stem.split("_")[-1]) if f.stem.split("_")[-1].isdigit() else 0,
    )
    return files


def _steps_success_expr():
    """steps, but only on runs that fixated (non-fixation rows contribute nothing)."""
    import polars as pl

    return (
        pl.when(pl.col("fixation"))
        .then(pl.col("steps"))
        .otherwise(None)
        .alias("steps_success")
    )


def _aggregate_chunked(files, progress_every=100):
    """Roll raw results up to one row per (graph, r), one shard at a time.

    Polars' group_by over a 1000-file / 7.2e9-row scan does NOT bound its memory: given a
    16GB limit it dies at 16GB, given 64GB it dies at 64GB, i.e. it accumulates with rows
    read rather than with group count. No amount of RAM fixes that. So instead of asking
    polars to hold the whole batch, we aggregate each ~7.2M-row shard on its own (peak
    memory = one shard) and combine the partials afterwards.

    This is exact, not an approximation, because every statistic we keep decomposes into
    additive partials:

        prob_fixation = sum(fixation) / count
        mean_steps    = sum(steps) / count(steps)
        std_steps     = sqrt( (sum(steps^2) - sum(steps)^2/n) / (n-1) )

    Combining is therefore just summing the partials per group -- which also handles a
    config split across several workers, as _create_task_list does when a configuration
    does not divide evenly into a worker's share.

    std_steps uses summed squares, which can cancel when std << mean. Fixation times have
    CV ~ 1 (std comparable to mean), so the subtraction keeps ~15 significant digits here;
    the variance is clipped at 0 to absorb the last-bit rounding.
    """
    import polars as pl

    partials = []
    for i, f in enumerate(files):
        if i > 0 and i % progress_every == 0:
            print(f"  aggregated {i}/{len(files)} shards...")
        partials.append(
            scan_results(f)
            .with_columns(_steps_success_expr())
            .group_by(GROUP_KEYS)
            .agg(
                [
                    pl.col("fixation").sum().cast(pl.Int64).alias("_fix_sum"),
                    pl.len().cast(pl.Int64).alias("_n_total"),
                    pl.col("steps_success").cast(pl.Float64).sum().alias("_steps_sum"),
                    pl.col("steps_success").count().cast(pl.Int64).alias("_steps_n"),
                    (pl.col("steps_success").cast(pl.Float64) ** 2)
                    .sum()
                    .alias("_steps_sq_sum"),
                ]
            )
            .collect()
        )

    # ~73 rows per shard, so the reduce input is tiny (~73k rows for 1000 shards).
    combined = (
        pl.concat(partials)
        .group_by(GROUP_KEYS)
        .agg([pl.col(c).sum().alias(c) for c in _PARTIAL_COLS])
    )

    n, s, sq = pl.col("_steps_n"), pl.col("_steps_sum"), pl.col("_steps_sq_sum")
    variance = ((sq - s**2 / n) / (n - 1)).clip(lower_bound=0)

    return (
        combined.with_columns(
            [
                (pl.col("_fix_sum") / pl.col("_n_total")).alias("prob_fixation"),
                # n == 0 -> no run fixated, so mean/std are undefined rather than 0.
                pl.when(n > 0).then(s / n).otherwise(None).alias("mean_steps"),
                pl.when(n > 1).then(variance.sqrt()).otherwise(None).alias("std_steps"),
                pl.col("_n_total").alias("n_grouped"),
            ]
        )
        .drop(_PARTIAL_COLS)
        .to_pandas()
    )


def build_graph_statistics(
    results_path,
    df_graphs,
    graph_statistics_path,
    category_filter=None,
    r_filter=None,
    include_order_stats=False,
):
    """Aggregate raw simulation results to one row per (graph, r) with fixation statistics.

    If graph_statistics_path already exists, loads it directly. Otherwise streams results
    from results_path (Parquet or CSV), merges with df_graphs, sorts, and saves.

    Filtering is applied as a view *after* the full table is built/loaded, so the cached
    graph_statistics.csv always holds every category and r value; only the returned frame
    is narrowed.

    Two classes of statistic are computed, and they pick different execution paths:

    - Moments (always) -- prob_fixation, mean_steps, std_steps, n_grouped. Each decomposes
      into additive per-shard partials, so these are computed shard-by-shard and combined
      (see _aggregate_chunked). Peak memory is one shard, whatever the batch size.
    - Order statistics (only if ``include_order_stats``) -- median_steps, q25_steps,
      q75_steps, iqr_steps. An exact quantile needs every value of a group held at once, so
      it does NOT decompose and forces a single group_by over the whole batch. Polars does
      not bound that (it died at 16GB given 16GB, and at 64GB given 64GB, on the 7.2e9-row
      batch), so this path is only viable for small batches. These columns are display-only
      in the current notebooks (no figure or ML feature reads them), which is why they
      default off. ``iqr_steps`` is just q75 - q25, so it rides along for free.

    NOTE: this flag only takes effect when graph_statistics.csv does NOT yet exist -- an
    existing file is loaded verbatim. To switch the set of columns, delete the cached CSV
    and rebuild.

    Args:
        results_path: raw results to scan -- a single raw_results.parquet/.csv, or a glob
            of per-job shards (see resolve_results_source, which prefers the glob and
            explains why the fused file breaks on large batches).
        df_graphs: DataFrame with graph structural properties (must have 'wl_hash', 'graph_name')
        graph_statistics_path: path where graph_statistics.csv is saved / loaded from
        category_filter: keep only these categories. A single value or a list/tuple/set;
            None keeps all categories.
        r_filter: keep only these selection coefficients. A single value or a
            list/tuple/set; None keeps all r values.
        include_order_stats: if True, also compute median/quartile/iqr of steps (slow).

    Returns:
        analysis_df: aggregated DataFrame ready for plotting
    """
    import polars as pl

    graph_statistics_path = Path(graph_statistics_path)

    if graph_statistics_path.exists():
        print(
            f"Aggregated statistics already exist -- loading {graph_statistics_path}..."
        )
        analysis_df = pd.read_csv(graph_statistics_path)
    else:
        files = _expand_shards(results_path)

        if include_order_stats:
            # Quantiles are NOT decomposable: an exact median needs every value of a group
            # held at once, so it cannot be combined from per-shard partials. This path
            # therefore does the whole batch in one group_by, which polars does not bound
            # (see _aggregate_chunked) -- fine for small batches, fatal for large ones.
            print(
                f"Aggregating {len(files)} file(s) in ONE pass (order stats requested)..."
            )
            agg_results_df = (
                scan_results(results_path)
                .with_columns(_steps_success_expr())
                .group_by(GROUP_KEYS)
                .agg(
                    [
                        pl.col("fixation").mean().alias("prob_fixation"),
                        pl.col("steps_success").mean().alias("mean_steps"),
                        pl.col("steps_success").std().alias("std_steps"),
                        pl.col("fixation").count().alias("n_grouped"),
                        pl.col("steps_success").median().alias("median_steps"),
                        pl.col("steps_success").quantile(0.25).alias("q25_steps"),
                        pl.col("steps_success").quantile(0.75).alias("q75_steps"),
                        (
                            pl.col("steps_success").quantile(0.75)
                            - pl.col("steps_success").quantile(0.25)
                        ).alias("iqr_steps"),
                    ]
                )
                .collect(engine="streaming")
                .to_pandas()
            )
        else:
            print(f"Aggregating {len(files)} shard(s) chunked (bounded memory)...")
            agg_results_df = _aggregate_chunked(files)

        print("Shape before merging: ", agg_results_df.shape)

        analysis_df = pd.merge(
            agg_results_df,
            df_graphs,
            on="wl_hash",
            how="left",
            suffixes=("", "_db"),
        )
        # graph_props is deduped to one row per wl_hash, so an unmatched hash here means a
        # result graph was never registered -- its n_nodes stays NaN and later crashes the
        # analytic baseline. Surface it plainly instead of failing 200 lines downstream.
        n_unmatched = analysis_df["n_nodes"].isna().sum()
        if n_unmatched:
            missing = analysis_df.loc[analysis_df["n_nodes"].isna(), "wl_hash"].unique()
            print(
                f"WARNING: {n_unmatched} result row(s) have no graph_props match "
                f"({len(missing)} wl_hash(es), e.g. {missing[:3].tolist()})."
            )
        analysis_df["z_order"] = (analysis_df["category"] != "Random").astype(int)
        analysis_df = analysis_df.sort_values("z_order").drop(columns="z_order")
        analysis_df.to_csv(graph_statistics_path, index=False)

    # Derived from n_nodes and r alone, so they are recomputed on both the cached and
    # the freshly-built path rather than being persisted. That keeps the on-disk
    # graph_statistics.csv schema unchanged, so existing batches need no migration.
    analysis_df = add_analytic_reference_columns(analysis_df)

    print("Shape after merging: ", analysis_df.shape)

    def _as_list(val):
        return list(val) if isinstance(val, (list, tuple, set)) else [val]

    if category_filter is not None:
        cats = _as_list(category_filter)
        analysis_df = analysis_df[analysis_df["category"].isin(cats)].copy()
        print(
            f"Filtered to categories {sorted(map(str, cats))}: {len(analysis_df):,} rows"
        )

    if r_filter is not None:
        rs = _as_list(r_filter)
        analysis_df = analysis_df[analysis_df["r"].isin(rs)].copy()
        print(f"Filtered to r in {sorted(rs)}: {len(analysis_df):,} rows")

    print(f"Graph statistics columns: {list(analysis_df.columns)}")
    return analysis_df


# --------------------------------------------------------------------------------------
# Fixation-steps sample: the one figure input that needs the RAW rows
# --------------------------------------------------------------------------------------
# graph_statistics.csv answers everything the analysis asks except the *shape* of the
# steps-to-fixation distribution, because a KDE and a rank test both need individual runs,
# not moments. So the violin and the p-value matrix are the only consumers left scanning
# the full 7.2e9-row batch. What they actually consume, though, is tiny: fixation rows
# only, subsampled to a cap per category. Caching that sample turns a multi-minute scan
# into a file read while staying faithful to the picture, and it is deliberately the
# *data* that is cached rather than the rendered figure, so the notebook can still
# restyle, reorder and recolor interactively.

CACHE_DIR_NAME = "cache"
FIXATION_STEPS_CACHE_STEM = "fixation_steps"


def fixation_steps_cache_path(batch_dir, r, max_points_per_category=50_000):
    """Path of the cached fixation-steps sample for one (r, subsample cap).

    Both the r value and the cap are in the filename. The cap has to be part of the key
    because it changes the contents (a sample capped at 50k is not the head of one capped
    at 200k), and putting it in the *name* rather than only in the sidecar means caches for
    different caps coexist: one exploratory call with a small cap can no longer clobber the
    expensive default-cap cache and silently send the next figure back to a full scan.

    The '.' in an r value is replaced ('1.1' -> '1p1') to keep the stem free of the
    extension separator, so the parquet and its .json sidecar are unambiguous.
    """
    r_tag = str(r).replace(".", "p")
    n_tag = "all" if max_points_per_category is None else str(max_points_per_category)
    return (
        Path(batch_dir)
        / CACHE_DIR_NAME
        / f"{FIXATION_STEPS_CACHE_STEM}_r{r_tag}_n{n_tag}.parquet"
    )


def _fixation_steps_cache_is_valid(batch_dir, r, max_points_per_category):
    """True if a cache for this (r, cap) exists, without paying to read the parquet.

    Both files must be there: the parquet alone is useless, since the per-category counts
    the violin annotates with live only in the sidecar.
    """
    path = fixation_steps_cache_path(batch_dir, r, max_points_per_category)
    return path.exists() and path.with_suffix(".json").exists()


def _read_fixation_steps_cache(batch_dir, r, max_points_per_category):
    """Return the cached loader tuple for this (r, cap), or None on a miss.

    A miss is not an error: the caller silently falls back to the full scan.
    """
    if not _fixation_steps_cache_is_valid(batch_dir, r, max_points_per_category):
        return None

    path = fixation_steps_cache_path(batch_dir, r, max_points_per_category)
    with open(path.with_suffix(".json")) as f:
        meta = json.load(f)

    print(f"Using cached fixation-steps sample: {path}")
    return (
        pd.read_parquet(path),
        meta["fixation_counts"],
        meta["total_counts"],
        meta["r"],
        meta["r_suffix"],
        meta["subsampled"],
    )


def _write_fixation_steps_cache(batch_dir, loaded, max_points_per_category):
    """Persist a loader result. The counts live in a JSON sidecar, not in the parquet.

    fixation_counts/total_counts are per-category scalars counted BEFORE subsampling (the
    violin's rho annotation depends on that), so they cannot be recovered from the sampled
    rows and cannot be columns of them either.
    """
    merged_raw, fixation_counts, total_counts, r, r_suffix, subsampled = loaded
    path = fixation_steps_cache_path(batch_dir, r, max_points_per_category)
    path.parent.mkdir(parents=True, exist_ok=True)
    merged_raw.to_parquet(path, index=False)
    with open(path.with_suffix(".json"), "w") as f:
        json.dump(
            {
                "r": r,
                "r_suffix": r_suffix,
                "subsampled": subsampled,
                "max_points_per_category": max_points_per_category,
                # JSON keys must be strings; None is a real category here (a graph with no
                # graph_props match), so it is preserved as the string "None" rather than
                # dropped, which would silently hide the unmatched rows.
                "fixation_counts": {str(k): int(v) for k, v in fixation_counts.items()},
                "total_counts": {str(k): int(v) for k, v in total_counts.items()},
            },
            f,
            indent=2,
        )
    print(f"Cached fixation-steps sample ({len(merged_raw):,} rows): {path}")
    return path


def load_fixation_steps_by_category(
    results_path,
    df_graphs,
    r=None,
    max_points_per_category=50_000,
    cache_dir=None,
):
    """Load fixation 'steps' joined to graph 'category' for a single r value.

    Shared loader for ``plot_steps_violin`` and ``plot_steps_pvalue_matrix`` so the
    two figures are always built from exactly the same rows (same r resolution, same
    fixation filter, same subsample). Returns:

    - a tidy pandas DataFrame with columns ['category', 'steps'] (subsampled);
    - ``fixation_counts``: true per-category fixation counts, read *before*
      subsampling so callers can annotate how much data backs each category;
    - ``total_counts``: per-category total run counts (fixation + non-fixation),
      counted before the fixation filter, so callers can report rho = fix / total;
    - the resolved ``r`` and an ``r_suffix`` label for titles;
    - ``subsampled``: whether the cap actually trimmed any category.

    See ``plot_steps_violin`` for why only fixation rows are materialised and why
    subsampling to ``max_points_per_category`` is faithful to the full distribution.

    Args:
        cache_dir: a batch directory. If given and it holds a cached sample for this
            (r, max_points_per_category), the cache is read and results_path is never
            scanned; on a miss the sample is computed and then written there. Requires an
            explicit ``r``, since resolving r=None needs the very scan the cache avoids.
            Build the cache ahead of time with ``pipeline.cache_violin_data``.
    """
    import polars as pl

    if cache_dir is not None and r is not None:
        cached = _read_fixation_steps_cache(cache_dir, r, max_points_per_category)
        if cached is not None:
            return cached

    _scanner = scan_results(results_path)
    _has_r = "r" in _scanner.collect_schema().names()

    lf = _scanner.select(["wl_hash", "steps", "fixation"] + (["r"] if _has_r else []))

    # Pooling several r values would silently overlay distributions, so resolve to a
    # single r before collecting.
    r_suffix = ""
    if _has_r:
        r_available = sorted(
            lf.select(pl.col("r")).unique().collect().to_series().to_list()
        )
        if r is None:
            if len(r_available) == 1:
                r = r_available[0]
            else:
                raise ValueError(
                    f"results contain multiple r values {r_available}; pass r=<value> "
                    f"(one r at a time)"
                )
        elif r not in r_available:
            raise ValueError(f"r={r} not found in results; available: {r_available}")
        lf = lf.filter(pl.col("r") == r)
        r_suffix = f"  (r={r})"

    # Attach category to every run (lazy). Totals per category must be counted before
    # the fixation filter so we can report rho = fixations / total runs, hence the join
    # happens here rather than after filtering.
    lf = lf.join(
        pl.from_pandas(df_graphs[["wl_hash", "category"]]).lazy(),
        on="wl_hash",
        how="left",
    )

    # Total runs per category (the rho denominator), counted before non-fixation rows
    # are dropped. Streamed, so the full frame is never materialised.
    _tot = (
        lf.group_by("category").agg(pl.len().alias("total")).collect(engine="streaming")
    )
    total_counts = dict(
        zip(_tot.get_column("category").to_list(), _tot.get_column("total").to_list())
    )

    # Only fixation events are ever drawn/tested, so materialise just those.
    merged_raw = lf.filter(pl.col("fixation")).collect()

    _vc = merged_raw["category"].value_counts()
    fixation_counts = dict(
        zip(_vc.get_column("category").to_list(), _vc.get_column("count").to_list())
    )

    # Subsample each category down to the cap with a within-category shuffle (uniform
    # sample), keeping every violin's KDE and every pairwise test cheap and faithful.
    subsampled = False
    if max_points_per_category is not None:
        largest_category = max(fixation_counts.values(), default=None)
        subsampled = (
            largest_category is not None and largest_category > max_points_per_category
        )
        merged_raw = (
            merged_raw.with_columns(
                pl.int_range(pl.len()).shuffle(seed=0).over("category").alias("_rn")
            )
            .filter(pl.col("_rn") < max_points_per_category)
            .drop("_rn")
        )

    loaded = (
        merged_raw.to_pandas(),
        fixation_counts,
        total_counts,
        r,
        r_suffix,
        subsampled,
    )

    if cache_dir is not None:
        _write_fixation_steps_cache(cache_dir, loaded, max_points_per_category)

    return loaded


def build_fixation_steps_cache(
    batch_dir,
    r_values=None,
    max_points_per_category=50_000,
    force=False,
):
    """Precompute the cached fixation-steps sample for every r in a finished batch.

    One scan of the raw shards per r value. Meant to run once, on a compute node, as the
    job chained after aggregation, so every later violin / p-value figure is a file read.

    Args:
        batch_dir: a finished batch (needs graph_props.csv and tmp/results/).
        r_values: which r values to cache. Defaults to every r present in
            graph_statistics.csv, which is the authoritative record of what was simulated.
        max_points_per_category: subsample cap; part of the cache key.
        force: rebuild even if a matching cache already exists.

    Returns:
        list of written cache paths.
    """
    batch_path = Path(batch_dir)

    results_source = resolve_results_source(batch_path)
    if results_source is None:
        raise FileNotFoundError(
            f"No raw results found under {batch_path / 'tmp' / 'results'}; the violin "
            f"sample can only be built from raw rows."
        )

    df_graphs = pd.read_csv(batch_path / "graph_props.csv")

    if r_values is None:
        stats_path = batch_path / "graph_statistics.csv"
        if not stats_path.exists():
            raise FileNotFoundError(
                f"{stats_path} missing; pass r_values explicitly or run aggregate_batch first."
            )
        r_values = sorted(pd.read_csv(stats_path, usecols=["r"])["r"].unique().tolist())

    written = []
    for r in r_values:
        path = fixation_steps_cache_path(batch_path, r, max_points_per_category)
        if not force and _fixation_steps_cache_is_valid(
            batch_path, r, max_points_per_category
        ):
            print(f"r={r}: cache already present, skipping.")
            written.append(path)
            continue
        print(f"r={r}: scanning raw results...")
        load_fixation_steps_by_category(
            results_source,
            df_graphs,
            r=r,
            max_points_per_category=max_points_per_category,
            cache_dir=batch_path,
        )
        written.append(path)
    return written

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

from .constants import HASH_DTYPES

from .provenance import load_batch_info
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
    "load_graph_statistics",
    "load_fixation_steps_by_category",
    "compute_fixation_steps_by_category",
    "build_fixation_steps_cache",
    "fixation_steps_cache_path",
    "list_cached_fixation_steps",
]


# Aggregated per-repeat results filename. "raw_results" = the raw, one-row-per-run
# table, as opposed to the aggregated graph_statistics.csv. (Historically named
# "full_results"; existing batches were migrated on disk to this name.)
RAW_RESULTS_STEM = "raw_results"

# Per-job temp files the worker writes into tmp/results/, one per LSF array index.
PER_JOB_RESULT_STEM = "raw_results_job"


def _as_list(val):
    """Wrap a scalar in a list; pass a list/tuple/set through unchanged.

    The idiom this module uses for every parameter that accepts either one value or
    several: category_filter, r_filter, and batch_dir itself.
    """
    return list(val) if isinstance(val, (list, tuple, set)) else [val]


def _batch_paths(batch_dir):
    """Normalise ``batch_dir`` (one directory, or a list of them) to a list of Paths."""
    return [Path(d) for d in _as_list(batch_dir)]


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
    include_order_stats=False,
):
    """Aggregate raw simulation results to one row per (graph, r) with fixation statistics.

    BUILDER. Streams results from results_path (Parquet or CSV), merges with df_graphs,
    sorts, and writes graph_statistics.csv, unconditionally overwriting any existing file.

    This is a job-sized operation: on the 100K-reps batch it reads ~39GB across 1000
    shards. It is therefore called from ``pipeline.aggregate_batch`` (i.e. from an LSF
    job) and nowhere else. Readers -- notebooks, streamlit, the ML step -- call
    ``load_graph_statistics`` instead, which reads the CSV and refuses to build it.

    The two used to be one "compute if missing, else load" function. That shape is right
    when the compute is cheap and wrong here: hitting the build branch by accident meant a
    Jupyter kernel silently starting a 1000-shard aggregation, indistinguishable from a
    hung cell. Splitting them makes the expensive path something you can only ask for.

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

    Args:
        results_path: raw results to scan -- a single raw_results.parquet/.csv, or a glob
            of per-job shards (see resolve_results_source, which prefers the glob and
            explains why the fused file breaks on large batches).
        df_graphs: DataFrame with graph structural properties (must have 'wl_hash', 'graph_name')
        graph_statistics_path: path where graph_statistics.csv is written
        include_order_stats: if True, also compute median/quartile/iqr of steps (slow).

    Returns:
        analysis_df: the table as written (no filters, no analytic reference columns --
            those are a reader concern, see load_graph_statistics)
    """
    import polars as pl

    graph_statistics_path = Path(graph_statistics_path)
    files = _expand_shards(results_path)

    if include_order_stats:
        # Quantiles are NOT decomposable: an exact median needs every value of a group
        # held at once, so it cannot be combined from per-shard partials. This path
        # therefore does the whole batch in one group_by, which polars does not bound
        # (see _aggregate_chunked) -- fine for small batches, fatal for large ones.
        print(f"Aggregating {len(files)} file(s) in ONE pass (order stats requested)...")
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

    graph_statistics_path.parent.mkdir(parents=True, exist_ok=True)
    analysis_df.to_csv(graph_statistics_path, index=False)
    print(f"Wrote {len(analysis_df):,} rows -> {graph_statistics_path}")
    return analysis_df


def _report_stats_stitch(frames, stitched):
    """Report what stitching several batches produced: size, ragged columns, collisions.

    Both findings are things a silent concat would hide until a figure looked wrong. A
    column only one batch has reads as "all NaN" downstream, and a (wl_hash, r) present in
    two batches draws the same graph twice.
    """
    print(
        f"Stitched {len(frames)} batches -> {len(stitched):,} rows; "
        f"column 'batch' identifies the source."
    )

    every_col = {col for frame in frames for col in frame.columns}
    ragged = sorted(c for c in every_col if any(c not in f.columns for f in frames))
    if ragged:
        print(f"Ragged columns (NaN outside their source batch): {', '.join(ragged)}")

    if not {"wl_hash", "r"} <= set(stitched.columns):
        return
    keys = stitched[["wl_hash", "r"]]
    collided = keys[keys.duplicated(keep=False)].drop_duplicates()
    if len(collided):
        print(
            f"WARNING: {len(collided):,} (wl_hash, r) key(s) appear in more than one "
            f"batch. Rows are kept separately, distinguished by 'batch'. Facet or color "
            f"by it, or the same graph is plotted twice."
        )


def load_graph_statistics(batch_dir, category_filter=None, r_filter=None):
    """READER. Load a batch's aggregated statistics; never builds them.

    Counterpart to ``build_graph_statistics``. Raises if the aggregation has not run,
    rather than quietly starting a 1000-shard scan inside whatever process asked --
    a notebook, streamlit, the ML step. If you see the error, run the aggregation job.

    ``batch_dir`` may be one directory or a **list** of them, stitched at read time: the
    frames are concatenated and a ``batch`` column records where each row came from. This
    replaces the old on-disk "combined batch", which unioned the same CSVs into a third
    directory and symlinked 2000 raw shards alongside them. None of that needed
    materialising: every statistic here decomposes into additive per-shard partials keyed
    by (wl_hash, r), so concatenating two rollups is exactly the rollup of their union.

    Two batches may or may not overlap, and the two cases mean different things:

    - **Disjoint** (the respiratory zoo plus the GA extreme graphs): the union is simply
      more categories to plot side by side.
    - **Identical** (the same zoo simulated twice at different n_repeats): every
      (wl_hash, r) collides, and comparing them is the whole point.

    Both are legitimate, so an overlap warns rather than raising, and ``batch`` is always
    present so a collision is labelled rather than silent.

    Columns present in only some batches (the respiratory-only construction parameters
    ``branching``, ``depth``, ``n_rods``, ...) are kept and filled with NaN elsewhere.
    That is the honest encoding, since a GA-evolved graph has no branching factor, and
    they are reported at load time so an empty column is never a surprise mid-figure.

    Filtering is applied as a view on the way out, so the on-disk graph_statistics.csv
    always holds every category and r value; only the returned frame is narrowed.

    Args:
        batch_dir: a batch directory holding graph_statistics.csv, or a list of them
            to stitch into one frame.
        category_filter: keep only these categories. A single value or a list/tuple/set;
            None keeps all categories.
        r_filter: keep only these selection coefficients. A single value or a
            list/tuple/set; None keeps all r values.

    Returns:
        analysis_df: aggregated DataFrame ready for plotting, with the analytic
            complete-graph reference columns attached and a ``batch`` column naming each
            row's source batch.
    """
    frames = []
    for batch_path in _batch_paths(batch_dir):
        stats_path = batch_path / "graph_statistics.csv"
        if not stats_path.exists():
            raise FileNotFoundError(
                f"{stats_path} does not exist: this batch has not been aggregated yet.\n"
                f"Build it with the aggregation job:\n"
                f"    uv run python -m moran_process.pipeline.aggregate_batch "
                f"--batch-dir {batch_path}\n"
                f"or submit the whole post-batch chain:\n"
                f"    uv run python -m moran_process.pipeline.post_batch "
                f"--batch-dir {batch_path}"
            )

        frame = pd.read_csv(stats_path, dtype=HASH_DTYPES)

        # Derived from n_nodes and r alone, so they are recomputed on read rather than
        # being persisted. That keeps the on-disk graph_statistics.csv schema unchanged,
        # so existing batches need no migration.
        frame = add_analytic_reference_columns(frame)
        frame.insert(0, "batch", batch_path.name)
        print(f"Loaded {len(frame):,} rows from {stats_path}")
        frames.append(frame)

    if len(frames) == 1:
        analysis_df = frames[0]
    else:
        analysis_df = pd.concat(frames, ignore_index=True, sort=False)
        _report_stats_stitch(frames, analysis_df)

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


def list_cached_fixation_steps(batch_dir):
    """Every (r, cap) pair this batch has a complete cache for, sorted by r.

    Used both to resolve ``r=None`` when exactly one r is cached and to make a cache
    miss say what IS available instead of just what is not.

    Returns:
        list of (r, max_points_per_category, path) tuples.
    """
    cache_dir = Path(batch_dir) / CACHE_DIR_NAME
    if not cache_dir.is_dir():
        return []

    found = []
    for path in sorted(cache_dir.glob(f"{FIXATION_STEPS_CACHE_STEM}_r*_n*.parquet")):
        sidecar = path.with_suffix(".json")
        if not sidecar.exists():
            continue
        with open(sidecar) as f:
            meta = json.load(f)
        found.append((meta["r"], meta.get("max_points_per_category"), path))
    return sorted(found, key=lambda t: (t[0], t[1] is None, t[1] or 0))


def _read_fixation_steps_cache(batch_dir, r, max_points_per_category):
    """Return the cached loader tuple for this (r, cap), or None on a miss."""
    if not _fixation_steps_cache_is_valid(batch_dir, r, max_points_per_category):
        return None

    path = fixation_steps_cache_path(batch_dir, r, max_points_per_category)
    with open(path.with_suffix(".json")) as f:
        meta = json.load(f)

    print(f"Using cached fixation-steps sample: {path}")
    sample = pd.read_parquet(path)
    # Caches written before direction existed have no 'group' column. Every batch from
    # that era was entirely undirected, so group == category is the true value there,
    # not a placeholder standing in for something unknown.
    if "group" not in sample.columns:
        sample["group"] = sample["category"]
    return (
        sample,
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


def _accumulate_counts(acc, df, key_col, val_col):
    """Fold a per-shard count frame into a running {category: count} dict."""
    for key, val in zip(df.get_column(key_col).to_list(), df.get_column(val_col).to_list()):
        acc[key] = acc.get(key, 0) + int(val)


def _trim_reservoir(reservoir, max_points_per_category):
    """Keep the ``cap`` smallest sampling keys per violin group, dropping the rest.

    Partitioned on 'group' rather than 'category' so a directed graph and its undirected
    twin each get the full cap, instead of sharing one.
    """
    import polars as pl

    return (
        reservoir.sort("_key")
        .with_columns(pl.int_range(pl.len()).over("group").alias("_rank"))
        .filter(pl.col("_rank") < max_points_per_category)
        .drop("_rank")
    )


def compute_fixation_steps_by_category(
    results_source,
    df_graphs,
    r_values,
    max_points_per_category=50_000,
    seed=0,
    progress_every=100,
):
    """BUILDER. Scan the raw shards ONCE and return a bounded sample for every r value.

    This is the job-sized half of the violin/p-value input. It never materialises more
    than one shard plus the reservoirs, whatever the batch size.

    Two things make that true, and they solve different problems.

    **Bounded memory: a reservoir keyed on a uniform draw**, not a per-shard quota. Every
    fixation row gets an iid uniform(0, 1) key; after each shard a reservoir keeps only
    the ``cap`` smallest keys per category. Keeping the k smallest of n iid uniforms is
    exactly a uniform random subset of size k, and that property does not care how the
    rows were split across files or what order they arrived in. So folding shard by shard
    gives the *same distribution* as shuffling all 7.2e9 rows and taking the first k, and
    allocation lands proportional to each shard's contribution automatically, with no
    counting pass and no second read. The implementation this replaced collected every
    fixation row for one r before subsampling (~1.2e9 rows) and died at 32GB, exit 137.

    **Bounded I/O: all r values share one pass.** Every shard holds rows for every r, so
    looping r on the outside and shards on the inside meant opening all 2000 files once
    per r value: six passes over 39GB to extract data that arrives together. Measured
    cold, reading one r from a shard costs 1.10-1.64s and reading all six costs 1.46s,
    because the cost is fetching the file off shared storage, not decoding rows. Row-group
    pruning does not help with that and rather disguised it. So the shard loop is on the
    outside and each read fans out to one reservoir per r.

    Args:
        results_source: raw results -- a single file or a glob of per-job shards
            (see resolve_results_source).
        df_graphs: DataFrame with at least 'wl_hash' and 'category'.
        r_values: the selection coefficients to sample. A single value is accepted and
            treated as a one-element list. Required rather than resolved from the data,
            because resolving would cost an extra full-column scan.
        max_points_per_category: reservoir size per category, per r. None keeps every
            fixation row, which removes the memory bound -- only sensible on small batches.
        seed: base seed. Shard i, r-slot j draws from ``default_rng((seed, i, j))``, so
            the sample is reproducible and every (shard, r) stream is independent.

    Returns:
        dict mapping r -> the same 6-tuple ``load_fixation_steps_by_category`` returns:
        (sample_df, fixation_counts, total_counts, r, r_suffix, subsampled).
    """
    import polars as pl

    if not isinstance(r_values, (list, tuple, set)):
        r_values = [r_values]
    r_values = sorted({float(r) for r in r_values})
    if not r_values:
        raise ValueError("r_values is empty; nothing to sample.")

    files = _expand_shards(results_source)

    # A violin needs one x position per distribution, and since an undirected graph and
    # its directed twin now share a category (direction lives in its own column), the
    # grouping key is the (category, is_directed) pair. It is materialised as one string
    # because the counts end up as JSON sidecar keys, which cannot be tuples. 'category'
    # is carried alongside so callers can still colour a pair with a single hue.
    graphs = df_graphs[["wl_hash", "category"]].copy()
    _directed = (
        df_graphs["is_directed"].fillna(False).astype(bool)
        if "is_directed" in df_graphs.columns
        else pd.Series(False, index=df_graphs.index)
    )
    graphs["group"] = graphs["category"].where(
        ~_directed, graphs["category"] + " (directed)"
    )
    cats = pl.from_pandas(graphs).lazy()

    total_counts = {r: {} for r in r_values}
    fixation_counts = {r: {} for r in r_values}
    reservoirs = {r: None for r in r_values}
    rows_seen = {r: 0 for r in r_values}

    print(
        f"sampling {len(files)} shard(s) in one pass for r={r_values}, "
        f"cap={max_points_per_category}"
    )
    for i, path in enumerate(files):
        if i > 0 and i % progress_every == 0:
            print(f"  sampled {i}/{len(files)} shards...")

        scanner = scan_results(path)
        names = scanner.collect_schema().names()
        has_r = "r" in names
        if not has_r and len(r_values) > 1:
            raise ValueError(
                f"{path} has no r column, so it cannot be split across "
                f"{len(r_values)} r values; pass a single r for this batch."
            )

        lf = scanner.select(["wl_hash", "steps", "fixation"] + (["r"] if has_r else []))
        if has_r:
            lf = lf.filter(pl.col("r").is_in(r_values))

        # Left-join category before the fixation filter: total_counts is the rho
        # denominator, so it has to count non-fixation runs too.
        shard = (
            lf.join(cats, on="wl_hash", how="left")
            .select(
                ["category", "group", "steps", "fixation"] + (["r"] if has_r else [])
            )
            .collect()
        )
        if shard.height == 0:
            continue

        for j, r in enumerate(r_values):
            slice_ = shard.filter(pl.col("r") == r).drop("r") if has_r else shard
            if slice_.height == 0:
                continue
            rows_seen[r] += slice_.height

            _accumulate_counts(
                total_counts[r],
                slice_.group_by("group").agg(pl.len().alias("n")),
                "group",
                "n",
            )

            fx = slice_.filter(pl.col("fixation")).select(
                ["category", "group", "steps"]
            )
            if fx.height == 0:
                continue

            _accumulate_counts(
                fixation_counts[r],
                fx.group_by("group").agg(pl.len().alias("n")),
                "group",
                "n",
            )

            if max_points_per_category is not None:
                keys = np.random.default_rng((seed, i, j)).random(fx.height)
                fx = fx.with_columns(pl.Series("_key", keys))

            current = reservoirs[r]
            current = fx if current is None else pl.concat([current, fx])
            if max_points_per_category is not None:
                current = _trim_reservoir(current, max_points_per_category)
            reservoirs[r] = current

        del shard

    missing = [r for r in r_values if rows_seen[r] == 0]
    if missing:
        raise ValueError(
            f"no rows matched r={missing} in {results_source}; check "
            f"graph_statistics.csv for the r values this batch actually holds."
        )

    results = {}
    for r in r_values:
        reservoir = reservoirs[r]
        if reservoir is None:
            sample = pd.DataFrame({"category": [], "group": [], "steps": []})
        else:
            sample = reservoir.drop("_key", strict=False).to_pandas()

        # Counted before the cap was applied, so this reports whether the picture you see
        # is a sample or the whole thing.
        largest = max(fixation_counts[r].values(), default=0)
        subsampled = (
            max_points_per_category is not None and largest > max_points_per_category
        )
        results[r] = (
            sample,
            fixation_counts[r],
            total_counts[r],
            r,
            f"  (r={r})",
            subsampled,
        )
    return results


def _resolve_stitch_r(batch_paths, r, max_points_per_category):
    """Pick which r to load when the caller passed ``r=None``.

    With one batch this is the old rule: if exactly one r is cached at this cap, use it.
    With several, the candidates are the r values cached at this cap in EVERY batch, so a
    stitch can never silently resolve to an r only one side is able to serve.
    """
    if r is not None:
        return r

    per_batch = [
        {e[0] for e in list_cached_fixation_steps(p) if e[1] == max_points_per_category}
        for p in batch_paths
    ]
    common = set.intersection(*per_batch) if per_batch else set()
    if len(common) == 1:
        return common.pop()

    raise ValueError(
        f"r=None needs exactly one r cached at cap {max_points_per_category} in every "
        f"batch, found {len(common)}: {sorted(common)}. Pass r=<value> explicitly.\n"
        + "\n".join(
            f"  {p.name}: {sorted(s) or 'nothing'}"
            for p, s in zip(batch_paths, per_batch)
        )
    )


def _missing_cache_message(batch_path, r, max_points_per_category):
    """Explain a cache miss, separating "not built yet" from "never simulated".

    Only the first is fixable by running a job. batch_info.json records which r values the
    batch actually ran, so telling the two apart is a dict lookup, and without it you
    would chase a cache job that can never succeed.
    """
    available = list_cached_fixation_steps(batch_path)
    expected = fixation_steps_cache_path(batch_path, r, max_points_per_category)
    simulated = load_batch_info(batch_path).get("simulation", {}).get("r_values")

    head = (
        f"No violin cache for r={r} at cap {max_points_per_category} in "
        f"{batch_path.name}.\n"
        f"  Expected: {expected}\n"
        f"  Cached in this batch: {[(e[0], e[1]) for e in available] or 'nothing'}\n"
    )

    if simulated is not None and r not in simulated:
        return head + (
            f"  Reason: r={r} was never simulated in this batch "
            f"(batch_info r_values = {simulated}).\n"
            f"  This is not fixable by building a cache. Drop r={r}, or drop this batch "
            f"from the stitch."
        )

    return head + (
        f"  Reason: r={r} was simulated but its cache has not been built.\n"
        f"  Build it:\n"
        f"    uv run python -m moran_process.pipeline.cache_violin_data "
        f"--batch-dir {batch_path} --r-values {r} "
        f"--max-points-per-category {max_points_per_category}\n"
        f"  or submit the whole post-batch chain:\n"
        f"    uv run python -m moran_process.pipeline.post_batch --batch-dir {batch_path}"
    )


def _report_cache_stitch(batch_paths, frames, merged_raw):
    """Report a stitched violin sample, warning about categories present in two batches.

    Unlike a (wl_hash, r) collision in the statistics, a shared category here is not
    wrong: pooling two batches' fixation steps for the same category estimates the same
    distribution, and the summed counts give a valid pooled rho. It is only surprising,
    so it is announced rather than refused.
    """
    print(
        f"Stitched {len(frames)} batches -> {len(merged_raw):,} sampled rows; "
        f"column 'batch' identifies the source."
    )

    seen = {}
    for path, frame in zip(batch_paths, frames):
        for category in frame["category"].dropna().unique():
            seen.setdefault(category, []).append(path.name)
    shared = sorted(c for c, batches in seen.items() if len(batches) > 1)
    if shared:
        print(
            f"WARNING: {len(shared)} category/categories appear in more than one batch "
            f"({', '.join(map(str, shared))}). Their violins POOL across batches unless "
            f"you facet by 'batch', and the rho annotation pools too (counts summed)."
        )


def load_fixation_steps_by_category(batch_dir, r=None, max_points_per_category=50_000):
    """READER. Load the cached fixation-steps sample for one (r, cap). Never scans.

    Shared loader for ``plot_steps_violin`` and ``plot_steps_pvalue_matrix`` so the two
    figures are always built from exactly the same rows. Returns:

    - a tidy pandas DataFrame with columns ['batch', 'category', 'steps'] (subsampled);
    - ``fixation_counts``: true per-category fixation counts, counted *before*
      subsampling so callers can annotate how much data backs each category;
    - ``total_counts``: per-category total run counts (fixation + non-fixation),
      counted before the fixation filter, so callers can report rho = fix / total;
    - the resolved ``r`` and an ``r_suffix`` label for titles;
    - ``subsampled``: whether the cap actually trimmed any category.

    A miss raises instead of falling back to a raw scan. The fallback is what made the
    notebook unpredictable: the same cell was either a 20ms file read or a 42GB scan
    depending on state you could not see from the call. Building the sample is a job
    (``pipeline.cache_violin_data``), so asking for one that does not exist is a
    missing prerequisite, not a slow path.

    ``batch_dir`` may be one directory or a **list** of them, matching
    ``load_graph_statistics``. Stitching is exact rather than approximate: the cache is a
    per-category reservoir, so concatenating two batches' samples is the same as sampling
    their union, and ``fixation_counts`` / ``total_counts`` are counted before subsampling
    and therefore sum. A batch that cannot serve the requested r raises rather than being
    skipped, because dropping one silently yields a figure that looks complete while
    missing half its categories.

    Args:
        batch_dir: a batch directory (the cache lives in <batch_dir>/cache/), or a list
            of them to stitch.
        r: which selection coefficient. If None, resolved to the single r cached at this
            cap across every batch; otherwise you are asked to pick.
        max_points_per_category: the cap the cache was built with. Part of the cache key.
    """
    batch_paths = _batch_paths(batch_dir)
    r = _resolve_stitch_r(batch_paths, r, max_points_per_category)

    frames, fixation_counts, total_counts = [], {}, {}
    subsampled, r_suffix = False, None

    for batch_path in batch_paths:
        cached = _read_fixation_steps_cache(batch_path, r, max_points_per_category)
        if cached is None:
            raise FileNotFoundError(
                _missing_cache_message(batch_path, r, max_points_per_category)
            )
        frame, batch_fix, batch_total, _, batch_suffix, batch_subsampled = cached

        frame = frame.copy()
        frame.insert(0, "batch", batch_path.name)
        frames.append(frame)

        for accumulator, counts in (
            (fixation_counts, batch_fix),
            (total_counts, batch_total),
        ):
            for category, count in counts.items():
                accumulator[category] = accumulator.get(category, 0) + int(count)

        subsampled = subsampled or batch_subsampled
        r_suffix = r_suffix if r_suffix is not None else batch_suffix

    if len(frames) == 1:
        merged_raw = frames[0]
    else:
        merged_raw = pd.concat(frames, ignore_index=True)
        _report_cache_stitch(batch_paths, frames, merged_raw)

    return merged_raw, fixation_counts, total_counts, r, r_suffix, subsampled


def build_fixation_steps_cache(
    batch_dir,
    r_values=None,
    max_points_per_category=50_000,
    force=False,
):
    """Precompute the cached fixation-steps sample for every r in a finished batch.

    **One** scan of the raw shards, covering every r value at once, not one scan per r.
    Every shard holds rows for every r, so a pass per r meant re-fetching the same 39GB
    off shared storage six times for data that arrives together (see
    compute_fixation_steps_by_category for the measurements). Meant to run once, on a
    compute node, as the job chained after aggregation, so every later violin / p-value
    figure is a file read.

    Args:
        batch_dir: a finished batch (needs graph_props.csv and tmp/results/).
        r_values: which r values to cache. Defaults to every r present in
            graph_statistics.csv, which is the authoritative record of what was simulated.
        max_points_per_category: subsample cap; part of the cache key.
        force: rebuild even if a matching cache already exists.

    Returns:
        list of cache paths that now exist (both freshly written and already present).
    """
    batch_path = Path(batch_dir)

    results_source = resolve_results_source(batch_path)
    if results_source is None:
        raise FileNotFoundError(
            f"No raw results found under {batch_path / 'tmp' / 'results'}; the violin "
            f"sample can only be built from raw rows."
        )

    df_graphs = pd.read_csv(batch_path / "graph_props.csv", dtype=HASH_DTYPES)

    if r_values is None:
        stats_path = batch_path / "graph_statistics.csv"
        if not stats_path.exists():
            raise FileNotFoundError(
                f"{stats_path} missing; pass r_values explicitly or run aggregate_batch first."
            )
        r_values = pd.read_csv(stats_path, usecols=["r"])["r"].unique().tolist()

    # Normalise to plain floats: these r values become dict keys and filename fragments,
    # and numpy scalars stringify differently across numpy versions.
    r_values = sorted(float(r) for r in r_values)

    # Decide what still needs building BEFORE the scan, so the single pass covers exactly
    # the missing r values. Skipping an r here is what makes the job cheap to re-run.
    todo, present = [], []
    for r in r_values:
        if not force and _fixation_steps_cache_is_valid(
            batch_path, r, max_points_per_category
        ):
            print(f"r={r}: cache already present, skipping.")
            present.append(r)
        else:
            todo.append(r)

    if todo:
        results = compute_fixation_steps_by_category(
            results_source,
            df_graphs,
            r_values=todo,
            max_points_per_category=max_points_per_category,
        )
        for r in todo:
            _write_fixation_steps_cache(
                batch_path, results[r], max_points_per_category
            )

    return [
        fixation_steps_cache_path(batch_path, r, max_points_per_category)
        for r in sorted(present + todo)
    ]

"""
Results IO: locate a batch's aggregated results, stream per-job files into one,
and roll raw results up to per-(graph, r) fixation statistics.

Heavy readers (polars, pyarrow) are imported lazily inside the functions that
need them, so importing this module stays cheap.
"""

import shutil
from pathlib import Path

import pandas as pd

__all__ = [
    "RAW_RESULTS_STEM",
    "PER_JOB_RESULT_STEM",
    "resolve_results_path",
    "aggregate_results_no_load",
    "build_graph_statistics",
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


def build_graph_statistics(
    results_path, df_graphs, graph_statistics_path, category_filter=None, r_filter=None
):
    """Aggregate raw simulation results to one row per (graph, r) with fixation statistics.

    If graph_statistics_path already exists, loads it directly. Otherwise streams results
    from results_path (Parquet or CSV), merges with df_graphs, sorts, and saves.

    Filtering is applied as a view *after* the full table is built/loaded, so the cached
    graph_statistics.csv always holds every category and r value; only the returned frame
    is narrowed.

    Args:
        results_path: path to raw_results.parquet (or .csv)
        df_graphs: DataFrame with graph structural properties (must have 'wl_hash', 'graph_name')
        graph_statistics_path: path where graph_statistics.csv is saved / loaded from
        category_filter: keep only these categories. A single value or a list/tuple/set;
            None keeps all categories.
        r_filter: keep only these selection coefficients. A single value or a
            list/tuple/set; None keeps all r values.

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
        results_path = Path(results_path)
        if results_path.suffix == ".parquet":
            lazy_df = pl.scan_parquet(str(results_path))
        else:
            lazy_df = pl.scan_csv(str(results_path))

        agg_results_df = (
            lazy_df.with_columns(
                pl.when(pl.col("fixation"))
                .then(pl.col("steps"))
                .otherwise(None)
                .alias("steps_success")
            )
            .group_by(["wl_hash", "r", "graph_name"])
            .agg(
                [
                    pl.col("fixation").mean().alias("prob_fixation"),
                    pl.col("steps_success").median().alias("median_steps"),
                    pl.col("steps_success").mean().alias("mean_steps"),
                    pl.col("steps_success").std().alias("std_steps"),
                    pl.col("steps_success").quantile(0.25).alias("q25_steps"),
                    pl.col("steps_success").quantile(0.75).alias("q75_steps"),
                    (
                        pl.col("steps_success").quantile(0.75)
                        - pl.col("steps_success").quantile(0.25)
                    ).alias("iqr_steps"),
                    pl.col("fixation").count().alias("n_grouped"),
                ]
            )
            .collect(engine="streaming")
            .to_pandas()
        )

        print("Shape before merging: ", agg_results_df.shape)

        analysis_df = pd.merge(
            agg_results_df,
            df_graphs,
            on=["wl_hash", "graph_name"],
            how="left",
            suffixes=("", "_db"),
        )
        analysis_df["z_order"] = (analysis_df["category"] != "Random").astype(int)
        analysis_df = analysis_df.sort_values("z_order").drop(columns="z_order")
        analysis_df.to_csv(graph_statistics_path, index=False)

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

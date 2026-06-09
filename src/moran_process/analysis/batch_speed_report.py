import re
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def _parse_all_log_times(log_dir: Path) -> pd.DataFrame:
    rows = []
    # Sort by filename so that a higher LSF array ID (later resubmission) comes last.
    for log_file in sorted(log_dir.glob("*.out")):
        if "register" in log_file.name:
            continue
        m_id = re.search(r"_(\d+)\.out$", log_file.name)
        if not m_id:
            continue
        text = log_file.read_text(errors="ignore")
        m_run    = re.search(r"Run time\s*:\s*(\d+)\s*sec", text)
        m_turn   = re.search(r"Turnaround time\s*:\s*(\d+)\s*sec", text)
        m_maxmem = re.search(r"Max Memory\s*:\s*([\d.]+)\s*MB", text)
        m_avgmem = re.search(r"Average Memory\s*:\s*([\d.]+)\s*MB", text)
        if m_run and m_turn:
            rows.append({
                "job_id":         int(m_id.group(1)),
                "run_sec":        int(m_run.group(1)),
                "turnaround_sec": int(m_turn.group(1)),
                "max_mem_mb":     float(m_maxmem.group(1)) if m_maxmem else None,
                "avg_mem_mb":     float(m_avgmem.group(1)) if m_avgmem else None,
            })
    df = pd.DataFrame(rows)
    # If the batch was resubmitted, duplicate job_ids appear (one per submission).
    # Keep only the most recent run (last entry after sorting by filename).
    if not df.empty:
        df = df.drop_duplicates(subset="job_id", keep="last")
    return df


def _fmt_sec(s: float) -> str:
    if s >= 3600:
        return f"{s/3600:.2f} h"
    if s >= 60:
        return f"{s/60:.1f} min"
    return f"{s:.1f} sec"


def batch_speed_report(batch_name: str, df: pd.DataFrame, batch_root: Path) -> None:
    """Print performance stats and show histograms for one batch.

    Parameters
    ----------
    batch_name:
        Display name of the batch (used as plot title and log-dir lookup key).
    df:
        Raw simulation results for this batch (must have columns: job_id, steps;
        the optional `duration` column enables the engine-throughput breakdown).
    batch_root:
        Path to the simulation_data/<batch_name> directory.
    """
    log_dir = batch_root / "logs"

    job_steps = (
        df.groupby("job_id")["steps"].sum()
        .reset_index()
        .rename(columns={"steps": "total_steps"})
    )

    all_log_times = _parse_all_log_times(log_dir)
    job_stats = job_steps.merge(all_log_times, on="job_id", how="inner")
    job_stats["steps_M"]       = job_stats["total_steps"] / 1e6
    job_stats["steps_per_sec"] = job_stats["total_steps"] / job_stats["run_sec"]

    # Pure-simulation seconds per job: sum of the worker-measured `duration`
    # (timed around sim.run() only). This isolates true engine throughput from
    # the fixed per-job startup/IO overhead that dominates the LSF wall-clock
    # (`run_sec`). Optional: callers that don't select `duration` skip these lines.
    has_dur = "duration" in df.columns
    if has_dur:
        job_dur = (
            df.groupby("job_id")["duration"].sum()
            .reset_index()
            .rename(columns={"duration": "sim_sec"})
        )
        job_stats = job_stats.merge(job_dur, on="job_id", how="left")

    total_steps_M = job_stats["steps_M"].sum()
    total_run_h   = job_stats["run_sec"].sum() / 3600
    total_turn_h  = job_stats["turnaround_sec"].sum() / 3600
    avg_run_sec   = job_stats["run_sec"].mean()
    max_run_sec   = job_stats["run_sec"].max()
    avg_speed_k   = job_stats["steps_per_sec"].mean() / 1e3
    min_speed_k   = job_stats["steps_per_sec"].min()  / 1e3
    max_speed_k   = job_stats["steps_per_sec"].max()  / 1e3

    max_mem_col = job_stats["max_mem_mb"].dropna()
    avg_mem_col = job_stats["avg_mem_mb"].dropna()
    has_mem = len(max_mem_col) > 0

    print(f"Batch: {batch_name}  ({len(job_stats)} jobs)")
    print(f"  Total steps      : {total_steps_M:.2f} M")
    print(f"  Total run time   : {_fmt_sec(total_run_h * 3600)}")
    print(f"  Total turnaround : {_fmt_sec(total_turn_h * 3600)}")
    print(f"  Avg job run time : {_fmt_sec(avg_run_sec)}  |  Max: {_fmt_sec(max_run_sec)}")
    print(f"  Wall throughput  : {avg_speed_k:.1f} k steps/s avg  |  min {min_speed_k:.1f}  |  max {max_speed_k:.1f}   (steps / wall-clock, incl. startup/IO)")

    if has_dur:
        total_steps   = job_stats["total_steps"].sum()
        total_run_sec = job_stats["run_sec"].sum()
        total_sim_sec = job_stats["sim_sec"].sum()
        overhead_sec  = max(total_run_sec - total_sim_sec, 0.0)
        sim_pct       = 100 * total_sim_sec / total_run_sec if total_run_sec else float("nan")
        sim_thru_M    = (total_steps / total_sim_sec / 1e6) if total_sim_sec else float("nan")
        wall_thru_M   = (total_steps / total_run_sec / 1e6) if total_run_sec else float("nan")
        speedup       = (sim_thru_M / wall_thru_M) if wall_thru_M else float("nan")
        print(f"  -- engine throughput (from `duration`, sim-only) --")
        print(f"  Pure sim time    : {_fmt_sec(total_sim_sec)}  ({sim_pct:.2f}% of wall; rest is startup/IO overhead)")
        print(f"  Sim throughput   : {sim_thru_M:.2f} M steps/s   (engine-only, what the C++ core accelerates)")
        print(f"  Overhead-masking : sim is {speedup:.0f}x faster than wall throughput; per-job startup hides it")

    if has_mem:
        print(f"  Peak RAM per job : avg {max_mem_col.mean():.0f} MB  |  min {max_mem_col.min():.0f} MB  |  max {max_mem_col.max():.0f} MB")
        print(f"  Avg  RAM per job : avg {avg_mem_col.mean():.0f} MB  |  min {avg_mem_col.min():.0f} MB  |  max {avg_mem_col.max():.0f} MB")

    n_plots = 3 if has_mem else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(5.5 * n_plots, 4))
    fig.suptitle(batch_name, fontsize=11)

    med = job_stats["run_sec"].median()
    if med >= 3600:
        run_vals = job_stats["run_sec"] / 3600
        run_unit = "h"
    elif med >= 60:
        run_vals = job_stats["run_sec"] / 60
        run_unit = "min"
    else:
        run_vals = job_stats["run_sec"]
        run_unit = "sec"

    axes[0].hist(run_vals, bins=30, color="steelblue", edgecolor="white")
    axes[0].set_xlabel(f"Run time per job ({run_unit})")
    axes[0].set_ylabel("Jobs")
    axes[0].set_title("Job run time")

    axes[1].hist(job_stats["steps_per_sec"] / 1e3, bins=30, color="seagreen", edgecolor="white")
    axes[1].set_xlabel("Speed (k steps / sec)")
    axes[1].set_ylabel("Jobs")
    axes[1].set_title("Job speed")

    if has_mem:
        axes[2].hist(max_mem_col, bins=30, color="mediumpurple", edgecolor="white", label="Peak", alpha=0.7)
        axes[2].hist(avg_mem_col, bins=30, color="orchid",       edgecolor="white", label="Avg",  alpha=0.7)
        axes[2].set_xlabel("Memory per job (MB)")
        axes[2].set_ylabel("Jobs")
        axes[2].set_title("RAM usage")
        axes[2].legend()

    plt.tight_layout()
    plt.show()

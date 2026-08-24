"""Figures for a simulation-driven GA run (``pipeline.ga_search``).

A separate module from ``plots`` for the same reason ``plots`` was split out of the
original monolithic ``analysis_utils``: that file is already ~1600 lines, and these
figures read a different artefact (``ga_history.csv``) than everything in it.

``plot_ga_history`` keeps the conventions of ``plot_multi_model_history`` in
``notebooks/extreme_graphs.ipynb``, so the ML-driven and simulation-driven searches can be
put side by side: left axis fixation time, right axis fixation probability, solid for time
and dashed for probability, complete-graph baselines drawn as the residual origin, and a
sorted unified legend below the axes.

What is added is what measurement makes available and prediction did not: the spread
across survivors, and error bars at the standard error of the measurement. A predicted
fitness has no error bar, so the ML figure could never show whether its trajectory
exceeded the noise floor. This one can.

**Color encodes the search direction.** A run is defined by an angle theta (see
``pipeline.ga_search``), and hue is mapped straight onto it, so the same direction is the
same color in every figure here and a sweep of directions reads as a color wheel. The
older convention -- a hand-picked color per named objective, and a thicker line for
"the metric this run optimized" -- does not survive the move to theta, because every run
now optimizes a combination of both metrics and there is no un-optimized one to gray out.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from .colors import (
    CATEGORY_COLOR_DICT,
    generate_robust_color_dict,
    theta_color,
    theta_color_dict,
)
from .ga_io import (
    _run_dirs as _ga_io_run_dirs,
    final_elite_properties,
    final_population_stats,
    load_elite_population,
    load_ga_history,
    load_ga_state,
)
from .plots import _resolve_figure_path
from .theory import analytic_moran_fc_fixation_prob, analytic_moran_fc_fixation_time

__all__ = [
    "plot_ga_history",
    "plot_ga_runs_comparison",
    "plot_ga_winners_in_context",
    "plot_ga_winners_scatter",
    "plot_ga_trails",
    "plot_ga_explorer_plotly",
    "plot_elite_grid",
    "plot_population",
    "plot_replicate_agreement",
    "plot_selection_efficiency",
    "plot_selection_response",
    "plot_ml_vs_simulation",
    "plot_ml_vs_simulation_scatter",
    "plot_ml_vs_simulation_scatter_plotly",
]

# The search space is fixed at avian_r4_l7's size (see pipeline.ga_search), so the
# complete-graph baselines are constants for every figure here.
N_NODES = 31
N_EDGES = 34
R_VALUE = 1.1

# The residual origin every figure here measures against. Recomputed from `theory` rather
# than imported from `pipeline.ga_search`, which would drag the whole submission stack
# (ProcessLab, joblib, polars) into a plotting import; the two call the same formula with
# the same constants, so they cannot disagree.
RHO_COMPLETE_PLOT = float(analytic_moran_fc_fixation_prob(N_NODES, R_VALUE))
T_COMPLETE_PLOT = float(analytic_moran_fc_fixation_time(N_NODES, R_VALUE))

_METRIC_LABEL = {
    "mean_steps": "Mean Fixation Time (steps)",
    "prob_fixation": "Fixation Probability",
    # ga_search --metric weighted. The units are random-graph SDs, which is the only
    # thing that makes the two terms addable, so the label says so.
    "weighted": "Combined objective (random-graph SDs)",
}
# The linestyle convention carried over from plot_multi_model_history.
_METRIC_LINESTYLE = {"mean_steps": "-", "prob_fixation": "--"}

# Marker shape encodes HOW a graph was selected, independently of the color, which encodes
# what it was selected for. Simulation-driven categories are filled and the two ML model
# families are open, so "measured" versus "predicted" survives even in greyscale, and LR
# versus XGBOOST is separable without reading the legend text.
_MARKER_BY_FAMILY = {"LR": "o", "XGBOOST": "s", "simulation": "D"}
_PLOTLY_MARKER_BY_FAMILY = {"LR": "circle", "XGBOOST": "square", "simulation": "diamond"}
_FALLBACK_MARKER, _PLOTLY_FALLBACK_MARKER = "^", "triangle-up"


def _ga_colors(frame):
    """Category -> color for a GA frame, hue driven by the search direction.

    One resolver for every figure in this module, so a given theta is the same color in
    the trajectory plot, the comparison plot and the winners scatter. Falls back to the
    hand-picked map (and then to husl) for any category without a theta, which is how the
    respiratory and random categories keep their established colors when they share axes
    with GA runs.
    """
    colors = generate_robust_color_dict(frame, CATEGORY_COLOR_DICT)
    if "theta" in frame:
        colors.update(theta_color_dict(frame))
    return colors


def _format_steps(value, _position=None):
    """Tick label for a step count: 2400 -> '2.4K', 35000 -> '35K', 1.2e6 -> '1.2M'.

    Fixation times here run from a few thousand to a few million steps, so a linear axis
    otherwise labels itself '1000000' and the reader counts zeros. One decimal is kept only
    when it carries information ('2.4K'), not when it is padding ('35K', not '35.0K').
    """
    for threshold, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(value) >= threshold:
            return f"{value / threshold:.1f}".rstrip("0").rstrip(".") + suffix
    return f"{value:g}"


def _sims_note(values):
    """How many simulations sit behind the plotted points: '100K', or '10K-100K' if mixed.

    Taken from the ``n_grouped`` column of whatever frame is being drawn, never from the
    run's configured ``n_repeats``. The two differ exactly when a generation came back
    short, which is the case where a stated repeat count would be a lie.

    A range rather than a single number is the honest answer when a figure overlays sources
    measured at different depths (the random cloud is a 10K-repeat batch, the winners are
    100K), so it is reported instead of averaged away.
    """
    counts = sorted({int(v) for v in pd.Series(values).dropna()})
    if not counts:
        return "?"
    if len(counts) == 1:
        return _format_steps(counts[0])
    return f"{_format_steps(counts[0])}-{_format_steps(counts[-1])}"


def _scale_steps_axis(axis, which="x", logscale=True):
    """Put a fixation-time axis on a log scale, or give it K/M tick labels instead.

    The two branches are alternatives, not a default plus an override: on a log axis the
    ticks are already decades and reformatting them to '10K' would fight matplotlib's
    LogFormatter for the minor ticks. Returns the axis label so the ', log scale' note is
    written in the same place the scale is set, and cannot fall out of step with it.
    """
    if logscale:
        (axis.set_xscale if which == "x" else axis.set_yscale)("log")
        return _METRIC_LABEL["mean_steps"] + ", log scale"
    (axis.xaxis if which == "x" else axis.yaxis).set_major_formatter(
        FuncFormatter(_format_steps)
    )
    return _METRIC_LABEL["mean_steps"]


def _selection_family(source, category):
    """Which selection procedure produced a category: 'simulation', 'LR', or 'XGBOOST'.

    Read off the category string because that is the only record of it -- the ML batch's
    categories were written by the notebook that produced them and carry the model name
    inline (``maximize XGBOOST Fixation Time (log-ratio)``). Anything unrecognized falls
    back to a distinct third shape rather than silently borrowing one of these.
    """
    if source == "simulation-driven":
        return "simulation"
    for family in ("LR", "XGBOOST"):
        if family in category:
            return family
    return "other"


def _survivor_trajectory(history):
    """Per-generation mean, spread and SEM of both metrics over the surviving elites.

    The SEM of a mean over k survivors combines their individual measurement errors as
    sqrt(sum(sem^2))/k, which is the error on the plotted point. It is NOT the spread
    across survivors -- that is a real property of the population, and is drawn as a band.
    """
    rows = []
    for generation, group in history.groupby("generation"):
        row = {"generation": generation, "n_survivors": len(group)}
        for metric in ("mean_steps", "prob_fixation"):
            sem = group[f"{metric}_sem"].to_numpy(dtype=float)
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_lo"] = group[metric].min()
            row[f"{metric}_hi"] = group[metric].max()
            row[f"{metric}_sem"] = np.sqrt(np.nansum(sem**2)) / len(group)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("generation")


def _draw_baselines(ax_time, ax_prob, t_complete, rho_complete):
    """The complete-graph reference lines, exactly as the ML figure drew them."""
    ax_time.axhline(t_complete, color="#7f7f7f", linestyle=":", linewidth=1.2, alpha=0.8)
    ax_time.text(
        0.01, t_complete, f" complete T={t_complete:.0f}",
        transform=ax_time.get_yaxis_transform(),
        va="bottom", ha="left", fontsize=8, color="#7f7f7f",
    )
    ax_prob.axhline(
        rho_complete, color="#7f7f7f", linestyle=(0, (1, 3)), linewidth=1.2, alpha=0.8
    )
    ax_prob.text(
        0.99, rho_complete, f"complete rho={rho_complete:.3f} ",
        transform=ax_prob.get_yaxis_transform(),
        va="bottom", ha="right", fontsize=8, color="#7f7f7f",
    )


def _sorted_legend(ax, lines):
    """Time before probability, then alphabetical. Mirrors the ML figure's legend order."""
    pairs = [(ln, ln.get_label()) for ln in lines]
    pairs.sort(key=lambda item: (1 if "Probability" in item[1] else 0, item[1]))
    ax.legend(
        [p[0] for p in pairs], [p[1] for p in pairs],
        loc="upper center", bbox_to_anchor=(0.5, -0.15),
        fancybox=True, shadow=True, ncol=2,
    )


def plot_ga_history(
    run_dir, figures_dir=None, show_spread=True, figsize=(12, 7), logscale=True
):
    """Measured fitness trajectory of one GA run, on physical axes.

    Both metrics are drawn for the same survivors -- the optimized one thick, the other
    thin -- which is the same two-axis layout the ML figure used for its primary and
    secondary models. Watching the un-optimized metric is the point: it shows whether
    driving fixation time also moved fixation probability, which is the amplifier /
    suppressor question.

    ``logscale`` applies to the fixation-time axis only. On by default because a maximizing
    run climbs by more than an order of magnitude, and on a linear axis its first thirty
    generations are flattened against the floor.
    """
    history = load_ga_history(run_dir, survivors_only=True)
    state = load_ga_state(run_dir)
    theta = state["theta"]
    trajectory = _survivor_trajectory(history)

    rho_complete = float(analytic_moran_fc_fixation_prob(N_NODES, R_VALUE))
    t_complete = float(analytic_moran_fc_fixation_time(N_NODES, R_VALUE))

    fig, ax_time = plt.subplots(figsize=figsize)
    ax_prob = ax_time.twinx()
    ax_time.set_xlabel("Generation", fontweight="bold")
    ax_time.set_ylabel(
        _scale_steps_axis(ax_time, which="y", logscale=logscale), fontweight="bold"
    )
    ax_prob.set_ylabel(_METRIC_LABEL["prob_fixation"], fontweight="bold")

    # Every run optimizes a combination of both metrics, so BOTH traces are the optimized
    # one and both take the run's color. Which of the two the search actually leans on is
    # the weights' business, and the subtitle states them rather than the linewidth
    # implying them.
    category = state["category"]
    color = theta_color(theta)
    lines = []
    for this_metric in ("mean_steps", "prob_fixation"):
        axis = ax_time if this_metric == "mean_steps" else ax_prob
        x = trajectory["generation"]
        y = trajectory[f"{this_metric}_mean"]

        if show_spread:
            axis.fill_between(
                x, trajectory[f"{this_metric}_lo"], trajectory[f"{this_metric}_hi"],
                color=color, alpha=0.12, linewidth=0,
            )
        container = axis.errorbar(
            x, y, yerr=trajectory[f"{this_metric}_sem"],
            color=color, linestyle=_METRIC_LINESTYLE[this_metric],
            linewidth=3.0,
            elinewidth=0.8, capsize=0, errorevery=max(1, len(x) // 40),
            label=_METRIC_LABEL[this_metric],
        )
        # The container, not container.lines[0]: errorbar() puts the label on the container,
        # so the bare Line2D inside it is labelled '_nolegend_' and the legend rendered two
        # entries reading exactly that.
        lines.append(container)

    _draw_baselines(ax_time, ax_prob, t_complete, rho_complete)

    if logscale:
        # Set the limit rather than autoscale it. errorbar draws its bars as a
        # LineCollection, and on a log axis matplotlib resolves a collection's extent
        # through a min-positive value that lands far below the data: the axis opened at
        # 2.4 steps against a minimum of 5.5K, spending three empty decades. The plotted
        # extent is already known here (the survivor min/max), so it is used directly.
        # The margin is multiplicative because a fixed one is asymmetric in log space.
        lo = min(trajectory["mean_steps_lo"].min(), t_complete)
        hi = max(trajectory["mean_steps_hi"].max(), t_complete)
        pad = (hi / lo) ** 0.05
        ax_time.set_ylim(lo / pad, hi * pad)

    _sorted_legend(ax_time, lines)
    ax_time.grid(True, linestyle=":", alpha=0.7)
    ax_time.set_title(
        "Evolution of Topologies (measured)\n"
        + f"{category} ({state.get('quadrant', '')})  |  "
        + f"w=({state.get('weight_prob', float('nan')):+.2f}, "
        + f"{state.get('weight_time', float('nan')):+.2f})  |  "
        + f"N={N_NODES}, r={R_VALUE}, {state.get('generations', '?')} generations  |  "
        + f"{_sims_note(history['n_grouped'])} simulations per graph",
        fontsize=14, pad=15,
    )
    fig.tight_layout()

    path = _resolve_figure_path(figures_dir, "plot_ga_history", theta=f"{theta:03.0f}")
    if path:
        fig.savefig(path, bbox_inches="tight", dpi=150, facecolor="white")
        print(f"Saved: {path}")
    return fig


def plot_ga_runs_comparison(run_dirs, figures_dir=None, figsize=(13, 6), logscale=True):
    """All runs' trajectories side by side: fixation time left, probability right.

    The ML notebook could not draw this -- its eight runs optimized eight different model
    outputs and so were not commensurable. These four optimize two measured quantities in
    two directions, so they share axes and the envelope they trace out is the reachable
    range for (31, 34) graphs.
    """
    history = load_ga_history(run_dirs, survivors_only=True)
    rho_complete = float(analytic_moran_fc_fixation_prob(N_NODES, R_VALUE))
    t_complete = float(analytic_moran_fc_fixation_time(N_NODES, R_VALUE))

    colors = _ga_colors(history)
    # Sorted by angle rather than by name, so the legend and the drawing order walk the
    # circle in order. Alphabetical on "theta=045, theta=135, ..." happens to agree, but
    # only because the label is zero-padded; sorting on the number says what is meant.
    by_theta = sorted(history.groupby("run"), key=lambda item: item[1]["theta"].iloc[0])

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for metric, axis in zip(("mean_steps", "prob_fixation"), axes):
        for run, group in by_theta:
            category = group["category"].iloc[0]
            trajectory = _survivor_trajectory(group)
            # Every run optimizes a combination of both metrics, so neither panel shows an
            # incidental quantity: both are drawn solid and equal. How hard a given run
            # pushes on this particular axis is its weight, which is the legend's job.
            axis.plot(
                trajectory["generation"], trajectory[f"{metric}_mean"],
                color=colors.get(category, "#2ca02c"),
                linewidth=2.6,
                label=category,
            )
        baseline = t_complete if metric == "mean_steps" else rho_complete
        axis.axhline(baseline, color="#7f7f7f", linestyle=":", linewidth=1.2)
        axis.set_xlabel("Generation", fontweight="bold")
        label = (
            _scale_steps_axis(axis, which="y", logscale=logscale)
            if metric == "mean_steps"
            else _METRIC_LABEL[metric]
        )
        axis.set_ylabel(label, fontweight="bold")
        axis.grid(True, linestyle=":", alpha=0.7)
        axis.set_title(_METRIC_LABEL[metric])

    # Deduplicated by label: one line is drawn per RUN but the legend describes
    # CATEGORIES, so a 3-replicate launch would otherwise repeat each entry three times
    # and a 12-run matrix would print a twelve-item legend naming four things.
    handles, labels = axes[0].get_legend_handles_labels()
    seen = {}
    for handle, label in zip(handles, labels):
        seen.setdefault(label, handle)
    fig.legend(
        list(seen.values()), list(seen), loc="lower center",
        ncol=min(4, len(seen)),
        bbox_to_anchor=(0.5, -0.06), fancybox=True,
    )
    fig.suptitle(
        "Simulation-driven GA: all runs, colored by search direction\n"
        f"{history['run'].nunique()} runs  |  N={N_NODES}, E={N_EDGES}, r={R_VALUE}  |  "
        f"{_sims_note(history['n_grouped'])} simulations per graph",
        fontsize=14,
    )
    fig.tight_layout()

    path = _resolve_figure_path(figures_dir, "plot_ga_runs_comparison", n_runs=len(set(history["run"])))
    if path:
        fig.savefig(path, bbox_inches="tight", dpi=150, facecolor="white")
        print(f"Saved: {path}")
    return fig


REPLICATE_PROPERTIES = [
    "degree_std",
    "max_degree",
    "average_clustering",
    "degree_assortativity",
    "average_shortest_path_length",
    "diameter",
]


def plot_replicate_agreement(
    run_dirs, properties=None, figures_dir=None, figsize=(14, 5.5)
):
    """Do independent repeats of the same search arrive at the same kind of graph?

    Left: the fitness each replicate reached, one marker per elite. Right: the structural
    fingerprint of those elites, z-scored per property across every graph shown, one line
    per (objective, replicate). Replicates that converged on the same kind of topology
    trace the same line; replicates that found different solutions to the same problem
    trace different ones.

    Deliberately not plotted: overlap of ``wl_hash``. It is zero between independent runs
    and will stay zero, because ~20k visited topologies out of an astronomical space
    guarantees it. Reporting it would only ever say "the runs disagree completely", which
    is an artefact of the space's size and not a finding.

    z-scoring is what makes six properties on incompatible scales (a clustering
    coefficient near 0.1, a diameter near 25) share one axis. It is computed across the
    graphs in this figure, so the reference is "unusual compared to the other winners",
    not "unusual compared to random graphs".
    """
    properties = list(properties or REPLICATE_PROPERTIES)
    elites = final_elite_properties(run_dirs)
    missing = [p for p in properties if p not in elites.columns]
    if missing:
        raise KeyError(
            f"graph_props.csv has no column(s) {missing}. Available structural columns: "
            f"{sorted(c for c in elites.columns if c not in ('run', 'category'))}"
        )
    colors = _ga_colors(elites)

    # A single-replicate launch has no -repN suffix, so label those by run instead of
    # dropping them: the figure is still the right way to compare several runs side by
    # side. The shared launch prefix ("2026_08_19-phase1-") is stripped first, since it is
    # identical on every tick and would otherwise take more width than the labels.
    # Truncated at the last separator, because commonprefix works per character and would
    # otherwise eat the shared digits of "theta045"/"theta135" and leave a ragged stub.
    prefix = os.path.commonprefix(sorted(elites["run"].unique()))
    prefix = prefix[: prefix.rfind("-") + 1]
    elites["label"] = np.where(
        elites["replicate"].isna(),
        elites["run"].str.slice(len(prefix)).replace("", np.nan).fillna(elites["run"]),
        elites["category"] + " rep" + elites["replicate"].astype(str),
    )

    fig, (ax_fitness, ax_shape) = plt.subplots(1, 2, figsize=figsize)

    for index, (label, group) in enumerate(elites.groupby("label")):
        category = group["category"].iloc[0]
        color = colors[category]
        offsets = np.linspace(-0.18, 0.18, len(group)) if len(group) > 1 else [0.0]
        ax_fitness.plot(
            index + np.asarray(offsets), group["weighted"] / group["weighted"].mean(),
            linestyle="none", marker="o", markersize=5,
            markerfacecolor=color, markeredgecolor=color, alpha=0.75,
        )

    # Fitness is shown relative to each group's own mean. Every run now scores in the same
    # unit (random-graph SDs along its own direction), but those scores still differ
    # several-fold between directions, since some directions are far more reachable than
    # others. Within a direction the replicates then sit on a common scale, and agreement
    # is "do the clouds line up at 1.0".
    ax_fitness.axhline(1.0, color="#7f7f7f", linestyle=":", linewidth=1.2)
    ax_fitness.set_xticks(range(elites["label"].nunique()))
    ax_fitness.set_xticklabels(sorted(elites["label"].unique()), rotation=30, ha="right")
    ax_fitness.set_ylabel("Fitness / group mean", fontweight="bold")
    ax_fitness.set_title("Did the replicates reach the same fitness?")

    z_scored = elites[properties].apply(
        lambda column: (column - column.mean()) / (column.std() or 1.0)
    )
    z_scored["label"] = elites["label"]
    z_scored["category"] = elites["category"]
    x = np.arange(len(properties))
    for label, group in z_scored.groupby("label"):
        ax_shape.plot(
            x, group[properties].mean().to_numpy(),
            color=colors[group["category"].iloc[0]],
            linewidth=2.0, marker="o", markersize=5, alpha=0.85, label=label,
        )
    ax_shape.axhline(0.0, color="#7f7f7f", linestyle=":", linewidth=1.2)
    ax_shape.set_xticks(x)
    ax_shape.set_xticklabels(
        [p.replace("_", " ") for p in properties], rotation=30, ha="right"
    )
    ax_shape.set_ylabel("z-score across the graphs shown", fontweight="bold")
    ax_shape.set_title("Did they arrive at the same kind of graph?")

    for axis in (ax_fitness, ax_shape):
        axis.grid(True, linestyle=":", alpha=0.7)

    handles, labels = ax_shape.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=4,
        bbox_to_anchor=(0.5, -0.20), fancybox=True,
    )
    fig.suptitle(
        "Replicate agreement: independent searches, same objective\n"
        f"{elites['run'].nunique()} runs  |  N={N_NODES}, E={N_EDGES}, r={R_VALUE}",
        fontweight="bold",
    )
    fig.tight_layout()

    path = _resolve_figure_path(
        figures_dir, "plot_replicate_agreement", n_runs=elites["run"].nunique()
    )
    if path:
        fig.savefig(path, bbox_inches="tight", dpi=150, facecolor="white")
        print(f"Saved: {path}")
    return fig


def _selection_efficiency(history):
    """Per-generation rho: how well measured fitness tracks true fitness.

    A generation ranks its candidates on a *measured* metric, so what selection actually
    responds to is the true value plus measurement noise. Under the usual additive model
    the response is proportional to the correlation between the two,

        rho = 1 / sqrt(1 + (SEM / SD_between)^2)

    which is 1 when measurement is perfect and falls toward 0 as noise swamps the real
    spread. It answers the question n_repeats is chosen to answer, and unlike a raw
    signal-to-noise ratio it is bounded, dimensionless and comparable across metrics.

    Computed over every candidate in the generation rather than the survivors alone,
    because the candidate pool is what selection ranked. Restricting to survivors would
    measure the spread that selection already narrowed and understate rho systematically.
    """
    rows = []
    for (run, generation), group in history.groupby(["run", "generation"]):
        if len(group) < 3:
            # Generation 0 is the initial population alone; a spread over fewer than a
            # handful of graphs is not an estimate of anything.
            continue
        sd = float(group["weighted"].std())
        sem = float(group["weighted_sem"].mean())
        if not np.isfinite(sd) or sd <= 0 or not np.isfinite(sem):
            continue
        rows.append({
            "run": run,
            "generation": generation,
            "category": group["category"].iloc[0],
            "theta": group["theta"].iloc[0],
            "sd_between": sd,
            "sem": sem,
            "rho": 1.0 / np.sqrt(1.0 + (sem / sd) ** 2),
        })
    return pd.DataFrame(rows).sort_values(["run", "generation"])


def plot_selection_efficiency(run_dirs, figures_dir=None, figsize=(13, 5.5)):
    """Whether selection was still responding to topology, or had started chasing noise.

    Left: rho per generation (see ``_selection_efficiency``). Right: the two quantities
    rho is built from, each divided by its own first-generation value so both are
    dimensionless and runs optimizing different metrics share one axis. Solid is the
    between-graph spread, dashed the measurement error.

    This is the figure that justifies (or refutes) the choice of ``n_repeats`` after the
    fact, and it costs no simulation: every input is already in ga_history.csv.

    The right panel is here because rho alone says a run degraded without saying why, and
    the two metrics degrade for opposite reasons. prob_fixation is a ratio, so its SEM,
    sqrt(p(1-p)/n), barely moves as p does: rho falls only when the population converges
    and the spread collapses. mean_steps is a scale, so its SEM is proportional to the
    mean -- a run that successfully maximizes it inflates its own noise floor in step with
    its own signal, and can lose rho while still visibly improving.
    """
    history = load_ga_history(run_dirs)
    efficiency = _selection_efficiency(history)
    if efficiency.empty:
        raise ValueError(
            "No generation had enough candidates to estimate rho. A run that has only "
            "finished generation 0 has nothing to plot here yet."
        )
    colors = _ga_colors(efficiency)

    fig, (ax_rho, ax_parts) = plt.subplots(1, 2, figsize=figsize)
    for category, group in efficiency.groupby("category"):
        color = colors[category]
        ax_rho.plot(
            group["generation"], group["rho"],
            color=color, linewidth=2.0, label=category,
        )
        first = group.iloc[0]
        ax_parts.plot(
            group["generation"], group["sd_between"] / first["sd_between"],
            color=color, linewidth=2.0, linestyle="-",
        )
        ax_parts.plot(
            group["generation"], group["sem"] / first["sem"],
            color=color, linewidth=1.6, linestyle="--",
        )

    # 1.0 is perfect measurement. The shading marks where fewer than half the ideal
    # response survives the noise, which is the point at which more repeats buy more than
    # more generations do.
    ax_rho.axhline(1.0, color="#7f7f7f", linestyle=":", linewidth=1.2)
    ax_rho.axhspan(0, 0.5, color="#d62728", alpha=0.06)
    ax_rho.set_ylim(0, 1.05)
    ax_rho.set_ylabel("Selection efficiency  rho", fontweight="bold")
    ax_rho.set_title("Fraction of the ideal selection response retained")

    ax_parts.axhline(1.0, color="#7f7f7f", linestyle=":", linewidth=1.2)
    ax_parts.set_yscale("log")
    ax_parts.set_ylabel("Relative to first generation", fontweight="bold")
    ax_parts.set_title("solid = between-graph spread,  dashed = measurement error")

    for axis in (ax_rho, ax_parts):
        axis.set_xlabel("Generation", fontweight="bold")
        axis.grid(True, linestyle=":", alpha=0.7)

    handles, labels = ax_rho.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=4,
        bbox_to_anchor=(0.5, -0.08), fancybox=True,
    )
    n_repeats = int(history["n_grouped"].median())
    fig.suptitle(
        "Was selection still tracking topology, or fitting noise?\n"
        f"{efficiency['run'].nunique()} runs  |  {n_repeats:,} repeats per graph  |  "
        f"N={N_NODES}, E={N_EDGES}, r={R_VALUE}",
        fontweight="bold",
    )
    fig.tight_layout()

    path = _resolve_figure_path(
        figures_dir, "plot_selection_efficiency", n_runs=efficiency["run"].nunique()
    )
    if path:
        fig.savefig(path, bbox_inches="tight", dpi=150, facecolor="white")
        print(f"Saved: {path}")
    return fig


def _selection_response(history, smooth):
    """Per-generation gain in mean survivor score, and how many elites were replaced.

    Two independent readings of "is selection still doing anything":

    * ``delta`` -- the change in the surviving elites' mean score from one generation to
      the next. This is the realized response to selection. It is what actually matters,
      and it is noisy, so it is also reported smoothed.
    * ``n_new`` -- how many of the surviving elites are newly arrived children rather than
      returning incumbents. Zero means no child managed to displace an elite, which is a
      frozen search regardless of what the fitness numbers do.

    ``sem_delta`` is the noise floor to read ``delta`` against: sqrt(2) x the standard
    error of a generation's mean, since a difference of two means combines two errors. It
    is an OVERESTIMATE, because consecutive generations share their elites (elites are
    re-simulated every generation) so the two means are positively correlated and the true
    error on their difference is smaller. Overestimating is the safe direction: it makes
    the "is this gain real" test conservative.
    """
    rows = []
    for run, group in history.groupby("run"):
        per_gen = group.groupby("generation")
        mean = per_gen["weighted"].mean()
        # Error on the mean of k survivors, combining their individual measurement errors.
        sem = per_gen["weighted_sem"].apply(
            lambda s: float(np.sqrt(np.nansum(np.square(s.to_numpy(dtype=float))))) / len(s)
        )
        new = per_gen["is_new"].sum()
        frame = pd.DataFrame({
            "run": run,
            "category": group["category"].iloc[0],
            "theta": group["theta"].iloc[0],
            "generation": mean.index,
            "mean_score": mean.to_numpy(),
            "delta": mean.diff().to_numpy(),
            "sem_delta": (np.sqrt(2) * sem).to_numpy(),
            "n_new": new.to_numpy(),
            "gain": (mean - mean.iloc[0]).to_numpy(),
        })
        frame["delta_smooth"] = (
            frame["delta"].rolling(smooth, center=True, min_periods=1).mean()
        )
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def plot_selection_response(
    run_dirs, smooth=5, figures_dir=None, figsize=(13.5, 5.5), show_runs=True
):
    """Is the population still IMPROVING, generation by generation?

    ``plot_selection_efficiency`` asks whether the ranking was trustworthy.
    This asks the complementary question -- whether that ranking was still buying anything.
    A search can have excellent rho and be completely stuck, if every candidate is already
    as good as its parent.

    Left: the per-generation change in the surviving elites' mean score, smoothed over
    ``smooth`` generations. The gray band is the measurement noise floor on that
    difference; a curve inside the band is a population whose apparent movement cannot be
    distinguished from remeasurement error. Selection has stopped paying when the curve
    settles onto zero AND stays inside the band.

    Right: how many of the ``pop_size`` elites each generation are newly arrived children.
    This is the mechanical version of the same question and needs no error bar: if nothing
    displaces an incumbent, nothing is happening, whatever the scores say. It usually
    decays toward a small nonzero number rather than to zero, because re-simulating the
    elites means an incumbent can lose its place to measurement noise alone.
    """
    history = load_ga_history(run_dirs, survivors_only=True)
    response = _selection_response(history, smooth)
    colors = _ga_colors(response)

    fig, (ax_delta, ax_new) = plt.subplots(1, 2, figsize=figsize)

    # Noise floor first, so the curves sit on top of it.
    floor = response.groupby("generation")["sem_delta"].median()
    ax_delta.fill_between(
        floor.index, -floor, floor, color="#bdbdbd", alpha=0.35, linewidth=0,
        label="measurement noise on the change",
    )

    for (category, theta), group in sorted(
        response.groupby(["category", "theta"]), key=lambda kv: kv[0][1]
    ):
        color = colors[category]
        if show_runs:
            for _, run_group in group.groupby("run"):
                ax_delta.plot(run_group["generation"], run_group["delta_smooth"],
                              color=color, linewidth=1.0, alpha=0.55)
                ax_new.plot(run_group["generation"],
                            run_group["n_new"].rolling(smooth, center=True,
                                                       min_periods=1).mean(),
                            color=color, linewidth=1.0, alpha=0.55)
        else:
            for axis, column in ((ax_delta, "delta_smooth"), (ax_new, "n_new")):
                mean = group.groupby("generation")[column].mean()
                axis.plot(mean.index, mean, color=color, linewidth=1.8, label=category)
        if show_runs:
            ax_delta.plot([], [], color=color, linewidth=1.8, label=category)

    ax_delta.axhline(0, color="#333333", linestyle="-", linewidth=1.0, zorder=1)
    ax_delta.set_ylabel("Gain per generation (random-graph SDs)", fontweight="bold")
    ax_delta.set_title(
        f"Response to selection, smoothed over {smooth} generations\n"
        "inside the gray band = indistinguishable from remeasurement",
        fontsize=11,
    )
    ax_new.set_ylabel("New elites per generation", fontweight="bold")
    ax_new.set_title(
        "Elites displaced by their own children\n"
        "flat near zero = the population has stopped moving",
        fontsize=11,
    )
    for axis in (ax_delta, ax_new):
        axis.set_xlabel("Generation", fontweight="bold")
        axis.grid(True, linestyle=":", alpha=0.7)
        axis.set_axisbelow(True)

    handles, labels = ax_delta.get_legend_handles_labels()
    unique = {}
    for handle, label in zip(handles, labels):
        unique.setdefault(label, handle)
    fig.legend(list(unique.values()), list(unique), loc="center left",
               bbox_to_anchor=(1.0, 0.5), fontsize=8, frameon=False)

    fig.suptitle(
        "Was selection still buying anything?\n"
        f"{response['run'].nunique()} runs, {response['category'].nunique()} directions  |  "
        f"N={N_NODES}, E={N_EDGES}, r={R_VALUE}",
        fontweight="bold",
    )
    fig.tight_layout()

    path = _resolve_figure_path(
        figures_dir, "plot_selection_response", n_runs=response["run"].nunique()
    )
    if path:
        fig.savefig(path, bbox_inches="tight", dpi=150, facecolor="white")
        print(f"Saved: {path}")
    return fig


def plot_ga_winners_in_context(
    run_dirs, reference_stats, metric="mean_steps", figures_dir=None, figsize=(11, 6),
    logscale=True,
):
    """The GA winners against the distribution of random graphs, with avian marked.

    This is the figure the whole experiment exists to produce. ``reference_stats`` is a
    graph_statistics frame from an ordinary batch (e.g. the respiratory-vs-random batch),
    filtered here to the same (N, E, r) the search ran at so the comparison is
    like-for-like.
    """
    reference = reference_stats[
        (reference_stats["n_nodes"] == N_NODES)
        & (reference_stats["n_edges"] == N_EDGES)
        & (np.isclose(reference_stats["r"], R_VALUE))
    ]
    winners = load_ga_history(run_dirs, survivors_only=True)
    winners = winners[winners["generation"] == winners.groupby("run")["generation"].transform("max")]

    # A log axis has to change the binning too, not just the scale: 45 linearly spaced bins
    # rendered on a log axis get visually narrower to the right, so equal-count bars look
    # like a falling density that is not in the data.
    use_log = logscale and metric == "mean_steps"
    values = reference[metric].to_numpy(dtype=float)
    bins = (
        np.logspace(np.log10(values.min()), np.log10(values.max()), 46)
        if use_log and len(values) and values.min() > 0
        else 45
    )

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(
        reference[metric], bins=bins, color="#c8c8c8", edgecolor="white",
        label=f"random ({N_NODES}, {N_EDGES}) graphs, n={len(reference)} "
              f"({_sims_note(reference['n_grouped'])} sims each)",
    )

    for run, group in winners.groupby("run"):
        category = group["category"].iloc[0]
        for value in group[metric]:
            ax.axvline(
                value, color=CATEGORY_COLOR_DICT.get(category, "#2ca02c"),
                linewidth=1.4, alpha=0.75,
            )
        ax.plot([], [], color=CATEGORY_COLOR_DICT.get(category, "#2ca02c"),
                linewidth=2.5, label=f"GA winners: {category}")

    avian = reference[reference["graph_name"].astype(str).str.startswith("avian")]
    if not avian.empty:
        ax.axvline(
            avian[metric].iloc[0], color=CATEGORY_COLOR_DICT["Avian"],
            linewidth=3.0, linestyle="--", label="avian_r4_l7",
        )

    label = (
        _scale_steps_axis(ax, which="x", logscale=logscale)
        if metric == "mean_steps"
        else _METRIC_LABEL[metric]
    )
    ax.set_xlabel(label, fontweight="bold")
    ax.set_ylabel("Number of random graphs", fontweight="bold")
    ax.set_title(
        f"Where the GA winners and the avian lung sit among random "
        f"({N_NODES}, {N_EDGES}) graphs  |  r={R_VALUE}\n"
        f"{winners['run'].nunique()} GA runs, {len(winners)} winners, "
        f"{_sims_note(winners['n_grouped'])} simulations per graph",
        fontsize=13,
    )
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.5, axis="y")
    fig.tight_layout()

    path = _resolve_figure_path(figures_dir, "plot_ga_winners_in_context", metric=metric)
    if path:
        fig.savefig(path, bbox_inches="tight", dpi=150, facecolor="white")
        print(f"Saved: {path}")
    return fig


def _reference_cloud(reference_stats):
    """The size-matched random graphs, and the respiratory graphs, at the GA's r.

    Split out because two figures now draw the same background and a second copy of the
    filter is a second place for it to drift out of step with the search space.
    Returns (cloud, organs); organs is every non-random category, which for the reference
    batch means the respiratory graphs plus the structural controls.
    """
    at_r = reference_stats[np.isclose(reference_stats["r"], R_VALUE)]
    sized = at_r[(at_r["n_nodes"] == N_NODES) & (at_r["n_edges"] == N_EDGES)]
    cloud = sized[sized["category"] == "Random"]
    if cloud.empty:
        raise ValueError(
            f"No random ({N_NODES}, {N_EDGES}) graphs at r={R_VALUE} in reference_stats. "
            f"The winners would have nothing to be extreme relative to."
        )
    return cloud, at_r


def _draw_cloud(ax, cloud):
    """The gray random-graph background, identical in every figure that shows it."""
    ax.scatter(
        cloud["mean_steps"], cloud["prob_fixation"],
        s=18, color="#cccccc", edgecolor="none", zorder=1,
        label=f"random ({N_NODES}, {N_EDGES}) graphs, n={len(cloud)}\n"
              f"({_sims_note(cloud['n_grouped'])} sims each)",
    )


def _draw_organs(ax, at_r, categories):
    """Star markers for the named real topologies, annotated when not size-matched."""
    organs = at_r[at_r["category"].isin(categories)]
    for _, row in organs.iterrows():
        note = (
            "" if (row["n_nodes"], row["n_edges"]) == (N_NODES, N_EDGES)
            else f"  [{int(row['n_nodes'])}, {int(row['n_edges'])}]"
        )
        ax.scatter(
            row["mean_steps"], row["prob_fixation"],
            s=230, marker="*",
            facecolor=CATEGORY_COLOR_DICT.get(row["category"], "#000000"),
            edgecolor="white", linewidth=1.0, zorder=7,
            label=f"{row['graph_name']}{note}",
        )
    return organs


def _dedup_legend(ax, **kwargs):
    """Legend with one entry per label. A sweep draws one series per RUN but names
    DIRECTIONS, so 36 entries would describe 12 things and overflow the axes."""
    handles, labels = ax.get_legend_handles_labels()
    unique = {}
    for handle, label in zip(handles, labels):
        unique.setdefault(label, handle)
    ax.legend(list(unique.values()), list(unique), **kwargs)


def plot_ga_trails(
    run_dirs,
    reference_stats,
    every=5,
    figures_dir=None,
    figsize=(12, 8),
    logscale=True,
    respiratory=("Avian", "Mammalian", "Fish"),
):
    """The PATH each search took through the (fixation time, probability) plane.

    ``plot_ga_winners_scatter`` shows where the searches ended. This shows how they got
    there: one trail per run, each point the mean over that generation's surviving elites,
    so 36 runs starting from statistically identical random populations fan out from the
    middle of the random cloud and end at the boundary. The colors are the search
    directions, so the figure reads as the objective pulling each population outward.

    Averaging over survivors rather than over all candidates is deliberate: the candidate
    pool at generation N includes that generation's untested mutants, most of which are
    worse, so its mean lags the population that selection actually kept.

    ``every`` subsamples generations (the first and last are always drawn). At 100
    generations x 36 runs the full trail is 3600 points, which renders as a hairball; every
    5th generation keeps the shape and loses only the jitter.
    """
    cloud, at_r = _reference_cloud(reference_stats)
    history = load_ga_history(run_dirs, survivors_only=True)
    colors = _ga_colors(history)

    fig, ax = plt.subplots(figsize=figsize)
    _draw_cloud(ax, cloud)

    # Angle order, so overlapping trails stack in a predictable sequence rather than an
    # alphabetical one, and the legend walks the circle.
    by_theta = sorted(history.groupby("run"), key=lambda kv: kv[1]["theta"].iloc[0])

    for run, group in by_theta:
        category = group["category"].iloc[0]
        color = colors[category]
        trail = (
            group.groupby("generation")[["mean_steps", "prob_fixation"]]
            .mean()
            .sort_index()
        )
        keep = trail.index.isin(trail.index[::every]) | (trail.index == trail.index[-1])
        trail = trail[keep]

        ax.plot(
            trail["mean_steps"], trail["prob_fixation"],
            color=color, linewidth=1.4, alpha=0.75, zorder=4, label=category,
        )
        # Where it started and where it ended. The start markers all pile up in the middle
        # of the cloud by construction (every run begins from random graphs), which is
        # precisely the point: the fan-out is the search, not the starting condition.
        ax.plot(*trail.iloc[0][["mean_steps", "prob_fixation"]], marker="o",
                markersize=4, color=color, alpha=0.9, zorder=5)
        ax.plot(*trail.iloc[-1][["mean_steps", "prob_fixation"]], marker="D",
                markersize=8, color=color, markeredgecolor="black",
                markeredgewidth=1.0, zorder=6)

    organs = _draw_organs(ax, at_r, respiratory)

    drawn = pd.concat([cloud, history, organs])
    ax.set_xlim(drawn["mean_steps"].min() * 0.85, drawn["mean_steps"].max() * 1.18)
    span = drawn["prob_fixation"].max() - drawn["prob_fixation"].min()
    ax.set_ylim(drawn["prob_fixation"].min() - 0.08 * span,
                drawn["prob_fixation"].max() + 0.08 * span)

    ax.set_xlabel(_scale_steps_axis(ax, which="x", logscale=logscale), fontweight="bold")
    ax.set_ylabel(_METRIC_LABEL["prob_fixation"], fontweight="bold")
    n_dir = history["category"].nunique()
    last = int(history["generation"].max())
    ax.set_title(
        "The path each search took\n"
        f"{n_dir} directions, {history['run'].nunique()} runs, "
        f"generations 0-{last} every {every}  |  "
        f"circle = generation 0, diamond = final  |  "
        f"N={N_NODES}, E={N_EDGES}, r={R_VALUE}",
        fontsize=13, pad=12,
    )
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_axisbelow(True)
    _dedup_legend(ax, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8,
                  frameon=False, labelspacing=0.6)
    fig.tight_layout()

    path = _resolve_figure_path(
        figures_dir, "plot_ga_trails", n_runs=history["run"].nunique(), every=every
    )
    if path:
        fig.savefig(path, bbox_inches="tight", dpi=150, facecolor="white")
        print(f"Saved: {path}")
    return fig


def plot_ga_explorer_plotly(
    run_dirs, reference_stats, trails=True, every=5, height=760, width=1150,
    logscale=True, respiratory=("Avian", "Mammalian", "Fish"),
):
    """Interactive twin of the winners scatter and the trails, in one figure.

    What the static versions cannot do with 36 overlapping series: hover any point for the
    graph that produced it, and click a legend entry to isolate one direction. With twelve
    directions x three replicates that is the difference between a figure you present and a
    figure you can actually interrogate.

    ``trails=True`` adds each run's generation-by-generation path as a line, so one click
    isolates a direction and shows both where it ended and how it got there. Trails are
    added first so plotly paints them under the markers; unlike matplotlib there is no
    zorder, only insertion order.

    plotly is imported inside the function following the precedent below: it is a hard
    dependency but a slow import, and ``analysis_utils/__init__`` pulls this module in
    eagerly, so a module-level import would tax every reader-only import too.
    """
    import plotly.graph_objects as go

    cloud, at_r = _reference_cloud(reference_stats)
    history = load_ga_history(run_dirs, survivors_only=True)
    winners = final_population_stats(run_dirs)
    colors = _ga_colors(history)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cloud["mean_steps"], y=cloud["prob_fixation"], mode="markers",
        name=f"random ({N_NODES}, {N_EDGES}), n={len(cloud)}",
        marker=dict(size=5, color="#cccccc"),
        text=cloud["graph_name"],
        hovertemplate="%{text}<br>T=%{x:.0f}<br>rho=%{y:.4f}<extra></extra>",
    ))

    ordered = sorted(history["category"].unique(),
                     key=lambda c: history.loc[history["category"] == c, "theta"].iloc[0])

    if trails:
        for category in ordered:
            block = history[history["category"] == category]
            for run, group in block.groupby("run"):
                trail = (group.groupby("generation")[["mean_steps", "prob_fixation"]]
                         .mean().sort_index())
                keep = trail.index.isin(trail.index[::every]) | (trail.index == trail.index[-1])
                trail = trail[keep]
                fig.add_trace(go.Scatter(
                    x=trail["mean_steps"], y=trail["prob_fixation"], mode="lines",
                    name=category, legendgroup=category, showlegend=False,
                    line=dict(color=colors[category], width=1.2),
                    opacity=0.55,
                    customdata=trail.index,
                    hovertemplate=(f"{run}<br>generation %{{customdata}}"
                                   "<br>T=%{x:.0f}<br>rho=%{y:.4f}<extra></extra>"),
                ))

    for category in ordered:
        group = winners[winners["category"] == category]
        fig.add_trace(go.Scatter(
            x=group["mean_steps"], y=group["prob_fixation"], mode="markers",
            name=category, legendgroup=category,
            marker=dict(size=9, symbol="diamond", color=colors[category],
                        line=dict(color="white", width=0.8)),
            customdata=np.stack([group["graph_name"], group["run"],
                                 group["weighted"]], axis=-1),
            hovertemplate=("%{customdata[0]}<br>%{customdata[1]}"
                           "<br>score=%{customdata[2]:.2f} SD"
                           "<br>T=%{x:.0f}<br>rho=%{y:.4f}<extra></extra>"),
        ))

    organs = at_r[at_r["category"].isin(respiratory)]
    for _, row in organs.iterrows():
        note = ("" if (row["n_nodes"], row["n_edges"]) == (N_NODES, N_EDGES)
                else f" [{int(row['n_nodes'])},{int(row['n_edges'])}]")
        fig.add_trace(go.Scatter(
            x=[row["mean_steps"]], y=[row["prob_fixation"]], mode="markers",
            name=f"{row['graph_name']}{note}",
            marker=dict(size=17, symbol="star",
                        color=CATEGORY_COLOR_DICT.get(row["category"], "#000000"),
                        line=dict(color="white", width=1)),
            hovertemplate=(f"{row['graph_name']}{note}"
                           "<br>T=%{x:.0f}<br>rho=%{y:.4f}<extra></extra>"),
        ))

    fig.update_layout(
        title=(f"GA explorer: {history['category'].nunique()} directions, "
               f"{history['run'].nunique()} runs"
               + ("  |  lines = search path, diamonds = final elites" if trails else "")
               + "<br><sup>click a legend entry to isolate a direction, "
                 "double-click to isolate it alone</sup>"),
        xaxis_title="Mean Fixation Time (steps)"
                    + (", log scale" if logscale else ""),
        yaxis_title="Fixation Probability",
        xaxis_type="log" if logscale else "linear",
        height=height, width=width, hovermode="closest",
        template="plotly_white",
    )
    return fig


def plot_population(
    ga_runs_dir, prefix, theta, replicate=None, n=None, figures_dir=None,
    per_row=5, size=2.8, with_labels=False,
):
    """Draw the final elite population of one direction. The everyday inspection call.

    ``plot_elite_grid`` shows one graph from every run; this shows every graph from one
    run, which is the other axis of the same question: a direction whose 20 elites are all
    the same shape has converged, and one whose elites are a mixed bag has not.

    ``theta`` is matched after normalization, so 315 and -45 both find the same runs.
    ``replicate=None`` draws every replicate of that direction stacked, each labelled;
    pass an integer for one. ``n`` limits how many elites are drawn per run (default all).

    Raises rather than returning empty when the direction matches nothing, and says which
    directions DO exist -- a silent empty figure is indistinguishable from a run that found
    nothing.
    """
    from moran_process.pipeline.ga_search import normalize_theta

    ga_runs_dir = Path(ga_runs_dir)
    wanted = normalize_theta(theta)
    matched = []
    for run_dir in sorted(p for p in ga_runs_dir.iterdir()
                          if p.name.startswith(prefix) and (p / "ga_config.json").exists()):
        state = load_ga_state(run_dir) or {}
        if state.get("theta") is None or not np.isclose(state["theta"], wanted):
            continue
        if replicate is not None and not run_dir.name.endswith(f"-rep{replicate}"):
            continue
        matched.append((run_dir, state))

    if not matched:
        available = sorted({
            (load_ga_state(p) or {}).get("theta")
            for p in ga_runs_dir.iterdir()
            if p.name.startswith(prefix) and (p / "ga_config.json").exists()
        } - {None})
        raise ValueError(
            f"No run at theta={wanted:g}"
            + (f" replicate {replicate}" if replicate is not None else "")
            + f" under prefix {prefix!r}. Directions present: "
            + ", ".join(f"{t:g}" for t in available)
        )

    panels = []
    for run_dir, state in matched:
        population = load_elite_population(run_dir)[:n]
        for index, graph in enumerate(population):
            panels.append((run_dir.name, state, index, graph))

    n_rows = int(np.ceil(len(panels) / per_row))
    fig, axes = plt.subplots(n_rows, per_row, figsize=(size * per_row, size * n_rows))
    axes = np.atleast_1d(axes).ravel()
    color = theta_color(wanted)

    for axis, (run, state, index, graph) in zip(axes, panels):
        graph.draw(ax=axis, with_labels=with_labels, descriptive=False)
        tag = run.rsplit("-", 1)[-1] if "-rep" in run else run
        axis.set_title(f"{tag}  #{index + 1}\n{graph.name}", fontsize=8, color=color)
        axis.set_xticks([]); axis.set_yticks([])
    for axis in axes[len(panels):]:
        axis.axis("off")

    state = matched[0][1]
    fig.suptitle(
        f"Final population: {state.get('category', f'theta={wanted:g}')} "
        f"({state.get('quadrant', '')})  |  "
        f"{len(matched)} replicate(s), {len(panels)} graphs  |  "
        f"N={N_NODES}, E={N_EDGES}, r={R_VALUE}",
        fontweight="bold", fontsize=13, color=color,
    )
    fig.tight_layout()

    path = _resolve_figure_path(
        figures_dir, "plot_population", theta=f"{wanted:03.0f}",
        rep="all" if replicate is None else replicate,
    )
    if path:
        fig.savefig(path, bbox_inches="tight", dpi=130, facecolor="white")
        print(f"Saved: {path}")
    return fig


def plot_elite_grid(
    run_dirs, figures_dir=None, per_row=3, size=3.2, rank=0, with_labels=False
):
    """Every run's best topology on one sheet, grouped so replicates sit side by side.

    The 36 individual PNGs answer "what does this one look like"; this answers the question
    the sweep was designed around -- did independent searches in the SAME direction arrive
    at the same KIND of graph. Replicates of one direction share a row, so convergence (or
    its absence) is a horizontal comparison rather than a hunt through a folder.

    ``rank`` picks which elite to draw: 0 is the best, 1 the runner-up, and so on. Drawing
    rank 1 as well is a cheap check that a direction's answer is a family of graphs rather
    than one lucky topology.
    """
    run_dirs = _ga_io_run_dirs(run_dirs)
    entries = []
    for run_dir in run_dirs:
        state = load_ga_state(run_dir) or {}
        population = load_elite_population(run_dir)
        if rank < len(population):
            entries.append((state.get("theta", np.inf), run_dir.name,
                            state.get("category", run_dir.name), population[rank]))
    entries.sort(key=lambda e: (e[0], e[1]))

    n_rows = int(np.ceil(len(entries) / per_row))
    fig, axes = plt.subplots(n_rows, per_row,
                             figsize=(size * per_row, size * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for axis, (theta, run, category, graph) in zip(axes, entries):
        graph.draw(ax=axis, with_labels=with_labels, descriptive=False)
        # The replicate tag, not the whole run name: the prefix is identical on all of
        # them and would take more width than the part that differs.
        replicate = run.rsplit("-", 1)[-1] if "-rep" in run else ""
        axis.set_title(f"{category}  {replicate}", fontsize=9,
                       color=theta_color(theta) if np.isfinite(theta) else "#333333",
                       fontweight="bold")
        axis.set_xticks([]); axis.set_yticks([])
    for axis in axes[len(entries):]:
        axis.axis("off")

    fig.suptitle(
        f"Best topology per run ({'winner' if rank == 0 else f'rank {rank + 1}'})  |  "
        f"{len(entries)} runs, N={N_NODES}, E={N_EDGES}, r={R_VALUE}",
        fontweight="bold", fontsize=13,
    )
    fig.tight_layout()

    path = _resolve_figure_path(
        figures_dir, "plot_elite_grid", n_runs=len(entries), rank=rank
    )
    if path:
        fig.savefig(path, bbox_inches="tight", dpi=130, facecolor="white")
        print(f"Saved: {path}")
    return fig


def _run_thetas(run_dirs):
    """``{run_name: theta_deg}``, read from each run's state.

    From ga_state.json rather than ga_config.json because the state holds theta already
    normalized to [0, 360), which is the form every consumer here wants. A run with no
    state yet is simply absent rather than defaulted, so a caller drawing per-direction
    decoration draws nothing for it instead of drawing it in the wrong direction.
    """
    thetas = {}
    for run_dir in _ga_io_run_dirs(run_dirs):
        state = load_ga_state(run_dir) or {}
        if state.get("theta") is not None:
            thetas[run_dir.name] = float(state["theta"])
    return thetas


def _support_line(w_prob, w_time, prob_fixation, mean_steps, half_length=8.0):
    """The objective's iso-line through one point, back in (mean_steps, prob) coordinates.

    The objective is linear in the standardized pair u = (rho - rho_c)/SD_PROB,
    v = log(T/T_c)/SD_LOG_TIME, so its level sets are straight lines there -- and since
    both coordinates are affine in (prob_fixation, log T), they stay straight on a log time
    axis too. Drawn through the winner, the line is the *supporting* line of the search: if
    the run did its job, every other graph in the plot lies on the losing side of it, which
    is the visual form of "the winner maximizes w . (u, v)".

    ``half_length`` is in standardized-SD units along the line, and is deliberately much
    larger than the plotted region so the caller can just set the axis limits and let
    matplotlib clip.
    """
    # Imported inside the function: ga_search pulls ProcessLab, joblib and the whole
    # submission stack behind it, and this module is imported to look at figures.
    from moran_process.pipeline.ga_search import (
        RHO_COMPLETE, SD_LOG_TIME_RESIDUAL, SD_PROB_RESIDUAL, T_COMPLETE,
    )

    u0 = (prob_fixation - RHO_COMPLETE) / SD_PROB_RESIDUAL
    v0 = np.log(mean_steps / T_COMPLETE) / SD_LOG_TIME_RESIDUAL
    norm = np.hypot(w_prob, w_time) or 1.0
    # Tangent to the level set is the normal rotated by 90 degrees.
    du, dv = -w_time / norm, w_prob / norm

    s = np.array([-half_length, half_length])
    rho = RHO_COMPLETE + (u0 + du * s) * SD_PROB_RESIDUAL
    steps = T_COMPLETE * np.exp((v0 + dv * s) * SD_LOG_TIME_RESIDUAL)
    return steps, rho


def plot_ga_winners_scatter(
    run_dirs,
    reference_stats,
    figures_dir=None,
    figsize=(12, 8),
    logscale=True,
    show_support_lines=True,
    show_hull=False,
    respiratory=("Avian", "Mammalian", "Fish"),
):
    """GA winners in the (fixation time, fixation probability) plane, over the random cloud.

    The 2-D companion to ``plot_ga_winners_in_context``, which draws one metric at a time
    and so cannot show the thing a weighted run is actually optimizing: a *direction* in
    the joint plane. ``plot_ml_vs_simulation_scatter`` draws the same plane but requires an
    ML batch to compare against, which a validation launch does not have.

    What this adds beyond a scatter:

    * ``show_support_lines`` draws each weighted run's objective iso-line through its own
      best winner. Maximizing a linear objective returns a support point of the achievable
      set, so a run that worked leaves the entire random cloud on one side of its line.
      This is the check that winners land where the objective says they should, and it is
      read off the figure rather than taken on trust.
    * ``show_hull`` outlines the convex hull of all winners together. With a single pair of
      opposed directions that is just a segment and is off by default; over a circle of
      directions it is the boundary those runs were launched to trace.
    * ``respiratory`` marks the real organ graphs. Only the avian graph is (31, 34); the
      mammalian and fish graphs are a different size, so their marker is annotated with it
      rather than being allowed to imply a like-for-like comparison.
    """
    cloud, at_r = _reference_cloud(reference_stats)

    from moran_process.pipeline.ga_search import theta_weights

    winners = final_population_stats(run_dirs)
    thetas = _run_thetas(run_dirs)
    colors = _ga_colors(winners)

    fig, ax = plt.subplots(figsize=figsize)
    _draw_cloud(ax, cloud)

    # Walked in angle order so the legend reads around the circle rather than around the
    # alphabet, which is what makes a K-direction sweep legible as a sweep.
    by_theta = sorted(
        winners.groupby("run"), key=lambda item: thetas.get(item[0], float("inf"))
    )
    for run, group in by_theta:
        category = group["category"].iloc[0]
        color = colors[category]
        best = group.loc[group["rank"].idxmin()]
        ax.scatter(
            group["mean_steps"], group["prob_fixation"],
            s=55, marker="D", facecolor=color, edgecolor="white", linewidth=0.6,
            alpha=0.85, zorder=5, label=f"{category}  ({len(group)} elites)",
        )
        ax.scatter(
            best["mean_steps"], best["prob_fixation"],
            s=170, marker="D", facecolor=color, edgecolor="black", linewidth=1.4,
            zorder=6,
        )
        if show_support_lines and run in thetas:
            steps, rho = _support_line(
                *theta_weights(thetas[run]),
                best["prob_fixation"], best["mean_steps"],
            )
            ax.plot(steps, rho, color=color, linewidth=1.2, linestyle="--", alpha=0.8,
                    zorder=3)

    if show_hull and len(winners) >= 3:
        from scipy.spatial import ConvexHull

        # Hulled in (log T, rho): that is the plane the objective is linear in, so this
        # outline is exactly the set of points some weight vector could have selected.
        points = np.column_stack([
            np.log10(winners["mean_steps"].to_numpy(dtype=float)),
            winners["prob_fixation"].to_numpy(dtype=float),
        ])
        hull = ConvexHull(points)
        loop = np.append(hull.vertices, hull.vertices[0])
        ax.plot(
            10 ** points[loop, 0], points[loop, 1],
            color="#333333", linewidth=1.3, linestyle="-", alpha=0.55, zorder=4,
            label="convex hull of all winners",
        )

    organs = _draw_organs(ax, at_r, respiratory)

    # Limits from every point actually drawn -- the organ graphs included, or the avian
    # star sits on the axis line. Set explicitly rather than autoscaled because the support
    # lines run far beyond the region on purpose and would otherwise drag the axes with them.
    drawn = pd.concat([cloud, winners, organs])
    ax.set_xlim(drawn["mean_steps"].min() * 0.85, drawn["mean_steps"].max() * 1.18)
    span = drawn["prob_fixation"].max() - drawn["prob_fixation"].min()
    ax.set_ylim(
        drawn["prob_fixation"].min() - 0.12 * span,
        drawn["prob_fixation"].max() + 0.12 * span,
    )

    # The residual origin, drawn only if it is inside the region. At (31, 34) the complete
    # graph is ~4x faster than anything plotted here, so on this figure the crosshair is
    # usually off-scale entirely and an invisible annotation is worse than none: it reads
    # as though the origin were somewhere in view.
    if ax.get_xlim()[0] <= T_COMPLETE_PLOT <= ax.get_xlim()[1]:
        ax.axvline(T_COMPLETE_PLOT, color="#7f7f7f", linestyle=":", linewidth=1.2, zorder=2)
    if ax.get_ylim()[0] <= RHO_COMPLETE_PLOT <= ax.get_ylim()[1]:
        ax.axhline(RHO_COMPLETE_PLOT, color="#7f7f7f", linestyle=":", linewidth=1.2, zorder=2)

    ax.set_xlabel(_scale_steps_axis(ax, which="x", logscale=logscale), fontweight="bold")
    ax.set_ylabel(_METRIC_LABEL["prob_fixation"], fontweight="bold")
    # Counted, not divided. runs/directions is 31/12 here, and integer division would
    # print "2 replicates" for a sweep that has 3 of most directions and 2 of a few.
    n_directions = winners["category"].nunique()
    per = winners.groupby("category")["run"].nunique()
    reps = f"{per.min()}" if per.min() == per.max() else f"{per.min()}-{per.max()}"
    ax.set_title(
        "Where the GA winners sit in the joint plane\n"
        f"{n_directions} search directions, {reps} replicates each "
        f"({winners['run'].nunique()} runs)  |  "
        f"N={N_NODES}, E={N_EDGES}, r={R_VALUE}  |  "
        f"{_sims_note(winners['n_grouped'])} sims/graph"
        + ("  |  dashed = objective iso-line through each best"
           if show_support_lines and thetas else ""),
        fontsize=13, pad=12,
    )
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_axisbelow(True)
    _dedup_legend(ax, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8,
                  frameon=False, labelspacing=0.6)
    fig.tight_layout()

    path = _resolve_figure_path(
        figures_dir, "plot_ga_winners_scatter", n_runs=winners["run"].nunique()
    )
    if path:
        fig.savefig(path, bbox_inches="tight", dpi=150, facecolor="white")
        print(f"Saved: {path}")
    return fig


def _ml_vs_simulation_frame(run_dirs, ml_stats, keep_name=False):
    """Both searches' winners in one long frame: source, category, and the two metrics.

    ``keep_name`` carries ``graph_name`` through as well, which the static figures have no
    room for but the interactive one puts in the hover.

    The ML batch is filtered to the GA's exact (N, E, r) rather than trusted to already be
    there. It happens to be entirely (31, 34) at r=1.1, so nothing is dropped today, but a
    silent size mismatch would make the whole comparison meaningless rather than merely
    wrong, so it is checked instead of assumed.
    """
    ml = ml_stats[
        (ml_stats["n_nodes"] == N_NODES)
        & (ml_stats["n_edges"] == N_EDGES)
        & (np.isclose(ml_stats["r"], R_VALUE))
    ]
    if ml.empty:
        raise ValueError(
            f"No ML-driven rows at N={N_NODES}, E={N_EDGES}, r={R_VALUE}. The batch passed "
            f"as ml_stats was measured on a different search space, so the two sets of "
            f"winners are not comparable."
        )

    winners = load_ga_history(run_dirs, survivors_only=True)
    winners = winners[
        winners["generation"] == winners.groupby("run")["generation"].transform("max")
    ]

    # n_grouped and run travel with the rows so the figures can state how much simulation is
    # behind each source. 'run' is meaningless for the ML batch (those graphs came out of a
    # regressor, not a search), so it is NaN there and only ever counted on the sim side.
    columns = ["source", "category", "run", "n_grouped", "mean_steps", "prob_fixation"]
    if keep_name:
        columns.append("graph_name")

    return pd.concat(
        [
            ml.assign(source="ML-driven", run=None)[columns],
            winners.assign(
                source="simulation-driven",
            )[columns],
        ],
        ignore_index=True,
    )


def _provenance_note(frame):
    """'4 GA runs, 100K sims/graph  |  ML-driven: 100K sims/graph' for the subtitle."""
    sim = frame[frame["source"] == "simulation-driven"]
    ml = frame[frame["source"] == "ML-driven"]
    return (
        f"{sim['run'].nunique()} GA runs, {len(sim)} winners, "
        f"{_sims_note(sim['n_grouped'])} sims/graph  |  "
        f"ML-driven: {len(ml)} graphs, {_sims_note(ml['n_grouped'])} sims/graph"
    )


def plot_ml_vs_simulation(
    run_dirs, ml_stats, figures_dir=None, figsize=(14, 9), logscale=True
):
    """Measured mean_steps and prob_fixation of every winner group, both searches.

    One row per category, individual winners drawn as points and the group mean as a bar,
    so the figure answers two questions at once: how far each group actually got, and how
    tightly its ten or twenty winners agree. A group whose points are scattered across the
    whole axis did not converge on anything, which a bar chart of means alone would hide.

    The ML-driven groups are predictions that were only later measured; the
    simulation-driven groups were selected on the measurement itself. Both are plotted from
    the same measured quantities, which is the whole point: this is the only figure in the
    project where the residual predictors can be checked against ground truth.
    """
    frame = _ml_vs_simulation_frame(run_dirs, ml_stats)

    # Known categories keep their hand-picked color and anything new falls through to husl
    # rather than to a KeyError. Note that a simulation-driven category shares its color
    # with the ML-driven category it corresponds to (see colors.py) -- deliberate, since
    # the two blocks are separated and labelled, so a shared hue reads as the pairing it is.
    colors = generate_robust_color_dict(frame, CATEGORY_COLOR_DICT)

    # ML block on top, simulation block below, alphabetical within each. Rows are built
    # top-down but matplotlib's y axis runs bottom-up, so the order is reversed at the end.
    order = [
        (source, category)
        for source in ("ML-driven", "simulation-driven")
        for category in sorted(frame.loc[frame["source"] == source, "category"].unique())
    ][::-1]
    y_of = {key: i for i, key in enumerate(order)}
    split_at = sum(1 for source, _ in order if source == "simulation-driven") - 0.5

    baseline = {
        "mean_steps": float(analytic_moran_fc_fixation_time(N_NODES, R_VALUE)),
        "prob_fixation": float(analytic_moran_fc_fixation_prob(N_NODES, R_VALUE)),
    }

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    for metric, axis in zip(("mean_steps", "prob_fixation"), axes):
        for (source, category), group in frame.groupby(["source", "category"]):
            y = y_of[(source, category)]
            color = colors[category]
            values = group[metric].to_numpy(dtype=float)

            # Bars grow from the complete-graph baseline rather than from zero. The axis
            # still reads in absolute units, but the anchor is the project's usual residual
            # origin, so bar direction is amplifier-versus-suppressor at a glance. Anchoring
            # at zero would spend 60% of the probability axis on empty space below 0.08 and
            # leave every group looking the same length.
            axis.barh(
                y, values.mean() - baseline[metric], left=baseline[metric],
                height=0.62, color=color,
                alpha=0.30 if source == "ML-driven" else 0.65,
                edgecolor=color, linewidth=1.2,
            )
            # Deterministic offsets rather than random jitter: the same data must redraw
            # identically, or two saved copies of one figure look like different results.
            offsets = np.linspace(-0.16, 0.16, len(values)) if len(values) > 1 else [0.0]
            family = _selection_family(source, category)
            axis.plot(
                values, y + np.asarray(offsets), linestyle="none",
                marker=_MARKER_BY_FAMILY.get(family, _FALLBACK_MARKER),
                markersize=4.5,
                markerfacecolor=color if family == "simulation" else "none",
                markeredgecolor=color, markeredgewidth=1.1, alpha=0.9,
            )

        axis.axvline(
            baseline[metric], color="#7f7f7f", linestyle=":", linewidth=1.4, zorder=0
        )
        axis.text(
            baseline[metric], 1.005, " complete graph",
            transform=axis.get_xaxis_transform(),
            fontsize=8, color="#7f7f7f", ha="left", va="bottom",
        )
        axis.axhline(split_at, color="#333333", linewidth=1.0, linestyle="--", alpha=0.6)
        # The bars still start at the complete-graph baseline on a log axis: both of a bar's
        # endpoints are positive step counts, so a bar pointing left is drawn the same way,
        # it just no longer has a length proportional to the residual.
        label = (
            _scale_steps_axis(axis, which="x", logscale=logscale)
            if metric == "mean_steps"
            else _METRIC_LABEL[metric]
        )
        axis.set_xlabel(label, fontweight="bold")
        axis.grid(True, axis="x", linestyle=":", alpha=0.6)
        axis.set_axisbelow(True)

    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([category for _, category in order], fontsize=9)
    axes[0].set_ylim(-0.6, len(order) - 0.4)

    # Which block is which, written outside the right spine so it can never land on a bar.
    for source, span in (
        ("simulation-driven", (0, split_at)),
        ("ML-driven", (split_at, len(order) - 1)),
    ):
        axes[1].text(
            1.015, sum(span) / 2, source, transform=axes[1].get_yaxis_transform(),
            rotation=90, ha="left", va="center",
            fontsize=10, fontweight="bold", color="#333333",
        )

    marker_key = [
        plt.Line2D([], [], linestyle="none", marker=_MARKER_BY_FAMILY["LR"],
                   markerfacecolor="none", markeredgecolor="#333333",
                   label="predicted by LR"),
        plt.Line2D([], [], linestyle="none", marker=_MARKER_BY_FAMILY["XGBOOST"],
                   markerfacecolor="none", markeredgecolor="#333333",
                   label="predicted by XGBOOST"),
        plt.Line2D([], [], linestyle="none", marker=_MARKER_BY_FAMILY["simulation"],
                   color="#333333", label="measured by simulation"),
        plt.Line2D([], [], linestyle=":", color="#7f7f7f",
                   label="complete graph (bar origin)"),
    ]
    fig.legend(
        handles=marker_key, loc="lower center", ncol=4,
        bbox_to_anchor=(0.5, -0.03), frameon=False, fontsize=9,
    )
    fig.suptitle(
        "Winner groups of both searches, measured\n"
        f"bars are the group mean, markers the individual winners  |  "
        f"N={N_NODES}, E={N_EDGES}, r={R_VALUE}\n"
        f"{_provenance_note(frame)}",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 0.97, 1))

    path = _resolve_figure_path(
        figures_dir, "plot_ml_vs_simulation", n_groups=len(order)
    )
    if path:
        fig.savefig(path, bbox_inches="tight", dpi=150, facecolor="white")
        print(f"Saved: {path}")
    return fig


def plot_ml_vs_simulation_scatter(
    run_dirs, ml_stats, reference_stats=None, figures_dir=None, figsize=(12, 8), logscale=True
):
    """Fixation probability against fixation time, one point per winner graph.

    The companion to ``plot_ml_vs_simulation``, which plots each metric on its own axis and
    so cannot show how the two move together. That joint structure is the amplifier /
    suppressor question directly: whether a topology can be slow *and* likely to fixate, or
    whether pushing one drags the other along.

    Pass ``reference_stats`` (an ordinary batch's graph_statistics frame) to draw the random
    (N, E) graphs as a gray cloud behind the winners. Without that background the winners
    have nothing to be extreme *relative to*, so it is worth supplying. The respiratory
    graphs in that batch are deliberately NOT singled out here; ``plot_ga_winners_in_context``
    is the figure that places the avian graph, and this one is about the two searches.

    Fixation time is on a log axis because the winners span roughly 2.4K to 35K steps, and
    on a linear axis the four fastest groups collapse into the left margin.
    """
    frame = _ml_vs_simulation_frame(run_dirs, ml_stats)
    colors = generate_robust_color_dict(frame, CATEGORY_COLOR_DICT)

    rho_complete = float(analytic_moran_fc_fixation_prob(N_NODES, R_VALUE))
    t_complete = float(analytic_moran_fc_fixation_time(N_NODES, R_VALUE))

    fig, ax = plt.subplots(figsize=figsize)

    if reference_stats is not None:
        reference = reference_stats[
            (reference_stats["n_nodes"] == N_NODES)
            & (reference_stats["n_edges"] == N_EDGES)
            & (np.isclose(reference_stats["r"], R_VALUE))
        ]
        ax.scatter(
            reference["mean_steps"], reference["prob_fixation"],
            s=18, color="#cccccc", edgecolor="none", zorder=1,
            label=f"random ({N_NODES}, {N_EDGES}) graphs, n={len(reference)}\n"
                  f"({_sims_note(reference['n_grouped'])} sims each)",
        )

    for (source, category), group in frame.groupby(["source", "category"]):
        family = _selection_family(source, category)
        is_sim = family == "simulation"
        ax.scatter(
            group["mean_steps"], group["prob_fixation"],
            s=70 if is_sim else 55,
            marker=_MARKER_BY_FAMILY.get(family, _FALLBACK_MARKER),
            facecolor=colors[category] if is_sim else "none",
            edgecolor=colors[category],
            linewidth=1.4,
            alpha=0.9,
            zorder=5 if is_sim else 4,
            # No family prefix: the marker shape already says LR / XGBOOST / simulation,
            # and the ML category strings repeat the model name anyway.
            label=category,
        )

    # The complete graph is the residual origin used everywhere else in the project, so the
    # crosshair splits the plane into the four amplifier / suppressor quadrants.
    ax.axvline(t_complete, color="#7f7f7f", linestyle=":", linewidth=1.2, zorder=2)
    ax.axhline(rho_complete, color="#7f7f7f", linestyle=":", linewidth=1.2, zorder=2)
    ax.annotate(
        f"complete graph\nT={t_complete:.0f}, rho={rho_complete:.3f}",
        xy=(t_complete, rho_complete), xytext=(6, 6), textcoords="offset points",
        fontsize=8, color="#7f7f7f", va="bottom", ha="left",
    )

    ax.set_xlabel(_scale_steps_axis(ax, which="x", logscale=logscale), fontweight="bold")
    ax.set_ylabel(_METRIC_LABEL["prob_fixation"], fontweight="bold")
    # The marker key that used to be spelled out here (circle = LR, square = XGBOOST,
    # diamond = simulation) is dropped: the legend draws each category with its own marker
    # and the ML category strings name their model, so the sentence restated the legend.
    ax.set_title(
        "Both metrics together, every winner graph\n"
        f"N={N_NODES}, E={N_EDGES}, r={R_VALUE}  |  {_provenance_note(frame)}",
        fontsize=13, pad=12,
    )
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(
        loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8,
        frameon=False, labelspacing=0.7,
    )
    fig.tight_layout()

    path = _resolve_figure_path(
        figures_dir, "plot_ml_vs_simulation_scatter",
        reference="yes" if reference_stats is not None else "no",
    )
    if path:
        fig.savefig(path, bbox_inches="tight", dpi=150, facecolor="white")
        print(f"Saved: {path}")
    return fig


def plot_ml_vs_simulation_scatter_plotly(
    run_dirs, ml_stats, reference_stats=None, height=700, width=1100, logscale=True
):
    """Interactive twin of ``plot_ml_vs_simulation_scatter``. Returns a plotly Figure.

    Hover gives the graph name and both measured values, and clicking a legend entry
    isolates a group -- which is what the static version cannot do with 12 overlapping
    categories on one pair of axes. Same colors, same marker families, same log time axis.

    plotly is imported inside the function rather than at module scope, following
    ``plotly_prototype.py``: it is a hard dependency but a slow import, and
    ``analysis_utils/__init__`` pulls this module in eagerly, so a module-level import would
    tax every ``import analysis_utils`` including the ones that only want a reader.
    """
    import plotly.graph_objects as go

    frame = _ml_vs_simulation_frame(run_dirs, ml_stats, keep_name=True)
    colors = generate_robust_color_dict(frame, CATEGORY_COLOR_DICT)

    rho_complete = float(analytic_moran_fc_fixation_prob(N_NODES, R_VALUE))
    t_complete = float(analytic_moran_fc_fixation_time(N_NODES, R_VALUE))

    fig = go.Figure()

    # Added first so plotly paints it underneath: unlike matplotlib there is no zorder.
    if reference_stats is not None:
        reference = reference_stats[
            (reference_stats["n_nodes"] == N_NODES)
            & (reference_stats["n_edges"] == N_EDGES)
            & (np.isclose(reference_stats["r"], R_VALUE))
        ]
        fig.add_trace(
            go.Scatter(
                x=reference["mean_steps"], y=reference["prob_fixation"],
                mode="markers",
                name=f"random ({N_NODES}, {N_EDGES}), n={len(reference)}"
                     f" ({_sims_note(reference['n_grouped'])} sims each)",
                marker=dict(size=5, color="#cccccc"),
                text=reference["graph_name"],
                hovertemplate="%{text}<br>T=%{x:.0f}<br>rho=%{y:.4f}<extra></extra>",
            )
        )

    for (source, category), group in frame.groupby(["source", "category"]):
        family = _selection_family(source, category)
        is_sim = family == "simulation"
        fig.add_trace(
            go.Scatter(
                x=group["mean_steps"], y=group["prob_fixation"],
                mode="markers",
                name=category,
                marker=dict(
                    size=10 if is_sim else 9,
                    symbol=_PLOTLY_MARKER_BY_FAMILY.get(
                        family, _PLOTLY_FALLBACK_MARKER
                    ),
                    # Open markers are drawn as a transparent fill with a colored line,
                    # which is plotly's only equivalent of matplotlib's facecolor='none'.
                    color=colors[category] if is_sim else "rgba(0,0,0,0)",
                    line=dict(color=colors[category], width=1.8),
                ),
                text=group["graph_name"],
                hovertemplate=(
                    "%{text}<br>" + category + "<br>T=%{x:.0f}<br>rho=%{y:.4f}"
                    "<extra></extra>"
                ),
            )
        )

    fig.add_vline(x=t_complete, line=dict(color="#7f7f7f", width=1.2, dash="dot"))
    fig.add_hline(y=rho_complete, line=dict(color="#7f7f7f", width=1.2, dash="dot"))
    # Annotation x is in axis coordinates, and for a log axis those are log10 units. Passing
    # the raw step count on a log axis would put the label at x=3.7 steps, off the left edge.
    fig.add_annotation(
        x=np.log10(t_complete) if logscale else t_complete, y=rho_complete,
        text=f"complete graph<br>T={t_complete:.0f}, rho={rho_complete:.3f}",
        showarrow=False, xanchor="left", yanchor="bottom",
        font=dict(size=10, color="#7f7f7f"),
    )

    fig.update_layout(
        title=(
            "Both metrics together, every winner graph<br>"
            f"<sup>N={N_NODES}, E={N_EDGES}, r={R_VALUE} | "
            f"{_provenance_note(frame)}</sup>"
        ),
        # plotly's own SI format ('~s' -> '2.4k', '1M') is the equivalent of the K/M tick
        # formatter the matplotlib twin uses when the axis is linear.
        xaxis=(
            dict(title=_METRIC_LABEL["mean_steps"] + ", log scale", type="log")
            if logscale
            else dict(title=_METRIC_LABEL["mean_steps"], tickformat="~s")
        ),
        yaxis=dict(title="Fixation Probability"),
        template="plotly_white",
        height=height,
        width=width,
        legend=dict(font=dict(size=10)),
        hovermode="closest",
    )
    return fig

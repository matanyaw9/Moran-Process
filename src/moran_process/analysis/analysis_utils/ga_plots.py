"""Figures for a simulation-driven GA run (``pipeline.ga_search``).

A separate module from ``plots`` for the same reason ``plots`` was split out of the
original monolithic ``analysis_utils``: that file is already ~1600 lines, and these
figures read a different artefact (``ga_history.csv``) than everything in it.

``plot_ga_history`` is deliberately the same figure as ``plot_multi_model_history`` in
``notebooks/extreme_graphs.ipynb``, so the ML-driven and simulation-driven searches can be
put side by side. Every convention is preserved: left axis fixation time, right axis
fixation probability, solid for time and dashed for probability, a thicker line for the
metric being optimized, complete-graph baselines drawn as the residual origin, and a
sorted unified legend below the axes.

What is added is what measurement makes available and prediction did not: the spread
across survivors, and error bars at the standard error of the measurement. A predicted
fitness has no error bar, so the ML figure could never show whether its trajectory
exceeded the noise floor. This one can.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from .colors import CATEGORY_COLOR_DICT, generate_robust_color_dict
from .ga_io import final_elite_properties, load_ga_history, load_ga_state
from .plots import _resolve_figure_path
from .theory import analytic_moran_fc_fixation_prob, analytic_moran_fc_fixation_time

__all__ = [
    "plot_ga_history",
    "plot_ga_runs_comparison",
    "plot_ga_winners_in_context",
    "plot_replicate_agreement",
    "plot_selection_efficiency",
    "plot_ml_vs_simulation",
    "plot_ml_vs_simulation_scatter",
    "plot_ml_vs_simulation_scatter_plotly",
]

# The search space is fixed at avian_r4_l7's size (see pipeline.ga_search), so the
# complete-graph baselines are constants for every figure here.
N_NODES = 31
N_EDGES = 34
R_VALUE = 1.1

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
    metric = state["metric"]
    objective = state["objective"]
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

    # A weighted run optimizes a combination, so its category names the corner it
    # chases rather than "objective metric", and BOTH traces are the optimized one.
    combined = metric == "weighted"
    # From the state, which records it, with a fallback for runs written before the
    # field existed.
    category = state.get("category") or f"{objective} {metric}"
    color = CATEGORY_COLOR_DICT.get(category, "#2ca02c")
    lines = []
    for this_metric in ("mean_steps", "prob_fixation"):
        axis = ax_time if this_metric == "mean_steps" else ax_prob
        is_main = combined or this_metric == metric
        # The optimized metric takes the run's category color; the other is drawn in gray
        # so the figure never implies the search was chasing it.
        line_color = color if is_main else "#9e9e9e"
        x = trajectory["generation"]
        y = trajectory[f"{this_metric}_mean"]

        if show_spread:
            axis.fill_between(
                x, trajectory[f"{this_metric}_lo"], trajectory[f"{this_metric}_hi"],
                color=line_color, alpha=0.12, linewidth=0,
            )
        container = axis.errorbar(
            x, y, yerr=trajectory[f"{this_metric}_sem"],
            color=line_color, linestyle=_METRIC_LINESTYLE[this_metric],
            linewidth=3.5 if is_main else 1.8,
            elinewidth=0.8, capsize=0, errorevery=max(1, len(x) // 40),
            label=f"{_METRIC_LABEL[this_metric]}{' (optimized)' if is_main else ''}",
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
        + (
            f"Optimizing: {category}  |  "
            if combined
            else f"Optimizing: {objective.title()} {_METRIC_LABEL[metric]}  |  "
        )
        + f"N={N_NODES}, r={R_VALUE}, {state.get('generations', '?')} generations  |  "
        + f"{_sims_note(history['n_grouped'])} simulations per graph",
        fontsize=14, pad=15,
    )
    fig.tight_layout()

    path = _resolve_figure_path(
        figures_dir, "plot_ga_history", objective=objective, metric=metric
    )
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

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for metric, axis in zip(("mean_steps", "prob_fixation"), axes):
        for run, group in history.groupby("run"):
            category = group["category"].iloc[0]
            trajectory = _survivor_trajectory(group)
            # A weighted run optimizes a combination of both, so neither trace is
            # the incidental one; without this both would be drawn dotted and thin,
            # reading as "this run was not chasing either of these".
            run_metric = group["metric"].iloc[0]
            optimizes_this = run_metric in (metric, "weighted")
            axis.plot(
                trajectory["generation"], trajectory[f"{metric}_mean"],
                color=CATEGORY_COLOR_DICT.get(category, "#2ca02c"),
                linestyle="-" if optimizes_this else ":",
                linewidth=3.0 if optimizes_this else 1.4,
                alpha=1.0 if optimizes_this else 0.55,
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
        "Simulation-driven GA: all runs (solid = the metric that run optimized)\n"
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
    colors = generate_robust_color_dict(elites, CATEGORY_COLOR_DICT)

    # A single-replicate launch has no -repN suffix, so label those by run instead of
    # dropping them: the figure is still the right way to compare four runs side by side.
    # The shared launch prefix ("2026_07_28-long-100-gen-run-") is stripped first, since it
    # is identical on every tick and would otherwise take more width than the labels.
    # Truncated at the last separator, because commonprefix works per character and would
    # otherwise eat the shared 'm' of maximize/minimize and leave 'aximize-mean_steps'.
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
        metric = group["metric"].iloc[0]
        color = colors[category]
        offsets = np.linspace(-0.18, 0.18, len(group)) if len(group) > 1 else [0.0]
        ax_fitness.plot(
            index + np.asarray(offsets), group[metric] / group[metric].mean(),
            linestyle="none", marker="o", markersize=5,
            markerfacecolor=color, markeredgecolor=color, alpha=0.75,
        )

    # Fitness is shown relative to each group's own mean because the four objectives live
    # on scales three orders of magnitude apart. Within a category the replicates then sit
    # on a common scale, and agreement is "do the clouds line up at 1.0".
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
        metric = group["metric"].iloc[0]
        if len(group) < 3:
            # Generation 0 is the initial population alone; a spread over fewer than a
            # handful of graphs is not an estimate of anything.
            continue
        sd = float(group[metric].std())
        sem = float(group[f"{metric}_sem"].mean())
        if not np.isfinite(sd) or sd <= 0 or not np.isfinite(sem):
            continue
        rows.append({
            "run": run,
            "generation": generation,
            "category": group["category"].iloc[0],
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
    colors = generate_robust_color_dict(efficiency, CATEGORY_COLOR_DICT)

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
        category = f"{group['objective'].iloc[0]} {group['metric'].iloc[0]}"
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

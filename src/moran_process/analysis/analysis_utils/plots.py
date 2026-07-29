"""
All figure-producing functions plus the small path/stamping infrastructure
they share.

Depends on the leaf modules: ``colors`` (palette + property metadata + the
``_sort_categories`` ordering) and ``provenance`` (``_bi_get`` for the batch
title card). Nothing imports from here, so this is the top of the dependency graph.

Shared interface
----------------
Every ``plot_*`` function ends with the same keyword-only output tail (after a
bare ``*``, so these must be passed by name), in this order:

    figures_dir=None      directory to save the PNG into; None = display only
    fig_title=None        override the auto-generated title
    batch_name=None       label stamped in the bottom-right corner
    show=True             display the figure (set False to build silently)
    save=True             write the PNG (requires figures_dir; else raises)

The leading positional arguments are the data and the columns each figure
needs; per-figure options (colors, r, density, cmap, ...) sit in between.
Per-function docstrings document only those specifics and refer back here for
the shared tail.
"""

import functools
import textwrap
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter, MaxNLocator
import seaborn as sns

from .colors import DEFAULT_FIG_SIZE, GRAPH_PROPERTY_DESCRIPTION, _sort_categories
# The fixation-steps loader lives in io.py (pure polars/pandas) so the LSF cache job
# can build the violin sample without importing matplotlib/seaborn.
from .io import load_fixation_steps_by_category as _load_fixation_steps_by_category
from .provenance import _bi_get
from .theory import *

__all__ = [
    "plot_batch_info_card",
    "plot_steps_violin",
    "plot_steps_pvalue_matrix",
    "plot_steps_histogram",
    "plot_outcome_vs_property",
    "plot_two_property_effect",
    "plot_two_property_effect_hexbin",
]


def _resolve_figure_path(figures_dir, func_name: str, **key_kwargs):
    """Build a descriptive Path to save a figure to, creating the directory if needed."""
    if figures_dir is None:
        return None
    p = Path(figures_dir)
    p.mkdir(parents=True, exist_ok=True)
    slug = "__".join(f"{k}={v}" for k, v in key_kwargs.items())
    slug = slug.replace("/", "-").replace(" ", "_").replace(",", "-")
    return p / f"{func_name}__{slug}.png"


def _timed(func):
    """Print how long a figure took to build.

    This replaces the PNG cache these functions used to carry, which returned a saved
    image instead of redrawing. That cache made sense when a miss meant scanning 42GB
    inside the plot call; now every heavy input is a file read built by a job, so the
    draw itself is all that is left and knowing its cost beats skipping it.

    Skipping was also becoming unsafe: the cached filename is keyed on the function name
    and a few kwargs, never on the data, so once ``batch_dir`` accepts a *list* the same
    filename can mean different batch combinations.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        started = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            print(f"[figure] {func.__name__} built in {time.perf_counter() - started:.2f}s")

    return wrapper


def _stamp_batch(fig, batch_name: str) -> None:
    """Add a source label to the bottom-right corner of the figure."""
    fig.text(
        0.99,
        0.01,
        f"source: {batch_name}",
        fontsize=8,
        color="#666666",
        ha="right",
        va="bottom",
        style="italic",
        transform=fig.transFigure,
    )


# Style for graph-property explanations: a serif, medium-gray face that is visually
# distinct from the sans-serif axis labels so the text reads as an explanatory gloss
# rather than a label. The description is italic; the leading property name is bold and
# upright (rendered via mathtext, see _gloss_text). Dark enough (#5a5a5a) to stay
# very readable.
_DESC_FONT = {"family": "serif", "style": "italic", "color": "#5a5a5a", "fontsize": 9}
# matplotlib's math-bold defaults to a sans-serif face; force the serif math fontset so
# the inline bold name matches the serif italic description it shares a line with.
_DESC_MATH_FONT = "dejavuserif"


def _gloss_text(prop, width):
    """Inline gloss string for `prop`: bold name then italic description, wrapped.

    The name is wrapped in mathtext ($\\bf{...}$) so it renders bold *on the same line*
    as the description within a single text artist (a plain Text carries one weight).
    The whole "Name: description" is wrapped together so it flows as one paragraph.
    Returns None when `prop` has no description.
    """
    desc = GRAPH_PROPERTY_DESCRIPTION.get(prop)
    if not desc:
        return None
    name = prop.title().replace("_", " ")
    plain = textwrap.fill(f"{name}: {desc}", width=width)
    # Bold just the leading "Name:" (spaces escaped for mathtext); the rest stays italic.
    bold_name = r"$\bf{" + name.replace(" ", r"\ ") + r":}$"
    return bold_name + plain[len(name) + 1 :]


def _add_property_description(ax, prop, axis="x", width=90) -> None:
    """Render the GRAPH_PROPERTY_DESCRIPTION gloss for `prop` next to its axis.

    The property name leads in serif bold, inline with the serif italic description
    (see _DESC_FONT / _gloss_text), kept separate from the axis label. No-op when no
    description exists for `prop`.

    axis="x" places the gloss horizontally below the x-axis; axis="y" places it
    rotated 90 degrees just outside the y-axis label, so each explanation sits beside
    the axis it describes.
    """
    text = _gloss_text(prop, width)
    if text is None:
        return
    if axis == "y":
        # Sit outside (left of) the y-axis title so the gloss never overlaps it.
        ax.text(
            -0.22,
            0.5,
            text,
            transform=ax.transAxes,
            ha="center",
            va="center",
            rotation=90,
            rotation_mode="anchor",
            math_fontfamily=_DESC_MATH_FONT,
            **_DESC_FONT,
        )
    else:
        ax.text(
            0.5,
            -0.16,
            text,
            transform=ax.transAxes,
            ha="center",
            va="top",
            math_fontfamily=_DESC_MATH_FONT,
            **_DESC_FONT,
        )


def _add_property_descriptions_below(ax, props, width=90) -> None:
    """Render several property glosses stacked horizontally below the x-axis.

    An alternative to the per-axis placement of `_add_property_description`: when
    two properties are shown (one per axis), a rotated y-axis gloss can be hard to
    read, so this lays every explanation flat below the plot as one centered block.
    Props without a description are skipped; a no-op if none have one.
    """
    blocks = [t for prop in props if (t := _gloss_text(prop, width)) is not None]
    if not blocks:
        return
    ax.text(
        0.5,
        -0.16,
        "\n" + "\n".join(blocks),
        transform=ax.transAxes,
        ha="center",
        va="top",
        math_fontfamily=_DESC_MATH_FONT,
        **_DESC_FONT,
    )


def _add_corr_box(ax, text, anchor=None, default=(0.03, 0.97), fontsize=9) -> None:
    """Draw a correlation summary text box on `ax`, just below a right-side guide.

    `anchor` is an artist with a measurable extent (a category legend or a colorbar's
    axes). It sits outside the axes and has dynamic size, so its extent is measured
    after a canvas draw and mapped back to axes-fraction coordinates rather than guessing
    a fixed position. The box is placed just beneath the anchor, left-aligned to its left
    edge. Call this AFTER tight_layout so the extent reflects the final axes size.

    With no anchor, falls back to `default` (axes-fraction, inside top-left).
    """
    box = dict(
        boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor="lightgray"
    )
    if anchor is None:
        ax.text(
            *default,
            text,
            transform=ax.transAxes,
            fontsize=fontsize,
            ha="left",
            va="top",
            bbox=box,
            zorder=5,
        )
        return
    ax.figure.canvas.draw()  # realize the anchor so it has a measurable extent
    disp = anchor.get_window_extent()
    x_left, y_bottom = ax.transAxes.inverted().transform((disp.x0, disp.y0))
    ax.text(
        x_left,
        y_bottom - 0.03,
        text,
        transform=ax.transAxes,
        fontsize=fontsize,
        ha="left",
        va="top",
        bbox=box,
        zorder=5,
    )


def _safe_corr(a, b, method="pearson"):
    """Correlation of two aligned series, ignoring NaN/inf pairs.

    `method` is passed through to pandas ('pearson' or 'spearman').
    Returns NaN when fewer than two valid pairs remain or either side is
    constant (correlation is undefined there). Shared by every figure that
    annotates a correlation, so the number means the same thing everywhere.
    """
    a = pd.Series(a).replace([np.inf, -np.inf], np.nan)
    b = pd.Series(b).replace([np.inf, -np.inf], np.nan)
    mask = a.notna() & b.notna()
    if mask.sum() > 1 and a[mask].std() > 0 and b[mask].std() > 0:
        return a[mask].corr(b[mask], method=method)
    return np.nan


def _human_tick(x, _pos=None):
    """Abbreviate large axis ticks: 10000 -> '10K', 2.5e6 -> '2.5M'.

    Values below 1000 (including fractional ones like density) are left
    as-is, so the same formatter is safe across every x property.
    """
    for div, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(x) >= div:
            v = x / div
            return f"{v:.0f}{suffix}" if v == int(v) else f"{v:g}{suffix}"
    return f"{x:g}"


def _outcome_color_norm(values, *, log_dynamic_range=50.0, clip_pct=(2, 98)):
    """Pick a color normalization that keeps the dense bulk of the data visible.

    The two-property outcomes are heavy right-tailed: e.g. mean_steps spans
    ~4000x, with most graphs near the median and a few large-N graphs far above.
    A plain linear Normalize then maps ~90% of points into the bottom sliver of
    the colormap (the indistinguishable dark end). To spread them out:

    - strictly-positive outcomes whose dynamic range (max/min) exceeds
      ``log_dynamic_range`` get a LogNorm, so the median graph lands near the
      middle of the colormap and the dense cluster uses its full span;
    - everything else (bounded or signed, e.g. prob_fixation) gets a linear
      Normalize clipped to the ``clip_pct`` percentiles, so a handful of
      outliers cannot eat the whole color range.

    Returns a matplotlib Normalize/LogNorm instance shared by the main artist
    and the highlight scatter so their colors stay on one scale.
    """
    s = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return mcolors.Normalize()
    vmin, vmax = float(s.min()), float(s.max())
    if vmin > 0 and vmax / vmin > log_dynamic_range:
        return mcolors.LogNorm(vmin=vmin, vmax=vmax)
    lo, hi = (float(v) for v in np.percentile(s.to_numpy(), clip_pct))
    if lo == hi:  # degenerate after clipping (near-constant): fall back to full range
        lo, hi = vmin, vmax
    return mcolors.Normalize(vmin=lo, vmax=hi)


def _finish_two_property_figure(
    fig,
    ax,
    plot_df,
    x_prop,
    y_prop,
    outcome,
    *,
    cmap,
    norm,
    cbar_ax,
    color_dict,
    highlight_categories,
    descriptions_below,
    default_title,
    fig_title,
    batch_name,
    fig_path,
    show,
    save,
    r_value=None,
    corr="spearman",
):
    """Draw the shared tail of the two-property figures (scatter and hexbin).

    Everything after the main artist is identical between
    ``plot_two_property_effect`` and ``plot_two_property_effect_hexbin``:
    optional highlight scatter, the analytic reference lines, the
    Pearson-correlation box, axis labels and property glosses, the title, the
    legend, the batch stamp, and the save/show handling. The two callers differ
    only in their primary artist (``scatter`` vs ``hexbin``) and the default
    title, so they pass those in and delegate the rest here.

    ``r_value`` is the single selection coefficient of the data (or ``None`` when
    the data mixes several r); it is only used to draw the analytic Moran curve
    when the axes are the canonical n_nodes-vs-fixation view.
    """
    # Highlight specific categories on top, colored by the same outcome scale.
    if highlight_categories:
        hl_df = plot_df[plot_df["category"].isin(highlight_categories)]
        if not hl_df.empty:
            for cat, grp in hl_df.groupby("category"):
                ax.scatter(
                    grp[x_prop],
                    grp[y_prop],
                    c=grp[outcome],
                    norm=norm,
                    cmap=cmap,
                    s=120,
                    linewidths=1.8,
                    edgecolors=color_dict.get(cat, "black"),
                    zorder=3,
                    label=cat,
                )

    # Analytic reference lines: only meaningful when x encodes N and y is the
    # fixation probability. Mirrors the curves drawn in plot_outcome_vs_property.
    ref_lines_drawn = False
    if x_prop == "n_nodes" and y_prop == "prob_fixation":
        n_col = plot_df["n_nodes"].dropna()
        if len(n_col) > 0:
            x_range = np.linspace(max(1, n_col.min()), n_col.max(), 300)
            # Neutral drift baseline y = 1/N (independent of r).
            ax.plot(
                x_range,
                1.0 / x_range,
                color="black",
                linestyle="--",
                linewidth=1.4,
                label=r"Neutral  $1/N$",
                zorder=4,
            )
            # Complete-graph Moran fixation probability rho(N, r); needs one r.
            if r_value is not None:
                ax.plot(
                    x_range,
                    analytic_moran_fc_fixation_prob(x_range, r_value),
                    color="tab:red",
                    linestyle="--",
                    linewidth=1.4,
                    label=r"Moran  $\rho=\frac{1-1/r}{1-1/r^{N}}$",
                    zorder=4,
                )
            ref_lines_drawn = True

    # Correlation box: corr is None / 'spearman' / 'pearson', matching
    # plot_outcome_vs_property. Each property is correlated against the outcome.
    corr_text = None
    if corr:
        corr_x = _safe_corr(plot_df[x_prop], plot_df[outcome], method=corr)
        corr_y = _safe_corr(plot_df[y_prop], plot_df[outcome], method=corr)
        corr_text = (
            f"{corr.capitalize()} corr with {outcome.replace('_', ' ')}\n"
            + "-" * 30
            + "\n"
            + f"{x_prop}: {corr_x:.3f}\n"
            + f"{y_prop}: {corr_y:.3f}"
        )

    ax.set_xlabel(x_prop.replace("_", " ").title(), fontsize=12)
    ax.set_ylabel(y_prop.replace("_", " ").title(), fontsize=12)
    if descriptions_below:
        _add_property_descriptions_below(ax, [x_prop, y_prop])
    else:
        _add_property_description(ax, x_prop, axis="x")
        _add_property_description(ax, y_prop, axis="y", width=60)
    ax.set_title(fig_title or default_title, fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.4)

    # One legend covering whatever labeled artists exist (category highlights
    # and/or the analytic reference lines). Title only reads "Category" when the
    # only labeled artists are highlights. Its final position is set after layout
    # (below the colorbar); the anchor here is just a sensible pre-layout value.
    if highlight_categories or ref_lines_drawn:
        legend_title = (
            "Category" if highlight_categories and not ref_lines_drawn else None
        )
        legend = ax.legend(
            title=legend_title, bbox_to_anchor=(1.02, 1.0), loc="upper left"
        )
    else:
        legend = None

    if batch_name:
        _stamp_batch(fig, batch_name)
    fig.tight_layout()

    # Stack the three right-margin guides top-to-bottom: colorbar, then legend,
    # then correlation box. This only needs rearranging when a legend exists;
    # otherwise the colorbar keeps its full height and the box sits beneath it.
    if legend is not None and cbar_ax is not None:
        fig.canvas.draw()  # realize positions before measuring/repositioning
        # Confine the colorbar to its top ~55% so there is room below it.
        pos = cbar_ax.get_position()
        new_h = pos.height * 0.55
        cbar_ax.set_position([pos.x0, pos.y0 + pos.height - new_h, pos.width, new_h])
        fig.canvas.draw()
        # Anchor the legend just below the colorbar, left-aligned to it.
        cb = cbar_ax.get_window_extent()
        cb_left, cb_bottom = ax.transAxes.inverted().transform((cb.x0, cb.y0))
        legend.set_bbox_to_anchor((cb_left, cb_bottom - 0.04), transform=ax.transAxes)

    # Under the legend when present (which now sits under the colorbar), else
    # directly under the colorbar. Skipped entirely when corr is None.
    if corr_text is not None:
        _add_corr_box(ax, corr_text, anchor=legend if legend is not None else cbar_ax)
    if fig_path is not None and save:
        fig.savefig(fig_path, bbox_inches="tight", dpi=150)
        print(f"[figure] Saved: {fig_path.name}")
    if show:
        plt.show()


@_timed
def plot_batch_info_card(
    batch_info,
    *,
    figures_dir=None,
    show=True,
    save=True,
):
    """Generate a standalone title-card figure for a batch, suitable as a first/catalog slide.

    Args:
        batch_info: dict returned by load_batch_info() or create_batch_info()
        figures_dir: directory where PNG is saved; None = display only, no save
        show: set False to build the figure without displaying it
        save: set False to skip writing the PNG even when figures_dir is given
    """
    fig_path = _resolve_figure_path(figures_dir, "batch_info_card")

    # Read fields from the nested batch_info (with flat fallback for legacy files).
    name = batch_info.get("name", "Unknown Batch")
    description = batch_info.get("description", "")
    notes = batch_info.get("notes", "")
    created_at = _bi_get(batch_info, "created_at") or _bi_get(
        batch_info, "date_created", default=""
    )
    graph_types = _bi_get(batch_info, "zoo", "graph_types", default=[])
    node_sizes = _bi_get(batch_info, "zoo", "node_sizes", default=[])
    n_graphs = _bi_get(batch_info, "zoo", "n_graphs")
    r_values = _bi_get(batch_info, "simulation", "r_values", default=[])
    n_repeats = _bi_get(batch_info, "simulation", "n_repeats")
    total_simulations = _bi_get(batch_info, "simulation", "total_simulations")
    engine = _bi_get(batch_info, "simulation", "engine")
    n_requested_jobs = _bi_get(batch_info, "hpc", "n_requested_jobs")
    queue = _bi_get(batch_info, "hpc", "queue")
    memory_mb = _bi_get(batch_info, "hpc", "memory_mb")
    lsf_job_id = _bi_get(batch_info, "hpc", "lsf_job_id")
    git_commit = _bi_get(batch_info, "provenance", "git_commit")
    git_branch = _bi_get(batch_info, "provenance", "git_branch")
    git_dirty = _bi_get(batch_info, "provenance", "git_dirty")
    hostname = _bi_get(batch_info, "provenance", "hostname")

    # 16:9 canvas so the card drops straight onto a widescreen slide.
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Title
    ax.text(
        0.05,
        0.93,
        name,
        transform=ax.transAxes,
        fontsize=30,
        fontweight="bold",
        va="top",
        ha="left",
        color="#1a1a1a",
    )

    # Subtitle: date + engine (muted, just under the title)
    subtitle_bits = []
    if created_at:
        subtitle_bits.append(str(created_at).replace("T", "  "))
    if engine:
        subtitle_bits.append(f"{engine} engine")
    if subtitle_bits:
        ax.text(
            0.05,
            0.845,
            "   ·   ".join(subtitle_bits),
            transform=ax.transAxes,
            fontsize=13,
            va="top",
            ha="left",
            color="#888888",
        )

    # Horizontal rule under the title block
    ax.plot(
        [0.04, 0.96],
        [0.80, 0.80],
        transform=ax.transAxes,
        color="#cccccc",
        linewidth=1.2,
        solid_capstyle="butt",
    )

    # Description
    if description:
        wrapped = textwrap.fill(description, width=95)
        ax.text(
            0.05,
            0.74,
            wrapped,
            transform=ax.transAxes,
            fontsize=14,
            va="top",
            ha="left",
            color="#333333",
            style="italic",
            linespacing=1.5,
        )

    # Dense grouped metadata rows: label on the left, a single packed value line.
    def _meta_row(label, value, y):
        ax.text(
            0.05,
            y,
            label,
            transform=ax.transAxes,
            fontsize=13,
            va="top",
            ha="left",
            fontweight="bold",
            color="#444444",
        )
        ax.text(
            0.20,
            y,
            value,
            transform=ax.transAxes,
            fontsize=13,
            va="top",
            ha="left",
            color="#222222",
        )

    def _join(parts):
        return "      ".join(p for p in parts if p)

    y = 0.56
    row_h = 0.105

    zoo_parts = []
    if n_graphs is not None:
        zoo_parts.append(f"{int(n_graphs):,} graphs")
    if graph_types:
        zoo_parts.append(f'types: {", ".join(graph_types)}')
    if node_sizes:
        zoo_parts.append(f'sizes: {", ".join(str(n) for n in node_sizes)}')
    if zoo_parts:
        _meta_row("Zoo", _join(zoo_parts), y)
        y -= row_h

    sim_parts = []
    if r_values:
        sim_parts.append(f'r = {", ".join(str(r) for r in r_values)}')
    if n_repeats is not None:
        sim_parts.append(f"{int(n_repeats):,} reps/config")
    if total_simulations is not None:
        sim_parts.append(f"{int(total_simulations):,} total sims")
    if sim_parts:
        _meta_row("Simulation", _join(sim_parts), y)
        y -= row_h

    hpc_parts = []
    if n_requested_jobs is not None:
        hpc_parts.append(f"{int(n_requested_jobs):,} jobs")
    if queue:
        hpc_parts.append(f"queue: {queue}")
    if memory_mb is not None:
        hpc_parts.append(f"{int(memory_mb):,} MB/job")
    if lsf_job_id:
        hpc_parts.append(f"job {lsf_job_id}")
    if hpc_parts:
        _meta_row("HPC", _join(hpc_parts), y)
        y -= row_h

    # Provenance + notes footer (muted, bottom of the slide)
    footer_bits = []
    if git_commit:
        commit = f"commit {git_commit[:8]}"
        if git_branch:
            commit += f" ({git_branch})"
        if git_dirty:
            commit += " +dirty"
        footer_bits.append(commit)
    if hostname:
        footer_bits.append(hostname)
    if footer_bits:
        ax.text(
            0.05,
            0.13,
            "   ·   ".join(footer_bits),
            transform=ax.transAxes,
            fontsize=10,
            va="top",
            ha="left",
            color="#aaaaaa",
        )
    if notes:
        ax.text(
            0.05,
            0.07,
            textwrap.fill(f"Notes: {notes}", width=110),
            transform=ax.transAxes,
            fontsize=10,
            va="top",
            ha="left",
            color="#999999",
            style="italic",
        )

    fig.tight_layout()
    if fig_path is not None and save:
        fig.savefig(fig_path, bbox_inches="tight", dpi=200, facecolor="white")
        print(f"[figure] Saved: {fig_path.name}")
    if show:
        plt.show()


@_timed
def plot_steps_violin(
    batch_dir,
    df_graphs,
    color_dict=None,
    categories=None,
    r=None,
    max_points_per_category=50_000,
    *,
    figures_dir=None,
    fig_title=None,
    batch_name=None,
    show=True,
    save=True,
):
    """Violin plot of steps-to-fixation distribution, one violin per graph category.

    A violin is a kernel density estimate, and KDE cost is linear in the number of
    points. The raw batch can hold tens of millions of fixation events, which makes
    seaborn spend many minutes building the KDEs. Two cheap measures keep it fast
    without changing the picture: only the fixation rows are materialised (non-fixation
    rows are never drawn), and each category is subsampled to ``max_points_per_category``
    points before the KDE (a 50k sample is visually identical to the full distribution).

    The sample is read from the batch's violin cache, built once by the post-batch job
    ``pipeline.cache_violin_data``. If it is missing this raises with the command to
    build it rather than silently starting a multi-GB scan.

    Args:
        batch_dir: the batch directory; the cached sample lives in <batch_dir>/cache/
        df_graphs: DataFrame with at least 'wl_hash' and 'category' columns
        color_dict: category -> hex color mapping for violin fills
        categories: x-axis order; defaults to sorted unique values in df_graphs['category']
        r: which selection coefficient to plot (violins show one r at a time). If None
            and exactly one r is cached at this cap, that value is used.
        max_points_per_category: the cap the cache was built with; part of the cache key.
            Default 50_000.

    See the module docstring for the shared output tail (figures_dir,
    fig_title, batch_name, show, save).
    """
    if save and not figures_dir:
        raise ValueError("figures_dir must be provided if save=True")
    if color_dict is None:
        color_dict = {}

    fig_path = _resolve_figure_path(figures_dir, "plot_steps_violin")

    if categories is None:
        categories = _sort_categories(df_graphs["category"].dropna().unique().tolist())

    merged_raw, fixation_counts, total_counts, r, r_suffix, subsampled = (
        _load_fixation_steps_by_category(
            batch_dir,
            r=r,
            max_points_per_category=max_points_per_category,
        )
    )

    # The loader left-joins every graph, so merged_raw carries categories outside the
    # requested set (e.g. 'Grid'). Because hue='category' equals x='category', seaborn
    # maps hue to every value present in the data and demands a palette key for each,
    # ignoring `order`. Drop out-of-filter rows so hue levels stay a subset of palette.
    merged_raw = merged_raw[merged_raw["category"].isin(categories)]

    # seaborn needs a palette entry for every hue level; fill any category the
    # caller did not color with a distinct husl fallback so a partial color_dict
    # never crashes the plot (same guard plot_outcome_vs_property uses).
    palette = {cat: color_dict[cat] for cat in categories if cat in color_dict}
    missing_cats = [c for c in categories if c not in palette]
    if missing_cats:
        palette.update(zip(missing_cats, sns.color_palette("husl", len(missing_cats))))

    fig, ax = plt.subplots(figsize=(max(12, len(categories) * 1.1), 7))
    sns.violinplot(
        data=merged_raw,
        x="category",
        y="steps",
        order=categories,
        hue="category",
        palette=palette,
        legend=False,
        inner="box",
        linewidth=1.2,
        ax=ax,
    )
    fig_title = fig_title or f"Distribution of Steps to Fixation by Category{r_suffix}"

    # Annotate each violin with its fixation probability and the raw fixation count
    # (n) on a second label line. rho = fixations / total runs makes clear that an
    # unequal n reflects a different success rate, not a different number of runs.
    def _violin_label(cat):
        fix = fixation_counts.get(cat, 0)
        tot = total_counts.get(cat, 0)
        rho = fix / tot if tot else 0.0
        return f"{cat}\nρ = {rho:.3f}  (n = {fix:,})"

    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(
        [_violin_label(cat) for cat in categories],
        rotation=45,
        ha="right",
        fontsize=10,
    )
    ax.set_xlabel("Category", fontsize=13)
    ax.set_ylabel("Steps to Fixation", fontsize=13)
    ax.set_title(fig_title, fontsize=14)
    if batch_name:
        _stamp_batch(fig, batch_name)
    if subsampled:
        fig.text(
            0.01,
            0.01,
            f"violins drawn from a random subsample of {max_points_per_category:,} points/category",
            fontsize=8,
            color="#666666",
            ha="left",
            va="bottom",
            style="italic",
            transform=fig.transFigure,
        )
    fig.tight_layout()
    if fig_path is not None and save:
        fig.savefig(fig_path, bbox_inches="tight", dpi=150)
        print(f"[figure] Saved: {fig_path.name}")
    if show:
        plt.show()


def _significance_stars(p):
    """Conventional significance markers for a (corrected) p-value."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


@_timed
def plot_steps_pvalue_matrix(
    batch_dir,
    df_graphs,
    categories=None,
    r=None,
    max_points_per_category=50_000,
    *,
    figures_dir=None,
    fig_title=None,
    batch_name=None,
    show=True,
    save=True,
):
    """Pairwise significance matrix for the steps-to-fixation violins.

    Companion to ``plot_steps_violin``: for every pair of categories it runs a
    two-sided Mann-Whitney U test on the steps-to-fixation distributions and shows
    two things per cell:

    - **cell color**: rank-biserial correlation, an effect size in [-1, 1] that is
      (in expectation) independent of sample size. 0 means the two categories'
      steps are interchangeable; +-1 means total separation. Sign is row-vs-column:
      positive (red) means the *row* category fixes slower (larger steps) than the
      *column* category; negative (blue) means faster.
    - **cell text**: significance stars from the Bonferroni-corrected p-value
      (*** < 0.001, ** < 0.01, * < 0.05, ns otherwise), with the effect size above.

    Why both: after subsampling each category still holds tens of thousands of
    events (millions before), so almost every pair is "significant" and a bare
    p-value matrix would be near-uniformly tiny. The effect size tells you which
    differences are large enough to matter; the stars tell you which survive
    multiple-comparison correction. Mann-Whitney is computed on the same subsample
    the violins are drawn from, so the two figures agree; the effect size is stable
    under subsampling while the p-value reflects the subsample size.

    Args mirror ``plot_steps_violin`` (no ``color_dict``: cells are colored by
    effect size, not by category). See the module docstring for the shared
    output tail (figures_dir, fig_title, batch_name, show, save).
    """
    from scipy.stats import mannwhitneyu

    if save and not figures_dir:
        raise ValueError("figures_dir must be provided if save=True")

    fig_path = _resolve_figure_path(figures_dir, "plot_steps_pvalue_matrix")

    if categories is None:
        categories = _sort_categories(df_graphs["category"].dropna().unique().tolist())

    merged, fixation_counts, _total_counts, r, r_suffix, subsampled = (
        _load_fixation_steps_by_category(
            batch_dir,
            r=r,
            max_points_per_category=max_points_per_category,
        )
    )

    # Only categories with fixation data can be tested.
    categories = [c for c in categories if fixation_counts.get(c, 0) > 0]
    if len(categories) < 2:
        print("[skip] need at least two categories with fixation events to compare")
        return

    groups = {
        c: merged.loc[merged["category"] == c, "steps"].to_numpy() for c in categories
    }

    k = len(categories)
    n_pairs = k * (k - 1) // 2
    effect = np.full((k, k), np.nan)  # rank-biserial, antisymmetric
    annot = np.full((k, k), "", dtype=object)
    for i in range(k):
        for j in range(i + 1, k):
            a, b = groups[categories[i]], groups[categories[j]]
            u, p = mannwhitneyu(a, b, alternative="two-sided")
            # Common-language effect P(row > col) = U/(n_a*n_b); rank-biserial = 2*P - 1.
            rb = 2.0 * u / (len(a) * len(b)) - 1.0
            p_corr = min(p * n_pairs, 1.0)  # Bonferroni over all pairs
            stars = _significance_stars(p_corr)
            # effect is antisymmetric, so each triangle's label carries its own sign.
            effect[i, j], effect[j, i] = rb, -rb
            annot[i, j] = f"{rb:+.2f}\n{stars}"
            annot[j, i] = f"{-rb:+.2f}\n{stars}"

    fig, ax = plt.subplots(figsize=(max(8, k * 0.95 + 2), max(6, k * 0.85 + 2)))
    sns.heatmap(
        effect,
        mask=np.eye(k, dtype=bool),
        annot=annot,
        fmt="",
        cmap="coolwarm",
        center=0.0,
        vmin=-1.0,
        vmax=1.0,
        square=True,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "rank-biserial effect (row slower → +1)"},
        annot_kws={"fontsize": 8},
        xticklabels=categories,
        yticklabels=categories,
        ax=ax,
    )
    fig_title = fig_title or f"Pairwise steps-to-fixation significance{r_suffix}"
    ax.set_title(fig_title, fontsize=14)
    ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(categories, rotation=0, fontsize=9)
    if batch_name:
        _stamp_batch(fig, batch_name)
    cap_note = (
        f"  n capped at {max_points_per_category:,}/category"
        if max_points_per_category is not None
        else ""
    )
    fig.text(
        0.01,
        0.01,
        "color = signed rank-biserial effect; stars = Bonferroni Mann-Whitney p "
        f"(*** <0.001, ** <0.01, * <0.05).{cap_note}",
        fontsize=8,
        color="#666666",
        ha="left",
        va="bottom",
        style="italic",
        transform=fig.transFigure,
    )
    fig.tight_layout()
    if fig_path is not None and save:
        fig.savefig(fig_path, bbox_inches="tight", dpi=150)
        print(f"[figure] Saved: {fig_path.name}")
    if show:
        plt.show()


@_timed
def plot_steps_histogram(
    df,
    metric="mean_steps",
    category=None,
    color_dict=None,
    bins=50,
    *,
    figures_dir=None,
    fig_title=None,
    batch_name=None,
    show=True,
    save=True,
):
    """Histogram of a steps/outcome metric, optionally filtered to one graph category.

    Args:
        df: aggregated graph-statistics DataFrame (output of the groupby/merge block)
        metric: column to histogram, e.g. 'mean_steps', 'prob_fixation'
        category: if given, restricts to df['category'] == category; None plots all graphs
        color_dict: category -> hex color; the category's color is used for the bars when set
        bins: number of histogram bins

    See the module docstring for the shared output tail (figures_dir,
    fig_title, batch_name, show, save).
    """
    if save and not figures_dir:
        raise ValueError("figures_dir must be provided if save=True")
    if color_dict is None:
        color_dict = {}

    r_vals = sorted(df["r"].dropna().unique().tolist()) if "r" in df.columns else []
    r_suffix = f"  (r={r_vals[0]})" if len(r_vals) == 1 else ""
    # Unlike the violin/matrix loaders, the histogram does not resolve to a single
    # r; with several r values present it pools them into one distribution. Warn so
    # that pooling is never silent (the blank r_suffix is the only other hint).
    if len(r_vals) > 1:
        print(
            f"[plot_steps_histogram] pooling {len(r_vals)} r values {r_vals} into one "
            f"histogram; pass a single-r df to separate them"
        )

    cat_key = category or "all"
    fig_path = _resolve_figure_path(
        figures_dir, "plot_steps_histogram", metric=metric, category=cat_key
    )

    plot_df = df if category is None else df.loc[df["category"] == category]
    data = plot_df[metric].dropna()

    if data.empty:
        print(
            f"[plot_steps_histogram] No data for metric={metric!r}, category={category!r}"
        )
        return

    label = category or "All Graphs"
    metric_label = metric.replace("_", " ").title()
    bar_color = color_dict.get(category, "#4c72b0") if category else "#4c72b0"

    fig, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)
    ax.hist(data, bins=bins, color=bar_color, edgecolor="black", alpha=0.7)
    ax.set_xlabel(metric_label, fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(
        fig_title or f"Distribution of {metric_label} - {label}{r_suffix}", fontsize=14
    )
    ax.grid(axis="y", alpha=0.3)

    if batch_name:
        _stamp_batch(fig, batch_name)
    fig.tight_layout()
    if fig_path is not None and save:
        fig.savefig(fig_path, bbox_inches="tight", dpi=150)
        print(f"[figure] Saved: {fig_path.name}")
    if show:
        plt.show()


@_timed
def plot_outcome_vs_property(
    df,
    x_prop,
    y_outcome="prob_fixation",
    color_dict=None,
    density_threshold=50,
    highlight_categories=None,
    filter_categories=None,
    size_property=None,
    *,
    figures_dir=None,
    fig_title=None,
    batch_name=None,
    corr="spearman",
    show=True,
    save=True,
):
    """Scatter plot of one graph property vs an evolutionary outcome, with auto-detected violins.

    Violins are drawn automatically for discrete x values that have >= density_threshold points.
    The 1/N neutral drift line is drawn as a curve when x_prop='n_nodes', as a flat line when
    N is homogeneous (CV < 5%), and omitted when N varies widely.

    Args:
        df: aggregated graph-statistics DataFrame (one row per graph per r value)
        x_prop: structural property column for the x-axis (e.g. 'n_nodes', 'avg_degree')
        y_outcome: outcome column for the y-axis; 'prob_fixation' triggers neutral-line logic
        color_dict: category -> hex color mapping
        density_threshold: min points at an x position before a violin is drawn (default 50). If None, no violins will be drawn.
        highlight_categories: categories that, if present in the data, are drawn with
            black outlines on top of the scatter; absent categories are simply ignored
        filter_categories: if given, restrict the plot to these categories only
            (affects correlation, scatter, and neutral line); None = use all categories
        size_property: column name to encode as marker size; None = uniform size

    See the module docstring for the shared output tail (figures_dir,
    fig_title, batch_name, show, save).
    """
    if save and not figures_dir:
        raise ValueError("figures_dir must be provided if save=True")
    if color_dict is None:
        color_dict = {}

    # filter_categories distinguishes the cached figure so a filtered plot does not
    # clobber the unfiltered one (and vice versa).
    cache_key = dict(x=x_prop, y=y_outcome)
    if filter_categories is not None:
        cache_key["cats"] = "-".join(map(str, filter_categories))
    fig_path = _resolve_figure_path(
        figures_dir, "plot_outcome_vs_property", **cache_key
    )

    if filter_categories is not None:
        df = df[df["category"].isin(filter_categories)]

    # --- 1. Labels ---
    prob_label = "Fixation Probability ($P_{fix}$)"
    is_prob = y_outcome == "prob_fixation"
    ylabel = prob_label if is_prob else y_outcome.replace("_", " ").title()

    if x_prop == "prob_fixation":
        xlabel_base = prob_label
    elif x_prop == "std_steps":
        xlabel_base = "Std. Steps to Fixation"
    else:
        xlabel_base = x_prop.replace("_", " ").title()

    # The property gloss is rendered separately (see _add_property_description) so it
    # can carry its own muted, distinct font; the axis label stays clean.
    xlabel = xlabel_base

    # --- 2. Correlation (per r value, compactly); corr is None / 'spearman' / 'pearson' ---
    r_values = sorted(df["r"].dropna().unique()) if "r" in df.columns else []
    stats_text = None
    if corr:
        cols_for_corr = [x_prop, y_outcome] + (["r"] if r_values else [])
        clean_df = df[cols_for_corr].replace([np.inf, -np.inf], np.nan).dropna()
        header = f"{corr.capitalize()} corr"

        if len(r_values) > 1:
            corr_lines = [header, "-" * 18]
            for rv in r_values:
                sub = clean_df[clean_df["r"] == rv]
                c = _safe_corr(sub[x_prop], sub[y_outcome], method=corr)
                corr_lines.append(f"r={rv}: {c:.3f}" if pd.notna(c) else f"r={rv}: N/A")
        else:
            c = _safe_corr(clean_df[x_prop], clean_df[y_outcome], method=corr)
            corr_lines = [header, f"{c:.3f}" if pd.notna(c) else "N/A"]
        stats_text = "\n".join(corr_lines)

    # --- 3. X-axis processing ---
    plot_df = df.copy()
    is_numeric_x = pd.api.types.is_numeric_dtype(plot_df[x_prop])

    if is_numeric_x:
        plot_df[x_prop] = pd.to_numeric(plot_df[x_prop], errors="coerce")
        plot_df["x_plot"] = plot_df[x_prop].round(3)
        unique_cats = None
    else:
        plot_df = plot_df.dropna(subset=[x_prop])
        unique_cats = sorted(plot_df[x_prop].unique())
        plot_df["x_plot"] = plot_df[x_prop].map(
            {v: i for i, v in enumerate(unique_cats)}
        )

    # --- 4. Auto-detect discrete x (few unique values relative to data size) ---
    valid_x = plot_df["x_plot"].dropna()
    n_unique = valid_x.nunique()
    n_total = len(valid_x)
    is_discrete_x = is_numeric_x and (n_unique <= max(20, n_total * 0.02))

    # --- 5. Figure ---
    fig, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)

    dense_x_values = set()
    if is_discrete_x and density_threshold is not None:
        counts = valid_x.value_counts()
        dense_x_values = set(counts[counts >= density_threshold].index)

        # Violin width: Smart proportional calculation
        all_x_sorted = sorted(valid_x.unique())
        if len(all_x_sorted) > 1:
            diffs = np.diff(all_x_sorted)
            total_span = all_x_sorted[-1] - all_x_sorted[0]
            if total_span == 0:
                total_span = 1.0

            # Filter out tiny sub-gaps (threshold: 2% of total span)
            min_valid_gap_threshold = total_span * 0.02
            valid_gaps = diffs[diffs > min_valid_gap_threshold]

            if len(valid_gaps) > 0:
                dist_basis = np.min(valid_gaps)
            else:
                dist_basis = total_span * 0.1

            violin_width = dist_basis * 0.7
        else:
            violin_width = 0.5

        for x_val in dense_x_values:
            subset = plot_df.loc[plot_df["x_plot"] == x_val, y_outcome].dropna()
            if len(subset) > 0:
                parts = ax.violinplot(
                    subset,
                    positions=[x_val],
                    widths=violin_width,
                    showmeans=False,
                    showextrema=False,
                )
                for pc in parts["bodies"]:
                    pc.set_facecolor("whitesmoke")
                    pc.set_edgecolor("lightgray")
                    pc.set_alpha(1.0)

        # Vectorized jitter -- much faster than apply(func, axis=1).
        # Seeded local RNG so the same data always jitters to the same x-offsets
        # (a fresh np.random draw would shift dots on every regeneration). Seed 0
        # matches the deterministic-plot convention used for the violin subsample.
        mask = plot_df["x_plot"].isin(dense_x_values) & plot_df["x_plot"].notna()
        jitter_half = violin_width * 0.15
        plot_df["x_jittered"] = plot_df["x_plot"].copy().astype(float)
        if mask.any():
            jitter_rng = np.random.default_rng(0)
            plot_df.loc[mask, "x_jittered"] = plot_df.loc[
                mask, "x_plot"
            ] + jitter_rng.uniform(-jitter_half, jitter_half, size=int(mask.sum()))
    else:
        plot_df["x_jittered"] = plot_df["x_plot"]

    # --- 6. Scatter (background) ---
    hue_order = _sort_categories(plot_df["category"].dropna().unique().tolist())
    # Draw order is the REVERSE of legend order. matplotlib paints last-on-top, and
    # _sort_categories puts 'Random' last (the right reading order for the legend),
    # which would paint the large, pale Random cloud OVER the biological categories.
    # Drawing in reverse lands Random in the back; the legend is re-sorted to
    # hue_order in step 12, so its reading order is unaffected.
    draw_order = list(reversed(hue_order))
    # seaborn requires a dict palette to cover every hue level. Keep the caller's
    # colors and fill any uncolored category with a distinct fallback so the plot
    # never crashes on a missing/partial color_dict.
    palette = dict(color_dict)
    missing_cats = [c for c in hue_order if c not in palette]
    if missing_cats:
        palette.update(zip(missing_cats, sns.color_palette("husl", len(missing_cats))))
    # Fixed dot size, applied only when not encoding a column as size (else seaborn's
    # size/sizes mapping owns 's' and passing both raises).
    base_dot_size = 55
    fixed_size = {"s": base_dot_size} if size_property is None else {}
    sns.scatterplot(
        data=plot_df,
        ax=ax,
        x="x_jittered",
        y=y_outcome,
        hue="category",
        hue_order=draw_order,
        style="r" if len(r_values) > 1 else None,
        size=size_property,
        sizes=(20, 100),
        palette=palette,
        alpha=0.7,
        edgecolor="w",
        linewidth=0.5,
        zorder=2,
        **fixed_size,
    )

    # --- 7. Highlighted categories (foreground) ---
    if highlight_categories:
        hl_df = plot_df[plot_df["category"].isin(highlight_categories)]
        if not hl_df.empty:
            sns.scatterplot(
                data=hl_df,
                ax=ax,
                x="x_jittered",
                y=y_outcome,
                hue="category",
                hue_order=hue_order,
                style="r" if len(r_values) > 1 else None,
                size=size_property,
                sizes=(20, 100),
                palette=palette,
                # 1.3 pt matches the Plotly version's 1.8 px edge at inline DPI (100):
                # matplotlib linewidth is in points, Plotly's is in pixels.
                alpha=1.0,
                edgecolor="black",
                linewidth=0.8,
                legend=False,
                zorder=3,
                **fixed_size,
            )

    # --- 8. Neutral 1/N reference line ---
    if is_prob and "n_nodes" in plot_df.columns:
        n_col = plot_df["n_nodes"].dropna()
        if len(n_col) > 0:
            if x_prop == "n_nodes":
                # x encodes N directly: draw the theoretical y = 1/x curve
                x_range = np.linspace(max(1, n_col.min()), n_col.max(), 300)
                # Neutral drift baseline: y = 1/N (independent of r). zorder 4 keeps it
                # above both dot layers (background 2, highlights 3).
                ax.plot(
                    x_range,
                    1.0 / x_range,
                    color="black",
                    linestyle="--",
                    linewidth=1.2,
                    label=r"Neutral  $1/N$",
                    zorder=4,
                )
                # Analytic complete-graph fixation probability rho(N, r). It depends
                # on a single r, so only draw it when the data has exactly one r.
                if len(r_values) == 1:
                    r = r_values[0]
                    ax.plot(
                        x_range,
                        analytic_moran_fc_fixation_prob(x_range, r),
                        color="tab:blue",
                        linestyle="--",
                        linewidth=1.2,
                        label=r"Moran  $\rho=\frac{1-1/r}{1-1/r^{N}}$",
                        zorder=1,
                    )
            else:
                # Only draw a flat line when N is homogeneous (CV < 5%)
                n_mean = n_col.mean()
                n_cv = n_col.std() / n_mean if n_mean > 0 else 1.0
                if n_cv < 0.05:
                    if len(r_values) == 1:
                        # The baseline that matters is the COMPLETE-graph rho(N, r):
                        # amplifier/suppressor is defined against it, and it sits inside
                        # the data. Neutral 1/N is the r=1 baseline, so at r>1 it lands
                        # far below every point (0.033 vs ~0.10 at N=30, r=1.1) and
                        # stretched the autoscaled y axis over probabilities that never
                        # occur. rho collapses to exactly 1/N at r=1, so that case still
                        # draws the neutral line, just via the general formula.
                        rv = r_values[0]
                        base = float(analytic_moran_fc_fixation_prob(n_mean, rv))
                        label = (
                            rf"Moran  $\rho$(N={n_mean:.0f}, r={rv:g})={base:.4f}"
                            if rv != 1
                            else f"Neutral (1/N={n_mean:.0f})"
                        )
                        color = "tab:blue" if rv != 1 else "black"
                    else:
                        # rho depends on r, so several r values have no single line.
                        # 1/N is the only r-independent reference left.
                        base = 1.0 / n_mean
                        label = f"Neutral (1/N={n_mean:.0f})"
                        color = "black"
                    ax.axhline(
                        base,
                        color=color,
                        linestyle=":",
                        linewidth=1.0,
                        label=label,
                        zorder=4,
                    )
                # else: N varies too much -- a flat line would be misleading, so skip

    # --- 9. Categorical x-axis ticks ---
    if not is_numeric_x and unique_cats is not None:
        ax.set_xticks(range(len(unique_cats)))
        ax.set_xticklabels(unique_cats)
    elif is_numeric_x:
        # Human-readable numeric ticks (10000 -> '10K'); harmless for small/fractional x.
        ax.xaxis.set_major_formatter(FuncFormatter(_human_tick))
        # If the property is integer-valued (e.g. n_nodes, n_edges, diameter), pin the
        # locator to integers so matplotlib never invents fractional ticks like 29.5.
        # Detected from the data, not the name, so it generalises to any integer column
        # while leaving genuinely fractional ones (density, centralities) alone.
        _xv = plot_df[x_prop].dropna()
        if len(_xv) and np.all(_xv == _xv.round()):
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # --- 10. Titles & labels ---
    r_suffix = f"  (r={r_values[0]})" if len(r_values) == 1 else ""
    fig_title = fig_title or f"{xlabel_base}  →  {ylabel}{r_suffix}"
    # Secondary title: how many Moran runs back each data point (n_grouped is the
    # per-config run count from aggregation). Use the typical value if it varies.
    if "n_grouped" in df.columns and df["n_grouped"].notna().any():
        reps = int(df["n_grouped"].dropna().mode().iloc[0])
        ax.set_title(fig_title, fontsize=13, pad=20)
        ax.text(
            0.5,
            1.012,
            f"{reps:,} simulation runs per configuration",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=9,
            color="dimgray",
        )
    else:
        ax.set_title(fig_title, fontsize=13, pad=8)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    _add_property_description(ax, x_prop)
    ax.grid(True, linestyle="--", alpha=0.4)

    # --- 12. Legend with highlight styling and sorted order ---
    handles, labels_leg = ax.get_legend_handles_labels()
    if highlight_categories:
        for h, lbl in zip(handles, labels_leg):
            if lbl in highlight_categories:
                if hasattr(h, "set_markeredgecolor"):
                    h.set_markeredgecolor("black")
                    h.set_markeredgewidth(1.3)
                    h.set_alpha(1.0)
                elif hasattr(h, "set_edgecolor"):
                    h.set_edgecolor("black")
                    h.set_linewidth(1.3)
                    h.set_alpha(1.0)

    # Sort category entries; non-category entries (neutral line, r marker styles) follow
    _cat_set = set(plot_df["category"].dropna().unique())
    _handle_map = dict(zip(labels_leg, handles))
    _sorted_cats = [(l, _handle_map[l]) for l in hue_order if l in _handle_map]
    _others = [(l, h) for l, h in zip(labels_leg, handles) if l not in _cat_set]
    labels_leg = [l for l, _ in _sorted_cats + _others]
    handles = [h for _, h in _sorted_cats + _others]
    # seaborn's style='r' inserts a bare 'r' sub-header; spell out what r means.
    labels_leg = ["r  (mutant relative fitness)" if l == "r" else l for l in labels_leg]

    legend = ax.legend(
        handles=handles,
        labels=labels_leg,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0.0,
        fontsize=9,
    )

    if batch_name:
        _stamp_batch(fig, batch_name)
    fig.tight_layout()
    # After layout so the legend's measured extent is final.
    if stats_text is not None:
        _add_corr_box(ax, stats_text, anchor=legend)
    if fig_path is not None and save:
        fig.savefig(fig_path, bbox_inches="tight", dpi=150)
        print(f"[figure] Saved: {fig_path.name}")
    if show:
        plt.show()


@_timed
def plot_two_property_effect(
    df,
    x_prop,
    y_prop,
    outcome="mean_steps",
    color_dict=None,
    cmap="viridis",
    highlight_categories=None,
    descriptions_below=False,
    *,
    figures_dir=None,
    fig_title=None,
    batch_name=None,
    corr="spearman",
    show=True,
    save=True,
):
    """
    Shows the combined effect of two graph properties on an outcome.

    Each point is one graph; position encodes (x_prop, y_prop); color encodes outcome.
    Animal / special categories can be highlighted with black outlines.

    Args:
        df: graph-statistics DataFrame (one row per graph per r value)
        x_prop: column name for the x-axis structural property
        y_prop: column name for the y-axis structural property
        outcome: column name for the outcome to color by (default: 'mean_steps')
        color_dict: category -> color mapping (used only for highlight outlines)
        cmap: matplotlib colormap name for the outcome gradient
        highlight_categories: list of category names to draw with black outlines on top
        descriptions_below: if True, both property glosses are stacked flat below the
            x-axis instead of x-below / y-rotated; often easier to read
        corr: correlation method for the box annotating each property's correlation
            with the outcome; None / 'spearman' / 'pearson' (default 'spearman').
            None suppresses the box entirely.

    See the module docstring for the shared output tail (figures_dir,
    fig_title, batch_name, show, save).
    """
    if save and not figures_dir:
        raise ValueError("figures_dir must be provided if save=True")
    if color_dict is None:
        color_dict = {}

    r_vals = sorted(df["r"].dropna().unique().tolist()) if "r" in df.columns else []
    r_suffix = f"  (r={r_vals[0]})" if len(r_vals) == 1 else ""

    fig_path = _resolve_figure_path(
        figures_dir, "plot_two_property_effect", x=x_prop, y=y_prop, outcome=outcome
    )

    cols = [x_prop, y_prop, outcome, "category"]
    plot_df = df[cols].replace([np.inf, -np.inf], np.nan).dropna()

    if plot_df.empty:
        print(f"No valid data for ({x_prop}, {y_prop}) -> {outcome}")
        return

    fig, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)
    norm = _outcome_color_norm(plot_df[outcome])
    sc = ax.scatter(
        plot_df[x_prop],
        plot_df[y_prop],
        c=plot_df[outcome],
        norm=norm,
        cmap=cmap,
        alpha=0.6,
        s=40,
        linewidths=0,
        zorder=2,
    )
    cbar = fig.colorbar(sc, ax=ax, label=outcome.replace("_", " ").title())

    outcome_label = outcome.replace("_", " ").title()
    default_title = (
        f"Combined effect of {x_prop.replace('_', ' ').title()} & "
        f"{y_prop.replace('_', ' ').title()}\non {outcome_label}{r_suffix}"
    )
    _finish_two_property_figure(
        fig,
        ax,
        plot_df,
        x_prop,
        y_prop,
        outcome,
        cmap=cmap,
        norm=norm,
        cbar_ax=cbar.ax,
        color_dict=color_dict,
        highlight_categories=highlight_categories,
        descriptions_below=descriptions_below,
        default_title=default_title,
        fig_title=fig_title,
        batch_name=batch_name,
        fig_path=fig_path,
        show=show,
        save=save,
        r_value=r_vals[0] if len(r_vals) == 1 else None,
        corr=corr,
    )


@_timed
def plot_two_property_effect_hexbin(
    df,
    x_prop,
    y_prop,
    outcome="mean_steps",
    color_dict=None,
    cmap="viridis",
    highlight_categories=None,
    descriptions_below=False,
    gridsize=25,
    reduce_C_function=np.mean,
    *,
    figures_dir=None,
    fig_title=None,
    batch_name=None,
    corr="spearman",
    show=True,
    save=True,
):
    """
    Hexbin version of plot_two_property_effect.

    Each hex cell aggregates the outcome for all graphs whose (x_prop, y_prop)
    falls inside it, using reduce_C_function (default: np.mean). Useful when
    points are dense and the population distribution matters more than individual
    graph identity.

    Args:
        df: graph-statistics DataFrame (one row per graph per r value)
        x_prop: column name for the x-axis structural property
        y_prop: column name for the y-axis structural property
        outcome: column name for the outcome to color by (default: 'mean_steps')
        color_dict: category -> color mapping (used for highlight outlines)
        cmap: matplotlib colormap name for the outcome gradient
        highlight_categories: list of category names to draw as scatter on top
        descriptions_below: if True, both property glosses are stacked flat below the
            x-axis instead of x-below / y-rotated; often easier to read
        gridsize: number of hexagons across the x-axis (higher = finer grid)
        reduce_C_function: aggregation applied per bin (np.mean, np.median, etc.)
        corr: correlation method for the box annotating each property's correlation
            with the outcome; None / 'spearman' / 'pearson' (default 'spearman').
            None suppresses the box entirely.

    See the module docstring for the shared output tail (figures_dir,
    fig_title, batch_name, show, save).
    """
    if save and not figures_dir:
        raise ValueError("figures_dir must be provided if save=True")
    if color_dict is None:
        color_dict = {}

    r_vals = sorted(df["r"].dropna().unique().tolist()) if "r" in df.columns else []
    r_suffix = f"  (r={r_vals[0]})" if len(r_vals) == 1 else ""

    fig_path = _resolve_figure_path(
        figures_dir,
        "plot_two_property_effect_hexbin",
        x=x_prop,
        y=y_prop,
        outcome=outcome,
    )

    cols = [x_prop, y_prop, outcome, "category"]
    plot_df = df[cols].replace([np.inf, -np.inf], np.nan).dropna()

    if plot_df.empty:
        print(f"No valid data for ({x_prop}, {y_prop}) -> {outcome}")
        return

    fig, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)
    # One shared norm (log for heavy-tailed positive outcomes, else robust-clipped
    # linear) drives both the hexbin cells and the highlight scatter, so their
    # colors stay on a single, readable scale instead of a dark-crowded linear one.
    norm = _outcome_color_norm(plot_df[outcome])
    hb = ax.hexbin(
        plot_df[x_prop],
        plot_df[y_prop],
        C=plot_df[outcome],
        gridsize=gridsize,
        cmap=cmap,
        norm=norm,
        reduce_C_function=reduce_C_function,
        mincnt=1,
        linewidths=0.2,
    )
    cbar = fig.colorbar(hb, ax=ax, label=outcome.replace("_", " ").title())

    outcome_label = outcome.replace("_", " ").title()
    reduce_name = getattr(reduce_C_function, "__name__", str(reduce_C_function))
    default_title = (
        f"Combined effect of {x_prop.replace('_', ' ').title()} & "
        f"{y_prop.replace('_', ' ').title()}\non {outcome_label}"
        f" (hex={reduce_name}){r_suffix}"
    )
    _finish_two_property_figure(
        fig,
        ax,
        plot_df,
        x_prop,
        y_prop,
        outcome,
        cmap=cmap,
        norm=norm,
        cbar_ax=cbar.ax,
        color_dict=color_dict,
        highlight_categories=highlight_categories,
        descriptions_below=descriptions_below,
        default_title=default_title,
        fig_title=fig_title,
        batch_name=batch_name,
        fig_path=fig_path,
        show=show,
        save=save,
        r_value=r_vals[0] if len(r_vals) == 1 else None,
        corr=corr,
    )

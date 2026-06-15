"""
All figure-producing functions plus the small caching/stamping infrastructure
they share.

Depends on the leaf modules: ``colors`` (palette + property metadata + the
``_sort_categories`` ordering) and ``provenance`` (``_bi_get`` for the batch
title card). Nothing imports from here, so this is the top of the dependency graph.
"""
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

from .colors import DEFAULT_FIG_SIZE, GRAPH_PROPERTY_DESCRIPTION, _sort_categories
from .provenance import _bi_get

__all__ = [
    'try_load_cached',
    'plot_batch_info_card',
    'plot_steps_violin',
    'plot_steps_pvalue_matrix',
    'plot_steps_histogram',
    'plot_outcome_vs_property',
    'plot_two_property_effect',
    'plot_two_property_effect_hexbin',
]


def _resolve_figure_path(figures_dir, func_name: str, **key_kwargs):
    """Build a descriptive Path for a cached figure, creating the directory if needed."""
    if figures_dir is None:
        return None
    p = Path(figures_dir)
    p.mkdir(parents=True, exist_ok=True)
    slug = "__".join(f"{k}={v}" for k, v in key_kwargs.items())
    slug = slug.replace("/", "-").replace(" ", "_").replace(",", "-")
    return p / f"{func_name}__{slug}.png"


def try_load_cached(path) -> bool:
    """Display a saved PNG from disk and return True; return False if not found."""
    if path is not None and Path(path).exists():
        try:
            from IPython.display import Image, display
            display(Image(str(path), width=int(DEFAULT_FIG_SIZE[0] * 100)))
            print(f"[cache] Loaded: {Path(path).name}")
            return True
        except ImportError:
            pass
    return False


def _stamp_batch(fig, batch_name: str) -> None:
    """Add a source label to the bottom-right corner of the figure."""
    fig.text(
        0.99, 0.01, f"source: {batch_name}",
        fontsize=8, color="#666666", ha="right", va="bottom",
        style="italic", transform=fig.transFigure,
    )


def plot_batch_info_card(
    batch_info,
    figures_dir=None,
    force_recompute=False,
):
    """Generate a standalone title-card figure for a batch, suitable as a first/catalog slide.

    Args:
        batch_info: dict returned by load_batch_info() or create_batch_info()
        figures_dir: directory where PNG is saved; None = display only, no save
        force_recompute: skip cache and regenerate even if PNG already exists
    """
    fig_path = _resolve_figure_path(figures_dir, 'batch_info_card')
    if not force_recompute and try_load_cached(fig_path):
        return

    # Read fields from the nested batch_info (with flat fallback for legacy files).
    name              = batch_info.get('name', 'Unknown Batch')
    description       = batch_info.get('description', '')
    notes             = batch_info.get('notes', '')
    created_at        = _bi_get(batch_info, 'created_at') or _bi_get(batch_info, 'date_created', default='')
    graph_types       = _bi_get(batch_info, 'zoo', 'graph_types', default=[])
    node_sizes        = _bi_get(batch_info, 'zoo', 'node_sizes', default=[])
    n_graphs          = _bi_get(batch_info, 'zoo', 'n_graphs')
    r_values          = _bi_get(batch_info, 'simulation', 'r_values', default=[])
    n_repeats         = _bi_get(batch_info, 'simulation', 'n_repeats')
    total_simulations = _bi_get(batch_info, 'simulation', 'total_simulations')
    engine            = _bi_get(batch_info, 'simulation', 'engine')
    n_requested_jobs  = _bi_get(batch_info, 'hpc', 'n_requested_jobs')
    queue             = _bi_get(batch_info, 'hpc', 'queue')
    memory_mb         = _bi_get(batch_info, 'hpc', 'memory_mb')
    lsf_job_id        = _bi_get(batch_info, 'hpc', 'lsf_job_id')
    git_commit        = _bi_get(batch_info, 'provenance', 'git_commit')
    git_branch        = _bi_get(batch_info, 'provenance', 'git_branch')
    git_dirty         = _bi_get(batch_info, 'provenance', 'git_dirty')
    hostname          = _bi_get(batch_info, 'provenance', 'hostname')

    # 16:9 canvas so the card drops straight onto a widescreen slide.
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    ax.axis('off')
    fig.patch.set_facecolor('white')

    # Title
    ax.text(0.05, 0.93, name, transform=ax.transAxes,
            fontsize=30, fontweight='bold', va='top', ha='left', color='#1a1a1a')

    # Subtitle: date + engine (muted, just under the title)
    subtitle_bits = []
    if created_at:
        subtitle_bits.append(str(created_at).replace('T', '  '))
    if engine:
        subtitle_bits.append(f'{engine} engine')
    if subtitle_bits:
        ax.text(0.05, 0.845, '   ·   '.join(subtitle_bits), transform=ax.transAxes,
                fontsize=13, va='top', ha='left', color='#888888')

    # Horizontal rule under the title block
    ax.plot([0.04, 0.96], [0.80, 0.80], transform=ax.transAxes,
            color='#cccccc', linewidth=1.2, solid_capstyle='butt')

    # Description
    if description:
        wrapped = textwrap.fill(description, width=95)
        ax.text(0.05, 0.74, wrapped, transform=ax.transAxes,
                fontsize=14, va='top', ha='left', color='#333333',
                style='italic', linespacing=1.5)

    # Dense grouped metadata rows: label on the left, a single packed value line.
    def _meta_row(label, value, y):
        ax.text(0.05, y, label, transform=ax.transAxes,
                fontsize=13, va='top', ha='left', fontweight='bold', color='#444444')
        ax.text(0.20, y, value, transform=ax.transAxes,
                fontsize=13, va='top', ha='left', color='#222222')

    def _join(parts):
        return '      '.join(p for p in parts if p)

    y = 0.56
    row_h = 0.105

    zoo_parts = []
    if n_graphs is not None:
        zoo_parts.append(f'{int(n_graphs):,} graphs')
    if graph_types:
        zoo_parts.append(f'types: {", ".join(graph_types)}')
    if node_sizes:
        zoo_parts.append(f'sizes: {", ".join(str(n) for n in node_sizes)}')
    if zoo_parts:
        _meta_row('Zoo', _join(zoo_parts), y);  y -= row_h

    sim_parts = []
    if r_values:
        sim_parts.append(f'r = {", ".join(str(r) for r in r_values)}')
    if n_repeats is not None:
        sim_parts.append(f'{int(n_repeats):,} reps/config')
    if total_simulations is not None:
        sim_parts.append(f'{int(total_simulations):,} total sims')
    if sim_parts:
        _meta_row('Simulation', _join(sim_parts), y);  y -= row_h

    hpc_parts = []
    if n_requested_jobs is not None:
        hpc_parts.append(f'{int(n_requested_jobs):,} jobs')
    if queue:
        hpc_parts.append(f'queue: {queue}')
    if memory_mb is not None:
        hpc_parts.append(f'{int(memory_mb):,} MB/job')
    if lsf_job_id:
        hpc_parts.append(f'job {lsf_job_id}')
    if hpc_parts:
        _meta_row('HPC', _join(hpc_parts), y);  y -= row_h

    # Provenance + notes footer (muted, bottom of the slide)
    footer_bits = []
    if git_commit:
        commit = f'commit {git_commit[:8]}'
        if git_branch:
            commit += f' ({git_branch})'
        if git_dirty:
            commit += ' +dirty'
        footer_bits.append(commit)
    if hostname:
        footer_bits.append(hostname)
    if footer_bits:
        ax.text(0.05, 0.13, '   ·   '.join(footer_bits), transform=ax.transAxes,
                fontsize=10, va='top', ha='left', color='#aaaaaa')
    if notes:
        ax.text(0.05, 0.07, textwrap.fill(f'Notes: {notes}', width=110),
                transform=ax.transAxes,
                fontsize=10, va='top', ha='left', color='#999999', style='italic')

    fig.tight_layout()
    if fig_path is not None:
        fig.savefig(fig_path, bbox_inches='tight', dpi=200, facecolor='white')
        print(f"[cache] Saved: {fig_path.name}")
    plt.show()


def _load_fixation_steps_by_category(
    results_path,
    df_graphs,
    r=None,
    max_points_per_category=50_000,
):
    """Load fixation 'steps' joined to graph 'category' for a single r value.

    Shared loader for ``plot_steps_violin`` and ``plot_steps_pvalue_matrix`` so the
    two figures are always built from exactly the same rows (same r resolution, same
    fixation filter, same subsample). Returns:

    - a tidy pandas DataFrame with columns ['category', 'steps'] (subsampled);
    - ``fixation_counts``: true per-category fixation counts, read *before*
      subsampling so callers can annotate how much data backs each category;
    - the resolved ``r`` and an ``r_suffix`` label for titles;
    - ``subsampled``: whether the cap actually trimmed any category.

    See ``plot_steps_violin`` for why only fixation rows are materialised and why
    subsampling to ``max_points_per_category`` is faithful to the full distribution.
    """
    import polars as pl

    _rp = Path(results_path)
    _scanner = pl.scan_parquet(str(_rp)) if _rp.suffix == '.parquet' else pl.scan_csv(str(_rp))
    _has_r = 'r' in _scanner.collect_schema().names()

    # Only fixation events are ever drawn/tested, so filter them lazily up front.
    lf = _scanner.select(['wl_hash', 'steps', 'fixation'] + (['r'] if _has_r else []))
    lf = lf.filter(pl.col('fixation'))

    # Pooling several r values would silently overlay distributions, so resolve to a
    # single r before collecting.
    r_suffix = ""
    if _has_r:
        r_available = sorted(lf.select(pl.col('r')).unique().collect().to_series().to_list())
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
        lf = lf.filter(pl.col('r') == r)
        r_suffix = f"  (r={r})"

    merged_raw = lf.join(
        pl.from_pandas(df_graphs[['wl_hash', 'category']]).lazy(),
        on='wl_hash',
        how='left',
    ).collect()

    _vc = merged_raw['category'].value_counts()
    fixation_counts = dict(zip(_vc.get_column('category').to_list(), _vc.get_column('count').to_list()))

    # Subsample each category down to the cap with a within-category shuffle (uniform
    # sample), keeping every violin's KDE and every pairwise test cheap and faithful.
    subsampled = False
    if max_points_per_category is not None:
        largest_category = max(fixation_counts.values(), default=None)
        subsampled = largest_category is not None and largest_category > max_points_per_category
        merged_raw = (
            merged_raw
            .with_columns(
                pl.int_range(pl.len()).shuffle(seed=0).over('category').alias('_rn')
            )
            .filter(pl.col('_rn') < max_points_per_category)
            .drop('_rn')
        )

    return merged_raw.to_pandas(), fixation_counts, r, r_suffix, subsampled


def plot_steps_violin(
    results_path,
    df_graphs,
    color_dict=None,
    categories=None,
    figures_dir=None,
    force_recompute=False,
    batch_name=None,
    fig_title=None,
    show=True,
    r=None,
    max_points_per_category=50_000,
    results_csv_path=None,  # deprecated alias for results_path
):
    """Violin plot of steps-to-fixation distribution, one violin per graph category.

    A violin is a kernel density estimate, and KDE cost is linear in the number of
    points. The raw batch can hold tens of millions of fixation events, which makes
    seaborn spend many minutes building the KDEs. Two cheap measures keep it fast
    without changing the picture: only the fixation rows are materialised (non-fixation
    rows are never drawn), and each category is subsampled to ``max_points_per_category``
    points before the KDE (a 50k sample is visually identical to the full distribution).

    Args:
        results_path: path to the raw_results.parquet or raw_results.csv (read lazily)
        df_graphs: DataFrame with at least 'wl_hash' and 'category' columns
        color_dict: category -> hex color mapping for violin fills
        categories: x-axis order; defaults to sorted unique values in df_graphs['category']
        figures_dir: directory where PNG is saved; None = display only, no save
        force_recompute: skip cache and regenerate even if PNG already exists
        batch_name: batch label stamped in the bottom-right corner of the figure
        show: change this to flase if you want the fig to be made but not shown
        r: which selection coefficient to plot (violins show one r at a time). If None
            and the data has a single r, that value is used; if None and several r
            values are present, a ValueError is raised asking you to pick one.
        max_points_per_category: cap on the number of fixation events fed to each
            category's KDE. None disables subsampling and plots every point (slow for
            large batches). Default 50_000.
    """
    if color_dict is None:
        color_dict = {}

    # Support deprecated alias
    if results_path is None and results_csv_path is not None:
        results_path = results_csv_path

    fig_path = _resolve_figure_path(figures_dir, 'plot_steps_violin')
    if not force_recompute and try_load_cached(fig_path):
        return

    if categories is None:
        categories = _sort_categories(df_graphs['category'].dropna().unique().tolist())

    merged_raw, fixation_counts, r, r_suffix, subsampled = _load_fixation_steps_by_category(
        results_path, df_graphs, r=r, max_points_per_category=max_points_per_category,
    )

    palette = {cat: color_dict[cat] for cat in categories if cat in color_dict}

    fig, ax = plt.subplots(figsize=(max(12, len(categories) * 1.1), 7))
    sns.violinplot(
        data=merged_raw,
        x='category',
        y='steps',
        order=categories,
        hue='category',
        palette=palette,
        legend=False,
        inner='box',
        linewidth=1.2,
        ax=ax,
    )
    fig_title = fig_title or f'Distribution of Steps to Fixation by Category{r_suffix}'
    # Annotate each violin with its true fixation count (n) on a second label line.
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(
        [f"{cat}\nn = {fixation_counts.get(cat, 0):,}" for cat in categories],
        rotation=45, ha='right', fontsize=10,
    )
    ax.set_xlabel('Category', fontsize=13)
    ax.set_ylabel('Steps to Fixation', fontsize=13)
    ax.set_title(fig_title, fontsize=14)
    if batch_name:
        _stamp_batch(fig, batch_name)
    if subsampled:
        fig.text(
            0.01, 0.01,
            f"violins drawn from a random subsample of {max_points_per_category:,} points/category",
            fontsize=8, color="#666666", ha="left", va="bottom",
            style="italic", transform=fig.transFigure,
        )
    fig.tight_layout()
    if fig_path is not None:
        fig.savefig(fig_path, bbox_inches='tight', dpi=150)
        print(f"[cache] Saved: {fig_path.name}")
    if show:
        plt.show()


def _significance_stars(p):
    """Conventional significance markers for a (corrected) p-value."""
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'ns'


def plot_steps_pvalue_matrix(
    results_path,
    df_graphs,
    categories=None,
    figures_dir=None,
    force_recompute=False,
    batch_name=None,
    fig_title=None,
    show=True,
    r=None,
    max_points_per_category=50_000,
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
    effect size, not by category).
    """
    from scipy.stats import mannwhitneyu

    fig_path = _resolve_figure_path(figures_dir, 'plot_steps_pvalue_matrix')
    if not force_recompute and try_load_cached(fig_path):
        return

    if categories is None:
        categories = _sort_categories(df_graphs['category'].dropna().unique().tolist())

    merged, fixation_counts, r, r_suffix, subsampled = _load_fixation_steps_by_category(
        results_path, df_graphs, r=r, max_points_per_category=max_points_per_category,
    )

    # Only categories with fixation data can be tested.
    categories = [c for c in categories if fixation_counts.get(c, 0) > 0]
    if len(categories) < 2:
        print("[skip] need at least two categories with fixation events to compare")
        return

    groups = {c: merged.loc[merged['category'] == c, 'steps'].to_numpy() for c in categories}

    k = len(categories)
    n_pairs = k * (k - 1) // 2
    effect = np.full((k, k), np.nan)   # rank-biserial, antisymmetric
    annot = np.full((k, k), '', dtype=object)
    for i in range(k):
        for j in range(i + 1, k):
            a, b = groups[categories[i]], groups[categories[j]]
            u, p = mannwhitneyu(a, b, alternative='two-sided')
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
        fmt='',
        cmap='coolwarm',
        center=0.0,
        vmin=-1.0,
        vmax=1.0,
        square=True,
        linewidths=0.5,
        linecolor='white',
        cbar_kws={'label': 'rank-biserial effect (row slower → +1)'},
        annot_kws={'fontsize': 8},
        xticklabels=categories,
        yticklabels=categories,
        ax=ax,
    )
    fig_title = fig_title or f'Pairwise steps-to-fixation significance{r_suffix}'
    ax.set_title(fig_title, fontsize=14)
    ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(categories, rotation=0, fontsize=9)
    if batch_name:
        _stamp_batch(fig, batch_name)
    cap_note = (
        f"  n capped at {max_points_per_category:,}/category"
        if max_points_per_category is not None else ""
    )
    fig.text(
        0.01, 0.01,
        "color = signed rank-biserial effect; stars = Bonferroni Mann-Whitney p "
        f"(*** <0.001, ** <0.01, * <0.05).{cap_note}",
        fontsize=8, color="#666666", ha="left", va="bottom",
        style="italic", transform=fig.transFigure,
    )
    fig.tight_layout()
    if fig_path is not None:
        fig.savefig(fig_path, bbox_inches='tight', dpi=150)
        print(f"[cache] Saved: {fig_path.name}")
    if show:
        plt.show()


def plot_steps_histogram(
    df,
    metric='mean_steps',
    category=None,
    color_dict=None,
    bins=50,
    figures_dir=None,
    force_recompute=False,
    batch_name=None,
    show=True,
):
    """Histogram of a steps/outcome metric, optionally filtered to one graph category.

    Args:
        df: aggregated graph-statistics DataFrame (output of the groupby/merge block)
        metric: column to histogram, e.g. 'mean_steps', 'prob_fixation'
        category: if given, restricts to df['category'] == category; None plots all graphs
        color_dict: category -> hex color; the category's color is used for the bars when set
        bins: number of histogram bins
        figures_dir: directory where PNG is saved; None = display only, no save
        force_recompute: skip cache and regenerate even if PNG already exists
        batch_name: batch label stamped in the bottom-right corner of the figure
        show: change this to flase if you want the fig to be made but not shown

    """
    if color_dict is None:
        color_dict = {}

    r_vals = sorted(df['r'].dropna().unique().tolist()) if 'r' in df.columns else []
    r_suffix = f"  (r={r_vals[0]})" if len(r_vals) == 1 else ""

    cat_key = category or 'all'
    fig_path = _resolve_figure_path(figures_dir, 'plot_steps_histogram',
                                    metric=metric, category=cat_key)
    if not force_recompute and try_load_cached(fig_path):
        return

    plot_df = df if category is None else df.loc[df['category'] == category]
    data = plot_df[metric].dropna()

    if data.empty:
        print(f"[plot_steps_histogram] No data for metric={metric!r}, category={category!r}")
        return

    label = category or 'All Graphs'
    metric_label = metric.replace('_', ' ').title()
    bar_color = color_dict.get(category, '#4c72b0') if category else '#4c72b0'

    fig, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)
    ax.hist(data, bins=bins, color=bar_color, edgecolor='black', alpha=0.7)
    ax.set_xlabel(metric_label, fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Distribution of {metric_label} — {label}{r_suffix}', fontsize=14)
    ax.grid(axis='y', alpha=0.3)

    if batch_name:
        _stamp_batch(fig, batch_name)
    fig.tight_layout()
    if fig_path is not None:
        fig.savefig(fig_path, bbox_inches='tight', dpi=150)
        print(f"[cache] Saved: {fig_path.name}")
    if show:
        plt.show()


def plot_outcome_vs_property(
    df,
    x_prop,
    y_outcome='prob_fixation',
    color_dict=None,
    density_threshold=50,
    highlight_categories=None,
    size_property=None,
    figures_dir=None,
    force_recompute=False,
    batch_name=None,
    fig_title=None,
    show=True,
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
        highlight_categories: list of categories drawn with black outlines on top of scatter
        size_property: column name to encode as marker size; None = uniform size
        figures_dir: directory where PNG is saved; None = display only, no save
        force_recompute: skip cache and regenerate even if PNG already exists
        batch_name: batch label stamped in the bottom-right corner of the figure
        show: change this to flase if you want the fig to be made but not shown

    """
    if color_dict is None:
        color_dict = {}

    fig_path = _resolve_figure_path(figures_dir, 'plot_outcome_vs_property',
                                    x=x_prop, y=y_outcome)
    if not force_recompute and try_load_cached(fig_path):
        return

    # --- 1. Labels ---
    prob_label = "Fixation Probability ($P_{fix}$)"
    is_prob = (y_outcome == 'prob_fixation')
    ylabel = prob_label if is_prob else y_outcome.replace('_', ' ').title()

    if x_prop == 'prob_fixation':
        xlabel_base = prob_label
    elif x_prop == 'std_steps':
        xlabel_base = 'Std. Steps to Fixation'
    else:
        xlabel_base = x_prop.replace('_', ' ').title()

    desc_text = GRAPH_PROPERTY_DESCRIPTION.get(x_prop, '')
    wrapped_desc = textwrap.fill(desc_text, width=90) if desc_text else ''
    xlabel = f"{xlabel_base}\n{wrapped_desc}" if wrapped_desc else xlabel_base

    # --- 2. Pearson correlation (per r value, compactly) ---
    r_values = sorted(df['r'].dropna().unique()) if 'r' in df.columns else []
    cols_for_corr = [x_prop, y_outcome] + (['r'] if r_values else [])
    clean_df = df[cols_for_corr].replace([np.inf, -np.inf], np.nan).dropna()

    def _safe_corr(a, b):
        if len(a) > 1 and a.std() > 0 and b.std() > 0:
            return a.corr(b)
        return np.nan

    if len(r_values) > 1:
        corr_lines = ["Pearson corr", "-" * 18]
        for rv in r_values:
            sub = clean_df[clean_df['r'] == rv]
            c = _safe_corr(sub[x_prop], sub[y_outcome])
            corr_lines.append(f"r={rv}: {c:.3f}" if pd.notna(c) else f"r={rv}: N/A")
    else:
        c = _safe_corr(clean_df[x_prop], clean_df[y_outcome])
        corr_lines = ["Pearson corr", f"{c:.3f}" if pd.notna(c) else "N/A"]
    stats_text = "\n".join(corr_lines)

    # --- 3. X-axis processing ---
    plot_df = df.copy()
    is_numeric_x = pd.api.types.is_numeric_dtype(plot_df[x_prop])

    if is_numeric_x:
        plot_df[x_prop] = pd.to_numeric(plot_df[x_prop], errors='coerce')
        plot_df['x_plot'] = plot_df[x_prop].round(3)
        unique_cats = None
    else:
        plot_df = plot_df.dropna(subset=[x_prop])
        unique_cats = sorted(plot_df[x_prop].unique())
        plot_df['x_plot'] = plot_df[x_prop].map({v: i for i, v in enumerate(unique_cats)})

    # --- 4. Auto-detect discrete x (few unique values relative to data size) ---
    valid_x = plot_df['x_plot'].dropna()
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
            subset = plot_df.loc[plot_df['x_plot'] == x_val, y_outcome].dropna()
            if len(subset) > 0:
                parts = ax.violinplot(subset, positions=[x_val], widths=violin_width,
                                      showmeans=False, showextrema=False)
                for pc in parts['bodies']:
                    pc.set_facecolor('whitesmoke')
                    pc.set_edgecolor('lightgray')
                    pc.set_alpha(1.0)

        # Vectorized jitter -- much faster than apply(func, axis=1)
        mask = plot_df['x_plot'].isin(dense_x_values) & plot_df['x_plot'].notna()
        jitter_half = violin_width * 0.15
        plot_df['x_jittered'] = plot_df['x_plot'].copy().astype(float)
        if mask.any():
            plot_df.loc[mask, 'x_jittered'] = (
                plot_df.loc[mask, 'x_plot']
                + np.random.uniform(-jitter_half, jitter_half, size=int(mask.sum()))
            )
    else:
        plot_df['x_jittered'] = plot_df['x_plot']

    # --- 6. Scatter (background) ---
    hue_order = _sort_categories(plot_df['category'].dropna().unique().tolist())
    sns.scatterplot(
        data=plot_df, ax=ax,
        x='x_jittered', y=y_outcome,
        hue='category', hue_order=hue_order,
        style='r' if len(r_values) > 1 else None,
        size=size_property, sizes=(20, 100),
        palette=color_dict,
        alpha=0.7, edgecolor='w', linewidth=0.5, zorder=2,
    )

    # --- 7. Highlighted categories (foreground) ---
    if highlight_categories:
        hl_df = plot_df[plot_df['category'].isin(highlight_categories)]
        if not hl_df.empty:
            sns.scatterplot(
                data=hl_df, ax=ax,
                x='x_jittered', y=y_outcome,
                hue='category', hue_order=hue_order,
                style='r' if len(r_values) > 1 else None,
                size=size_property, sizes=(20, 100),
                palette=color_dict,
                alpha=1.0, edgecolor='black', linewidth=1.8,
                legend=False, zorder=3,
            )

    # --- 8. Neutral 1/N reference line ---
    if is_prob and 'n_nodes' in plot_df.columns:
        n_col = plot_df['n_nodes'].dropna()
        if len(n_col) > 0:
            if x_prop == 'n_nodes':
                # x encodes N directly: draw the theoretical y = 1/x curve
                x_range = np.linspace(max(1, n_col.min()), n_col.max(), 300)
                ax.plot(x_range, 1.0 / x_range, color='black', linestyle='--',
                        linewidth=1.2, label='Neutral (1/N)', zorder=1)
            else:
                # Only draw a flat line when N is homogeneous (CV < 5%)
                n_mean = n_col.mean()
                n_cv = n_col.std() / n_mean if n_mean > 0 else 1.0
                if n_cv < 0.05:
                    ax.axhline(1.0 / n_mean, color='black', linestyle=':',
                               linewidth=1.0, label=f'Neutral (1/N={n_mean:.0f})', zorder=1)
                # else: N varies too much -- a flat line would be misleading, so skip

    # --- 9. Categorical x-axis ticks ---
    if not is_numeric_x and unique_cats is not None:
        ax.set_xticks(range(len(unique_cats)))
        ax.set_xticklabels(unique_cats)

    # --- 10. Titles & labels ---
    r_suffix = f"  (r={r_values[0]})" if len(r_values) == 1 else ""
    fig_title = fig_title or f'{xlabel_base}  →  {ylabel}{r_suffix}'
    ax.set_title(fig_title, fontsize=13, pad=8)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.4)

    # --- 11. Correlation text box (bottom-right inside axes) ---
    ax.text(
        0.97, 0.04, stats_text,
        transform=ax.transAxes, fontsize=9,
        verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='lightgray'),
        zorder=5,
    )

    # --- 12. Legend with highlight styling and sorted order ---
    handles, labels_leg = ax.get_legend_handles_labels()
    if highlight_categories:
        for h, lbl in zip(handles, labels_leg):
            if lbl in highlight_categories:
                if hasattr(h, 'set_markeredgecolor'):
                    h.set_markeredgecolor('black')
                    h.set_markeredgewidth(1.8)
                    h.set_alpha(1.0)
                elif hasattr(h, 'set_edgecolor'):
                    h.set_edgecolor('black')
                    h.set_linewidth(1.8)
                    h.set_alpha(1.0)

    # Sort category entries; non-category entries (neutral line, r marker styles) follow
    _cat_set = set(plot_df['category'].dropna().unique())
    _handle_map = dict(zip(labels_leg, handles))
    _sorted_cats = [(l, _handle_map[l]) for l in hue_order if l in _handle_map]
    _others      = [(l, h) for l, h in zip(labels_leg, handles) if l not in _cat_set]
    labels_leg = [l for l, _ in _sorted_cats + _others]
    handles    = [h for _, h in _sorted_cats + _others]

    ax.legend(handles=handles, labels=labels_leg,
              bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., fontsize=9)

    if batch_name:
        _stamp_batch(fig, batch_name)
    fig.tight_layout()
    if fig_path is not None:
        fig.savefig(fig_path, bbox_inches='tight', dpi=150)
        print(f"[cache] Saved: {fig_path.name}")
    if show:
        plt.show()


def plot_two_property_effect(
    df,
    x_prop,
    y_prop,
    outcome='mean_steps',
    color_dict=None,
    highlight_categories=None,
    cmap='viridis',
    figures_dir=None,
    force_recompute=False,
    batch_name=None,
    show=True,
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
        highlight_categories: list of category names to draw with black outlines on top
        cmap: matplotlib colormap name for the outcome gradient
        show: change this to flase if you want the fig to be made but not shown

    """
    if color_dict is None:
        color_dict = {}

    r_vals = sorted(df['r'].dropna().unique().tolist()) if 'r' in df.columns else []
    r_suffix = f"  (r={r_vals[0]})" if len(r_vals) == 1 else ""

    fig_path = _resolve_figure_path(figures_dir, 'plot_two_property_effect',
                                    x=x_prop, y=y_prop, outcome=outcome)
    if not force_recompute and try_load_cached(fig_path):
        return

    cols = [x_prop, y_prop, outcome, 'category']
    plot_df = df[cols].replace([np.inf, -np.inf], np.nan).dropna()

    if plot_df.empty:
        print(f"No valid data for ({x_prop}, {y_prop}) -> {outcome}")
        return

    fig, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)

    norm = mcolors.Normalize(vmin=plot_df[outcome].min(), vmax=plot_df[outcome].max())

    sc = ax.scatter(
        plot_df[x_prop], plot_df[y_prop],
        c=plot_df[outcome], norm=norm, cmap=cmap,
        alpha=0.6, s=40, linewidths=0, zorder=2,
    )
    fig.colorbar(sc, ax=ax, label=outcome.replace("_", " ").title())

    # --- Highlight specific categories on top ---
    if highlight_categories:
        hl_df = plot_df[plot_df['category'].isin(highlight_categories)]
        if not hl_df.empty:
            for cat, grp in hl_df.groupby('category'):
                ax.scatter(
                    grp[x_prop], grp[y_prop],
                    c=grp[outcome], norm=norm, cmap=cmap,
                    s=120, linewidths=1.8,
                    edgecolors=color_dict.get(cat, 'black'),
                    zorder=3, label=cat,
                )

    # --- Correlations text ---
    def _safe_corr(a, b):
        mask = pd.notna(a) & pd.notna(b)
        if mask.sum() > 1 and a[mask].std() > 0 and b[mask].std() > 0:
            return a[mask].corr(b[mask])
        return np.nan

    corr_x = _safe_corr(plot_df[x_prop], plot_df[outcome])
    corr_y = _safe_corr(plot_df[y_prop], plot_df[outcome])
    corr_text = (
        f"Pearson corr with {outcome.replace('_', ' ')}\n"
        + "-" * 30 + "\n"
        + f"{x_prop}: {corr_x:.3f}\n"
        + f"{y_prop}: {corr_y:.3f}"
    )
    ax.text(
        0.0, -0.14, corr_text,
        transform=ax.transAxes, fontsize=9,
        verticalalignment='top',
        clip_on=False,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor="lightgray"),
        zorder=5,
    )

    # --- Labels & formatting ---
    outcome_label = outcome.replace("_", " ").title()
    ax.set_xlabel(x_prop.replace("_", " ").title(), fontsize=12)
    ax.set_ylabel(y_prop.replace("_", " ").title(), fontsize=12)
    ax.set_title(
        f"Combined effect of {x_prop.replace('_', ' ').title()} & "
        f"{y_prop.replace('_', ' ').title()}\non {outcome_label}{r_suffix}",
        fontsize=13,
    )
    ax.grid(True, linestyle='--', alpha=0.4)

    if highlight_categories:
        ax.legend(title="Category", bbox_to_anchor=(1.18, 1), loc='upper left')

    if batch_name:
        _stamp_batch(fig, batch_name)
    plt.tight_layout()
    if fig_path is not None:
        plt.savefig(fig_path, bbox_inches='tight', dpi=150)
        print(f"[cache] Saved: {fig_path.name}")
    if show:
        plt.show()


def plot_two_property_effect_hexbin(
    df,
    x_prop,
    y_prop,
    outcome='mean_steps',
    color_dict=None,
    highlight_categories=None,
    cmap='viridis',
    gridsize=25,
    reduce_C_function=np.mean,
    figures_dir=None,
    force_recompute=False,
    batch_name=None,
    show=True,
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
        highlight_categories: list of category names to draw as scatter on top
        cmap: matplotlib colormap name for the outcome gradient
        gridsize: number of hexagons across the x-axis (higher = finer grid)
        reduce_C_function: aggregation applied per bin (np.mean, np.median, etc.)
        show: change this to flase if you want the fig to be made but not shown

    """
    if color_dict is None:
        color_dict = {}

    r_vals = sorted(df['r'].dropna().unique().tolist()) if 'r' in df.columns else []
    r_suffix = f"  (r={r_vals[0]})" if len(r_vals) == 1 else ""

    fig_path = _resolve_figure_path(figures_dir, 'plot_two_property_effect_hexbin',
                                    x=x_prop, y=y_prop, outcome=outcome)
    if not force_recompute and try_load_cached(fig_path):
        return

    cols = [x_prop, y_prop, outcome, 'category']
    plot_df = df[cols].replace([np.inf, -np.inf], np.nan).dropna()

    if plot_df.empty:
        print(f"No valid data for ({x_prop}, {y_prop}) -> {outcome}")
        return

    fig, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)

    hb = ax.hexbin(
        plot_df[x_prop], plot_df[y_prop],
        C=plot_df[outcome],
        gridsize=gridsize,
        cmap=cmap,
        reduce_C_function=reduce_C_function,
        mincnt=1,
        linewidths=0.2,
    )
    fig.colorbar(hb, ax=ax, label=outcome.replace("_", " ").title())

    # --- Highlight specific categories on top ---
    norm = mcolors.Normalize(vmin=plot_df[outcome].min(), vmax=plot_df[outcome].max())
    if highlight_categories:
        hl_df = plot_df[plot_df['category'].isin(highlight_categories)]
        if not hl_df.empty:
            for cat, grp in hl_df.groupby('category'):
                ax.scatter(
                    grp[x_prop], grp[y_prop],
                    c=grp[outcome], norm=norm, cmap=cmap,
                    s=120, linewidths=1.8,
                    edgecolors=color_dict.get(cat, 'black'),
                    zorder=3, label=cat,
                )

    # --- Correlations text ---
    def _safe_corr(a, b):
        mask = pd.notna(a) & pd.notna(b)
        if mask.sum() > 1 and a[mask].std() > 0 and b[mask].std() > 0:
            return a[mask].corr(b[mask])
        return np.nan

    corr_x = _safe_corr(plot_df[x_prop], plot_df[outcome])
    corr_y = _safe_corr(plot_df[y_prop], plot_df[outcome])
    reduce_name = getattr(reduce_C_function, '__name__', str(reduce_C_function))
    corr_text = (
        f"Pearson corr with {outcome.replace('_', ' ')}\n"
        + "-" * 30 + "\n"
        + f"{x_prop}: {corr_x:.3f}\n"
        + f"{y_prop}: {corr_y:.3f}"
    )
    ax.text(
        0.03, 0.97, corr_text,
        transform=ax.transAxes, fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor="lightgray"),
        zorder=5,
    )

    # --- Labels & formatting ---
    outcome_label = outcome.replace("_", " ").title()
    ax.set_xlabel(x_prop.replace("_", " ").title(), fontsize=12)
    ax.set_ylabel(y_prop.replace("_", " ").title(), fontsize=12)
    ax.set_title(
        f"Combined effect of {x_prop.replace('_', ' ').title()} & "
        f"{y_prop.replace('_', ' ').title()}\non {outcome_label}"
        f" (hex={reduce_name}){r_suffix}",
        fontsize=13,
    )
    ax.grid(True, linestyle='--', alpha=0.4)

    if highlight_categories:
        ax.legend(title="Category", bbox_to_anchor=(1.18, 1), loc='upper left')

    if batch_name:
        _stamp_batch(fig, batch_name)
    plt.tight_layout()
    if fig_path is not None:
        plt.savefig(fig_path, bbox_inches='tight', dpi=150)
        print(f"[cache] Saved: {fig_path.name}")
    if show:
        plt.show()

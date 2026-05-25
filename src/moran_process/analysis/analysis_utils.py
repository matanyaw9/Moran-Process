"""
Utility functions for analysis notebooks
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
import textwrap
from pathlib import Path
import matplotlib.colors as mcolors
import hashlib
import json
from datetime import datetime


CATEGORY_COLOR_DICT = dict({
    'Random': 'lightgray',     
    'Avian': "#2DB806",       
    'Fish': '#1f77b4',        
    'Mammalian': "#833105",   
    'Complete': 'black',
    'Decelerator': "#5e54e0",   
    'Accelerator': "#d1234e",       

    'Other': 'yellow'          
})

# Use a defaultdict to return 'lightgray' for unknown categories
# Paste your dictionary here (or ensure it's in the global scope)
GRAPH_PROPERTY_DESCRIPTION = {
    'n_nodes': "The total number of vertices (individuals) in the graph.",
    'n_edges': "The total number of connections (links) between nodes in the graph.",
    'density': "The ratio of actual edges to the maximum possible number of edges (0 = empty, 1 = fully connected).",
    'diameter': "The longest shortest path between any pair of nodes (the 'width' of the network).",
    'avg_degree': "The average number of connections a node has.",
    'average_clustering': "A measure of how much nodes tend to cluster together (how likely a node's neighbors are also neighbors).",
    'average_shortest_path_length': "The average number of steps required to get from one node to any other node.",
    'degree_assortativity': "The correlation between a node's degree and the degree of its neighbors (positive = high-degree nodes connect to other high-degree nodes).",
    'avg_betweenness_centrality': "The average frequency that nodes act as a bridge along the shortest path between two other nodes.",
    'max_degree': "The highest number of connections held by a single node in the graph (the 'hub' size).",
    'min_degree': "The lowest number of connections held by a single node.",
    'degree_std': "The standard deviation of degrees; measures how much variation there is in connectivity (high = mixture of hubs and leaves).",
    'transitivity': "The overall probability that two neighbors of a node are connected (similar to clustering but calculated globally).",
    'radius': "The minimum eccentricity in the graph (the shortest distance from the 'center' of the graph to the furthest node).",
    'avg_degree_centrality': "The average fraction of the total possible nodes that any given node is connected to.",
    'max_degree_centrality': "The highest centrality score; indicates the most central or well-connected node relative to network size.",
    'max_betweenness_centrality': "The score of the node that acts as the most critical bridge or bottleneck in the network.",
    'avg_closeness_centrality': "The average speed at which nodes can access all other nodes (inverse of average distance).",
    'max_closeness_centrality': "The score of the node that can reach all other nodes in the fewest number of steps."
}


GRAPH_PROPERTY_COLUMNS = [  
    'n_nodes',
    'n_edges',
    'is_directed',
    'density',
    'is_connected',
    'avg_degree',
    'max_degree',
    'min_degree',
    'degree_std',
    'degree_assortativity',
    'average_clustering',
    'transitivity',
    'diameter',
    'radius',
    'average_shortest_path_length',
    'avg_degree_centrality',
    'max_degree_centrality',
    'avg_betweenness_centrality',
    'max_betweenness_centrality',
    'avg_closeness_centrality',
    'max_closeness_centrality'
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
            display(Image(str(path)))
            print(f"[cache] Loaded: {Path(path).name}")
            return True
        except ImportError:
            pass
    return False


def load_batch_info(batch_dir) -> dict:
    """Read batch_info.json from batch_dir; returns name-only fallback if not found."""
    path = Path(batch_dir) / "batch_info.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"name": Path(batch_dir).name, "description": ""}


def create_batch_info(batch_dir, name, description, graph_types=None, r_values=None,
                      n_repeats=None, n_nodes_range=None, notes="") -> dict:
    """Write batch_info.json in batch_dir. Safe to re-run -- overwrites existing file."""
    info = {
        "name": name,
        "description": description,
        "date_created": datetime.now().strftime("%Y-%m-%d"),
        "graph_types": graph_types or [],
        "r_values": r_values or [],
        "n_repeats": n_repeats,
        "n_nodes_range": n_nodes_range or {},
        "notes": notes,
    }
    path = Path(batch_dir) / "batch_info.json"
    with open(path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"[batch_info] Written: {path}")
    return info


def _stamp_batch(fig, batch_name: str) -> None:
    """Add a source label to the bottom-right corner of the figure."""
    fig.text(
        0.99, 0.01, f"source: {batch_name}",
        fontsize=8, color="#666666", ha="right", va="bottom",
        style="italic", transform=fig.transFigure,
    )


def generate_robust_color_dict(df, existing_colors, default_palette='husl'):
    """
    Generates a color dictionary ensuring all categories have a unique color.
    Uses existing colors where available, and generates distinct colors for new ones.
    """
    # 1. Get unique, non-null categories
    categories = sorted(df['category'].dropna().unique().tolist())
    
    # 2. Separate known and unknown categories
    known_cats = [c for c in categories if c in existing_colors]
    unknown_cats = [c for c in categories if c not in existing_colors]
    
    # 3. Initialize the final dictionary with known colors
    final_color_dict = {c: existing_colors[c] for c in known_cats}
    
    # 4. Handle unknown categories
    if unknown_cats:
        num_unknown = len(unknown_cats)
        
        # Strategy A: If few unknowns, generate a nicely spaced palette
        if num_unknown <= 20: 
            # 'husl' creates perceptually distinct colors
            new_colors = sns.color_palette(default_palette, n_colors=num_unknown)
            
            for i, cat in enumerate(unknown_cats):
                final_color_dict[cat] = mcolors.to_hex(new_colors[i])
                
        # Strategy B: If many unknowns, use a deterministic hash to pick colors
        # This prevents the palette from becoming an indistinguishable rainbow
        else:
             # Use a very large palette to draw from
            large_palette = sns.color_palette("hls", 50) 
            
            for cat in unknown_cats:
                # Create a deterministic integer from the category name
                hash_val = int(hashlib.md5(cat.encode('utf-8')).hexdigest(), 16)
                # Pick a color from the palette based on the hash
                color_idx = hash_val % len(large_palette)
                final_color_dict[cat] = mcolors.to_hex(large_palette[color_idx])
                
    return final_color_dict


def aggregate_results_no_load(batch_dir, delete_temp=False, output_file=None):
    """
    Version that doesn't load the result into memory - just creates the file.
    Use this for very large datasets where you don't need the DataFrame immediately.
    
    Returns the path to the output file instead of a DataFrame.
    """
    batch_path = Path(batch_dir)
    if not output_file:
        output_file = batch_path / "full_results.csv"
    else:
        output_file = Path(output_file)
    
    tmp_results_path = batch_path / "tmp" / "results"
    
    # Check if already done
    if output_file.exists():
        print(f"File {output_file} already exists!")
        return output_file

    # Find and sort files
    all_files = sorted(
        tmp_results_path.glob("result_job_*.csv"),
        key=lambda p: int(p.stem.split('_')[-1])
    )
    
    if not all_files:
        print(f"No result files found in {tmp_results_path}")
        return None

    print(f"Found {len(all_files)} files. Aggregating...")

    # Stream aggregation
    try:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for i, fpath in enumerate(all_files):
                if i > 0 and i % 100 == 0:
                    print(f"  Processed {i}/{len(all_files)} files...")
                
                with open(fpath, 'r', encoding='utf-8') as infile:
                    if i == 0:
                        shutil.copyfileobj(infile, outfile)
                    else:
                        next(infile)
                        shutil.copyfileobj(infile, outfile)
        
        print(f"✓ Master file saved at: {output_file}")
    
    except Exception as e:
        print(f"Error during aggregation: {e}")
        if output_file.exists():
            output_file.unlink()
        raise

    # Delete temp files
    if delete_temp:
        tmp_dir = batch_path / "tmp"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
            print(f"Deleted temporary directory: {tmp_dir}")
    
    # Return path instead of loading into memory
    return output_file


def plot_steps_violin(
    results_csv_path,
    df_graphs,
    color_dict=None,
    categories=None,
    figures_dir=None,
    force_recompute=False,
    batch_name=None,
):
    """
    Violin plot of steps-to-fixation distribution, one violin per graph category.

    Uses polars lazy scan so only three columns are pulled from the (potentially huge) CSV.
    categories controls the x-axis order; defaults to sorted unique values in df_graphs.
    """
    import polars as pl

    if color_dict is None:
        color_dict = {}

    fig_path = _resolve_figure_path(figures_dir, 'plot_steps_violin')
    if not force_recompute and try_load_cached(fig_path):
        return

    if categories is None:
        categories = sorted(df_graphs['category'].dropna().unique().tolist())

    merged_raw = (
        pl.scan_csv(results_csv_path)
        .select(['wl_hash', 'steps', 'fixation'])
        .with_columns(
            pl.when(pl.col('fixation')).then(pl.col('steps')).otherwise(None).alias('steps_success')
        )
        .join(
            pl.from_pandas(df_graphs[['wl_hash', 'category']]).lazy(),
            on='wl_hash',
            how='left',
        )
        .collect()
        .to_pandas()
    )

    palette = {cat: color_dict[cat] for cat in categories if cat in color_dict}

    fig, ax = plt.subplots(figsize=(max(12, len(categories) * 1.1), 7))
    sns.violinplot(
        data=merged_raw,
        x='category',
        y='steps_success',
        order=categories,
        palette=palette,
        inner='box',
        linewidth=1.2,
        ax=ax,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    ax.set_xlabel('Category', fontsize=13)
    ax.set_ylabel('Steps to Fixation', fontsize=13)
    ax.set_title('Distribution of Steps to Fixation by Category', fontsize=14)
    if batch_name:
        _stamp_batch(fig, batch_name)
    fig.tight_layout()
    if fig_path is not None:
        fig.savefig(fig_path, bbox_inches='tight', dpi=150)
        print(f"[cache] Saved: {fig_path.name}")
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
):
    """
    Histogram of a steps/outcome metric, optionally filtered to one graph category.

    metric: column in df to histogram (default: 'mean_steps')
    category: if given, restricts to rows where df['category'] == category; None = all graphs
    color_dict: category -> color mapping; uses the category's color for the bars when available
    """
    if color_dict is None:
        color_dict = {}

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

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(data, bins=bins, color=bar_color, edgecolor='black', alpha=0.7)
    ax.set_xlabel(metric_label, fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Distribution of {metric_label} — {label}', fontsize=14)
    ax.grid(axis='y', alpha=0.3)

    if batch_name:
        _stamp_batch(fig, batch_name)
    fig.tight_layout()
    if fig_path is not None:
        fig.savefig(fig_path, bbox_inches='tight', dpi=150)
        print(f"[cache] Saved: {fig_path.name}")
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
):
    """
    Improved replacement for plot_hybrid_density.

    Fixes over the original:
    - 1/N neutral line: draws y=1/x curve when x_prop='n_nodes'; draws axhline
      only when N is homogeneous across graphs; silently omits it when N varies
      widely (a flat line would be misleading).
    - Auto-detects whether x is discrete enough to warrant violins -- no manual
      with_violin flag needed.
    - Vectorized jitter (numpy) instead of slow row-wise apply().
    - Consistent fig/ax API throughout (no mixed plt.*/ax.* state).
    - Single clean title; description embedded in x-axis label.
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
    fig, ax = plt.subplots(figsize=(9, 6.5))

    dense_x_values = set()
    if is_discrete_x:
        counts = valid_x.value_counts()
        dense_x_values = set(counts[counts >= density_threshold].index)

        # Violin width: smallest gap between any two adjacent x positions
        all_x_sorted = sorted(valid_x.unique())
        if len(all_x_sorted) > 1:
            violin_width = float(np.min(np.diff(all_x_sorted))) * 0.7
        else:
            violin_width = 0.5
        violin_width = max(0.01, violin_width)

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
    sns.scatterplot(
        data=plot_df, ax=ax,
        x='x_jittered', y=y_outcome,
        hue='category',
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
                hue='category',
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
    ax.set_title(f'{xlabel_base}  →  {ylabel}', fontsize=13, pad=8)
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

    # --- 12. Legend with highlight styling ---
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
    ax.legend(handles=handles, labels=labels_leg,
              bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., fontsize=9)

    if batch_name:
        _stamp_batch(fig, batch_name)
    fig.tight_layout()
    if fig_path is not None:
        fig.savefig(fig_path, bbox_inches='tight', dpi=150)
        print(f"[cache] Saved: {fig_path.name}")
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
    """
    if color_dict is None:
        color_dict = {}

    fig_path = _resolve_figure_path(figures_dir, 'plot_two_property_effect',
                                    x=x_prop, y=y_prop, outcome=outcome)
    if not force_recompute and try_load_cached(fig_path):
        return

    cols = [x_prop, y_prop, outcome, 'category']
    plot_df = df[cols].replace([np.inf, -np.inf], np.nan).dropna()

    if plot_df.empty:
        print(f"No valid data for ({x_prop}, {y_prop}) -> {outcome}")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

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
        f"{y_prop.replace('_', ' ').title()}\non {outcome_label}",
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
    """
    if color_dict is None:
        color_dict = {}

    fig_path = _resolve_figure_path(figures_dir, 'plot_two_property_effect_hexbin',
                                    x=x_prop, y=y_prop, outcome=outcome)
    if not force_recompute and try_load_cached(fig_path):
        return

    cols = [x_prop, y_prop, outcome, 'category']
    plot_df = df[cols].replace([np.inf, -np.inf], np.nan).dropna()

    if plot_df.empty:
        print(f"No valid data for ({x_prop}, {y_prop}) -> {outcome}")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

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
        f" (hex={reduce_name})",
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
    plt.show()
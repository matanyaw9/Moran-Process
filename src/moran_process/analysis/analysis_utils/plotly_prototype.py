"""
PROTOTYPE: a Plotly port of ``plot_two_property_effect`` (the most
annotation-heavy static figure) for evaluating whether interactive figures are
worth adopting. Not wired into the package __init__ and not used by anything;
delete freely if the experiment doesn't pan out.

What it deliberately REUSES from the matplotlib code (proves these survive a
port untouched):
  - ``_outcome_color_norm``       : the LogNorm-vs-linear decision for skewed outcomes
  - ``basic_moran_fixation_prob`` : the analytic Moran reference curve

What it has to RE-IMPLEMENT (the real porting cost; matplotlib gave this free):
  - colorbar tick mapping     : plotly markers have no LogNorm, so we normalize
                                colors to [0,1] ourselves and rebuild the colorbar
                                tick labels back in data units

What it INTENTIONALLY drops for this prototype (no automatic plotly equivalent;
each is a manual paper-space annotation, skipped per the eval scope):
  - the Pearson correlation box
  - the property-description glosses under the axes
  - the PNG cache (try_load_cached / _resolve_figure_path): interactive figures
    are HTML, not PNG, so the existing cache model does not apply. Writes .html.

Run it:
    uv run python -m moran_process.analysis.analysis_utils.plotly_prototype
    # -> writes two_property_prototype.html in the cwd; open in a browser.

Or from a notebook with real data:
    from moran_process.analysis.analysis_utils.plotly_prototype import plot_two_property_effect_plotly
    fig = plot_two_property_effect_plotly(df, 'n_nodes', 'prob_fixation',
                                          outcome='mean_steps',
                                          highlight_categories=['Mammalian', 'Avian'],
                                          color_dict=CATEGORY_COLOR_DICT)
    fig.show()            # interactive in the notebook
"""
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
import plotly.graph_objects as go

from .colors import CATEGORY_COLOR_DICT, DEFAULT_FIG_SIZE, _sort_categories
from .plots import _outcome_color_norm, basic_moran_fixation_prob


def _colorbar_ticks(norm):
    """Reconstruct a data-unit colorbar for colors we pre-normalized to [0,1].

    matplotlib's colorbar reads the norm and labels itself in real units. Here we
    feed plotly normalized colors (so a LogNorm actually shows), which means the
    colorbar would otherwise read 0..1. So we pick representative data values,
    push them through the SAME norm to get their [0,1] positions, and hand plotly
    (tickvals, ticktext) to relabel the bar in data units. This is the single
    biggest piece of free matplotlib behavior we have to rebuild by hand.
    """
    is_log = isinstance(norm, mcolors.LogNorm)
    vmin, vmax = float(norm.vmin), float(norm.vmax)
    if is_log:
        data_ticks = np.logspace(np.log10(vmin), np.log10(vmax), 5)
    else:
        data_ticks = np.linspace(vmin, vmax, 5)
    tickvals = [float(norm(v)) for v in data_ticks]          # -> [0,1] positions
    ticktext = [f"{v:.3g}" for v in data_ticks]              # -> real-unit labels
    return tickvals, ticktext


def plot_two_property_effect_plotly(
    df,
    x_prop,
    y_prop,
    outcome='mean_steps',
    color_dict=None,
    colorscale='Viridis',
    highlight_categories=None,
    fig_title=None,
    html_path=None,
):
    """Interactive twin of ``plot_two_property_effect``. Returns a plotly Figure.

    Same leading signature as the static version. Color encodes ``outcome`` on the
    same norm; each point is a graph and hovering reveals its category and exact
    outcome value (the capability the static figure cannot offer).
    """
    if color_dict is None:
        color_dict = {}

    r_vals = sorted(df['r'].dropna().unique().tolist()) if 'r' in df.columns else []
    r_value = r_vals[0] if len(r_vals) == 1 else None
    r_suffix = f"  (r={r_vals[0]})" if len(r_vals) == 1 else ""

    cols = [x_prop, y_prop, outcome, 'category']
    plot_df = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    if plot_df.empty:
        raise ValueError(f"No valid data for ({x_prop}, {y_prop}) -> {outcome}")

    # --- color: reuse the matplotlib norm, then normalize to [0,1] for plotly ---
    norm = _outcome_color_norm(plot_df[outcome])
    norm_colors = norm(plot_df[outcome].to_numpy())          # array in ~[0,1]
    tickvals, ticktext = _colorbar_ticks(norm)
    cbar_label = outcome.replace("_", " ").title()

    fig = go.Figure()

    # Base scatter: one point per graph. customdata carries category + the RAW
    # outcome so the hover label shows real units, not the normalized color.
    fig.add_trace(go.Scattergl(
        x=plot_df[x_prop], y=plot_df[y_prop],
        mode='markers',
        marker=dict(
            size=7, opacity=0.6,
            color=norm_colors, colorscale=colorscale, cmin=0, cmax=1,
            colorbar=dict(
                title=cbar_label, tickvals=tickvals, ticktext=ticktext,
                len=0.55, y=1.0, yanchor='top',     # top ~55%, mirrors the static stack
            ),
        ),
        customdata=np.stack([plot_df['category'], plot_df[outcome]], axis=-1),
        hovertemplate=(
            f"<b>%{{customdata[0]}}</b><br>"
            f"{x_prop}: %{{x}}<br>"
            f"{y_prop}: %{{y}}<br>"
            f"{outcome}: %{{customdata[1]:.4g}}<extra></extra>"
        ),
        showlegend=False,
    ))

    # Highlight categories: one trace each, outlined in the category color, color
    # still on the shared outcome scale. Legend entries come for free from `name`.
    if highlight_categories:
        for cat in highlight_categories:
            grp = plot_df[plot_df['category'] == cat]
            if grp.empty:
                continue
            fig.add_trace(go.Scatter(
                x=grp[x_prop], y=grp[y_prop],
                mode='markers', name=cat,
                marker=dict(
                    size=13,
                    color=norm(grp[outcome].to_numpy()),
                    colorscale=colorscale, cmin=0, cmax=1,
                    line=dict(width=2, color=color_dict.get(cat, 'black')),
                ),
                customdata=np.stack([grp['category'], grp[outcome]], axis=-1),
                hovertemplate=(
                    f"<b>%{{customdata[0]}}</b><br>"
                    f"{x_prop}: %{{x}}<br>{y_prop}: %{{y}}<br>"
                    f"{outcome}: %{{customdata[1]:.4g}}<extra></extra>"
                ),
            ))

    # Analytic reference lines: only when x encodes N and y is fixation prob.
    if x_prop == 'n_nodes' and y_prop == 'prob_fixation':
        n_col = plot_df['n_nodes'].dropna()
        if len(n_col) > 0:
            xr = np.linspace(max(1, n_col.min()), n_col.max(), 300)
            fig.add_trace(go.Scatter(
                x=xr, y=1.0 / xr, mode='lines', name="Neutral  1/N",
                line=dict(color='black', dash='dash', width=1.6),
            ))
            if r_value is not None:
                fig.add_trace(go.Scatter(
                    x=xr, y=basic_moran_fixation_prob(xr, r_value),
                    mode='lines', name="Moran  ρ(N,r)",
                    line=dict(color='crimson', dash='dash', width=1.6),
                ))

    # NOTE: the correlation box and property glosses are intentionally omitted in
    # this prototype. They have no automatic plotly equivalent (they are manual
    # paper-space annotations + margin bookkeeping); per the eval scope we skip
    # the manual-only pieces for now and keep what plotly gives natively.

    default_title = (
        f"Combined effect of {x_prop.replace('_', ' ').title()} & "
        f"{y_prop.replace('_', ' ').title()} on {cbar_label}{r_suffix}"
    )
    fig.update_layout(
        title=fig_title or default_title,
        xaxis_title=x_prop.replace("_", " ").title(),
        yaxis_title=y_prop.replace("_", " ").title(),
        template='plotly_white',
        width=950, height=600,
        margin=dict(r=210),                      # room for the colorbar + legend
        legend=dict(x=1.02, y=0.40, xanchor='left'),   # under the colorbar
    )
    fig.update_xaxes(showgrid=True, griddash='dash', gridcolor='rgba(0,0,0,0.12)')
    fig.update_yaxes(showgrid=True, griddash='dash', gridcolor='rgba(0,0,0,0.12)')

    if html_path is not None:
        fig.write_html(html_path)
        print(f"[prototype] wrote {html_path}")
    return fig


def _hover(plot_df, x_prop, y_outcome):
    """Build (customdata, hovertemplate) that name the graph when those columns
    exist. The single capability the static figure cannot offer: hover a point
    and read which graph it is, not just its coordinates."""
    id_cols = [c for c in ('graph_name', 'wl_hash', 'category') if c in plot_df.columns]
    customdata = plot_df[id_cols].to_numpy() if id_cols else None
    id_lines = "".join(
        f"{c.replace('_', ' ')}: %{{customdata[{i}]}}<br>"
        for i, c in enumerate(id_cols)
    )
    template = (
        id_lines
        + f"{x_prop}: %{{x}}<br>{y_outcome}: %{{y:.4g}}<extra></extra>"
    )
    return customdata, template


def plot_outcome_vs_property_plotly(
    df,
    x_prop,
    y_outcome='prob_fixation',
    color_dict=None,
    density_threshold=50,
    highlight_categories=None,
    filter_categories=None,
    fig_title=None,
    html_path=None,
):
    """Interactive twin of ``plot_outcome_vs_property``. Returns a plotly Figure.

    Color encodes category (one trace each -> native legend). Violins are drawn
    for dense discrete x positions, exactly as the static version decides them.
    Jitter, the correlation box, and the property gloss are intentionally dropped
    (see the module docstring): hover and the violins replace what jitter bought.
    """
    if color_dict is None:
        color_dict = {}
    if filter_categories is not None:
        df = df[df['category'].isin(filter_categories)]

    r_values = sorted(df['r'].dropna().unique()) if 'r' in df.columns else []
    is_prob = (y_outcome == 'prob_fixation')
    ylabel = "Fixation Probability (P_fix)" if is_prob else y_outcome.replace('_', ' ').title()
    if x_prop == 'prob_fixation':
        xlabel = "Fixation Probability (P_fix)"
    elif x_prop == 'std_steps':
        xlabel = 'Std. Steps to Fixation'
    else:
        xlabel = x_prop.replace('_', ' ').title()

    plot_df = df.copy()
    is_numeric_x = pd.api.types.is_numeric_dtype(plot_df[x_prop])
    if is_numeric_x:
        plot_df[x_prop] = pd.to_numeric(plot_df[x_prop], errors='coerce')
        plot_df['x_plot'] = plot_df[x_prop].round(3)
    else:
        plot_df = plot_df.dropna(subset=[x_prop])
        plot_df['x_plot'] = plot_df[x_prop]            # plotly handles categorical x natively

    # Dense-x detection: same rule as the static figure (numeric x only, few
    # unique values relative to the data, each with >= density_threshold points).
    valid_x = plot_df['x_plot'].dropna()
    n_unique, n_total = valid_x.nunique(), len(valid_x)
    is_discrete_x = is_numeric_x and (n_unique <= max(20, n_total * 0.02))
    dense_x_values, violin_width = set(), 0.5
    if is_discrete_x and density_threshold is not None:
        counts = valid_x.value_counts()
        dense_x_values = set(counts[counts >= density_threshold].index)
        all_x_sorted = sorted(valid_x.unique())
        if len(all_x_sorted) > 1:                      # proportional width, ported verbatim
            diffs = np.diff(all_x_sorted)
            total_span = (all_x_sorted[-1] - all_x_sorted[0]) or 1.0
            valid_gaps = diffs[diffs > total_span * 0.02]
            dist_basis = np.min(valid_gaps) if len(valid_gaps) else total_span * 0.1
            violin_width = dist_basis * 0.7

    fig = go.Figure()

    # Violins first so the scatter sits on top. Native go.Violin, no manual KDE.
    for x_val in sorted(dense_x_values):
        ys = plot_df.loc[plot_df['x_plot'] == x_val, y_outcome].dropna()
        if len(ys):
            fig.add_trace(go.Violin(
                x=[x_val] * len(ys), y=ys, width=violin_width,
                points=False, hoverinfo='skip', showlegend=False,
                line_color='lightgray', fillcolor='whitesmoke', opacity=0.9,
                meanline_visible=False,
            ))

    # One scatter trace per category -> native, clickable legend.
    # Two orderings, deliberately opposite: legend reads in _sort_categories order
    # (biological first, 'Random' last), but plotly paints last-on-top, so we ADD
    # traces in reverse (Random first -> back, biological last -> front) and pin the
    # legend back to canonical order with legendrank. Without this the large, pale
    # Random cloud would cover the biological points.
    customdata, template = _hover(plot_df, x_prop, y_outcome)
    highlight_categories = highlight_categories or []
    legend_order = _sort_categories(plot_df['category'].dropna().unique().tolist())

    # With several r values in one frame, color still encodes category while marker
    # SHAPE encodes r, so both are readable at once. The symbol is attached per point
    # (marker.symbol accepts an array), keeping each category a single trace and the
    # category legend clean; the shape -> r mapping is shown via proxy traces below.
    multi_r = len(r_values) > 1
    R_SYMBOLS = ['circle', 'diamond', 'square', 'triangle-up', 'cross', 'x', 'star', 'hexagon']
    r_symbol = {r: R_SYMBOLS[i % len(R_SYMBOLS)] for i, r in enumerate(r_values)}

    for cat in reversed(legend_order):
        m = (plot_df['category'] == cat).to_numpy()
        line = dict(width=1.8, color='black') if cat in highlight_categories else dict(width=0.5, color='white')
        marker = dict(size=8, color=color_dict.get(cat, 'lightgray'), opacity=0.75, line=line)
        if multi_r:
            marker['symbol'] = plot_df.loc[m, 'r'].map(r_symbol).to_numpy()
        fig.add_trace(go.Scatter(
            x=plot_df['x_plot'][m], y=plot_df[y_outcome][m],
            mode='markers', name=cat,
            legendrank=legend_order.index(cat),
            marker=marker,
            # When symbol is a per-point array (r-encoding), the legend swatch would
            # pick element [0]'s shape, so categories would show mixed shapes. Hide the
            # data trace from the legend and add a fixed-circle color proxy below, so
            # the category legend reflects only color.
            showlegend=not multi_r,
            customdata=customdata[m] if customdata is not None else None,
            hovertemplate=template,
        ))

    # Category legend (color only): one circle swatch per category, decoupled from the
    # per-point r-shapes above. Only needed in the multi-r case.
    if multi_r:
        for cat in legend_order:
            line = dict(width=1.8, color='black') if cat in highlight_categories else dict(width=0.5, color='white')
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode='markers', name=cat,
                legendrank=legend_order.index(cat),
                marker=dict(size=8, symbol='circle', color=color_dict.get(cat, 'lightgray'),
                            opacity=0.75, line=line),
                hoverinfo='skip',
            ))

    # Proxy traces (no data points) that label which marker shape maps to which r.
    if multi_r:
        for i, r in enumerate(r_values):
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode='markers', name=f"r = {r:g}",
                legendrank=1000 + i,
                legendgroup='r_legend', legendgrouptitle_text='r (mutant relative fitness)',
                marker=dict(size=8, symbol=r_symbol[r], color='dimgray',
                            line=dict(width=0.5, color='white')),
                hoverinfo='skip',
            ))

    # Neutral 1/N reference logic, conditions ported exactly from the static fig.
    if is_prob and 'n_nodes' in plot_df.columns:
        n_col = plot_df['n_nodes'].dropna()
        if len(n_col) > 0:
            if x_prop == 'n_nodes':
                xr = np.linspace(max(1, n_col.min()), n_col.max(), 300)
                fig.add_trace(go.Scatter(x=xr, y=1.0 / xr, mode='lines',
                    name="Neutral  1/N", line=dict(color='black', dash='dash', width=1.4)))
                if len(r_values) == 1:
                    fig.add_trace(go.Scatter(x=xr, y=basic_moran_fixation_prob(xr, r_values[0]),
                        mode='lines', name="Moran  ρ(N,r)",
                        line=dict(color='royalblue', dash='dash', width=1.4)))
            else:
                n_mean = n_col.mean()
                if n_mean > 0 and (n_col.std() / n_mean) < 0.05:     # only when N homogeneous
                    fig.add_hline(y=1.0 / n_mean, line=dict(color='black', dash='dot', width=1.2),
                                  annotation_text=f"Neutral (1/N={n_mean:.0f})")

    r_suffix = f"  (r={r_values[0]})" if len(r_values) == 1 else ""
    fig.update_layout(
        title=fig_title or f"{xlabel}  →  {ylabel}{r_suffix}",
        xaxis_title=xlabel, yaxis_title=ylabel,
        # Match the matplotlib plot_outcome_vs_property: same DEFAULT_FIG_SIZE
        # (inches) scaled by 100 px/inch, the factor its inline display uses.
        template='plotly_white',
        width=round(DEFAULT_FIG_SIZE[0] * 100), height=round(DEFAULT_FIG_SIZE[1] * 100),
        margin=dict(r=180), legend=dict(x=1.02, y=1.0, xanchor='left'),
        violingap=0, violinmode='overlay',
    )
    fig.update_xaxes(showgrid=True, griddash='dash', gridcolor='rgba(0,0,0,0.12)')
    fig.update_yaxes(showgrid=True, griddash='dash', gridcolor='rgba(0,0,0,0.12)')
    if html_path is not None:
        fig.write_html(html_path)
        print(f"[prototype] wrote {html_path}")
    return fig


def _demo_dataframe(seed=0):
    """Synthetic graph-stats frame exercising the n_nodes vs prob_fixation case
    (so the analytic reference lines fire) with a right-skewed mean_steps (so the
    LogNorm path fires) and two highlightable categories."""
    rng = np.random.default_rng(seed)
    n = 400
    n_nodes = rng.integers(10, 300, n)
    prob_fixation = np.clip(1.0 / n_nodes + rng.normal(0, 0.01, n), 1e-4, 1)
    mean_steps = n_nodes ** 1.8 * rng.lognormal(0, 0.4, n)      # heavy right tail
    category = rng.choice(
        ['Random', 'Mammalian', 'Avian'], size=n, p=[0.8, 0.1, 0.1]
    )
    avg_degree = rng.choice([2, 3, 4, 5, 6], size=n)   # few unique -> triggers violins
    return pd.DataFrame(dict(
        n_nodes=n_nodes, prob_fixation=prob_fixation, avg_degree=avg_degree,
        mean_steps=mean_steps, category=category, r=1.5,
        graph_name=[f"{c.lower()}_{i}" for i, c in enumerate(category)],
    ))


if __name__ == '__main__':
    # Smoke test: build all three figures and write the HTML previews to a temp dir
    # (NOT the repo root). Each file inlines ~4.7 MB of plotly.js, so they are large,
    # throwaway artifacts -- regenerated on demand, never committed.
    import os
    import tempfile
    out_dir = tempfile.gettempdir()

    df = _demo_dataframe()
    plot_two_property_effect_plotly(
        df, 'n_nodes', 'prob_fixation', outcome='mean_steps',
        highlight_categories=['Mammalian', 'Avian'],
        color_dict=CATEGORY_COLOR_DICT,
        html_path=os.path.join(out_dir, 'two_property_prototype.html'),
    )
    # n_nodes vs prob_fixation: exercises the neutral 1/N + Moran reference curves.
    plot_outcome_vs_property_plotly(
        df, 'n_nodes', 'prob_fixation',
        color_dict=CATEGORY_COLOR_DICT, highlight_categories=['Mammalian', 'Avian'],
        html_path=os.path.join(out_dir, 'outcome_vs_property_refs.html'),
    )
    # avg_degree (discrete) vs prob_fixation: exercises the dense-x violins.
    plot_outcome_vs_property_plotly(
        df, 'avg_degree', 'prob_fixation', density_threshold=20,
        color_dict=CATEGORY_COLOR_DICT, highlight_categories=['Mammalian', 'Avian'],
        html_path=os.path.join(out_dir, 'outcome_vs_property_violins.html'),
    )

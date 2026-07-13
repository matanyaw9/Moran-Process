"""
Streamlit dashboard for presenting Moran-process simulation figures.

This is a thin presentation shell: every figure is produced by the *existing*
``moran_process.analysis`` functions. The app does not contain any plotting
logic of its own; it only discovers batches, loads their data the way the
analysis notebook does, and renders the figures those functions build.

Run it with::

    uv run streamlit run streamlit_app.py --server.port 8600

IMPORTANT: some batches' raw_results files are several GB. The "Fixation time"
figures and the speed report must scan the whole file, which is CPU/IO heavy.
Run this app from an ``inode`` session, NOT the WEXAC login node, or the
login-node watchdog will kill the scan (signal 16).

Why a page selector instead of tabs: Streamlit executes the body of *every*
st.tabs() on every rerun, even hidden ones. With multi-GB batches that meant the
heavy violin/p-value scans ran on load before anything was shown. A sidebar
radio runs only the selected page's code, so heavy work happens on demand.
"""

import io
import contextlib
from pathlib import Path

# Streamlit runs headless, so force the non-interactive Agg backend before any
# pyplot state is created. The plot_* functions call plt.show() internally; with
# show=False they never do, and we grab the figure they built via plt.gcf().
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pandas as pd
import streamlit as st

from moran_process.analysis.batch_speed_report import batch_speed_report

# _resolve_figure_path is the same path-builder the plot_* functions use
# internally; reusing it (rather than re-deriving the slug here) guarantees the
# app's cache filenames never drift from the ones the notebook writes.
from moran_process.analysis.analysis_utils.plots import _resolve_figure_path
from moran_process.analysis.analysis_utils import (
    CATEGORY_COLOR_DICT,
    GRAPH_PROPERTY_COLUMNS,
    generate_robust_color_dict,
    resolve_results_path,
    build_graph_statistics,
    load_batch_info,
    plot_batch_info_card,
    plot_steps_violin,
    plot_steps_pvalue_matrix,
    plot_steps_histogram,
    plot_outcome_vs_property,
    plot_two_property_effect,
    plot_two_property_effect_hexbin,
)

# PROTOTYPE: interactive (plotly) twin of plot_two_property_effect. Not part of
# the analysis_utils public API, so imported directly from its module.
from moran_process.analysis.analysis_utils.plotly_prototype import (
    plot_two_property_effect_plotly,
    plot_outcome_vs_property_plotly,
)

SIM_DATA_DIR = Path(__file__).parent / "simulation_data"
LOGO_PATH = Path(__file__).parent / "images" / "logo-5.png"

# Outcome columns produced by build_graph_statistics that make sense to plot.
OUTCOME_COLUMNS = [
    "prob_fixation",
    "mean_steps",
    "median_steps",
    "std_steps",
    "iqr_steps",
]

st.set_page_config(
    page_title="Moran Process Figures",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else None,
    layout="wide",
)


# --------------------------------------------------------------------------- #
# Data discovery + loading (cached)
# --------------------------------------------------------------------------- #
def _human_size(num_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num_bytes < 1024:
            return f"{num_bytes:.0f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.0f} PB"


def list_batches():
    """Batches with both an aggregated results file and graph_props.csv.

    Returns a list of (name, results_size_bytes) so the picker can warn about
    multi-GB batches before the user selects one.
    """
    if not SIM_DATA_DIR.exists():
        return []
    out = []
    for d in sorted(SIM_DATA_DIR.iterdir()):
        if not d.is_dir():
            continue
        rp = resolve_results_path(d)
        if rp is not None and (d / "graph_props.csv").exists():
            out.append((d.name, rp.stat().st_size))
    return out


@st.cache_data(show_spinner="Loading batch statistics...")
def load_batch(batch_name: str):
    """Load the light, aggregated data a batch's property figures need.

    This deliberately does NOT touch the multi-GB raw_results file beyond what
    build_graph_statistics already cached to graph_statistics.csv. Returns
    (df_graphs, analysis_df, results_path, color_dict, batch_info).
    """
    batch_dir = SIM_DATA_DIR / batch_name
    df_graphs = pd.read_csv(batch_dir / "graph_props.csv")
    results_path = resolve_results_path(batch_dir)
    analysis_df = build_graph_statistics(
        results_path, df_graphs, batch_dir / "graph_statistics.csv"
    )
    color_dict = generate_robust_color_dict(analysis_df, CATEGORY_COLOR_DICT)
    batch_info = load_batch_info(batch_dir)
    return df_graphs, analysis_df, str(results_path), color_dict, batch_info


@st.cache_data(show_spinner=False)
def load_speed_agg(results_path: str) -> pd.DataFrame:
    """Per-job sums of steps (and duration) via a streaming scan of raw_results.

    The raw file can be many GB, so we never load it into pandas. polars scans it
    lazily, keeps only the columns the speed report needs, and reduces to one row
    per job_id. Because batch_speed_report re-groups by job_id and sums, feeding
    it these pre-summed rows yields identical numbers.
    """
    import polars as pl

    p = Path(results_path)
    lf = pl.scan_parquet(str(p)) if p.suffix == ".parquet" else pl.scan_csv(str(p))
    cols = lf.collect_schema().names()
    select_cols = ["job_id", "steps"] + (["duration"] if "duration" in cols else [])
    aggs = [pl.col("steps").sum()] + (
        [pl.col("duration").sum()] if "duration" in cols else []
    )
    return (
        lf.select(select_cols)
        .group_by("job_id")
        .agg(aggs)
        .collect(engine="streaming")
        .to_pandas()
    )


# --------------------------------------------------------------------------- #
# Figure rendering helpers
# --------------------------------------------------------------------------- #
def _build_png(plot_fn, *args, spinner=None, save_kwargs=None, **kwargs):
    """Run a plot_* function silently and return the figure it built as PNG bytes.

    The plot functions create their own figure and would normally show/save it;
    we suppress both (show=False, save=False) and grab the figure they left on the
    pyplot stack. Returning bytes (rather than st.pyplot-ing here) lets the caller
    both display the image and, on demand, persist those *exact* bytes to disk
    without rebuilding the figure - which matters because the violin/p-value
    figures scan multi-GB files. A spinner gives feedback during those scans.
    """
    kwargs.setdefault("show", False)
    kwargs.setdefault("save", False)
    plt.close("all")
    ctx = st.spinner(spinner) if spinner else contextlib.nullcontext()
    try:
        with ctx:
            plot_fn(*args, **kwargs)
    except Exception as exc:  # surface the error in the page instead of a blank crash
        st.error(f"{plot_fn.__name__} failed: {exc}")
        return None
    if not plt.get_fignums():
        st.info("No figure produced for this selection (likely no matching data).")
        return None
    fig = plt.figure(plt.get_fignums()[-1])
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", **(save_kwargs or {"dpi": 150}))
    plt.close("all")
    return buf.getvalue()


def cached_figure(
    plot_fn,
    *args,
    cache_name,
    figures_dir,
    cache_key=None,
    spinner=None,
    save_kwargs=None,
    **kwargs,
):
    """Cache-aware figure rendering, mirroring the plot_* on-disk PNG cache.

    Two independent actions, driven entirely by whether freshly-built PNG bytes
    are stashed in session_state (no separate "edit mode" flag):
      * Regenerate -> rebuild the figure from the raw data NOW and show that
        fresh, unsaved version. Available both from the cached view and from an
        already-regenerated view (to rebuild again).
      * Save / Overwrite -> write the currently displayed (freshly regenerated)
        bytes to the cache, replacing the old PNG. Nothing touches disk until
        this is clicked.

    State machine:
      * Fresh PNG stashed   -> show it + Save / Regenerate / Discard.
      * Cache exists, no fresh -> show cached PNG + Regenerate.
      * No cache, no fresh  -> build once (nothing else to show) and show it.

    The cache path is built with the *same* _resolve_figure_path the plot_*
    functions call internally, so files are interchangeable with the notebook's.
    `cache_key` should hold everything that changes the figure (r, selected
    columns, ...) so distinct selections get distinct files. The stashed bytes
    let the Save click write instantly instead of triggering a second
    (possibly multi-GB) scan.
    """
    fig_path = _resolve_figure_path(figures_dir, cache_name, **(cache_key or {}))
    base = str(fig_path)
    png_key = f"png::{base}"

    def regenerate():
        """Rebuild from data and stash the bytes; rerun so they render."""
        png = _build_png(
            plot_fn, *args, spinner=spinner, save_kwargs=save_kwargs, **kwargs
        )
        if png is not None:
            st.session_state[png_key] = png
            st.rerun()
        # png is None -> _build_png already surfaced why (no data / error).

    # Cached view: show the saved PNG; Regenerate rebuilds from data on demand.
    if not st.session_state.get(png_key) and fig_path.exists():
        st.image(str(fig_path), width="stretch")
        st.caption(f"Cached: {fig_path.name}")
        if st.button("Regenerate", key=f"btn-regen::{base}"):
            regenerate()
        return

    # No cache and nothing stashed: build once, since there is nothing to show.
    if not st.session_state.get(png_key):
        png = _build_png(
            plot_fn, *args, spinner=spinner, save_kwargs=save_kwargs, **kwargs
        )
        if png is None:
            return
        st.session_state[png_key] = png

    # Fresh, unsaved figure: show it and offer Save / Regenerate / Discard.
    png = st.session_state[png_key]
    st.image(png, width="stretch")
    st.caption("Regenerated from data (not saved)")
    cols = st.columns(3 if fig_path.exists() else 2)
    save_label = "Save / Overwrite" if fig_path.exists() else "Save to cache"
    if cols[0].button(save_label, key=f"btn-save::{base}", type="primary"):
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        fig_path.write_bytes(png)
        st.session_state.pop(png_key, None)
        st.toast(f"Saved {fig_path.name}")
        st.rerun()
    if cols[1].button("Regenerate", key=f"btn-regen2::{base}"):
        st.session_state.pop(png_key, None)
        regenerate()
    if fig_path.exists() and cols[2].button("Discard", key=f"btn-discard::{base}"):
        st.session_state.pop(png_key, None)
        st.rerun()


def render_speed_report(batch_name: str, results_path: str):
    """Render batch_speed_report's printed stats + histograms inside the page.

    Skipped with a message when the batch has no logs/ dir (the report parses LSF
    .out files for run time / memory and is meaningless without them).
    """
    batch_dir = SIM_DATA_DIR / batch_name
    if not (batch_dir / "logs").is_dir():
        st.info("No `logs/` directory for this batch, so no speed report is available.")
        return
    with st.spinner("Scanning raw results for per-job speed stats..."):
        slim = load_speed_agg(results_path)
        plt.close("all")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            batch_speed_report(batch_name, slim, batch_dir)
    text = buf.getvalue()
    if text.strip():
        st.code(text)
    if plt.get_fignums():
        st.pyplot(plt.figure(plt.get_fignums()[-1]))
        plt.close("all")


# --------------------------------------------------------------------------- #
# Sidebar: batch + page selection
# --------------------------------------------------------------------------- #
batches = list_batches()
if not batches:
    st.error(f"No batches with results found under {SIM_DATA_DIR}")
    st.stop()

st.sidebar.title("Moran Process Figures")

# Default to a moderately-sized batch with a parquet results file when present,
# so a fresh load does not immediately hit a 12 GB scan.
names = [n for n, _ in batches]
default_idx = next(
    (i for i, (n, _) in enumerate(batches) if n == "2026-06-10_scaling_study_6"), 0
)
batch_name = st.sidebar.selectbox(
    "Batch",
    names,
    index=default_idx,
    format_func=lambda n: f"{n}  ({_human_size(dict(batches)[n])})",
)
if dict(batches)[batch_name] > 2 * 1024**3:
    st.sidebar.warning(
        "This batch's raw results are >2 GB. The speed report and fixation-time "
        "figures will be slow; run from an `inode`, not the login node."
    )

page = st.sidebar.radio("Page", ["Overview", "Fixation time", "Property effects"])

df_graphs, analysis_df, results_path, color_dict, batch_info = load_batch(batch_name)

# Where cached figure PNGs live for this batch. Same `figures_dir` you'd pass the
# plot_* functions in a notebook, so the caches are shared in both directions.
figures_dir = SIM_DATA_DIR / batch_name / "figures"

r_values = (
    sorted(analysis_df["r"].dropna().unique().tolist()) if "r" in analysis_df else []
)
categories = sorted(analysis_df["category"].dropna().unique().tolist())

st.sidebar.markdown("---")
selected_r = (
    st.sidebar.selectbox("Selection coefficient r", r_values, index=0)
    if r_values
    else None
)
st.sidebar.caption(
    f"{len(df_graphs):,} graphs · {len(categories)} categories · {len(r_values)} r values"
)

# analysis_df restricted to the chosen r, for the per-r property figures.
df_r = (
    analysis_df[analysis_df["r"] == selected_r]
    if selected_r is not None
    else analysis_df
)


# --------------------------------------------------------------------------- #
# Pages (only the selected one runs -> heavy work is on demand)
# --------------------------------------------------------------------------- #
if page == "Overview":
    st.subheader(batch_info.get("name", batch_name))
    if batch_info.get("description"):
        st.write(batch_info["description"])
    cached_figure(
        plot_batch_info_card,
        batch_info,
        cache_name="batch_info_card",
        figures_dir=figures_dir,
        save_kwargs={"dpi": 200, "facecolor": "white"},
    )

    st.markdown("### Run speed & resource usage")
    render_speed_report(batch_name, results_path)

elif page == "Fixation time":
    st.caption(f"Showing r = {selected_r}" if selected_r is not None else "")

    st.markdown("#### Steps-to-fixation distribution by category")
    cached_figure(
        plot_steps_violin,
        results_path,
        df_graphs,
        cache_name="plot_steps_violin",
        cache_key={"r": selected_r},
        figures_dir=figures_dir,
        color_dict=color_dict,
        r=selected_r,
        batch_name=batch_name,
        spinner="Building violins (scanning raw results)...",
    )

    st.markdown("#### Pairwise significance (Mann-Whitney, effect size)")
    cached_figure(
        plot_steps_pvalue_matrix,
        results_path,
        df_graphs,
        cache_name="plot_steps_pvalue_matrix",
        cache_key={"r": selected_r},
        figures_dir=figures_dir,
        r=selected_r,
        batch_name=batch_name,
        spinner="Running pairwise Mann-Whitney tests...",
    )

    st.markdown("#### Per-graph metric histogram")
    c1, c2 = st.columns(2)
    hist_metric = c1.selectbox("Metric", OUTCOME_COLUMNS, key="hist_metric")
    hist_cat = c2.selectbox("Category", ["All"] + categories, key="hist_cat")
    cached_figure(
        plot_steps_histogram,
        df_r,
        cache_name="plot_steps_histogram",
        cache_key={"r": selected_r, "metric": hist_metric, "category": hist_cat},
        figures_dir=figures_dir,
        metric=hist_metric,
        category=None if hist_cat == "All" else hist_cat,
        color_dict=color_dict,
        batch_name=batch_name,
    )

elif page == "Property effects":
    st.caption(f"Showing r = {selected_r}" if selected_r is not None else "")

    st.markdown("#### Outcome vs. a single structural property")
    available_props = [p for p in GRAPH_PROPERTY_COLUMNS if p in df_r.columns]
    c1, c2 = st.columns(2)
    x_prop = c1.selectbox("X property", available_props, key="ovp_x")
    y_outcome = c2.selectbox("Y outcome", OUTCOME_COLUMNS, key="ovp_y")
    ovp_style = st.radio(
        "Style", ["static", "interactive"], horizontal=True, key="ovp_style"
    )
    if ovp_style == "interactive":
        # PROTOTYPE: plotly twin. Built from df_r (light), so it skips the PNG
        # cache. Hover a point for its graph name / category; biological
        # topologies are outlined; dense discrete x values get violins.
        try:
            fig = plot_outcome_vs_property_plotly(
                df_r,
                x_prop,
                y_outcome=y_outcome,
                color_dict=color_dict,
                highlight_categories=[
                    c for c in ("Mammalian", "Avian", "Fish") if c in categories
                ],
            )
            st.plotly_chart(fig, width="stretch")
        except Exception as exc:
            st.error(f"interactive figure failed: {exc}")
    else:
        cached_figure(
            plot_outcome_vs_property,
            df_r,
            x_prop,
            cache_name="plot_outcome_vs_property",
            cache_key={"r": selected_r, "x": x_prop, "y": y_outcome},
            figures_dir=figures_dir,
            y_outcome=y_outcome,
            color_dict=color_dict,
            batch_name=batch_name,
        )

    st.markdown("#### Combined effect of two properties")
    # X, Y and color can each show either a structural trait or a simulation
    # result. Order traits first, then results, and prefix each label with its
    # group so the flat selectbox reads as two grouped sections.
    available_outcomes = [o for o in OUTCOME_COLUMNS if o in df_r.columns]
    tp_options = available_props + available_outcomes

    def _grouped_label(col):
        group = "Result" if col in OUTCOME_COLUMNS else "Property"
        return f"{group} · {col.replace('_', ' ').title()}"

    def _opt_index(col):
        return tp_options.index(col) if col in tp_options else 0

    c1, c2, c3 = st.columns(3)
    tp_x = c1.selectbox(
        "X axis",
        tp_options,
        key="tp_x",
        index=_opt_index("n_nodes"),
        format_func=_grouped_label,
    )
    tp_y = c2.selectbox(
        "Y axis",
        tp_options,
        key="tp_y",
        index=_opt_index("prob_fixation"),
        format_func=_grouped_label,
    )
    tp_outcome = c3.selectbox(
        "Color by",
        tp_options,
        key="tp_outcome",
        index=_opt_index("mean_steps"),
        format_func=_grouped_label,
    )
    style = st.radio(
        "Style", ["scatter", "hexbin", "interactive"], horizontal=True, key="tp_style"
    )
    if style == "interactive":
        # PROTOTYPE: plotly twin of the scatter. Built straight from df_r (light,
        # already loaded), so it skips the PNG cache entirely - it is interactive
        # in the browser, there is nothing to persist. Hover a point to read its
        # category and exact outcome; biological topologies are outlined.
        try:
            fig = plot_two_property_effect_plotly(
                df_r,
                tp_x,
                tp_y,
                outcome=tp_outcome,
                color_dict=color_dict,
                highlight_categories=[
                    c for c in ("Mammalian", "Avian", "Fish") if c in categories
                ],
            )
            st.plotly_chart(fig, width="stretch")
        except Exception as exc:  # mirror cached_figure: show the error, don't crash
            st.error(f"interactive figure failed: {exc}")
    else:
        if style == "scatter":
            plot_fn, tp_cache_name = (
                plot_two_property_effect,
                "plot_two_property_effect",
            )
        else:
            plot_fn, tp_cache_name = (
                plot_two_property_effect_hexbin,
                "plot_two_property_effect_hexbin",
            )
        cached_figure(
            plot_fn,
            df_r,
            tp_x,
            tp_y,
            cache_name=tp_cache_name,
            cache_key={"r": selected_r, "x": tp_x, "y": tp_y, "outcome": tp_outcome},
            figures_dir=figures_dir,
            outcome=tp_outcome,
            color_dict=color_dict,
            batch_name=batch_name,
            descriptions_below=True,
        )

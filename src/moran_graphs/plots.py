"""Figures for finished batches. Reads only the report written by `hpc finalize`.

    python -m moran_graphs.plots <batch> [<batch> ...] [--out DIR]

Several batches are stitched by concatenation with a `batch` column. A graph that sits in
two batches stays two rows (the runs are not pooled), so it is drawn twice.
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns

from moran_graphs.engine import HIST_BINS
from moran_graphs.hpc import DATA_ROOT, batch_path, load_report
from moran_graphs.plan import load_plan
from moran_graphs.properties import properties

# Set on import, so notebooks that import plots draw in the same style as the saved files.
sns.set_theme(style="whitegrid", context="notebook", font_scale=1.2)
# TrueType fonts in PDFs: journals reject the default Type 3, and the text stays editable.
plt.rcParams["pdf.fonttype"] = 42

# The moran-process colors, so figures from both repos match. Asymmetric lung is new: a
# lighter brown from the Mammalian palette.
COLORS = {
    "Mammalian": "#8C510A",
    "Asymmetric lung": "#D8B365",
    "Avian": "#2E7D32",
    "Fish": "#084182",
    "Random": "#A09E9E",
    "Complete": "#000000",
    "Cycle": "#5C6BC0",
    "Star": "#E4C306",
    "Line": "#C2185B",
    "Grid": "#00897B",
}
BACKGROUND = ("Random",)  # drawn as a faint cloud, every other category on top with CIs
PROPERTIES = list(properties(nx.path_graph(3)))
LABELS = {
    "rho": "fixation probability",
    "steps_mean_given_fixed": "mean fixation time (steps), given fixation",
    "steps_sd_given_fixed": "sd of fixation time (steps), given fixation",
    "rho_minus_random": "fixation probability minus matched random mean",
    "steps_mean_given_fixed_minus_random": "mean fixation time minus matched random mean (steps)",
    "n_nodes": "N",
    "n_edges": "E",
}
MATCHED_METRICS = ("rho", "steps_mean_given_fixed")  # compared with random graphs of the same N, E and r
HOVER = ("n_nodes", "n_edges", "r", "batch", "kwargs")  # shown under the graph's name, in this order
# matplotlib marker -> the same shape in plotly, so both scatter functions agree
MARKERS = {"o": "circle", "s": "square", "^": "triangle-up", "D": "diamond", "v": "triangle-down", "P": "cross",
           "X": "x", "*": "star"}


def print_batch(batch):
    """What the batch ran (plan, submission, graphs built), then its verification report."""
    path = batch_path(batch)
    plan = load_plan(path / "plan.json")
    sim, lsf = plan["sim"], plan["lsf"]
    submissions = json.loads((path / "submissions.json").read_text()) if (path / "submissions.json").exists() else []
    graphs = pd.read_parquet(path / "graphs.parquet", columns=["category"])
    print(f"BATCH {plan['batch_name']}: {plan['description'] or '(no description)'}")
    for sub in submissions:
        jobs = ", ".join(f"{k} {v}" for k, v in sub.items() if k not in ("time", "n_workers"))
        print(f"  submitted {time.strftime('%Y-%m-%d %H:%M', time.localtime(sub['time']))}: "
              f"{sub['n_workers']} workers on {lsf['queue']} ({jobs})")
    print(f"  {sim['update_rule']} updating, r = {sim['r_values']}, {sim['n_repeats']:,} repeats, "
          f"max_steps {sim['max_steps']:,}, keep_runs {sim['keep_runs']}")
    print(f"  zoo_seed {plan['zoo_seed']}, batch_seed {plan['batch_seed']}")
    for i, spec in enumerate(plan["specs"]):
        kwargs = ", ".join(f"{k}={v!r}" for k, v in spec["kwargs"].items())
        skip = "  skip_null_steps" if spec["skip_null_steps"] else ""
        print(f"  [{i:>3}] {spec['factory']}({kwargs}) x{spec['count']}{skip}")
    print(f"  {len(graphs)} graphs built: {graphs['category'].value_counts().to_dict()}\n")
    print((path / "report" / "verification.txt").read_text())


def print_run_report(batch):
    """How the batch went: what you waited, where the time went, how much of the workers'
    time was wasted, and the cost of each r. Only time inside simulate() is measured per
    chunk; job startup and graph loading are not."""
    path = batch_path(batch)
    report = load_report(batch)
    t = report["chunk_timing"].assign(wall=lambda d: d["end"] - d["start"])
    submissions = json.loads((path / "submissions.json").read_text())
    submitted, requested = submissions[0]["time"], submissions[0]["n_workers"]
    # finalize writes verification.txt last, so its mtime is when the report was done
    finished = (path / "report" / "verification.txt").stat().st_mtime
    first_start, last_end = t["start"].min(), t["end"].max()
    workers = t.groupby(["job", "index"]).agg(first=("start", "min"), last=("end", "max"), busy=("wall", "sum"),
                                              chunks=("chunk_id", "count"))
    held = (workers["last"] - workers["first"]).sum()  # from a worker's first chunk to its last
    gaps = held - workers["busy"].sum()  # claiming the next chunk and writing the last result
    tail = last_end - workers["last"].min()
    cpu_s, steps, runs = t["cpu"].sum(), t["steps"].sum(), t["count"].sum()

    def clock(seconds):
        return time.strftime("%H:%M:%S", time.localtime(seconds))

    def minutes(seconds):
        return f"{seconds / 60:.1f} min" if seconds < 3600 else f"{seconds / 3600:.2f} h"

    print(f"{path.name}: verification {report['verification']['overall']}, "
          f"{t['graph_idx'].nunique()} graphs x {t['r'].nunique()} r values, {runs:,} runs in {len(t):,} chunks\n")
    print(f"YOUR WAIT: {minutes(finished - submitted)}")
    print(f"  {clock(submitted)}  submitted")
    print(f"  {clock(first_start)}  first chunk started   {minutes(first_start - submitted):>9}  queue + prepare job")
    print(f"  {clock(last_end)}  last chunk finished   {minutes(last_end - first_start):>9}  simulating")
    print(f"  {clock(finished)}  report written        {minutes(finished - last_end):>9}  finalize job\n")

    print("COMPUTE")
    print(f"  CPU time simulating   {cpu_s / 3600:.2f} CPU hours ({cpu_s / 60:,.0f} CPU min)")
    print(f"  speed                 {steps / cpu_s / 1e6:.0f}M Moran steps per CPU second, "
          f"{steps / runs:,.0f} steps per run on average")
    print(f"  chunk length          median {t['wall'].median():.1f} s "
          f"(p10 {t['wall'].quantile(0.1):.1f}, p90 {t['wall'].quantile(0.9):.1f}, max {t['wall'].max():.1f})\n")

    print("WASTE")
    print(f"  workers               {len(workers)} of {requested} requested got a chunk")
    print(f"  worker time held      {minutes(held)} (each worker from its first chunk to its last)")
    print(f"    simulating          {workers['busy'].sum() / held:.0%}")
    print(f"    between chunks      {gaps / held:.0%}  ({gaps / len(t):.2f} s per chunk: claim + write result)")
    print(f"  tail                  {minutes(tail)} from the first worker running out of chunks to the last finishing")
    print("  not measured          job startup, graph loading, queue time of each worker\n")

    per_r = t.groupby("r").agg(cpu_s=("cpu", "sum"), steps=("steps", "sum"), runs=("count", "sum"))
    per_r = pd.DataFrame({
        "CPU min": (per_r["cpu_s"] / 60).round(1),
        "share of CPU": (per_r["cpu_s"] / cpu_s * 100).round(0).astype(int).astype(str) + "%",
        "steps per run": per_r["steps"].div(per_r["runs"]).round(0).astype(int),
        "M steps per CPU s": (per_r["steps"] / per_r["cpu_s"] / 1e6).round(0).astype(int),
    })
    print("PER r (all r values run mixed on the same workers, so only CPU time can be split)")
    print("  " + per_r.to_string().replace("\n", "\n  "))


def load(batches):
    """results and histograms of every batch, with a `batch` column and the differences from
    matched random graphs (see _minus_random). Raises if a batch has not been finalized."""
    results, hists = [], []
    for batch in batches:
        report = load_report(batch)
        name = batch_path(batch).name
        results.append(report["results"].assign(batch=name))
        hists.append(report["histograms"].assign(batch=name))
    results = pd.concat(results, ignore_index=True)
    repeated = results.groupby(["hash", "r"])["batch"].nunique().gt(1).sum()
    if repeated:
        print(f"{repeated} (graph, r) cells appear in more than one batch, each is drawn once per batch")
    return _minus_random(results), pd.concat(hists, ignore_index=True)


def _minus_random(results):
    """Adds <metric>_minus_random: the metric minus its mean over the random graphs with the
    same N, E and r (in any loaded batch), and n_random_matches, how many there are. NaN
    where there are none. A random graph gets its own distance from that mean."""
    size = ["n_nodes", "n_edges", "r"]
    random = results[results["category"] == "Random"].groupby(size)
    matched = random[list(MATCHED_METRICS)].mean().join(random.size().rename("n_random_matches"))
    results = results.join(matched, on=size, rsuffix="_random_mean")
    for metric in MATCHED_METRICS:
        results[f"{metric}_minus_random"] = results[metric] - results.pop(f"{metric}_random_mean")
    results["n_random_matches"] = results["n_random_matches"].fillna(0).astype(int)
    return results


def _error(df, col):
    """Error bar of a column as [below, above]: its 95% interval if the report has one, else None."""
    if f"{col}_lo" in df:
        return [df[col] - df[f"{col}_lo"], df[f"{col}_hi"] - df[col]]
    se = col.replace("_mean_", "_se_")
    return [1.96 * df[se]] * 2 if se != col and se in df else None


def scatter(ax, df, x, y, marker_by=None):
    """y against x, one color per category. Background categories are a faint cloud.
    marker_by names a column (e.g. "r") whose values get one marker shape each."""
    markers = _markers(df, marker_by)
    for category, part in df.groupby("category"):
        label = category  # one legend entry per category, not per marker
        for value, sub in part.groupby(marker_by) if marker_by else [(None, part)]:
            if category in BACKGROUND:
                ax.scatter(sub[x], sub[y], s=10, alpha=0.5, linewidths=0, marker=markers[value],
                           color=COLORS[category], label=label)
            else:
                ax.errorbar(sub[x], sub[y], xerr=_error(sub, x), yerr=_error(sub, y), fmt=markers[value], ms=5,
                            capsize=2, color=COLORS[category], label=label, zorder=3)
            label = None
    if marker_by:
        for value, marker in markers.items():
            ax.scatter([], [], marker=marker, color="0.4", label=f"{LABELS.get(marker_by, marker_by)} = {value}")
    ax.set(xlabel=LABELS.get(x, x), ylabel=LABELS.get(y, y))


def _markers(df, marker_by):
    """Marker shape per value of the marker_by column, in sorted order."""
    if marker_by is None:
        return {None: "o"}
    values = sorted(df[marker_by].unique())
    if len(values) > len(MARKERS):
        raise ValueError(f"{len(values)} values of {marker_by}, only {len(MARKERS)} marker shapes")
    return dict(zip(values, MARKERS))


def hover_scatter(df, x, y, hover=HOVER, marker_by=None):
    """scatter() as an interactive plotly figure. Hovering a point shows the graph's name,
    x, y, and the `hover` columns of df."""
    errors = {}
    for axis, col in (("x", x), ("y", y)):
        err = _error(df, col)
        if err is not None:
            df = df.assign(**{f"{axis}_below": err[0], f"{axis}_above": err[1]})
            errors.update({f"error_{axis}": f"{axis}_above", f"error_{axis}_minus": f"{axis}_below"})
    background_first = sorted(df["category"].unique(), key=lambda c: c not in BACKGROUND)
    markers = _markers(df, marker_by)
    fig = px.scatter(df, x=x, y=y, color="category", color_discrete_map=COLORS, symbol=marker_by,
                     symbol_map={value: MARKERS[marker] for value, marker in markers.items()}, hover_name="name",
                     hover_data=dict.fromkeys(hover, True), labels=LABELS,
                     category_orders={"category": background_first}, template="simple_white",
                     width=800, height=500, **errors)

    def style(trace):
        category = trace.name.split(", ")[0]  # with marker_by, plotly names traces "<category>, <value>"
        trace.update(showlegend=False, legendgroup=category)
        if category in BACKGROUND:
            trace.update(marker_opacity=0.5, error_x_visible=False, error_y_visible=False)

    # Plotly's legend would list every (category, value) pair: instead one entry per color and
    # one per shape. Clicking a category hides all its points.
    fig.for_each_trace(style)
    for category in background_first:
        fig.add_scatter(x=[None], y=[None], mode="markers", marker_color=COLORS[category], name=category,
                        legendgroup=category)
    if marker_by:
        for value, marker in markers.items():
            fig.add_scatter(x=[None], y=[None], mode="markers", marker=dict(color="grey", symbol=MARKERS[marker]),
                            name=f"{LABELS.get(marker_by, marker_by)} = {value}")
    return fig


def sweep(ax, df, category, kwarg, y):
    """y of one category's graphs against a kwarg of their recipe, colored by N (a sweep
    can change it). At each graph's x, a grey bar spans its random matches: the random
    graphs with the same N and E, min to max. Returns the points, for a colorbar."""
    graphs = df[df["category"] == category]
    x = graphs["kwargs"].map(lambda kwargs: json.loads(kwargs)[kwarg])
    random = df[df["category"] == "Random"].groupby(["n_nodes", "n_edges"])[y].agg(["min", "max"])
    matched = graphs.join(random, on=["n_nodes", "n_edges"])
    ax.vlines(x, matched["min"], matched["max"], color=COLORS["Random"], lw=5, alpha=0.6,
              label="random, same N and E")
    ax.errorbar(x, graphs[y], yerr=_error(graphs, y), fmt="none", ecolor="0.4", capsize=2)
    points = ax.scatter(x, graphs[y], c=graphs["n_nodes"], cmap="viridis", s=30, zorder=3, label=category)
    ax.set(xlabel=kwarg, ylabel=LABELS.get(y, y))
    return points


def lines(ax, df, x, y, by, cmap="viridis"):
    """y against x, one line per value of `by`, colored in order along cmap, with the
    same error bars as scatter()."""
    values = sorted(df[by].unique())
    for value, color in zip(values, plt.get_cmap(cmap)(np.linspace(0, 0.9, len(values)))):
        part = df[df[by] == value].sort_values(x)
        ax.errorbar(part[x], part[y], yerr=_error(part, y), fmt="o-", ms=4, lw=1.5, capsize=0, color=color,
                    label=f"{LABELS.get(by, by)} = {value}")
    ax.set(xlabel=LABELS.get(x, x), ylabel=LABELS.get(y, y))


def _complete_graph_time(n, r):
    """Mean Bd fixation time (steps, given fixation) from one mutant on the complete graph of n
    nodes, exact. Traulsen & Hauert 2009 (Stochastic evolutionary game dynamics), eq. 26, where
    T-/T+ = 1/r at every mutant count i collapses the inner product to a geometric sum."""
    i = np.arange(1, n)
    t_plus = r * i / (r * i + n - i) * (n - i) / (n - 1)  # no self-loops: offspring go to the other n - 1
    phi = (1 - r ** -i) / (1 - r ** -n)  # fixation probability from i mutants
    return np.sum(phi / t_plus * (1 - r ** -(n - i)) / (1 - 1 / r))


def versus_random(results, directed):
    """One row of panels per graph type against its random cloud: the random graphs with the
    same N, E and r, drawn as their median and interquartile band (the question is whether a
    graph falls outside the cloud, so its spread matters, not the SE of its mean). Panels:
    fixation probability, fixation time, and random time over the graph's time (above 1 the
    graph fixes faster). A type is drawn only at sizes with a cloud and needs one graph per
    size; what is left out is printed. random_graph is undirected, so for a directed type the
    cloud matches N and E but not the edge directions."""
    if results["r"].nunique() != 1:
        raise ValueError(f"versus_random needs exactly one r value, got {sorted(results['r'].unique())}")
    r = results["r"].iloc[0]
    kind = "directed" if directed else "undirected"
    size = ["n_nodes", "n_edges"]
    random = results[results["category"] == "Random"]
    cloud = random.groupby(size)[["rho", "steps_mean_given_fixed"]].quantile([0.25, 0.5, 0.75]).unstack()
    cloud.columns = [f"{metric}_q{round(q * 100)}" for metric, q in cloud.columns]
    graphs = results[(results["category"] != "Random") & (results["is_directed"] == directed)]
    matched = graphs.join(cloud, on=size, how="inner")
    for name in sorted(set(graphs["name"]) - set(matched["name"])):
        print(f"  no random graphs with its N and E, not drawn: {name}")
    types = []
    for category, part in matched.groupby("category"):
        if part["n_nodes"].duplicated().any():
            print(f"  several {category} graphs share one N, not drawn (a shape sweep, not a size sweep)")
        else:
            types.append((category, part.sort_values("n_nodes")))
    if not types:
        raise ValueError(f"no {kind} graph type has random graphs of its N and E")

    fig, axes = plt.subplots(len(types), 3, figsize=(17, 4.6 * len(types)), squeeze=False, layout="constrained")
    for (category, part), (ax_rho, ax_time, ax_ratio) in zip(types, axes):
        n = part["n_nodes"]
        color = COLORS[category]
        for ax, y in ((ax_time, "steps_mean_given_fixed"), (ax_rho, "rho")):
            ax.fill_between(n, part[f"{y}_q25"], part[f"{y}_q75"], color=COLORS["Random"], alpha=0.3, lw=0)
            ax.plot(n, part[f"{y}_q50"], "o-", ms=4, color=COLORS["Random"], label="random, same N and E")
            ax.errorbar(n, part[y], yerr=_error(part, y), fmt="o-", ms=5, capsize=2, color=color, label=category,
                        zorder=3)
        # Moran's well-mixed formula, the baseline an amplifier beats and a suppressor misses
        ax_rho.plot(n, (1 - 1 / r) / (1 - r ** -n), "--", color="0.2", lw=1, label="well-mixed")
        ax_time.plot(n, [_complete_graph_time(size, r) for size in n], "--", color="0.2", lw=1,
                     label="well-mixed")
        ax_ratio.fill_between(n, part["steps_mean_given_fixed_q25"] / part["steps_mean_given_fixed"],
                              part["steps_mean_given_fixed_q75"] / part["steps_mean_given_fixed"],
                              color=color, alpha=0.3, lw=0)
        ax_ratio.plot(n, part["steps_mean_given_fixed_q50"] / part["steps_mean_given_fixed"], "o-", ms=5,
                      color=color)
        ax_ratio.axhline(1, color="0.2", ls="--", lw=1)
        ax_time.set(xscale="log", yscale="log", ylabel=LABELS["steps_mean_given_fixed"],
                    title="fixation time")
        ax_rho.set(xscale="log", ylabel=LABELS["rho"], title=f"{category} ({kind}): fixation probability")
        ax_ratio.set(xscale="log", yscale="log", ylabel=f"random / {category.lower()} (mean time)",
                     title="speed-up over random (band: random IQR)")
        for ax in (ax_rho, ax_time, ax_ratio):
            ax.set_xlabel("N")
        ax_rho.legend(frameon=False, fontsize=9)
    fig.suptitle(f"Each {kind} graph type against random graphs of the same N and E, r = {r}")
    return fig


def property_grid(df, y, properties, n_cols=4):
    """One scatter panel per structural property, all against y."""
    n_rows = -(-len(properties) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.2 * n_rows), squeeze=False,
                             layout="constrained")
    for ax, prop in zip(axes.flat, properties):
        scatter(ax, df, prop, y)
        ax.set_ylabel("")
    for ax in axes.flat[len(properties):]:
        ax.set_visible(False)
    fig.supylabel(LABELS.get(y, y))
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="outside lower center", ncols=df["category"].nunique(),
               frameon=False)
    return fig


def time_distributions(results, hists):
    """Distribution of fixation time per category, pooling the runs of all its graphs."""
    # Pooling across r would mix distributions with very different time scales.
    if results["r"].nunique() != 1:
        raise ValueError(f"time_distributions needs exactly one r value, got {sorted(results['r'].unique())}")
    fixed =hists[hists["winner"] == 0].merge(results[["batch", "graph_idx", "r", "category"]],
                                               on=["batch", "graph_idx", "r"])
    # Bin k starts at 2**(k/4) - 1 steps (engine.steps_bin). Bin 0 (0 steps) cannot fix and
    # would put 0 on a log axis.
    edges = 2.0 ** (np.arange(1, HIST_BINS + 1) / 4) - 1
    fig, ax = plt.subplots(figsize=(7, 4.5))
    total = np.zeros(HIST_BINS - 1)
    for category, part in fixed.groupby("category"):
        counts = np.vstack(part["steps_hist"].to_numpy()).sum(axis=0)[1:].astype(np.float64)
        total += counts
        ax.stairs(counts / counts.sum(), edges, color=COLORS[category], label=f"{category} ({len(part)} graphs)")
    used = np.flatnonzero(total)
    ax.set(xscale="log", xlim=(edges[used[0]], edges[used[-1] + 1]),
           xlabel="fixation time (steps), runs that fixed", ylabel="fraction of runs (4 bins per doubling)")
    ax.legend(frameon=False)
    return fig


def _save(fig, path):
    """Writes path (a .png, for a quick look) and the same name as .pdf, for the paper."""
    for out in (path, path.with_suffix(".pdf")):
        fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path} and its .pdf")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("batches", nargs="+")
    parser.add_argument("--out", type=Path, help="default: <data root>/figures/<batch names joined by +>")
    a = parser.parse_args()

    for batch in a.batches:
        print_batch(batch)
        print_run_report(batch)
    results, hists = load(a.batches)
    out = a.out or DATA_ROOT / "figures" / "+".join(batch_path(b).name for b in a.batches)
    out.mkdir(parents=True, exist_ok=True)
    varying = [p for p in PROPERTIES if results[p].dtype != bool and results[p].nunique() > 1]

    for r, part in results.groupby("r"):
        fig, (ax_joint, ax_spread) = plt.subplots(1, 2, figsize=(12, 4.5))
        scatter(ax_joint, part, "steps_mean_given_fixed", "rho")
        scatter(ax_spread, part, "steps_mean_given_fixed", "steps_sd_given_fixed")
        ax_joint.legend(frameon=False)
        fig.suptitle(f"r = {r}")
        _save(fig, out / f"rho_vs_time_r{r}.png")
        for y in ("rho", "steps_mean_given_fixed"):
            fig = property_grid(part, y, varying)
            fig.suptitle(f"r = {r}")
            _save(fig, out / f"{y}_vs_properties_r{r}.png")
        fig = time_distributions(part, hists)
        fig.axes[0].set_title(f"r = {r}")
        _save(fig, out / f"time_distribution_r{r}.png")

    if results["r"].nunique() > 1:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        scatter(ax, results, "r", "rho")
        ax.legend(frameon=False)
        _save(fig, out / "rho_vs_r.png")


if __name__ == "__main__":
    main()

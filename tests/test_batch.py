import subprocess
import sys
from collections import Counter

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from moran_graphs import PopulationGraph as pg
from moran_graphs import hpc
from moran_graphs.engine import simulate, summarize
from moran_graphs.plan import build_zoo, make_plan, save_plan, spec
from moran_graphs.properties import graph_from_W, properties, wl_hash


def tiny_plan(name, n_repeats=1000, chunk_target_steps=2_000, n_workers=8):
    specs = [spec("complete_graph", n_nodes=5), spec("cycle_graph", n_nodes=6),
             spec("random_graph", count=2, n_nodes=7, n_edges=8)]
    return make_plan(name, specs, [1.0, 2.0], n_repeats, zoo_seed=1, batch_seed=2,
                     n_workers=n_workers, chunk_target_steps=chunk_target_steps)


def prepared(tmp_path, **kw):
    batch = tmp_path / "batch"
    save_plan(tiny_plan("batch", **kw), batch / "plan.json")
    hpc.prepare(batch)
    hpc.compute_properties(batch)
    return batch


def test_chunks_cover_every_cell_exactly(tmp_path):
    batch = prepared(tmp_path)
    chunks = pd.read_parquet(batch / "chunks.parquet")
    graphs = pd.read_parquet(batch / "graphs.parquet")
    assert (chunks.groupby(["graph_idx", "r"])["n_runs"].sum() == 1000).all()
    assert len(chunks.groupby(["graph_idx", "r"])) == len(graphs) * 2
    assert chunks["seed"].is_unique and (chunks["chunk_id"] == np.arange(len(chunks))).all()
    n = graphs.set_index("graph_idx").loc[chunks["graph_idx"], "n_nodes"].to_numpy()
    assert (np.diff(n) <= 0).all()  # largest graphs first


def test_workers_claim_each_chunk_once(tmp_path):
    batch = prepared(tmp_path, n_repeats=200, chunk_target_steps=10)  # ~1 run per chunk
    n_chunks = len(pd.read_parquet(batch / "chunks.parquet"))
    assert n_chunks >= 200
    procs = [subprocess.Popen([sys.executable, "-m", "moran_graphs.hpc", "worker", str(batch),
                               "--n-workers", "8", "--index", str(i)], stdout=subprocess.DEVNULL)
             for i in range(1, 9)]
    assert all(p.wait() == 0 for p in procs)
    # All 8 share LSB_JOBID "manual", so the journals are keyed by index alone.
    claims = Counter(int(c) for j in (batch / "journals").iterdir() for c in j.read_text().split())
    assert set(claims) == set(range(n_chunks)) and max(claims.values()) == 1
    assert hpc.finalize(batch) == 0


def test_chunk_summaries_merge_exactly():
    W = pg.star_graph(8).csr()
    parts = [simulate(W, [1], [1.5], 3000, 10**8, seed) for seed in (1, 2, 3)]
    res = pd.concat([summarize(p).assign(chunk_id=i, graph_idx=0, r=1.5) for i, p in enumerate(parts)])
    cells, _ = hpc._aggregate(res, pd.DataFrame({"graph_idx": [0]}))
    winner = np.concatenate([p["winner"] for p in parts])
    steps = np.concatenate([p["steps"] for p in parts]).astype(float)[winner == 0]
    row = cells.iloc[0]
    assert row["n"] == 9000 and row["n_fixed"] == len(steps)
    assert np.isclose(row["steps_mean_given_fixed"], steps.mean())
    assert np.isclose(row["steps_sd_given_fixed"], steps.std(ddof=1))


def test_wl_hash_identity():
    cycle = pg.cycle_graph(6).csr()
    assert wl_hash(cycle) == wl_hash(cycle * 3.0)  # unweighted: the weight value does not matter
    assert wl_hash(cycle) != wl_hash(pg.cycle_graph(6, directed=True).csr())
    weighted = cycle.copy()
    weighted.data[:2] = 5.0
    assert wl_hash(weighted) != wl_hash(cycle)


@pytest.mark.parametrize("graph", [pg.star_graph(7, directed=True), pg.avian_graph(3, 2, directed=True),
                                   pg.fish_graph(2, 2), pg.random_graph(12, 16, seed=0)], ids=lambda g: g.name)
def test_properties_from_W_match_the_original_graph(graph):
    pd.testing.assert_series_equal(pd.Series(properties(graph_from_W(graph.csr()))),
                                   pd.Series(properties(graph.graph)))


def test_csr_roundtrip(tmp_path):
    mats = [pg.star_graph(5, directed=True).csr(), pg.mammalian_lung_graph(3).csr()]
    hpc.save_csr(tmp_path / "g.npz", mats)
    z = np.load(tmp_path / "g.npz")
    for i, W in enumerate(mats):
        assert (hpc.read_csr(z, i) != sp.csr_array(W, dtype=float)).nnz == 0


def test_series_graphs_are_indexed():
    specs = [spec("random_graph", count=3, n_nodes=31, n_edges=40), spec("star_graph", n_nodes=5),
             spec("random_graph", count=2, n_nodes=31, n_edges=40)]
    names = [rec["graph"].name for rec in build_zoo(make_plan("x", specs, [1.0], 10, zoo_seed=0, batch_seed=0))]
    assert names == ["random_n31_e40_i0", "random_n31_e40_i1", "random_n31_e40_i2", "star_n5",
                     "random_n31_e40_i3", "random_n31_e40_i4"]
    assert [g.name for g in pg.sample_random_graphs(31, 40, 2, seed=0)] == ["random_n31_e40_i0", "random_n31_e40_i1"]


def test_random_matches_share_one_cloud_per_size():
    specs = [spec("star_graph", n_nodes=5, random_matches=2), spec("mammalian_lung_graph", depth=3, random_matches=2),
             spec("line_graph", n_nodes=5, random_matches=3), spec("random_graph", count=1, n_nodes=15, n_edges=14)]
    plan = make_plan("x", specs, [1.0], 10, zoo_seed=0, batch_seed=0)
    matches = [(s["count"], s["up_to"], s["kwargs"]["n_nodes"], s["kwargs"]["n_edges"]) for s in plan["specs"][4:]]
    assert matches == [(3, True, 5, 4), (1, True, 15, 14)]


def test_random_matches_skip_null_steps_like_their_graph():
    specs = [spec("star_graph", n_nodes=5, random_matches=2),
             spec("line_graph", n_nodes=5, skip_null_steps=True, random_matches=2),
             spec("cycle_graph", n_nodes=5, random_matches=2)]
    plan = make_plan("x", specs, [1.0], 10, zoo_seed=0, batch_seed=0)
    assert [(s["kwargs"]["n_edges"], s["skip_null_steps"]) for s in plan["specs"][3:]] == [(4, True), (5, False)]


def test_random_matches_stop_when_size_is_exhausted():
    plan = make_plan("x", [spec("complete_graph", n_nodes=5, random_matches=3)], [1.0], 10, zoo_seed=0, batch_seed=0)
    assert [rec["graph"].name for rec in build_zoo(plan)] == ["complete_n5"]


def test_too_few_distinct_random_graphs_raises():
    plan = make_plan("x", [spec("random_graph", count=2, n_nodes=3, n_edges=3)], [1.0], 10, zoo_seed=0, batch_seed=0)
    with pytest.raises(ValueError, match="only 1 of 2"):
        build_zoo(plan)

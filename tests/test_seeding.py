import shutil

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from moran_graphs import PopulationGraph as pg
from moran_graphs import hpc, simulate
from moran_graphs.plan import make_plan, save_plan, spec

FIELDS = ("winner", "steps", "n_changes")


def weighted_directed():
    W = sp.csr_array(pg.mammalian_lung_graph(4, directed=True).csr(), dtype=float)
    W.data = np.linspace(0.5, 3.0, W.nnz)
    W = W + sp.csr_array(pg.mammalian_lung_graph(4).csr()) * 0.1  # back edges, lighter
    return W


@pytest.mark.parametrize("skip", [False, True])
@pytest.mark.parametrize("graph", ["star", "weighted directed lung"])
def test_same_seed_same_runs(skip, graph):
    W = pg.star_graph(20).csr() if graph == "star" else weighted_directed()
    for n_mutants, fitness in (([1], [1.5]), ([2, 1, 1], [1.1, 0.0, 2.0])):
        a, b = (simulate(W, n_mutants, fitness, 2000, 10**8, 12345, skip_null_steps=skip) for _ in range(2))
        for f in FIELDS:
            assert np.array_equal(a[f], b[f]), f
        c = simulate(W, n_mutants, fitness, 2000, 10**8, 12346, skip_null_steps=skip)
        assert not np.array_equal(a["steps"], c["steps"])  # the seed is actually used


def test_rerunning_a_batch_reproduces_every_chunk(tmp_path):
    """A chunk's numbers depend only on its seed, not on which worker ran it or when."""
    specs = [spec("star_graph", n_nodes=10), spec("random_graph", count=2, n_nodes=12, n_edges=15)]
    plan = make_plan("b", specs, [1.2], 3000, zoo_seed=3, batch_seed=4, n_workers=3, chunk_target_steps=50_000)
    batch = tmp_path / "b"
    save_plan(plan, batch / "plan.json")
    hpc.prepare(batch)

    def run_all(n_workers):
        for d in ("claims", "journals", "results"):
            shutil.rmtree(batch / d, ignore_errors=True)
        for i in range(1, n_workers + 1):
            hpc.worker(batch, n_workers, index=i)
        res = pd.read_parquet(batch / "results")
        return res.drop(columns=["job", "index", "host", "start", "end", "cpu"]).sort_values(["chunk_id", "winner"])

    first, second = run_all(3), run_all(1)  # different worker counts, so different chunk-to-worker splits
    assert len(first) > 3
    for col in first.columns:
        a, b = first[col].to_numpy(), second[col].to_numpy()
        if col == "steps_hist":
            assert all(np.array_equal(x, y) for x, y in zip(a, b))
        else:
            assert np.array_equal(a, b), col

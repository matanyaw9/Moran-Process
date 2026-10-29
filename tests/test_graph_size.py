import inspect

import pytest

from moran_graphs.PopulationGraph import PopulationGraph, graph_size
from moran_graphs.plan import factories

SHAPES = [
    ("complete_graph", {"n_nodes": 15}),
    ("cycle_graph", {"n_nodes": 3}),
    ("cycle_graph", {"n_nodes": 31}),
    ("line_graph", {"n_nodes": 2}),
    ("line_graph", {"n_nodes": 31}),
    ("star_graph", {"n_nodes": 2}),
    ("star_graph", {"n_nodes": 31}),
    ("grid_graph", {"width": 1, "height": 5}),
    ("grid_graph", {"width": 5, "height": 6}),
    ("random_graph", {"n_nodes": 20, "n_edges": 19, "seed": 0}),
    ("random_graph", {"n_nodes": 20, "n_edges": 40, "seed": 0}),
    *[("mammalian_lung_graph", {"depth": d, "branching_factor": b}) for b in (1, 2, 3) for d in (1, 4)],
    ("mammalian_lung_graph", {"depth": 4}),
    *[("avian_graph", {"n_parabronchi": n, "l_parabronchi": l}) for n in (1, 2, 7) for l in (1, 4)],
    *[("fish_graph", {"n_rods": r, "l_rod": l, "fillaments": f}) for r in (1, 3) for l in (1, 3) for f in (0, 1, 2, 5)],
    ("fish_graph", {"n_rods": 3, "l_rod": 3}),
    *[("asymmetric_lung_graph", {"flow_ratio": f, "n_nodes": n}) for f in (0.05, 0.326, 0.5) for n in (15, 31, 255)],
    *[("asymmetric_lung_graph", {"flow_ratio": f, "cutoff_diameter": 1.0}) for f in (0.1, 0.5)],
    ("asymmetric_lung_graph", {"n_nodes": 31}),
]


def with_directed(shapes):
    for factory, kwargs in shapes:
        yield factory, kwargs
        if "directed" in inspect.signature(getattr(PopulationGraph, factory)).parameters:
            yield factory, {**kwargs, "directed": True}


@pytest.mark.parametrize("factory, kwargs", list(with_directed(SHAPES)))
def test_size_matches_built_graph(factory, kwargs):
    G = getattr(PopulationGraph, factory)(**kwargs).graph
    assert graph_size(factory, **kwargs) == (G.number_of_nodes(), G.number_of_edges())


def test_every_factory_has_a_formula():
    assert {factory for factory, _ in SHAPES} == set(factories())


def test_misspelled_kwarg_raises():
    with pytest.raises(TypeError):
        graph_size("fish_graph", n_rods=3, l_rods=3)

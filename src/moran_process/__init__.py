"""moran_process package.

Top-level names are exported lazily (PEP 562). Importing the package no longer
eagerly pulls in matplotlib / scipy / seaborn / networkx via the analysis, core
and pipeline submodules, which keeps HPC-worker startup lean: a worker that only
runs simulations from a GraphCore shard never imports the plotting/analysis
stack. Each name is resolved (and cached) on first access, so the flat API
(`from moran_process import GraphZoo`) keeps working exactly as before.
"""
import importlib

# Public name -> "submodule:attribute". Nothing here is imported until accessed.
_LAZY_EXPORTS = {
    "CATEGORY_COLOR_DICT": "moran_process.analysis.analysis_utils:CATEGORY_COLOR_DICT",
    "GRAPH_PROPERTY_COLUMNS": "moran_process.analysis.analysis_utils:GRAPH_PROPERTY_COLUMNS",
    "GRAPH_PROPERTY_DESCRIPTION": "moran_process.analysis.analysis_utils:GRAPH_PROPERTY_DESCRIPTION",
    "GraphZoo": "moran_process.core.graph_zoo:GraphZoo",
    "GRAPH_PROPS": "moran_process.core.population_graph:GRAPH_PROPS",
    "PopulationGraph": "moran_process.core.population_graph:PopulationGraph",
    "ProcessLab": "moran_process.pipeline.process_lab:ProcessLab",
    "MoranProcess": "moran_process.simulations.moran_process:MoranProcess",
    "MultiColorMoranProcess": "moran_process.simulations.multi_color_moran_process:MultiColorMoranProcess",
    "SimulationProcess": "moran_process.simulations.simulation_process:SimulationProcess",
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name):
    """Resolve a top-level export on first access (PEP 562)."""
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = target.split(":")
    value = getattr(importlib.import_module(module_name), attr)
    globals()[name] = value  # cache: subsequent access skips __getattr__
    return value


def __dir__():
    return sorted(__all__)

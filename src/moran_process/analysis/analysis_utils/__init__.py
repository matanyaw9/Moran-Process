"""
Utility functions for analysis notebooks.

This used to be a single ``analysis_utils.py``. It was split into focused
submodules once it crossed ~1500 lines:

    colors      -- category palette, graph-property metadata, shared plot constants
    provenance  -- read/write batch_info.json + git/host/python run provenance
    theory      -- exact complete-graph Moran baselines (the amplifier/suppressor zero)
    io          -- locate / aggregate raw results, roll up graph statistics
    plots       -- every figure-producing function + its caching infrastructure

``colors``, ``provenance`` and ``theory`` are leaves; ``io`` and ``plots`` build on
them. ``theory`` needs only numpy and scipy, so the data path can import the analytic
baselines without pulling in the plotting stack.

The public API is re-exported here, so the import path is unchanged:
``from moran_process.analysis.analysis_utils import <name>`` (and ``import *``)
keep working exactly as before. Import from a submodule directly
(``...analysis_utils.io import aggregate_results_no_load``) when you want only
that slice without pulling in the plotting stack.
"""

from . import colors, provenance, theory, io, plots
from .colors import *  # noqa: F401,F403
from .provenance import *  # noqa: F401,F403
from .theory import *  # noqa: F401,F403
from .io import *  # noqa: F401,F403
from .plots import *  # noqa: F401,F403

__all__ = [
    *colors.__all__,
    *provenance.__all__,
    *theory.__all__,
    *io.__all__,
    *plots.__all__,
]

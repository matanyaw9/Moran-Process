"""C++-backed engines, drop-in replacements for the pure-Python simulations.

Two wrappers live here, one per Python reference class:

===============================  ==========================================
:class:`CppMoranProcess`         ``moran_process.MoranProcess``
:class:`CppMultiColorMoranProcess`  ``multi_color_moran_process.MultiColorMoranProcess``
===============================  ==========================================

Each mirrors its reference's public interface exactly (constructor signature,
initialiser, ``run``, ``run_repeats``) and delegates the hot inner loop to the
compiled ``_moran_cpp`` extension, so a caller swaps engines by swapping the
name and changes nothing else.

Both are *statistically* equivalent to their reference rather than bit-exact:
given the same seed individual trajectories differ, because the core uses
xoshiro256++ where NumPy uses PCG64. The aggregate quantities match within Monte
Carlo error. See ``_cpp/moran_core.cpp`` for the algorithms and RNG details.

Usage (drop-in in a worker)::

    from moran_process.simulations.cpp_moran_wrapper import CppMoranProcess as MoranProcess
"""

import numpy as np

from moran_process import _moran_cpp


class CppMoranProcess:
    """Moran process with a C++ core. Interface-compatible with ``MoranProcess``."""

    def __init__(
        self,
        graph_core,
        selection_coefficient: float = 1.0,
        max_steps: int = 1_000_000,
        seed=None,
    ):
        self.r = selection_coefficient
        self.n_nodes = graph_core.n_nodes
        self.max_steps = max_steps

        # The C++ core copies these into owned vectors; ensure contiguous int32
        # to match the GraphCore CSR dtype and avoid surprise casts.
        nbrs = np.ascontiguousarray(graph_core.nbrs, dtype=np.int32)
        offsets = np.ascontiguousarray(graph_core.offsets, dtype=np.int32)

        # seed=None -> -1 tells the C++ core to draw OS entropy (NumPy parity).
        c_seed = -1 if seed is None else int(seed)

        self._core = _moran_cpp.MoranProcessCore(
            self.n_nodes,
            nbrs,
            offsets,
            float(selection_coefficient),
            int(max_steps),
            c_seed,
        )

    @property
    def mutant_count(self) -> int:
        return self._core.mutant_count

    def initialize_random_mutant(self, n_mutants: int = 1) -> list[int]:
        """Place n_mutants mutants at randomly chosen nodes. Returns the indices."""
        return list(self._core.initialize_random_mutant(n_mutants))

    def run(self, track_history: bool = False) -> dict:
        """Run until fixation, extinction, or max_steps.

        Returns the same keys as the Python reference: fixation, steps,
        initial_mutants, selection_coeff, duration, and (if track_history=True)
        history.
        """
        return self._core.run(track_history)

    def run_repeats(self, n_repeats: int, n_mutants: int = 1) -> dict:
        """Run n_repeats simulations entirely in C++ (one boundary crossing).

        Mirrors ``SimulationProcess.run_repeats``: returns a dict of three
        equal-length arrays (fixation: bool, steps: int64, duration: float64).
        The repeat loop, mutant placement and timing all happen in the C++ core,
        so no per-repeat ``py::dict`` is allocated.
        """
        return self._core.run_repeats(n_repeats, n_mutants)


class CppMultiColorMoranProcess:
    """Multi-color Moran process with a C++ core.

    Interface-compatible with ``MultiColorMoranProcess``: same constructor,
    ``initialize_unique_colors``, ``run``, ``run_repeats``, and a ``state``
    array. Neutral by construction, so there is no selection coefficient here
    (the Python reference inherits one from ``SimulationProcess`` and ignores
    it; this class does not take one at all).

    The core's absorption test is O(1) rather than the reference's O(n)
    ``np.all(state == state[0])`` scan, using a live count of surviving
    lineages. That is exact: a lineage that goes extinct can never reappear,
    since a node only ever adopts a color a neighbour already holds.
    """

    def __init__(self, graph_core, max_steps: int = 1_000_000, seed=None):
        # Same duck-typed conversion as SimulationProcess.__init__: HPC workers
        # hand over a GraphCore, notebooks and `multicolor_batch.preview` hand
        # over a PopulationGraph. Checked by method rather than by isinstance so
        # this module stays numpy-only and never imports NetworkX.
        if hasattr(graph_core, "to_simulation_struct"):
            graph_core = graph_core.to_simulation_struct()

        self.n_nodes = graph_core.n_nodes
        self.max_steps = max_steps

        nbrs = np.ascontiguousarray(graph_core.nbrs, dtype=np.int32)
        offsets = np.ascontiguousarray(graph_core.offsets, dtype=np.int32)

        # seed=None -> -1 tells the C++ core to draw OS entropy (NumPy parity).
        c_seed = -1 if seed is None else int(seed)

        self._core = _moran_cpp.MultiColorCore(
            self.n_nodes,
            nbrs,
            offsets,
            int(max_steps),
            c_seed,
        )

    @property
    def state(self) -> np.ndarray:
        """Current color of every node (color == the founding node's index)."""
        return self._core.state

    @property
    def n_distinct(self) -> int:
        """Number of lineages still holding at least one node."""
        return self._core.n_distinct

    def initialize_unique_colors(self) -> None:
        """Assign each node its own color (color = node index)."""
        self._core.initialize_unique_colors()

    def run(self, track_history: bool = False) -> dict:
        """Run until one color fixes or max_steps is reached.

        Returns the same keys as the Python reference: winner, fixed, steps,
        duration, and (if track_history=True) history, a (snapshots x n_nodes)
        int array of full state snapshots. ``winner`` is -1 when the run was cut
        short by max_steps.
        """
        return self._core.run(track_history)

    def run_repeats(self, n_repeats: int, n_mutants: int = 1) -> dict:
        """Run n_repeats independent takeovers entirely in C++.

        Returns four equal-length arrays: winner (int64, -1 where censored),
        fixed (bool), steps (int64), duration (float64).

        ``n_mutants`` is accepted and ignored, matching the Python reference:
        every node is its own lineage, so there is no mutant count to set. It is
        in the signature only so the two engines stay call-compatible.
        """
        return self._core.run_repeats(n_repeats)

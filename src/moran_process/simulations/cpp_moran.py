"""C++-backed Moran process, a drop-in replacement for :class:`MoranProcess`.

This wrapper mirrors the public interface of
``moran_process.simulations.moran_simulation_process.MoranProcess`` exactly
(constructor signature, ``initialize_random_mutant``, ``run``), but delegates the
hot inner loop to the compiled ``_moran_cpp`` extension.

It is *statistically* equivalent to the pure-Python reference rather than
bit-exact: given the same seed individual trajectories differ, but the aggregate
fixation probability and fixation-time distribution match within Monte Carlo
error. See ``_cpp/moran_core.cpp`` for the algorithm and RNG details.

Usage (drop-in in the worker)::

    from moran_process.simulations.cpp_moran import CppMoranProcess as MoranProcess
"""

import numpy as np

from moran_process import _moran_cpp


class CppMoranProcess:
    """Moran process with a C++ core. Interface-compatible with ``MoranProcess``."""

    def __init__(self, graph_core, selection_coefficient: float = 1.0,
                 max_steps: int = 1_000_000, seed=None):
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
            self.n_nodes, nbrs, offsets,
            float(selection_coefficient), int(max_steps), c_seed,
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

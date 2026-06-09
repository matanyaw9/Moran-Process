from abc import ABC, abstractmethod
import numpy as np

from moran_process.core.graph_core import GraphCore


class SimulationProcess(ABC):
    """Abstract base class for all graph-based evolutionary simulation processes."""

    def __init__(
        self,
        graph_core: GraphCore,
        max_steps: int = 1_000_000,
        seed=None,
    ):
        self.max_steps = max_steps
        self.n_nodes = graph_core.n_nodes
        self.nbrs = graph_core.nbrs
        self.offsets = graph_core.offsets
        self.state = np.zeros(self.n_nodes, dtype=int)
        self._rng = np.random.default_rng(seed)

    def reset(self) -> None:
        """Reset all nodes to wild-type without reallocating the state array."""
        self.state.fill(0)

    @abstractmethod
    def step(self) -> None:
        """Execute one simulation step."""
        ...

    @abstractmethod
    def run(self, track_history: bool = False) -> dict:
        """Run the simulation to completion and return a result dict."""
        ...

    def run_repeats(self, n_repeats: int, n_mutants: int = 1) -> dict:
        """Run ``n_repeats`` independent simulations and return a column table.

        Each repeat re-randomises the mutant placement and advances this
        object's RNG stream, exactly as the per-task worker loop did. Returns a
        dict of three equal-length arrays: ``fixation`` (bool), ``steps``
        (int64), ``duration`` (float64).

        This default is a plain Python loop over :meth:`run`; engines with a
        native repeat loop (e.g. ``CppMoranProcess``) override it to keep the
        whole set of repeats on one side of the language boundary.
        """
        fixation = np.empty(n_repeats, dtype=np.bool_)
        steps = np.empty(n_repeats, dtype=np.int64)
        duration = np.empty(n_repeats, dtype=np.float64)
        for i in range(n_repeats):
            self.initialize_random_mutant(n_mutants)
            raw = self.run()
            fixation[i] = raw["fixation"]
            steps[i] = raw["steps"]
            duration[i] = raw["duration"]
        return {"fixation": fixation, "steps": steps, "duration": duration}

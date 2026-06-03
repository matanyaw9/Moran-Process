from abc import ABC, abstractmethod
import numpy as np

from moran_process.core.population_graph import GraphCore


class SimulationProcess(ABC):
    """Abstract base class for all graph-based evolutionary simulation processes."""

    def __init__(
        self,
        graph_core: GraphCore,
        selection_coefficient: float = 1.0,
        max_steps: int = 1_000_000,
        seed=None,
    ):
        self.r = selection_coefficient
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

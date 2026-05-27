from abc import ABC, abstractmethod
import numpy as np

from moran_process.core.population_graph import PopulationGraph


class SimulationProcess(ABC):
    """Abstract base class for all graph-based evolutionary simulation processes."""

    def __init__(
        self,
        population_graph: PopulationGraph,
        selection_coefficient: float = 1.0,
        max_steps: int = 1_000_000,
    ):
        self.pop_graph = population_graph
        self.r = selection_coefficient
        self.max_steps = max_steps
        self.n_nodes = self.pop_graph.number_of_nodes()
        self.state = np.zeros(self.n_nodes, dtype=int)
        self.adj_list = [
            list(self.pop_graph.graph.neighbors(n)) for n in range(self.n_nodes)
        ]

    @abstractmethod
    def step(self) -> None:
        """Execute one simulation step."""
        ...

    @abstractmethod
    def run(self, track_history: bool = False) -> dict:
        """Run the simulation to completion and return a result dict."""
        ...

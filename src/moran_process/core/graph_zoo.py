import math
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from moran_process.core.population_graph import PopulationGraph


class GraphZoo:
    """An ordered collection of PopulationGraph objects for a simulation study."""

    def __init__(self, name: str = ""):
        self.name = name
        self.graphs: list[PopulationGraph] = []

    # --- Adding graphs ---

    def add(self, graph: PopulationGraph) -> "GraphZoo":
        self.graphs.append(graph)
        return self


    # --- Visualization ---

    def draw_all(self, cols: int = 3) -> None:
        """Draw all graphs in a matplotlib grid. Each graph occupies one subplot."""
        n = len(self.graphs)
        if n == 0:
            print("No graphs in zoo.")
            return

        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        axes = np.array(axes).flatten()
        for i, graph in enumerate(self.graphs):
            graph.draw(ax=axes[i])
        for j in range(n, len(axes)):
            axes[j].set_visible(False)
        fig.suptitle(self.name or "Graph Zoo", fontsize=16)
        plt.tight_layout()
        plt.show()

    # --- Persistence ---

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Zoo saved to {path}")

    @staticmethod
    def load(path: str) -> "GraphZoo":
        with open(path, "rb") as f:
            return pickle.load(f)

    # --- Collection interface ---

    def __len__(self) -> int:
        return len(self.graphs)

    def __iter__(self):
        return iter(self.graphs)

    def __getitem__(self, idx: int) -> PopulationGraph:
        return self.graphs[idx]

    def __repr__(self) -> str:
        names = [g.name for g in self.graphs]
        return f"GraphZoo(name={self.name!r}, graphs={names})"

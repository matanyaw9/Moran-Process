import numpy as np
import random
import time

from moran_process.simulations.simulation_process import SimulationProcess


class MultiColorMoranProcess(SimulationProcess):
    """
    Multi-color Moran process: each node starts as its own unique lineage (color == node index).
    Reproduction is uniform (all colors neutral -- selection_coefficient is ignored).
    Terminates when one lineage takes over all nodes.

    State encoding: state[i] = color of node i, where color is the index of the
    founding ancestor node.
    """

    def initialize_unique_colors(self) -> None:
        """Assign each node its own color (color = node index)."""
        for i in range(self.n_nodes):
            self.state[i] = i

    def step(self) -> None:
        reproducer = np.random.randint(self.n_nodes)
        neighbors = self.nbrs[self.offsets[reproducer] : self.offsets[reproducer + 1]]
        if len(neighbors) > 0:
            victim = random.choice(neighbors)
            self.state[victim] = self.state[reproducer]

    def run(self, track_history: bool = False) -> dict:
        """
        Run until one color fixes or max_steps is reached.

        Returns keys: winner, steps, duration, and (if track_history=True)
        history -- a (steps x n_nodes) int array of full state snapshots.
        """
        start_time = time.perf_counter()
        steps = 0
        history = []

        while steps < self.max_steps:
            if track_history:
                history.append(self.state.copy())

            if np.all(self.state == self.state[0]):
                break

            self.step()
            steps += 1

        end_time = time.perf_counter()

        winner = int(self.state[0])
        result = {
            "winner": winner,
            "steps": steps,
            "duration": end_time - start_time,
        }
        if track_history:
            result["history"] = np.array(history)
        return result

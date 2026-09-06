import numpy as np
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
        reproducer = self._rng.integers(self.n_nodes)
        neighbors = self.nbrs[self.offsets[reproducer] : self.offsets[reproducer + 1]]
        if len(neighbors) > 0:
            victim = self._rng.choice(neighbors)
            self.state[victim] = self.state[reproducer]

    def run(self, track_history: bool = False) -> dict:
        """
        Run until one color fixes or max_steps is reached.

        Returns keys: winner, fixed, steps, duration, and (if track_history=True)
        history -- a (steps x n_nodes) int array of full state snapshots.

        ``fixed`` is False when the run was cut short by max_steps, and then
        ``winner`` is -1 rather than a node id. Reporting state[0] as the winner
        of a run that never reached consensus does not merely bias a statistic,
        it invents a data point: whichever lineage happened to sit on node 0
        would be credited with a win it never earned. This mirrors the
        ``censored`` column the two-color worker writes for the same reason.
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

        # Checked after the loop, not inside it: a run whose final step produced
        # consensus exits on the `steps < max_steps` condition without ever
        # reaching the in-loop test, and would otherwise be recorded as censored.
        fixed = bool(np.all(self.state == self.state[0]))
        result = {
            "winner": int(self.state[0]) if fixed else -1,
            "fixed": fixed,
            "steps": steps,
            "duration": end_time - start_time,
        }
        if track_history:
            result["history"] = np.array(history)
        return result

    def run_repeats(self, n_repeats: int, n_mutants: int = 1) -> dict:
        """Run ``n_repeats`` independent takeovers and return a column table.

        Overrides the base loop, which calls ``initialize_random_mutant``: that
        is the two-color setup and this class does not have it. Each repeat
        re-seeds every node with its own lineage and advances this object's RNG
        stream, so one seeded instance yields a reproducible but non-repeating
        sequence of trials.

        ``n_mutants`` is accepted and ignored, for the same reason
        ``selection_coefficient`` is: every node is its own lineage here, so
        there is no mutant count to set.

        Returns four equal-length arrays: ``winner`` (int64, -1 where the run was
        censored), ``fixed`` (bool), ``steps`` (int64), ``duration`` (float64).
        """
        winner = np.empty(n_repeats, dtype=np.int64)
        fixed = np.empty(n_repeats, dtype=np.bool_)
        steps = np.empty(n_repeats, dtype=np.int64)
        duration = np.empty(n_repeats, dtype=np.float64)
        for i in range(n_repeats):
            self.initialize_unique_colors()
            raw = self.run()
            winner[i] = raw["winner"]
            fixed[i] = raw["fixed"]
            steps[i] = raw["steps"]
            duration[i] = raw["duration"]
        return {
            "winner": winner,
            "fixed": fixed,
            "steps": steps,
            "duration": duration,
        }

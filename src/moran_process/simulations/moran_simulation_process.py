import numpy as np
import time

from moran_process.simulations.simulation_process import SimulationProcess


class MoranProcess(SimulationProcess):
    """
    Moran process: fitness-weighted birth, uniform-random neighbor replacement.
    Wild-type fitness = 1.0; mutant fitness = r (selection_coefficient).

    State encoding: 0 = wild-type, 1 = mutant.
    Terminates at fixation (all 1) or extinction (all 0).
    """

    def __init__(self, graph_core, selection_coefficient: float = 1.0,
                 max_steps: int = 1_000_000, seed=None):
        super().__init__(graph_core, selection_coefficient, max_steps, seed=seed)
        self.mutant_count: int = 0

    def reset(self) -> None:
        """Reset all nodes to wild-type and zero the mutant counter."""
        super().reset()
        self.mutant_count = 0

    def initialize_random_mutant(self, n_mutants: int = 1, seed=None) -> list[int]:
        """Place n_mutants mutants at randomly chosen nodes. Returns chosen node indices.

        seed: if given, re-anchors this instance's RNG from that point forward,
              making the placement and all subsequent steps reproducible.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.state.fill(0)
        if n_mutants > self.n_nodes:
            raise ValueError("Number of mutants exceeds number of nodes in the graph.")
        chosen = self._rng.choice(self.n_nodes, size=n_mutants, replace=False)
        self.state[chosen] = 1
        self.mutant_count = n_mutants
        return list(chosen)

    def step(self) -> None:
        fitness = np.where(self.state == 1, self.r, 1.0)
        probs = fitness / fitness.sum()
        reproducer = self._rng.choice(self.n_nodes, p=probs)
        neighbors = self.nbrs[self.offsets[reproducer] : self.offsets[reproducer + 1]]
        if len(neighbors) > 0:
            victim = self._rng.choice(neighbors)
            old_state = self.state[victim]
            self.state[victim] = self.state[reproducer]
            self.mutant_count += self.state[reproducer] - old_state

    def run(self, track_history: bool = False) -> dict:
        """
        Run until fixation, extinction, or max_steps.

        Returns keys: fixation, steps, initial_mutants, selection_coeff, duration,
        and (if track_history=True) history (array of mutant counts per step).
        """
        start_time = time.perf_counter()
        steps = 0
        fixation = False
        initial_mutants = self.mutant_count
        history = []

        while steps < self.max_steps:
            if track_history:
                history.append(self.mutant_count)

            if self.mutant_count == 0:
                break
            if self.mutant_count == self.n_nodes:
                fixation = True
                break

            self.step()
            steps += 1

        end_time = time.perf_counter()

        result = {
            "fixation": fixation,
            "steps": steps,
            "initial_mutants": initial_mutants,
            "selection_coeff": self.r,
            "duration": end_time - start_time,
        }
        if track_history:
            result["history"] = np.array(history)
        return result

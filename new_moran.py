
"""
new_moran.py  —  Core Moran process + optional Tkinter GUI

Usage examples
--------------
Headless (no GUI):
    python new_moran.py --N 100 --mutant-fitness 1.5 --initial-mutants 1

The core logic is in the MoranProcess class, which can be imported and
used independently of the GUI, e.g.:

    from moran import MoranProcess, make_well_mixed_adjacency
"""

from __future__ import annotations

import argparse
import math
import random
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple


def make_well_mixed_adjacency(n: int) -> NDArray[np.int_]:
    """
    Create a fully-connected (well-mixed) replacement graph:
    individual i can replace j for all i != j.
    Represented as an n x n matrix of 0/1.
    """
    if n <= 0:
        raise ValueError("Population size n must be positive.")
    return np.ones((n, n), dtype=int)- np.eye(n, dtype=int)


@dataclass
class MoranProcess:
    """
    Birth-death Moran process on a general directed graph.

    - Population of N individuals, each with a discrete 'type' (0, 1, 2, ...).
    - fitness[type] gives the relative reproductive rate of that type.
    - adjacency[i,j] != 0 means individual i is allowed to replace j.

    Step dynamics (birth-death variant):
    1. Pick a reproducing individual i with probability ~ fitness[type_i].
    2. Pick a neighbor j from the outgoing neighbors of i (adjacency[i,j] != 0),
       uniformly at random.
    3. Set type_j = type_i (j is replaced by offspring of i).
    """

    N: int
    blobs: NDArray[np.str_]
    fitness: Dict[str, float]
    adjacency: Optional[NDArray[np.int_]] = None
    rng: random.Random = field(default_factory=random.Random)

    def __post_init__(self) -> None:
        if self.N <= 0:
            raise ValueError("Population size N must be positive.")
        if len(self.blobs) != self.N:
            raise ValueError("Length of 'blobs' must equal N.")

        # Fill in default fitness 1.0 for any unseen type
        unique_types = set(self.blobs)
        for t in unique_types:
            self.fitness.setdefault(t, 1.0)

        # Default: well-mixed population
        if self.adjacency is None:
            self.adjacency = make_well_mixed_adjacency(self.N)

        # Basic sanity checks on adjacency:
        # Normalize/validate adjacency as an ndarray N x N with 0/1 entries
        self.adjacency = np.asarray(self.adjacency)
        
        if self.adjacency.ndim != 2 or self.adjacency.shape != (self.N, self.N):
            raise ValueError("Adjacency must be an N x N matrix.")
        
        if not np.all(np.isin(self.adjacency, [0, 1])):
            raise ValueError("Adjacency entries must be 0 or 1.")
        # ensure integer dtype for subsequent indexing/logic
        self.adjacency = self.adjacency.astype(np.int_)
        if not np.any(self.adjacency):
            raise ValueError("Adjacency must have at least one allowed replacement edge.")

    def step(self) -> Tuple[int, int]:
        """
        Perform one birth-death step.

        Returns
        -------
        (reproducer_index, replacee_index)
        """
        # 1) Choose reproducer according to fitness
        weights = [self.fitness[self.blobs[i]] for i in range(self.N)]
        total = sum(weights)
        if total <= 0:
            raise RuntimeError("Total fitness is non-positive; cannot sample reproducer.")

        r = self.rng.random() * total
        cumulative = 0.0
        reproducer = 0
        for i, w in enumerate(weights):
            cumulative += w
            if r <= cumulative:
                reproducer = i
                break

        # 2) Choose neighbor to replace among outgoing neighbors (numpy)
        neighbors = np.flatnonzero(self.adjacency[reproducer])  # ndarray of indices with non-zero entry
        if neighbors.size == 0:
            raise RuntimeError(f"Individual {reproducer} has no outgoing neighbors in adjacency matrix.")
        replacee = int(np.random.choice(neighbors))

        # 3) Offspring replaces neighbor
        self.blobs[replacee] = self.blobs[reproducer]
        return reproducer, replacee

    # ---- Convenience methods for analysis / non-GUI usage -----------------

    def counts(self) -> Dict[str, int]:
        """Return a dictionary mapping type -> number of individuals of that type."""
        arr = np.asarray(self.blobs)
        unique, counts = np.unique(arr, return_counts=True)
        return {u.item(): int(c) for u, c in zip(unique, counts)}

    def is_fixated(self) -> bool:
        """Return True if the population is monomorphic (all the same type)."""
        return len(set(self.blobs)) == 1

    def run(
        self,
        max_steps: int,
        callback: Optional[Callable[[int, "MoranProcess"], None]] = None,
    ) -> Tuple[int, bool]:
        """
        Run the process for up to max_steps or until fixation.

        Parameters
        ----------
        max_steps : int
            Maximum number of steps to simulate.
        callback : Optional[Callable[[int, MoranProcess], None]]
            If provided, called after each step with (step_index, self).

        Returns
        -------
        steps_run : int
        fixated : bool
        """
        for step_index in range(1, max_steps + 1):
            self.step()
            if callback is not None:
                callback(step_index, self)
            if self.is_fixated():
                return step_index, True
        return max_steps, self.is_fixated()


# # ---------------------------------------------------------------------------
# # GUI wrapper (optional)
# # ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Command-line interface (headless + GUI switch)
# ---------------------------------------------------------------------------


def run_headless(
    N: int,
    mutant_fitness: float,
    initial_mutants: int,
    max_steps: int,
    seed: Optional[int] = None,
) -> None:
    rng = random.Random(seed)
    if initial_mutants <= 0 or initial_mutants > N:
        raise ValueError("initial_mutants must be between 1 and N.")

    blobs = [0 for _ in range(N)]
    indices = list(range(N))
    rng.shuffle(indices)
    for idx in indices[:initial_mutants]:
        blobs[idx] = 1

    fitness = {0: 1.0, 1: mutant_fitness}
    process = MoranProcess(N=N, blobs=blobs, fitness=fitness, rng=rng)

    print(f"Starting headless Moran process with N={N}, initial_mutants={initial_mutants}, "
          f"mutant_fitness={mutant_fitness}, max_steps={max_steps}")
    print(f"Initial counts: {process.counts()}")

    def cb(step_index: int, proc: MoranProcess) -> None:
        if step_index % max(1, max_steps // 10) == 0:
            print(f"Step {step_index}: counts={proc.counts()}")

    steps_run, fixated = process.run(max_steps=max_steps, callback=cb)

    print(f"Finished after {steps_run} steps. Fixated={fixated}")
    print(f"Final counts: {process.counts()}")
    if fixated:
        t = next(iter(process.counts().keys()))
        print(f"Fixated type: {t}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Moran process simulator with optional GUI.")
    parser.add_argument("--N", type=int, default=100, help="Population size.")
    parser.add_argument("--mutant-fitness", type=float, default=1.5, help="Fitness of mutant type (wild-type=1.0).")
    parser.add_argument("--initial-mutants", type=int, default=1, help="Number of mutants in the initial population.")
    parser.add_argument("--max-steps", type=int, default=10000, help="Maximum number of steps for headless run.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    args = parser.parse_args()

    run_headless(
        N=args.N,
        mutant_fitness=args.mutant_fitness,
        initial_mutants=args.initial_mutants,
        max_steps=args.max_steps,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

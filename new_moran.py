
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
import logging
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("moran")


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

    - blobs is an array of N individuals, each with a discrete 'type' ('A', 'B', ...).
    - fitness[type] gives the relative reproductive rate of that type.
    - adjacency[i,j] != 0 means individual i is allowed to replace j.

    Step dynamics (birth-death variant):
    1. Pick a reproducing individual i with probability ~ fitness[type_i].
    2. Pick a neighbor j from the outgoing neighbors of i (adjacency[i,j] != 0),
       uniformly at random.
    3. Set type_j = type_i (j is replaced by offspring of i).
    """

    blobs: NDArray[np.str_]
    fitness: Dict[str, float]
    adjacency: Optional[NDArray[np.int_]] = None
    rng: random.Random = field(default_factory=random.Random)

    def __post_init__(self) -> None:

        # Fill in default fitness 1.0 for any unseen type
        self.N = self.blobs.size
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

    # ---- Convenience methods for analysis -----------------

    def counts(self) -> Dict[str, int]:
        """Return a dictionary mapping type -> number of individuals of that type."""
        if type(self.blobs) is not np.ndarray:
            arr = np.asarray(self.blobs)
        else:
            arr = self.blobs
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
                return step_index, self.blobs[0]  # all same type
        return max_steps, 0  # not fixated

def run_headless(
    N: int,
    initial_mutants: int,
    mutant_fitness: float=1.0,
    max_steps: int=10_000,
    seed: Optional[int] = None,
) -> None:
    rng = random.Random(seed)   # if seed is None, uses system time or entropy source
    if initial_mutants <= 0 or initial_mutants > N:
        raise ValueError("initial_mutants must be between 1 and N.")

    blobs = np.full(N, 'A', dtype='<U1')  # all 'A' initially
    # pick initial_mutants distinct indices and mark them as 'B'
    mutant_indices = rng.sample(range(N), initial_mutants)
    blobs[mutant_indices] = 'B'
    fitness = {'A': 1.0, 'B': mutant_fitness}
    process = MoranProcess(blobs=blobs, fitness=fitness, rng=rng)

    logger.info(f"Starting headless Moran process with N={N}, initial_mutants={initial_mutants}, "
          f"mutant_fitness={mutant_fitness}, max_steps={max_steps}")
    logger.info(f"Initial counts: {process.counts()}")

    def basic_callback(step_index: int, proc: MoranProcess) -> None:
        pass
        # if step_index % 10 == 0:
        #     logger.info(f"Step {step_index}: counts={proc.counts()}")

    steps_run, fixated = process.run(max_steps=max_steps, callback=basic_callback)

    logger.info(f"Finished after {steps_run} steps. Fixated={fixated}")
    logger.info(f"Final counts: {process.counts()}")
    if fixated:
        logger.info(f"Fixated type: {fixated}")
    return fixated


def main() -> None:
    parser = argparse.ArgumentParser(description="Moran process simulator with optional GUI.")
    parser.add_argument("--N", type=int, default=20, help="Population size.")
    parser.add_argument("--mutant-fitness", type=float, default=1, help="Fitness of mutant type (wild-type=1.0).")
    parser.add_argument("--initial-mutants", type=int, default=10, help="Number of mutants in the initial population.")
    parser.add_argument("--max-steps", type=int, default=10000, help="Maximum number of steps for headless run.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--verbose", action="store_true", help="Enable detailed output.")
    args = parser.parse_args()
    logging.basicConfig(
            level=logging.INFO if args.verbose else logging.WARNING,
            format="%(message)s",
        )

    run_headless(
        N=args.N,
        initial_mutants=args.initial_mutants,
        mutant_fitness=args.mutant_fitness,
        max_steps=args.max_steps,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

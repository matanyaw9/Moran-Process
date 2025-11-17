
"""
moran.py  —  Core Moran process + optional Tkinter GUI

Usage examples
--------------
Headless (no GUI):
    python moran.py --N 100 --mutant-fitness 1.5 --initial-mutants 1

With GUI (if Tkinter and a display are available):
    python moran.py --gui

The core logic is in the MoranProcess class, which can be imported and
used independently of the GUI, e.g.:

    from moran import MoranProcess, make_well_mixed_adjacency
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

# Tkinter is optional: we import it lazily so the core logic works without a GUI.
try:
    import tkinter as tk
    from tkinter import ttk
except Exception:  # ImportError or no Tk on the system
    tk = None
    ttk = None


def make_well_mixed_adjacency(n: int) -> List[List[int]]:
    """
    Create a fully-connected (well-mixed) replacement graph:
    individual i can replace j for all i != j.
    Represented as an n x n matrix of 0/1.
    """
    if n <= 0:
        raise ValueError("Population size n must be positive.")
    return [[1 if i != j else 0 for j in range(n)] for i in range(n)]


@dataclass
class MoranProcess:
    """
    Birth-death Moran process on a general directed graph.

    - Population of N individuals, each with a discrete 'type' (0, 1, 2, ...).
    - fitness[type] gives the relative reproductive rate of that type.
    - adjacency[i][j] != 0 means individual i is allowed to replace j.

    Step dynamics (birth-death variant):
    1. Pick a reproducing individual i with probability ~ fitness[type_i].
    2. Pick a neighbor j from the outgoing neighbors of i (adjacency[i][j] != 0),
       uniformly at random.
    3. Set type_j = type_i (j is replaced by offspring of i).
    """

    N: int
    types: List[int]
    fitness: Dict[int, float]
    adjacency: Optional[List[List[int]]] = None
    rng: random.Random = field(default_factory=random.Random)

    # last_event stores the last (reproducer_index, replacee_index), useful for GUI arrows later
    last_event: Optional[Tuple[int, int]] = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.N <= 0:
            raise ValueError("Population size N must be positive.")
        if len(self.types) != self.N:
            raise ValueError("Length of 'types' must equal N.")

        # Fill in default fitness 1.0 for any unseen type
        unique_types = set(self.types)
        for t in unique_types:
            self.fitness.setdefault(t, 1.0)

        # Default: well-mixed population
        if self.adjacency is None:
            self.adjacency = make_well_mixed_adjacency(self.N)

        # Basic sanity checks on adjacency
        if len(self.adjacency) != self.N or any(len(row) != self.N for row in self.adjacency):
            raise ValueError("Adjacency must be an N x N matrix.")
        if all(all(val == 0 for val in row) for row in self.adjacency):
            raise ValueError("Adjacency must have at least one allowed replacement edge.")

    def step(self) -> Tuple[int, int]:
        """
        Perform one birth-death step.

        Returns
        -------
        (reproducer_index, replacee_index)
        """
        # 1) Choose reproducer according to fitness
        weights = [self.fitness[self.types[i]] for i in range(self.N)]
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

        # 2) Choose neighbor to replace among outgoing neighbors
        neighbors = [j for j, val in enumerate(self.adjacency[reproducer]) if val]
        if not neighbors:
            raise RuntimeError(f"Individual {reproducer} has no outgoing neighbors in adjacency matrix.")
        replacee = self.rng.choice(neighbors)

        # 3) Offspring replaces neighbor
        self.types[replacee] = self.types[reproducer]
        self.last_event = (reproducer, replacee)
        return reproducer, replacee

    # ---- Convenience methods for analysis / non-GUI usage -----------------

    def counts(self) -> Dict[int, int]:
        """Return a dictionary mapping type -> number of individuals of that type."""
        out: Dict[int, int] = {}
        for t in self.types:
            out[t] = out.get(t, 0) + 1
        return out

    def is_fixated(self) -> bool:
        """Return True if the population is monomorphic (all the same type)."""
        return len(set(self.types)) == 1

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


# ---------------------------------------------------------------------------
# Tkinter GUI wrapper (optional)
# ---------------------------------------------------------------------------

TYPE_COLORS = {
    0: "#DDDDDD",  # grey
    1: "#E41A1C",  # red
    2: "#377EB8",  # blue
    3: "#4DAF4A",  # green
    4: "#984EA3",  # purple
}


def _type_to_color(t: int) -> str:
    return TYPE_COLORS.get(t, "#000000")  # default black


class MoranGUI:
    """
    Simple Tkinter GUI wrapper around a MoranProcess.

    - Shows individuals as colored squares on a grid.
    - Start / Stop / Step / Reset controls.
    - Speed slider.
    - Designed so that future features like arrows between reproducer and
      replacee, or adjacency visualization, can be added in one place.
    """

    def __init__(self, root: "tk.Tk", N: int = 100) -> None:
        if tk is None:
            raise RuntimeError("Tkinter is not available; GUI cannot be created.")

        self.root = root
        self.N = N
        self.canvas_size = 500
        self.running = False
        self.after_id: Optional[str] = None
        self.step_count = 0

        # GUI variables
        self.speed_ms = tk.IntVar(value=100)  # delay between steps in ms
        self.mutant_fitness = tk.DoubleVar(value=1.5)

        # Layout
        main = ttk.Frame(root, padding=5)
        main.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(
            main, width=self.canvas_size, height=self.canvas_size, bg="white", highlightthickness=0
        )
        self.canvas.grid(row=0, column=0, columnspan=4, pady=(0, 5))

        self.start_button = ttk.Button(main, text="Start", command=self.start)
        self.start_button.grid(row=1, column=0, sticky="ew")

        self.stop_button = ttk.Button(main, text="Stop", command=self.stop)
        self.stop_button.grid(row=1, column=1, sticky="ew")

        self.step_button = ttk.Button(main, text="Step", command=self.single_step)
        self.step_button.grid(row=1, column=2, sticky="ew")

        self.reset_button = ttk.Button(main, text="Reset", command=self.reset_process)
        self.reset_button.grid(row=1, column=3, sticky="ew")

        ttk.Label(main, text="Speed (ms):").grid(row=2, column=0, sticky="w")
        self.speed_scale = ttk.Scale(
            main, from_=10, to=1000, orient="horizontal", variable=self.speed_ms
        )
        self.speed_scale.grid(row=2, column=1, columnspan=2, sticky="ew")

        ttk.Label(main, text="Mutant fitness:").grid(row=3, column=0, sticky="w")
        self.fitness_entry = ttk.Entry(main, textvariable=self.mutant_fitness, width=6)
        self.fitness_entry.grid(row=3, column=1, sticky="w")

        self.step_var = tk.IntVar(value=0)
        ttk.Label(main, text="Step:").grid(row=3, column=2, sticky="e")
        self.step_label = ttk.Label(main, textvariable=self.step_var, width=6)
        self.step_label.grid(row=3, column=3, sticky="w")

        # Make columns expand nicely
        for col in range(4):
            main.columnconfigure(col, weight=1)

        # Internal state for drawing
        self.rectangles: List[int] = []
        self.rows: int = 0
        self.cols: int = 0

        # Create underlying process and first drawing
        self.process: MoranProcess
        self.reset_process()

    # ---- Simulation / GUI interaction -------------------------------------

    def reset_process(self) -> None:
        """Reset to a single mutant in a population of wild-type."""
        self.stop()
        self.step_count = 0
        self.step_var.set(0)

        types = [0 for _ in range(self.N)]
        # Put a few mutants at random positions
        num_mutants = max(1, self.N // 10)  # 10% mutants by default
        mutant_indices = list(range(self.N))
        random.shuffle(mutant_indices)
        for idx in mutant_indices[:num_mutants]:
            types[idx] = 1

        fitness = {0: 1.0, 1: float(self.mutant_fitness.get())}

        self.process = MoranProcess(N=self.N, types=types, fitness=fitness)
        self._init_grid()
        self._redraw_population()

    def _init_grid(self) -> None:
        self.canvas.delete("all")
        self.rectangles.clear()

        # Compute grid dimensions (roughly square)
        self.cols = int(math.ceil(math.sqrt(self.N)))
        self.rows = int(math.ceil(self.N / self.cols))

        cell_w = self.canvas_size / self.cols
        cell_h = self.canvas_size / self.rows

        for idx in range(self.N):
            row = idx // self.cols
            col = idx % self.cols
            x0 = col * cell_w
            y0 = row * cell_h
            x1 = (col + 1) * cell_w
            y1 = (row + 1) * cell_h
            rect = self.canvas.create_rectangle(
                x0, y0, x1, y1, outline="#888888", fill=_type_to_color(self.process.types[idx])
            )
            self.rectangles.append(rect)

    def _redraw_population(self) -> None:
        for idx, t in enumerate(self.process.types):
            self.canvas.itemconfig(self.rectangles[idx], fill=_type_to_color(t))

    def start(self) -> None:
        if not self.running:
            self.running = True
            self._schedule_next_step()

    def stop(self) -> None:
        self.running = False
        if self.after_id is not None:
            try:
                self.root.after_cancel(self.after_id)
            except Exception:
                pass
            self.after_id = None

    def _schedule_next_step(self) -> None:
        if self.running:
            delay = max(1, int(self.speed_ms.get()))
            self.after_id = self.root.after(delay, self._timer_step)

    def _timer_step(self) -> None:
        self.single_step()
        self._schedule_next_step()

    def single_step(self) -> None:
        if self.process.is_fixated():
            self.stop()
            return

        self.process.step()
        self.step_count += 1
        self.step_var.set(self.step_count)
        self._redraw_population()

        # NOTE: in the future you can use self.process.last_event here to draw
        # an arrow from reproducer to replacee, based on the adjacency matrix.

    # ------------------------------------------------------------------


def run_gui(N: int = 100) -> None:
    if tk is None:
        raise RuntimeError("Tkinter is not available; GUI cannot be started.")
    root = tk.Tk()
    root.title("Moran Process Simulation")
    MoranGUI(root, N=N)
    root.mainloop()


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

    types = [0 for _ in range(N)]
    indices = list(range(N))
    rng.shuffle(indices)
    for idx in indices[:initial_mutants]:
        types[idx] = 1

    fitness = {0: 1.0, 1: mutant_fitness}
    process = MoranProcess(N=N, types=types, fitness=fitness, rng=rng)

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
    parser.add_argument("--gui", action="store_true", help="Run with Tkinter GUI.")
    args = parser.parse_args()

    if args.gui:
        if tk is None:
            parser.error("Tkinter is not available on this system; cannot run GUI.")
        run_gui(N=args.N)
    else:
        run_headless(
            N=args.N,
            mutant_fitness=args.mutant_fitness,
            initial_mutants=args.initial_mutants,
            max_steps=args.max_steps,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()

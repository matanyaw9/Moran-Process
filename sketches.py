from new_moran import run_headless
import numpy as np

if __name__ == "__main__":
    # Simple test run of the Moran process
    num_runs = 10000
    fixations = np.empty(num_runs, dtype=np.str_)
    for i in range(num_runs):
        fixations[i] = run_headless(N=10, initial_mutants=1)
    unique, counts = np.unique(fixations, return_counts=True)
    print("Fixation results over", num_runs, "runs:")
    for u, c in zip(unique, counts):
        print(f"Type {u}: {c} times")

    

    
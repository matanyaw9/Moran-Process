import argparse
import os
import sys
import pandas as pd
import joblib

from moran_process.simulations.moran_simulation_process import MoranProcess


def load_data(zoo_shard_dir, task_manifest_path, worker_index):
    """Load this worker's GraphCore shard and the shared task manifest.

    The shard is a list[GraphCore] containing only the graphs this worker
    references, already in compact CSR form. No NetworkX objects are loaded.
    """
    shard_path = os.path.join(zoo_shard_dir, f"zoo_worker_{worker_index}.pkl")
    if not os.path.exists(shard_path):
        raise FileNotFoundError(f"Shard not found: {shard_path}")
    if not os.path.exists(task_manifest_path):
        raise FileNotFoundError(f"Manifest not found: {task_manifest_path}")

    with open(shard_path, "rb") as f:
        graph_zoo = joblib.load(f)
    print(f"[Worker {worker_index}] Loaded {len(graph_zoo)} graphs from shard.")

    manifest_df = pd.read_csv(task_manifest_path)
    print(f"[Worker {worker_index}] Loaded manifest ({len(manifest_df)} total tasks across all workers).")

    return graph_zoo, manifest_df


def run_worker_slice(batch_dir, zoo_shard_dir, manifest_path, worker_index):
    """Run simulations for all tasks assigned to this LSF job index.

    1. Load this worker's GraphCore shard and filter manifest to our rows.
    2. Run simulations for each (graph, r) task.
    3. Write results to a per-job CSV in batch_dir/results/.
    """
    print(f"--- Worker {worker_index} started ---")

    # 1. Load data
    graph_zoo, manifest_df = load_data(zoo_shard_dir, manifest_path, worker_index)

    # 2. Filter to tasks assigned to this worker (worker_id is 1-based, matches LSB_JOBINDEX)
    my_tasks = manifest_df[manifest_df['worker_id'] == worker_index]
    print('='*60)
    print(my_tasks)
    print('='*60)

    if my_tasks.empty:
        print(f"[Worker {worker_index}] No tasks found for this worker_id. Exiting.")
        return

    n_tasks = len(my_tasks)
    total_sims = int(my_tasks['n_repeats'].sum())
    print(f"[Worker {worker_index}] {n_tasks} tasks / {total_sims} simulations.")

    # 3. Run simulations
    results_buffer = []
    for task_num, row in enumerate(my_tasks.itertuples(), start=1):
        try:
            # Shard uses local_graph_idx (0-based within this worker's shard)
            graph_core = graph_zoo[row.local_graph_idx]
            r_val = row.r_value
            n_repeats = row.n_repeats

            print(f"[Worker {worker_index}] Task {task_num}/{n_tasks} | "
                  f"graph={graph_core.name} r={r_val} reps={n_repeats}")

            for _ in range(n_repeats):
                sim = MoranProcess(graph_core=graph_core, selection_coefficient=r_val)
                sim.initialize_random_mutant()
                raw_result = sim.run()

                record = {
                    "task_id": row.task_id,
                    "job_id": worker_index,
                    "wl_hash": graph_core.wl_hash,
                    "graph_name": graph_core.name,
                    "r": r_val,
                    "fixation": raw_result["fixation"],
                    "steps": raw_result["steps"],
                    "initial_mutants": raw_result["initial_mutants"],
                    "duration": raw_result["duration"],
                }
                results_buffer.append(record)

        except Exception as e:
            print(f"[Worker {worker_index}] ERROR in task {row.task_id}: {e}")
            continue

    # 4. Write results
    if results_buffer:
        results_df = pd.DataFrame(results_buffer)
        filename = f"result_job_{worker_index}.csv"
        save_path = os.path.join(batch_dir, "results", filename)
        results_df.to_csv(save_path, index=False)
        print(f"--- Worker {worker_index} finished. Saved {len(results_df)} rows to {filename} ---")
    else:
        print(f"--- Worker {worker_index} finished. No results generated. ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HPC worker: runs Moran simulations for one LSF array job index."
    )
    parser.add_argument("--zoo-shard-dir", required=True,
                        help="Directory containing per-worker GraphCore shards (zoo_worker_N.pkl)")
    parser.add_argument("--manifest-path", required=True,
                        help="Path to the shared task_manifest.csv")
    parser.add_argument("--batch-dir", required=True,
                        help="Path to the batch tmp/ directory; results written to batch-dir/results/")
    parser.add_argument("--job-index", type=int, default=None,
                        help="Override job index (default: read from $LSB_JOBINDEX)")

    args = parser.parse_args()

    # Resolve job index: explicit arg takes precedence, then LSF env variable
    job_idx = args.job_index
    if job_idx is None:
        env_idx = os.environ.get("LSB_JOBINDEX")
        if env_idx:
            job_idx = int(env_idx)
        else:
            print("ERROR: No job index found. Pass --job-index or submit via bsub.")
            sys.exit(1)

    run_worker_slice(args.batch_dir, args.zoo_shard_dir, args.manifest_path, job_idx)

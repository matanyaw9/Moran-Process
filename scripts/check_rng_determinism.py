"""
Verify RNG determinism at both the graph-generation and simulation levels.

Batch setup:
  - Batches 1 & 2: GRAPH_ZOO_SEED=42, batch_seed=42
      -> identical random graph topologies AND identical simulation results
  - Batches 3 & 4: GRAPH_ZOO_SEED=None, batch_seed=None
      -> different random graph topologies AND different simulation results

Run with:
    uv run python scripts/check_rng_determinism.py

Not part of the automated test suite — run manually after generating new test batches.
"""

import os
import glob
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

BATCH_DIR = "simulation_data"
BATCHES = {
    1: "2026-06-04_graph_creation_randomness-1",
    2: "2026-06-04_graph_creation_randomness-2",
    3: "2026-06-04_graph_creation_randomness-3",
    4: "2026-06-04_graph_creation_randomness-4",
}

SEEDED     = [1, 2]   # same seed=42
UNSEEDED   = [3, 4]   # seed=None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_manifest(batch_num: int) -> pd.DataFrame:
    path = os.path.join(BATCH_DIR, BATCHES[batch_num], "tmp", "task_manifest.csv")
    return pd.read_csv(path)


def load_results(batch_num: int) -> pd.DataFrame:
    """Aggregate all per-job parquet files into one sorted DataFrame.

    Columns compared: wl_hash, graph_name, r, fixation, steps.
    duration and job_id are excluded (wall-clock / LSF-assigned, non-deterministic).
    Rows are sorted by (task_id, row_within_task) so order is canonical.
    """
    pattern = os.path.join(BATCH_DIR, BATCHES[batch_num], "tmp", "results", "*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No result parquet files found for batch {batch_num}")

    tables = [pq.read_table(f) for f in files]
    df = pa.concat_tables(tables).to_pandas()

    # Assign a stable within-task row index so cross-batch row order is comparable
    df["row_within_task"] = df.groupby("task_id").cumcount()
    df = df.sort_values(["task_id", "row_within_task"]).reset_index(drop=True)

    return df[["task_id", "wl_hash", "graph_name", "r", "fixation", "steps", "row_within_task"]]


def load_graph_hashes(batch_num: int) -> set:
    path = os.path.join(BATCH_DIR, BATCHES[batch_num], "graph_props.csv")
    return set(pd.read_csv(path)["wl_hash"])


def results_equal(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    return a.reset_index(drop=True).equals(b.reset_index(drop=True))


def results_differ(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    return not results_equal(a, b)


# ---------------------------------------------------------------------------
# Load everything
# ---------------------------------------------------------------------------

print("Loading manifests ...")
manifests = {i: load_manifest(i) for i in range(1, 5)}

print("Loading results (this may take a moment) ...")
results = {i: load_results(i) for i in range(1, 5)}

print("Loading zoo graph hashes ...")
hashes = {i: load_graph_hashes(i) for i in range(1, 5)}

print()

# ---------------------------------------------------------------------------
# 1. Manifest seed identity: batches 1 and 2 must be identical
# ---------------------------------------------------------------------------

print("=" * 60)
print("CHECK 1: Manifest seeds — batches 1 and 2 must be identical")
print("=" * 60)

m1_seeds = manifests[1]["seed"].tolist()
m2_seeds = manifests[2]["seed"].tolist()
assert m1_seeds == m2_seeds, (
    f"FAIL: manifest seeds differ between batch 1 and 2\n"
    f"  batch1 seeds[:5]: {m1_seeds[:5]}\n"
    f"  batch2 seeds[:5]: {m2_seeds[:5]}"
)
print(f"  PASS: {len(m1_seeds)} task seeds are identical between batches 1 and 2.")

# ---------------------------------------------------------------------------
# 2. Manifest seeds: batches 3 and 4 must differ from each other and from 1/2
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("CHECK 2: Manifest seeds — batches 3 & 4 must differ (from 1/2 and each other)")
print("=" * 60)

m3_seeds = manifests[3]["seed"].tolist()
m4_seeds = manifests[4]["seed"].tolist()

assert m3_seeds != m1_seeds, "FAIL: batch 3 seeds identical to batch 1 (expected random)"
assert m4_seeds != m1_seeds, "FAIL: batch 4 seeds identical to batch 1 (expected random)"
assert m3_seeds != m4_seeds, "FAIL: batch 3 and 4 seeds are identical (both should be random)"
print("  PASS: batch 3 and 4 seeds each differ from batch 1/2 and from each other.")
print(f"  Batch 1 seeds[:3]: {m1_seeds[:3]}")
print(f"  Batch 2 seeds[:3]: {m2_seeds[:3]}")
print(f"  Batch 3 seeds[:3]: {m3_seeds[:3]}")
print(f"  Batch 4 seeds[:3]: {m4_seeds[:3]}")

# ---------------------------------------------------------------------------
# 3. Results identity: batches 1 and 2 must produce identical outcomes
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("CHECK 3: Simulation results — batches 1 and 2 must be identical")
print("=" * 60)

assert results_equal(results[1], results[2]), (
    "FAIL: results differ between batch 1 and batch 2 despite identical seeds.\n"
    + results[1].compare(results[2]).to_string()
)
print(f"  PASS: {len(results[1])} rows are identical between batches 1 and 2 "
      f"(fixation + steps match row-for-row).")

# ---------------------------------------------------------------------------
# 4. Results differ: batches 3 & 4 must differ from each other and from 1/2
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("CHECK 4: Simulation results — batches 3 & 4 must differ from 1/2 and each other")
print("=" * 60)

pairs = [(1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
for a, b in pairs:
    assert results_differ(results[a], results[b]), (
        f"FAIL: batches {a} and {b} produced identical results — "
        f"expected them to differ (different RNG seeds)."
    )
    print(f"  PASS: batch {a} vs batch {b} — results differ as expected.")

# ---------------------------------------------------------------------------
# 5. Zoo graph hashes: batches 1&2 identical, batches 3&4 differ
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("CHECK 5: Zoo graph hashes")
print("=" * 60)

# Batches 1 & 2: same GRAPH_ZOO_SEED=42 -> identical random graph topologies
assert hashes[1] == hashes[2], (
    f"FAIL: batches 1 and 2 have different graph zoos despite same GRAPH_ZOO_SEED.\n"
    f"  only in batch 1: {hashes[1] - hashes[2]}\n"
    f"  only in batch 2: {hashes[2] - hashes[1]}"
)
print(f"  PASS: batches 1 and 2 have identical graph zoos ({len(hashes[1])} graphs).")

# Batches 3 & 4: GRAPH_ZOO_SEED=None -> different random graph topologies
assert hashes[3] != hashes[1], "FAIL: batch 3 graphs identical to batch 1 (expected random topology)"
assert hashes[4] != hashes[1], "FAIL: batch 4 graphs identical to batch 1 (expected random topology)"
assert hashes[3] != hashes[4], "FAIL: batch 3 and 4 have identical graphs (both should be random)"
print("  PASS: batches 3 and 4 each have different graph zoos from batch 1/2 and from each other.")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("ALL CHECKS PASSED")
print("=" * 60)

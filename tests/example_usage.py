#!/usr/bin/env python
"""
Example demonstrating the simplified ProcessLab usage
"""

from population_graph import PopulationGraph
from process_lab import ProcessLab

# Create some test graphs
graphs = [
    PopulationGraph.complete_graph(N=10, register_in_db=False),
    PopulationGraph.cycle_graph(N=10, register_in_db=False)
]

# Initialize lab
lab = ProcessLab()

# Example 1: Run without saving (original behavior)
print("Example 1: Run without saving")
df1 = lab.run_comparative_study(
    graphs, 
    r_values=[1.0, 1.2], 
    n_repeats=10
)
print(f"Returned DataFrame with {len(df1)} rows\n")

# Example 2: Run with automatic saving
print("Example 2: Run with automatic saving")
df2 = lab.run_comparative_study(
    graphs, 
    r_values=[1.5, 2.0], 
    n_repeats=10,
    output_path="simulation_data/example_results.csv"
)
print(f"Returned DataFrame with {len(df2)} rows")
print("Results automatically saved to CSV\n")

# Example 3: Run again - will append to existing file
print("Example 3: Run again - appends to existing file")
df3 = lab.run_comparative_study(
    graphs, 
    r_values=[2.5], 
    n_repeats=10,
    output_path="simulation_data/example_results.csv"
)
print(f"Returned DataFrame with {len(df3)} rows")
print("Results appended to existing CSV")

# Example 4: HPC job submission (simplified)
print("\nExample 4: HPC job submission")
try:
    tracking_info = lab.submit_jobs(
        graphs=graphs,
        r_values=[1.0, 1.2, 1.5],
        n_repeats=100,
        n_jobs=5,
        memory="2GB",  # Use less memory for test
        walltime="4:00"
    )
    print(f"Submitted job array: {tracking_info['job_id']}")
except Exception as e:
    print(f"Job submission test (expected to fail in test environment): {type(e).__name__}")
    print("This would work in a real LSF environment")

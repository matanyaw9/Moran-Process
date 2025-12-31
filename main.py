import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from population_graph import PopulationGraph
from process_lab import ProcessLab

# 1. SETUP EXPERIMENT SUITE
# Define a list of dictionaries. Each entry is one "Configuration"
experiments = [
    {
        "name": "Complete (Control)",
        "factory": lambda: PopulationGraph.complete_graph(30)
    },
    {
        "name": "Cycle",
        "factory": lambda: PopulationGraph.cycle_graph(30)
    },
    {
        "name": "Mammalian Lung",
        "factory": lambda: PopulationGraph.mammalian_lung_graph(branching_factor=2, depth=4) 
    },
    {
        "name": "Avian Lung",
        "factory": lambda: PopulationGraph.avian_graph(n_rods=3, rod_length=8)
    },
    {
        "name": "Fish Gills",
        # Custom params to keep N ~ 30
        "factory": lambda: PopulationGraph.fish_graph(n_rods=3, rod_length=3) 
    }
]

# 2. RUN BATCHES
lab = ProcessLab()
r_values = [1.0, 1.05, 1.1, 1.2]
all_data = []

print("--- Starting Comparative Study ---")
for exp in experiments:
    print(f"Running: {exp['name']}")
    
    # 1. Run the Simulations
    # (Note: You might need to add a 'clear_buffer' method to ProcessLab or access .results_buffer directly)
    lab.results_buffer = [] 
    lab.run_experiment(exp['factory'], r_values, n_repeats=100)
    
    # 2. Convert to DataFrame
    df = pd.DataFrame(lab.results_buffer)
    
    # 3. TAG THE DATA (Crucial Step!)
    df['Graph_Type'] = exp['name']
    
    # 4. Add Graph Metrics (N is critical for 1/N comparison)
    temp_graph = exp['factory']()
    df['N'] = temp_graph.number_of_nodes()
    
    all_data.append(df)

# 3. COMBINE & SAVE
master_df = pd.concat(all_data, ignore_index=True)
master_df.to_csv("master_simulation_results.csv", index=False)
print("Saved master_simulation_results.csv")

# 4. INSTANT ANALYSIS (Optional Check)
# Calculate Relative Fixation Prob (rho = P_fix * N)
stats = master_df.groupby(['Graph_Type', 'r', 'N']).agg(
    P_fix=('fixation', 'mean')
).reset_index()
stats['rho'] = stats['P_fix'] * stats['N']

print("\nRelative Fixation Probabilities (rho):")
print(stats.pivot(index='Graph_Type', columns='r', values='rho'))
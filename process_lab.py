import pandas as pd
import time
import os 
from datetime import datetime
from population_graph import PopulationGraph
from process_run import ProcessRun

class ProcessLab:
    """ Manages multiple process runs and stores their results"""
    def __init__(self, output_dir="simulation_data"):
        """
        :param output_dir: Directory to save simulation data.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.results_buffer = []

    def run_experiment(self, graph_factory_func, r_values, n_repeats=100, max_steps=1_000_000):
        """
        Runs a batch of simulations across different selection coefficients (r).
        
        graph_type: String identifier (e.g., "complete", "star", "respiratory")
        n_nodes: Size of the graph
        r_values: List of fitness values to test (e.g., [1.0, 1.1, 1.2])
        replicates: How many times to repeat the simulation for each r
        """
        
        print(f"--- Starting Experiment ---")        
        graph = graph_factory_func()
        for r in r_values:
            print(f' Running simulations for r={r} ({n_repeats} repeats)...')
            for i in range(n_repeats):
                sim = ProcessRun(graph, selection_coefficient=r)
                sim.initialize_random_mutant()
                result = sim.run()

                # Log Data
                record = {
                    "r": r,
                    "replicate_id": i,
                    "fixation": result["fixation"],
                    "steps": result["steps"],
                    "mutant_count": result["mutant_count"]
                }
                self.results_buffer.append(record)

            print("  > Batch finished.")
    
    def save_results(self, filename=None):
            """
            Flushes the buffer to a CSV file.
            """
            if not self.results_buffer:
                print("No results to save.")
                return

            df = pd.DataFrame(self.results_buffer)
            
            if filename is None:
                # Generate unique timestamped filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"results_{timestamp}.csv"
            
            full_path = os.path.join(self.output_dir, filename)
            df.to_csv(full_path, index=False)
            print(f"Saved {len(df)} rows to {full_path}")
            
            # Clear buffer to free memory
            self.results_buffer = []

# --- TEST BLOCK ---
if __name__ == "__main__":
    # Initialize Lab
    lab = ProcessLab()
    
    # Define Experiment Parameters
    selection_values = [1.0, 1.1, 1.5] # Neutral, Slight Advantage, Strong Advantage
    
    # Run Batch
    lab.run_experiment(
        graph_factory_func = (lambda: PopulationGraph.avian_graph(n_rods=5, rod_length=10)), 
        r_values=selection_values, 
        n_repeats=50
    )
    
    # Save Data
    lab.save_results()
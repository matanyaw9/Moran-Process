import networkx as nx
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt


def generate_refined_lung(generations, D, fitness_mutant):
    G = nx.Graph()  # Using undirected graph for bi-directional replacement
    G.add_node(0, fitness=1.0, type='WT')
    
    # Scaling factor r based on Fractal Dimension D: r = (1/N)^(1/D)
    # For a binary tree, N=2
    r = (0.5)**(1/D)
    
    nodes_at_gen = {0: [0]}
    next_id = 1
    
    for g in range(generations):
        nodes_at_gen[g+1] = []
        for parent in nodes_at_gen[g]:
            for _ in range(2): # Binary branching
                G.add_node(next_id, fitness=1.0, type='WT')
                # Edge weight represents the 'link strength' or radius
                # which influences the probability of replacement
                weight = r**(g+1) 
                G.add_edge(parent, next_id, weight=weight)
                nodes_at_gen[g+1].append(next_id)
                next_id += 1
    return G

def run_moran(G, fitness_mutant, trials=500):
    N = G.number_of_nodes()
    fixation_success = 0
    times = []

    for _ in range(trials):
        # State: 0 for WT, 1 for Mutant
        # Mutation starts at a random leaf (distal bronchiole/alveolus)
        leaves = [n for n, d in G.degree() if d == 1 and n != 0]
        state = {node: 0 for node in G.nodes()}
        start_node = random.choice(leaves)
        state[start_node] = 1
        
        steps = 0
        max_steps = N * N # Safety cutoff
        
        while 0 < sum(state.values()) < N and steps < max_steps:
            steps += 1
            # 1. Select for Birth (proportional to fitness)
            nodes = list(G.nodes())
            fitnesses = [fitness_mutant if state[n] == 1 else 1.0 for n in nodes]
            parent = random.choices(nodes, weights=fitnesses)[0]
            
            # 2. Select for Death (weighted by edge strength/radius)
            neighbors = list(G.neighbors(parent))
            weights = [G[parent][nb]['weight'] for nb in neighbors]
            child = random.choices(neighbors, weights=weights)[0]
            
            # Update state
            state[child] = state[parent]
            
        if sum(state.values()) == N:
            fixation_success += 1
            times.append(steps)
            
    return fixation_success / trials, np.mean(times) if times else 0

# --- Experiment Execution ---
fitness_r = 1.1 # Mutant is 10% more fit (e.g., a pre-cancerous cell)
results = []

for D in [1.5, 2.1, 2.7, 3.0]:
    lung = generate_refined_lung(generations=5, D=D, fitness_mutant=fitness_r)
    prob, time = run_moran(lung, fitness_mutant=fitness_r, trials=1000)
    results.append({"Dimension": D, "P_Fixation": prob, "Avg_Time": time})
    # nx.draw(lung)
    # plt.show()

print(pd.DataFrame(results))

# This script will generate a randome sample of graphs and will run a genetic algorithm to get the
#  most extreme graphs in terms of time to fixation. 


import numpy as np
import joblib
import pandas as pd
from population_graph import PopulationGraph


def add_new_random_graph(graph_zoo: list[PopulationGraph], 
                         wl_set:set, 
                         n_nodes:int, 
                         n_edges:int, 
                         name:str, 
                         seed=None):
    
    new_graph, new_wl = None, None
    while(new_wl is None or new_wl in wl_set):
        new_graph = PopulationGraph.random_connected_graph(n_nodes, n_edges, name=name, seed=seed)
        new_wl = new_graph.wl_hash
    graph_zoo.append(new_graph)
    wl_set.add(new_wl)
    return wl_set



def next_generation(graph_zoo:list[PopulationGraph], size:int, rng:np.random.Generator=None):
    new_generation = []
    wl_set = set([graph.wl_hash for graph in graph_zoo])
    for graph in graph_zoo:
        for i in range(size):
            new_graph, new_wl = None, None
            while(new_wl is None or new_wl in wl_set):
                seed = rng.integers(0, 2**32) if rng else None
                new_graph = graph.mutate_graph(seed=seed)
                new_wl = new_graph.wl_hash
            new_generation.append(new_graph)
            wl_set.add(new_wl)
    return new_generation



def main():

    # Create some random graphs
    seed = 42
    rng = np.random.default_rng(seed)

    INITIAL_GRAPH_POPULATION = 5
    NUMBER_OF_CHILDREN = 5
    N_NODES = 31
    N_EDGES = 34
    graph_zoo:list[PopulationGraph] = []
    wl_set = set()
    
    for i in range(INITIAL_GRAPH_POPULATION):
        add_new_random_graph(graph_zoo, wl_set, N_NODES, N_EDGES, name=f"random_{i}", seed=int(rng.integers(0, 2**32)))

    
    new_generation = next_generation(graph_zoo, size=NUMBER_OF_CHILDREN, rng=rng)
    
    lr = load_model()
    all_graphs = graph_zoo + new_generation
    graph_properties = []
    for graph in all_graphs:
        props = graph.calculate_graph_properties()
        graph_properties.append(props)
    df_properties = pd.DataFrame(graph_properties)
    print(df_properties)




    # predictions = [lr.predict([[graph.wl_hash]])[0] for graph in all_graphs]
    # df = pd.DataFrame({
    #     'graph_name': [graph.name for graph in all_graphs],
    #     'wl_hash': [graph.wl_hash for graph in all_graphs],
    #     'prediction_score': predictions
    # })


    # print(df)






def load_model(): 
    lr = joblib.load('./ml_models/linear_regression_pipeline.joblib')
    return lr
    
    
def test_random_seed():
    seed = 42
    G1 = PopulationGraph.random_connected_graph(30, 29, name=f"random_1_seed_{seed}", seed=seed, labeled_edges=False)
    G2 = PopulationGraph.random_connected_graph(30, 29, name=f"random_2_seed_{seed}", seed=seed, labeled_edges=False)
    G1.draw(filename=f"./tmp_images/random_check/{G1.name}.png", with_labels = True)
    G2.draw(filename=f"./tmp_images/random_check/{G2.name}.png", with_labels = True) 
    assert G1.wl_hash == G2.wl_hash, "WL hash should be the same if they are created with the same seed!"

    # now let's mutate
    G1_mutate = G1.mutate_graph(seed=seed)
    G2_mutate = G2.mutate_graph(seed=seed)
    G1_mutate.draw(filename=f"./tmp_images/random_check/{G1_mutate.name}.png", with_labels = True)
    G2_mutate.draw(filename=f"./tmp_images/random_check/{G2_mutate.name}.png", with_labels = True)
    assert G1_mutate.wl_hash == G2_mutate.wl_hash, "WL hash should be the same if they are mutated from the same seed graph!"


if __name__ == "__main__":

    main()
    # test_random_seed()
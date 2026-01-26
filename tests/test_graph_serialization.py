import pytest
import os
import pickle
import networkx as nx
# import sys
# from pathlib import Path

# # Add parent directory to path to import population_graph
# sys.path.insert(0, str(Path(__file__).parent.parent))
from population_graph import PopulationGraph

def test_save_load_roundtrip(tmp_path):
    """
    Verifies that a graph can be saved to disk and loaded back 
    with all its properties and topology intact.
    """
    # 1. Setup: Create a complex graph (Avian)
    original_graph = PopulationGraph.avian_graph(n_rods=3, rod_length=5)
    
    # Define a temporary path using pytest's tmp_path fixture
    save_file = tmp_path / "test_avian.pkl"
    
    # 2. Action: Save
    original_graph.save(str(save_file))
    
    # Check file exists
    assert os.path.exists(save_file)
    assert os.path.getsize(save_file) > 0

    # 3. Action: Load
    loaded_graph = PopulationGraph.load(str(save_file))

    # 4. Assertions: Verify Integrity
    assert isinstance(loaded_graph, PopulationGraph)
    
    # check metadata
    assert loaded_graph.name == original_graph.name
    assert loaded_graph.category == "Avian"
    assert loaded_graph.params == original_graph.params
    assert loaded_graph.wl_hash == original_graph.wl_hash
    
    # check topology (structure)
    assert loaded_graph.graph.number_of_nodes() == original_graph.graph.number_of_nodes()
    assert loaded_graph.graph.number_of_edges() == original_graph.graph.number_of_edges()
    
    # Strict check: Are they isomorphic? (Structure is identical)
    # Note: convert_node_labels_to_integers in your generator ensures consistent IDs
    assert nx.is_isomorphic(original_graph.graph, loaded_graph.graph)

def test_load_non_existent_file():
    """Ensure proper error is raised for missing files."""
    with pytest.raises(FileNotFoundError):
        PopulationGraph.load("non_existent_ghost_file.pkl")

def test_load_invalid_object(tmp_path):
    """
    Ensure we catch cases where the file exists but contains 
    something that isn't a PopulationGraph (e.g., a dictionary).
    """
    # Create a pickle file containing just a Python dictionary
    bad_file = tmp_path / "bad_data.pkl"
    with open(bad_file, "wb") as f:
        pickle.dump({"i_am": "not_a_graph"}, f)
        
    # Expect a TypeError because of the isinstance check in your load method
    with pytest.raises(TypeError, match="not a PopulationGraph"):
        PopulationGraph.load(str(bad_file))
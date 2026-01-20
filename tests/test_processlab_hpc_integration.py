#!/usr/bin/env python
"""
Integration test for ProcessLab HPC functionality.
Tests that ProcessLab can serialize and deserialize graphs correctly.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from population_graph import PopulationGraph
from process_lab import ProcessLab
from hpc.serialization import SerializationError


def test_processlab_serialization_workflow():
    """Test complete ProcessLab serialization workflow."""
    print("Testing ProcessLab HPC serialization workflow...")
    
    # Create ProcessLab instance
    lab = ProcessLab()
    
    # Create a realistic graph zoo for testing
    graphs = [
        PopulationGraph.complete_graph(8, register_in_db=False),
        PopulationGraph.cycle_graph(10, register_in_db=False),
        PopulationGraph.mammalian_lung_graph(branching_factor=2, depth=3, register_in_db=False),
        PopulationGraph.avian_graph(n_rods=3, rod_length=5, register_in_db=False),
        PopulationGraph.fish_graph(n_rods=2, rod_length=4, register_in_db=False),
        PopulationGraph.random_connected_graph(12, 18, seed=123, register_in_db=False)
    ]
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test serialization using ProcessLab method
        serialize_path = os.path.join(temp_dir, "processlab_graphs.pkl")
        
        try:
            print(f"Serializing {len(graphs)} graphs...")
            metadata = lab._serialize_graphs(graphs, serialize_path)
            
            # Verify metadata
            assert metadata['n_graphs'] == len(graphs), "Metadata graph count mismatch"
            assert len(metadata['graph_names']) == len(graphs), "Metadata names mismatch"
            assert 'checksum' in metadata, "Missing checksum in metadata"
            assert 'serialization_time' in metadata, "Missing timestamp in metadata"
            
            print("✓ Serialization completed with valid metadata")
            
            # Test deserialization using ProcessLab method
            print("Deserializing graphs...")
            loaded_graphs, loaded_metadata = lab._deserialize_graphs(serialize_path)
            
            # Verify loaded data
            assert len(loaded_graphs) == len(graphs), "Loaded graph count mismatch"
            assert loaded_metadata['n_graphs'] == len(graphs), "Loaded metadata mismatch"
            
            # Verify graph integrity
            for i, (original, loaded) in enumerate(zip(graphs, loaded_graphs)):
                assert original.name == loaded.name, f"Graph {i} name mismatch"
                assert original.wl_hash == loaded.wl_hash, f"Graph {i} WL hash mismatch"
                assert original.N == loaded.N, f"Graph {i} node count mismatch"
                assert original.category == loaded.category, f"Graph {i} category mismatch"
            
            print("✓ Deserialization completed with data integrity preserved")
            
            # Test that graphs can still be used for simulations
            print("Testing graph functionality after deserialization...")
            
            # Pick a simple graph and verify it works
            test_graph = loaded_graphs[0]  # Complete graph
            assert test_graph.number_of_nodes() > 0, "Graph has no nodes"
            assert len(test_graph.get_nodes()) == test_graph.N, "Node list mismatch"
            
            # Test adjacency matrix generation
            adj_matrix = test_graph.to_adjacency_matrix()
            assert adj_matrix.shape == (test_graph.N, test_graph.N), "Adjacency matrix shape mismatch"
            
            print("✓ Deserialized graphs maintain full functionality")
            
            return True
            
        except Exception as e:
            print(f"✗ ProcessLab serialization workflow failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_processlab_error_handling():
    """Test ProcessLab error handling for serialization failures."""
    print("\nTesting ProcessLab error handling...")
    
    lab = ProcessLab()
    
    # Test serialization with invalid inputs
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            invalid_path = os.path.join(temp_dir, "invalid_test.pkl")
            lab._serialize_graphs([], invalid_path)  # Empty list should fail
        print("✗ Should have failed with empty graph list")
        return False
    except SerializationError:
        print("✓ Correctly handled empty graph list in ProcessLab")
    except Exception as e:
        print(f"✗ Unexpected error type: {e}")
        return False
    
    # Test deserialization with non-existent file
    try:
        lab._deserialize_graphs("/nonexistent/path/file.pkl")
        print("✗ Should have failed with non-existent file")
        return False
    except SerializationError:
        print("✓ Correctly handled non-existent file in ProcessLab")
    except Exception as e:
        print(f"✗ Unexpected error type: {e}")
        return False
    
    return True


def test_file_path_handling():
    """Test that ProcessLab handles various file path formats correctly."""
    print("\nTesting file path handling...")
    
    lab = ProcessLab()
    graphs = [PopulationGraph.complete_graph(4, register_in_db=False)]
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test absolute path
        abs_path = os.path.join(temp_dir, "absolute_test.pkl")
        
        try:
            metadata = lab._serialize_graphs(graphs, abs_path)
            loaded_graphs, _ = lab._deserialize_graphs(abs_path)
            assert len(loaded_graphs) == 1, "Absolute path test failed"
            print("✓ Absolute path handling works")
        except Exception as e:
            print(f"✗ Absolute path test failed: {e}")
            return False
        
        # Test nested directory creation
        nested_path = os.path.join(temp_dir, "nested", "dir", "test.pkl")
        
        try:
            metadata = lab._serialize_graphs(graphs, nested_path)
            assert os.path.exists(nested_path), "Nested directory not created"
            loaded_graphs, _ = lab._deserialize_graphs(nested_path)
            assert len(loaded_graphs) == 1, "Nested path test failed"
            print("✓ Nested directory creation works")
        except Exception as e:
            print(f"✗ Nested path test failed: {e}")
            return False
    
    return True


def main():
    """Run all ProcessLab HPC integration tests."""
    print("=== ProcessLab HPC Integration Tests ===")
    
    tests = [
        test_processlab_serialization_workflow,
        test_processlab_error_handling,
        test_file_path_handling
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"✗ {test_func.__name__} failed")
        except Exception as e:
            print(f"✗ {test_func.__name__} crashed: {e}")
    
    print(f"\n=== Integration Test Results: {passed}/{total} passed ===")
    
    if passed == total:
        print("✓ All ProcessLab HPC integration tests passed!")
        return True
    else:
        print("✗ Some integration tests failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
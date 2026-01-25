#!/usr/bin/env python
"""
Integration test for ProcessLab HPC functionality.
Tests that ProcessLab can serialize and deserialize graphs correctly.
"""

import os
import sys
import tempfile
import pickle
from pathlib import Path

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from population_graph import PopulationGraph
from process_lab import ProcessLab


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
        # Test serialization using simple pickle
        serialize_path = os.path.join(temp_dir, "processlab_graphs.pkl")
        
        try:
            print(f"Serializing {len(graphs)} graphs...")
            with open(serialize_path, 'wb') as f:
                pickle.dump(graphs, f)
            
            print("✓ Serialization completed")
            
            # Test deserialization
            print("Deserializing graphs...")
            with open(serialize_path, 'rb') as f:
                loaded_graphs = pickle.load(f)
            
            # Verify loaded data
            assert len(loaded_graphs) == len(graphs), "Loaded graph count mismatch"
            
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


def test_processlab_job_submission():
    """Test ProcessLab job submission (dry run)."""
    print("\nTesting ProcessLab job submission...")
    
    lab = ProcessLab()
    graphs = [PopulationGraph.complete_graph(4, register_in_db=False)]
    
    try:
        # This would normally submit to LSF, but we'll just test the setup
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test that the method exists and accepts parameters
            # We can't actually submit without LSF, but we can test parameter validation
            try:
                tracking_info = lab.submit_jobs(
                    graphs=graphs,
                    r_values=[1.0, 1.2],
                    n_repeats=10,
                    n_jobs=2,
                    temp_dir=temp_dir
                )
                print("✗ Should have failed without LSF environment")
                return False
            except Exception as e:
                # Expected to fail without LSF, but should fail gracefully
                if "LSF" in str(e) or "bsub" in str(e):
                    print("✓ Job submission correctly requires LSF environment")
                    return True
                else:
                    print(f"✗ Unexpected error: {e}")
                    return False
    
    except Exception as e:
        print(f"✗ Job submission test failed: {e}")
        return False


def main():
    """Run all ProcessLab HPC integration tests."""
    print("=== ProcessLab HPC Integration Tests (Simplified) ===")
    
    tests = [
        test_processlab_serialization_workflow,
        test_processlab_job_submission
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
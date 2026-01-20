#!/usr/bin/env python
"""
Property-based tests for graph serialization round-trip consistency.
Tests that serialization and deserialization preserve all graph properties.
"""

import os
import sys
import tempfile
import random
from pathlib import Path

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from population_graph import PopulationGraph
from hpc.serialization import GraphSerializer, SerializationError


def generate_test_graphs(n_graphs=10):
    """Generate a diverse set of test graphs for property testing."""
    graphs = []
    
    # Complete graphs of various sizes
    for n in [3, 5, 8, 10]:
        graphs.append(PopulationGraph.complete_graph(n, register_in_db=False))
    
    # Cycle graphs
    for n in [4, 6, 9]:
        graphs.append(PopulationGraph.cycle_graph(n, register_in_db=False))
    
    # Mammalian lung graphs with different parameters
    for branching, depth in [(2, 2), (3, 2), (2, 3)]:
        graphs.append(PopulationGraph.mammalian_lung_graph(
            branching_factor=branching, depth=depth, register_in_db=False
        ))
    
    # Avian graphs with different parameters
    for n_rods, rod_length in [(2, 3), (3, 4), (4, 2)]:
        graphs.append(PopulationGraph.avian_graph(
            n_rods=n_rods, rod_length=rod_length, register_in_db=False
        ))
    
    # Fish graphs
    for n_rods, rod_length in [(2, 3), (3, 2)]:
        graphs.append(PopulationGraph.fish_graph(
            n_rods=n_rods, rod_length=rod_length, register_in_db=False
        ))
    
    # Random connected graphs with seeds for reproducibility
    for i in range(5):
        n_nodes = random.randint(5, 15)
        n_edges = random.randint(n_nodes - 1, min(n_nodes * (n_nodes - 1) // 2, n_nodes + 10))
        graphs.append(PopulationGraph.random_connected_graph(
            n_nodes=n_nodes, n_edges=n_edges, seed=42 + i, register_in_db=False
        ))
    
    return graphs[:n_graphs]


def test_round_trip_consistency():
    """
    Property Test 1: Graph Serialization Round-trip Consistency
    
    For any collection of PopulationGraph objects, serializing then deserializing 
    should produce equivalent objects with identical graph structure, metadata, 
    and calculated properties.
    """
    print("Testing Property 1: Graph Serialization Round-trip Consistency")
    
    # Generate diverse test graphs
    test_graphs = generate_test_graphs(15)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, "round_trip_test.pkl")
        
        try:
            # Serialize graphs
            metadata = GraphSerializer.serialize_graphs(test_graphs, test_file)
            
            # Deserialize graphs
            loaded_graphs, loaded_metadata = GraphSerializer.deserialize_graphs(test_file)
            
            # Verify count consistency
            assert len(loaded_graphs) == len(test_graphs), \
                f"Graph count mismatch: {len(loaded_graphs)} != {len(test_graphs)}"
            
            # Verify each graph's properties are preserved
            for i, (original, loaded) in enumerate(zip(test_graphs, loaded_graphs)):
                # Basic properties
                assert original.name == loaded.name, \
                    f"Graph {i}: Name mismatch {original.name} != {loaded.name}"
                
                assert original.category == loaded.category, \
                    f"Graph {i}: Category mismatch {original.category} != {loaded.category}"
                
                assert original.N == loaded.N, \
                    f"Graph {i}: Node count mismatch {original.N} != {loaded.N}"
                
                assert original.wl_hash == loaded.wl_hash, \
                    f"Graph {i}: WL hash mismatch {original.wl_hash} != {loaded.wl_hash}"
                
                # Graph structure
                assert original.graph.number_of_nodes() == loaded.graph.number_of_nodes(), \
                    f"Graph {i}: NetworkX node count mismatch"
                
                assert original.graph.number_of_edges() == loaded.graph.number_of_edges(), \
                    f"Graph {i}: NetworkX edge count mismatch"
                
                # Parameters (if any)
                assert original.params == loaded.params, \
                    f"Graph {i}: Parameters mismatch {original.params} != {loaded.params}"
                
                # Metadata
                assert original.metadata == loaded.metadata, \
                    f"Graph {i}: Metadata mismatch {original.metadata} != {loaded.metadata}"
            
            # Verify metadata consistency
            assert metadata['n_graphs'] == len(test_graphs), \
                "Metadata graph count mismatch"
            
            assert len(metadata['graph_names']) == len(test_graphs), \
                "Metadata names list length mismatch"
            
            assert len(metadata['wl_hashes']) == len(test_graphs), \
                "Metadata hashes list length mismatch"
            
            print(f"✓ Round-trip consistency verified for {len(test_graphs)} diverse graphs")
            return True
            
        except Exception as e:
            print(f"✗ Round-trip consistency test failed: {e}")
            return False


def test_serialization_with_edge_cases():
    """Test serialization with edge cases and boundary conditions."""
    print("\nTesting serialization edge cases...")
    
    edge_case_graphs = [
        # Minimal graphs
        PopulationGraph.complete_graph(1, register_in_db=False),  # Single node
        PopulationGraph.complete_graph(2, register_in_db=False),  # Two nodes
        PopulationGraph.cycle_graph(3, register_in_db=False),     # Minimal cycle
        
        # Larger graphs
        PopulationGraph.complete_graph(20, register_in_db=False), # Larger complete
        PopulationGraph.mammalian_lung_graph(branching_factor=4, depth=4, register_in_db=False),
        
        # Graphs with special parameters
        PopulationGraph.random_connected_graph(10, 15, seed=999, register_in_db=False)
    ]
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, "edge_cases_test.pkl")
        
        try:
            # Test serialization and deserialization
            metadata = GraphSerializer.serialize_graphs(edge_case_graphs, test_file)
            loaded_graphs, loaded_metadata = GraphSerializer.deserialize_graphs(test_file)
            
            # Verify all edge cases preserved correctly
            for original, loaded in zip(edge_case_graphs, loaded_graphs):
                assert original.wl_hash == loaded.wl_hash, \
                    f"Edge case failed: {original.name} WL hash mismatch"
            
            print(f"✓ Edge cases handled correctly for {len(edge_case_graphs)} graphs")
            return True
            
        except Exception as e:
            print(f"✗ Edge cases test failed: {e}")
            return False


def test_multiple_serialization_cycles():
    """Test that multiple serialization cycles don't degrade data."""
    print("\nTesting multiple serialization cycles...")
    
    # Start with a few test graphs
    original_graphs = [
        PopulationGraph.complete_graph(6, register_in_db=False),
        PopulationGraph.avian_graph(n_rods=3, rod_length=4, register_in_db=False)
    ]
    
    current_graphs = original_graphs
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Perform multiple serialization/deserialization cycles
            for cycle in range(5):
                test_file = os.path.join(temp_dir, f"cycle_{cycle}.pkl")
                
                # Serialize current graphs
                GraphSerializer.serialize_graphs(current_graphs, test_file)
                
                # Deserialize to get new graphs
                current_graphs, _ = GraphSerializer.deserialize_graphs(test_file)
            
            # Verify final graphs match original
            assert len(current_graphs) == len(original_graphs), \
                "Graph count changed during cycles"
            
            for original, final in zip(original_graphs, current_graphs):
                assert original.wl_hash == final.wl_hash, \
                    f"Graph degraded during cycles: {original.name}"
                assert original.N == final.N, \
                    f"Node count changed during cycles: {original.name}"
            
            print("✓ Multiple serialization cycles preserve data integrity")
            return True
            
        except Exception as e:
            print(f"✗ Multiple cycles test failed: {e}")
            return False


def main():
    """Run all property-based serialization tests."""
    print("=== Property-Based Serialization Tests ===")
    
    # Set random seed for reproducible tests
    random.seed(42)
    
    tests = [
        test_round_trip_consistency,
        test_serialization_with_edge_cases,
        test_multiple_serialization_cycles
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
    
    print(f"\n=== Property Test Results: {passed}/{total} passed ===")
    
    if passed == total:
        print("✓ All property-based serialization tests passed!")
        return True
    else:
        print("✗ Some property tests failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
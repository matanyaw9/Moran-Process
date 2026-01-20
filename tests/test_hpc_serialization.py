#!/usr/bin/env python
"""
Test script for HPC graph serialization functionality.
Tests the core serialization and deserialization methods.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from population_graph import PopulationGraph
from process_lab import ProcessLab
from hpc.serialization import GraphSerializer, SerializationError


def test_basic_serialization():
    """Test basic serialization and deserialization functionality."""
    print("Testing basic graph serialization...")
    
    # Create test graphs
    graphs = [
        PopulationGraph.complete_graph(5, register_in_db=False),
        PopulationGraph.cycle_graph(6, register_in_db=False),
        PopulationGraph.mammalian_lung_graph(branching_factor=2, depth=2, register_in_db=False)
    ]
    
    # Create temporary directory for test
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, "test_graphs.pkl")
        
        # Test serialization
        try:
            metadata = GraphSerializer.serialize_graphs(graphs, test_file)
            print(f"✓ Serialization successful: {metadata['n_graphs']} graphs")
            
            # Verify file exists and has content
            assert os.path.exists(test_file), "Serialized file not created"
            assert os.path.getsize(test_file) > 0, "Serialized file is empty"
            
        except Exception as e:
            print(f"✗ Serialization failed: {e}")
            return False
        
        # Test deserialization
        try:
            loaded_graphs, loaded_metadata = GraphSerializer.deserialize_graphs(test_file)
            print(f"✓ Deserialization successful: {len(loaded_graphs)} graphs loaded")
            
            # Verify graph count matches
            assert len(loaded_graphs) == len(graphs), "Graph count mismatch"
            
            # Verify graph properties are preserved
            for orig, loaded in zip(graphs, loaded_graphs):
                assert orig.name == loaded.name, f"Name mismatch: {orig.name} != {loaded.name}"
                assert orig.N == loaded.N, f"Node count mismatch: {orig.N} != {loaded.N}"
                assert orig.wl_hash == loaded.wl_hash, f"WL hash mismatch: {orig.wl_hash} != {loaded.wl_hash}"
            
            print("✓ Graph properties preserved correctly")
            
        except Exception as e:
            print(f"✗ Deserialization failed: {e}")
            return False
    
    return True


def test_processlab_integration():
    """Test ProcessLab integration with serialization methods."""
    print("\nTesting ProcessLab serialization integration...")
    
    # Create ProcessLab instance
    lab = ProcessLab()
    
    # Create test graphs
    graphs = [
        PopulationGraph.complete_graph(4, register_in_db=False),
        PopulationGraph.cycle_graph(5, register_in_db=False)
    ]
    
    # Create temporary directory for test
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, "processlab_test.pkl")
        
        try:
            # Test ProcessLab serialization method
            metadata = lab._serialize_graphs(graphs, test_file)
            print(f"✓ ProcessLab serialization successful")
            
            # Test ProcessLab deserialization method
            loaded_graphs, loaded_metadata = lab._deserialize_graphs(test_file)
            print(f"✓ ProcessLab deserialization successful")
            
            # Verify results
            assert len(loaded_graphs) == len(graphs), "Graph count mismatch in ProcessLab methods"
            
        except Exception as e:
            print(f"✗ ProcessLab integration failed: {e}")
            return False
    
    return True


def test_error_handling():
    """Test error handling for various failure scenarios."""
    print("\nTesting error handling...")
    
    # Test empty graph list
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, "empty_test.pkl")
            GraphSerializer.serialize_graphs([], test_file)
        print("✗ Should have failed with empty graph list")
        return False
    except SerializationError:
        print("✓ Correctly handled empty graph list")
    
    # Test non-existent file deserialization
    try:
        GraphSerializer.deserialize_graphs("/nonexistent/path/file.pkl")
        print("✗ Should have failed with non-existent file")
        return False
    except SerializationError:
        print("✓ Correctly handled non-existent file")
    
    # Test invalid graph objects
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, "invalid_test.pkl")
            GraphSerializer.serialize_graphs(["not_a_graph", 123], test_file)
        print("✗ Should have failed with invalid graph objects")
        return False
    except SerializationError:
        print("✓ Correctly handled invalid graph objects")
    
    return True


def main():
    """Run all serialization tests."""
    print("=== HPC Graph Serialization Tests ===")
    
    tests = [
        test_basic_serialization,
        test_processlab_integration,
        test_error_handling
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
    
    print(f"\n=== Test Results: {passed}/{total} passed ===")
    
    if passed == total:
        print("✓ All serialization tests passed!")
        return True
    else:
        print("✗ Some tests failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
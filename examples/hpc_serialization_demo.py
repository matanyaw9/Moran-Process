#!/usr/bin/env python
"""
Demonstration of HPC graph serialization functionality.

This script shows how to use the new ProcessLab HPC serialization methods
to save and load graph collections for distributed computing.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from population_graph import PopulationGraph
from process_lab import ProcessLab


def main():
    """Demonstrate HPC serialization functionality."""
    print("=== HPC Graph Serialization Demo ===")
    
    # Create ProcessLab instance
    lab = ProcessLab()
    
    # Create a diverse collection of graphs for HPC execution
    print("\n1. Creating graph collection...")
    graphs = [
        PopulationGraph.complete_graph(10, register_in_db=False),
        PopulationGraph.cycle_graph(12, register_in_db=False),
        PopulationGraph.mammalian_lung_graph(branching_factor=3, depth=3, register_in_db=False),
        PopulationGraph.avian_graph(n_rods=4, rod_length=5, register_in_db=False),
        PopulationGraph.fish_graph(n_rods=3, rod_length=4, register_in_db=False),
        PopulationGraph.random_connected_graph(15, 25, seed=42, register_in_db=False)
    ]
    
    print(f"Created {len(graphs)} graphs:")
    for i, graph in enumerate(graphs):
        print(f"  {i+1}. {graph.name} ({graph.category}) - {graph.N} nodes")
    
    # Serialize graphs for HPC execution
    print("\n2. Serializing graphs for HPC execution...")
    output_dir = "hpc_temp"
    os.makedirs(output_dir, exist_ok=True)
    serialize_path = os.path.join(output_dir, "demo_graphs.pkl")
    
    try:
        metadata = lab._serialize_graphs(graphs, serialize_path)
        
        print(f"✓ Serialization successful!")
        print(f"  File: {serialize_path}")
        print(f"  Size: {Path(serialize_path).stat().st_size} bytes")
        print(f"  Graphs: {metadata['n_graphs']}")
        print(f"  Checksum: {metadata['checksum']}")
        print(f"  Timestamp: {metadata['serialization_time']}")
        
    except Exception as e:
        print(f"✗ Serialization failed: {e}")
        return False
    
    # Simulate loading graphs on a compute node
    print("\n3. Simulating compute node graph loading...")
    
    try:
        loaded_graphs, loaded_metadata = lab._deserialize_graphs(serialize_path)
        
        print(f"✓ Deserialization successful!")
        print(f"  Loaded {len(loaded_graphs)} graphs")
        print(f"  Data integrity verified (checksum match)")
        
        # Verify graphs are identical
        print("\n4. Verifying graph integrity...")
        for i, (original, loaded) in enumerate(zip(graphs, loaded_graphs)):
            if original.wl_hash != loaded.wl_hash:
                print(f"✗ Graph {i+1} integrity check failed!")
                return False
            print(f"  ✓ {loaded.name} - integrity verified")
        
        # Show that loaded graphs are fully functional
        print("\n5. Testing loaded graph functionality...")
        test_graph = loaded_graphs[0]  # Complete graph
        
        print(f"  Graph: {test_graph.name}")
        print(f"  Nodes: {test_graph.number_of_nodes()}")
        print(f"  Edges: {test_graph.graph.number_of_edges()}")
        print(f"  WL Hash: {test_graph.wl_hash}")
        
        # Test adjacency matrix
        adj_matrix = test_graph.to_adjacency_matrix()
        print(f"  Adjacency matrix shape: {adj_matrix.shape}")
        
        # Test neighbor access
        neighbors = test_graph.get_neighbors(0)
        print(f"  Node 0 neighbors: {len(neighbors)} nodes")
        
        print("  ✓ All graph operations work correctly")
        
    except Exception as e:
        print(f"✗ Deserialization failed: {e}")
        return False
    
    print("\n=== Demo completed successfully! ===")
    print(f"\nThe serialized graphs are ready for HPC execution.")
    print(f"Worker nodes can load them using:")
    print(f"  graphs, metadata = lab._deserialize_graphs('{serialize_path}')")
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
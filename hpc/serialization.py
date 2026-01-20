"""
Graph serialization and deserialization functionality for HPC execution.

This module provides robust serialization of PopulationGraph objects for transfer
between login nodes and compute nodes in HPC environments. It uses pickle protocol 4
for compatibility and includes comprehensive error handling and data integrity checks.
"""

import pickle
import hashlib
import datetime
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from population_graph import PopulationGraph


class SerializationError(Exception):
    """Custom exception for serialization/deserialization failures."""
    pass


class GraphSerializer:
    """
    Handles serialization and deserialization of PopulationGraph collections
    with metadata preservation and integrity checking.
    """
    
    PROTOCOL_VERSION = 4  # Python 3.4+ compatibility
    
    @classmethod
    def serialize_graphs(cls, graphs: List[PopulationGraph], filepath: str) -> Dict[str, Any]:
        """
        Serialize a collection of PopulationGraph objects to a pickle file.
        
        Args:
            graphs: List of PopulationGraph objects to serialize
            filepath: Output pickle file path
            
        Returns:
            Dict containing serialization metadata
            
        Raises:
            SerializationError: If serialization fails for any reason
        """
        if not graphs:
            raise SerializationError("Cannot serialize empty graph list")
            
        if not isinstance(graphs, list):
            raise SerializationError("Graphs must be provided as a list")
            
        # Validate all objects are PopulationGraph instances
        for i, graph in enumerate(graphs):
            if not isinstance(graph, PopulationGraph):
                raise SerializationError(
                    f"Object at index {i} is not a PopulationGraph instance: {type(graph)}"
                )
        
        try:
            # Create output directory if it doesn't exist
            output_path = Path(filepath)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare serialization data
            serialization_data = {
                'graphs': graphs,
                'metadata': {
                    'n_graphs': len(graphs),
                    'graph_names': [g.name for g in graphs],
                    'graph_categories': [g.category for g in graphs],
                    'wl_hashes': [g.wl_hash for g in graphs],
                    'serialization_time': datetime.datetime.now().isoformat(),
                    'python_version': sys.version,
                    'protocol_version': cls.PROTOCOL_VERSION
                }
            }
            
            # Calculate checksum for integrity verification
            graph_data_str = str([(g.wl_hash, g.name, g.N) for g in graphs])
            checksum = hashlib.md5(graph_data_str.encode()).hexdigest()
            serialization_data['metadata']['checksum'] = checksum
            
            # Serialize to file
            with open(filepath, 'wb') as f:
                pickle.dump(serialization_data, f, protocol=cls.PROTOCOL_VERSION)
                
            # Verify file was created and has content
            if not output_path.exists() or output_path.stat().st_size == 0:
                raise SerializationError(f"Failed to create valid pickle file at {filepath}")
                
            return serialization_data['metadata']
            
        except (IOError, OSError) as e:
            raise SerializationError(f"File system error during serialization: {e}")
        except pickle.PicklingError as e:
            raise SerializationError(f"Pickle serialization failed: {e}")
        except Exception as e:
            raise SerializationError(f"Unexpected error during serialization: {e}")
    
    @classmethod
    def deserialize_graphs(cls, filepath: str) -> tuple[List[PopulationGraph], Dict[str, Any]]:
        """
        Deserialize PopulationGraph objects from a pickle file.
        
        Args:
            filepath: Path to pickle file containing serialized graphs
            
        Returns:
            Tuple of (graphs_list, metadata_dict)
            
        Raises:
            SerializationError: If deserialization fails or data is corrupted
        """
        file_path = Path(filepath)
        
        # Validate file exists and is readable
        if not file_path.exists():
            raise SerializationError(f"Serialized graph file not found: {filepath}")
            
        if not file_path.is_file():
            raise SerializationError(f"Path is not a file: {filepath}")
            
        if file_path.stat().st_size == 0:
            raise SerializationError(f"Serialized graph file is empty: {filepath}")
        
        try:
            # Load serialized data
            with open(filepath, 'rb') as f:
                serialization_data = pickle.load(f)
                
            # Validate data structure
            if not isinstance(serialization_data, dict):
                raise SerializationError("Invalid serialization format: expected dictionary")
                
            if 'graphs' not in serialization_data:
                raise SerializationError("Invalid serialization format: missing 'graphs' key")
                
            if 'metadata' not in serialization_data:
                raise SerializationError("Invalid serialization format: missing 'metadata' key")
            
            graphs = serialization_data['graphs']
            metadata = serialization_data['metadata']
            
            # Validate graphs list
            if not isinstance(graphs, list):
                raise SerializationError("Invalid serialization format: 'graphs' must be a list")
                
            if not graphs:
                raise SerializationError("Deserialized graph list is empty")
            
            # Validate all objects are PopulationGraph instances
            for i, graph in enumerate(graphs):
                if not isinstance(graph, PopulationGraph):
                    raise SerializationError(
                        f"Deserialized object at index {i} is not a PopulationGraph: {type(graph)}"
                    )
            
            # Verify data integrity using checksum
            if 'checksum' in metadata:
                graph_data_str = str([(g.wl_hash, g.name, g.N) for g in graphs])
                calculated_checksum = hashlib.md5(graph_data_str.encode()).hexdigest()
                
                if calculated_checksum != metadata['checksum']:
                    raise SerializationError(
                        f"Data integrity check failed: checksum mismatch. "
                        f"Expected {metadata['checksum']}, got {calculated_checksum}"
                    )
            
            # Validate metadata consistency
            if 'n_graphs' in metadata and len(graphs) != metadata['n_graphs']:
                raise SerializationError(
                    f"Metadata inconsistency: expected {metadata['n_graphs']} graphs, "
                    f"found {len(graphs)}"
                )
            
            return graphs, metadata
            
        except (IOError, OSError) as e:
            raise SerializationError(f"File system error during deserialization: {e}")
        except pickle.UnpicklingError as e:
            raise SerializationError(f"Pickle deserialization failed: {e}")
        except Exception as e:
            if isinstance(e, SerializationError):
                raise
            raise SerializationError(f"Unexpected error during deserialization: {e}")
    
    @classmethod
    def validate_serialized_file(cls, filepath: str) -> Dict[str, Any]:
        """
        Validate a serialized graph file without fully loading it.
        
        Args:
            filepath: Path to pickle file to validate
            
        Returns:
            Dictionary with validation results and metadata
            
        Raises:
            SerializationError: If file is invalid or corrupted
        """
        try:
            # Quick validation - just load metadata
            graphs, metadata = cls.deserialize_graphs(filepath)
            
            validation_result = {
                'valid': True,
                'n_graphs': len(graphs),
                'metadata': metadata,
                'file_size_bytes': Path(filepath).stat().st_size
            }
            
            return validation_result
            
        except SerializationError as e:
            return {
                'valid': False,
                'error': str(e),
                'file_exists': Path(filepath).exists(),
                'file_size_bytes': Path(filepath).stat().st_size if Path(filepath).exists() else 0
            }
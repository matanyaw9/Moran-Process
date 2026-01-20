"""
HPC (High Performance Computing) module for ProcessLab distributed execution.

This module provides functionality for running ProcessLab simulations on HPC clusters
using LSF (Load Sharing Facility) job arrays. It includes:

- Graph serialization/deserialization for data transfer between nodes
- Job distribution and work assignment algorithms  
- LSF job submission and management
- Worker execution logic for compute nodes
- Result aggregation and file management

The module maintains backward compatibility with existing ProcessLab functionality
while adding new distributed computing capabilities.
"""

__version__ = "0.1.0"

# Import main classes for easy access
from .serialization import GraphSerializer, SerializationError
from .job_distribution import JobDistributor
from .worker import ProcessLabWorker

__all__ = [
    'GraphSerializer',
    'SerializationError',
    'JobDistributor', 
    'ProcessLabWorker'
]
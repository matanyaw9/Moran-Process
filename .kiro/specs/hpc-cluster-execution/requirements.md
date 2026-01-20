# Requirements Document

## Introduction

This document specifies the requirements for refactoring ProcessLab, an evolutionary simulation system, to support distributed High Performance Computing (HPC) execution using LSF (Load Sharing Facility). The current system runs simulations sequentially on a single machine, but needs to scale to thousands of parallel jobs on HPC clusters for large-scale comparative studies.

## Glossary

- **ProcessLab**: Main orchestrator class that manages comparative studies across multiple graphs and selection coefficients
- **ProcessRun**: The simulation engine that executes individual Moran process simulations on population graphs
- **PopulationGraph**: Represents different population structures (complete graphs, respiratory systems, etc.) with metadata
- **LSF**: Load Sharing Facility - the job scheduler used on HPC clusters
- **Job_Array**: LSF feature that allows submitting thousands of jobs with a single command
- **Submitter**: The component running on the login node that prepares data and submits jobs
- **Worker**: The component running on compute nodes that executes individual simulation batches
- **Graph_Zoo**: Collection of PopulationGraph objects to be used in comparative studies
- **Serialization**: Process of converting Python objects to files for transfer between nodes

## Requirements

### Requirement 1: Graph Serialization and Deserialization

**User Story:** As a researcher, I want to serialize graph collections to files, so that worker nodes can load and use the exact same graph objects.

#### Acceptance Criteria

1. WHEN a graph zoo is provided, THE Serializer SHALL save all PopulationGraph objects to a pickle file
2. WHEN worker nodes load the pickle file, THE Deserializer SHALL reconstruct identical PopulationGraph objects with all metadata preserved
3. WHEN serialization fails, THE System SHALL return a descriptive error message
4. THE Serializer SHALL preserve graph structure, metadata, and all calculated properties during round-trip serialization

### Requirement 2: Job Submission Management

**User Story:** As a researcher, I want to submit thousands of simulation jobs with a single command, so that I can efficiently utilize HPC cluster resources.

#### Acceptance Criteria

1. WHEN submit_jobs is called with parameters, THE Submitter SHALL calculate the total number of jobs needed based on graphs and r_values
2. WHEN submitting to LSF, THE Submitter SHALL use job arrays to minimize submission overhead
3. WHEN LSF commands are executed, THE Submitter SHALL use subprocess to run bsub commands with proper error handling
4. WHEN job submission fails, THE Submitter SHALL return detailed error information including LSF error codes
5. THE Submitter SHALL distribute multiple repeats per job to optimize job overhead versus parallelism

### Requirement 3: Worker Node Execution

**User Story:** As a compute node, I want to execute assigned simulation batches independently, so that simulations can run in parallel across the cluster.

#### Acceptance Criteria

1. WHEN a worker starts, THE Worker SHALL read the LSF job index from environment variable LSB_JOBINDEX
2. WHEN command-line arguments are provided, THE Worker SHALL parse r_values and repeat counts using argparse
3. WHEN calculating work assignment, THE Worker SHALL determine which graph and r_value combination to process based on job index
4. WHEN running simulations, THE Worker SHALL load the serialized graph zoo and execute the assigned simulation batch
5. WHEN simulations complete, THE Worker SHALL save results to a unique CSV file using the job ID to prevent conflicts
6. WHEN file writing fails, THE Worker SHALL handle errors gracefully and report the failure

### Requirement 4: Result Aggregation

**User Story:** As a researcher, I want to combine results from all worker jobs into a single dataset, so that I can analyze the complete study results.

#### Acceptance Criteria

1. WHEN aggregating results, THE Aggregator SHALL merge all individual CSV files into one master CSV file
2. WHEN CSV files are missing, THE Aggregator SHALL handle missing files gracefully and report which jobs failed
3. WHEN duplicate data is detected, THE Aggregator SHALL prevent data duplication in the final results
4. WHEN aggregation completes, THE Aggregator SHALL provide a summary of total simulations and any missing data
5. THE Aggregator SHALL preserve all metadata and result columns from individual worker outputs

### Requirement 5: Backward Compatibility

**User Story:** As an existing ProcessLab user, I want the local execution mode to continue working unchanged, so that I can run small studies without HPC infrastructure.

#### Acceptance Criteria

1. WHEN using existing ProcessLab methods, THE System SHALL maintain all current functionality without modification
2. WHEN run_comparative_study is called, THE System SHALL execute simulations locally as before
3. WHEN ProcessRun logic is accessed, THE System SHALL use the unchanged simulation engine
4. THE System SHALL not modify any existing method signatures or return values

### Requirement 6: File Management and Safety

**User Story:** As a system administrator, I want to prevent file conflicts and data loss, so that concurrent jobs don't interfere with each other.

#### Acceptance Criteria

1. WHEN workers write output files, THE System SHALL ensure each worker writes to a unique filename
2. WHEN temporary files are created, THE System SHALL use job-specific naming to prevent conflicts
3. WHEN file operations fail, THE System SHALL not overwrite existing data
4. THE System SHALL create necessary directories if they don't exist
5. WHEN cleanup is requested, THE System SHALL safely remove temporary files without affecting results

### Requirement 7: Error Handling and Monitoring

**User Story:** As a researcher, I want comprehensive error reporting and job monitoring, so that I can identify and resolve issues in large-scale studies.

#### Acceptance Criteria

1. WHEN LSF commands fail, THE System SHALL capture and report LSF error codes and messages
2. WHEN worker jobs fail, THE System SHALL log detailed error information including job ID and failure reason
3. WHEN serialization or deserialization fails, THE System SHALL provide specific error messages about the failure
4. WHEN jobs are submitted, THE System SHALL provide job tracking information for monitoring
5. THE System SHALL distinguish between recoverable errors and fatal failures

### Requirement 8: Configuration and Flexibility

**User Story:** As a researcher, I want to configure job parameters for different cluster environments, so that I can optimize performance for various HPC systems.

#### Acceptance Criteria

1. WHEN submitting jobs, THE System SHALL accept configurable LSF parameters including queue, memory, and walltime
2. WHEN distributing work, THE System SHALL allow configuration of repeats per job for optimal resource utilization
3. WHEN specifying output paths, THE System SHALL support both absolute and relative paths
4. THE System SHALL provide sensible defaults for all configuration parameters
5. WHEN invalid configurations are provided, THE System SHALL validate parameters and report specific errors
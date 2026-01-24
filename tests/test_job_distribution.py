#!/usr/bin/env python
"""
Test script for HPC job distribution functionality.
Tests job count calculation, work distribution, and work assignment algorithms.
"""

import os
import sys

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from hpc.job_distribution import JobDistributor


def test_job_count_calculation():
    """Test job count calculation with various scenarios."""
    print("Testing job count calculation...")
    
    distributor = JobDistributor()
    
    # Test basic calculation
    n_jobs, repeats_per_job = distributor.calculate_job_count(2, 3, 100)
    print(f"✓ Basic test: {n_jobs} jobs, {repeats_per_job} repeats per job")
    assert n_jobs > 0, "Number of jobs must be positive"
    assert repeats_per_job > 0, "Repeats per job must be positive"
    
    # Test single job scenario
    n_jobs, repeats_per_job = distributor.calculate_job_count(1, 1, 1)
    print(f"✓ Single job test: {n_jobs} jobs, {repeats_per_job} repeats per job")
    assert n_jobs == 1, "Single scenario should result in 1 job"
    assert repeats_per_job == 1, "Single scenario should have 1 repeat per job"
    
    # Test with specified repeats_per_job
    n_jobs, actual_repeats = distributor.calculate_job_count(2, 2, 50, repeats_per_job=10)
    print(f"✓ Specified repeats test: {n_jobs} jobs, {actual_repeats} repeats per job")
    assert n_jobs == 5, "Should have 5 jobs for 50 repeats with 10 per job"
    
    # Test edge cases
    try:
        distributor.calculate_job_count(0, 1, 1)
        assert False, "Should raise ValueError for zero graphs"
    except ValueError:
        print("✓ Correctly handled zero graphs")
    
    try:
        distributor.calculate_job_count(1, 0, 1)
        assert False, "Should raise ValueError for zero r_values"
    except ValueError:
        print("✓ Correctly handled zero r_values")
    
    try:
        distributor.calculate_job_count(1, 1, 0)
        assert False, "Should raise ValueError for zero repeats"
    except ValueError:
        print("✓ Correctly handled zero repeats")
    
    return True


def test_work_distribution():
    """Test work distribution algorithm."""
    print("Testing work distribution...")
    
    distributor = JobDistributor()
    
    # Test basic distribution
    result = distributor.distribute_work(2, 3, 100, 10)
    print(f"✓ Basic distribution: {result}")
    assert result['total_combinations'] == 6, "Should have 6 combinations (2*3)"
    assert result['repeats_per_job'] == 10, "Should have 10 repeats per job"
    assert result['total_work_units'] == 100, "Should have 100 total work units"
    
    # Test with uneven division
    result = distributor.distribute_work(3, 2, 50, 7)
    print(f"✓ Uneven division: {result}")
    assert result['total_combinations'] == 6, "Should have 6 combinations (3*2)"
    assert result['repeats_per_job'] == 8, "Should have 8 repeats per job (50/7 rounded up)"
    assert result['total_work_units'] == 56, "Should have 56 total work units (7*8)"
    assert result['excess_work'] == 6, "Should have 6 excess work units"
    
    # Test edge cases
    try:
        distributor.distribute_work(0, 1, 1, 1)
        assert False, "Should raise ValueError for zero graphs"
    except ValueError:
        print("✓ Correctly handled zero graphs in distribution")
    
    try:
        distributor.distribute_work(1, 1, 1, 0)
        assert False, "Should raise ValueError for zero jobs"
    except ValueError:
        print("✓ Correctly handled zero jobs in distribution")
    
    return True


def test_work_assignment():
    """Test work assignment calculation for specific jobs."""
    print("Testing work assignment calculation...")
    
    distributor = JobDistributor()
    
    # Test basic assignment with 2 graphs, 3 r_values, 10 repeats per job
    # Total combinations = 6, so jobs 1-6 should get different combinations
    
    # Job 1 should get graph 0, r_value 0, repeats 0-9
    graph_idx, r_idx, start, end = distributor.calculate_work_assignment(1, 2, 3, 10)
    print(f"✓ Job 1 assignment: graph={graph_idx}, r={r_idx}, repeats={start}-{end-1}")
    assert graph_idx == 0 and r_idx == 0, "Job 1 should get first combination"
    assert start == 0 and end == 10, "Job 1 should get repeats 0-9"
    
    # Job 2 should get graph 0, r_value 1, repeats 0-9
    graph_idx, r_idx, start, end = distributor.calculate_work_assignment(2, 2, 3, 10)
    print(f"✓ Job 2 assignment: graph={graph_idx}, r={r_idx}, repeats={start}-{end-1}")
    assert graph_idx == 0 and r_idx == 1, "Job 2 should get second r_value"
    
    # Job 4 should get graph 1, r_value 0, repeats 0-9
    graph_idx, r_idx, start, end = distributor.calculate_work_assignment(4, 2, 3, 10)
    print(f"✓ Job 4 assignment: graph={graph_idx}, r={r_idx}, repeats={start}-{end-1}")
    assert graph_idx == 1 and r_idx == 0, "Job 4 should get second graph, first r_value"
    
    # Job 7 should cycle back to graph 0, r_value 0, but with repeats 10-19
    graph_idx, r_idx, start, end = distributor.calculate_work_assignment(7, 2, 3, 10)
    print(f"✓ Job 7 assignment: graph={graph_idx}, r={r_idx}, repeats={start}-{end-1}")
    assert graph_idx == 0 and r_idx == 0, "Job 7 should cycle back to first combination"
    assert start == 10 and end == 20, "Job 7 should get repeats 10-19"
    
    # Test edge cases
    try:
        distributor.calculate_work_assignment(0, 2, 3, 10)
        assert False, "Should raise ValueError for job index 0"
    except ValueError:
        print("✓ Correctly handled job index 0")
    
    try:
        distributor.calculate_work_assignment(1, 0, 3, 10)
        assert False, "Should raise ValueError for zero graphs"
    except ValueError:
        print("✓ Correctly handled zero graphs in assignment")
    
    return True


def test_comprehensive_workflow():
    """Test complete workflow from job count calculation to work assignment."""
    print("Testing comprehensive workflow...")
    
    distributor = JobDistributor()
    
    # Scenario: 3 graphs, 4 r_values, 100 total repeats
    n_graphs, n_r_values, n_repeats = 3, 4, 100
    
    # Step 1: Calculate job count
    n_jobs, repeats_per_job = distributor.calculate_job_count(n_graphs, n_r_values, n_repeats)
    print(f"✓ Calculated {n_jobs} jobs with {repeats_per_job} repeats per job")
    
    # Step 2: Distribute work
    distribution = distributor.distribute_work(n_graphs, n_r_values, n_repeats, n_jobs)
    print(f"✓ Work distribution: {distribution['total_combinations']} combinations")
    
    # Step 3: Test work assignments for first few jobs
    assignments = []
    for job_idx in range(1, min(6, n_jobs + 1)):  # Test first 5 jobs
        assignment = distributor.calculate_work_assignment(job_idx, n_graphs, n_r_values, repeats_per_job)
        assignments.append((job_idx, assignment))
        print(f"✓ Job {job_idx}: graph={assignment[0]}, r={assignment[1]}, repeats={assignment[2]}-{assignment[3]-1}")
    
    # Verify that different jobs get different combinations (at least initially)
    first_combinations = [(a[1][0], a[1][1]) for a in assignments[:min(4, len(assignments))]]
    unique_combinations = set(first_combinations)
    assert len(unique_combinations) >= min(4, distribution['total_combinations']), "Jobs should get different combinations initially"
    
    print(f"✓ Verified {len(unique_combinations)} unique combinations in first jobs")
    
    return True


def main():
    """Run job distribution tests."""
    print("=== HPC Job Distribution Tests ===")
    
    tests = [
        test_job_count_calculation,
        test_work_distribution,
        test_work_assignment,
        test_comprehensive_workflow,
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
        print("✓ All job distribution tests passed!")
        return True
    else:
        print("✗ Some tests failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
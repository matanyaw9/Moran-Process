#!/usr/bin/env python
"""
Test script for result aggregation functionality.
Tests the aggregate_results method and related validation/deduplication features.
"""

import os
import sys
import tempfile
import shutil
import pandas as pd
from pathlib import Path

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from process_lab import ProcessLab


def create_test_csv_files(temp_dir: Path, num_files: int = 3):
    """Create test CSV files with sample data."""
    created_files = []
    
    for i in range(1, num_files + 1):
        # Create sample data for each job
        data = {
            'wl_hash': [f'hash_{i}_{j}' for j in range(5)],
            'graph_name': [f'test_graph_{i}' for _ in range(5)],
            'r': [1.1 + i * 0.1] * 5,
            'fixation': [True, False, True, False, True],
            'steps': [100 + j * 10 for j in range(5)],
            'initial_mutants': [1] * 5,
            'selection_coeff': [1.1 + i * 0.1] * 5,
            'duration': [0.001 + j * 0.001 for j in range(5)],
            'job_id': [i] * 5,
            'repeat_id': list(range(5))
        }
        
        df = pd.DataFrame(data)
        filename = f"results_job_{i}_20240101_120000.csv"
        filepath = temp_dir / filename
        df.to_csv(filepath, index=False)
        created_files.append(str(filepath))
        
    return created_files


def test_basic_aggregation():
    """Test basic result aggregation functionality."""
    print("Testing basic result aggregation...")
    
    lab = ProcessLab()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test CSV files
        created_files = create_test_csv_files(temp_path, num_files=3)
        print(f"Created {len(created_files)} test CSV files")
        
        # Test aggregation
        output_path = temp_path / "aggregated_results.csv"
        
        try:
            summary = lab.aggregate_results(
                temp_dir=str(temp_path),
                output_path=str(output_path),
                cleanup=False  # Don't cleanup for inspection
            )
            
            print(f"✓ Aggregation completed successfully")
            print(f"  Files processed: {summary['total_files_processed']}")
            print(f"  Total rows: {summary['total_rows']}")
            print(f"  Final rows: {summary['final_rows']}")
            
            # Verify output file exists and has correct content
            assert output_path.exists(), "Output file not created"
            
            result_df = pd.read_csv(output_path)
            expected_rows = 3 * 5  # 3 files × 5 rows each
            assert len(result_df) == expected_rows, f"Expected {expected_rows} rows, got {len(result_df)}"
            
            # Verify required columns exist
            required_columns = ['wl_hash', 'graph_name', 'r', 'fixation', 'steps']
            for col in required_columns:
                assert col in result_df.columns, f"Missing required column: {col}"
            
            print("✓ Output file validation passed")
            return True
            
        except Exception as e:
            print(f"✗ Basic aggregation failed: {e}")
            return False


def test_duplicate_prevention():
    """Test duplicate prevention functionality."""
    print("\nTesting duplicate prevention...")
    
    lab = ProcessLab()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test files with some duplicates
        # File 1: Original data
        data1 = {
            'wl_hash': ['hash1', 'hash2'],
            'graph_name': ['graph1', 'graph2'],
            'r': [1.1, 1.2],
            'fixation': [True, False],
            'steps': [100, 200],
            'job_id': [1, 1],
            'repeat_id': [0, 1]
        }
        df1 = pd.DataFrame(data1)
        df1.to_csv(temp_path / "results_job_1_20240101_120000.csv", index=False)
        
        # File 2: Contains exact duplicate of first row from file 1
        data2 = {
            'wl_hash': ['hash1', 'hash3'],  # hash1 is exact duplicate
            'graph_name': ['graph1', 'graph3'],
            'r': [1.1, 1.3],
            'fixation': [True, True],
            'steps': [100, 300],  # steps=100 matches first row exactly
            'job_id': [1, 2],     # Make job_id match for exact duplicate
            'repeat_id': [0, 1]   # Make repeat_id match for exact duplicate
        }
        df2 = pd.DataFrame(data2)
        df2.to_csv(temp_path / "results_job_2_20240101_120000.csv", index=False)
        
        # Test aggregation
        output_path = temp_path / "aggregated_results.csv"
        
        try:
            summary = lab.aggregate_results(
                temp_dir=str(temp_path),
                output_path=str(output_path),
                cleanup=False
            )
            
            print(f"✓ Aggregation with duplicates completed")
            print(f"  Original rows: {summary['original_rows']}")
            print(f"  Final rows: {summary['final_rows']}")
            print(f"  Duplicates removed: {summary.get('duplicates_removed', 0)}")
            
            # Verify duplicate was removed
            result_df = pd.read_csv(output_path)
            expected_unique_rows = 3  # Should have 3 unique rows after deduplication
            assert len(result_df) == expected_unique_rows, f"Expected {expected_unique_rows} rows after deduplication, got {len(result_df)}"
            
            print("✓ Duplicate prevention validation passed")
            return True
            
        except Exception as e:
            print(f"✗ Duplicate prevention test failed: {e}")
            return False


def test_error_handling():
    """Test error handling for various scenarios."""
    print("\nTesting error handling...")
    
    lab = ProcessLab()
    
    # Test with non-existent directory
    try:
        lab.aggregate_results("/nonexistent/directory", "/tmp/output.csv")
        print("✗ Should have failed with non-existent directory")
        return False
    except ValueError:
        print("✓ Correctly handled non-existent directory")
    
    # Test with empty directory
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "output.csv"
        
        try:
            summary = lab.aggregate_results(temp_dir, str(output_path))
            print("✓ Correctly handled empty directory")
            assert summary['total_files_found'] == 0
            assert summary['total_files_processed'] == 0
        except Exception as e:
            print(f"✗ Empty directory handling failed: {e}")
            return False
    
    return True


def main():
    """Run all result aggregation tests."""
    print("=== Result Aggregation Tests ===")
    
    tests = [
        test_basic_aggregation,
        test_duplicate_prevention,
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
        print("✓ All result aggregation tests passed!")
        return True
    else:
        print("✗ Some tests failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
#!/usr/bin/env python
"""
Test script to verify data loading works correctly from the tests directory
"""

import os
import sys

print("Testing data loading from tests directory...")
print(f"Current working directory: {os.getcwd()}")
print(f"Script location: {__file__}")

# Add parent directory to path so we can import from analysis
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Test the utility functions
try:
    # Import from analysis directory
    sys.path.insert(0, os.path.join(parent_dir, 'analysis'))
    from analysis_utils import setup_analysis_environment, load_all_data, get_data_path
    
    print("\n✓ Successfully imported analysis_utils")
    
    # Setup environment
    setup_analysis_environment()
    
    # Test path detection
    data_path = get_data_path()
    print(f"Data path: {data_path}")
    print(f"Absolute data path: {os.path.abspath(data_path)}")
    print(f"Data directory exists: {os.path.exists(data_path)}")
    
    # Test data loading
    print("\nTesting data loading...")
    data = load_all_data()
    
    # Print summary
    print("\nData loading summary:")
    for key, df in data.items():
        if not df.empty:
            print(f"  ✓ {key}: {df.shape}")
        else:
            print(f"  ✗ {key}: empty")
    
    print("\n✓ Data loading test completed successfully!")
    
except Exception as e:
    print(f"\n✗ Error during testing: {e}")
    import traceback
    traceback.print_exc()
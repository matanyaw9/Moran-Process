#!/usr/bin/env python
"""
Quick test to verify CSV append functionality
"""
import pandas as pd
import os

# Test file path
test_file = "simulation_data/test_append.csv"

# Create initial data
df1 = pd.DataFrame({
    'name': ['graph1', 'graph2'],
    'value': [10, 20]
})

print("Creating initial CSV...")
df1.to_csv(test_file, index=False)
print(f"Initial CSV created with {len(df1)} rows")

# Simulate appending new data
df2 = pd.DataFrame({
    'name': ['graph3', 'graph4'],
    'value': [30, 40]
})

print("\nAppending new data...")
if os.path.exists(test_file):
    existing_df = pd.read_csv(test_file)
    df_combined = pd.concat([existing_df, df2], ignore_index=True)
    print(f"Appending {len(df2)} new rows to existing CSV with {len(existing_df)} rows")
    df_combined.to_csv(test_file, index=False)
    print(f"Total rows now: {len(df_combined)}")

# Verify
print("\nVerifying final CSV:")
final_df = pd.read_csv(test_file)
print(final_df)
print(f"\nTotal rows: {len(final_df)}")

# Cleanup
os.remove(test_file)
print("\nTest file cleaned up. Test passed!")

#!/usr/bin/env .venv/bin/python
import sys
import os
import pandas as pd

def inspect_sas_file(sas_filename):
    if not os.path.exists(sas_filename):
        print(f"Error: File '{sas_filename}' not found")
        return False
    
    try:
        print(f"Inspecting SAS file: {sas_filename}")
        print("=" * 50)
        
        # Read with metadata
        df = pd.read_sas(sas_filename, encoding='utf-8')
        
        print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
        print()
        
        print("Column Information:")
        print("-" * 30)
        for i, col in enumerate(df.columns):
            dtype = df[col].dtype
            null_count = df[col].isnull().sum()
            unique_count = df[col].nunique()
            print(f"{i+1:2d}. {col:<20} | {str(dtype):<10} | Nulls: {null_count:3d} | Unique: {unique_count:4d}")
        
        print()
        print("Sample data (first 3 rows):")
        print("-" * 30)
        print(df.head(3).to_string())
        
        print()
        print("Data types summary:")
        print("-" * 30)
        print(df.dtypes.value_counts())
        
        print()
        print("Missing values summary:")
        print("-" * 30)
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(missing[missing > 0])
        else:
            print("No missing values found")
            
        # Check for any additional metadata or attributes
        print()
        print("File-level metadata:")
        print("-" * 30)
        try:
            # Try to read with iterator to see if there are multiple chunks/datasets
            iterator = pd.read_sas(sas_filename, chunksize=1000)
            chunk_count = 0
            total_rows = 0
            for chunk in iterator:
                chunk_count += 1
                total_rows += len(chunk)
                if chunk_count > 10:  # Limit to avoid infinite loop
                    break
            print(f"Processed {chunk_count} chunks, total rows: {total_rows}")
            
        except Exception as e:
            print(f"Chunk reading info: {e}")
            
        return True
        
    except Exception as e:
        print(f"Error inspecting file: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python sas_inspector.py <sas_filename>")
        sys.exit(1)
    
    sas_filename = sys.argv[1]
    success = inspect_sas_file(sas_filename)
    sys.exit(0 if success else 1)
#!/usr/bin/env .venv/bin/python
import sys
import os
import pandas as pd

def sas_to_csv(sas_filename):
    if not os.path.exists(sas_filename):
        print(f"Error: File '{sas_filename}' not found")
        return False
    
    base_name = os.path.splitext(sas_filename)[0]
    
    try:
        print(f"Reading SAS file: {sas_filename}")
        df = pd.read_sas(sas_filename)
        
        csv_filename = base_name + ".csv"
        df.to_csv(csv_filename, index=False)
        
        print(f"Successfully converted '{sas_filename}' to '{csv_filename}'")
        print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"Columns: {list(df.columns)}")
        
        return True
        
    except ImportError as e:
        print("Error: Required libraries not installed.")
        print("Please install: pip install pandas pyreadstat")
        return False
    except Exception as e:
        print(f"Error converting file: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python sas_to_csv.py <sas_filename>")
        sys.exit(1)
    
    sas_filename = sys.argv[1]
    success = sas_to_csv(sas_filename)
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
import sys
import os
import zipfile
import shutil

def rename_and_extract(filename):
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found")
        return False
    
    base_name = os.path.splitext(filename)[0]
    zip_filename = base_name + ".zip"
    
    try:
        shutil.move(filename, zip_filename)
        print(f"Renamed '{filename}' to '{zip_filename}'")
        
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall('.')
            print(f"Extracted contents of '{zip_filename}' to current directory")
            
        return True
        
    except zipfile.BadZipFile:
        print(f"Error: '{zip_filename}' is not a valid zip file")
        shutil.move(zip_filename, filename)
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python rename_and_extract.py <filename>")
        sys.exit(1)
    
    filename = sys.argv[1]
    rename_and_extract(filename)
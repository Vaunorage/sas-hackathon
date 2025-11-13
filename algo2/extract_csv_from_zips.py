#!/usr/bin/env python3
"""
Script to extract CSV files from zip archives into a destination folder.
"""
import zipfile
from pathlib import Path

from paths import HERE


def extract_csv_from_zips(zip_files, destination_folder):
    """
    Extract all CSV files from the given zip files into the destination folder.
    
    Args:
        zip_files: List of paths to zip files
        destination_folder: Path to destination folder
    """
    destination = Path(destination_folder)
    destination.mkdir(parents=True, exist_ok=True)
    
    extracted_count = 0
    
    for zip_path in zip_files:
        zip_path = Path(zip_path)
        
        if not zip_path.exists():
            print(f"Warning: {zip_path} does not exist, skipping...")
            continue
        
        print(f"\nProcessing {zip_path.name}...")
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Get all files in the zip
                all_files = zip_ref.namelist()
                
                # Filter for CSV files
                csv_files = [f for f in all_files if f.lower().endswith('.csv')]
                
                if not csv_files:
                    print(f"  No CSV files found in {zip_path.name}")
                    continue
                
                # Extract each CSV file
                for csv_file in csv_files:
                    # Extract to destination folder
                    zip_ref.extract(csv_file, destination)
                    
                    # If the file is in a subdirectory, move it to the root of destination
                    extracted_path = destination / csv_file
                    if extracted_path.parent != destination:
                        # Move file to destination root
                        final_path = destination / extracted_path.name
                        extracted_path.rename(final_path)
                        
                        # Clean up empty directories
                        try:
                            extracted_path.parent.rmdir()
                        except OSError:
                            pass  # Directory not empty or other issue
                    
                    print(f"  ✓ Extracted: {csv_file}")
                    extracted_count += 1
                    
        except zipfile.BadZipFile:
            print(f"Error: {zip_path.name} is not a valid zip file")
        except Exception as e:
            print(f"Error processing {zip_path.name}: {e}")
    
    print(f"\n{'='*60}")
    print(f"Extraction complete! Total CSV files extracted: {extracted_count}")
    print(f"Destination folder: {destination.absolute()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Define the zip files to process
    # Check if running in Docker (working dir is /app) or locally
    if HERE.name == 'app' or (HERE / 'flask_app.py').exists():
        # Running in Docker or from algo2 directory
        base_dir = HERE
    else:
        # Running from repo root
        base_dir = HERE / 'algo2'
    
    zip_data_dir = base_dir / "default_data"
    
    # Find all zip files in the default_data directory
    zip_files = list(zip_data_dir.glob("*.zip"))
    
    if not zip_files:
        print(f"Warning: No zip files found in {zip_data_dir}")
        # Fallback to specific files if directory doesn't exist
        zip_files = [
            zip_data_dir / "data_1.zip",
            zip_data_dir / "data_2.zip",
            zip_data_dir / "data_3.zip",
            zip_data_dir / "data_4.zip",
            zip_data_dir / "data_5.zip",
        ]
    
    # Define destination folder (you can change this)
    destination_folder = base_dir / "data_in"
    
    print("CSV Extraction Script")
    print(f"{'='*60}")
    print(f"Base directory: {base_dir}")
    print(f"Source zip files: {len(zip_files)}")
    print(f"Destination: {destination_folder}")
    print(f"{'='*60}")
    
    extract_csv_from_zips(zip_files, destination_folder)

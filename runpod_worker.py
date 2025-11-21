import runpod
import os
import tempfile
from pathlib import Path
import pandas as pd
import gzip
import gpu # Import the main gpu projection script

def handler(job):
    """
    RunPod serverless handler for the GPU projection.

    The job input is expected to be a dictionary containing:
    - 'nb_an_projection': Number of years for the projection.
    - 'nb_scenarios': Number of economic scenarios.
    - 'data_files': A dictionary where keys are file names (e.g., 'POPULATION.csv')
                    and values are the file contents as base64 encoded strings.
    """
    job_input = job['input']

    # --- 1. Get parameters from job input ---
    try:
        nb_an_projection = int(job_input.get('nb_an_projection', 10))
        nb_scenarios = int(job_input.get('nb_scenarios', 100))
        data_files_b64 = job_input.get('data_files', {})
    except (ValueError, TypeError) as e:
        return {'error': f"Invalid input parameter: {e}"}

    # --- 2. Create temporary directories for input and output ---
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir) / 'data_in'
        output_path = Path(tmpdir) / 'data_out'
        data_path.mkdir()
        output_path.mkdir()

        # --- 3. Decode and write uploaded files, use defaults for the rest ---
        file_paths = {}
        default_data_path = Path('/app/data_in')  # Default CSVs baked into Docker image
        
        required_files = [
            'POPULATION.csv', 'MORTALITE.csv', 'RENDEMENTS.csv', 'DEPOTS_FUTURS.csv',
            'FRAIS_ADMIN.csv', 'MIN_FERR.csv', 'TX_LAPSE_PART.csv', 'TX_LAPSE_TOT.csv',
            'ACQUISITION.csv', 'COUSSINS_ESCAP.csv'
        ]
        
        try:
            import base64
            import shutil
            
            for filename in required_files:
                arg_name = f"{filename.split('.')[0].lower()}_path"
                
                if filename in data_files_b64:
                    # Use uploaded file
                    file_path = data_path / filename
                    with open(file_path, 'wb') as f:
                        # Decode from base64
                        compressed_bytes = base64.b64decode(data_files_b64[filename])
                        # Decompress with gzip
                        decompressed_bytes = gzip.decompress(compressed_bytes)
                        f.write(decompressed_bytes)
                    file_paths[arg_name] = file_path
                    print(f"  Using uploaded: {filename}")
                else:
                    # Use default file from Docker image
                    default_file = default_data_path / filename
                    if default_file.exists():
                        # Copy default to temp directory
                        dest_file = data_path / filename
                        shutil.copy(default_file, dest_file)
                        file_paths[arg_name] = dest_file
                        print(f"  Using default: {filename}")
                    else:
                        return {'error': f"Required file not found: {filename} (not uploaded and no default available)"}
                        
        except Exception as e:
            return {'error': f"Failed to process data files: {e}"}

        # --- 4. Run the GPU projection ---
        try:
            print(f"Starting GPU projection with {nb_an_projection} years and {nb_scenarios} scenarios.")
            # The run_projection_gpu function returns the final aggregated DataFrame
            results_df = gpu.run_projection_gpu(
                data_path=data_path,
                output_path=output_path,
                nb_an_projection=nb_an_projection,
                nb_scenarios=nb_scenarios,
                **file_paths # Pass the specific paths for each data file
            )
            print("GPU projection completed successfully.")

            # --- 5. Convert results to JSON and return ---
            if isinstance(results_df, pd.DataFrame):
                # Convert DataFrame to a list of dictionaries for JSON serialization
                result_json = results_df.to_dict(orient='records')
                return {'results': result_json}
            else:
                return {'error': 'Projection did not return a DataFrame.'}

        except Exception as e:
            # Catch exceptions from the GPU projection and return an error
            import traceback
            return {
                'error': f"An error occurred during GPU projection: {e}",
                'traceback': traceback.format_exc()
            }

# --- Start the RunPod serverless worker ---
runpod.serverless.start({"handler": handler})

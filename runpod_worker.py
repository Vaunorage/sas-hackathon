import runpod
import os
import tempfile
from pathlib import Path
import pandas as pd
import requests
import gpu # Import the main gpu projection script

def handler(job):
    """
    RunPod serverless handler for the GPU projection.

    The job input is expected to be a dictionary containing:
    - 'nb_an_projection': Number of years for the projection.
    - 'nb_scenarios': Number of economic scenarios.
    - 'data_file_urls': A dictionary where keys are file names (e.g., 'POPULATION.csv')
                        and values are temporary download URLs (from tmpfiles.org).
    """
    job_input = job['input']

    # --- 1. Get parameters from job input ---
    try:
        nb_an_projection = int(job_input.get('nb_an_projection', 10))
        nb_scenarios = int(job_input.get('nb_scenarios', 100))
        max_accounts = job_input.get('max_accounts', None)
        if max_accounts is not None:
            max_accounts = int(max_accounts)
        debug_account = job_input.get('debug_account', None)
        if debug_account is not None:
            debug_account = int(debug_account)
        debug_scenario = job_input.get('debug_scenario', None)
        if debug_scenario is not None:
            debug_scenario = int(debug_scenario)
        data_file_urls = job_input.get('data_file_urls', {})
    except (ValueError, TypeError) as e:
        return {'error': f"Invalid input parameter: {e}"}

    # --- 2. Create temporary directories for input and output ---
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir) / 'data_in'
        output_path = Path(tmpdir) / 'data_out'
        data_path.mkdir()
        output_path.mkdir()

        # --- 3. Download uploaded files from URLs, use defaults for the rest ---
        file_paths = {}
        default_data_path = Path('/data_in')  # Default CSVs baked into Docker image
        
        required_files = [
            'POPULATION.csv', 'MORTALITE.csv', 'RENDEMENTS.csv', 'DEPOTS_FUTURS.csv',
            'FRAIS_ADMIN.csv', 'MIN_FERR.csv', 'TX_LAPSE_PART.csv', 'TX_LAPSE_TOT.csv',
            'ACQUISITION.csv', 'COUSSINS_ESCAP.csv'
        ]
        
        try:
            import shutil
            
            for filename in required_files:
                arg_name = f"{filename.split('.')[0].lower()}_path"
                
                if filename in data_file_urls:
                    # Download uploaded file from URL
                    file_path = data_path / filename
                    url = data_file_urls[filename]
                    
                    print(f"  Downloading {filename} from tmpfiles.org...")
                    response = requests.get(url, timeout=120)
                    response.raise_for_status()
                    
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                    
                    file_paths[arg_name] = file_path
                    print(f"  ✓ Downloaded: {filename} ({len(response.content)} bytes)")
                else:
                    # Use default file from Docker image
                    default_file = default_data_path / filename
                    if default_file.exists():
                        # Copy default to temp directory
                        dest_file = data_path / filename
                        shutil.copy(default_file, dest_file)
                        file_paths[arg_name] = dest_file
                        print(f"  ✓ Using default: {filename}")
                    else:
                        return {'error': f"Required file not found: {filename} (not uploaded and no default available)"}
                        
        except Exception as e:
            return {'error': f"Failed to process data files: {str(e)}"}

        # --- 4. Run the GPU projection ---
        try:
            # Build log message
            log_parts = [f"{nb_an_projection} years", f"{nb_scenarios} scenarios"]
            if max_accounts:
                log_parts.append(f"max {max_accounts} accounts")
            if debug_account is not None:
                log_parts.append(f"debug account {debug_account}")
            if debug_scenario is not None:
                log_parts.append(f"debug scenario {debug_scenario}")
            print(f"Starting GPU projection with {', '.join(log_parts)}.")
            
            # Progress callback to report batch progress to RunPod
            def progress_callback(current_batch, total_batches):
                progress_percent = int((current_batch / total_batches) * 100)
                try:
                    runpod.serverless.progress_update(
                        job,
                        f"Processing batch {current_batch}/{total_batches} ({progress_percent}%)"
                    )
                    print(f"  Progress reported to RunPod: Batch {current_batch}/{total_batches} ({progress_percent}%)")
                except Exception as e:
                    print(f"  Warning: Failed to report progress: {e}")
            
            # The run_projection_gpu function returns a dict with 3 DataFrames
            results_dict = gpu.run_projection_gpu(
                data_path=data_path,
                output_path=output_path,
                nb_an_projection=nb_an_projection,
                nb_scenarios=nb_scenarios,
                max_accounts=max_accounts,
                debug_account=debug_account,
                debug_scenario=debug_scenario,
                progress_callback=progress_callback,
                **file_paths # Pass the specific paths for each data file
            )
            print("GPU projection completed successfully.")

            # --- 5. Convert results to JSON and return ---
            if isinstance(results_dict, dict):
                # Convert each DataFrame to JSON format
                output = {}
                for key, df in results_dict.items():
                    if df is not None and isinstance(df, pd.DataFrame):
                        output[key] = df.to_dict(orient='records')
                        print(f"  ✓ Converted {key}: {len(df)} rows")
                    else:
                        output[key] = None
                        print(f"  ⚠ Skipped {key}: empty or None")
                
                return {'results': output}
            else:
                return {'error': 'Projection did not return expected dictionary format.'}

        except Exception as e:
            # Catch exceptions from the GPU projection and return an error
            import traceback
            return {
                'error': f"An error occurred during GPU projection: {e}",
                'traceback': traceback.format_exc()
            }

# --- Start the RunPod serverless worker ---
runpod.serverless.start({"handler": handler})

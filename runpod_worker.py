import runpod
import os
import tempfile
from pathlib import Path
import pandas as pd
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

    if not data_files_b64:
        return {'error': 'No data files provided in the job input.'}

    # --- 2. Create temporary directories for input and output ---
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir) / 'data_in'
        output_path = Path(tmpdir) / 'data_out'
        data_path.mkdir()
        output_path.mkdir()

        # --- 3. Decode and write data files to the temp directory ---
        file_paths = {}
        try:
            for filename, content_b64 in data_files_b64.items():
                file_path = data_path / filename
                with open(file_path, 'wb') as f:
                    import base64
                    f.write(base64.b64decode(content_b64))
                # Store the path with a key that matches the argument names in run_projection_gpu
                arg_name = f"{filename.split('.')[0].lower()}_path"
                file_paths[arg_name] = file_path
        except Exception as e:
            return {'error': f"Failed to decode or write data file: {e}"}

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

import runpod
import tempfile
from pathlib import Path
import pandas as pd
import requests
import importlib
import traceback
import os
import sys

# Store original kernel content for restoration
# Use environment variable or default path for Docker compatibility
KERNELS_PATH = Path(os.environ.get('KERNELS_PATH', Path(__file__).parent / 'calculations' / 'kernels.py'))
_original_kernel_content = None


def apply_custom_kernel(kernel_code: str) -> dict:
    """
    Apply custom kernel code by writing to kernels.py and reloading modules.
    
    Returns dict with 'success' or 'error' key.
    """
    global _original_kernel_content
    
    try:
        # Backup original content (only once)
        if _original_kernel_content is None and KERNELS_PATH.exists():
            _original_kernel_content = KERNELS_PATH.read_text()
        
        # Validate syntax first
        try:
            compile(kernel_code, 'kernels.py', 'exec')
        except SyntaxError as e:
            return {
                'error': f'Syntax error at line {e.lineno}: {e.msg}',
                'line': e.lineno
            }
        
        # Write new kernel code
        KERNELS_PATH.write_text(kernel_code)
        print(f"[KERNEL] Custom kernel code written ({len(kernel_code)} bytes)")
        sys.stdout.flush()
        
        # Reload modules
        import calculations.kernels
        importlib.reload(calculations.kernels)
        print("[KERNEL] Reloaded calculations.kernels")
        sys.stdout.flush()
        
        import calculations.gpu
        importlib.reload(calculations.gpu)
        print("[KERNEL] Reloaded calculations.gpu")
        sys.stdout.flush()

        try:
            calculations.gpu.validate_kernel_compatibility()
        except Exception as e:
            restore_original_kernel()
            return {
                'error': 'Kernel not compatible with the running methods',
                'details': str(e)
            }
        
        return {'success': True}
        
    except Exception as e:
        # Restore original on failure
        restore_original_kernel()
        return {
            'error': f'Failed to apply custom kernel: {str(e)}',
            'traceback': traceback.format_exc()
        }


def restore_original_kernel():
    """Restore the original kernel code."""
    global _original_kernel_content
    
    if _original_kernel_content is not None:
        try:
            KERNELS_PATH.write_text(_original_kernel_content)
            
            import calculations.kernels
            importlib.reload(calculations.kernels)
            
            import calculations.gpu
            importlib.reload(calculations.gpu)
            
            print("[KERNEL] Restored original kernel code")
        except Exception as e:
            print(f"[KERNEL] Warning: Failed to restore original kernel: {e}")


def handler(job):
    """
    RunPod serverless handler for the GPU projection.

    The job input is expected to be a dictionary containing:
    - 'nb_an_projection': Number of years for the projection.
    - 'nb_ext_scenarios': Number of external (real-world) scenarios.
    - 'nb_int_scenarios': Number of internal (risk-neutral) scenarios per node.
    - 'data_file_urls': A dictionary where keys are file names (e.g., 'POPULATION.csv')
                        and values are temporary download URLs (from tmpfiles.org).
    - 'kernel_code': (Optional) Custom kernels.py code to use for this job.
    """
    job_input = job['input']
    custom_kernel_applied = False

    # --- 0. Apply custom kernel if provided ---
    kernel_code = job_input.get('kernel_code')
    if kernel_code:
        print(f"[KERNEL] Custom kernel code provided ({len(kernel_code)} bytes)")
        result = apply_custom_kernel(kernel_code)
        if 'error' in result:
            return {
                'error': f"Failed to apply custom kernel: {result['error']}",
                'kernel_error': result
            }
        custom_kernel_applied = True
        print("[KERNEL] Custom kernel applied successfully")

    # --- 1. Get parameters from job input ---
    try:
        nb_an_projection = int(job_input.get('nb_an_projection', 100))
        nb_ext_scenarios = int(job_input.get('nb_ext_scenarios', job_input.get('nb_scenarios', 100)))
        nb_int_scenarios = int(job_input.get('nb_int_scenarios', 100))
        shock_capital_pct = float(job_input.get('shock_capital_pct', 0.35))
        max_accounts = job_input.get('max_accounts', None)
        if max_accounts is not None:
            max_accounts = int(max_accounts)
        debug_account = job_input.get('debug_account', -1)
        if debug_account is not None:
            debug_account = int(debug_account)
        debug_scenario = job_input.get('debug_scenario', -1)
        if debug_scenario is not None:
            debug_scenario = int(debug_scenario)
        debug_year = job_input.get('debug_year', -1)
        if debug_year is not None:
            debug_year = int(debug_year)
        debug_month = job_input.get('debug_month', -1)
        if debug_month is not None:
            debug_month = int(debug_month)
        debug_int_scenario = job_input.get('debug_int_scenario', -1)
        if debug_int_scenario is not None:
            debug_int_scenario = int(debug_int_scenario)
        debug_int_year = job_input.get('debug_int_year', -1)
        if debug_int_year is not None:
            debug_int_year = int(debug_int_year)
        debug_only = job_input.get('debug_only', False)
        data_file_urls = job_input.get('data_file_urls', {})
    except (ValueError, TypeError) as e:
        if custom_kernel_applied:
            restore_original_kernel()
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
            log_parts = [f"{nb_an_projection} years", f"{nb_ext_scenarios} ext scenarios", f"{nb_int_scenarios} int scenarios"]
            if max_accounts:
                log_parts.append(f"max {max_accounts} accounts")
            if debug_account is not None and debug_account >= 0:
                log_parts.append(f"debug account {debug_account}")
            if debug_scenario is not None and debug_scenario >= 0:
                log_parts.append(f"debug scenario {debug_scenario}")
            if custom_kernel_applied:
                log_parts.append("custom kernel")
            print(f"Starting GPU nested projection with {', '.join(log_parts)}.")
            sys.stdout.flush()
            
            # Progress callback to report batch progress to RunPod
            def progress_callback(current_batch, total_batches):
                progress_percent = int((current_batch / total_batches) * 100)
                try:
                    runpod.serverless.progress_update(
                        job,
                        f"Processing batch {current_batch}/{total_batches} ({progress_percent}%)"
                    )
                    print(f"  Progress reported to RunPod: Batch {current_batch}/{total_batches} ({progress_percent}%)")
                    sys.stdout.flush()
                except Exception as e:
                    print(f"  Warning: Failed to report progress: {e}")
                    sys.stdout.flush()
            
            # Import run_projection_gpu_nested from the (possibly reloaded) module
            print("[DEBUG] Importing run_projection_gpu_nested...")
            sys.stdout.flush()
            from calculations.gpu import run_projection_gpu_nested
            print("[DEBUG] Import successful, starting projection...")
            sys.stdout.flush()
            
            # run_projection_gpu_nested returns a ProjectionResult dataclass
            result = run_projection_gpu_nested(
                data_path=data_path,
                output_path=output_path,
                nb_an_projection=nb_an_projection,
                nb_ext_scenarios=nb_ext_scenarios,
                nb_int_scenarios=nb_int_scenarios,
                shock_capital_pct=shock_capital_pct,
                max_accounts=max_accounts,
                debug_account=debug_account if debug_account is not None else -1,
                debug_scenario=debug_scenario if debug_scenario is not None else -1,
                debug_year=debug_year if debug_year is not None else -1,
                debug_month=debug_month if debug_month is not None else -1,
                debug_int_scenario=debug_int_scenario if debug_int_scenario is not None else -1,
                debug_int_year=debug_int_year if debug_int_year is not None else -1,
                debug_only=debug_only,
                progress_callback=progress_callback,
                **file_paths  # Pass the specific paths for each data file
            )
            print("GPU nested projection completed successfully.")

            # --- 5. Convert results to JSON and return ---
            output = {
                'total_duration': result.total_duration,
                'saved_files': result.saved_files,
                'custom_kernel_used': custom_kernel_applied,
            }
            
            # Convert DataFrames to JSON format
            if result.results is not None:
                output['results'] = result.results.to_dict(orient='records')
                print(f"  ✓ Converted results: {len(result.results)} rows")
            
            if result.vp_flux_total is not None:
                output['vp_flux_total'] = result.vp_flux_total.to_dict(orient='records')
                print(f"  ✓ Converted vp_flux_total: {len(result.vp_flux_total)} rows")
            
            if result.results_5chocs is not None:
                output['results_5chocs'] = result.results_5chocs.to_dict(orient='records')
                print(f"  ✓ Converted results_5chocs: {len(result.results_5chocs)} rows")
            
            if result.sensitivities is not None:
                output['sensitivities'] = result.sensitivities.to_dict(orient='records')
                print(f"  ✓ Converted sensitivities: {len(result.sensitivities)} rows")
            
            if result.chocs_summary is not None:
                output['chocs_summary'] = result.chocs_summary.to_dict(orient='records')
                print(f"  ✓ Converted chocs_summary: {len(result.chocs_summary)} rows")
            
            if result.ext_debug_df is not None:
                output['ext_debug'] = result.ext_debug_df.to_dict(orient='records')
                print(f"  ✓ Converted ext_debug: {len(result.ext_debug_df)} rows")
            
            if result.int_debug_df is not None:
                output['int_debug'] = result.int_debug_df.to_dict(orient='records')
                print(f"  ✓ Converted int_debug: {len(result.int_debug_df)} rows")
            
            if result.flux_projetes_df is not None:
                output['flux_projetes'] = result.flux_projetes_df.to_dict(orient='records')
                print(f"  ✓ Converted flux_projetes: {len(result.flux_projetes_df)} rows")
            
            # Restore original kernel after job completes
            if custom_kernel_applied:
                restore_original_kernel()
            
            return {'results': output}

        except Exception as e:
            print(f"[ERROR] GPU projection failed: {e}")
            print(f"[ERROR] Traceback:\n{traceback.format_exc()}")
            sys.stdout.flush()
            # Restore original kernel on error
            if custom_kernel_applied:
                restore_original_kernel()
            # Catch exceptions from the GPU projection and return an error
            return {
                'error': f"An error occurred during GPU projection: {e}",
                'traceback': traceback.format_exc()
            }

# --- Start the RunPod serverless worker ---
runpod.serverless.start({"handler": handler})

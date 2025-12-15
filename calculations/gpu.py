import os

from calculations.kernels import (
    external_generator_kernel, nested_valuation_kernel_five_chocs, STATE_SIZE,
    EXT_DEBUG_SIZE, INT_DEBUG_SIZE,
)
from calculations.constants import (
    MAX_SEXE, MAX_AGE, MAX_LAPSE_LEVELS, MAX_DURATION, DEFAULT_AGE_MAX_DEPOSIT,
    RN_DEFAULT_FORWARD_RATE, RN_DEFAULT_REND_DEX, RN_DEFAULT_REND_MM,
    RN_DEFAULT_REND_TSX, RN_DEFAULT_REND_SP500, RN_DEFAULT_REND_EAFE,
    NUM_CHOCS, CHOC_NAMES, METRICS_RESERVE_IDX, METRICS_CAPITAL_IDX, METRICS_OUTPUT_SIZE,
    LOOKUP_TABLE_OVERHEAD_MB, DEFAULT_GPU_MEMORY_GB, MEMORY_SAFETY_FACTOR, MEMORY_BATCH_THRESHOLD,
    DEFAULT_THREADS_PER_BLOCK_1D,
)

os.environ['NUMBA_CUDA_ENABLE_PYNVJITLINK'] = '1'

from calculations.utils import logger, CONFIG, load_all_data, prepare_account_data
from numba import cuda
from paths import HERE
import argparse
import pandas as pd
import numpy as np
import polars as pl
import gc
from pathlib import Path
from typing import Optional, List, TypedDict
from dataclasses import dataclass
from datetime import datetime
from fastparquet import write as fastparquet_write

# Try to import cuDF for GPU-accelerated DataFrame operations
try:
    import cudf
    import cupy as cp
except ImportError:
    print("⚠ CuDF not available - falling back to pandas (CPU). Install with: pip install cudf-cu12")


@dataclass
class ProjectionResult:
    """Result of run_projection_gpu_nested containing all output DataFrames."""
    results: pd.DataFrame
    results_5chocs: Optional[pd.DataFrame]
    sensitivities: Optional[pd.DataFrame]
    total_duration: float
    vp_flux_total: pd.DataFrame
    chocs_summary: Optional[pd.DataFrame]
    ext_debug_df: Optional[pd.DataFrame]
    int_debug_df: Optional[pd.DataFrame]
    saved_files: List[str]


def create_gpu_mortality_lookup(df: pd.DataFrame):
    """Create flattened array for mortality lookup on GPU."""
    from calculations.constants import DEFAULT_MORTALITY_RATE
    # Create a 4D array indexed by: [i_sexe, age, year, i_produit_regr]
    max_year = df['ANNEE_REELLE'].max() + 1
    max_produit = df['I_PRODUIT_REGR'].max() + 1

    # Initialize with default value
    lookup = np.full((MAX_SEXE, MAX_AGE, max_year, max_produit), DEFAULT_MORTALITY_RATE, dtype=np.float32)

    for _, row in df.iterrows():
        i_sexe = int(row['I_SEXE'])
        age = int(row['AGE_MORTALITE'])
        year = int(row['ANNEE_REELLE'])
        i_produit = int(row['I_PRODUIT_REGR'])
        lookup[i_sexe, age, year, i_produit] = float(row['QX'])

    return lookup


def create_gpu_returns_lookup(df: pd.DataFrame):
    """Create flattened arrays for returns lookup on GPU."""
    max_scn = df['SCN_EVAL'].max() + 1
    max_an = df['AN_EVAL'].max() + 1
    max_mois = df['MOIS_EVAL'].max() + 1

    # Create separate arrays for each return type
    forward_rate = np.zeros((max_scn, max_an, max_mois), dtype=np.float32)
    ajust_forward = np.zeros((max_scn, max_an, max_mois), dtype=np.float32)
    rend_dex = np.zeros((max_scn, max_an, max_mois), dtype=np.float32)
    rend_mm = np.zeros((max_scn, max_an, max_mois), dtype=np.float32)
    rend_tsx = np.zeros((max_scn, max_an, max_mois), dtype=np.float32)
    rend_sp500 = np.zeros((max_scn, max_an, max_mois), dtype=np.float32)
    rend_eafe = np.zeros((max_scn, max_an, max_mois), dtype=np.float32)

    for _, row in df.iterrows():
        scn = int(row['SCN_EVAL'])
        an = int(row['AN_EVAL'])
        mois = int(row['MOIS_EVAL'])
        forward_rate[scn, an, mois] = float(row['FORWARD_RATE'])
        ajust_forward[scn, an, mois] = float(row['AJUST_FORWARD_RATE_VM_0'])
        rend_dex[scn, an, mois] = float(row['RENDDEX_AN'])
        rend_mm[scn, an, mois] = float(row['RENDMM_AN'])
        rend_tsx[scn, an, mois] = float(row['RENDTSX_AN'])
        rend_sp500[scn, an, mois] = float(row['RENDSP500_AN'])
        rend_eafe[scn, an, mois] = float(row['RENDEAFE_AN'])

    return (forward_rate, ajust_forward, rend_dex, rend_mm, rend_tsx, rend_sp500, rend_eafe)


def create_gpu_min_ferr_lookup(df: pd.DataFrame):
    """Create array for minimum FERR lookup."""
    lookup = np.zeros(MAX_AGE, dtype=np.float32)
    for _, row in df.iterrows():
        age = int(row['AGE'])
        lookup[age] = float(row['MIN_FERR'])
    return lookup


def create_gpu_lapse_part_lookup(df: pd.DataFrame):
    """Create arrays for partial lapse lookup."""
    max_id_lapse = df['ID_LAPSE'].max() + 1
    max_regime = df['I_REGIME_2'].max() + 1

    tx_min = np.zeros((MAX_AGE, max_id_lapse, max_regime, MAX_LAPSE_LEVELS), dtype=np.float32)
    tx_max = np.zeros((MAX_AGE, max_id_lapse, max_regime, MAX_LAPSE_LEVELS), dtype=np.float32)

    for _, row in df.iterrows():
        age = int(row['AGE'])
        id_lapse = int(row['ID_LAPSE'])
        regime = int(row['I_REGIME_2'])
        niv = int(row['LAPSE_NIV_PART'])
        tx_min[age, id_lapse, regime, niv] = float(row['TX_LAPSE_PART_MIN'])
        tx_max[age, id_lapse, regime, niv] = float(row['TX_LAPSE_PART_MAX'])

    return tx_min, tx_max


def create_gpu_lapse_tot_lookup(df: pd.DataFrame):
    """Create arrays for total lapse lookup."""
    max_id_lapse = df['ID_LAPSE'].max() + 1

    tx_min = np.zeros((MAX_DURATION, max_id_lapse, MAX_LAPSE_LEVELS), dtype=np.float32)
    tx_max = np.zeros((MAX_DURATION, max_id_lapse, MAX_LAPSE_LEVELS), dtype=np.float32)
    fact_dim = np.ones((MAX_DURATION, max_id_lapse, MAX_LAPSE_LEVELS), dtype=np.float32)

    for _, row in df.iterrows():
        duree = int(row['DUREE_MAX10'])
        id_lapse = int(row['ID_LAPSE'])
        niv = int(row['LAPSE_NIV_TOT'])
        tx_min[duree, id_lapse, niv] = float(row['TX_LAPSE_TOT_MIN'])
        tx_max[duree, id_lapse, niv] = float(row['TX_LAPSE_TOT_MAX'])
        fact_dim[duree, id_lapse, niv] = float(row['FACT_DIM'])

    return tx_min, tx_max, fact_dim


def create_gpu_fees_lookup(df: pd.DataFrame):
    """Create array for fees lookup."""
    max_produit = df['ID_PRODUIT'].max() + 1
    max_year = df['ANNEE_REELLE'].max() + 1

    lookup = np.zeros((max_produit, max_year), dtype=np.float32)
    for _, row in df.iterrows():
        produit = int(row['ID_PRODUIT'])
        year = int(row['ANNEE_REELLE'])
        lookup[produit, year] = float(row['FRAIS'])

    return lookup


def create_gpu_deposits_lookup(df: pd.DataFrame):
    """Create arrays for deposits lookup."""
    max_id_depot = df['ID_DEPOT'].max() + 1

    pc_depot = np.zeros((MAX_DURATION, max_id_depot), dtype=np.float32)
    var_fct = np.zeros((MAX_DURATION, max_id_depot), dtype=np.int32)
    age_max = np.full((MAX_DURATION, max_id_depot), DEFAULT_AGE_MAX_DEPOSIT, dtype=np.int32)
    i_even = np.zeros((MAX_DURATION, max_id_depot), dtype=np.int32)

    for _, row in df.iterrows():
        duree = int(row['DUREE_MAX10'])
        id_depot = int(row['ID_DEPOT'])
        pc_depot[duree, id_depot] = float(row['PC_DEPOT_ANNUEL'])
        var_fct[duree, id_depot] = int(row['VAR_DEPOT_FCT'])
        age_max[duree, id_depot] = int(row['AGE_MAX_DEPOT'])
        i_even[duree, id_depot] = int(row['I_EVEN_CESSE_DEPOT'])

    return pc_depot, var_fct, age_max, i_even


def create_gpu_acquisition_lookup(df: pd.DataFrame):
    """Create arrays for acquisition lookup."""
    max_id_acqui = df['ID_ACQUI'].max() + 1

    pc_vente_rf = np.zeros((MAX_DURATION, max_id_acqui), dtype=np.float32)
    pc_vente_ac = np.zeros((MAX_DURATION, max_id_acqui), dtype=np.float32)
    pc_maintien_rf = np.zeros((MAX_DURATION, max_id_acqui), dtype=np.float32)
    pc_maintien_ac = np.zeros((MAX_DURATION, max_id_acqui), dtype=np.float32)
    pc_frais_ac = np.zeros((MAX_DURATION, max_id_acqui), dtype=np.float32)
    pc_frais_rf = np.zeros((MAX_DURATION, max_id_acqui), dtype=np.float32)

    for _, row in df.iterrows():
        duree = int(row['DUREE_MAX10'])
        id_acqui = int(row['ID_ACQUI'])
        pc_vente_rf[duree, id_acqui] = float(row['PC_COMMISSION_VENTE_RF'])
        pc_vente_ac[duree, id_acqui] = float(row['PC_COMMISSION_VENTE_AC'])
        pc_maintien_rf[duree, id_acqui] = float(row['PC_COMMISSION_MAINTIEN_RF'])
        pc_maintien_ac[duree, id_acqui] = float(row['PC_COMMISSION_MAINTIEN_AC'])
        pc_frais_ac[duree, id_acqui] = float(row['PC_FRAIS_AN_AC'])
        pc_frais_rf[duree, id_acqui] = float(row['PC_FRAIS_AN_RF'])

    return pc_vente_rf, pc_vente_ac, pc_maintien_rf, pc_maintien_ac, pc_frais_ac, pc_frais_rf


def initialize_gpu():
    """
    Initialize GPU and check availability.
    
    Returns:
        Tuple of (gpu_device, free_mem, total_mem) or raises RuntimeError
    """
    try:
        if not cuda.is_available():
            raise RuntimeError("CUDA is not available")
        
        gpu = cuda.get_current_device()
        print(f"GPU Device: {gpu.name.decode()}")
        
        try:
            free_mem, total_mem = cuda.current_context().get_memory_info()
            print(f"GPU Memory: {free_mem / 1024**3:.2f} GB free / {total_mem / 1024**3:.2f} GB total")
        except NotImplementedError:
            free_mem, total_mem = None, None
        
        return gpu, free_mem, total_mem
    except Exception as e:
        raise RuntimeError(f"Failed to initialize GPU: {e}")


def calculate_batch_size(n_accounts: int, nb_ext_scenarios: int, nb_an_projection: int, 
                         nb_int_scenarios: int, account_data_cols: int):
    """
    Calculate optimal batch size based on memory requirements.
    
    Returns:
        Tuple of (batch_size, num_batches, total_mem_per_account, lookup_overhead)
    """
    print("\nCalculating memory requirements...")
    
    # State tensor: (Batch, Ext_Scenarios, Years, STATE_SIZE)
    state_mem_per_account = nb_ext_scenarios * nb_an_projection * STATE_SIZE * 4  # float32
    
    # Cashflow tensor: (Batch, Ext_Scenarios, Years, 1)
    cf_mem_per_account = nb_ext_scenarios * nb_an_projection * 1 * 4
    
    # Metrics tensor: (Batch, Ext_Scenarios, Years, NUM_CHOCS, METRICS_OUTPUT_SIZE) - chocs × (Reserve & Capital)
    metrics_mem_per_account = nb_ext_scenarios * nb_an_projection * NUM_CHOCS * METRICS_OUTPUT_SIZE * 4
    
    total_mem_per_account = (state_mem_per_account + cf_mem_per_account + 
                             metrics_mem_per_account + account_data_cols * 4)
    
    # Estimate lookup table memory overhead (always resident on GPU)
    lookup_overhead = 0
    lookup_overhead += 6 * nb_ext_scenarios * nb_an_projection * 12 * 4
    lookup_overhead += 6 * nb_int_scenarios * nb_an_projection * 4
    lookup_overhead += LOOKUP_TABLE_OVERHEAD_MB * 1024**2
    
    print(f"  State tensor per account: {state_mem_per_account / 1024**2:.2f} MB")
    print(f"  Total memory per account: {total_mem_per_account / 1024**2:.2f} MB")
    print(f"  Lookup table overhead: {lookup_overhead / 1024**2:.2f} MB")
    
    # Calculate batch size (conservative for nested scenarios)
    try:
        free_mem, total_mem = cuda.current_context().get_memory_info()
        print(f"  GPU free memory: {free_mem / 1024**3:.2f} GB")
        print(f"  GPU total memory: {total_mem / 1024**3:.2f} GB")
        available_mem = max(0, (free_mem - lookup_overhead) * MEMORY_SAFETY_FACTOR)
    except NotImplementedError:
        print("  Warning: Cannot query GPU memory, using conservative estimate")
        available_mem = max(0, DEFAULT_GPU_MEMORY_GB * 1024**3 - lookup_overhead)
    
    batch_size = max(1, int(available_mem // total_mem_per_account))
    batch_size = min(batch_size, n_accounts)
    num_batches = (n_accounts + batch_size - 1) // batch_size
    
    print(f"  Batch size: {batch_size} accounts")
    print(f"  Total batches: {num_batches}")
    
    return batch_size, num_batches, total_mem_per_account, lookup_overhead


class ProcessBatchResult(TypedDict):
    """Typed result from process_batch function."""
    batch_reserves: np.ndarray
    batch_capital: np.ndarray
    batch_reserves_5chocs: np.ndarray
    batch_capital_5chocs: np.ndarray
    ext_debug: Optional[np.ndarray]  # Debug output from external kernel
    int_debug: Optional[np.ndarray]  # Debug output from internal kernel


def check_gpu_memory(batch_size: int, mem_per_account: float, batch_idx: int = 0):
    """Log GPU memory status and raise if insufficient for batch."""
    try:
        free_mem, _ = cuda.current_context().get_memory_info()
        estimated_mem = batch_size * mem_per_account
        logger.info(f"  Free GPU memory: {free_mem / 1024 ** 3:.2f} GB")
        logger.info(f"  Estimated batch memory: {estimated_mem / 1024 ** 3:.2f} GB")
        
        if estimated_mem > free_mem * MEMORY_BATCH_THRESHOLD:
            raise RuntimeError(
                f"Insufficient GPU memory for batch {batch_idx + 1}. "
                f"Need {estimated_mem / 1024**3:.2f} GB but only "
                f"{free_mem / 1024**3:.2f} GB available. "
                f"Try reducing batch size or number of scenarios."
            )
    except NotImplementedError:
        pass


def process_batch(
    batch_account_data: np.ndarray,
    nb_ext_scenarios: int,
    nb_an_projection: int,
    nb_int_scenarios: int,
    total_mem_per_account: float,
    threads_per_block: tuple,
    gpu_lookups: dict,
    batch_idx: int = 0,
    num_batches: int = 1,
    debug_account: int = -1,
    debug_scenario: int = -1,
    debug_year: int = -1,
    debug_month: int = -1,
    debug_int_scenario: int = -1,
    debug_int_year: int = -1,
) -> ProcessBatchResult:
    """
    Process a single batch through both kernels.
    
    Args:
        batch_account_data: 2D array of account data for this batch (n_batch_accounts, n_features)
        nb_ext_scenarios: Number of external scenarios
        nb_an_projection: Number of projection years
        nb_int_scenarios: Number of internal scenarios
        total_mem_per_account: Estimated GPU memory per account (bytes)
        threads_per_block: CUDA thread block dimensions for Kernel A
        gpu_lookups: Dictionary of GPU device arrays with lookup tables
        batch_idx: Index of current batch (for logging)
        num_batches: Total number of batches (for logging)
        debug_account: Account index to debug (-1 = disabled)
        debug_scenario: External scenario index to debug (-1 = disabled)
        debug_year: Year (an_eval) to debug (-1 = disabled)
        debug_month: Month (mois_eval) to debug (-1 = disabled)
        debug_int_scenario: Internal scenario to debug (-1 = disabled)
        debug_int_year: Internal year to debug (-1 = disabled)
    
    Returns:
        ProcessBatchResult with batch results
    """
    batch_start = datetime.now()
    current_batch_size = len(batch_account_data)
    
    logger.info(f"\n--- Batch {batch_idx + 1}/{num_batches} ({current_batch_size} accounts) ---")
    check_gpu_memory(current_batch_size, total_mem_per_account, batch_idx)
    
    # Prepare batch data
    batch_account_data_contiguous = np.ascontiguousarray(batch_account_data)
    d_batch_accounts = cuda.to_device(batch_account_data_contiguous)
    
    # Allocate tensors
    try:
        d_states = cuda.device_array(
            (current_batch_size, nb_ext_scenarios, nb_an_projection, STATE_SIZE),
            dtype=np.float32
        )
        d_cashflows = cuda.device_array(
            (current_batch_size, nb_ext_scenarios, nb_an_projection, 1),
            dtype=np.float32
        )
        d_metrics = cuda.device_array(
            (current_batch_size, nb_ext_scenarios, nb_an_projection, NUM_CHOCS, METRICS_OUTPUT_SIZE),
            dtype=np.float32
        )
        
        # Allocate debug arrays (always allocate, use -1 flags to disable)
        enable_ext_debug = debug_account >= 0 or debug_scenario >= 0 or debug_year >= 0 or debug_month >= 0
        enable_int_debug = enable_ext_debug  # Internal debug only if external debug is enabled
        
        if enable_ext_debug:
            logger.info(f"  Debug mode: account={debug_account}, scenario={debug_scenario}, year={debug_year}, month={debug_month}")
        if enable_int_debug:
            logger.info(f"  Internal debug: int_scenario={debug_int_scenario}, int_year={debug_int_year}")
        
        # Always allocate debug arrays (kernel uses -1 flags to skip writing)
        d_ext_debug = cuda.device_array((EXT_DEBUG_SIZE,), dtype=np.float32)
        d_int_debug = cuda.device_array((NUM_CHOCS, INT_DEBUG_SIZE), dtype=np.float32)
    except Exception as e:
        raise RuntimeError(
            f"Failed to allocate GPU memory for batch {batch_idx+1}. "
            f"Try reducing --max-accounts or --ext-scenarios. Original error: {e}"
        )
    
    # === KERNEL A: EXTERNAL GENERATOR ===
    logger.info("  Launching Kernel A (External Generator)...")
    blocks_x = (current_batch_size + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_y = (nb_ext_scenarios + threads_per_block[1] - 1) // threads_per_block[1]
    grid_A = (blocks_x, blocks_y)
    
    kernel_a_start = datetime.now()
    external_generator_kernel[grid_A, threads_per_block](
        d_batch_accounts,
        nb_ext_scenarios, nb_an_projection,
        CONFIG['FREQ_EVAL'],
        gpu_lookups['mortality'],
        gpu_lookups['returns'],
        gpu_lookups['lapse'],
        gpu_lookups['policy'],
        gpu_lookups['commission'],
        d_states,
        d_cashflows,
        d_ext_debug,
        debug_account,
        debug_scenario,
        debug_year,
        debug_month,
    )
    cuda.synchronize()
    kernel_a_time = (datetime.now() - kernel_a_start).total_seconds()
    logger.info(f"  Kernel A complete: {kernel_a_time:.2f}s")
    
    # === KERNEL B: NESTED VALUATOR WITH 5 CHOCS ===
    logger.info(f"  Launching Kernel B (Five Chocs Nested Valuator)...")
    total_nodes = current_batch_size * nb_ext_scenarios * nb_an_projection
    threads_per_block_B = DEFAULT_THREADS_PER_BLOCK_1D
    blocks_B = (total_nodes + threads_per_block_B - 1) // threads_per_block_B
    
    kernel_b_start = datetime.now()
    
    nested_valuation_kernel_five_chocs[blocks_B, threads_per_block_B](
        d_states,
        d_batch_accounts,
        nb_int_scenarios,
        nb_an_projection,
        gpu_lookups['rn_returns'],
        gpu_lookups['mortality'],
        d_metrics,
        d_int_debug,
        debug_int_scenario,
        debug_int_year,
        debug_account,
        debug_scenario,
        debug_year,
    )
    cuda.synchronize()
    
    kernel_b_time = (datetime.now() - kernel_b_start).total_seconds()
    logger.info(f"  Kernel B complete: {kernel_b_time:.2f}s")
    
    # Copy results back
    logger.info("  Copying results to CPU...")
    h_metrics = d_metrics.copy_to_host()
    
    # Copy debug arrays if enabled
    h_ext_debug = None
    h_int_debug = None
    if enable_ext_debug:
        logger.info("  Copying external debug output to CPU...")
        h_ext_debug = d_ext_debug.copy_to_host()
    if enable_int_debug:
        logger.info("  Copying internal debug output to CPU...")
        h_int_debug = d_int_debug.copy_to_host()
    
    # Process metrics
    batch_reserves_5chocs = h_metrics[:, :, :, :, METRICS_RESERVE_IDX].mean(axis=(1, 2))
    batch_capital_5chocs = h_metrics[:, :, :, :, METRICS_CAPITAL_IDX].mean(axis=(1, 2))
    batch_reserves = batch_reserves_5chocs[:, 0]
    batch_capital = batch_capital_5chocs[:, 0]
    
    # Cleanup
    del d_batch_accounts, d_states, d_cashflows, d_metrics, d_ext_debug, d_int_debug
    cuda.synchronize()
    del h_metrics
    gc.collect()
    
    try:
        import rmm
        rmm.mr.get_current_device_resource().deallocate(0, 0)
    except (ImportError, AttributeError):
        pass
    
    batch_time = (datetime.now() - batch_start).total_seconds()
    logger.info(f"  Batch complete: {batch_time:.2f}s (KernelA: {kernel_a_time:.2f}s, KernelB: {kernel_b_time:.2f}s)")
    
    return {
        'batch_reserves': batch_reserves,
        'batch_capital': batch_capital,
        'batch_reserves_5chocs': batch_reserves_5chocs,
        'batch_capital_5chocs': batch_capital_5chocs,
        'ext_debug': h_ext_debug,
        'int_debug': h_int_debug,
    }


def create_results_dataframes(
    population_ids: np.ndarray,
    all_reserves: list,
    all_capital: list,
    all_reserves_5chocs: list,
    all_capital_5chocs: list,
    n_accounts: int
):
    """
    Create results DataFrames from accumulated batch results.
    
    Returns:
        Tuple of (results_df, results_5chocs_df, sensitivities_df)
    """
    results_df = pd.DataFrame({
        'ID_COMPTE': population_ids[:n_accounts],
        'RESERVE_BE': all_reserves,
        'CAPITAL_REQ': all_capital,
        'SCR': [cap - res for res, cap in zip(all_reserves, all_capital)]
    })
    
    results_5chocs_df = None
    sensitivities_df = None
    
    if all_reserves_5chocs:
        all_reserves_5chocs_array = np.array(all_reserves_5chocs)
        all_capital_5chocs_array = np.array(all_capital_5chocs)
        choc_rows = []
        
        for acc_idx in range(n_accounts):
            account_id = population_ids[acc_idx]
            for choc_idx, choc_name in enumerate(CHOC_NAMES):
                reserve = all_reserves_5chocs_array[acc_idx, choc_idx]
                capital = all_capital_5chocs_array[acc_idx, choc_idx]
                choc_rows.append({
                    'ID_COMPTE': account_id,
                    'CHOC_TYPE': choc_name,
                    'CHOC_IDX': choc_idx,
                    'RESERVE_BE': reserve,
                    'CAPITAL_REQ': capital,
                    'SCR': capital - reserve
                })
        
        results_5chocs_df = pd.DataFrame(choc_rows)
        
        base_reserves = all_reserves_5chocs_array[:, 0]
        base_capital = all_capital_5chocs_array[:, 0]
        
        sensitivities_df = pd.DataFrame({
            'ID_COMPTE': population_ids[:n_accounts],
            'DELTA_SP500_RESERVE': all_reserves_5chocs_array[:, 1] - base_reserves,
            'DELTA_TSX_RESERVE': all_reserves_5chocs_array[:, 2] - base_reserves,
            'DELTA_EAFE_RESERVE': all_reserves_5chocs_array[:, 3] - base_reserves,
            'DELTA_DEX_RESERVE': all_reserves_5chocs_array[:, 4] - base_reserves,
            'DELTA_SP500_CAPITAL': all_capital_5chocs_array[:, 1] - base_capital,
            'DELTA_TSX_CAPITAL': all_capital_5chocs_array[:, 2] - base_capital,
            'DELTA_EAFE_CAPITAL': all_capital_5chocs_array[:, 3] - base_capital,
            'DELTA_DEX_CAPITAL': all_capital_5chocs_array[:, 4] - base_capital,
        })
        
        logger.info(f"Created 5 chocs results with {len(results_5chocs_df)} rows and sensitivities for {len(sensitivities_df)} accounts")
    
    return results_df, results_5chocs_df, sensitivities_df


def save_results(
    output_path: Path,
    results_df: pd.DataFrame,
    results_5chocs_df: Optional[pd.DataFrame],
    sensitivities_df: Optional[pd.DataFrame],
    n_accounts: int,
    ext_debug: Optional[np.ndarray] = None,
    int_debug: Optional[np.ndarray] = None,
    debug_params: Optional[dict] = None,
    flux_projetes_periods: Optional[pd.DataFrame] = None,
):
    """
    Save all results (final simulation results and debug output) to CSV files.
    
    Args:
        output_path: Directory to save CSV files
        results_df: Main results DataFrame with reserves/capital per account
        results_5chocs_df: Optional DataFrame with 5 chocs results
        sensitivities_df: Optional DataFrame with sensitivities/Greeks
        n_accounts: Number of accounts processed
        ext_debug: Optional external kernel debug array (EXT_DEBUG_SIZE,) - single row
        int_debug: Optional internal kernel debug array (NUM_CHOCS, INT_DEBUG_SIZE) - one row per choc
        debug_params: Optional dictionary with debug filter parameters for context
    
    Returns:
        Dictionary containing all created DataFrames:
        - 'saved_files': List of saved file names
        - 'vp_flux_total': Portfolio totals DataFrame
        - 'chocs_summary': 5 chocs summary DataFrame (if results_5chocs_df provided)
        - 'ext_debug_df': External kernel debug DataFrame (if ext_debug provided)
        - 'int_debug_df': Internal kernel debug DataFrame (if int_debug provided)
    """
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Column names for debug CSV files
    EXT_DEBUG_COLUMNS = [
        'VM', 'AGE', 'QX', 'LAPSE_TOT', 'LAPSE_PART', 'TX_SURVIE',
        'FORWARD_RATE', 'REND_SP500', 'REND_TSX', 'REND_EAFE', 'REND_DEX',
        'RETRAIT', 'PREST_DECES', 'PRIMES_GARANTIES', 'VM_VG_RATIO'
    ]
    
    INT_DEBUG_COLUMNS = [
        'START_VM', 'VM_CHOC', 'AVG_PV_FLUX', 'RESERVE', 'CAPITAL',
        'START_TX_SURVIE', 'START_AGE',
        # Values captured at specific internal scenario/year
        'INT_CURR_VM', 'INT_FEES', 'INT_PV_PATH', 'INT_R_PORTFOLIO', 'INT_FWD_RATE'
    ]
    
    print("\n" + "=" * 80)
    print("SAVING OUTPUT FILES")
    print("=" * 80)
    
    saved_files = []
    chocs_summary_df = None
    ext_debug_df = None
    int_debug_df = None
    flux_projetes_df = None
    
    # ===========================================
    # 1. FINAL SIMULATION RESULTS
    # ===========================================
    
    # 1a. VP_FLUX_TOTAL_GPU.csv - Portfolio totals
    vp_flux_total_path = output_path / "VP_FLUX_TOTAL_GPU.csv"
    vp_flux_total_df = pd.DataFrame({
        'CATEGORIE': ['TOTAL'],
        'VP_RESERVE_BE': [results_df['RESERVE_BE'].sum()],
        'VP_CAPITAL_REQ': [results_df['CAPITAL_REQ'].sum()],
        'VP_SCR': [results_df['SCR'].sum()],
        'AVG_RESERVE_BE': [results_df['RESERVE_BE'].mean()],
        'AVG_CAPITAL_REQ': [results_df['CAPITAL_REQ'].mean()],
        'AVG_SCR': [results_df['SCR'].mean()],
        'N_ACCOUNTS': [len(results_df)]
    })
    vp_flux_total_df.to_csv(vp_flux_total_path, index=False, sep=';')
    print(f"✓ Saved VP_FLUX_TOTAL_GPU.csv")
    print(f"  Total Reserve (BE): ${vp_flux_total_df['VP_RESERVE_BE'].iloc[0]:,.2f}")
    print(f"  Total Capital Req:  ${vp_flux_total_df['VP_CAPITAL_REQ'].iloc[0]:,.2f}")
    print(f"  Total SCR:          ${vp_flux_total_df['VP_SCR'].iloc[0]:,.2f}")
    saved_files.append("VP_FLUX_TOTAL_GPU.csv (portfolio totals)")

    if flux_projetes_periods is not None and len(flux_projetes_periods) > 0:
        flux_cols = [
            'AN_EVAL', 'MOIS_EVAL',
            'PRIMES_GARANTIES', 'PREST_DECES', 'PREST_ECH', 'PREST_MRV',
            'FRAIS_ACQUIS', 'COMM_VENTE', 'PRIMES_VARIABLES',
            'FRAIS_FIXES', 'HON_GEST', 'COMM_MAINTIEN',
            'VALEUR_MARCHANDE', 'PASSIF_REDRESSE',
            'COUSSIN_CREDIT', 'COUSSIN_MARCHE', 'COUSSIN_DEPENSE',
            'COUSSIN_DECHEANCE', 'COUSSIN_MORTALITE', 'COUSSIN_DEPOT'
        ]

        flux_projetes_df = flux_projetes_periods.copy()
        for key in ('AN_EVAL', 'MOIS_EVAL'):
            if key not in flux_projetes_df.columns:
                flux_projetes_df[key] = 0

        for col in flux_cols:
            if col not in flux_projetes_df.columns:
                if col in ('AN_EVAL', 'MOIS_EVAL'):
                    continue
                flux_projetes_df[col] = 0.0

        flux_projetes_df = flux_projetes_df[flux_cols]
        flux_projetes_path = output_path / "FLUX_PROJETES_GPU.csv"
        flux_projetes_df.to_csv(flux_projetes_path, index=False, sep=';')
        print(f"✓ Saved FLUX_PROJETES_GPU.csv")
        saved_files.append("FLUX_PROJETES_GPU.csv (external loop logs)")
    
    # 1b. Five Chocs Results
    if results_5chocs_df is not None:
        print(f"\n✓ [FIVE_CHOCS] Saving five chocs results...")
        
        chocs_detailed_path = output_path / "VP_FLUX_5CHOCS_DETAILED_GPU.csv"
        results_5chocs_df.to_csv(chocs_detailed_path, index=False, sep=';')
        print(f"  Saved VP_FLUX_5CHOCS_DETAILED_GPU.csv")
        print(f"  Contains {len(results_5chocs_df)} rows (5 chocs × {n_accounts} accounts)")
        
        sensitivities_path = output_path / "VP_FLUX_SENSITIVITIES_GPU.csv"
        sensitivities_df.to_csv(sensitivities_path, index=False, sep=';')
        print(f"  Saved VP_FLUX_SENSITIVITIES_GPU.csv")
        print(f"  Contains {len(sensitivities_df)} rows with Greeks/Deltas")
        
        chocs_summary_df = results_5chocs_df.groupby('CHOC_TYPE').agg({
            'RESERVE_BE': ['sum', 'mean'],
            'CAPITAL_REQ': ['sum', 'mean'], 
            'SCR': ['sum', 'mean']
        }).round(2)
        chocs_summary_df.columns = ['_'.join(col).strip() for col in chocs_summary_df.columns]
        chocs_summary_df = chocs_summary_df.reset_index()
        
        chocs_summary_path = output_path / "VP_FLUX_5CHOCS_SUMMARY_GPU.csv"
        chocs_summary_df.to_csv(chocs_summary_path, index=False, sep=';')
        print(f"  Saved VP_FLUX_5CHOCS_SUMMARY_GPU.csv")
        
        print(f"\n  Key Portfolio Sensitivities (Total):")
        total_sensitivities = sensitivities_df.sum()
        print(f"    SP500 Delta (Reserve): ${total_sensitivities['DELTA_SP500_RESERVE']:,.2f}")
        print(f"    TSX Delta (Reserve):   ${total_sensitivities['DELTA_TSX_RESERVE']:,.2f}")
        print(f"    EAFE Delta (Reserve):  ${total_sensitivities['DELTA_EAFE_RESERVE']:,.2f}")
        print(f"    DEX Delta (Reserve):   ${total_sensitivities['DELTA_DEX_RESERVE']:,.2f}")
        
        saved_files.extend([
            "VP_FLUX_5CHOCS_DETAILED_GPU.csv (5 chocs × accounts)",
            "VP_FLUX_SENSITIVITIES_GPU.csv (Greeks/Deltas per account)",
            "VP_FLUX_5CHOCS_SUMMARY_GPU.csv (aggregated by choc type)"
        ])
    
    # ===========================================
    # 2. DEBUG OUTPUT (if enabled)
    # ===========================================
    
    if ext_debug is not None:
        print(f"\n✓ [DEBUG] Saving external kernel debug output...")
        
        # Create single-row DataFrame with debug filter context
        row = {}
        if debug_params:
            row['DEBUG_ACCOUNT'] = debug_params.get('account', -1)
            row['DEBUG_SCENARIO'] = debug_params.get('scenario', -1)
            row['DEBUG_YEAR'] = debug_params.get('year', -1)
            row['DEBUG_MONTH'] = debug_params.get('month', -1)
        
        for col_idx, col_name in enumerate(EXT_DEBUG_COLUMNS):
            row[col_name] = ext_debug[col_idx]
        
        ext_debug_df = pd.DataFrame([row])
        ext_debug_path = output_path / "DEBUG_EXTERNAL_KERNEL.csv"
        ext_debug_df.to_csv(ext_debug_path, index=False, sep=';')
        print(f"  Saved DEBUG_EXTERNAL_KERNEL.csv (1 row)")
        if debug_params:
            print(f"  Filter: account={debug_params.get('account', -1)}, scenario={debug_params.get('scenario', -1)}, year={debug_params.get('year', -1)}, month={debug_params.get('month', -1)}")
        saved_files.append("DEBUG_EXTERNAL_KERNEL.csv (external kernel debug)")
    
    if int_debug is not None:
        print(f"\n✓ [DEBUG] Saving internal kernel debug output...")
        n_chocs = int_debug.shape[0]
        
        rows = []
        for choc_idx in range(n_chocs):
            choc_name = CHOC_NAMES[choc_idx] if choc_idx < len(CHOC_NAMES) else f"CHOC_{choc_idx}"
            row = {
                'CHOC_IDX': choc_idx,
                'CHOC_NAME': choc_name,
            }
            if debug_params:
                row['DEBUG_INT_SCENARIO'] = debug_params.get('int_scenario', -1)
                row['DEBUG_INT_YEAR'] = debug_params.get('int_year', -1)
            
            for col_idx, col_name in enumerate(INT_DEBUG_COLUMNS):
                row[col_name] = int_debug[choc_idx, col_idx]
            rows.append(row)
        
        int_debug_df = pd.DataFrame(rows)
        int_debug_path = output_path / "DEBUG_INTERNAL_KERNEL.csv"
        int_debug_df.to_csv(int_debug_path, index=False, sep=';')
        print(f"  Saved DEBUG_INTERNAL_KERNEL.csv ({len(int_debug_df)} rows - one per choc)")
        if debug_params:
            print(f"  Filter: int_scenario={debug_params.get('int_scenario', -1)}, int_year={debug_params.get('int_year', -1)}")
        saved_files.append("DEBUG_INTERNAL_KERNEL.csv (internal kernel debug)")
    
    # ===========================================
    # 3. SUMMARY
    # ===========================================
    
    print("\n" + "=" * 80)
    print("FILE SAVING SUMMARY")
    print("=" * 80)
    for idx, file_name in enumerate(saved_files, 1):
        print(f"  {idx}. {file_name}")
    print("=" * 80)
    
    # Build return dictionary with all created DataFrames
    result = {
        'saved_files': saved_files,
        'vp_flux_total': vp_flux_total_df,
        'chocs_summary': chocs_summary_df if results_5chocs_df is not None else None,
        'ext_debug_df': ext_debug_df if ext_debug is not None else None,
        'int_debug_df': int_debug_df if int_debug is not None else None,
        'flux_projetes_df': flux_projetes_df,
    }
    
    return result


def create_all_lookup_tables(data: dict, nb_int_scenarios: int, nb_an_projection: int):
    """
    Create all CPU lookup tables from loaded data.
    
    Args:
        data: Dictionary containing loaded DataFrames (mortalite, rendements, etc.)
        nb_int_scenarios: Number of internal scenarios for risk-neutral tables
        nb_an_projection: Number of projection years
    
    Returns:
        Dictionary containing all lookup tables (CPU numpy arrays)
    """
    print("\nCreating CPU lookup tables...")
    
    lookups = {}
    
    lookups['mortality'] = create_gpu_mortality_lookup(data['mortalite'])
    (lookups['forward_rate'], lookups['ajust_forward'], lookups['rend_dex'], 
     lookups['rend_mm'], lookups['rend_tsx'], lookups['rend_sp500'], 
     lookups['rend_eafe']) = create_gpu_returns_lookup(data['rendements'])
    
    lookups['min_ferr'] = create_gpu_min_ferr_lookup(data['min_ferr'])
    lookups['lapse_part_min'], lookups['lapse_part_max'] = create_gpu_lapse_part_lookup(data['tx_lapse_part'])
    lookups['lapse_tot_min'], lookups['lapse_tot_max'], lookups['lapse_tot_fact'] = create_gpu_lapse_tot_lookup(data['tx_lapse_tot'])
    (lookups['deposits_pc'], lookups['deposits_var'], lookups['deposits_age_max'],
     lookups['deposits_i_even']) = create_gpu_deposits_lookup(data['depots_futurs'])
    lookups['fees'] = create_gpu_fees_lookup(data['frais_admin'])
    (lookups['acq_vente_rf'], lookups['acq_vente_ac'], lookups['acq_maintien_rf'], 
     lookups['acq_maintien_ac'], lookups['acq_frais_ac'], 
     lookups['acq_frais_rf']) = create_gpu_acquisition_lookup(data['acquisition'])
    
    print("✓ All CPU lookup tables created")
    
    # Create risk-neutral scenario tables
    print("\nCreating risk-neutral scenario tables...")
    lookups['rn_forward_rate'] = np.full((nb_int_scenarios, nb_an_projection), RN_DEFAULT_FORWARD_RATE, dtype=np.float32)
    lookups['rn_rend_dex'] = np.full((nb_int_scenarios, nb_an_projection), RN_DEFAULT_REND_DEX, dtype=np.float32)
    lookups['rn_rend_mm'] = np.full((nb_int_scenarios, nb_an_projection), RN_DEFAULT_REND_MM, dtype=np.float32)
    lookups['rn_rend_tsx'] = np.full((nb_int_scenarios, nb_an_projection), RN_DEFAULT_REND_TSX, dtype=np.float32)
    lookups['rn_rend_sp500'] = np.full((nb_int_scenarios, nb_an_projection), RN_DEFAULT_REND_SP500, dtype=np.float32)
    lookups['rn_rend_eafe'] = np.full((nb_int_scenarios, nb_an_projection), RN_DEFAULT_REND_EAFE, dtype=np.float32)
    
    print("✓ Risk-neutral tables created")
    
    return lookups


def copy_lookups_to_gpu(lookups: dict):
    """
    Copy all CPU lookup tables to GPU memory.
    
    Args:
        lookups: Dictionary of CPU numpy arrays from create_all_lookup_tables()
    
    Returns:
        Dictionary containing grouped GPU device arrays as tuples
    """
    print("\nCopying lookup tables to GPU...")
    
    gpu_lookups = {}
    
    # Mortality table
    gpu_lookups['mortality'] = cuda.to_device(lookups['mortality'])
    
    # Returns lookups (7 arrays): forward_rate, ajust_forward, rend_dex, rend_mm, rend_tsx, rend_sp500, rend_eafe
    gpu_lookups['returns'] = (
        cuda.to_device(lookups['forward_rate']),
        cuda.to_device(lookups['ajust_forward']),
        cuda.to_device(lookups['rend_dex']),
        cuda.to_device(lookups['rend_mm']),
        cuda.to_device(lookups['rend_tsx']),
        cuda.to_device(lookups['rend_sp500']),
        cuda.to_device(lookups['rend_eafe']),
    )
    
    # Lapse lookups (6 arrays): min_ferr, lapse_part_min, lapse_part_max, lapse_tot_min, lapse_tot_max, lapse_tot_fact
    gpu_lookups['lapse'] = (
        cuda.to_device(lookups['min_ferr']),
        cuda.to_device(lookups['lapse_part_min']),
        cuda.to_device(lookups['lapse_part_max']),
        cuda.to_device(lookups['lapse_tot_min']),
        cuda.to_device(lookups['lapse_tot_max']),
        cuda.to_device(lookups['lapse_tot_fact']),
    )
    
    # Policy lookups (5 arrays): deposits_pc, deposits_var, deposits_age_max, deposits_i_even, fees
    gpu_lookups['policy'] = (
        cuda.to_device(lookups['deposits_pc']),
        cuda.to_device(lookups['deposits_var']),
        cuda.to_device(lookups['deposits_age_max']),
        cuda.to_device(lookups['deposits_i_even']),
        cuda.to_device(lookups['fees']),
    )
    
    # Commission lookups (6 arrays): acq_vente_rf, acq_vente_ac, acq_maintien_rf, acq_maintien_ac, acq_frais_ac, acq_frais_rf
    gpu_lookups['commission'] = (
        cuda.to_device(lookups['acq_vente_rf']),
        cuda.to_device(lookups['acq_vente_ac']),
        cuda.to_device(lookups['acq_maintien_rf']),
        cuda.to_device(lookups['acq_maintien_ac']),
        cuda.to_device(lookups['acq_frais_ac']),
        cuda.to_device(lookups['acq_frais_rf']),
    )
    
    # Risk-neutral returns (6 arrays): rn_forward_rate, rn_rend_dex, rn_rend_mm, rn_rend_tsx, rn_rend_sp500, rn_rend_eafe
    gpu_lookups['rn_returns'] = (
        cuda.to_device(lookups['rn_forward_rate']),
        cuda.to_device(lookups['rn_rend_dex']),
        cuda.to_device(lookups['rn_rend_mm']),
        cuda.to_device(lookups['rn_rend_tsx']),
        cuda.to_device(lookups['rn_rend_sp500']),
        cuda.to_device(lookups['rn_rend_eafe']),
    )
    
    print("✓ Lookup tables on GPU")
    
    return gpu_lookups


def run_projection_gpu_nested(
        data_path: Path, 
        output_path: Path, 
        nb_an_projection: int,
        nb_ext_scenarios: int,
        nb_int_scenarios: int,
        shock_capital_pct: float = 0.35,
        max_accounts: int = None,
        threads_per_block=(16, 16),
        use_pinned_memory=True,
        population_path: Optional[Path] = None,
        mortalite_path: Optional[Path] = None,
        rendements_path: Optional[Path] = None,
        depots_futurs_path: Optional[Path] = None,
        frais_admin_path: Optional[Path] = None,
        min_ferr_path: Optional[Path] = None,
        tx_lapse_part_path: Optional[Path] = None,
        tx_lapse_tot_path: Optional[Path] = None,
        acquisition_path: Optional[Path] = None,
        coussins_escap_path: Optional[Path] = None,
        progress_callback: Optional[callable] = None,
        debug_account: int = -1,
        debug_scenario: int = -1,
        debug_year: int = -1,
        debug_month: int = -1,
        debug_int_scenario: int = -1,
        debug_int_year: int = -1,
        debug_only: bool = False):
    """
    Run GPU-accelerated nested stochastic projection using Two-Pass architecture.
    
    Architecture:
    - Kernel A (Generator): Runs external scenarios, outputs state tensors to VRAM
    - Kernel B (Valuator): Reads states, runs internal scenarios with 5 chocs, outputs reserves & capital
    
    Args:
        debug_only: If True and debug_account >= 0, only process the single account 
                   specified by debug_account (filters population to that account only).
    """
    start_time = datetime.now()
    print(f"Starting NESTED STOCHASTIC GPU projection at {start_time}")
    print("=" * 80)
    print(f"Architecture: Two-Pass (Generator → Valuator with 5 Chocs)")
    print(f"External scenarios: {nb_ext_scenarios}")
    print(f"Internal scenarios per node: {nb_int_scenarios}")
    print(f"Capital shock: {shock_capital_pct*100:.1f}%")
    enable_debug = debug_account >= 0 or debug_scenario >= 0 or debug_year >= 0 or debug_month >= 0
    if enable_debug:
        print(f"Debug mode: ENABLED (account={debug_account}, scenario={debug_scenario}, year={debug_year}, month={debug_month})")
        print(f"  Internal debug: int_scenario={debug_int_scenario}, int_year={debug_int_year}")
        if debug_only and debug_account >= 0:
            print(f"  DEBUG_ONLY: Will process ONLY account {debug_account}")
    else:
        print(f"Debug mode: disabled")
    print("=" * 80)
    
    # Initialize GPU
    initialize_gpu()

    # Update config
    CONFIG['NB_AN_PROJECTION'] = nb_an_projection
    CONFIG['NB_SC'] = nb_ext_scenarios

    # Load data
    print("\nLoading data files...")
    data = load_all_data(data_path,
                         population_path=population_path,
                         mortalite_path=mortalite_path,
                         rendements_path=rendements_path,
                         depots_futurs_path=depots_futurs_path,
                         frais_admin_path=frais_admin_path,
                         min_ferr_path=min_ferr_path,
                         tx_lapse_part_path=tx_lapse_part_path,
                         tx_lapse_tot_path=tx_lapse_tot_path,
                         acquisition_path=acquisition_path,
                         coussins_escap_path=coussins_escap_path)
    print("✓ Data loaded successfully")

    flux_projetes_periods = None
    if 'rendements' in data and data['rendements'] is not None:
        try:
            flux_projetes_periods = (
                data['rendements'][['AN_EVAL', 'MOIS_EVAL']]
                .drop_duplicates()
                .sort_values(['AN_EVAL', 'MOIS_EVAL'])
                .reset_index(drop=True)
            )
            if not ((flux_projetes_periods['AN_EVAL'] == 0) & (flux_projetes_periods['MOIS_EVAL'] == 12)).any():
                flux_projetes_periods = pd.concat(
                    [pd.DataFrame([{'AN_EVAL': 0, 'MOIS_EVAL': 12}]), flux_projetes_periods],
                    ignore_index=True,
                ).sort_values(['AN_EVAL', 'MOIS_EVAL']).reset_index(drop=True)
        except Exception:
            flux_projetes_periods = None

    # Filter to single account if debug_only mode
    if debug_only and debug_account >= 0:
        # Find the account by ID (assuming there's an ID column like 'NO_COMPTE' or index)
        pop_df = data['population']
        if 'NO_COMPTE' in pop_df.columns:
            filtered = pop_df[pop_df['NO_COMPTE'] == debug_account]
        else:
            # Fall back to using index/row position
            if debug_account < len(pop_df):
                filtered = pop_df.iloc[[debug_account]]
            else:
                raise ValueError(f"debug_account {debug_account} is out of range (max: {len(pop_df)-1})")
        
        if len(filtered) == 0:
            raise ValueError(f"Account {debug_account} not found in population data")
        
        data['population'] = filtered.reset_index(drop=True)
        print(f"⚠️  DEBUG_ONLY: Filtered to single account {debug_account}")
        # Override max_accounts since we're only doing one
        max_accounts = None
    elif max_accounts:
        data['population'] = data['population'].head(max_accounts)

    n_accounts = len(data['population'])
    print(f"\nPreparing {n_accounts} accounts for GPU processing...")

    # Prepare account data
    all_account_data, _ = prepare_account_data(data['population'])
    print("✓ Account data prepared")

    # Create all CPU lookup tables
    lookups = create_all_lookup_tables(data, nb_int_scenarios, nb_an_projection)

    # Calculate batch size
    batch_size, num_batches, total_mem_per_account, _ = calculate_batch_size(
        n_accounts, nb_ext_scenarios, nb_an_projection, 
        nb_int_scenarios, all_account_data.shape[1]
    )

    # Copy lookup tables to GPU
    gpu_lookups = copy_lookups_to_gpu(lookups)

    # Process batches
    print("\n" + "=" * 80)
    print("RUNNING TWO-PASS NESTED STOCHASTIC PROJECTION")
    print("=" * 80)
    
    all_reserves = []
    all_capital = []
    all_reserves_5chocs = []
    all_capital_5chocs = []
    ext_debug_result = None
    int_debug_result = None
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_accounts)
        batch_account_data = all_account_data[start_idx:end_idx]
        
        # Adjust debug_account for batch offset (only debug if account is in this batch)
        batch_debug_account = -1
        if debug_account >= 0:
            if start_idx <= debug_account < end_idx:
                batch_debug_account = debug_account - start_idx
        
        batch_result = process_batch(
            batch_account_data=batch_account_data,
            nb_ext_scenarios=nb_ext_scenarios,
            nb_an_projection=nb_an_projection,
            nb_int_scenarios=nb_int_scenarios,
            total_mem_per_account=total_mem_per_account,
            threads_per_block=threads_per_block,
            gpu_lookups=gpu_lookups,
            batch_idx=i,
            num_batches=num_batches,
            debug_account=batch_debug_account,
            debug_scenario=debug_scenario,
            debug_year=debug_year,
            debug_month=debug_month,
            debug_int_scenario=debug_int_scenario,
            debug_int_year=debug_int_year,
        )
        
        # Accumulate results
        all_reserves.extend(batch_result['batch_reserves'])
        all_capital.extend(batch_result['batch_capital'])
        all_reserves_5chocs.extend(batch_result['batch_reserves_5chocs'])
        all_capital_5chocs.extend(batch_result['batch_capital_5chocs'])
        
        # Store debug output (only one batch will have it if account filter is used)
        if batch_result['ext_debug'] is not None:
            ext_debug_result = batch_result['ext_debug']
        if batch_result['int_debug'] is not None:
            int_debug_result = batch_result['int_debug']
        
        # Call progress callback if provided
        if progress_callback is not None:
            progress_callback(i + 1, num_batches)
    
    # Create results DataFrames
    results_df, results_5chocs_df, sensitivities_df = create_results_dataframes(
        population_ids=data['population']['ID_COMPTE'].values,
        all_reserves=all_reserves,
        all_capital=all_capital,
        all_reserves_5chocs=all_reserves_5chocs,
        all_capital_5chocs=all_capital_5chocs,
        n_accounts=n_accounts
    )
    
    # Print summary
    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()
    
    print("\n" + "=" * 80)
    print("NESTED STOCHASTIC PROJECTION COMPLETE")
    print("=" * 80)
    print(f"Total time: {total_duration:.2f}s ({total_duration/60:.2f} minutes)")
    print(f"Accounts processed: {n_accounts}")
    print(f"External scenarios: {nb_ext_scenarios}")
    print(f"Internal scenarios per node: {nb_int_scenarios}")
    print(f"Total nested simulations: {n_accounts * nb_ext_scenarios * nb_an_projection * nb_int_scenarios:,}")
    print(f"\nResults Summary:")
    print(f"  Total Best Estimate Reserve: ${results_df['RESERVE_BE'].sum():,.2f}")
    print(f"  Total Capital Requirement:   ${results_df['CAPITAL_REQ'].sum():,.2f}")
    print(f"  Total SCR (Capital - Reserve): ${results_df['SCR'].sum():,.2f}")
    print(f"\n  Average per account:")
    print(f"    Reserve: ${results_df['RESERVE_BE'].mean():,.2f}")
    print(f"    Capital: ${results_df['CAPITAL_REQ'].mean():,.2f}")
    print(f"    SCR:     ${results_df['SCR'].mean():,.2f}")
    print("=" * 80)
    
    # Build debug params if debug is enabled
    debug_params = None
    if enable_debug:
        debug_params = {
            'account': debug_account,
            'scenario': debug_scenario,
            'year': debug_year,
            'month': debug_month,
            'int_scenario': debug_int_scenario,
            'int_year': debug_int_year,
        }
    
    # Save all results (including debug output if enabled)
    save_result = save_results(
        output_path=output_path,
        results_df=results_df,
        results_5chocs_df=results_5chocs_df,
        sensitivities_df=sensitivities_df,
        n_accounts=n_accounts,
        ext_debug=ext_debug_result,
        int_debug=int_debug_result,
        debug_params=debug_params,
        flux_projetes_periods=flux_projetes_periods,
    )
    
    return ProjectionResult(
        results=results_df,
        results_5chocs=results_5chocs_df,
        sensitivities=sensitivities_df if results_5chocs_df is not None else None,
        total_duration=total_duration,
        vp_flux_total=save_result['vp_flux_total'],
        chocs_summary=save_result['chocs_summary'],
        ext_debug_df=save_result['ext_debug_df'],
        int_debug_df=save_result['int_debug_df'],
        saved_files=save_result['saved_files'],
    )


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Run GPU-accelerated actuarial projections',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic nested stochastic
  python gpu.py --ext-scenarios 100 --int-scenarios 500 --max-accounts 1000
  
  # Full production run
  python gpu.py --ext-scenarios 1000 --int-scenarios 1000 --years 100
  
  # Debug specific account/scenario/year/month
  python gpu.py --max-accounts 10 --debug-account 0 --debug-scenario 0 --debug-year 5 --debug-month 12
        """
    )

    parser.add_argument('--max-accounts', type=int, default=2000,
                        help='Maximum number of accounts to process (for testing)')
    parser.add_argument('--years', type=int, default=100,
                        help='Number of years to project (default: 100)')

    # Nested mode parameters
    parser.add_argument('--ext-scenarios', type=int, default=100,
                        help='Number of external (real-world) scenarios for nested mode (default: 100)')
    parser.add_argument('--int-scenarios', type=int, default=100,
                        help='Number of internal (risk-neutral) scenarios per node for nested mode (default: 100)')
    parser.add_argument('--shock', type=float, default=0.35,
                        help='Capital shock percentage for nested mode (default: 0.35 = 35%%)')
    
    # Debug filter parameters
    parser.add_argument('--debug-account', type=int, default=-1,
                        help='Account index to debug (-1 = disabled)')
    parser.add_argument('--debug-scenario', type=int, default=-1,
                        help='External scenario index to debug (-1 = disabled)')
    parser.add_argument('--debug-year', type=int, default=-1,
                        help='Year (an_eval) to debug (-1 = disabled)')
    parser.add_argument('--debug-month', type=int, default=-1,
                        help='Month (mois_eval) to debug (-1 = disabled)')
    parser.add_argument('--debug-int-scenario', type=int, default=-1,
                        help='Internal scenario to debug (-1 = disabled)')
    parser.add_argument('--debug-int-year', type=int, default=-1,
                        help='Internal year to debug (-1 = disabled)')

    args = parser.parse_args()
    
    try:
        if not cuda.is_available():
            print("ERROR: CUDA is not available. Please check your GPU setup.")
            exit(1)

        print(f"CUDA Device: {cuda.get_current_device().name}")
        
        DATA_PATH = HERE.joinpath("data_in")
        OUTPUT_PATH = HERE.joinpath("data_out_gpu")

        print("\n" + "=" * 80)
        print("RUNNING NESTED STOCHASTIC MODE (Tier 2 & 3: Reserves & Capital)")
        print("=" * 80)

        results = run_projection_gpu_nested(
            data_path=DATA_PATH,
            output_path=OUTPUT_PATH,
            nb_an_projection=args.years,
            nb_ext_scenarios=args.ext_scenarios,
            nb_int_scenarios=args.int_scenarios,
            shock_capital_pct=args.shock,
            max_accounts=args.max_accounts,
            threads_per_block=(16, 16),
            debug_account=args.debug_account,
            debug_scenario=args.debug_scenario,
            debug_year=args.debug_year,
            debug_month=args.debug_month,
            debug_int_scenario=args.debug_int_scenario,
            debug_int_year=args.debug_int_year,
        )

        if results:
            print("\n" + "=" * 80)
            print("NESTED STOCHASTIC RESULTS")
            print("=" * 80)
            print("\nTop 10 accounts by SCR:")
            print(results.results.nlargest(10, 'SCR')[['ID_COMPTE', 'RESERVE_BE', 'CAPITAL_REQ', 'SCR']])

            print("\nSummary Statistics:")
            print(results.results[['RESERVE_BE', 'CAPITAL_REQ', 'SCR']].describe())

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()
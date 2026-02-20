"""
Low-memory GPU projection orchestration.

This module is a drop-in replacement for calculations/gpu.py that uses
in-kernel atomic aggregation (via kernels_lowmem.py) to eliminate the
massive cashflow output tensor.

Memory savings:
  Original d_cashflows: (batch × scenarios × months × 41) × 4 bytes
    e.g., 1000 × 50 × 1200 × 41 × 4 = 9.4 GB

  Low-mem replacement: three small tensors
    vp_per_account:   (1000 × 41 × 4)   = 164 KB
    flux_projete_agg: (1200 × 41 × 4)   = 197 KB
    mean_cashflows:   (1000 × 1200 × 41 × 4) = 188 MB  (optional)
    Total: ~188 MB worst case vs 9.4 GB = 98% reduction

Usage:
    python -m calculations.gpu_lowmem --outer-loop-only --ext-scenarios 50 --max-accounts 999999
"""
import os
import sys
import gc
import argparse
import math
from pathlib import Path
from datetime import datetime
from typing import Optional

os.environ['NUMBA_CUDA_ENABLE_PYNVJITLINK'] = '1'

import numpy as np
import pandas as pd
from numba import cuda

from calculations.utils import logger, CONFIG, load_all_data, prepare_account_data
from calculations.constants import (
    NUM_CHOCS, CHOC_NAMES, METRICS_RESERVE_IDX, METRICS_CAPITAL_IDX, METRICS_OUTPUT_SIZE,
    FLUX_COMP_IDX_SIZE, CF_OUT_IDX_SIZE,
    CF_OUT_IDX_VP_FRAIS_ACQUIS, CF_OUT_IDX_VP_COMM_VENTE, CF_OUT_IDX_VP_PRIMES_GARANTIES,
    CF_OUT_IDX_VP_PRIMES_VARIABLES, CF_OUT_IDX_VP_FRAIS_FIXES, CF_OUT_IDX_VP_HON_GEST,
    CF_OUT_IDX_VP_COMM_MAINTIEN, CF_OUT_IDX_VP_PREST_ECH, CF_OUT_IDX_VP_PREST_MRV,
    CF_OUT_IDX_VP_PREST_DECES,
    INT_TS_DEBUG_IDX_CURR_VM, INT_TS_DEBUG_IDX_FEES, INT_TS_DEBUG_IDX_PV_PATH,
    INT_TS_DEBUG_IDX_R_PORTFOLIO, INT_TS_DEBUG_IDX_FWD_RATE, INT_TS_DEBUG_IDX_DF,
    INT_TS_DEBUG_IDX_SIZE,
    LOOKUP_TABLE_OVERHEAD_MB, DEFAULT_GPU_MEMORY_GB,
    DEFAULT_THREADS_PER_BLOCK_1D,
)
from calculations.kernels_lowmem import (
    external_generator_kernel,
    nested_valuation_kernel_five_chocs,
    STATE_SIZE, EXT_DEBUG_SIZE, INT_DEBUG_SIZE,
)

# Import reusable components from the original gpu.py
from calculations.gpu import (
    ProjectionResult, ProcessBatchResult, KernelIncompatibilityError,
    initialize_gpu, check_gpu_memory,
    create_all_lookup_tables, copy_lookups_to_gpu,
    create_results_dataframes, save_results,
    write_cashflows_batch, write_vp_flux_compte_batch,
    accumulate_flux_projete, write_flux_projete,
    _device_array_cupy, _to_device_contiguous, _clear_cupy_refs,
)
from paths import HERE

# Low-memory kernel has 20 args (3 output arrays replacing the single d_cashflows)
EXPECTED_EXTERNAL_GENERATOR_ARGCOUNT = 20
EXPECTED_NESTED_VALUATION_FIVE_CHOCS_ARGCOUNT = 18


def validate_kernel_compatibility():
    """Validate that kernel signatures match expected argument counts."""
    kernel_a_argcount = external_generator_kernel.py_func.__code__.co_argcount
    if kernel_a_argcount != EXPECTED_EXTERNAL_GENERATOR_ARGCOUNT:
        raise KernelIncompatibilityError(
            f"Low-mem external_generator_kernel has {kernel_a_argcount} params "
            f"but gpu_lowmem expects {EXPECTED_EXTERNAL_GENERATOR_ARGCOUNT}. "
            "Check calculations/kernels_lowmem.py."
        )

    kernel_b_argcount = nested_valuation_kernel_five_chocs.py_func.__code__.co_argcount
    if kernel_b_argcount != EXPECTED_NESTED_VALUATION_FIVE_CHOCS_ARGCOUNT:
        raise KernelIncompatibilityError(
            f"nested_valuation_kernel_five_chocs has {kernel_b_argcount} params "
            f"but gpu_lowmem expects {EXPECTED_NESTED_VALUATION_FIVE_CHOCS_ARGCOUNT}. "
            "Check calculations/kernels.py."
        )


def calculate_batch_size(n_accounts: int, nb_ext_scenarios: int, nb_an_projection: int,
                         nb_int_scenarios: int, account_data_cols: int,
                         write_detailed_cashflows: bool = False,
                         run_nested_valuation: bool = True):
    """
    Calculate optimal batch size based on LOW-MEMORY requirements.

    Only counts memory for tensors that are actually allocated:
    - State tensor: only when run_nested_valuation=True
    - Mean cashflows: only when write_detailed_cashflows=True
    - Metrics tensor: only when run_nested_valuation=True
    """
    print("\n[LOW-MEM] Calculating memory requirements...")

    # State tensor: (Batch, Ext_Scenarios, Years, STATE_SIZE)  — only for Kernel B
    if run_nested_valuation:
        state_mem_per_account = nb_ext_scenarios * nb_an_projection * STATE_SIZE * 4
    else:
        # Outer-loop-only: tiny dummy (1,1,1,STATE_SIZE) shared across accounts
        state_mem_per_account = 0

    # No d_cashflows!  Replaced by three tiny/moderate tensors:
    # vp_per_account:   (Batch, CF_OUT_IDX_SIZE) — negligible
    vp_mem_per_account = CF_OUT_IDX_SIZE * 4

    # flux_projete_agg: (n_months, CF_OUT_IDX_SIZE) — shared across accounts, not per-account
    flux_projete_mem_total = nb_an_projection * 12 * CF_OUT_IDX_SIZE * 4

    # mean_cashflows:   (Batch, n_months, CF_OUT_IDX_SIZE) — only when enabled
    if write_detailed_cashflows:
        mean_cf_mem_per_account = nb_an_projection * 12 * CF_OUT_IDX_SIZE * 4
    else:
        mean_cf_mem_per_account = 0

    # Metrics tensor: (Batch, Ext_Scenarios, Years, NUM_CHOCS, METRICS_OUTPUT_SIZE) — only for Kernel B
    if run_nested_valuation:
        metrics_mem_per_account = nb_ext_scenarios * nb_an_projection * NUM_CHOCS * METRICS_OUTPUT_SIZE * 4
    else:
        metrics_mem_per_account = 0

    total_mem_per_account = (state_mem_per_account + vp_mem_per_account +
                             mean_cf_mem_per_account + metrics_mem_per_account +
                             account_data_cols * 4)

    original_cf_mem_per_account = nb_ext_scenarios * nb_an_projection * 12 * CF_OUT_IDX_SIZE * 4
    noncf_mem = total_mem_per_account if total_mem_per_account > 0 else 1
    savings_factor = original_cf_mem_per_account / noncf_mem

    # Estimate lookup table memory overhead
    lookup_overhead = 0
    lookup_overhead += 6 * nb_ext_scenarios * nb_an_projection * 12 * 4
    lookup_overhead += 6 * nb_int_scenarios * nb_an_projection * 4
    lookup_overhead += LOOKUP_TABLE_OVERHEAD_MB * 1024 ** 2
    lookup_overhead += flux_projete_mem_total  # shared tensor

    print(f"  State tensor per account: {state_mem_per_account / 1024 ** 2:.2f} MB")
    print(f"  Mean CF per account:      {mean_cf_mem_per_account / 1024 ** 2:.2f} MB")
    print(f"  VP per account:           {vp_mem_per_account / 1024:.2f} KB")
    print(f"  Total mem per account:    {total_mem_per_account / 1024 ** 2:.2f} MB")
    print(f"  Original CF per account:  {original_cf_mem_per_account / 1024 ** 2:.2f} MB")
    print(f"  Memory savings:           {savings_factor:.1f}x vs original")
    print(f"  Lookup table overhead:    {lookup_overhead / 1024 ** 2:.2f} MB")

    # Calculate available GPU memory
    try:
        free_mem, total_mem = cuda.current_context().get_memory_info()
        print(f"  GPU free memory:          {free_mem / 1024 ** 3:.2f} GB")
        print(f"  GPU total memory:         {total_mem / 1024 ** 3:.2f} GB")
        available_mem = max(0, (free_mem - lookup_overhead) * 0.70)
    except Exception as e:
        try:
            import cupy as cp
            mempool = cp.get_default_memory_pool()
            free_mem = mempool.free_bytes()
            total_mem = cp.cuda.Device().mem_info[1]
            if free_mem == 0:
                total_mem, used_mem = cp.cuda.Device().mem_info
                free_mem = total_mem
            print(f"  GPU free memory (cupy):   {free_mem / 1024 ** 3:.2f} GB")
            available_mem = max(0, (free_mem - lookup_overhead) * 0.70)
        except Exception:
            print(f"  Warning: Cannot query GPU memory ({e}), using conservative estimate")
            available_mem = max(0, DEFAULT_GPU_MEMORY_GB * 1024 ** 3 - lookup_overhead)

    batch_size = max(1, int(available_mem // total_mem_per_account))
    batch_size = min(batch_size, n_accounts)
    num_batches = (n_accounts + batch_size - 1) // batch_size

    print(f"  Batch size:               {batch_size} accounts")
    print(f"  Total batches:            {num_batches}")

    return batch_size, num_batches, total_mem_per_account, lookup_overhead


def process_batch(
        batch_account_data: np.ndarray,
        nb_ext_scenarios: int,
        nb_an_projection: int,
        nb_int_scenarios: int,
        shock_capital_pct: float,
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
        run_nested_valuation: bool = True,
        write_detailed_cashflows: bool = False,
) -> ProcessBatchResult:
    """
    Process a single batch using LOW-MEMORY kernel (no massive cashflow tensor).

    The external_generator_kernel accumulates results directly into three
    small output tensors via cuda.atomic.add, eliminating the need for the
    giant (batch × scenarios × months × 41) cashflow tensor.
    """
    batch_start = datetime.now()
    current_batch_size = len(batch_account_data)
    n_months = nb_an_projection * 12

    logger.info(f"\n--- [LOWMEM] Batch {batch_idx + 1}/{num_batches} ({current_batch_size} accounts) ---")
    check_gpu_memory(current_batch_size, total_mem_per_account, batch_idx)

    # Prepare batch data
    batch_account_data_contiguous = np.ascontiguousarray(batch_account_data)
    d_batch_accounts = _to_device_contiguous(batch_account_data_contiguous)

    try:
        # State tensor: full allocation for nested valuation, tiny dummy for outer-loop-only
        if run_nested_valuation:
            d_states = _device_array_cupy(
                (current_batch_size, nb_ext_scenarios, nb_an_projection, STATE_SIZE)
            )
        else:
            # Tiny dummy — kernel has bounds checks on all 3 dims so no OOB writes
            d_states = _to_device_contiguous(
                np.zeros((1, 1, 1, STATE_SIZE), dtype=np.float32)
            )

        # ============================================================
        # LOW-MEMORY OUTPUT TENSORS (replace the massive d_cashflows)
        # ============================================================

        # 1. VP per account: (batch, CF_OUT_IDX_SIZE) — sum over scenarios & months
        #    ~164 KB for 1000 accounts
        d_vp_per_account = _device_array_cupy(
            (current_batch_size, CF_OUT_IDX_SIZE)
        )
        # Zero-initialize (atomic adds accumulate into this)
        import cupy as cp
        cp.asarray(d_vp_per_account).fill(0.0)

        # 2. Flux projete aggregated: (n_months, CF_OUT_IDX_SIZE) — sum over accounts & scenarios
        #    ~197 KB for 1200 months
        d_flux_projete_agg = _device_array_cupy(
            (n_months, CF_OUT_IDX_SIZE)
        )
        cp.asarray(d_flux_projete_agg).fill(0.0)

        # 3. Mean cashflows per account/month: (batch, n_months, CF_OUT_IDX_SIZE)
        #    ~188 MB for 1000 accounts × 1200 months — still much smaller than 9.4 GB
        #    Or tiny dummy (1, 1, 1) if not needed
        if write_detailed_cashflows:
            d_mean_cashflows = _device_array_cupy(
                (current_batch_size, n_months, CF_OUT_IDX_SIZE)
            )
            cp.asarray(d_mean_cashflows).fill(0.0)
            mem_saved_msg = f"(mean_cf allocated: {current_batch_size * n_months * CF_OUT_IDX_SIZE * 4 / 1024 ** 2:.1f} MB)"
        else:
            # Tiny dummy — kernel checks shape before writing
            d_mean_cashflows = _to_device_contiguous(
                np.zeros((1, 1, 1), dtype=np.float32)
            )
            mem_saved_msg = "(mean_cf disabled — maximum memory savings)"

        original_cf_size = current_batch_size * nb_ext_scenarios * n_months * CF_OUT_IDX_SIZE * 4
        logger.info(f"  Memory saved: skipped {original_cf_size / 1024 ** 3:.2f} GB cashflow tensor {mem_saved_msg}")

        # Metrics tensor (only if running nested valuation)
        if run_nested_valuation:
            d_metrics = _device_array_cupy(
                (current_batch_size, nb_ext_scenarios, nb_an_projection, NUM_CHOCS, METRICS_OUTPUT_SIZE)
            )
        else:
            d_metrics = None

        # Debug arrays
        enable_ext_debug = debug_account >= 0 or debug_scenario >= 0 or debug_year >= 0 or debug_month >= 0
        enable_int_debug = enable_ext_debug and run_nested_valuation

        d_ext_debug = _device_array_cupy((EXT_DEBUG_SIZE,))

        if run_nested_valuation:
            d_int_debug = _device_array_cupy((NUM_CHOCS, INT_DEBUG_SIZE))
        else:
            d_int_debug = None

        freq_eval_int = int(CONFIG['FREQ_EVAL'])
        if enable_ext_debug:
            d_debug_flux = _to_device_contiguous(
                np.zeros((nb_an_projection + 1, freq_eval_int + 1, FLUX_COMP_IDX_SIZE), dtype=np.float32)
            )
        else:
            d_debug_flux = _to_device_contiguous(
                np.zeros((1, 1, FLUX_COMP_IDX_SIZE), dtype=np.float32)
            )

        enable_int_debug_ts = enable_int_debug and debug_int_scenario >= 0
        if enable_int_debug_ts:
            d_int_debug_ts = _to_device_contiguous(
                np.zeros((NUM_CHOCS, nb_an_projection, INT_TS_DEBUG_IDX_SIZE), dtype=np.float32)
            )
        elif run_nested_valuation:
            d_int_debug_ts = _to_device_contiguous(
                np.zeros((1, 1, INT_TS_DEBUG_IDX_SIZE), dtype=np.float32)
            )
        else:
            d_int_debug_ts = None

    except Exception as e:
        raise RuntimeError(
            f"Failed to allocate GPU memory for batch {batch_idx + 1}. "
            f"Try reducing --max-accounts or --ext-scenarios. Error: {e}"
        )

    # === KERNEL A: EXTERNAL GENERATOR (Low-Memory) ===
    logger.info("  Launching Kernel A (External Generator — LOW MEMORY)...")
    blocks_x = (current_batch_size + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_y = (nb_ext_scenarios + threads_per_block[1] - 1) // threads_per_block[1]
    grid_A = (blocks_x, blocks_y)

    kernel_a_start = datetime.now()
    if 'coussins' not in gpu_lookups:
        raise RuntimeError("gpu_lookups['coussins'] is missing.")

    # Call the low-memory kernel with 3 small output arrays instead of d_cashflows
    external_generator_kernel[grid_A, threads_per_block](
        d_batch_accounts,
        nb_ext_scenarios, nb_an_projection,
        CONFIG['FREQ_EVAL'],
        gpu_lookups['mortality'],
        gpu_lookups['returns'],
        gpu_lookups['lapse'],
        gpu_lookups['policy'],
        gpu_lookups['commission'],
        gpu_lookups['coussins'],
        d_states,
        d_vp_per_account,       # NEW: small (batch, 41)
        d_flux_projete_agg,     # NEW: small (months, 41)
        d_mean_cashflows,       # NEW: moderate or dummy (batch, months, 41) or (1,1,1)
        d_ext_debug,
        d_debug_flux,
        debug_account,
        debug_scenario,
        debug_year,
        debug_month,
    )
    cuda.synchronize()
    kernel_a_time = (datetime.now() - kernel_a_start).total_seconds()
    logger.info(f"  Kernel A complete: {kernel_a_time:.2f}s")

    # Free states if not running nested valuation (tiny dummy, negligible)
    if not run_nested_valuation:
        del d_states
        cuda.synchronize()

    # === KERNEL B: NESTED VALUATOR (unchanged) ===
    kernel_b_time = 0.0
    if run_nested_valuation:
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
            gpu_lookups['lapse'],
            gpu_lookups['policy'],
            gpu_lookups['commission'],
            d_metrics,
            d_int_debug,
            d_int_debug_ts,
            debug_int_scenario,
            debug_int_year,
            debug_account,
            debug_scenario,
            debug_year,
            float(shock_capital_pct),
        )
        cuda.synchronize()
        kernel_b_time = (datetime.now() - kernel_b_start).total_seconds()
        logger.info(f"  Kernel B complete: {kernel_b_time:.2f}s")
    else:
        logger.info(f"  Kernel B skipped (outer loop only)")

    # ============================================================
    # COPY RESULTS — No CuPy aggregation needed, already done in kernel!
    # ============================================================
    logger.info("  Copying small aggregated results to CPU...")

    # Debug arrays
    h_ext_debug = None
    h_int_debug = None
    h_int_debug_ts = None
    h_debug_flux = None
    if enable_ext_debug:
        h_ext_debug = d_ext_debug.copy_to_host()
        h_debug_flux = d_debug_flux.copy_to_host()
    if enable_int_debug:
        h_int_debug = d_int_debug.copy_to_host()
        if enable_int_debug_ts:
            h_int_debug_ts = d_int_debug_ts.copy_to_host()

    # Copy the small aggregated tensors and divide by n_scenarios to get means
    h_vp_flux_compte = d_vp_per_account.copy_to_host()
    h_vp_flux_compte /= nb_ext_scenarios
    logger.info(f"  VP per account: {h_vp_flux_compte.nbytes / 1024:.1f} KB copied")

    h_flux_projete = d_flux_projete_agg.copy_to_host()
    h_flux_projete /= nb_ext_scenarios
    logger.info(f"  Flux projete agg: {h_flux_projete.nbytes / 1024:.1f} KB copied")

    h_mean_cashflows = None
    if write_detailed_cashflows:
        h_mean_cashflows = d_mean_cashflows.copy_to_host()
        h_mean_cashflows /= nb_ext_scenarios
        logger.info(f"  Mean cashflows: {h_mean_cashflows.nbytes / 1024 ** 2:.1f} MB copied")

    # Process metrics (nested valuation)
    if run_nested_valuation:
        h_metrics = d_metrics.copy_to_host()
        batch_reserves_5chocs = h_metrics[:, :, :, :, METRICS_RESERVE_IDX].mean(axis=(1, 2))
        batch_capital_5chocs = h_metrics[:, :, :, :, METRICS_CAPITAL_IDX].mean(axis=(1, 2))
        batch_reserves = batch_reserves_5chocs[:, 0]
        batch_capital = batch_capital_5chocs[:, 0]
    else:
        # Compute reserves from vp_per_account (already divided by n_scenarios)
        logger.info("  Computing reserves from VP per account (no mean cashflows needed)...")
        h_metrics = None
        batch_reserves = (
            h_vp_flux_compte[:, CF_OUT_IDX_VP_FRAIS_ACQUIS] +
            h_vp_flux_compte[:, CF_OUT_IDX_VP_COMM_VENTE] +
            h_vp_flux_compte[:, CF_OUT_IDX_VP_PRIMES_GARANTIES] +
            h_vp_flux_compte[:, CF_OUT_IDX_VP_PRIMES_VARIABLES] +
            h_vp_flux_compte[:, CF_OUT_IDX_VP_FRAIS_FIXES] +
            h_vp_flux_compte[:, CF_OUT_IDX_VP_HON_GEST] +
            h_vp_flux_compte[:, CF_OUT_IDX_VP_COMM_MAINTIEN] +
            h_vp_flux_compte[:, CF_OUT_IDX_VP_PREST_ECH] +
            h_vp_flux_compte[:, CF_OUT_IDX_VP_PREST_MRV] +
            h_vp_flux_compte[:, CF_OUT_IDX_VP_PREST_DECES]
        )
        batch_capital = np.zeros(current_batch_size, dtype=np.float32)
        batch_reserves_5chocs = np.zeros((current_batch_size, NUM_CHOCS), dtype=np.float32)
        batch_capital_5chocs = np.zeros((current_batch_size, NUM_CHOCS), dtype=np.float32)
        batch_reserves_5chocs[:, 0] = batch_reserves

    # Cleanup GPU memory
    del d_batch_accounts, d_ext_debug, d_debug_flux
    del d_vp_per_account, d_flux_projete_agg, d_mean_cashflows
    if not run_nested_valuation:
        pass  # d_states already deleted
    else:
        del d_states
    if d_metrics is not None:
        del d_metrics
    if d_int_debug is not None:
        del d_int_debug
    if d_int_debug_ts is not None:
        del d_int_debug_ts
    cuda.synchronize()
    if h_metrics is not None:
        del h_metrics
    gc.collect()

    _clear_cupy_refs()
    gc.collect()

    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except (ImportError, AttributeError):
        pass

    try:
        import rmm
        rmm.mr.get_current_device_resource().deallocate(0, 0)
    except (ImportError, AttributeError):
        pass

    cuda.synchronize()
    gc.collect()

    batch_time = (datetime.now() - batch_start).total_seconds()
    logger.info(f"  Batch complete: {batch_time:.2f}s (KernelA: {kernel_a_time:.2f}s, KernelB: {kernel_b_time:.2f}s)")

    return {
        'batch_reserves': batch_reserves,
        'batch_capital': batch_capital,
        'batch_reserves_5chocs': batch_reserves_5chocs,
        'batch_capital_5chocs': batch_capital_5chocs,
        'batch_mean_cashflows': h_mean_cashflows,
        'batch_vp_flux_compte': h_vp_flux_compte,
        'batch_flux_projete': h_flux_projete,
        'ext_debug': h_ext_debug,
        'int_debug': h_int_debug,
        'int_debug_ts': h_int_debug_ts,
        'debug_flux': h_debug_flux,
    }


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
        rendements_int_path: Optional[Path] = None,
        depots_futurs_path: Optional[Path] = None,
        frais_admin_path: Optional[Path] = None,
        min_ferr_path: Optional[Path] = None,
        tx_lapse_part_path: Optional[Path] = None,
        tx_lapse_tot_path: Optional[Path] = None,
        acquisition_path: Optional[Path] = None,
        coussins_escap_path: Optional[Path] = None,
        progress_callback: Optional[callable] = None,
        debug_account_id: Optional[int] = None,
        debug_account: int = -1,
        debug_scenario: int = -1,
        debug_year: int = -1,
        debug_month: int = -1,
        debug_int_scenario: int = -1,
        debug_int_year: int = -1,
        debug_only: bool = False,
        run_nested_valuation: bool = True,
        write_detailed_cashflows: bool = False):
    """
    Run GPU-accelerated projection using LOW-MEMORY kernels.

    Same interface as calculations.gpu.run_projection_gpu_nested but uses
    in-kernel atomic aggregation to eliminate the massive cashflow tensor.
    """
    start_time = datetime.now()

    # Clear output directory
    if output_path.exists():
        import shutil
        print(f"Clearing output directory: {output_path}")
        for item in output_path.glob('*.csv'):
            item.unlink()
    else:
        output_path.mkdir(parents=True, exist_ok=True)

    mode_str = 'NESTED STOCHASTIC' if run_nested_valuation else 'OUTER LOOP ONLY'
    print(f"[LOW-MEM] Starting {mode_str} GPU projection at {start_time}")
    print("=" * 80)
    print(f"Architecture: LOW-MEMORY {'Two-Pass' if run_nested_valuation else 'Single-Pass'}")
    print(f"External scenarios: {nb_ext_scenarios}")
    if run_nested_valuation:
        print(f"Internal scenarios per node: {nb_int_scenarios}")
        print(f"Capital shock: {shock_capital_pct * 100:.1f}%")
    sys.stdout.flush()

    enable_debug = debug_account >= 0 or debug_scenario >= 0 or debug_year >= 0 or debug_month >= 0
    if enable_debug:
        print(f"Debug: account={debug_account}, scenario={debug_scenario}, year={debug_year}, month={debug_month}")
        if debug_only and debug_account >= 0:
            print(f"  DEBUG_ONLY: Will process ONLY account {debug_account}")
    print("=" * 80)
    sys.stdout.flush()

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
                         coussins_escap_path=coussins_escap_path,
                         rendements_int_path=rendements_int_path)
    print("Data loaded successfully")

    # Filter for debug_only
    if debug_only and debug_account >= 0:
        pop_df = data['population']
        if 'ID_COMPTE' in pop_df.columns:
            filtered = pop_df[pop_df['ID_COMPTE'] == debug_account]
        elif 'NO_COMPTE' in pop_df.columns:
            filtered = pop_df[pop_df['NO_COMPTE'] == debug_account]
        else:
            if debug_account < len(pop_df):
                filtered = pop_df.iloc[[debug_account]]
            else:
                raise ValueError(f"debug_account {debug_account} out of range (max: {len(pop_df) - 1})")
        if len(filtered) == 0:
            raise ValueError(f"Account {debug_account} not found")
        data['population'] = filtered.reset_index(drop=True)
        max_accounts = None
    elif max_accounts:
        data['population'] = data['population'].head(max_accounts)

    if debug_account_id is not None:
        if 'ID_COMPTE' not in data['population'].columns:
            raise ValueError("Population data does not contain ID_COMPTE")
        matches = np.where(data['population']['ID_COMPTE'].values == debug_account_id)[0]
        if len(matches) == 0:
            raise ValueError(f"debug_account_id={debug_account_id} not found")
        debug_account = int(matches[0])
        enable_debug = True

    n_accounts = len(data['population'])
    print(f"\nPreparing {n_accounts} accounts for LOW-MEM GPU processing...")

    all_account_data, _ = prepare_account_data(data['population'])
    lookups = create_all_lookup_tables(data, nb_int_scenarios, nb_an_projection)

    batch_size, num_batches, total_mem_per_account, _ = calculate_batch_size(
        n_accounts, nb_ext_scenarios, nb_an_projection,
        nb_int_scenarios, all_account_data.shape[1],
        write_detailed_cashflows=write_detailed_cashflows,
        run_nested_valuation=run_nested_valuation,
    )

    gpu_lookups = copy_lookups_to_gpu(lookups)

    # Process batches
    print("\n" + "=" * 80)
    print(f"RUNNING LOW-MEMORY {mode_str} PROJECTION")
    print("=" * 80)

    all_reserves = []
    all_capital = []
    all_reserves_5chocs = []
    all_capital_5chocs = []
    ext_debug_result = None
    int_debug_result = None
    int_debug_ts_result = None
    debug_flux_result = None
    total_cashflow_rows = 0
    total_vp_flux_compte_rows = 0
    accumulated_flux_projete = None
    population_ids = data['population']['ID_COMPTE'].values

    validate_kernel_compatibility()

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_accounts)
        batch_account_data = all_account_data[start_idx:end_idx]

        batch_debug_account = -1
        if debug_account >= 0:
            if start_idx <= debug_account < end_idx:
                batch_debug_account = debug_account - start_idx

        batch_result = process_batch(
            batch_account_data=batch_account_data,
            nb_ext_scenarios=nb_ext_scenarios,
            nb_an_projection=nb_an_projection,
            nb_int_scenarios=nb_int_scenarios,
            shock_capital_pct=shock_capital_pct,
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
            run_nested_valuation=run_nested_valuation,
            write_detailed_cashflows=write_detailed_cashflows,
        )

        # Accumulate results
        all_reserves.extend(batch_result['batch_reserves'])
        all_capital.extend(batch_result['batch_capital'])
        all_reserves_5chocs.extend(batch_result['batch_reserves_5chocs'])
        all_capital_5chocs.extend(batch_result['batch_capital_5chocs'])

        # Write detailed cashflows incrementally
        if write_detailed_cashflows and batch_result.get('batch_mean_cashflows') is not None:
            csv_start = datetime.now()
            rows_written = write_cashflows_batch(
                output_path=output_path,
                batch_mean_cashflows=batch_result['batch_mean_cashflows'],
                population_ids=population_ids,
                start_idx=start_idx,
                is_first_batch=(i == 0),
            )
            csv_time = (datetime.now() - csv_start).total_seconds()
            logger.info(f"  FLUX_PROJETE CSV: {csv_time:.2f}s ({rows_written} rows)")
            total_cashflow_rows += rows_written
        if batch_result.get('batch_mean_cashflows') is not None:
            del batch_result['batch_mean_cashflows']

        # Write VP_FLUX_COMPTE incrementally
        if batch_result.get('batch_vp_flux_compte') is not None:
            vp_rows_written = write_vp_flux_compte_batch(
                output_path=output_path,
                batch_vp_flux_compte=batch_result['batch_vp_flux_compte'],
                population_ids=population_ids,
                start_idx=start_idx,
                is_first_batch=(i == 0),
            )
            total_vp_flux_compte_rows += vp_rows_written
            del batch_result['batch_vp_flux_compte']

        # Accumulate FLUX_PROJETE
        if batch_result.get('batch_flux_projete') is not None:
            accumulated_flux_projete = accumulate_flux_projete(
                accumulated_flux_projete,
                batch_result['batch_flux_projete']
            )
            del batch_result['batch_flux_projete']

        # Store debug output
        if batch_result['ext_debug'] is not None:
            ext_debug_result = batch_result['ext_debug']
        if batch_result['int_debug'] is not None:
            int_debug_result = batch_result['int_debug']
        if batch_result.get('int_debug_ts') is not None:
            int_debug_ts_result = batch_result['int_debug_ts']
        if batch_result.get('debug_flux') is not None:
            debug_flux_result = batch_result['debug_flux']

        if progress_callback is not None:
            progress_callback(i + 1, num_batches)

    # Write final FLUX_PROJETE
    total_flux_projete_rows = 0
    if accumulated_flux_projete is not None:
        total_flux_projete_rows = write_flux_projete(
            output_path=output_path,
            flux_projete=accumulated_flux_projete,
            nb_an_projection=nb_an_projection,
        )

    # Create results DataFrames
    results_df, results_5chocs_df, sensitivities_df = create_results_dataframes(
        population_ids=population_ids,
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
    print(f"[LOW-MEM] {mode_str} PROJECTION COMPLETE")
    print("=" * 80)
    print(f"Total time: {total_duration:.2f}s ({total_duration / 60:.2f} minutes)")
    print(f"Accounts processed: {n_accounts}")
    print(f"External scenarios: {nb_ext_scenarios}")
    if run_nested_valuation:
        print(f"Total Reserve: ${results_df['RESERVE_BE'].sum():,.2f}")
        print(f"Total Capital: ${results_df['CAPITAL_REQ'].sum():,.2f}")
        print(f"Total SCR:     ${results_df['SCR'].sum():,.2f}")
    else:
        print(f"Total Reserve (PV): ${results_df['RESERVE_BE'].sum():,.2f}")
        print(f"(Capital requires nested valuation)")
    print("=" * 80)

    # Build debug params
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

    int_debug_ts_df = None
    if enable_debug and int_debug_ts_result is not None:
        rows = []
        max_years = int_debug_ts_result.shape[1]
        years_iter = [debug_int_year] if debug_int_year is not None and debug_int_year >= 0 else range(max_years)
        for choc_idx in range(int_debug_ts_result.shape[0]):
            choc_name = CHOC_NAMES[choc_idx] if choc_idx < len(CHOC_NAMES) else f"CHOC_{choc_idx}"
            for t_int in years_iter:
                if t_int < 0 or t_int >= max_years:
                    continue
                id_compte = int(population_ids[debug_account]) if population_ids is not None and 0 <= debug_account < len(population_ids) else -1
                rows.append({
                    'CHOC_IDX': choc_idx, 'CHOC_NAME': choc_name, 'T_INT': int(t_int),
                    'DEBUG_ACCOUNT_IDX': debug_account, 'ID_COMPTE': id_compte,
                    'DEBUG_SCENARIO': debug_scenario, 'DEBUG_YEAR': debug_year,
                    'DEBUG_INT_SCENARIO': debug_int_scenario,
                    'CURR_VM': float(int_debug_ts_result[choc_idx, t_int, INT_TS_DEBUG_IDX_CURR_VM]),
                    'FEES': float(int_debug_ts_result[choc_idx, t_int, INT_TS_DEBUG_IDX_FEES]),
                    'PV_PATH': float(int_debug_ts_result[choc_idx, t_int, INT_TS_DEBUG_IDX_PV_PATH]),
                    'R_PORTFOLIO': float(int_debug_ts_result[choc_idx, t_int, INT_TS_DEBUG_IDX_R_PORTFOLIO]),
                    'FWD_RATE': float(int_debug_ts_result[choc_idx, t_int, INT_TS_DEBUG_IDX_FWD_RATE]),
                    'DF': float(int_debug_ts_result[choc_idx, t_int, INT_TS_DEBUG_IDX_DF]),
                })
        if rows:
            int_debug_ts_df = pd.DataFrame(rows)

    # Save all results
    save_result = save_results(
        output_path=output_path,
        results_df=results_df,
        results_5chocs_df=results_5chocs_df,
        sensitivities_df=sensitivities_df,
        n_accounts=n_accounts,
        ext_debug=ext_debug_result,
        int_debug=int_debug_result,
        int_debug_ts_df=int_debug_ts_df,
        debug_params=debug_params,
        population_ids=population_ids,
        debug_flux=debug_flux_result,
        population_df=data.get('population'),
        lookup_data=data,
        total_cashflow_rows=total_cashflow_rows,
        total_vp_flux_compte_rows=total_vp_flux_compte_rows,
        total_flux_projete_rows=total_flux_projete_rows,
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
        int_debug_ts_df=save_result['int_debug_ts_df'],
        flux_projetes_df=save_result['flux_projetes_df'],
        saved_files=save_result['saved_files'],
    )


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run LOW-MEMORY GPU-accelerated actuarial projections',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Low-memory version: eliminates the massive cashflow tensor (~98% GPU RAM savings).

Examples:
  # Outer loop only (most common use case for low-mem)
  python -m calculations.gpu_lowmem --outer-loop-only --ext-scenarios 50 --max-accounts 999999

  # Full nested stochastic (still saves ~98% RAM from cashflow tensor)
  python -m calculations.gpu_lowmem --ext-scenarios 100 --int-scenarios 500 --max-accounts 1000

  # With detailed per-account cashflows
  python -m calculations.gpu_lowmem --outer-loop-only --ext-scenarios 50 --write-detailed-cashflows
        """
    )

    parser.add_argument('--max-accounts', type=int, default=200)
    parser.add_argument('--years', type=int, default=100)
    parser.add_argument('--ext-scenarios', type=int, default=100)
    parser.add_argument('--int-scenarios', type=int, default=100)
    parser.add_argument('--shock', type=float, default=0.35)
    parser.add_argument('--outer-loop-only', action='store_true')
    parser.add_argument('--debug-account', type=int, default=0)
    parser.add_argument('--debug-account-id', type=int, default=None)
    parser.add_argument('--debug-scenario', type=int, default=0)
    parser.add_argument('--debug-year', type=int, default=-1)
    parser.add_argument('--debug-month', type=int, default=-1)
    parser.add_argument('--debug-int-scenario', type=int, default=0)
    parser.add_argument('--debug-int-year', type=int, default=1)
    parser.add_argument('--write-detailed-cashflows', action='store_true',
                        help='Write per-account cashflows (moderate memory). Default: disabled for max savings.')

    args = parser.parse_args()

    try:
        if not cuda.is_available():
            print("ERROR: CUDA not available.")
            exit(1)

        print(f"CUDA Device: {cuda.get_current_device().name}")

        DATA_PATH = HERE.joinpath("data_in")
        OUTPUT_PATH = HERE.joinpath("data_out_gpu")

        print("\n" + "=" * 80)
        print("RUNNING LOW-MEMORY GPU PROJECTION")
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
            debug_account_id=args.debug_account_id,
            debug_account=args.debug_account,
            debug_scenario=args.debug_scenario,
            debug_year=args.debug_year,
            debug_month=args.debug_month,
            debug_int_scenario=args.debug_int_scenario,
            debug_int_year=args.debug_int_year,
            run_nested_valuation=not args.outer_loop_only,
            write_detailed_cashflows=args.write_detailed_cashflows,
        )

        if results:
            print("\nTop 10 accounts by SCR:")
            print(results.results.nlargest(10, 'SCR')[['ID_COMPTE', 'RESERVE_BE', 'CAPITAL_REQ', 'SCR']])
            print("\nSummary:")
            print(results.results[['RESERVE_BE', 'CAPITAL_REQ', 'SCR']].describe())

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

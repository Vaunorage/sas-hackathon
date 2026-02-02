#!/usr/bin/env python3
"""
Benchmark: Numba CUDA Kernels vs CuPy (Paul's approach)

Matches Paul's exact problem size and calculations:
- 100 years × 500 scenarios × 500 policies × 17 calculations
- Same calculation logic (MV projection, guarantee ratchet, claims)
- Measures Load, Loop, Summary phases separately

Target to beat: 1.6s (Paul's best with CuPy satellite arrays)
"""

import numpy as np
import cupy as cp
import math
from numba import cuda
from datetime import datetime
import warnings

from numba.core.errors import NumbaPerformanceWarning
warnings.filterwarnings('ignore', category=NumbaPerformanceWarning)

# =============================================================================
# CONFIGURATION - Match Paul's benchmark
# =============================================================================
OUTER_YEARS = 100
OUTER_SCENS = 500
NB_POLICIES = 500
NB_CALCS = 17  # 16 output calcs + 1 for claims (index 0)

# Field indices for input data (OL = Outer Loop)
_T = 0
_S = 1
_POLICY = 2
_MV = 3
_RETURN = 4
_TRANSAC = 5
_GUARANTEE = 6
NB_INPUT_FIELDS = 7

# Field indices for output calculations (OC = Outer Calcs)
_CLAIMS = 0
_CALC1 = 1
_CALC2 = 2
_CALC3 = 3
_CALC4 = 4
_CALC5 = 5
_CALC6 = 6
_CALC7 = 7
_CALC8 = 8
_CALC9 = 9
_CALC10 = 10
_CALC11 = 11
_CALC12 = 12
_CALC13 = 13
_CALC14 = 14
_CALC15 = 15
_CALC16 = 16


# =============================================================================
# DATA GENERATION
# =============================================================================
def generate_test_data():
    """Generate test data matching Paul's structure."""
    np.random.seed(42)
    
    # Create base data
    # Shape: (OUTER_YEARS, OUTER_SCENS, NB_POLICIES, NB_INPUT_FIELDS)
    OL_np = np.zeros((OUTER_YEARS, OUTER_SCENS, NB_POLICIES, NB_INPUT_FIELDS), dtype=np.float64)
    
    # Initialize T, S, POLICY indices
    for t in range(OUTER_YEARS):
        for s in range(OUTER_SCENS):
            OL_np[t, s, :, _T] = t
            OL_np[t, s, :, _S] = s
            OL_np[t, s, :, _POLICY] = np.arange(NB_POLICIES)
    
    # Initialize MV (market value) - random starting values
    OL_np[0, :, :, _MV] = np.random.uniform(10000, 100000, (OUTER_SCENS, NB_POLICIES))
    
    # Returns - random per scenario/year
    for t in range(OUTER_YEARS):
        OL_np[t, :, :, _RETURN] = np.random.normal(0.05, 0.15, (OUTER_SCENS, NB_POLICIES))
    
    # Transactions - random
    OL_np[:, :, :, _TRANSAC] = np.random.uniform(-1000, 1000, (OUTER_YEARS, OUTER_SCENS, NB_POLICIES))
    
    # Guarantee - initial value based on MV
    OL_np[0, :, :, _GUARANTEE] = OL_np[0, :, :, _MV] * 1.0
    
    return OL_np


# =============================================================================
# APPROACH 1: CUPY (Paul's approach with satellite arrays)
# =============================================================================
def run_cupy_approach(OL_np):
    """Paul's CuPy approach with satellite arrays (best performing)."""
    
    # === LOAD PHASE ===
    t_load_start = datetime.now()
    
    # Transfer input data to GPU
    OL_cp = cp.asarray(OL_np)
    
    # Create output array directly on GPU (satellite - not transferred)
    OC_cp = cp.zeros((OUTER_YEARS, OUTER_SCENS, NB_POLICIES, NB_CALCS), dtype=cp.float64)
    
    cp.cuda.Stream.null.synchronize()
    t_load_end = datetime.now()
    
    # === LOOP PHASE ===
    t_loop_start = datetime.now()
    
    for T in range(1, OUTER_YEARS - 1):
        OL_cp[T, ..., _MV] = OL_cp[T-1, ..., _MV] * (1 + OL_cp[T, ..., _RETURN]) + OL_cp[T, ..., _TRANSAC]
        OL_cp[T, ..., _GUARANTEE] = OL_cp[T-1, ..., _GUARANTEE]
        
        OC_cp[T, ..., _CALC1] = (OL_cp[T, ..., _MV] + OL_cp[T, ..., _GUARANTEE]) / 2
        OC_cp[T, ..., _CALC2] = (OL_cp[T, ..., _MV] / 1000) ** 2
        OC_cp[T, ..., _CALC3] = OL_cp[T, ..., _MV] * (1 + OL_cp[T, ..., _RETURN]) ** 1.5
        OC_cp[T, ..., _CALC4] = cp.nan_to_num(OL_cp[T, ..., _TRANSAC] / OL_cp[T, ..., _MV])
        OC_cp[T, ..., _CALC5] = 1 - (1 + OL_cp[T, ..., _RETURN]) ** 0.5
        OC_cp[T, ..., _CALC6] = (OL_cp[T, ..., _MV] - OL_cp[T, ..., _GUARANTEE]) / 2
        OC_cp[T, ..., _CALC7] = (-OL_cp[T, ..., _MV] / 1000) ** 2
        OC_cp[T, ..., _CALC8] = OL_cp[T, ..., _MV] * (1 - OL_cp[T, ..., _RETURN]) ** 1.5
        OC_cp[T, ..., _CALC9] = cp.nan_to_num(OL_cp[T, ..., _TRANSAC] / OL_cp[T, ..., _MV])
        OC_cp[T, ..., _CALC10] = 1 - (1 - OL_cp[T, ..., _RETURN]) ** 0.5
        OC_cp[T, ..., _CALC11] = (OL_cp[T, ..., _MV] - OL_cp[T, ..., _GUARANTEE]) / 2
        OC_cp[T, ..., _CALC12] = (-OL_cp[T, ..., _MV] / 1000) ** 2
        OC_cp[T, ..., _CALC13] = OL_cp[T, ..., _MV] * (1 - OL_cp[T, ..., _RETURN]) ** 1.5
        OC_cp[T, ..., _CALC14] = cp.nan_to_num(OL_cp[T, ..., _TRANSAC] / OL_cp[T, ..., _MV])
        OC_cp[T, ..., _CALC15] = 1 - (1 - OL_cp[T, ..., _RETURN]) ** 0.5
        OC_cp[T, ..., _CALC16] = OL_cp[T, ..., _MV] * 0.001  # Extra calc to match 17
        
        if T % 10 == 0:
            OL_cp[T, ..., _GUARANTEE] = cp.maximum(OL_cp[T-1, ..., _GUARANTEE], OL_cp[T, ..., _MV])
        
        OC_cp[T, ..., _CLAIMS] = cp.maximum((OL_cp[T, ..., _GUARANTEE] - OL_cp[T, ..., _MV]), 0)
    
    cp.cuda.Stream.null.synchronize()
    t_loop_end = datetime.now()
    
    # === SUMMARY PHASE ===
    t_summary_start = datetime.now()
    
    result = cp.nansum(cp.nanmean(cp.nansum(OC_cp[..., _CLAIMS], axis=0), axis=0), axis=0)
    result_value = float(result)
    
    cp.cuda.Stream.null.synchronize()
    t_summary_end = datetime.now()
    
    # Cleanup
    del OL_cp, OC_cp
    cp.get_default_memory_pool().free_all_blocks()
    
    return {
        'load': (t_load_end - t_load_start).total_seconds(),
        'loop': (t_loop_end - t_loop_start).total_seconds(),
        'summary': (t_summary_end - t_summary_start).total_seconds(),
        'result': result_value
    }


# =============================================================================
# APPROACH 2: NUMBA CUDA KERNEL (Account-First)
# =============================================================================
@cuda.jit
def numba_projection_kernel(
    OL_data,    # (n_years, n_scens, n_policies, n_fields) - input/state
    OC_data,    # (n_years, n_scens, n_policies, n_calcs) - output
    n_years,
    n_scens,
    n_policies,
):
    """
    Account-first kernel: each thread processes one (scenario, policy) pair
    through ALL years sequentially.
    
    This fuses all operations into a single kernel launch.
    """
    # 2D grid: (scenario, policy)
    scn_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    pol_idx = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    
    if scn_idx >= n_scens or pol_idx >= n_policies:
        return
    
    # Load initial state
    mv = OL_data[0, scn_idx, pol_idx, _MV]
    guarantee = OL_data[0, scn_idx, pol_idx, _GUARANTEE]
    
    # Process all years sequentially
    for T in range(1, n_years - 1):
        # Get inputs for this timestep
        ret = OL_data[T, scn_idx, pol_idx, _RETURN]
        transac = OL_data[T, scn_idx, pol_idx, _TRANSAC]
        
        # Update MV
        mv = mv * (1.0 + ret) + transac
        
        # Store updated MV back (needed for consistency)
        OL_data[T, scn_idx, pol_idx, _MV] = mv
        OL_data[T, scn_idx, pol_idx, _GUARANTEE] = guarantee
        
        # Calculate all outputs
        OC_data[T, scn_idx, pol_idx, _CALC1] = (mv + guarantee) / 2.0
        OC_data[T, scn_idx, pol_idx, _CALC2] = (mv / 1000.0) ** 2
        OC_data[T, scn_idx, pol_idx, _CALC3] = mv * math.pow(1.0 + ret, 1.5)
        
        # Safe division
        if mv != 0.0:
            OC_data[T, scn_idx, pol_idx, _CALC4] = transac / mv
        else:
            OC_data[T, scn_idx, pol_idx, _CALC4] = 0.0
        
        OC_data[T, scn_idx, pol_idx, _CALC5] = 1.0 - math.pow(1.0 + ret, 0.5)
        OC_data[T, scn_idx, pol_idx, _CALC6] = (mv - guarantee) / 2.0
        OC_data[T, scn_idx, pol_idx, _CALC7] = (-mv / 1000.0) ** 2
        OC_data[T, scn_idx, pol_idx, _CALC8] = mv * math.pow(1.0 - ret, 1.5)
        
        if mv != 0.0:
            OC_data[T, scn_idx, pol_idx, _CALC9] = transac / mv
        else:
            OC_data[T, scn_idx, pol_idx, _CALC9] = 0.0
        
        OC_data[T, scn_idx, pol_idx, _CALC10] = 1.0 - math.pow(1.0 - ret, 0.5)
        OC_data[T, scn_idx, pol_idx, _CALC11] = (mv - guarantee) / 2.0
        OC_data[T, scn_idx, pol_idx, _CALC12] = (-mv / 1000.0) ** 2
        OC_data[T, scn_idx, pol_idx, _CALC13] = mv * math.pow(1.0 - ret, 1.5)
        
        if mv != 0.0:
            OC_data[T, scn_idx, pol_idx, _CALC14] = transac / mv
        else:
            OC_data[T, scn_idx, pol_idx, _CALC14] = 0.0
        
        OC_data[T, scn_idx, pol_idx, _CALC15] = 1.0 - math.pow(1.0 - ret, 0.5)
        OC_data[T, scn_idx, pol_idx, _CALC16] = mv * 0.001
        
        # Guarantee ratchet every 10 years
        if T % 10 == 0:
            guarantee = max(guarantee, mv)
            OL_data[T, scn_idx, pol_idx, _GUARANTEE] = guarantee
        
        # Claims calculation
        OC_data[T, scn_idx, pol_idx, _CLAIMS] = max(guarantee - mv, 0.0)


def run_numba_approach(OL_np):
    """Numba CUDA kernel approach (account-first)."""
    
    # === LOAD PHASE ===
    t_load_start = datetime.now()
    
    # Transfer input data to GPU
    d_OL = cuda.to_device(OL_np)
    
    # Create output array directly on GPU
    d_OC = cuda.device_array((OUTER_YEARS, OUTER_SCENS, NB_POLICIES, NB_CALCS), dtype=np.float64)
    
    cuda.synchronize()
    t_load_end = datetime.now()
    
    # === LOOP PHASE ===
    t_loop_start = datetime.now()
    
    # Grid configuration: (scenarios, policies)
    threads_per_block = (16, 16)
    blocks_x = (OUTER_SCENS + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_y = (NB_POLICIES + threads_per_block[1] - 1) // threads_per_block[1]
    grid = (blocks_x, blocks_y)
    
    # Single kernel launch - processes all years internally
    numba_projection_kernel[grid, threads_per_block](
        d_OL, d_OC,
        OUTER_YEARS, OUTER_SCENS, NB_POLICIES
    )
    
    cuda.synchronize()
    t_loop_end = datetime.now()
    
    # === SUMMARY PHASE ===
    t_summary_start = datetime.now()
    
    # Copy claims column back for summarization
    # Use CuPy for fast reduction (interop with Numba device arrays)
    OC_cp = cp.asarray(d_OC)
    result = cp.nansum(cp.nanmean(cp.nansum(OC_cp[..., _CLAIMS], axis=0), axis=0), axis=0)
    result_value = float(result)
    
    cuda.synchronize()
    t_summary_end = datetime.now()
    
    # Cleanup
    del d_OL, d_OC, OC_cp
    cp.get_default_memory_pool().free_all_blocks()
    
    return {
        'load': (t_load_end - t_load_start).total_seconds(),
        'loop': (t_loop_end - t_loop_start).total_seconds(),
        'summary': (t_summary_end - t_summary_start).total_seconds(),
        'result': result_value
    }


# =============================================================================
# MAIN BENCHMARK
# =============================================================================
def run_benchmark():
    print("=" * 80)
    print("BENCHMARK: Numba CUDA Kernels vs CuPy (Paul's approach)")
    print("=" * 80)
    
    # Check GPU
    if not cuda.is_available():
        print("ERROR: CUDA not available!")
        return
    
    gpu = cuda.get_current_device()
    print(f"GPU: {gpu.name.decode()}")
    print(f"Problem size: {OUTER_YEARS} years × {OUTER_SCENS} scenarios × {NB_POLICIES} policies × {NB_CALCS} calcs")
    
    # Estimate memory
    input_mem = OUTER_YEARS * OUTER_SCENS * NB_POLICIES * NB_INPUT_FIELDS * 8 / 1e9
    output_mem = OUTER_YEARS * OUTER_SCENS * NB_POLICIES * NB_CALCS * 8 / 1e9
    print(f"Estimated memory: Input={input_mem:.2f}GB, Output={output_mem:.2f}GB, Total={input_mem+output_mem:.2f}GB")
    print()
    
    # Generate test data
    print("[1] Generating test data...")
    OL_np = generate_test_data()
    print(f"    Input shape: {OL_np.shape}")
    print(f"    Input memory: {OL_np.nbytes / 1e9:.2f} GB")
    print()
    
    # Warm-up runs
    print("[2] Warm-up runs...")
    _ = run_cupy_approach(OL_np.copy())
    _ = run_numba_approach(OL_np.copy())
    print("    Done")
    print()
    
    # Benchmark runs
    n_runs = 3
    print(f"[3] Running benchmark ({n_runs} runs each)...")
    print()
    
    cupy_results = []
    numba_results = []
    
    for i in range(n_runs):
        print(f"    Run {i+1}/{n_runs}...")
        
        # CuPy
        cp_res = run_cupy_approach(OL_np.copy())
        cupy_results.append(cp_res)
        
        # Numba
        nb_res = run_numba_approach(OL_np.copy())
        numba_results.append(nb_res)
    
    print()
    
    # Calculate averages
    def avg(results, key):
        return sum(r[key] for r in results) / len(results)
    
    cp_load = avg(cupy_results, 'load')
    cp_loop = avg(cupy_results, 'loop')
    cp_summary = avg(cupy_results, 'summary')
    cp_total = cp_load + cp_loop + cp_summary
    cp_result = cupy_results[-1]['result']
    
    nb_load = avg(numba_results, 'load')
    nb_loop = avg(numba_results, 'loop')
    nb_summary = avg(numba_results, 'summary')
    nb_total = nb_load + nb_loop + nb_summary
    nb_result = numba_results[-1]['result']
    
    # Results table
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()
    print(f"{'Approach':<20} {'Load':>10} {'Loop':>10} {'Summary':>10} {'Total':>10} {'Result':>20}")
    print("-" * 80)
    print(f"{'CuPy (Paul)':<20} {cp_load:>9.2f}s {cp_loop:>9.2f}s {cp_summary:>9.2f}s {cp_total:>9.2f}s {cp_result:>20.2f}")
    print(f"{'Numba CUDA':<20} {nb_load:>9.2f}s {nb_loop:>9.2f}s {nb_summary:>9.2f}s {nb_total:>9.2f}s {nb_result:>20.2f}")
    print("-" * 80)
    
    # Speedup
    speedup = cp_total / nb_total if nb_total > 0 else 0
    loop_speedup = cp_loop / nb_loop if nb_loop > 0 else 0
    
    print()
    print("COMPARISON:")
    print(f"  Total speedup:     {speedup:.2f}x {'(Numba faster)' if speedup > 1 else '(CuPy faster)'}")
    print(f"  Loop speedup:      {loop_speedup:.2f}x {'(Numba faster)' if loop_speedup > 1 else '(CuPy faster)'}")
    print(f"  Results match:     {'YES' if abs(cp_result - nb_result) < 1e-6 else 'NO (diff=' + str(abs(cp_result - nb_result)) + ')'}")
    print()
    print(f"  Paul's target:     1.6s")
    print(f"  Numba achieved:    {nb_total:.2f}s")
    print(f"  vs target:         {1.6/nb_total:.2f}x {'FASTER' if nb_total < 1.6 else 'SLOWER'}")
    print()
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()

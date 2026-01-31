#!/usr/bin/env python3
"""
Benchmark: Account-First vs Time-First GPU Projection

Scales up the number of accounts to find the crossover point where
one approach becomes faster than the other.
"""

import numpy as np
import math
from numba import cuda
from datetime import datetime
import warnings

# Suppress Numba performance warnings for cleaner output
warnings.filterwarnings('ignore', category=cuda.NumbaPerformanceWarning)

# =============================================================================
# CONSTANTS
# =============================================================================
NUM_EXT_SCENARIOS = 50
NUM_INT_SCENARIOS = 100
NUM_YEARS = 10
FREQ_EVAL = 12
TOTAL_MONTHS = NUM_YEARS * FREQ_EVAL

# State/Cashflow indices
STATE_VM, STATE_GAR_DECES, STATE_TX_SURVIE, STATE_AGE, STATE_SIZE = 0, 1, 2, 3, 4
CF_PRIMES, CF_PREST_DECES, CF_HON_GEST, CF_RETRAIT, CF_SIZE = 0, 1, 2, 3, 4
ACC_VM_INIT, ACC_AGE_INIT, ACC_GAR_DECES, ACC_PC_FRAIS, ACC_SIZE = 0, 1, 2, 3, 4


# =============================================================================
# ACCOUNT-FIRST KERNELS
# =============================================================================

@cuda.jit
def account_first_kernel(
    accounts, returns_lookup, mortality_lookup,
    states, cashflows,
    n_ext_scenarios, n_years,
):
    """Account-first: each thread processes full time series for one (acc, scn)."""
    acc_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    scn_idx = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    
    n_accounts = accounts.shape[0]
    if acc_idx >= n_accounts or scn_idx >= n_ext_scenarios:
        return
    
    vm = accounts[acc_idx, ACC_VM_INIT]
    age = accounts[acc_idx, ACC_AGE_INIT]
    gar_deces = accounts[acc_idx, ACC_GAR_DECES]
    pc_frais = accounts[acc_idx, ACC_PC_FRAIS]
    tx_survie = 1.0
    
    for year in range(n_years):
        market_return = returns_lookup[scn_idx, year]
        vm = vm * (1.0 + market_return)
        
        for month in range(FREQ_EVAL):
            month_idx = year * FREQ_EVAL + month
            age_idx = min(int(age), mortality_lookup.shape[0] - 1)
            qx_annual = mortality_lookup[age_idx]
            qx_monthly = 1.0 - math.pow(1.0 - qx_annual, 1.0 / FREQ_EVAL)
            
            tx_survie_prev = tx_survie
            tx_survie = tx_survie * (1.0 - qx_monthly)
            deaths = tx_survie_prev - tx_survie
            
            hon_gest = vm * pc_frais / FREQ_EVAL * tx_survie
            primes = gar_deces * 0.005 / FREQ_EVAL * tx_survie
            prest_deces = max(0.0, gar_deces - vm) * deaths
            retrait = vm * 0.04 / FREQ_EVAL * tx_survie
            vm = vm - retrait
            
            cashflows[acc_idx, scn_idx, month_idx, CF_PRIMES] = primes
            cashflows[acc_idx, scn_idx, month_idx, CF_PREST_DECES] = prest_deces
            cashflows[acc_idx, scn_idx, month_idx, CF_HON_GEST] = hon_gest
            cashflows[acc_idx, scn_idx, month_idx, CF_RETRAIT] = retrait
        
        states[acc_idx, scn_idx, year, STATE_VM] = vm
        states[acc_idx, scn_idx, year, STATE_TX_SURVIE] = tx_survie
        age = age + 1


# =============================================================================
# TIME-FIRST KERNELS
# =============================================================================

@cuda.jit
def time_first_init_kernel(accounts, state_curr, n_scenarios):
    """Initialize state at t=0."""
    acc_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    scn_idx = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    
    n_accounts = accounts.shape[0]
    if acc_idx >= n_accounts or scn_idx >= n_scenarios:
        return
    
    state_curr[acc_idx, scn_idx, STATE_VM] = accounts[acc_idx, ACC_VM_INIT]
    state_curr[acc_idx, scn_idx, STATE_GAR_DECES] = accounts[acc_idx, ACC_GAR_DECES]
    state_curr[acc_idx, scn_idx, STATE_TX_SURVIE] = 1.0
    state_curr[acc_idx, scn_idx, STATE_AGE] = accounts[acc_idx, ACC_AGE_INIT]


@cuda.jit
def time_first_step_kernel(
    accounts, state_prev, state_curr, cashflows,
    returns_lookup, mortality_lookup,
    month_idx, n_scenarios, freq_eval,
):
    """Process one timestep for all accounts."""
    acc_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    scn_idx = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    
    n_accounts = accounts.shape[0]
    if acc_idx >= n_accounts or scn_idx >= n_scenarios:
        return
    
    vm = state_prev[acc_idx, scn_idx, STATE_VM]
    gar_deces = state_prev[acc_idx, scn_idx, STATE_GAR_DECES]
    tx_survie = state_prev[acc_idx, scn_idx, STATE_TX_SURVIE]
    age = state_prev[acc_idx, scn_idx, STATE_AGE]
    pc_frais = accounts[acc_idx, ACC_PC_FRAIS]
    
    year_idx = month_idx // freq_eval
    month_in_year = month_idx % freq_eval
    
    if month_in_year == 0:
        market_return = returns_lookup[scn_idx, year_idx]
        vm = vm * (1.0 + market_return)
    
    age_idx = min(int(age), mortality_lookup.shape[0] - 1)
    qx_annual = mortality_lookup[age_idx]
    qx_monthly = 1.0 - math.pow(1.0 - qx_annual, 1.0 / freq_eval)
    
    tx_survie_prev = tx_survie
    tx_survie = tx_survie * (1.0 - qx_monthly)
    deaths = tx_survie_prev - tx_survie
    
    hon_gest = vm * pc_frais / freq_eval * tx_survie
    primes = gar_deces * 0.005 / freq_eval * tx_survie
    prest_deces = max(0.0, gar_deces - vm) * deaths
    retrait = vm * 0.04 / freq_eval * tx_survie
    vm = vm - retrait
    
    if month_in_year == freq_eval - 1:
        age = age + 1
    
    state_curr[acc_idx, scn_idx, STATE_VM] = vm
    state_curr[acc_idx, scn_idx, STATE_GAR_DECES] = gar_deces
    state_curr[acc_idx, scn_idx, STATE_TX_SURVIE] = tx_survie
    state_curr[acc_idx, scn_idx, STATE_AGE] = age
    
    cashflows[acc_idx, scn_idx, CF_PRIMES] = primes
    cashflows[acc_idx, scn_idx, CF_PREST_DECES] = prest_deces
    cashflows[acc_idx, scn_idx, CF_HON_GEST] = hon_gest
    cashflows[acc_idx, scn_idx, CF_RETRAIT] = retrait


# =============================================================================
# BENCHMARK FUNCTIONS
# =============================================================================

def create_data(n_accounts):
    """Generate sample data for n_accounts."""
    np.random.seed(42)
    
    accounts = np.zeros((n_accounts, ACC_SIZE), dtype=np.float32)
    accounts[:, ACC_VM_INIT] = np.random.uniform(50000, 500000, n_accounts)
    accounts[:, ACC_AGE_INIT] = np.random.uniform(40, 70, n_accounts)
    accounts[:, ACC_GAR_DECES] = accounts[:, ACC_VM_INIT] * 1.1
    accounts[:, ACC_PC_FRAIS] = np.random.uniform(0.01, 0.03, n_accounts)
    
    returns = np.random.normal(0.05, 0.15, (NUM_EXT_SCENARIOS, NUM_YEARS)).astype(np.float32)
    
    max_age = 120
    mortality = np.zeros(max_age, dtype=np.float32)
    for age in range(max_age):
        mortality[age] = min(0.0001 * math.exp(0.08 * age), 1.0)
    
    return accounts, returns, mortality


def run_account_first(n_accounts, accounts, returns, mortality):
    """Run account-first approach and return execution time."""
    d_accounts = cuda.to_device(accounts)
    d_returns = cuda.to_device(returns)
    d_mortality = cuda.to_device(mortality)
    
    d_states = cuda.device_array(
        (n_accounts, NUM_EXT_SCENARIOS, NUM_YEARS, STATE_SIZE), dtype=np.float32
    )
    d_cashflows = cuda.device_array(
        (n_accounts, NUM_EXT_SCENARIOS, TOTAL_MONTHS, CF_SIZE), dtype=np.float32
    )
    
    threads_per_block = (16, 16)
    blocks_x = (n_accounts + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_y = (NUM_EXT_SCENARIOS + threads_per_block[1] - 1) // threads_per_block[1]
    grid = (blocks_x, blocks_y)
    
    # Warm-up run
    account_first_kernel[grid, threads_per_block](
        d_accounts, d_returns, d_mortality, d_states, d_cashflows,
        NUM_EXT_SCENARIOS, NUM_YEARS
    )
    cuda.synchronize()
    
    # Timed run
    t0 = datetime.now()
    account_first_kernel[grid, threads_per_block](
        d_accounts, d_returns, d_mortality, d_states, d_cashflows,
        NUM_EXT_SCENARIOS, NUM_YEARS
    )
    cuda.synchronize()
    t1 = datetime.now()
    
    # Cleanup
    del d_accounts, d_returns, d_mortality, d_states, d_cashflows
    
    return (t1 - t0).total_seconds()


def run_time_first(n_accounts, accounts, returns, mortality):
    """Run time-first approach and return execution time."""
    d_accounts = cuda.to_device(accounts)
    d_returns = cuda.to_device(returns)
    d_mortality = cuda.to_device(mortality)
    
    d_state_A = cuda.device_array((n_accounts, NUM_EXT_SCENARIOS, STATE_SIZE), dtype=np.float32)
    d_state_B = cuda.device_array((n_accounts, NUM_EXT_SCENARIOS, STATE_SIZE), dtype=np.float32)
    d_cashflows_step = cuda.device_array((n_accounts, NUM_EXT_SCENARIOS, CF_SIZE), dtype=np.float32)
    
    threads_per_block = (16, 16)
    blocks_x = (n_accounts + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_y = (NUM_EXT_SCENARIOS + threads_per_block[1] - 1) // threads_per_block[1]
    grid = (blocks_x, blocks_y)
    
    # Warm-up run
    time_first_init_kernel[grid, threads_per_block](d_accounts, d_state_A, NUM_EXT_SCENARIOS)
    cuda.synchronize()
    state_prev, state_curr = d_state_A, d_state_B
    for month_idx in range(TOTAL_MONTHS):
        time_first_step_kernel[grid, threads_per_block](
            d_accounts, state_prev, state_curr, d_cashflows_step,
            d_returns, d_mortality, month_idx, NUM_EXT_SCENARIOS, FREQ_EVAL
        )
        state_prev, state_curr = state_curr, state_prev
    cuda.synchronize()
    
    # Timed run
    t0 = datetime.now()
    time_first_init_kernel[grid, threads_per_block](d_accounts, d_state_A, NUM_EXT_SCENARIOS)
    cuda.synchronize()
    state_prev, state_curr = d_state_A, d_state_B
    for month_idx in range(TOTAL_MONTHS):
        time_first_step_kernel[grid, threads_per_block](
            d_accounts, state_prev, state_curr, d_cashflows_step,
            d_returns, d_mortality, month_idx, NUM_EXT_SCENARIOS, FREQ_EVAL
        )
        state_prev, state_curr = state_curr, state_prev
    cuda.synchronize()
    t1 = datetime.now()
    
    # Cleanup
    del d_accounts, d_returns, d_mortality, d_state_A, d_state_B, d_cashflows_step
    
    return (t1 - t0).total_seconds()


def run_benchmark():
    """Run benchmark across different account counts."""
    print("=" * 70)
    print("BENCHMARK: Account-First vs Time-First GPU Projection")
    print("=" * 70)
    
    if not cuda.is_available():
        print("ERROR: CUDA not available!")
        return
    
    gpu = cuda.get_current_device()
    print(f"GPU: {gpu.name.decode()}")
    print(f"Scenarios: {NUM_EXT_SCENARIOS} external × {NUM_YEARS} years × {FREQ_EVAL} months")
    print()
    
    # Test different account counts
    account_counts = [100, 500, 1000, 2000, 5000, 10000, 20000, 50000]
    
    print(f"{'Accounts':>10} | {'Account-First':>14} | {'Time-First':>14} | {'Speedup':>10} | {'Winner':>12}")
    print("-" * 70)
    
    results = []
    
    for n_accounts in account_counts:
        # Check memory requirements
        state_mem = n_accounts * NUM_EXT_SCENARIOS * NUM_YEARS * STATE_SIZE * 4
        cf_mem = n_accounts * NUM_EXT_SCENARIOS * TOTAL_MONTHS * CF_SIZE * 4
        total_mem = (state_mem + cf_mem) / 1024**3  # GB
        
        if total_mem > 16:  # Skip if would exceed typical GPU memory
            print(f"{n_accounts:>10} | {'SKIPPED (memory)':>14} | {'-':>14} | {'-':>10} | {'-':>12}")
            continue
        
        try:
            accounts, returns, mortality = create_data(n_accounts)
            
            time_account_first = run_account_first(n_accounts, accounts, returns, mortality)
            time_time_first = run_time_first(n_accounts, accounts, returns, mortality)
            
            speedup = time_account_first / time_time_first
            winner = "Time-First" if speedup > 1 else "Account-First"
            
            print(f"{n_accounts:>10} | {time_account_first:>12.4f}s | {time_time_first:>12.4f}s | {speedup:>9.2f}x | {winner:>12}")
            
            results.append({
                'accounts': n_accounts,
                'account_first': time_account_first,
                'time_first': time_time_first,
                'speedup': speedup,
                'winner': winner
            })
            
        except Exception as e:
            print(f"{n_accounts:>10} | ERROR: {str(e)[:40]}")
    
    print("-" * 70)
    print()
    
    # Summary
    if results:
        tf_wins = sum(1 for r in results if r['winner'] == 'Time-First')
        af_wins = len(results) - tf_wins
        
        print("SUMMARY:")
        print(f"  Time-First wins: {tf_wins}/{len(results)} cases")
        print(f"  Account-First wins: {af_wins}/{len(results)} cases")
        
        # Find crossover point
        crossover = None
        for i, r in enumerate(results):
            if r['speedup'] < 1 and (i == 0 or results[i-1]['speedup'] >= 1):
                crossover = r['accounts']
                break
        
        if crossover:
            print(f"  Crossover point: ~{crossover} accounts (Account-First becomes faster)")
        else:
            print(f"  Time-First faster across all tested scales")
    
    print()
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark()

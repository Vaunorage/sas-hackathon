#!/usr/bin/env python3
"""
Time-First GPU Actuarial Projection Engine

Alternative architecture where time is the outer loop:
- Each timestep launches a kernel that processes ALL accounts in parallel
- State persists between kernel calls (current + previous buffers)
- Better for time-dependent global calculations and debugging

Comparison:
- Account-First: Thread(acc,scn) loops over time internally
- Time-First: Loop over time, each step processes all (acc,scn) in parallel
"""

import numpy as np
import math
from numba import cuda
from datetime import datetime

# =============================================================================
# CONSTANTS
# =============================================================================
NUM_ACCOUNTS = 100
NUM_EXT_SCENARIOS = 50
NUM_INT_SCENARIOS = 100
NUM_YEARS = 10
FREQ_EVAL = 12
TOTAL_MONTHS = NUM_YEARS * FREQ_EVAL

# State tensor indices (what we track per account/scenario at each timestep)
STATE_VM = 0                # Market value
STATE_GAR_DECES = 1         # Death guarantee
STATE_TX_SURVIE = 2         # Survival rate
STATE_AGE = 3               # Current age
STATE_SIZE = 4

# Cashflow indices
CF_PRIMES = 0
CF_PREST_DECES = 1
CF_HON_GEST = 2
CF_RETRAIT = 3
CF_SIZE = 4

# Account data indices
ACC_VM_INIT = 0
ACC_AGE_INIT = 1
ACC_GAR_DECES = 2
ACC_PC_FRAIS = 3
ACC_SIZE = 4


# =============================================================================
# TIME-FIRST KERNELS
# =============================================================================

@cuda.jit
def init_state_kernel(
    accounts,       # (n_accounts, ACC_SIZE)
    state_curr,     # (n_accounts, n_scenarios, STATE_SIZE) - output
    n_scenarios,
):
    """
    Initialize state at t=0 from account data.
    Each thread handles one (account, scenario) pair.
    """
    acc_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    scn_idx = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    
    n_accounts = accounts.shape[0]
    if acc_idx >= n_accounts or scn_idx >= n_scenarios:
        return
    
    # Initialize state from account data
    state_curr[acc_idx, scn_idx, STATE_VM] = accounts[acc_idx, ACC_VM_INIT]
    state_curr[acc_idx, scn_idx, STATE_GAR_DECES] = accounts[acc_idx, ACC_GAR_DECES]
    state_curr[acc_idx, scn_idx, STATE_TX_SURVIE] = 1.0  # Start with 100% survival
    state_curr[acc_idx, scn_idx, STATE_AGE] = accounts[acc_idx, ACC_AGE_INIT]


@cuda.jit
def timestep_kernel(
    accounts,           # (n_accounts, ACC_SIZE)
    state_prev,         # (n_accounts, n_scenarios, STATE_SIZE) - input (previous timestep)
    state_curr,         # (n_accounts, n_scenarios, STATE_SIZE) - output (current timestep)
    cashflows,          # (n_accounts, n_scenarios, CF_SIZE) - output for this timestep
    returns_lookup,     # (n_scenarios, n_years) - market returns
    mortality_lookup,   # (max_age,)
    month_idx,          # Current month index (0 to TOTAL_MONTHS-1)
    n_scenarios,
    freq_eval,
):
    """
    Process ONE timestep for ALL accounts/scenarios in parallel.
    
    This kernel is called once per month. All accounts advance together.
    """
    acc_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    scn_idx = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    
    n_accounts = accounts.shape[0]
    if acc_idx >= n_accounts or scn_idx >= n_scenarios:
        return
    
    # Load previous state
    vm = state_prev[acc_idx, scn_idx, STATE_VM]
    gar_deces = state_prev[acc_idx, scn_idx, STATE_GAR_DECES]
    tx_survie = state_prev[acc_idx, scn_idx, STATE_TX_SURVIE]
    age = state_prev[acc_idx, scn_idx, STATE_AGE]
    
    # Load account parameters
    pc_frais = accounts[acc_idx, ACC_PC_FRAIS]
    
    # Calculate year index for return lookup
    year_idx = month_idx // freq_eval
    month_in_year = month_idx % freq_eval
    
    # Apply market return at start of each year (month 0)
    if month_in_year == 0:
        market_return = returns_lookup[scn_idx, year_idx]
        vm = vm * (1.0 + market_return)
    
    # Mortality calculation
    age_idx = min(int(age), mortality_lookup.shape[0] - 1)
    qx_annual = mortality_lookup[age_idx]
    qx_monthly = 1.0 - math.pow(1.0 - qx_annual, 1.0 / freq_eval)
    
    # Update survival
    tx_survie_prev = tx_survie
    tx_survie = tx_survie * (1.0 - qx_monthly)
    deaths = tx_survie_prev - tx_survie
    
    # Calculate cashflows
    hon_gest = vm * pc_frais / freq_eval * tx_survie
    primes = gar_deces * 0.005 / freq_eval * tx_survie
    prest_deces = max(0.0, gar_deces - vm) * deaths
    retrait = vm * 0.04 / freq_eval * tx_survie
    
    # Update VM after withdrawal
    vm = vm - retrait
    
    # Age increases at year boundary
    if month_in_year == freq_eval - 1:
        age = age + 1
    
    # Store current state
    state_curr[acc_idx, scn_idx, STATE_VM] = vm
    state_curr[acc_idx, scn_idx, STATE_GAR_DECES] = gar_deces
    state_curr[acc_idx, scn_idx, STATE_TX_SURVIE] = tx_survie
    state_curr[acc_idx, scn_idx, STATE_AGE] = age
    
    # Store cashflows for this timestep
    cashflows[acc_idx, scn_idx, CF_PRIMES] = primes
    cashflows[acc_idx, scn_idx, CF_PREST_DECES] = prest_deces
    cashflows[acc_idx, scn_idx, CF_HON_GEST] = hon_gest
    cashflows[acc_idx, scn_idx, CF_RETRAIT] = retrait


@cuda.jit
def nested_valuation_timestep_kernel(
    state_curr,         # (n_accounts, n_ext_scenarios, STATE_SIZE) - current outer state
    accounts,           # (n_accounts, ACC_SIZE)
    rn_returns,         # (n_int_scenarios, n_years)
    rn_forward_rates,   # (n_int_scenarios, n_years)
    mortality_lookup,   # (max_age,)
    metrics,            # (n_accounts, n_ext_scenarios, 2) - output [reserve, capital]
    current_year,       # Current year index
    n_ext_scenarios,
    n_int_scenarios,
    n_years,
):
    """
    Nested valuation for ONE timestep (end of year).
    Runs internal scenarios from current state to compute reserve/capital.
    """
    acc_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    scn_idx = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    
    n_accounts = state_curr.shape[0]
    if acc_idx >= n_accounts or scn_idx >= n_ext_scenarios:
        return
    
    # Get current state
    vm_start = state_curr[acc_idx, scn_idx, STATE_VM]
    age_start = state_curr[acc_idx, scn_idx, STATE_AGE]
    tx_survie_start = state_curr[acc_idx, scn_idx, STATE_TX_SURVIE]
    gar_deces = state_curr[acc_idx, scn_idx, STATE_GAR_DECES]
    pc_frais = accounts[acc_idx, ACC_PC_FRAIS]
    
    remaining_years = n_years - current_year - 1
    if remaining_years <= 0:
        metrics[acc_idx, scn_idx, 0] = 0.0
        metrics[acc_idx, scn_idx, 1] = 0.0
        return
    
    sum_pv = 0.0
    sum_pv_sq = 0.0
    
    for int_scn in range(n_int_scenarios):
        vm = vm_start
        tx_survie = tx_survie_start
        age = age_start
        pv_total = 0.0
        discount_factor = 1.0
        
        for t in range(remaining_years):
            proj_year = current_year + 1 + t
            if proj_year >= n_years:
                break
            
            rn_return = rn_returns[int_scn, proj_year]
            fwd_rate = rn_forward_rates[int_scn, proj_year]
            
            discount_factor = discount_factor * math.exp(-fwd_rate)
            vm = vm * (1.0 + rn_return)
            
            age_idx = min(int(age), mortality_lookup.shape[0] - 1)
            qx = mortality_lookup[age_idx]
            tx_survie_prev = tx_survie
            tx_survie = tx_survie * (1.0 - qx)
            deaths = tx_survie_prev - tx_survie
            
            hon_gest = vm * pc_frais * tx_survie
            primes = gar_deces * 0.005 * tx_survie
            prest_deces = max(0.0, gar_deces - vm) * deaths
            
            net_cf = hon_gest + primes - prest_deces
            pv_total = pv_total + net_cf * discount_factor
            
            age = age + 1
        
        sum_pv = sum_pv + pv_total
        sum_pv_sq = sum_pv_sq + pv_total * pv_total
    
    reserve = sum_pv / n_int_scenarios
    variance = (sum_pv_sq / n_int_scenarios) - (reserve * reserve)
    std_dev = math.sqrt(max(0.0, variance))
    capital = reserve - 2.0 * std_dev
    
    metrics[acc_idx, scn_idx, 0] = reserve
    metrics[acc_idx, scn_idx, 1] = capital


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def create_sample_data():
    """Generate sample account and lookup data."""
    np.random.seed(42)
    
    accounts = np.zeros((NUM_ACCOUNTS, ACC_SIZE), dtype=np.float32)
    accounts[:, ACC_VM_INIT] = np.random.uniform(50000, 500000, NUM_ACCOUNTS)
    accounts[:, ACC_AGE_INIT] = np.random.uniform(40, 70, NUM_ACCOUNTS)
    accounts[:, ACC_GAR_DECES] = accounts[:, ACC_VM_INIT] * 1.1
    accounts[:, ACC_PC_FRAIS] = np.random.uniform(0.01, 0.03, NUM_ACCOUNTS)
    
    returns = np.random.normal(0.05, 0.15, (NUM_EXT_SCENARIOS, NUM_YEARS)).astype(np.float32)
    rn_returns = np.random.normal(0.03, 0.10, (NUM_INT_SCENARIOS, NUM_YEARS)).astype(np.float32)
    rn_forward_rates = np.full((NUM_INT_SCENARIOS, NUM_YEARS), 0.03, dtype=np.float32)
    rn_forward_rates += np.random.normal(0, 0.005, rn_forward_rates.shape).astype(np.float32)
    
    max_age = 120
    mortality = np.zeros(max_age, dtype=np.float32)
    for age in range(max_age):
        mortality[age] = min(0.0001 * math.exp(0.08 * age), 1.0)
    
    return accounts, returns, rn_returns, rn_forward_rates, mortality


def run_time_first_projection():
    """Main projection using time-first approach."""
    print("=" * 60)
    print("TIME-FIRST GPU ACTUARIAL PROJECTION")
    print("=" * 60)
    
    if not cuda.is_available():
        print("ERROR: CUDA is not available!")
        return
    
    gpu = cuda.get_current_device()
    print(f"GPU: {gpu.name.decode()}")
    
    # Create data
    print("\n[1] Creating sample data...")
    accounts, returns, rn_returns, rn_forward_rates, mortality = create_sample_data()
    print(f"    Accounts: {NUM_ACCOUNTS}")
    print(f"    External scenarios: {NUM_EXT_SCENARIOS}")
    print(f"    Total months: {TOTAL_MONTHS}")
    
    # Transfer to GPU
    print("\n[2] Allocating GPU memory...")
    d_accounts = cuda.to_device(accounts)
    d_returns = cuda.to_device(returns)
    d_rn_returns = cuda.to_device(rn_returns)
    d_rn_forward_rates = cuda.to_device(rn_forward_rates)
    d_mortality = cuda.to_device(mortality)
    
    # Double-buffer for state (ping-pong pattern)
    d_state_A = cuda.device_array((NUM_ACCOUNTS, NUM_EXT_SCENARIOS, STATE_SIZE), dtype=np.float32)
    d_state_B = cuda.device_array((NUM_ACCOUNTS, NUM_EXT_SCENARIOS, STATE_SIZE), dtype=np.float32)
    
    # Single timestep cashflow buffer (reused each month)
    d_cashflows_step = cuda.device_array((NUM_ACCOUNTS, NUM_EXT_SCENARIOS, CF_SIZE), dtype=np.float32)
    
    # Accumulated cashflows on CPU (or could accumulate on GPU)
    h_cashflows_total = np.zeros((NUM_ACCOUNTS, NUM_EXT_SCENARIOS, TOTAL_MONTHS, CF_SIZE), dtype=np.float32)
    
    # Metrics buffer (updated at each year end)
    d_metrics = cuda.device_array((NUM_ACCOUNTS, NUM_EXT_SCENARIOS, 2), dtype=np.float32)
    h_metrics_history = np.zeros((NUM_ACCOUNTS, NUM_EXT_SCENARIOS, NUM_YEARS, 2), dtype=np.float32)
    
    # Grid configuration
    threads_per_block = (16, 16)
    blocks_x = (NUM_ACCOUNTS + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_y = (NUM_EXT_SCENARIOS + threads_per_block[1] - 1) // threads_per_block[1]
    grid = (blocks_x, blocks_y)
    
    print(f"    State buffer: {d_state_A.nbytes / 1024:.2f} KB × 2 (double buffer)")
    print(f"    Cashflow step buffer: {d_cashflows_step.nbytes / 1024:.2f} KB")
    
    # === INITIALIZE STATE ===
    print("\n[3] Initializing state at t=0...")
    init_state_kernel[grid, threads_per_block](
        d_accounts, d_state_A, NUM_EXT_SCENARIOS
    )
    cuda.synchronize()
    
    # === TIME LOOP ===
    print(f"\n[4] Running time-first projection ({TOTAL_MONTHS} months)...")
    t0 = datetime.now()
    
    # Ping-pong buffers
    state_prev = d_state_A
    state_curr = d_state_B
    
    for month_idx in range(TOTAL_MONTHS):
        # Run timestep kernel
        timestep_kernel[grid, threads_per_block](
            d_accounts,
            state_prev,
            state_curr,
            d_cashflows_step,
            d_returns,
            d_mortality,
            month_idx,
            NUM_EXT_SCENARIOS,
            FREQ_EVAL,
        )
        cuda.synchronize()
        
        # Copy cashflows for this timestep to CPU
        h_cashflows_total[:, :, month_idx, :] = d_cashflows_step.copy_to_host()
        
        # At year end, run nested valuation
        if (month_idx + 1) % FREQ_EVAL == 0:
            year_idx = month_idx // FREQ_EVAL
            
            nested_valuation_timestep_kernel[grid, threads_per_block](
                state_curr,
                d_accounts,
                d_rn_returns,
                d_rn_forward_rates,
                d_mortality,
                d_metrics,
                year_idx,
                NUM_EXT_SCENARIOS,
                NUM_INT_SCENARIOS,
                NUM_YEARS,
            )
            cuda.synchronize()
            
            # Store metrics for this year
            h_metrics_history[:, :, year_idx, :] = d_metrics.copy_to_host()
            
            print(f"    Year {year_idx + 1}/{NUM_YEARS} complete")
        
        # Swap buffers (ping-pong)
        state_prev, state_curr = state_curr, state_prev
    
    t1 = datetime.now()
    print(f"\n    Total time: {(t1-t0).total_seconds():.3f}s")
    print(f"    Kernel launches: {TOTAL_MONTHS} timestep + {NUM_YEARS} valuation = {TOTAL_MONTHS + NUM_YEARS}")
    
    # === RESULTS ===
    print("\n[5] Computing summary statistics...")
    
    # Cashflow summary
    mean_cashflows = h_cashflows_total.mean(axis=1)  # Average across scenarios
    total_primes = mean_cashflows[:, :, CF_PRIMES].sum()
    total_prest_deces = mean_cashflows[:, :, CF_PREST_DECES].sum()
    total_hon_gest = mean_cashflows[:, :, CF_HON_GEST].sum()
    total_retrait = mean_cashflows[:, :, CF_RETRAIT].sum()
    
    print(f"\n    CASHFLOW SUMMARY:")
    print(f"    ─────────────────────────────────────────────")
    print(f"    Total Guarantee Premiums:  ${total_primes:>15,.2f}")
    print(f"    Total Death Benefits:      ${total_prest_deces:>15,.2f}")
    print(f"    Total Management Fees:     ${total_hon_gest:>15,.2f}")
    print(f"    Total Withdrawals:         ${total_retrait:>15,.2f}")
    
    # Valuation summary (last year)
    final_reserve = h_metrics_history[:, :, -1, 0].mean(axis=1).sum()
    final_capital = h_metrics_history[:, :, -1, 1].mean(axis=1).sum()
    
    print(f"\n    VALUATION SUMMARY (Year {NUM_YEARS}):")
    print(f"    ─────────────────────────────────────────────")
    print(f"    Total Portfolio Reserve:   ${final_reserve:>15,.2f}")
    print(f"    Total Portfolio Capital:   ${final_capital:>15,.2f}")
    print(f"    SCR:                       ${final_capital - final_reserve:>15,.2f}")
    
    # Final state
    h_final_state = state_prev.copy_to_host()  # Last written buffer
    final_vm = h_final_state[:, :, STATE_VM].mean()
    final_survie = h_final_state[:, :, STATE_TX_SURVIE].mean()
    
    print(f"\n    FINAL STATE:")
    print(f"    ─────────────────────────────────────────────")
    print(f"    Avg Market Value:          ${final_vm:>15,.2f}")
    print(f"    Avg Survival Rate:         {final_survie:>15.2%}")
    
    print("\n" + "=" * 60)
    print("TIME-FIRST PROJECTION COMPLETE")
    print("=" * 60)
    
    return h_cashflows_total, h_metrics_history, h_final_state


if __name__ == "__main__":
    run_time_first_projection()

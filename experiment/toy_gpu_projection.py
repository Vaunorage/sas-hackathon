#!/usr/bin/env python3
"""
Toy GPU Actuarial Projection Engine

A simplified, self-contained version demonstrating the two-kernel architecture:
- Kernel A: External scenario generator (market scenarios, cashflow projection)
- Kernel B: Nested valuation (present value calculation under risk-neutral scenarios)

This is a minimal example for educational purposes.
"""

import numpy as np
import math
from numba import cuda
from datetime import datetime

# =============================================================================
# CONSTANTS
# =============================================================================
NUM_ACCOUNTS = 100
NUM_EXT_SCENARIOS = 50      # External (market) scenarios
NUM_INT_SCENARIOS = 100     # Internal (risk-neutral) scenarios for nested valuation
NUM_YEARS = 10
FREQ_EVAL = 12              # Monthly evaluation

# State tensor indices (what we track per account/scenario/year)
STATE_VM = 0                # Market value
STATE_GAR_DECES = 1         # Death guarantee
STATE_TX_SURVIE = 2         # Survival rate
STATE_AGE = 3               # Current age
STATE_SIZE = 4

# Cashflow indices
CF_PRIMES = 0               # Guarantee premiums (revenue)
CF_PREST_DECES = 1          # Death benefits (payout)
CF_HON_GEST = 2             # Management fees (revenue)
CF_RETRAIT = 3              # Withdrawals
CF_SIZE = 4

# Account data indices
ACC_VM_INIT = 0             # Initial market value
ACC_AGE_INIT = 1            # Initial age
ACC_GAR_DECES = 2           # Death guarantee amount
ACC_PC_FRAIS = 3            # Fee percentage
ACC_SIZE = 4


# =============================================================================
# CUDA KERNELS
# =============================================================================

@cuda.jit
def external_generator_kernel(
    accounts,           # (n_accounts, ACC_SIZE) - account data
    returns_lookup,     # (n_ext_scenarios, n_years) - market returns
    mortality_lookup,   # (max_age,) - mortality rates by age
    states,             # (n_accounts, n_ext_scenarios, n_years, STATE_SIZE) - output
    cashflows,          # (n_accounts, n_ext_scenarios, n_years*12, CF_SIZE) - output
    n_ext_scenarios,
    n_years,
):
    """
    Kernel A: External Scenario Generator
    
    Projects each account through time under each external market scenario.
    Calculates market value evolution, mortality, and cashflows.
    """
    # 2D grid: (account_idx, scenario_idx)
    acc_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    scn_idx = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    
    n_accounts = accounts.shape[0]
    if acc_idx >= n_accounts or scn_idx >= n_ext_scenarios:
        return
    
    # Load account data
    vm = accounts[acc_idx, ACC_VM_INIT]
    age = accounts[acc_idx, ACC_AGE_INIT]
    gar_deces = accounts[acc_idx, ACC_GAR_DECES]
    pc_frais = accounts[acc_idx, ACC_PC_FRAIS]
    
    tx_survie = 1.0  # Start with 100% survival
    
    # Project through years
    for year in range(n_years):
        # Get market return for this scenario/year
        market_return = returns_lookup[scn_idx, year]
        
        # Apply return to market value (annual)
        vm_before = vm
        vm = vm * (1.0 + market_return)
        
        # Monthly calculations
        for month in range(FREQ_EVAL):
            month_idx = year * FREQ_EVAL + month
            
            # Mortality lookup (clamp age to valid range)
            age_idx = min(int(age), mortality_lookup.shape[0] - 1)
            qx_annual = mortality_lookup[age_idx]
            qx_monthly = 1.0 - math.pow(1.0 - qx_annual, 1.0 / FREQ_EVAL)
            
            # Update survival rate
            tx_survie_prev = tx_survie
            tx_survie = tx_survie * (1.0 - qx_monthly)
            
            # Calculate cashflows (scaled by survival)
            # Management fees (revenue)
            hon_gest = vm * pc_frais / FREQ_EVAL * tx_survie
            
            # Guarantee premium (simplified: % of guarantee amount)
            primes = gar_deces * 0.005 / FREQ_EVAL * tx_survie
            
            # Death benefit (payout when people die)
            deaths = tx_survie_prev - tx_survie
            prest_deces = max(0.0, gar_deces - vm) * deaths  # Pay shortfall
            
            # Withdrawal (simplified: 4% annual withdrawal rate)
            retrait = vm * 0.04 / FREQ_EVAL * tx_survie
            vm = vm - retrait
            
            # Store cashflows
            cashflows[acc_idx, scn_idx, month_idx, CF_PRIMES] = primes
            cashflows[acc_idx, scn_idx, month_idx, CF_PREST_DECES] = prest_deces
            cashflows[acc_idx, scn_idx, month_idx, CF_HON_GEST] = hon_gest
            cashflows[acc_idx, scn_idx, month_idx, CF_RETRAIT] = retrait
        
        # Store end-of-year state
        states[acc_idx, scn_idx, year, STATE_VM] = vm
        states[acc_idx, scn_idx, year, STATE_GAR_DECES] = gar_deces
        states[acc_idx, scn_idx, year, STATE_TX_SURVIE] = tx_survie
        states[acc_idx, scn_idx, year, STATE_AGE] = age + 1
        
        age = age + 1


@cuda.jit
def nested_valuation_kernel(
    states,             # (n_accounts, n_ext_scenarios, n_years, STATE_SIZE)
    accounts,           # (n_accounts, ACC_SIZE)
    rn_returns,         # (n_int_scenarios, n_years) - risk-neutral returns
    rn_forward_rates,   # (n_int_scenarios, n_years) - discount rates
    mortality_lookup,   # (max_age,)
    metrics,            # (n_accounts, n_ext_scenarios, n_years, 2) - output [reserve, capital]
    n_int_scenarios,
    n_years,
):
    """
    Kernel B: Nested Valuation
    
    For each (account, ext_scenario, year) node, run internal scenarios
    to compute present value of future cashflows (reserve) and capital requirement.
    """
    # 1D grid: flattened (account, ext_scenario, year)
    idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    
    n_accounts = states.shape[0]
    n_ext_scenarios = states.shape[1]
    total_nodes = n_accounts * n_ext_scenarios * n_years
    
    if idx >= total_nodes:
        return
    
    # Decode indices
    acc_idx = idx // (n_ext_scenarios * n_years)
    remainder = idx % (n_ext_scenarios * n_years)
    scn_idx = remainder // n_years
    year_idx = remainder % n_years
    
    # Get starting state for this node
    vm_start = states[acc_idx, scn_idx, year_idx, STATE_VM]
    age_start = states[acc_idx, scn_idx, year_idx, STATE_AGE]
    tx_survie_start = states[acc_idx, scn_idx, year_idx, STATE_TX_SURVIE]
    gar_deces = states[acc_idx, scn_idx, year_idx, STATE_GAR_DECES]
    pc_frais = accounts[acc_idx, ACC_PC_FRAIS]
    
    # Remaining projection years
    remaining_years = n_years - year_idx - 1
    if remaining_years <= 0:
        metrics[acc_idx, scn_idx, year_idx, 0] = 0.0  # Reserve
        metrics[acc_idx, scn_idx, year_idx, 1] = 0.0  # Capital
        return
    
    # Run internal scenarios and accumulate PV
    sum_pv = 0.0
    sum_pv_sq = 0.0  # For VaR/capital calculation
    
    for int_scn in range(n_int_scenarios):
        vm = vm_start
        tx_survie = tx_survie_start
        age = age_start
        pv_total = 0.0
        discount_factor = 1.0
        
        # Project forward under risk-neutral measure
        for t in range(remaining_years):
            proj_year = year_idx + 1 + t
            if proj_year >= n_years:
                break
            
            # Risk-neutral return and discount rate
            rn_return = rn_returns[int_scn, proj_year]
            fwd_rate = rn_forward_rates[int_scn, proj_year]
            
            # Update discount factor
            discount_factor = discount_factor * math.exp(-fwd_rate)
            
            # Project VM
            vm = vm * (1.0 + rn_return)
            
            # Mortality
            age_idx = min(int(age), mortality_lookup.shape[0] - 1)
            qx = mortality_lookup[age_idx]
            tx_survie_prev = tx_survie
            tx_survie = tx_survie * (1.0 - qx)
            deaths = tx_survie_prev - tx_survie
            
            # Cashflows (simplified annual)
            hon_gest = vm * pc_frais * tx_survie
            primes = gar_deces * 0.005 * tx_survie
            prest_deces = max(0.0, gar_deces - vm) * deaths
            
            # Net cashflow (revenue - payouts)
            net_cf = hon_gest + primes - prest_deces
            
            # Accumulate PV
            pv_total = pv_total + net_cf * discount_factor
            
            age = age + 1
        
        sum_pv = sum_pv + pv_total
        sum_pv_sq = sum_pv_sq + pv_total * pv_total
    
    # Average PV = Best Estimate Reserve
    reserve = sum_pv / n_int_scenarios
    
    # Capital = Reserve + Risk Margin (simplified: use std dev as proxy)
    variance = (sum_pv_sq / n_int_scenarios) - (reserve * reserve)
    std_dev = math.sqrt(max(0.0, variance))
    capital = reserve - 2.0 * std_dev  # 97.5% VaR approximation (negative = need capital)
    
    metrics[acc_idx, scn_idx, year_idx, 0] = reserve
    metrics[acc_idx, scn_idx, year_idx, 1] = capital


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def create_sample_data():
    """Generate sample account and lookup data."""
    np.random.seed(42)
    
    # Account data: (n_accounts, ACC_SIZE)
    accounts = np.zeros((NUM_ACCOUNTS, ACC_SIZE), dtype=np.float32)
    accounts[:, ACC_VM_INIT] = np.random.uniform(50000, 500000, NUM_ACCOUNTS)  # Market value
    accounts[:, ACC_AGE_INIT] = np.random.uniform(40, 70, NUM_ACCOUNTS)        # Age
    accounts[:, ACC_GAR_DECES] = accounts[:, ACC_VM_INIT] * 1.1                # 110% guarantee
    accounts[:, ACC_PC_FRAIS] = np.random.uniform(0.01, 0.03, NUM_ACCOUNTS)    # 1-3% fees
    
    # Market returns: (n_ext_scenarios, n_years)
    # Simulate different market scenarios
    returns = np.random.normal(0.05, 0.15, (NUM_EXT_SCENARIOS, NUM_YEARS)).astype(np.float32)
    
    # Risk-neutral returns: (n_int_scenarios, n_years)
    # Under risk-neutral measure, expected return = risk-free rate
    rn_returns = np.random.normal(0.03, 0.10, (NUM_INT_SCENARIOS, NUM_YEARS)).astype(np.float32)
    
    # Forward rates: (n_int_scenarios, n_years)
    rn_forward_rates = np.full((NUM_INT_SCENARIOS, NUM_YEARS), 0.03, dtype=np.float32)
    rn_forward_rates += np.random.normal(0, 0.005, rn_forward_rates.shape).astype(np.float32)
    
    # Mortality table: qx by age (simplified)
    max_age = 120
    mortality = np.zeros(max_age, dtype=np.float32)
    for age in range(max_age):
        # Gompertz-like mortality curve
        mortality[age] = 0.0001 * math.exp(0.08 * age)
        mortality[age] = min(mortality[age], 1.0)
    
    return accounts, returns, rn_returns, rn_forward_rates, mortality


def run_projection():
    """Main projection function."""
    print("=" * 60)
    print("TOY GPU ACTUARIAL PROJECTION ENGINE")
    print("=" * 60)
    
    # Check GPU availability
    if not cuda.is_available():
        print("ERROR: CUDA is not available!")
        return
    
    gpu = cuda.get_current_device()
    print(f"GPU: {gpu.name.decode()}")
    
    # Create sample data
    print("\n[1] Creating sample data...")
    accounts, returns, rn_returns, rn_forward_rates, mortality = create_sample_data()
    print(f"    Accounts: {NUM_ACCOUNTS}")
    print(f"    External scenarios: {NUM_EXT_SCENARIOS}")
    print(f"    Internal scenarios: {NUM_INT_SCENARIOS}")
    print(f"    Projection years: {NUM_YEARS}")
    
    # Allocate GPU arrays
    print("\n[2] Allocating GPU memory...")
    d_accounts = cuda.to_device(accounts)
    d_returns = cuda.to_device(returns)
    d_rn_returns = cuda.to_device(rn_returns)
    d_rn_forward_rates = cuda.to_device(rn_forward_rates)
    d_mortality = cuda.to_device(mortality)
    
    # Output tensors
    d_states = cuda.device_array(
        (NUM_ACCOUNTS, NUM_EXT_SCENARIOS, NUM_YEARS, STATE_SIZE), 
        dtype=np.float32
    )
    d_cashflows = cuda.device_array(
        (NUM_ACCOUNTS, NUM_EXT_SCENARIOS, NUM_YEARS * FREQ_EVAL, CF_SIZE),
        dtype=np.float32
    )
    d_metrics = cuda.device_array(
        (NUM_ACCOUNTS, NUM_EXT_SCENARIOS, NUM_YEARS, 2),
        dtype=np.float32
    )
    
    # Calculate memory usage
    state_mem = d_states.nbytes / 1024**2
    cf_mem = d_cashflows.nbytes / 1024**2
    metrics_mem = d_metrics.nbytes / 1024**2
    print(f"    State tensor: {state_mem:.2f} MB")
    print(f"    Cashflow tensor: {cf_mem:.2f} MB")
    print(f"    Metrics tensor: {metrics_mem:.2f} MB")
    print(f"    Total: {state_mem + cf_mem + metrics_mem:.2f} MB")
    
    # === KERNEL A: External Generator ===
    print("\n[3] Running Kernel A (External Generator)...")
    threads_per_block = (16, 16)
    blocks_x = (NUM_ACCOUNTS + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_y = (NUM_EXT_SCENARIOS + threads_per_block[1] - 1) // threads_per_block[1]
    grid_A = (blocks_x, blocks_y)
    
    t0 = datetime.now()
    external_generator_kernel[grid_A, threads_per_block](
        d_accounts, d_returns, d_mortality,
        d_states, d_cashflows,
        NUM_EXT_SCENARIOS, NUM_YEARS
    )
    cuda.synchronize()
    t1 = datetime.now()
    print(f"    Kernel A time: {(t1-t0).total_seconds():.3f}s")
    print(f"    Processed: {NUM_ACCOUNTS * NUM_EXT_SCENARIOS:,} account-scenarios")
    
    # === KERNEL B: Nested Valuation ===
    print("\n[4] Running Kernel B (Nested Valuation)...")
    total_nodes = NUM_ACCOUNTS * NUM_EXT_SCENARIOS * NUM_YEARS
    threads_per_block_B = 256
    blocks_B = (total_nodes + threads_per_block_B - 1) // threads_per_block_B
    
    t0 = datetime.now()
    nested_valuation_kernel[blocks_B, threads_per_block_B](
        d_states, d_accounts, d_rn_returns, d_rn_forward_rates, d_mortality,
        d_metrics,
        NUM_INT_SCENARIOS, NUM_YEARS
    )
    cuda.synchronize()
    t1 = datetime.now()
    print(f"    Kernel B time: {(t1-t0).total_seconds():.3f}s")
    print(f"    Processed: {total_nodes:,} nodes × {NUM_INT_SCENARIOS} internal scenarios")
    print(f"    Total scenarios: {total_nodes * NUM_INT_SCENARIOS:,}")
    
    # === Copy results back ===
    print("\n[5] Copying results to CPU...")
    h_states = d_states.copy_to_host()
    h_cashflows = d_cashflows.copy_to_host()
    h_metrics = d_metrics.copy_to_host()
    
    # === Compute summary statistics ===
    print("\n[6] Computing summary statistics...")
    
    # Average cashflows across scenarios
    mean_cashflows = h_cashflows.mean(axis=1)  # (accounts, months, CF_SIZE)
    total_primes = mean_cashflows[:, :, CF_PRIMES].sum()
    total_prest_deces = mean_cashflows[:, :, CF_PREST_DECES].sum()
    total_hon_gest = mean_cashflows[:, :, CF_HON_GEST].sum()
    total_retrait = mean_cashflows[:, :, CF_RETRAIT].sum()
    
    print(f"\n    CASHFLOW SUMMARY (averaged across scenarios):")
    print(f"    ─────────────────────────────────────────────")
    print(f"    Total Guarantee Premiums:  ${total_primes:>15,.2f}")
    print(f"    Total Death Benefits:      ${total_prest_deces:>15,.2f}")
    print(f"    Total Management Fees:     ${total_hon_gest:>15,.2f}")
    print(f"    Total Withdrawals:         ${total_retrait:>15,.2f}")
    
    # Reserves and capital (average across scenarios and years)
    mean_reserve = h_metrics[:, :, :, 0].mean()
    mean_capital = h_metrics[:, :, :, 1].mean()
    total_reserve = h_metrics[:, :, -1, 0].mean(axis=1).sum()  # Last year, sum across accounts
    total_capital = h_metrics[:, :, -1, 1].mean(axis=1).sum()
    
    print(f"\n    VALUATION SUMMARY:")
    print(f"    ─────────────────────────────────────────────")
    print(f"    Average Reserve (BE):      ${mean_reserve:>15,.2f}")
    print(f"    Average Capital Req:       ${mean_capital:>15,.2f}")
    print(f"    Total Portfolio Reserve:   ${total_reserve:>15,.2f}")
    print(f"    Total Portfolio Capital:   ${total_capital:>15,.2f}")
    print(f"    SCR (Capital - Reserve):   ${total_capital - total_reserve:>15,.2f}")
    
    # Final market values
    final_vm = h_states[:, :, -1, STATE_VM].mean()
    final_survie = h_states[:, :, -1, STATE_TX_SURVIE].mean()
    
    print(f"\n    FINAL STATE (Year {NUM_YEARS}):")
    print(f"    ─────────────────────────────────────────────")
    print(f"    Avg Market Value:          ${final_vm:>15,.2f}")
    print(f"    Avg Survival Rate:         {final_survie:>15.2%}")
    
    print("\n" + "=" * 60)
    print("PROJECTION COMPLETE")
    print("=" * 60)
    
    return h_states, h_cashflows, h_metrics


if __name__ == "__main__":
    run_projection()

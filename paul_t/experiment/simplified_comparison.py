#!/usr/bin/env python3
"""
Simplified comparison: Account-First vs Time-First GPU approaches.

Account-First: Each thread handles ONE account's FULL time series (all 120 months)
Time-First:    Each timestep is a separate kernel; ALL accounts processed together
"""

import numpy as np
import math
from numba import cuda

# Configuration
N_ACCOUNTS = 1000
N_SCENARIOS = 50
N_MONTHS = 120  # 10 years × 12 months


# =============================================================================
# APPROACH 1: ACCOUNT-FIRST (Single kernel, loop over time inside)
# =============================================================================

@cuda.jit
def account_first_kernel(vm_init, cashflows, returns, n_scenarios, n_months):
    """
    Each thread processes ONE (account, scenario) pair through ALL months.
    
    Thread (acc=5, scn=3) handles account 5, scenario 3 for months 0-119.
    """
    acc = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    scn = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    
    if acc >= vm_init.shape[0] or scn >= n_scenarios:
        return
    
    # Initialize state for this account
    vm = vm_init[acc]
    
    # Process all months sequentially (inside the thread)
    for month in range(n_months):
        year = month // 12
        
        # Apply annual return at start of each year
        if month % 12 == 0:
            vm = vm * (1.0 + returns[scn, year])
        
        # Calculate cashflow
        cashflows[acc, scn, month] = vm * 0.01  # 1% fee
        vm = vm * 0.995  # Deduct fee


# =============================================================================
# APPROACH 2: TIME-FIRST (Multiple kernels, one per timestep)
# =============================================================================

@cuda.jit
def time_first_kernel(vm_prev, vm_curr, cashflows, returns, month, n_scenarios):
    """
    ALL accounts processed for ONE month.
    
    Called 120 times (once per month). Each call processes all accounts.
    """
    acc = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    scn = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    
    if acc >= vm_prev.shape[0] or scn >= n_scenarios:
        return
    
    # Read previous state
    vm = vm_prev[acc, scn]
    
    # Apply annual return at start of each year
    year = month // 12
    if month % 12 == 0:
        vm = vm * (1.0 + returns[scn, year])
    
    # Calculate cashflow
    cashflows[acc, scn] = vm * 0.01
    vm = vm * 0.995
    
    # Write new state
    vm_curr[acc, scn] = vm


# =============================================================================
# RUN BOTH APPROACHES
# =============================================================================

def run_account_first():
    """Account-First: 1 kernel launch, loop inside."""
    vm_init = np.random.uniform(100000, 500000, N_ACCOUNTS).astype(np.float32)
    returns = np.random.normal(0.05, 0.15, (N_SCENARIOS, 10)).astype(np.float32)
    cashflows = np.zeros((N_ACCOUNTS, N_SCENARIOS, N_MONTHS), dtype=np.float32)
    
    d_vm = cuda.to_device(vm_init)
    d_returns = cuda.to_device(returns)
    d_cashflows = cuda.to_device(cashflows)
    
    threads = (16, 16)
    blocks = (
        (N_ACCOUNTS + 15) // 16,
        (N_SCENARIOS + 15) // 16
    )
    
    # ONE kernel launch
    account_first_kernel[blocks, threads](d_vm, d_cashflows, d_returns, N_SCENARIOS, N_MONTHS)
    cuda.synchronize()
    
    return d_cashflows.copy_to_host()


def run_time_first():
    """Time-First: 120 kernel launches, one per month."""
    vm_init = np.random.uniform(100000, 500000, N_ACCOUNTS).astype(np.float32)
    returns = np.random.normal(0.05, 0.15, (N_SCENARIOS, 10)).astype(np.float32)
    
    # Double-buffer for state (ping-pong)
    vm_A = np.zeros((N_ACCOUNTS, N_SCENARIOS), dtype=np.float32)
    vm_B = np.zeros((N_ACCOUNTS, N_SCENARIOS), dtype=np.float32)
    
    # Initialize
    for scn in range(N_SCENARIOS):
        vm_A[:, scn] = vm_init
    
    d_vm_A = cuda.to_device(vm_A)
    d_vm_B = cuda.to_device(vm_B)
    d_returns = cuda.to_device(returns)
    d_cashflows = cuda.device_array((N_ACCOUNTS, N_SCENARIOS), dtype=np.float32)
    
    threads = (16, 16)
    blocks = (
        (N_ACCOUNTS + 15) // 16,
        (N_SCENARIOS + 15) // 16
    )
    
    all_cashflows = np.zeros((N_ACCOUNTS, N_SCENARIOS, N_MONTHS), dtype=np.float32)
    
    # 120 kernel launches (one per month)
    vm_prev, vm_curr = d_vm_A, d_vm_B
    for month in range(N_MONTHS):
        time_first_kernel[blocks, threads](vm_prev, vm_curr, d_cashflows, d_returns, month, N_SCENARIOS)
        cuda.synchronize()
        
        all_cashflows[:, :, month] = d_cashflows.copy_to_host()
        vm_prev, vm_curr = vm_curr, vm_prev  # Swap buffers
    
    return all_cashflows


if __name__ == "__main__":
    from datetime import datetime
    
    print("=" * 60)
    print("SIMPLIFIED COMPARISON: Account-First vs Time-First")
    print("=" * 60)
    print(f"Accounts: {N_ACCOUNTS}, Scenarios: {N_SCENARIOS}, Months: {N_MONTHS}")
    print()
    
    # Warm-up
    _ = run_account_first()
    _ = run_time_first()
    
    # Benchmark
    t0 = datetime.now()
    cf1 = run_account_first()
    t1 = datetime.now()
    cf2 = run_time_first()
    t2 = datetime.now()
    
    time_af = (t1 - t0).total_seconds()
    time_tf = (t2 - t1).total_seconds()
    
    print(f"Account-First: {time_af:.4f}s  (1 kernel launch)")
    print(f"Time-First:    {time_tf:.4f}s  (120 kernel launches)")
    print(f"Speedup:       {time_tf/time_af:.1f}x faster with Account-First")
    print()
    print("=" * 60)

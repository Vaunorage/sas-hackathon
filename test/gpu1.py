import os
os.environ['NUMBA_CUDA_COMPUTE_CAPABILITY'] = '8.0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Force Numba to use lower PTX version
import numba
numba.config.CUDA_DEFAULT_PTX_CC = (8, 0)

import pandas as pd
import numpy as np
from numba import cuda, jit
import numba
from typing import Dict, Tuple, List
import warnings
import logging
import math

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants for array indexing
STATE_ACCOUNT_ID = 0
STATE_SCENARIO = 1
STATE_ACCOUNT_IDX = 2
STATE_AGE_DEB = 3
STATE_MT_VM_PROJ = 4
STATE_MT_GAR_DECES_PROJ = 5
STATE_TX_SURVIE = 6
STATE_AGE = 7
STATE_IS_TERMINATED = 8
STATE_SIZE = 9

# Constants for initial data indexing
DATA_MT_VM = 0
DATA_MT_GAR_DECES = 1
DATA_AGE_DEB = 2
DATA_TX_COMM_VENTE = 3
DATA_FRAIS_ACQUI = 4
DATA_PC_REVENU_FDS = 5
DATA_PC_HONORAIRES_GEST = 6
DATA_TX_COMM_MAINTIEN = 7
DATA_FRAIS_ADMIN = 8
DATA_FREQ_RESET_DECES = 9
DATA_MAX_RESET_DECES = 10
DATA_SIZE = 11


def load_input_data(data_path: str = ".", nb_accounts: int = None) -> Dict:
    """Load all input data files"""
    try:
        population = pd.read_csv(f"{data_path}/population_fixed.csv")
        if nb_accounts is not None:
            population = population.head(nb_accounts)
        rendement = pd.read_csv(f"{data_path}/rendement1.csv")
        tx_deces = pd.read_csv(f"{data_path}/tx_deces_fixed.csv")
        tx_interet = pd.read_csv(f"{data_path}/tx_interet_fixed.csv")
        tx_interet_int = pd.read_csv(f"{data_path}/tx_interet_int_fixed.csv")
        tx_retrait = pd.read_csv(f"{data_path}/tx_retrait_fixed.csv")

        if 'TYPE' in rendement.columns:
            rendement['TYPE'] = rendement['TYPE'].apply(
                lambda x: x.decode('utf-8') if isinstance(x, bytes) else str(x)
            )

        logger.info(f"Input files loaded - Population: {len(population)} accounts")
        return {
            'population': population,
            'rendement': rendement,
            'tx_deces': tx_deces,
            'tx_interet': tx_interet,
            'tx_interet_int': tx_interet_int,
            'tx_retrait': tx_retrait
        }
    except Exception as e:
        logger.error(f"Error loading input files: {e}")
        raise


def create_gpu_lookup_tables(data: Dict, max_age: int = 120, max_year: int = 50, max_scenarios: int = 1000) -> Dict:
    """Create GPU-friendly lookup tables as NumPy arrays"""
    mortality_array = np.zeros(max_age + 1, dtype=np.float64)
    for _, row in data['tx_deces'].iterrows():
        age = int(row['AGE'])
        if age <= max_age:
            mortality_array[age] = float(row['QX'])

    lapse_array = np.zeros(max_year + 1, dtype=np.float64)
    for _, row in data['tx_retrait'].iterrows():
        year = int(row['an_proj'])
        if year <= max_year:
            lapse_array[year] = float(row['WX'])

    discount_ext_array = np.ones(max_year + 1, dtype=np.float64)
    for _, row in data['tx_interet'].iterrows():
        year = int(row['an_proj'])
        if year <= max_year:
            discount_ext_array[year] = float(row['TX_ACTU'])

    discount_int_array = np.ones(max_year + 1, dtype=np.float64)
    for _, row in data['tx_interet_int'].iterrows():
        year = int(row['an_eval'])
        if year <= max_year:
            discount_int_array[year] = float(row['TX_ACTU_INT'])

    returns_ext_array = np.zeros((max_year + 1, max_scenarios + 1), dtype=np.float64)
    returns_int_array = np.zeros((max_year + 1, max_scenarios + 1), dtype=np.float64)

    for _, row in data['rendement'].iterrows():
        year = int(row['an_proj'])
        scenario = int(row['scn_proj'])
        if year <= max_year and scenario <= max_scenarios:
            if row['TYPE'] == 'EXTERNE':
                returns_ext_array[year, scenario] = float(row['RENDEMENT'])
            elif row['TYPE'] == 'INTERNE':
                returns_int_array[year, scenario] = float(row['RENDEMENT'])

    return {
        'mortality': mortality_array,
        'lapse': lapse_array,
        'discount_ext': discount_ext_array,
        'discount_int': discount_int_array,
        'returns_ext': returns_ext_array,
        'returns_int': returns_int_array
    }


def prepare_gpu_data(data, nb_accounts, nb_scenarios):
    """
    Improved version with explicit validation
    """
    from datetime import datetime

    total_combinations = min(nb_accounts, len(data['population'])) * nb_scenarios

    # Initialize arrays
    states = np.zeros((total_combinations, 9), dtype=np.float64)
    initial_data = np.zeros((min(nb_accounts, len(data['population'])), 11), dtype=np.float64)
    account_ids = np.zeros(min(nb_accounts, len(data['population'])), dtype=np.float64)

    # CRITICAL: Determine the maximum account ID first
    max_account_id = 0
    account_id_to_idx = {}

    for account_idx in range(min(nb_accounts, len(data['population']))):
        account_data = data['population'].iloc[account_idx]
        account_id = int(account_data['ID_COMPTE'])
        max_account_id = max(max_account_id, account_id)
        account_id_to_idx[account_id] = account_idx

    print(f"\nAccount ID range: 1 to {max_account_id}")
    print(f"Number of accounts: {len(account_id_to_idx)}")
    print(f"Account IDs: {sorted(account_id_to_idx.keys())}")

    # Create mapping array with size based on max account ID
    account_mapping = np.full(max_account_id + 1, -1, dtype=np.int32)

    # Populate data arrays
    combination_idx = 0
    for account_idx in range(min(nb_accounts, len(data['population']))):
        account_data = data['population'].iloc[account_idx]
        account_id = int(account_data['ID_COMPTE'])

        # Store account ID
        account_ids[account_idx] = float(account_id)

        # Update mapping
        account_mapping[account_id] = account_idx

        # Store initial data (using correct indices from constants)
        initial_data[account_idx, 0] = float(account_data['MT_VM'])  # DATA_MT_VM
        initial_data[account_idx, 1] = float(account_data['MT_GAR_DECES'])  # DATA_MT_GAR_DECES
        initial_data[account_idx, 2] = int(account_data['age_deb'])  # DATA_AGE_DEB
        initial_data[account_idx, 3] = float(account_data.get('TX_COMM_VENTE', 0.0))
        initial_data[account_idx, 4] = float(account_data['FRAIS_ACQUI'])
        initial_data[account_idx, 5] = float(account_data['PC_REVENU_FDS'])
        initial_data[account_idx, 6] = float(account_data['PC_HONORAIRES_GEST'])
        initial_data[account_idx, 7] = float(account_data['TX_COMM_MAINTIEN'])
        initial_data[account_idx, 8] = float(account_data['FRAIS_ADMIN'])
        initial_data[account_idx, 9] = float(account_data['FREQ_RESET_DECES'])
        initial_data[account_idx, 10] = float(account_data['MAX_RESET_DECES'])

        # Initialize states for all scenarios
        for scenario in range(1, nb_scenarios + 1):
            states[combination_idx, 0] = float(account_id)  # STATE_ACCOUNT_ID
            states[combination_idx, 1] = float(scenario)  # STATE_SCENARIO
            states[combination_idx, 2] = float(account_idx)  # STATE_ACCOUNT_IDX
            states[combination_idx, 3] = float(account_data['age_deb'])  # STATE_AGE_DEB
            states[combination_idx, 4] = 0.0  # STATE_MT_VM_PROJ
            states[combination_idx, 5] = 0.0  # STATE_MT_GAR_DECES_PROJ
            states[combination_idx, 6] = 0.0  # STATE_TX_SURVIE
            states[combination_idx, 7] = float(account_data['age_deb'])  # STATE_AGE
            states[combination_idx, 8] = 0.0  # STATE_IS_TERMINATED
            combination_idx += 1

    # Verify the mapping
    print("\nVerifying account mapping...")
    for account_id, expected_idx in account_id_to_idx.items():
        actual_idx = account_mapping[account_id]
        if actual_idx != expected_idx:
            print(f"ERROR: Account {account_id} should map to {expected_idx} but maps to {actual_idx}")
        else:
            print(f"  Account {account_id} -> Index {actual_idx} OK")

    return states, initial_data, account_ids, account_mapping


@cuda.jit
def gpu_calculate_year_transition(
    # --- Input/Output State ---
    states,                 # The dynamic state of each account-scenario pair
    results,                # The final detailed output array for all years

    # --- Static & Lookup Data ---
    initial_data,           # Static account data (initial VM, age_deb, fees, etc.)
    lookups_mortality,
    lookups_lapse,
    lookups_discount_ext,
    lookups_discount_int,
    lookups_returns_ext,
    lookups_returns_int,

    # --- Configuration ---
    year,                   # The current projection year being calculated (0, 1, 2, ...)
    projection_type,        # 0 for EXTERNE, 1 for INTERNE
    fund_shock,
    start_year,
    max_years_global        # The global maximum number of projection years (e.g., 100)
):
    """
    GPU kernel for calculating a single year's cash flows and state transition.

    This kernel is launched repeatedly for each year of the projection.

    THE FIX: This version includes an explicit age-based termination check
    that mirrors the CPU's `min(nb_years, 99 - age_deb)` logic. This ensures
    that projections for each account stop at the correct, individual time horizon.
    """
    # 1. SETUP: Determine which account-scenario this thread will process
    # ----------------------------------------------------------------------
    combination_idx = cuda.grid(1)
    if combination_idx >= states.shape[0]:
        return  # Exit if thread is out of bounds

    # If this policy has already been marked as terminated, do no more work.
    if states[combination_idx, STATE_IS_TERMINATED] > 0:
        return

    # Get the dense index for this account's static data
    account_idx = int(states[combination_idx, STATE_ACCOUNT_IDX])
    if account_idx >= initial_data.shape[0]:
        return # Safety: account index is out of bounds


    # 2. YEAR 0 LOGIC: Special initialization case
    # ----------------------------------------------------------------------
    if year == 0:
        # This block handles the unique cash flows at the start of a policy.
        # It is only executed once and its logic was already correct.
        if projection_type == 0:  # EXTERNE
            MT_VM_PROJ = initial_data[account_idx, DATA_MT_VM]
            MT_GAR_DECES_PROJ = initial_data[account_idx, DATA_MT_GAR_DECES]
            TX_SURVIE = 1.0
            AGE = initial_data[account_idx, DATA_AGE_DEB]

            # Initial, one-time fees
            COMMISSIONS = -initial_data[account_idx, DATA_TX_COMM_VENTE] * MT_VM_PROJ
            FRAIS_GEN = -initial_data[account_idx, DATA_FRAIS_ACQUI]
            FLUX_NET = FRAIS_GEN + COMMISSIONS
            VP_FLUX_NET = FLUX_NET
        else:  # INTERNE (Not used by the external projection, but kept for completeness)
            MT_VM_PROJ = initial_data[account_idx, DATA_MT_VM]
            if fund_shock > 0:
                MT_VM_PROJ *= (1 - fund_shock)

            MT_GAR_DECES_PROJ = initial_data[account_idx, DATA_MT_GAR_DECES]
            TX_SURVIE = 1.0
            AGE = initial_data[account_idx, DATA_AGE_DEB] + start_year
            FLUX_NET = 0.0
            VP_FLUX_NET = 0.0

        # Update state and results arrays for year 0
        states[combination_idx, STATE_MT_VM_PROJ] = MT_VM_PROJ
        states[combination_idx, STATE_MT_GAR_DECES_PROJ] = MT_GAR_DECES_PROJ
        states[combination_idx, STATE_TX_SURVIE] = TX_SURVIE
        states[combination_idx, STATE_AGE] = AGE
        # ... store results (omitted for brevity, same as your original)
        return # Important to exit after handling year 0


    # 3. YEAR > 0 LOGIC: Standard projection step
    # ----------------------------------------------------------------------

    # 🎯🎯🎯 THE FIX: PER-ACCOUNT HORIZON CHECK 🎯🎯🎯
    # Replicate the CPU's `min(nb_years, 99 - age_deb)` logic inside the kernel.
    age_deb = initial_data[account_idx, DATA_AGE_DEB]
    max_year_for_this_account = 99 - age_deb

    # If the current global `year` is beyond what is allowed for this specific
    # account, terminate it now and perform no further calculations.
    if year > max_year_for_this_account:
        states[combination_idx, STATE_IS_TERMINATED] = 1.0
        return

    # Standard termination checks (policy lapses before reaching age 99)
    current_survie = states[combination_idx, STATE_TX_SURVIE]
    current_vm = states[combination_idx, STATE_MT_VM_PROJ]
    if current_survie < 1e-15 or current_vm < 1e-15:
        states[combination_idx, STATE_IS_TERMINATED] = 1.0
        return

    # --- Calculations will only proceed if the policy is still valid for this year ---

    # Determine current age and projection year for lookups
    scenario = int(states[combination_idx, STATE_SCENARIO])
    new_age = int(age_deb + year)
    an_proj = year

    # Fund value evolution
    MT_VM_DEB = current_vm
    RENDEMENT_rate = lookups_returns_ext[an_proj, scenario]
    RENDEMENT = MT_VM_DEB * RENDEMENT_rate
    MT_VM_HALF_REND = MT_VM_DEB + RENDEMENT / 2.0
    FRAIS = -MT_VM_HALF_REND * initial_data[account_idx, DATA_PC_REVENU_FDS]
    new_MT_VM_PROJ = max(0.0, current_vm + RENDEMENT + FRAIS)

    # Death benefit guarantee reset
    new_MT_GAR_DECES_PROJ = states[combination_idx, STATE_MT_GAR_DECES_PROJ]
    if (initial_data[account_idx, DATA_FREQ_RESET_DECES] == 1 and
            new_age <= initial_data[account_idx, DATA_MAX_RESET_DECES]):
        new_MT_GAR_DECES_PROJ = max(new_MT_GAR_DECES_PROJ, new_MT_VM_PROJ)

    # Survival probability update
    QX = lookups_mortality[new_age]
    WX = lookups_lapse[an_proj]
    TX_SURVIE_DEB = current_survie
    new_TX_SURVIE = TX_SURVIE_DEB * (1.0 - QX) * (1.0 - WX)

    # Cash Flow Calculations
    REVENUS = -FRAIS * TX_SURVIE_DEB
    FRAIS_GEST = -MT_VM_HALF_REND * initial_data[account_idx, DATA_PC_HONORAIRES_GEST] * TX_SURVIE_DEB
    COMMISSIONS = -MT_VM_HALF_REND * initial_data[account_idx, DATA_TX_COMM_MAINTIEN] * TX_SURVIE_DEB
    FRAIS_GEN = -initial_data[account_idx, DATA_FRAIS_ADMIN] * TX_SURVIE_DEB
    PMT_GARANTIE = -max(0.0, new_MT_GAR_DECES_PROJ - new_MT_VM_PROJ) * QX * TX_SURVIE_DEB
    FLUX_NET = REVENUS + FRAIS_GEST + COMMISSIONS + FRAIS_GEN + PMT_GARANTIE

    # Discounting
    TX_ACTU = lookups_discount_ext[an_proj]
    VP_FLUX_NET = FLUX_NET * TX_ACTU

    # 4. UPDATE STATE: Store the new values for the next year's calculation
    # ----------------------------------------------------------------------
    states[combination_idx, STATE_MT_VM_PROJ] = new_MT_VM_PROJ
    states[combination_idx, STATE_MT_GAR_DECES_PROJ] = new_MT_GAR_DECES_PROJ
    states[combination_idx, STATE_TX_SURVIE] = new_TX_SURVIE
    states[combination_idx, STATE_AGE] = new_age

    # Mark as terminated if conditions are met, for the next iteration
    if new_TX_SURVIE < 1e-15 or new_MT_VM_PROJ < 1e-15:
        states[combination_idx, STATE_IS_TERMINATED] = 1.0

    # 5. STORE RESULTS: Write the calculated values to the output array
    # ----------------------------------------------------------------------
    result_idx = combination_idx * (max_years_global + 1) + year
    if result_idx < results.shape[0]:
        # Writing all columns for completeness
        results[result_idx, 0] = states[combination_idx, STATE_ACCOUNT_ID]
        results[result_idx, 1] = states[combination_idx, STATE_SCENARIO]
        results[result_idx, 2] = year
        results[result_idx, 3] = new_age
        results[result_idx, 4] = new_MT_VM_PROJ
        results[result_idx, 5] = new_MT_GAR_DECES_PROJ
        results[result_idx, 6] = new_TX_SURVIE
        results[result_idx, 7] = FLUX_NET
        results[result_idx, 8] = VP_FLUX_NET

def run_gpu_projection(states, initial_data, lookups, nb_years: int, projection_type: str,
                       fund_shock: float = 0.0, start_year: int = 0) -> np.ndarray:
    """Run projection on GPU"""
    proj_type_num = 0 if projection_type == "EXTERNE" else 1

    max_results = states.shape[0] * (nb_years + 1)
    results = np.zeros((max_results, 9), dtype=np.float64)

    print(f"DEBUG: States shape: {states.shape}, Max results: {max_results}")

    d_states = cuda.to_device(states)
    d_initial_data = cuda.to_device(initial_data)
    d_results = cuda.to_device(results)

    d_mortality = cuda.to_device(lookups['mortality'])
    d_lapse = cuda.to_device(lookups['lapse'])
    d_discount_ext = cuda.to_device(lookups['discount_ext'])
    d_discount_int = cuda.to_device(lookups['discount_int'])
    d_returns_ext = cuda.to_device(lookups['returns_ext'])
    d_returns_int = cuda.to_device(lookups['returns_int'])

    threads_per_block = 256
    blocks_per_grid = (states.shape[0] + threads_per_block - 1) // threads_per_block

    print(f"DEBUG: GPU grid: {blocks_per_grid} blocks, {threads_per_block} threads per block")

    for year in range(nb_years + 1):
        print(f"DEBUG: Processing year {year}")
        gpu_calculate_year_transition[blocks_per_grid, threads_per_block](
            d_states, d_initial_data, d_mortality, d_lapse, d_discount_ext, d_discount_int,
            d_returns_ext, d_returns_int, d_results, year, proj_type_num,
            fund_shock, start_year, nb_years
        )
        cuda.synchronize()

        if year == 0:
            temp_results = d_results.copy_to_host()
            year_0_results = temp_results[temp_results[:, 2] == 0]
            year_0_accounts = np.unique(year_0_results[:, 0][year_0_results[:, 0] != 0])
            print(f"DEBUG: Year 0 - Accounts with results: {year_0_accounts}")

    results = d_results.copy_to_host()
    states = d_states.copy_to_host()

    return results, states


@cuda.jit
def gpu_calculate_internal_scenarios(
        # --- Input Arrays ---
        external_results,  # Results from the external projection (state at each year)
        initial_data,  # Static account data (initial VM, age_deb, fees, etc.)
        account_mapping,  # Maps sparse account_id to dense initial_data index

        # --- Lookup Tables ---
        lookups_mortality,
        lookups_lapse,
        lookups_discount_ext,
        lookups_discount_int,
        lookups_returns_int,  # Using internal returns for this projection

        # --- Output Array ---
        internal_results,  # The final calculated value (reserve or capital) for each external_result row

        # --- Configuration Parameters ---
        nb_sc_int,  # Number of internal scenarios to run (e.g., 2)
        nb_an_projection_int,  # Maximum possible projection years (e.g., 100)
        fund_shock  # Capital shock factor (0.0 for reserves, 0.35 for capital)
):
    """
    GPU kernel to calculate the average present value of future cash flows for
    a given state (account, scenario, year) from an external projection.

    This kernel is launched for each row in the `external_results` array. Each thread
    will run a full Monte Carlo simulation (nb_sc_int scenarios) from the
    starting point defined by that row.

    The key fix to match the CPU logic is the dynamic calculation of the
    projection horizon based on the account's age.
    """

    # 1. SETUP: Determine which external state this thread will process
    # ------------------------------------------------------------------
    external_idx = cuda.grid(1)
    if external_idx >= external_results.shape[0]:
        return  # Exit if this thread is outside the bounds of the input data

    # 2. DATA UNPACKING & INITIAL CHECKS: Get the starting state for our projection
    # --------------------------------------------------------------------------------
    year = int(external_results[external_idx, 2])  # Column 2: an_proj (start year)

    # Internal calculations are only performed for future years (t > 0)
    if year == 0:
        internal_results[external_idx] = 0.0
        return

    # Unpack the state variables from the external projection result
    account_id = int(external_results[external_idx, 0])
    fund_value = external_results[external_idx, 4]  # MT_VM_PROJ at this year
    death_benefit = external_results[external_idx, 5]  # MT_GAR_DECES_PROJ at this year
    survival_prob = external_results[external_idx, 6]  # TX_SURVIE at this year

    # If the policy has already terminated, there are no future cash flows
    if survival_prob < 1e-15 or fund_value < 1e-15:
        internal_results[external_idx] = 0.0
        return

    # 3. ACCOUNT MAPPING: Find the account's static data
    # ----------------------------------------------------------------
    # The `account_id` can be large and sparse (e.g., 1, 5, 1002).
    # We use the mapping array to find its index in our dense `initial_data` array.
    if account_id < 0 or account_id >= account_mapping.shape[0]:
        internal_results[external_idx] = 0.0
        return  # Safety check: account_id is out of bounds for the mapping array

    account_idx = account_mapping[account_id]

    # Further safety checks on the mapping result
    if account_idx == -1 or account_idx >= initial_data.shape[0]:
        internal_results[external_idx] = 0.0
        return

    # 4. ** THE FIX **: DYNAMICALLY CALCULATE PROJECTION HORIZON
    # ----------------------------------------------------------------
    # This is the critical logic to ensure GPU results match the CPU.
    # The projection must stop when the person reaches age 99.
    AGE_BASE = int(initial_data[account_idx, 2])  # Index 2 is DATA_AGE_DEB

    # Calculate how many years are left until the person reaches age 99
    max_years_for_age = 99 - AGE_BASE - year

    # The actual projection horizon is the minimum of the global setting
    # and the account-specific age limit.
    actual_max_years = min(nb_an_projection_int, max_years_for_age)

    # If the person is already too old, there's no projection to run
    if actual_max_years < 0:
        internal_results[external_idx] = 0.0
        return

    # 5. MONTE CARLO SIMULATION: Run internal scenarios
    # ----------------------------------------------------------------
    # Use Kahan summation to improve numerical stability for floating-point sums.
    sum_vp = 0.0  # Holds the sum of PVs across all scenarios
    sum_compensation = 0.0  # Kahan compensation term for the total sum

    for internal_scenario in range(1, nb_sc_int + 1):
        # --- A. Initialize state for this scenario ---
        MT_VM_PROJ = fund_value
        MT_GAR_DECES_PROJ = death_benefit
        TX_SURVIE = survival_prob

        # Apply capital shock if applicable (fund_shock > 0)
        if fund_shock > 0.0:
            MT_VM_PROJ = MT_VM_PROJ * (1.0 - fund_shock)

        # Kahan variables for summing the PVs within this single scenario
        scenario_sum = 0.0
        scenario_compensation = 0.0

        # --- B. Project cash flows for this scenario year by year ---
        # ** NOTE: We use `actual_max_years` calculated in step 4 **
        for internal_year in range(1, actual_max_years + 1):

            # Terminate this scenario early if policy lapses or fund is depleted
            if TX_SURVIE < 1e-15 or MT_VM_PROJ < 1e-15:
                break

            # Calculate current age and projection year for lookups
            an_proj = year + internal_year
            current_age = AGE_BASE + an_proj

            # Bounds checks for lookup tables
            if current_age >= lookups_mortality.shape[0] or \
                    an_proj >= lookups_returns_int.shape[0]:
                break

            # --- C. Core Financial Calculations (for one year) ---
            # Get investment return rate for this year and scenario
            RENDEMENT_rate = lookups_returns_int[an_proj, internal_scenario]

            # Fund value evolution
            MT_VM_DEB = MT_VM_PROJ
            RENDEMENT = MT_VM_DEB * RENDEMENT_rate
            MT_VM_HALF_REND = MT_VM_DEB + RENDEMENT / 2.0  # Pre-calculate for fee base
            FRAIS = -MT_VM_HALF_REND * initial_data[account_idx, 5]  # Index 5: PC_REVENU_FDS
            MT_VM_PROJ = max(0.0, MT_VM_DEB + RENDEMENT + FRAIS)

            # Death benefit guarantee reset logic
            if initial_data[account_idx, 9] == 1 and current_age <= initial_data[account_idx, 10]:
                MT_GAR_DECES_PROJ = max(MT_GAR_DECES_PROJ, MT_VM_PROJ)

            # Survival probability update
            QX = lookups_mortality[current_age]
            WX = lookups_lapse[an_proj]
            TX_SURVIE_DEB = TX_SURVIE
            TX_SURVIE = TX_SURVIE_DEB * (1.0 - QX) * (1.0 - WX)

            # Cash Flow Calculations
            REVENUS = -FRAIS * TX_SURVIE_DEB
            FRAIS_GEST = -MT_VM_HALF_REND * initial_data[account_idx, 6] * TX_SURVIE_DEB  # Idx 6: PC_HONORAIRES_GEST
            COMMISSIONS = -MT_VM_HALF_REND * initial_data[account_idx, 7] * TX_SURVIE_DEB  # Idx 7: TX_COMM_MAINTIEN
            FRAIS_GEN = -initial_data[account_idx, 8] * TX_SURVIE_DEB  # Idx 8: FRAIS_ADMIN
            PMT_GARANTIE = -max(0.0, MT_GAR_DECES_PROJ - MT_VM_PROJ) * QX * TX_SURVIE_DEB
            FLUX_NET = REVENUS + FRAIS_GEST + COMMISSIONS + FRAIS_GEN + PMT_GARANTIE

            # Discounting to find Present Value (PV)
            TX_ACTU_EXT = lookups_discount_ext[an_proj]
            VP_FLUX_NET = FLUX_NET * TX_ACTU_EXT

            # Additional internal discounting based on the starting year of this projection
            TX_ACTU_INT = lookups_discount_int[year]
            if TX_ACTU_INT > 1e-15:
                VP_FLUX_NET /= TX_ACTU_INT

            # --- D. Accumulate PV for this scenario using Kahan sum ---
            y = VP_FLUX_NET - scenario_compensation
            t = scenario_sum + y
            scenario_compensation = (t - scenario_sum) - y
            scenario_sum = t

        # --- E. Add this scenario's total PV to the grand total (Kahan sum) ---
        y = scenario_sum - sum_compensation
        t = sum_vp + y
        sum_compensation = (t - sum_vp) - y
        sum_vp = t

    # 6. FINALIZATION: Calculate the mean and write the result
    # -----------------------------------------------------------
    # The final result is the average PV across all internal scenarios.
    internal_results[external_idx] = sum_vp / float(nb_sc_int)

def gpu_acfc_algorithm_complete(data_path: str = ".", nb_accounts: int = 4, nb_scenarios: int = 10,
                                nb_years: int = 10, nb_sc_int: int = 10, nb_an_projection_int: int = 10,
                                choc_capital: float = 0.35, hurdle_rt: float = 0.10) -> pd.DataFrame:
    """
    Complete GPU-Accelerated ACFC Algorithm - FIXED VERSION
    """

    print("Phase 1: Loading input data...")
    data = load_input_data(data_path, nb_accounts)

    print(f"DEBUG: Loaded {len(data['population'])} accounts:")
    for i, row in data['population'].iterrows():
        print(f"  Account {i}: ID_COMPTE = {row['ID_COMPTE']}")

    print("Phase 2: Creating GPU lookup tables...")
    lookups = create_gpu_lookup_tables(data)

    print("Phase 3: Preparing GPU data...")
    states, initial_data, account_ids, account_mapping = prepare_gpu_data(data, nb_accounts, nb_scenarios)

    print(f"DEBUG: Prepared {len(states)} state combinations")
    print(f"DEBUG: Account IDs in prepared data: {account_ids}")
    print(f"DEBUG: Unique account IDs in states: {np.unique(states[:, STATE_ACCOUNT_ID])}")
    print(f"DEBUG: Account mapping array size: {len(account_mapping)}")

    print("Phase 4: Running GPU external projections...")
    external_results, final_states = run_gpu_projection(
        states, initial_data, lookups, nb_years, 'EXTERNE'
    )

    # FIX 3: Better debugging of external results
    print("\n=== DEBUGGING EXTERNAL RESULTS ===")
    print(f"Total external results before filtering: {len(external_results)}")
    print(f"Non-zero account IDs in results: {np.unique(external_results[:, 0][external_results[:, 0] != 0])}")

    valid_mask = external_results[:, 0] != 0
    valid_external_results = external_results[valid_mask]

    print(f"Valid external results after filtering: {len(valid_external_results)}")

    for account_id in sorted(np.unique(valid_external_results[:, 0])):
        account_mask = valid_external_results[:, 0] == account_id
        account_results = valid_external_results[account_mask]
        scenarios = sorted(set(account_results[:, 1]))
        print(f"\nAccount {int(account_id)}:")
        print(f"  Total results: {len(account_results)}")
        print(f"  Scenarios found: {scenarios}")
        if len(scenarios) > 0:
            print(f"  Years per scenario: {len(account_results) // len(scenarios)}")

        year_0 = account_results[account_results[:, 2] == 0]
        if len(year_0) > 0:
            print(f"  Year 0 MT_VM_PROJ range: {year_0[:, 4].min():.2f} to {year_0[:, 4].max():.2f}")

    if len(valid_external_results) == 0:
        print("WARNING: No valid external results found!")
        return pd.DataFrame()

    print("Phase 6: Running GPU internal calculations for reserves and capital...")

    reserve_results = np.zeros(len(valid_external_results), dtype=np.float64)
    capital_results = np.zeros(len(valid_external_results), dtype=np.float64)

    d_external_results = cuda.to_device(valid_external_results)
    d_initial_data = cuda.to_device(initial_data)
    d_reserve_results = cuda.to_device(reserve_results)
    d_capital_results = cuda.to_device(capital_results)
    d_account_mapping = cuda.to_device(account_mapping)

    d_mortality = cuda.to_device(lookups['mortality'])
    d_lapse = cuda.to_device(lookups['lapse'])
    d_discount_ext = cuda.to_device(lookups['discount_ext'])
    d_discount_int = cuda.to_device(lookups['discount_int'])
    d_returns_ext = cuda.to_device(lookups['returns_ext'])
    d_returns_int = cuda.to_device(lookups['returns_int'])

    threads_per_block = 256
    blocks_per_grid = (len(valid_external_results) + threads_per_block - 1) // threads_per_block

    # Call for reserves
    print("  -> Calculating reserves...")
    gpu_calculate_internal_scenarios[blocks_per_grid, threads_per_block](
        d_external_results,
        d_initial_data,
        d_account_mapping,  # MOVED to 3rd position
        d_mortality,
        d_lapse,
        d_discount_ext,
        d_discount_int,
        d_returns_int,  # d_returns_ext is REMOVED
        d_reserve_results,
        nb_sc_int,
        nb_an_projection_int,
        0.0  # fund_shock for reserves
    )
    cuda.synchronize()

    # Call for capital
    print("  -> Calculating capital...")
    gpu_calculate_internal_scenarios[blocks_per_grid, threads_per_block](
        d_external_results,
        d_initial_data,
        d_account_mapping,  # MOVED to 3rd position
        d_mortality,
        d_lapse,
        d_discount_ext,
        d_discount_int,
        d_returns_int,  # d_returns_ext is REMOVED
        d_capital_results,
        nb_sc_int,
        nb_an_projection_int,
        choc_capital  # fund_shock for capital
    )
    cuda.synchronize()

    reserve_results = d_reserve_results.copy_to_host()
    capital_results = d_capital_results.copy_to_host()

    print("Phase 7: Calculating distributable flows...")

    final_results = []

    from collections import defaultdict
    grouped_external = defaultdict(list)
    grouped_reserves = defaultdict(list)
    grouped_capital = defaultdict(list)

    print(f"DEBUG: Processing {len(valid_external_results)} external results")

    for i, row in enumerate(valid_external_results):
        account_id = int(row[0])
        scenario = int(row[1])
        year = int(row[2])
        key = f"{account_id}_{scenario}"

        grouped_external[key].append({
            'year': year,
            'FLUX_NET': row[7],
            'VP_FLUX_NET': row[8]
        })
        grouped_reserves[key].append((year, reserve_results[i]))
        grouped_capital[key].append((year, capital_results[i] - reserve_results[i]))

    print(f"DEBUG: Unique account-scenario combinations: {len(grouped_external)}")

    for key in grouped_external:
        account_id, scenario = key.split('_')
        account_id = int(account_id)
        scenario = int(scenario)

        external_data = sorted(grouped_external[key], key=lambda x: x['year'])
        reserve_data = dict(sorted(grouped_reserves[key], key=lambda x: x[0]))
        capital_data = dict(sorted(grouped_capital[key], key=lambda x: x[0]))

        distributable_pvs = []
        prev_reserve = 0.0
        prev_capital = 0.0

        for ext_data in external_data:
            year = ext_data['year']
            external_cf = ext_data['FLUX_NET']

            current_reserve = reserve_data.get(year, 0.0)
            current_capital = capital_data.get(year, 0.0)

            if year == 0:
                profit = external_cf + current_reserve
                distributable = profit + current_capital
            else:
                profit = external_cf + (current_reserve - prev_reserve)
                distributable = profit + (current_capital - prev_capital)

            if year > 0:
                pv_distributable = distributable / ((1 + hurdle_rt) ** year)
            else:
                pv_distributable = distributable

            distributable_pvs.append(pv_distributable)

            prev_reserve = current_reserve
            prev_capital = current_capital

        total_pv_distributable = sum(distributable_pvs)

        final_results.append({
            'ID_COMPTE': account_id,
            'scn_eval': scenario,
            'VP_FLUX_DISTRIBUABLES': total_pv_distributable
        })

    print("Phase 8: Converting to DataFrame...")
    output_df = pd.DataFrame(final_results)

    print(f"Complete GPU algorithm finished. Generated {len(output_df)} results.")
    return output_df


if __name__ == "__main__":
    if not cuda.is_available():
        print("CUDA is not available. Please install CUDA and ensure your GPU supports it.")
        exit(1)

    print(f"CUDA devices available: {cuda.gpus}")

    data_path = "data_in"

    results = gpu_acfc_algorithm_complete(
        data_path=data_path,
        nb_accounts=2,
        nb_scenarios=2,
        nb_years=100,
        nb_sc_int=2,
        nb_an_projection_int=100,
        choc_capital=0.35,
        hurdle_rt=0.10
    )

    print("\nFinal Results:")
    print(results)

    results.to_csv('test/gpu_results_complete.csv', index=False)
    print(f"\nMean VP_FLUX_DISTRIBUABLES: {results['VP_FLUX_DISTRIBUABLES'].mean():.2f}")
    print(f"Range: {results['VP_FLUX_DISTRIBUABLES'].min():.2f} to {results['VP_FLUX_DISTRIBUABLES'].max():.2f}")
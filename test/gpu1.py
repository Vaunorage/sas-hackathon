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
def gpu_calculate_year_transition(states, initial_data, lookups_mortality, lookups_lapse,
                                  lookups_discount_ext, lookups_discount_int, lookups_returns_ext,
                                  lookups_returns_int, results, year, projection_type, fund_shock, start_year,
                                  max_years):
    """GPU kernel for year transition calculations - FIXED version"""

    combination_idx = cuda.grid(1)
    if combination_idx >= states.shape[0]:
        return

    if states[combination_idx, STATE_IS_TERMINATED] > 0:
        return

    account_idx = int(states[combination_idx, STATE_ACCOUNT_IDX])

    # FIXED: Add bounds checking for account_idx
    if account_idx >= initial_data.shape[0] or account_idx < 0:
        return

    scenario = int(states[combination_idx, STATE_SCENARIO])

    # Handle year 0 special cases
    if year == 0:
        if projection_type == 0:  # EXTERNE
            MT_VM_PROJ = initial_data[account_idx, DATA_MT_VM]
            MT_GAR_DECES_PROJ = initial_data[account_idx, DATA_MT_GAR_DECES]
            TX_SURVIE = 1.0
            AGE = initial_data[account_idx, DATA_AGE_DEB]

            COMMISSIONS = -initial_data[account_idx, DATA_TX_COMM_VENTE] * MT_VM_PROJ
            FRAIS_GEN = -initial_data[account_idx, DATA_FRAIS_ACQUI]
            FLUX_NET = FRAIS_GEN + COMMISSIONS
            VP_FLUX_NET = FLUX_NET
        else:  # INTERNE
            if fund_shock > 0:
                MT_VM_PROJ = initial_data[account_idx, DATA_MT_VM] * (1 - fund_shock)
            else:
                MT_VM_PROJ = initial_data[account_idx, DATA_MT_VM]

            MT_GAR_DECES_PROJ = initial_data[account_idx, DATA_MT_GAR_DECES]
            TX_SURVIE = 1.0
            AGE = initial_data[account_idx, DATA_AGE_DEB] + start_year

            FLUX_NET = 0.0
            VP_FLUX_NET = 0.0

        states[combination_idx, STATE_MT_VM_PROJ] = MT_VM_PROJ
        states[combination_idx, STATE_MT_GAR_DECES_PROJ] = MT_GAR_DECES_PROJ
        states[combination_idx, STATE_TX_SURVIE] = TX_SURVIE
        states[combination_idx, STATE_AGE] = AGE

        nb_years_total = max_years + 1
        result_idx = combination_idx * nb_years_total + year
        if result_idx < results.shape[0]:
            results[result_idx, 0] = states[combination_idx, STATE_ACCOUNT_ID]
            results[result_idx, 1] = states[combination_idx, STATE_SCENARIO]
            results[result_idx, 2] = year
            results[result_idx, 3] = AGE
            results[result_idx, 4] = MT_VM_PROJ
            results[result_idx, 5] = MT_GAR_DECES_PROJ
            results[result_idx, 6] = TX_SURVIE
            results[result_idx, 7] = FLUX_NET
            results[result_idx, 8] = VP_FLUX_NET
        return

    current_survie = states[combination_idx, STATE_TX_SURVIE]
    current_vm = states[combination_idx, STATE_MT_VM_PROJ]

    if current_survie <= 0.0 or current_vm <= 0.0:
        states[combination_idx, STATE_IS_TERMINATED] = 1.0
        nb_years_total = max_years + 1
        result_idx = combination_idx * nb_years_total + year
        if result_idx < results.shape[0]:
            results[result_idx, 0] = states[combination_idx, STATE_ACCOUNT_ID]
            results[result_idx, 1] = states[combination_idx, STATE_SCENARIO]
            results[result_idx, 2] = year
            results[result_idx, 3] = states[combination_idx, STATE_AGE]
            results[result_idx, 4] = 0.0
            results[result_idx, 5] = 0.0
            results[result_idx, 6] = 0.0
            results[result_idx, 7] = 0.0
            results[result_idx, 8] = 0.0
        return

    if projection_type == 1:  # INTERNE
        new_age = int(initial_data[account_idx, DATA_AGE_DEB] + start_year + year)
        an_proj = start_year + year
    else:  # EXTERNE
        new_age = int(initial_data[account_idx, DATA_AGE_DEB] + year)
        an_proj = year

    if (new_age >= lookups_mortality.shape[0] or new_age < 0 or
            an_proj >= lookups_returns_ext.shape[0] or an_proj < 0):
        states[combination_idx, STATE_IS_TERMINATED] = 1.0
        return

    MT_VM_DEB = states[combination_idx, STATE_MT_VM_PROJ]

    RENDEMENT_rate = 0.0
    if projection_type == 0:  # EXTERNE
        if (scenario >= 0 and scenario < lookups_returns_ext.shape[1] and
                an_proj >= 0 and an_proj < lookups_returns_ext.shape[0]):
            RENDEMENT_rate = lookups_returns_ext[an_proj, scenario]
    else:  # INTERNE
        if (scenario >= 0 and scenario < lookups_returns_int.shape[1] and
                an_proj >= 0 and an_proj < lookups_returns_int.shape[0]):
            RENDEMENT_rate = lookups_returns_int[an_proj, scenario]

    RENDEMENT = MT_VM_DEB * RENDEMENT_rate
    FRAIS = -(MT_VM_DEB + RENDEMENT / 2) * initial_data[account_idx, DATA_PC_REVENU_FDS]
    new_MT_VM_PROJ = max(0.0, states[combination_idx, STATE_MT_VM_PROJ] + RENDEMENT + FRAIS)

    new_MT_GAR_DECES_PROJ = states[combination_idx, STATE_MT_GAR_DECES_PROJ]
    if (initial_data[account_idx, DATA_FREQ_RESET_DECES] == 1 and
            new_age <= initial_data[account_idx, DATA_MAX_RESET_DECES]):
        new_MT_GAR_DECES_PROJ = max(states[combination_idx, STATE_MT_GAR_DECES_PROJ], new_MT_VM_PROJ)

    QX = 0.0
    WX = 0.0
    if new_age >= 0 and new_age < lookups_mortality.shape[0]:
        QX = lookups_mortality[new_age]
    if an_proj >= 0 and an_proj < lookups_lapse.shape[0]:
        WX = lookups_lapse[an_proj]

    TX_SURVIE_DEB = states[combination_idx, STATE_TX_SURVIE]
    new_TX_SURVIE = TX_SURVIE_DEB * (1 - QX) * (1 - WX)

    REVENUS = -FRAIS * TX_SURVIE_DEB
    FRAIS_GEST = -(MT_VM_DEB + RENDEMENT / 2) * initial_data[account_idx, DATA_PC_HONORAIRES_GEST] * TX_SURVIE_DEB
    COMMISSIONS = -(MT_VM_DEB + RENDEMENT / 2) * initial_data[account_idx, DATA_TX_COMM_MAINTIEN] * TX_SURVIE_DEB
    FRAIS_GEN = -initial_data[account_idx, DATA_FRAIS_ADMIN] * TX_SURVIE_DEB
    PMT_GARANTIE = -max(0.0, new_MT_GAR_DECES_PROJ - new_MT_VM_PROJ) * QX * TX_SURVIE_DEB

    FLUX_NET = REVENUS + FRAIS_GEST + COMMISSIONS + FRAIS_GEN + PMT_GARANTIE

    TX_ACTU = 1.0
    if an_proj >= 0 and an_proj < lookups_discount_ext.shape[0]:
        TX_ACTU = lookups_discount_ext[an_proj]
    VP_FLUX_NET = FLUX_NET * TX_ACTU

    if projection_type == 1 and start_year > 0:  # INTERNE
        TX_ACTU_INT = 1.0
        if start_year >= 0 and start_year < lookups_discount_int.shape[0]:
            TX_ACTU_INT = lookups_discount_int[start_year]
        if TX_ACTU_INT != 0:
            VP_FLUX_NET = VP_FLUX_NET / TX_ACTU_INT

    states[combination_idx, STATE_MT_VM_PROJ] = new_MT_VM_PROJ
    states[combination_idx, STATE_MT_GAR_DECES_PROJ] = new_MT_GAR_DECES_PROJ
    states[combination_idx, STATE_TX_SURVIE] = new_TX_SURVIE
    states[combination_idx, STATE_AGE] = new_age

    if new_TX_SURVIE <= 0.0 or new_MT_VM_PROJ <= 0.0:
        states[combination_idx, STATE_IS_TERMINATED] = 1.0

    nb_years_total = max_years + 1
    result_idx = combination_idx * nb_years_total + year
    if result_idx < results.shape[0]:
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
def gpu_calculate_internal_scenarios(external_results, initial_data, lookups_mortality,
                                           lookups_lapse, lookups_discount_ext, lookups_discount_int,
                                           lookups_returns_ext, lookups_returns_int, internal_results,
                                           nb_sc_int, nb_an_projection_int, fund_shock, account_mapping):
    """
    Final corrected version matching CPU exactly.

    Key fixes for age-dependent drift:
    1. Match CPU's order of operations EXACTLY
    2. Use explicit max logic instead of max() function
    3. Kahan summation to handle long accumulation periods
    4. Careful handling of repeated discount operations
    """

    external_idx = cuda.grid(1)
    if external_idx >= external_results.shape[0]:
        return

    year = int(external_results[external_idx, 2])
    if year == 0 or external_results[external_idx, 0] == 0:
        return

    account_id = int(external_results[external_idx, 0])
    fund_value = external_results[external_idx, 4]
    death_benefit = external_results[external_idx, 5]
    survival_prob = external_results[external_idx, 6]

    if survival_prob == 0.0 or fund_value == 0.0:
        internal_results[external_idx] = 0.0
        return

    # Validate account mapping
    if account_id >= account_mapping.shape[0] or account_id < 0:
        internal_results[external_idx] = 0.0
        return

    account_idx = account_mapping[account_id]
    if account_idx == -1 or account_idx >= initial_data.shape[0] or account_idx < 0:
        internal_results[external_idx] = 0.0
        return

    # Kahan summation variables
    sum_vp = 0.0
    sum_compensation = 0.0

    for internal_scenario in range(1, nb_sc_int + 1):
        # Initialize state - MATCH CPU EXACTLY
        MT_VM_PROJ = fund_value
        MT_GAR_DECES_PROJ = death_benefit
        TX_SURVIE = survival_prob

        # Apply shock AFTER initialization
        if fund_shock > 0.0:
            MT_VM_PROJ = MT_VM_PROJ * (1.0 - fund_shock)

        AGE_BASE = int(initial_data[account_idx, 2])

        # Scenario accumulation with Kahan
        scenario_sum = 0.0
        scenario_compensation = 0.0

        # CRITICAL: Start from year 1, not 0 (CPU skips year 0)
        for internal_year in range(1, nb_an_projection_int + 1):
            # Early termination
            if TX_SURVIE < 1e-15 or MT_VM_PROJ < 1e-15:
                break

            an_proj = year + internal_year
            current_age = AGE_BASE + year + internal_year

            # Bounds check
            if current_age >= lookups_mortality.shape[0]:
                break
            if an_proj >= lookups_returns_int.shape[0]:
                break

            # Get return rate
            RENDEMENT_rate = 0.0
            if internal_scenario < lookups_returns_int.shape[1] and an_proj < lookups_returns_int.shape[0]:
                RENDEMENT_rate = lookups_returns_int[an_proj, internal_scenario]

            # Fund evolution - EXACT CPU sequence
            MT_VM_DEB = MT_VM_PROJ
            RENDEMENT = MT_VM_DEB * RENDEMENT_rate

            # CRITICAL: Calculate average in one step to match CPU
            MT_VM_HALF_REND = MT_VM_DEB + RENDEMENT / 2.0
            FRAIS = -MT_VM_HALF_REND * initial_data[account_idx, 5]

            # Update fund value
            MT_VM_PROJ = MT_VM_DEB + RENDEMENT + FRAIS

            # Ensure non-negative (match CPU's max(MT_VM_PROJ, 0))
            if MT_VM_PROJ < 0.0:
                MT_VM_PROJ = 0.0

            # Death benefit reset - CRITICAL FIX
            # Use explicit comparison to avoid max() function issues
            FREQ_RESET = initial_data[account_idx, 9]
            MAX_RESET = initial_data[account_idx, 10]

            # Match CPU logic EXACTLY
            if FREQ_RESET > 0.5:  # More robust than == 1.0
                if current_age <= MAX_RESET:
                    # Explicit comparison instead of max()
                    if MT_VM_PROJ > MT_GAR_DECES_PROJ:
                        MT_GAR_DECES_PROJ = MT_VM_PROJ

            # Get rates
            QX = 0.0
            WX = 0.0
            if current_age < lookups_mortality.shape[0]:
                QX = lookups_mortality[current_age]
            if an_proj < lookups_lapse.shape[0]:
                WX = lookups_lapse[an_proj]

            # Update survival
            TX_SURVIE_DEB = TX_SURVIE
            TX_SURVIE = TX_SURVIE_DEB * (1.0 - QX) * (1.0 - WX)

            # Cash flows - use pre-calculated average
            REVENUS = -FRAIS * TX_SURVIE_DEB
            FRAIS_GEST = -MT_VM_HALF_REND * initial_data[account_idx, 6] * TX_SURVIE_DEB
            COMMISSIONS = -MT_VM_HALF_REND * initial_data[account_idx, 7] * TX_SURVIE_DEB
            FRAIS_GEN = -initial_data[account_idx, 8] * TX_SURVIE_DEB

            # Death benefit payment - explicit max
            shortfall = MT_GAR_DECES_PROJ - MT_VM_PROJ
            if shortfall > 0.0:
                PMT_GARANTIE = -shortfall * QX * TX_SURVIE_DEB
            else:
                PMT_GARANTIE = 0.0

            FLUX_NET = REVENUS + FRAIS_GEST + COMMISSIONS + FRAIS_GEN + PMT_GARANTIE

            # External discount
            TX_ACTU = 1.0
            if an_proj < lookups_discount_ext.shape[0]:
                TX_ACTU = lookups_discount_ext[an_proj]

            VP_FLUX_NET = FLUX_NET * TX_ACTU

            # Internal discount - CRITICAL: Match CPU exactly
            if year > 0:
                if year < lookups_discount_int.shape[0]:
                    TX_ACTU_INT = lookups_discount_int[year]
                    if TX_ACTU_INT > 1e-15:
                        VP_FLUX_NET = VP_FLUX_NET / TX_ACTU_INT

            # Kahan summation for scenario
            y = VP_FLUX_NET - scenario_compensation
            t = scenario_sum + y
            scenario_compensation = (t - scenario_sum) - y
            scenario_sum = t

        # Kahan summation for total
        y = scenario_sum - sum_compensation
        t = sum_vp + y
        sum_compensation = (t - sum_vp) - y
        sum_vp = t

    # Calculate mean
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

    gpu_calculate_internal_scenarios[blocks_per_grid, threads_per_block](
        d_external_results, d_initial_data, d_mortality, d_lapse, d_discount_ext, d_discount_int,
        d_returns_ext, d_returns_int, d_reserve_results, nb_sc_int, nb_an_projection_int,
        0.0, d_account_mapping
    )
    cuda.synchronize()

    gpu_calculate_internal_scenarios[blocks_per_grid, threads_per_block](
        d_external_results, d_initial_data, d_mortality, d_lapse, d_discount_ext, d_discount_int,
        d_returns_ext, d_returns_int, d_capital_results, nb_sc_int, nb_an_projection_int,
        choc_capital, d_account_mapping
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
        nb_accounts=10,
        nb_scenarios=10,
        nb_years=100,
        nb_sc_int=10,
        nb_an_projection_int=100,
        choc_capital=0.35,
        hurdle_rt=0.10
    )

    print("\nFinal Results:")
    print(results)

    results.to_csv('test/gpu_results_complete.csv', index=False)
    print(f"\nMean VP_FLUX_DISTRIBUABLES: {results['VP_FLUX_DISTRIBUABLES'].mean():.2f}")
    print(f"Range: {results['VP_FLUX_DISTRIBUABLES'].min():.2f} to {results['VP_FLUX_DISTRIBUABLES'].max():.2f}")
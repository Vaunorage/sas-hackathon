import pandas as pd
import numpy as np
from numba import cuda, jit
import numba
from typing import Dict, Tuple, List
import warnings
import logging
import math


def _initialize_cuda():
    """Initialize CUDA at module import time"""
    if cuda.is_available():
        try:
            # Force context creation
            cuda.select_device(0)
            # Create a dummy device array to establish context
            dummy = cuda.device_array(1, dtype=np.float32)
            del dummy
            return True
        except Exception as e:
            print(f"Warning: Could not initialize CUDA context: {e}")
            return False
    return False


# Initialize CUDA context when module loads
_CUDA_INITIALIZED = _initialize_cuda()

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


def prepare_gpu_data(data, nb_accounts, nb_scenarios, verbose=True):
    """
    Improved version with explicit validation and detailed logging
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

    if verbose:
        print("\n" + "=" * 80)
        print("INITIAL DATA PREPARATION - DETAILED LOG")
        print("=" * 80)

    for account_idx in range(min(nb_accounts, len(data['population']))):
        account_data = data['population'].iloc[account_idx]
        account_id = int(account_data['ID_COMPTE'])
        max_account_id = max(max_account_id, account_id)
        account_id_to_idx[account_id] = account_idx

        if verbose:
            print(f"\n--- Account {account_id} (Index {account_idx}) ---")
            print(f"  MT_VM (Initial Fund Value):       {account_data['MT_VM']:,.2f}")
            print(f"  MT_GAR_DECES (Death Benefit):     {account_data['MT_GAR_DECES']:,.2f}")
            print(f"  AGE_DEB (Starting Age):           {int(account_data['age_deb'])}")
            print(f"  TX_COMM_VENTE (Sales Commission): {account_data.get('TX_COMM_VENTE', 0.0):.4f}")
            print(f"  FRAIS_ACQUI (Acquisition Fee):    {account_data['FRAIS_ACQUI']:.2f}")
            print(f"  PC_REVENU_FDS (Fund Revenue):     {account_data['PC_REVENU_FDS']:.4f}")
            print(f"  PC_HONORAIRES_GEST (Mgmt Fee):    {account_data['PC_HONORAIRES_GEST']:.4f}")
            print(f"  TX_COMM_MAINTIEN (Ongoing Comm):  {account_data['TX_COMM_MAINTIEN']:.4f}")
            print(f"  FRAIS_ADMIN (Admin Fee):          {account_data['FRAIS_ADMIN']:.2f}")
            print(f"  FREQ_RESET_DECES (Reset Freq):    {account_data['FREQ_RESET_DECES']:.0f}")
            print(f"  MAX_RESET_DECES (Max Reset Age):  {account_data['MAX_RESET_DECES']:.0f}")

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

        # Store initial data
        initial_data[account_idx, 0] = float(account_data['MT_VM'])
        initial_data[account_idx, 1] = float(account_data['MT_GAR_DECES'])
        initial_data[account_idx, 2] = int(account_data['age_deb'])
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
            states[combination_idx, 0] = float(account_id)
            states[combination_idx, 1] = float(scenario)
            states[combination_idx, 2] = float(account_idx)
            states[combination_idx, 3] = float(account_data['age_deb'])
            states[combination_idx, 4] = 0.0
            states[combination_idx, 5] = 0.0
            states[combination_idx, 6] = 0.0
            states[combination_idx, 7] = float(account_data['age_deb'])
            states[combination_idx, 8] = 0.0
            combination_idx += 1

    if verbose:
        print("\n" + "=" * 80)
        print("Account mapping verification:")
        for account_id, expected_idx in account_id_to_idx.items():
            actual_idx = account_mapping[account_id]
            status = "✓ OK" if actual_idx == expected_idx else "✗ ERROR"
            print(f"  Account {account_id} -> Index {actual_idx} {status}")

    return states, initial_data, account_ids, account_mapping


@cuda.jit
def gpu_calculate_year_transition(states, initial_data, lookups_mortality, lookups_lapse,
                                  lookups_discount_ext, lookups_discount_int, lookups_returns_ext,
                                  lookups_returns_int, results, year, projection_type, fund_shock, start_year,
                                  max_years):
    """GPU kernel for year transition calculations - NOW STORES RENDEMENT"""

    combination_idx = cuda.grid(1)
    if combination_idx >= states.shape[0]:
        return

    if states[combination_idx, STATE_IS_TERMINATED] > 0:
        return

    account_idx = int(states[combination_idx, STATE_ACCOUNT_IDX])

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
            RENDEMENT = 0.0  # No return at year 0

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
            RENDEMENT = 0.0  # No return at year 0

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
            results[result_idx, 9] = RENDEMENT  # Store RENDEMENT
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
            results[result_idx, 9] = 0.0  # RENDEMENT = 0
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

    RENDEMENT = MT_VM_DEB * RENDEMENT_rate  # Calculate actual RENDEMENT amount
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
        results[result_idx, 9] = RENDEMENT  # Store RENDEMENT


def log_external_year_details(external_results, year, account_id=None, scenario=None):
    """Log detailed information for a specific year"""
    year_data = external_results[external_results[:, 2] == year]

    valid_year_data = year_data[year_data[:, 0] != 0]

    if account_id is not None:
        valid_year_data = valid_year_data[valid_year_data[:, 0] == account_id]
    if scenario is not None:
        valid_year_data = valid_year_data[valid_year_data[:, 1] == scenario]

    if len(valid_year_data) == 0:
        return

    print(f"\n--- Year {year} Details ---")
    for row in valid_year_data:
        acc_id, scn, yr, age, vm, death_ben, survie, flux, vp_flux, rendement = row
        print(f"  Account {int(acc_id)}, Scenario {int(scn)}:")
        print(f"    Age: {int(age)}")
        print(f"    Fund Value: {vm:,.2f}")
        print(f"    Death Benefit: {death_ben:,.2f}")
        print(f"    Survival Prob: {survie:.6f}")
        print(f"    Rendement: {rendement:,.2f}")
        print(f"    Net Cash Flow: {flux:,.2f}")
        print(f"    PV Net Cash Flow: {vp_flux:,.2f}")


def run_gpu_projection(states, initial_data, lookups, nb_years: int, projection_type: str,
                       fund_shock: float = 0.0, start_year: int = 0, verbose=True) -> np.ndarray:
    """Run projection on GPU with detailed logging"""
    proj_type_num = 0 if projection_type == "EXTERNE" else 1

    # CHANGED: Now 10 columns instead of 9 to include RENDEMENT
    max_results = states.shape[0] * (nb_years + 1)
    results = np.zeros((max_results, 10), dtype=np.float64)

    if verbose:
        print(f"\n{'=' * 80}")
        print(f"RUNNING {projection_type} PROJECTION")
        print(f"{'=' * 80}")
        print(f"States shape: {states.shape}")
        print(f"Max results: {max_results}")
        print(f"Fund shock: {fund_shock}")
        print(f"Start year: {start_year}")

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

    if verbose:
        print(f"GPU grid: {blocks_per_grid} blocks, {threads_per_block} threads per block")

    for year in range(nb_years + 1):
        if verbose:
            print(f"\nProcessing year {year}...")

        gpu_calculate_year_transition[blocks_per_grid, threads_per_block](
            d_states, d_initial_data, d_mortality, d_lapse, d_discount_ext, d_discount_int,
            d_returns_ext, d_returns_int, d_results, year, proj_type_num,
            fund_shock, start_year, nb_years
        )
        cuda.synchronize()

        if verbose and year <= 2:  # Log first few years in detail
            temp_results = d_results.copy_to_host()
            log_external_year_details(temp_results, year)

    results = d_results.copy_to_host()
    states = d_states.copy_to_host()

    return results, states


@cuda.jit
def gpu_calculate_internal_scenarios(external_results, initial_data, lookups_mortality,
                                     lookups_lapse, lookups_discount_ext, lookups_discount_int,
                                     lookups_returns_ext, lookups_returns_int, internal_results,
                                     nb_sc_int, nb_an_projection_int, fund_shock, account_mapping):
    """GPU kernel for internal scenario calculations"""

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

    if account_id >= account_mapping.shape[0] or account_id < 0:
        internal_results[external_idx] = 0.0
        return

    account_idx = account_mapping[account_id]
    if account_idx == -1 or account_idx >= initial_data.shape[0] or account_idx < 0:
        internal_results[external_idx] = 0.0
        return

    AGE_BASE = int(initial_data[account_idx, 2])
    max_years_for_age = 99 - AGE_BASE - year
    actual_max_years = min(nb_an_projection_int, max_years_for_age)

    if actual_max_years < 0:
        internal_results[external_idx] = 0.0
        return

    sum_vp = 0.0
    sum_compensation = 0.0

    for internal_scenario in range(1, nb_sc_int + 1):
        MT_VM_PROJ = fund_value
        MT_GAR_DECES_PROJ = death_benefit
        TX_SURVIE = survival_prob

        if fund_shock > 0.0:
            MT_VM_PROJ = MT_VM_PROJ * (1.0 - fund_shock)

        scenario_sum = 0.0
        scenario_compensation = 0.0

        for internal_year in range(1, actual_max_years + 1):
            if TX_SURVIE < 1e-15 or MT_VM_PROJ < 1e-15:
                break

            an_proj = year + internal_year
            current_age = AGE_BASE + year + internal_year

            if current_age >= lookups_mortality.shape[0]:
                break
            if an_proj >= lookups_returns_int.shape[0]:
                break

            RENDEMENT_rate = 0.0
            if internal_scenario < lookups_returns_int.shape[1] and an_proj < lookups_returns_int.shape[0]:
                RENDEMENT_rate = lookups_returns_int[an_proj, internal_scenario]

            MT_VM_DEB = MT_VM_PROJ
            RENDEMENT = MT_VM_DEB * RENDEMENT_rate
            MT_VM_HALF_REND = MT_VM_DEB + RENDEMENT / 2.0
            FRAIS = -MT_VM_HALF_REND * initial_data[account_idx, 5]

            MT_VM_PROJ = MT_VM_DEB + RENDEMENT + FRAIS
            if MT_VM_PROJ < 0.0:
                MT_VM_PROJ = 0.0

            FREQ_RESET = initial_data[account_idx, 9]
            MAX_RESET = initial_data[account_idx, 10]

            if FREQ_RESET > 0.5:
                if current_age <= MAX_RESET:
                    if MT_VM_PROJ > MT_GAR_DECES_PROJ:
                        MT_GAR_DECES_PROJ = MT_VM_PROJ

            QX = 0.0
            WX = 0.0
            if current_age < lookups_mortality.shape[0]:
                QX = lookups_mortality[current_age]
            if an_proj < lookups_lapse.shape[0]:
                WX = lookups_lapse[an_proj]

            TX_SURVIE_DEB = TX_SURVIE
            TX_SURVIE = TX_SURVIE_DEB * (1.0 - QX) * (1.0 - WX)

            REVENUS = -FRAIS * TX_SURVIE_DEB
            FRAIS_GEST = -MT_VM_HALF_REND * initial_data[account_idx, 6] * TX_SURVIE_DEB
            COMMISSIONS = -MT_VM_HALF_REND * initial_data[account_idx, 7] * TX_SURVIE_DEB
            FRAIS_GEN = -initial_data[account_idx, 8] * TX_SURVIE_DEB

            shortfall = MT_GAR_DECES_PROJ - MT_VM_PROJ
            if shortfall > 0.0:
                PMT_GARANTIE = -shortfall * QX * TX_SURVIE_DEB
            else:
                PMT_GARANTIE = 0.0

            FLUX_NET = REVENUS + FRAIS_GEST + COMMISSIONS + FRAIS_GEN + PMT_GARANTIE

            TX_ACTU = 1.0
            if an_proj < lookups_discount_ext.shape[0]:
                TX_ACTU = lookups_discount_ext[an_proj]

            VP_FLUX_NET = FLUX_NET * TX_ACTU

            if year > 0:
                if year < lookups_discount_int.shape[0]:
                    TX_ACTU_INT = lookups_discount_int[year]
                    if TX_ACTU_INT > 1e-15:
                        VP_FLUX_NET = VP_FLUX_NET / TX_ACTU_INT

            y = VP_FLUX_NET - scenario_compensation
            t = scenario_sum + y
            scenario_compensation = (t - scenario_sum) - y
            scenario_sum = t

        y = scenario_sum - sum_compensation
        t = sum_vp + y
        sum_compensation = (t - sum_vp) - y
        sum_vp = t

    internal_results[external_idx] = sum_vp / float(nb_sc_int)


def gpu_acfc_algorithm_complete(data_path: str = ".", nb_accounts: int = 4, nb_scenarios: int = 10,
                                nb_years: int = 10, nb_sc_int: int = 10, nb_an_projection_int: int = 10,
                                choc_capital: float = 0.35, hurdle_rt: float = 0.10,
                                verbose: bool = True,
                                log_account_id: int = None, log_scenario: int = None,
                                log_max_years: int = None) -> pd.DataFrame:
    """
    Complete GPU-Accelerated ACFC Algorithm with detailed logging including TX_SURVIE and RENDEMENT

    Args:
        log_account_id: If specified, only log details for this account ID
        log_scenario: If specified, only log details for this scenario
        log_max_years: If specified, only log details up to this year (e.g., 10 for years 0-9)
    """

    if verbose:
        print("\n" + "=" * 80)
        print("GPU-ACCELERATED ACFC ALGORITHM - COMPLETE EXECUTION LOG")
        print("=" * 80)
        print(f"Parameters:")
        print(f"  Accounts: {nb_accounts}")
        print(f"  External Scenarios: {nb_scenarios}")
        print(f"  Projection Years: {nb_years}")
        print(f"  Internal Scenarios: {nb_sc_int}")
        print(f"  Internal Projection Years: {nb_an_projection_int}")
        print(f"  Capital Shock: {choc_capital}")
        print(f"  Hurdle Rate: {hurdle_rt}")
        print(f"\nDetailed Logging Filters:")
        print(f"  Account ID Filter: {log_account_id if log_account_id is not None else 'All accounts'}")
        print(f"  Scenario Filter: {log_scenario if log_scenario is not None else 'All scenarios'}")
        print(f"  Max Years Filter: {log_max_years if log_max_years is not None else 'All years'}")

    print("\nPhase 1: Loading input data...")
    data = load_input_data(data_path, nb_accounts)

    print("\nPhase 2: Creating GPU lookup tables...")
    lookups = create_gpu_lookup_tables(data)

    print("\nPhase 3: Preparing GPU data...")
    states, initial_data, account_ids, account_mapping = prepare_gpu_data(
        data, nb_accounts, nb_scenarios, verbose=verbose
    )

    print("\nPhase 4: Running GPU external projections...")
    external_results, final_states = run_gpu_projection(
        states, initial_data, lookups, nb_years, 'EXTERNE', verbose=verbose
    )

    if verbose:
        print(f"\n{'=' * 80}")
        print("EXTERNAL PROJECTION RESULTS SUMMARY")
        print(f"{'=' * 80}")
        print(f"Total results: {len(external_results)}")
        valid_mask = external_results[:, 0] != 0
        valid_external_results = external_results[valid_mask]
        print(f"Valid results: {len(valid_external_results)}")

        for account_id in sorted(np.unique(valid_external_results[:, 0])):
            account_mask = valid_external_results[:, 0] == account_id
            account_results = valid_external_results[account_mask]
            print(f"\nAccount {int(account_id)}:")
            print(f"  Total results: {len(account_results)}")
            print(f"  Scenarios: {sorted(set(account_results[:, 1]))}")

            # Show year 0 and year 1 details for first scenario
            for yr in [0, 1]:
                yr_data = account_results[(account_results[:, 2] == yr) & (account_results[:, 1] == 1)]
                if len(yr_data) > 0:
                    row = yr_data[0]
                    print(f"  Year {yr} (Scenario 1):")
                    print(f"    Fund Value: {row[4]:,.2f}")
                    print(f"    Death Benefit: {row[5]:,.2f}")
                    print(f"    Survival Prob: {row[6]:.6f}")
                    print(f"    Rendement: {row[9]:,.2f}")
                    print(f"    Net Cash Flow: {row[7]:,.2f}")
                    print(f"    PV Cash Flow: {row[8]:,.2f}")
    else:
        valid_mask = external_results[:, 0] != 0
        valid_external_results = external_results[valid_mask]

    if len(valid_external_results) == 0:
        print("ERROR: No valid external results found!")
        return pd.DataFrame(), pd.DataFrame()

    print("\nPhase 5: Running GPU internal calculations...")

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

    if verbose:
        print(f"Calculating reserves (no shock)...")
    gpu_calculate_internal_scenarios[blocks_per_grid, threads_per_block](
        d_external_results, d_initial_data, d_mortality, d_lapse, d_discount_ext, d_discount_int,
        d_returns_ext, d_returns_int, d_reserve_results, nb_sc_int, nb_an_projection_int,
        0.0, d_account_mapping
    )
    cuda.synchronize()

    if verbose:
        print(f"Calculating capital (with {choc_capital} shock)...")
    gpu_calculate_internal_scenarios[blocks_per_grid, threads_per_block](
        d_external_results, d_initial_data, d_mortality, d_lapse, d_discount_ext, d_discount_int,
        d_returns_ext, d_returns_int, d_capital_results, nb_sc_int, nb_an_projection_int,
        choc_capital, d_account_mapping
    )
    cuda.synchronize()

    reserve_results = d_reserve_results.copy_to_host()
    capital_results = d_capital_results.copy_to_host()

    print("\nPhase 6: Calculating distributable flows...")

    final_results = []
    detailed_results = []
    from collections import defaultdict
    grouped_external = defaultdict(list)
    grouped_reserves = defaultdict(list)
    grouped_capital = defaultdict(list)

    for i, row in enumerate(valid_external_results):
        account_id = int(row[0])
        scenario = int(row[1])
        year = int(row[2])
        key = f"{account_id}_{scenario}"

        grouped_external[key].append({
            'year': year,
            'TX_SURVIE': row[6],  # Extract TX_SURVIE
            'RENDEMENT': row[9],  # Extract RENDEMENT
            'FLUX_NET': row[7],
            'VP_FLUX_NET': row[8]
        })
        grouped_reserves[key].append((year, reserve_results[i]))
        grouped_capital[key].append((year, capital_results[i] - reserve_results[i]))

    if verbose:
        print(f"\n{'=' * 80}")
        print("DISTRIBUTABLE FLOWS CALCULATION")
        print(f"{'=' * 80}")
        if log_account_id is not None or log_scenario is not None or log_max_years is not None:
            print(f"Detailed logging filters:")
            if log_account_id is not None:
                print(f"  Account ID: {log_account_id}")
            if log_scenario is not None:
                print(f"  Scenario: {log_scenario}")
            if log_max_years is not None:
                print(f"  Max Years: {log_max_years}")

    for key in grouped_external:
        account_id, scenario = key.split('_')
        account_id = int(account_id)
        scenario = int(scenario)

        external_data = sorted(grouped_external[key], key=lambda x: x['year'])
        reserve_data = dict(sorted(grouped_reserves[key], key=lambda x: x[0]))
        capital_data = dict(sorted(grouped_capital[key], key=lambda x: x[0]))

        if verbose and scenario == 1:
            print(f"\nAccount {account_id}, Scenario {scenario}:")
            print(
                f"  {'Year':<6} {'TX_SURVIE':<12} {'Rendement':<15} {'Ext CF':<12} {'Reserve':<12} {'Capital':<12} {'Profit':<12} {'Distrib':<12} {'PV Distrib':<12}")
            print(
                f"  {'-' * 6} {'-' * 12} {'-' * 15} {'-' * 12} {'-' * 12} {'-' * 12} {'-' * 12} {'-' * 12} {'-' * 12}")

        distributable_pvs = []
        prev_reserve = 0.0
        prev_capital = 0.0

        for ext_data in external_data:
            year = ext_data['year']
            tx_survie = ext_data['TX_SURVIE']
            rendement = ext_data['RENDEMENT']
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

            # Apply filtering for detailed logging
            should_log = True
            if log_account_id is not None and account_id != log_account_id:
                should_log = False
            if log_scenario is not None and scenario != log_scenario:
                should_log = False
            if log_max_years is not None and year >= log_max_years:
                should_log = False

            # Store detailed year-by-year results with TX_SURVIE and RENDEMENT (if passes filter)
            if should_log:
                detailed_results.append({
                    'ID_COMPTE': account_id,
                    'scn_eval': scenario,
                    'an_proj': year,
                    'TX_SURVIE': tx_survie,  # Added
                    'RENDEMENT': rendement,  # Added
                    'FLUX_NET_EXT': external_cf,
                    'RESERVE': current_reserve,
                    'CAPITAL_REQUIREMENT': current_capital,
                    'PROFIT': profit,
                    'FLUX_DISTRIBUABLE': distributable,
                    'VP_FLUX_DISTRIBUABLE_YEARLY': pv_distributable
                })

            if verbose and scenario == 1:
                print(
                    f"  {year:<6} {tx_survie:>12.6f} {rendement:>15,.2f} {external_cf:>12,.2f} {current_reserve:>12,.2f} {current_capital:>12,.2f} "
                    f"{profit:>12,.2f} {distributable:>12,.2f} {pv_distributable:>12,.2f}")

            prev_reserve = current_reserve
            prev_capital = current_capital

        total_pv_distributable = sum(distributable_pvs)

        if verbose and scenario == 1:
            print(
                f"  {'TOTAL':<6} {'':<12} {'':<15} {'':<12} {'':<12} {'':<12} {'':<12} {'':<12} {total_pv_distributable:>12,.2f}")

        final_results.append({
            'ID_COMPTE': account_id,
            'scn_eval': scenario,
            'VP_FLUX_DISTRIBUABLES': total_pv_distributable
        })

    print("\nPhase 7: Converting to DataFrame...")
    output_df = pd.DataFrame(final_results)
    detailed_df = pd.DataFrame(detailed_results)

    if verbose:
        print(f"\n{'=' * 80}")
        print("FINAL RESULTS SUMMARY")
        print(f"{'=' * 80}")
        print(f"Total results: {len(output_df)}")
        print(f"\nMean VP_FLUX_DISTRIBUABLES: {output_df['VP_FLUX_DISTRIBUABLES'].mean():,.2f}")
        print(f"Min: {output_df['VP_FLUX_DISTRIBUABLES'].min():,.2f}")
        print(f"Max: {output_df['VP_FLUX_DISTRIBUABLES'].max():,.2f}")
        print(f"\nResults by account:")
        for account_id in sorted(output_df['ID_COMPTE'].unique()):
            account_data = output_df[output_df['ID_COMPTE'] == account_id]
            print(f"  Account {account_id}: Mean = {account_data['VP_FLUX_DISTRIBUABLES'].mean():,.2f}, "
                  f"Scenarios = {len(account_data)}")

        print(f"\n{'=' * 80}")
        print("DETAILED RESULTS")
        print(f"{'=' * 80}")
        print(f"Total detailed records: {len(detailed_df)}")
        print(f"\nFirst few rows:")
        print(detailed_df.head(10))

    return output_df, detailed_df


def initialize_cuda_context():
    """Initialize CUDA context explicitly"""
    try:
        device = cuda.select_device(0)
        ctx = cuda.current_context()

        @cuda.jit
        def dummy_kernel(output):
            output[0] = 1.0

        test_array = np.zeros(1, dtype=np.float64)
        d_test = cuda.to_device(test_array)

        dummy_kernel[1, 1](d_test)
        cuda.synchronize()

        result = d_test.copy_to_host()

        if result[0] == 1.0:
            print(f"✓ CUDA context initialized successfully on device: {device.name.decode()}")
            print(f"✓ Context verification passed")
            return True
        else:
            print("✗ Context verification failed")
            return False

    except Exception as e:
        print(f"✗ Failed to initialize CUDA context: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_cuda_environment():
    """Comprehensive CUDA environment check"""
    print("\n" + "=" * 80)
    print("CUDA ENVIRONMENT DIAGNOSTICS")
    print("=" * 80)

    print(f"CUDA Available: {cuda.is_available()}")

    if cuda.is_available():
        try:
            print(f"Number of GPUs: {len(cuda.gpus)}")
            for i, gpu in enumerate(cuda.gpus):
                print(f"  GPU {i}: {gpu.name.decode()}")
        except Exception as e:
            print(f"Error accessing GPU info: {e}")

    print(f"Numba version: {numba.__version__}")

    try:
        from numba.cuda.cudadrv.libs import test
        test()
        print("✓ CUDA libraries detected")
    except Exception as e:
        print(f"✗ CUDA libraries not properly detected: {e}")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    check_cuda_environment()

    if not cuda.is_available():
        print("CUDA is not available. Please install CUDA and ensure your GPU supports it.")
        exit(1)

    if not _CUDA_INITIALIZED:
        print("\n⚠ CUDA context could not be initialized at module load time.")
        print("Attempting manual initialization...")

        try:
            cuda.close()
            cuda.select_device(0)

            test = cuda.device_array(10, dtype=np.float32)
            del test

            print("✓ Manual initialization successful")

        except Exception as e:
            print(f"✗ Manual initialization failed: {e}")
            print("\nThis might be a Docker/permissions issue.")
            print("Try running with: docker run --gpus all --privileged ...")
            exit(1)
    else:
        print("✓ CUDA context already initialized at module load")

    print(f"\nCUDA devices detected: {len(cuda.gpus)}")
    for i, gpu in enumerate(cuda.gpus):
        print(f"  Device {i}: {gpu.name.decode()}")

    data_path = "data_in"

    results, detailed_results = gpu_acfc_algorithm_complete(
        data_path=data_path,
        nb_accounts=1,
        nb_scenarios=1,
        nb_years=100,
        nb_sc_int=1,
        nb_an_projection_int=100,
        choc_capital=0.35,
        hurdle_rt=0.10,
        verbose=True,
        log_account_id=1,  # Only log acco
        log_scenario=1,  # Only log scenario 1
        log_max_years=10  # Only log first 10 years (0-9)
    )

    print("\nFinal Summary Results:")
    print(results)
    results.to_csv('test/gpu_results_complete.csv', index=False)

    print("\nSaving detailed year-by-year results...")
    detailed_results.to_csv('test/gpu_results_detailed.csv', index=False)
    filter_msg = ""
    if log_account_id is not None or log_scenario is not None or log_max_years is not None:
        filters = []
        if log_account_id is not None:
            filters.append(f"account={log_account_id}")
        if log_scenario is not None:
            filters.append(f"scenario={log_scenario}")
        if log_max_years is not None:
            filters.append(f"years=0-{log_max_years - 1}")
        filter_msg = f" [Filtered: {', '.join(filters)}]"
    print(f"✓ Detailed results saved to 'test/gpu_results_detailed.csv' ({len(detailed_results)} rows{filter_msg})")
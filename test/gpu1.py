import logging
import math
import warnings
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from numba import cuda
import numba
from paths import HERE

# --- Logger Setup ---
# We initialize the logger here, but configure it in the __main__ block
# so that other scripts importing this module can define their own logging configuration.
logger = logging.getLogger(__name__)


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
            # Use WARNING for non-critical failures at import time
            logger.warning(f"Could not initialize CUDA context on module load: {e}", exc_info=True)
            return False
    return False


# Initialize CUDA context when module loads
_CUDA_INITIALIZED = _initialize_cuda()

warnings.filterwarnings('ignore')

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


def load_input_data(data_path: Path, nb_accounts: int = None) -> Dict:
    """Load all input data files"""
    try:
        population = pd.read_csv(data_path.joinpath("population_fixed.csv"))
        if nb_accounts is not None:
            population = population.head(nb_accounts)
        rendement = pd.read_csv(data_path.joinpath("rendement1.csv"))
        tx_deces = pd.read_csv(data_path.joinpath("tx_deces_fixed.csv"))
        tx_interet = pd.read_csv(data_path.joinpath("tx_interet_fixed.csv"))
        tx_interet_int = pd.read_csv(data_path.joinpath("tx_interet_int_fixed.csv"))
        tx_retrait = pd.read_csv(data_path.joinpath("tx_retrait_fixed.csv"))

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
        logger.error(f"Error loading input files: {e}", exc_info=True)
        raise


def create_gpu_lookup_tables(data: Dict, max_age: int = 120, max_year: int = 50, max_scenarios: int = 1000) -> Dict:
    """Create GPU-friendly lookup tables as NumPy arrays"""
    # This function is fast and has no prints, so no changes needed.
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
    Prepare data for GPU, with detailed DEBUG-level logging.
    """
    total_combinations = min(nb_accounts, len(data['population'])) * nb_scenarios

    # Initialize arrays
    states = np.zeros((total_combinations, 9), dtype=np.float64)
    initial_data = np.zeros((min(nb_accounts, len(data['population'])), 11), dtype=np.float64)
    account_ids = np.zeros(min(nb_accounts, len(data['population'])), dtype=np.float64)

    # Determine the maximum account ID first
    max_account_id = 0
    account_id_to_idx = {}

    logger.debug("=" * 80)
    logger.debug("INITIAL DATA PREPARATION - DETAILED LOG")
    logger.debug("=" * 80)

    for account_idx in range(min(nb_accounts, len(data['population']))):
        account_data = data['population'].iloc[account_idx]
        account_id = int(account_data['ID_COMPTE'])
        max_account_id = max(max_account_id, account_id)
        account_id_to_idx[account_id] = account_idx

        logger.debug(f"--- Account {account_id} (Index {account_idx}) ---")
        logger.debug(f"  MT_VM (Initial Fund Value):       {account_data['MT_VM']:,.2f}")
        logger.debug(f"  MT_GAR_DECES (Death Benefit):     {account_data['MT_GAR_DECES']:,.2f}")
        logger.debug(f"  AGE_DEB (Starting Age):           {int(account_data['age_deb'])}")
        logger.debug(f"  TX_COMM_VENTE (Sales Commission): {account_data.get('TX_COMM_VENTE', 0.0):.4f}")
        logger.debug(f"  FRAIS_ACQUI (Acquisition Fee):    {account_data['FRAIS_ACQUI']:.2f}")
        logger.debug(f"  PC_REVENU_FDS (Fund Revenue):     {account_data['PC_REVENU_FDS']:.4f}")
        logger.debug(f"  PC_HONORAIRES_GEST (Mgmt Fee):    {account_data['PC_HONORAIRES_GEST']:.4f}")
        logger.debug(f"  TX_COMM_MAINTIEN (Ongoing Comm):  {account_data['TX_COMM_MAINTIEN']:.4f}")
        logger.debug(f"  FRAIS_ADMIN (Admin Fee):          {account_data['FRAIS_ADMIN']:.2f}")
        logger.debug(f"  FREQ_RESET_DECES (Reset Freq):    {account_data['FREQ_RESET_DECES']:.0f}")
        logger.debug(f"  MAX_RESET_DECES (Max Reset Age):  {account_data['MAX_RESET_DECES']:.0f}")

    logger.debug(f"Account ID range: 1 to {max_account_id}")
    logger.debug(f"Number of accounts: {len(account_id_to_idx)}")
    logger.debug(f"Account IDs: {sorted(account_id_to_idx.keys())}")

    # Create mapping array with size based on max account ID
    account_mapping = np.full(max_account_id + 1, -1, dtype=np.int32)

    # Populate data arrays
    combination_idx = 0
    for account_idx in range(min(nb_accounts, len(data['population']))):
        account_data = data['population'].iloc[account_idx]
        account_id = int(account_data['ID_COMPTE'])
        account_ids[account_idx] = float(account_id)
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

    logger.debug("=" * 80)
    logger.debug("Account mapping verification:")
    for account_id, expected_idx in account_id_to_idx.items():
        actual_idx = account_mapping[account_id]
        status = "✓ OK" if actual_idx == expected_idx else "✗ ERROR"
        logger.debug(f"  Account {account_id} -> Index {actual_idx} {status}")

    return states, initial_data, account_ids, account_mapping


@cuda.jit
def gpu_calculate_year_transition(states, initial_data, lookups_mortality, lookups_lapse,
                                  lookups_discount_ext, lookups_discount_int, lookups_returns_ext,
                                  lookups_returns_int, results, year, projection_type, fund_shock, start_year,
                                  max_years):
    """GPU kernel for year transition calculations - NOW STORES RENDEMENT"""
    # Kernels cannot log, so this remains unchanged.
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


def _log_external_year_details(external_results, year, account_id=None, scenario=None):
    """Log detailed DEBUG information for a specific year"""
    year_data = external_results[external_results[:, 2] == year]
    valid_year_data = year_data[year_data[:, 0] != 0]

    if account_id is not None:
        valid_year_data = valid_year_data[valid_year_data[:, 0] == account_id]
    if scenario is not None:
        valid_year_data = valid_year_data[valid_year_data[:, 1] == scenario]

    if len(valid_year_data) == 0:
        return

    logger.debug(f"--- Year {year} Details ---")
    for row in valid_year_data:
        acc_id, scn, yr, age, vm, death_ben, survie, flux, vp_flux, rendement = row
        log_message = (
            f"  Account {int(acc_id)}, Scenario {int(scn)}:\n"
            f"    Age: {int(age)}\n"
            f"    Fund Value: {vm:,.2f}\n"
            f"    Death Benefit: {death_ben:,.2f}\n"
            f"    Survival Prob: {survie:.6f}\n"
            f"    Rendement: {rendement:,.2f}\n"
            f"    Net Cash Flow: {flux:,.2f}\n"
            f"    PV Net Cash Flow: {vp_flux:,.2f}"
        )
        logger.debug(log_message)


def run_gpu_projection(states, initial_data, lookups, nb_years: int, projection_type: str,
                       fund_shock: float = 0.0, start_year: int = 0) -> np.ndarray:
    """Run projection on GPU with INFO for progress and DEBUG for details"""
    proj_type_num = 0 if projection_type == "EXTERNE" else 1
    max_results = states.shape[0] * (nb_years + 1)
    results = np.zeros((max_results, 10), dtype=np.float64)

    logger.info(f"Running {projection_type} projection...")
    logger.debug(f"States shape: {states.shape}, Max results: {max_results}")
    logger.debug(f"Fund shock: {fund_shock}, Start year: {start_year}")

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

    logger.debug(f"GPU grid: {blocks_per_grid} blocks, {threads_per_block} threads per block")

    for year in range(nb_years + 1):
        logger.debug(f"Processing {projection_type} year {year}...")

        gpu_calculate_year_transition[blocks_per_grid, threads_per_block](
            d_states, d_initial_data, d_mortality, d_lapse, d_discount_ext, d_discount_int,
            d_returns_ext, d_returns_int, d_results, year, proj_type_num,
            fund_shock, start_year, nb_years
        )
        cuda.synchronize()

        # Log first few years in detail if debug is enabled
        if logger.isEnabledFor(logging.DEBUG) and year <= 2:
            temp_results = d_results.copy_to_host()
            _log_external_year_details(temp_results, year)

    results = d_results.copy_to_host()
    states = d_states.copy_to_host()

    return results, states


@cuda.jit
def gpu_calculate_internal_scenarios(external_results, initial_data, lookups_mortality,
                                     lookups_lapse, lookups_discount_ext, lookups_discount_int,
                                     lookups_returns_ext, lookups_returns_int, internal_results,
                                     nb_sc_int, nb_an_projection_int, fund_shock, account_mapping,
                                     detailed_internal_results, log_internal_scenario_id):
    """GPU kernel for internal scenario calculations with optional detailed logging for a specific internal scenario"""
    # Kernels cannot log, so this remains unchanged.
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

            # Store detailed internal scenario results if this is the logged scenario
            if log_internal_scenario_id > 0 and internal_scenario == log_internal_scenario_id:
                detail_idx = external_idx * nb_an_projection_int + (internal_year - 1)
                if detail_idx < detailed_internal_results.shape[0]:
                    detailed_internal_results[detail_idx, 0] = account_id
                    detailed_internal_results[detail_idx, 1] = external_results[external_idx, 1]  # external scenario
                    detailed_internal_results[detail_idx, 2] = year  # external year
                    detailed_internal_results[detail_idx, 3] = internal_scenario
                    detailed_internal_results[detail_idx, 4] = internal_year
                    detailed_internal_results[detail_idx, 5] = current_age
                    detailed_internal_results[detail_idx, 6] = MT_VM_PROJ
                    detailed_internal_results[detail_idx, 7] = MT_GAR_DECES_PROJ
                    detailed_internal_results[detail_idx, 8] = TX_SURVIE
                    detailed_internal_results[detail_idx, 9] = RENDEMENT
                    detailed_internal_results[detail_idx, 10] = FLUX_NET
                    detailed_internal_results[detail_idx, 11] = VP_FLUX_NET

            y = VP_FLUX_NET - scenario_compensation
            t = scenario_sum + y
            scenario_compensation = (t - scenario_sum) - y
            scenario_sum = t

        y = scenario_sum - sum_compensation
        t = sum_vp + y
        sum_compensation = (t - sum_vp) - y
        sum_vp = t

    internal_results[external_idx] = sum_vp / float(nb_sc_int)


def gpu_acfc_algorithm_complete(data_path: Path = None, nb_accounts: int = 4, nb_scenarios: int = 10,
                                nb_years: int = 10, nb_sc_int: int = 10, nb_an_projection_int: int = 10,
                                choc_capital: float = 0.35, hurdle_rt: float = 0.10,
                                log_account_id: int = None, log_scenario: int = None,
                                log_max_years: int = None, log_internal_scenario: int = None,
                                population: pd.DataFrame = None,
                                rendement: pd.DataFrame = None,
                                tx_deces: pd.DataFrame = None,
                                tx_interet: pd.DataFrame = None,
                                tx_interet_int: pd.DataFrame = None,
                                tx_retrait: pd.DataFrame = None) -> pd.DataFrame:
    """
    Complete GPU-Accelerated ACFC Algorithm with structured logging.

    Args:
        (Same as before, but 'verbose' is removed)
    """
    log_params = [log_account_id, log_scenario, log_max_years]
    num_specified = sum(p is not None for p in log_params)

    if 0 < num_specified < len(log_params):
        raise ValueError(
            "Inconsistent detailed logging parameters. Please specify all or none of: "
            "log_account_id, log_scenario, log_max_years"
        )

    logger.info("=" * 80)
    logger.info("STARTING GPU-ACCELERATED ACFC ALGORITHM")
    logger.info("=" * 80)
    logger.info("Parameters:")
    logger.info(f"  Accounts: {nb_accounts}, External Scenarios: {nb_scenarios}, Projection Years: {nb_years}")
    logger.info(f"  Internal Scenarios: {nb_sc_int}, Internal Projection Years: {nb_an_projection_int}")
    logger.info(f"  Capital Shock: {choc_capital}, Hurdle Rate: {hurdle_rt}")
    logger.info("Detailed Logging Filters:")
    logger.info(f"  Account ID Filter: {log_account_id if log_account_id is not None else 'All'}")
    logger.info(f"  External Scenario Filter: {log_scenario if log_scenario is not None else 'All'}")
    logger.info(f"  Max Years Filter: {log_max_years if log_max_years is not None else 'All'}")
    logger.info(f"  Internal Scenario Filter: {log_internal_scenario if log_internal_scenario is not None else 'None'}")

    logger.info("--- Phase 1: Loading input data ---")
    data = {}
    df_map = {'population': population, 'rendement': rendement, 'tx_deces': tx_deces,
              'tx_interet': tx_interet, 'tx_interet_int': tx_interet_int, 'tx_retrait': tx_retrait}

    for name, df in df_map.items():
        if df is not None: data[name] = df

    if len(data) < len(df_map):
        if data_path is None: raise ValueError("Must provide either data_path or all dataframes")
        files_to_load = {'population': "population_fixed.csv", 'rendement': "rendement1.csv",
                         'tx_deces': "tx_deces_fixed.csv", 'tx_interet': "tx_interet_fixed.csv",
                         'tx_interet_int': "tx_interet_int_fixed.csv", 'tx_retrait': "tx_retrait_fixed.csv"}
        for name, filename in files_to_load.items():
            if name not in data: data[name] = pd.read_csv(data_path.joinpath(filename))

    if nb_accounts is not None:
        data['population'] = data['population'].head(nb_accounts)

    logger.info("--- Phase 2: Creating GPU lookup tables ---")
    lookups = create_gpu_lookup_tables(data, max_year=max(nb_years, nb_an_projection_int))

    logger.info("--- Phase 3: Preparing GPU data ---")
    states, initial_data, _, account_mapping = prepare_gpu_data(data, nb_accounts, nb_scenarios)

    logger.info("--- Phase 4: Running GPU external projections ---")
    external_results, final_states = run_gpu_projection(
        states, initial_data, lookups, nb_years, 'EXTERNE'
    )

    valid_mask = external_results[:, 0] != 0
    valid_external_results = external_results[valid_mask]

    logger.info("--- External Projection Summary ---")
    logger.info(f"Total results generated: {len(external_results)}")
    logger.info(f"Valid (non-zero) results: {len(valid_external_results)}")
    for account_id in sorted(np.unique(valid_external_results[:, 0])):
        account_results = valid_external_results[valid_external_results[:, 0] == account_id]
        logger.debug(f"Account {int(account_id)}: Total results: {len(account_results)}, "
                     f"Scenarios: {sorted(set(account_results[:, 1]))}")

    if len(valid_external_results) == 0:
        logger.error("No valid external results found! Cannot continue.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    logger.info("--- Phase 5: Running GPU internal calculations ---")
    reserve_results = np.zeros(len(valid_external_results), dtype=np.float64)
    capital_results = np.zeros(len(valid_external_results), dtype=np.float64)
    detailed_internal_results = np.zeros((len(valid_external_results) * nb_an_projection_int, 12), dtype=np.float64)

    d_external_results = cuda.to_device(valid_external_results)
    d_initial_data = cuda.to_device(initial_data)
    d_reserve_results = cuda.to_device(reserve_results)
    d_capital_results = cuda.to_device(capital_results)
    d_account_mapping = cuda.to_device(account_mapping)
    d_detailed_internal_results = cuda.to_device(detailed_internal_results)
    d_mortality = cuda.to_device(lookups['mortality'])
    d_lapse = cuda.to_device(lookups['lapse'])
    d_discount_ext = cuda.to_device(lookups['discount_ext'])
    d_discount_int = cuda.to_device(lookups['discount_int'])
    d_returns_ext = cuda.to_device(lookups['returns_ext'])
    d_returns_int = cuda.to_device(lookups['returns_int'])

    threads_per_block = 256
    blocks_per_grid = (len(valid_external_results) + threads_per_block - 1) // threads_per_block
    log_int_scn = log_internal_scenario if log_internal_scenario is not None else 0

    logger.info("Calculating reserves (no shock)...")
    gpu_calculate_internal_scenarios[blocks_per_grid, threads_per_block](
        d_external_results, d_initial_data, d_mortality, d_lapse, d_discount_ext, d_discount_int,
        d_returns_ext, d_returns_int, d_reserve_results, nb_sc_int, nb_an_projection_int,
        0.0, d_account_mapping, d_detailed_internal_results, log_int_scn)
    cuda.synchronize()

    logger.info(f"Calculating capital (with {choc_capital} shock)...")
    gpu_calculate_internal_scenarios[blocks_per_grid, threads_per_block](
        d_external_results, d_initial_data, d_mortality, d_lapse, d_discount_ext, d_discount_int,
        d_returns_ext, d_returns_int, d_capital_results, nb_sc_int, nb_an_projection_int,
        choc_capital, d_account_mapping, d_detailed_internal_results, 0)
    cuda.synchronize()

    reserve_results = d_reserve_results.copy_to_host()
    capital_results = d_capital_results.copy_to_host()
    detailed_internal_results = d_detailed_internal_results.copy_to_host()
    internal_df = pd.DataFrame()
    if log_internal_scenario is not None:
        valid_internal_mask = detailed_internal_results[:, 0] != 0
        valid_internal_results = detailed_internal_results[valid_internal_mask]
        if len(valid_internal_results) > 0:
            internal_df = pd.DataFrame(valid_internal_results, columns=['ID_COMPTE', 'scn_eval_ext', 'an_proj_ext', 'scn_int', 'an_proj_int', 'AGE', 'MT_VM_PROJ', 'MT_GAR_DECES_PROJ', 'TX_SURVIE', 'RENDEMENT', 'FLUX_NET', 'VP_FLUX_NET'])
            if log_account_id is not None: internal_df = internal_df[internal_df['ID_COMPTE'] == log_account_id]
            if log_scenario is not None: internal_df = internal_df[internal_df['scn_eval_ext'] == log_scenario]
            if log_max_years is not None: internal_df = internal_df[internal_df['an_proj_ext'] < log_max_years]
            logger.info(f"Generated {len(internal_df)} rows of detailed internal scenario results.")
            logger.debug(f"Sample internal scenario projections:\n{internal_df.head(10)}")

    logger.info("--- Phase 6: Calculating distributable flows ---")
    final_results, detailed_results = [], []
    from collections import defaultdict
    grouped_external = defaultdict(list)
    grouped_reserves = defaultdict(list)
    grouped_capital = defaultdict(list)

    for i, row in enumerate(valid_external_results):
        key = f"{int(row[0])}_{int(row[1])}"
        grouped_external[key].append({'year': int(row[2]), 'TX_SURVIE': row[6], 'RENDEMENT': row[9], 'FLUX_NET': row[7], 'VP_FLUX_NET': row[8]})
        grouped_reserves[key].append((int(row[2]), reserve_results[i]))
        grouped_capital[key].append((int(row[2]), capital_results[i] - reserve_results[i]))

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("--- DISTRIBUTABLE FLOWS CALCULATION (DETAIL) ---")
        if log_account_id is not None:
            logger.debug(f"Detailed logging filters: Account={log_account_id}, Scenario={log_scenario}, MaxYears={log_max_years}")

    for key in grouped_external:
        account_id, scenario = map(int, key.split('_'))
        external_data = sorted(grouped_external[key], key=lambda x: x['year'])
        reserve_data = dict(sorted(grouped_reserves[key], key=lambda x: x[0]))
        capital_data = dict(sorted(grouped_capital[key], key=lambda x: x[0]))

        # Build log table for debug
        log_table = []
        if logger.isEnabledFor(logging.DEBUG) and scenario == 1:
            log_table.append(f"\nAccount {account_id}, External Scenario {scenario}:")
            log_table.append(f"  {'Year':<6} {'TX_SURVIE':<12} {'Rendement':<15} {'Ext CF':<12} {'Reserve':<12} {'Capital':<12} {'Profit':<12} {'Distrib':<12} {'PV Distrib':<12}")
            log_table.append(f"  {'-' * 6} {'-' * 12} {'-' * 15} {'-' * 12} {'-' * 12} {'-' * 12} {'-' * 12} {'-' * 12} {'-' * 12}")

        distributable_pvs, prev_reserve, prev_capital = [], 0.0, 0.0
        for ext_data in external_data:
            year, tx_survie, rendement, external_cf = ext_data['year'], ext_data['TX_SURVIE'], ext_data['RENDEMENT'], ext_data['FLUX_NET']
            current_reserve, current_capital = reserve_data.get(year, 0.0), capital_data.get(year, 0.0)
            profit = external_cf + (current_reserve - prev_reserve) if year > 0 else external_cf + current_reserve
            distributable = profit + (current_capital - prev_capital) if year > 0 else profit + current_capital
            pv_distributable = distributable / ((1 + hurdle_rt) ** year) if year > 0 else distributable
            distributable_pvs.append(pv_distributable)

            if log_account_id is not None and account_id == log_account_id and scenario == log_scenario and year < log_max_years:
                detailed_results.append({'ID_COMPTE': account_id, 'scn_eval': scenario, 'an_proj': year, 'TX_SURVIE': tx_survie, 'RENDEMENT': rendement, 'FLUX_NET_EXT': external_cf, 'RESERVE': current_reserve, 'CAPITAL_REQUIREMENT': current_capital, 'PROFIT': profit, 'FLUX_DISTRIBUABLE': distributable, 'VP_FLUX_DISTRIBUABLE_YEARLY': pv_distributable})

            if logger.isEnabledFor(logging.DEBUG) and scenario == 1:
                log_table.append(f"  {year:<6} {tx_survie:>12.6f} {rendement:>15,.2f} {external_cf:>12,.2f} {current_reserve:>12,.2f} {current_capital:>12,.2f} {profit:>12,.2f} {distributable:>12,.2f} {pv_distributable:>12,.2f}")

            prev_reserve, prev_capital = current_reserve, current_capital

        total_pv_distributable = sum(distributable_pvs)
        if logger.isEnabledFor(logging.DEBUG) and scenario == 1:
            log_table.append(f"  {'TOTAL':<6} {'':<12} {'':<15} {'':<12} {'':<12} {'':<12} {'':<12} {'':<12} {total_pv_distributable:>12,.2f}")
            logger.debug("\n".join(log_table))

        final_results.append({'ID_COMPTE': account_id, 'scn_eval': scenario, 'VP_FLUX_DISTRIBUABLES': total_pv_distributable})

    logger.info("--- Phase 7: Finalizing results ---")
    output_df = pd.DataFrame(final_results)
    detailed_df = pd.DataFrame(detailed_results)

    logger.info("=" * 80)
    logger.info("FINAL RESULTS SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total results: {len(output_df)}")
    logger.info(f"Mean VP_FLUX_DISTRIBUABLES: {output_df['VP_FLUX_DISTRIBUABLES'].mean():,.2f} "
                f"(Min: {output_df['VP_FLUX_DISTRIBUABLES'].min():,.2f}, "
                f"Max: {output_df['VP_FLUX_DISTRIBUABLES'].max():,.2f})")
    for account_id in sorted(output_df['ID_COMPTE'].unique()):
        account_data = output_df[output_df['ID_COMPTE'] == account_id]
        logger.info(f"  Account {account_id}: Mean = {account_data['VP_FLUX_DISTRIBUABLES'].mean():,.2f}, Scenarios = {len(account_data)}")

    logger.info("=" * 80)
    logger.info("DETAILED RESULTS")
    logger.info("=" * 80)
    logger.info(f"Total detailed records generated: {len(detailed_df)}")
    logger.info(f"First few rows of detailed results:\n{detailed_df.head(10)}")

    return output_df, detailed_df, internal_df


def check_cuda_environment():
    """Comprehensive CUDA environment check using logging."""
    logger.info("=" * 80)
    logger.info("CUDA ENVIRONMENT DIAGNOSTICS")
    logger.info("=" * 80)
    logger.info(f"Numba version: {numba.__version__}")
    logger.info(f"CUDA Available via Numba: {cuda.is_available()}")

    if cuda.is_available():
        try:
            logger.info(f"Number of GPUs: {len(cuda.gpus)}")
            for i, gpu in enumerate(cuda.gpus):
                logger.info(f"  GPU {i}: {gpu.name.decode()}")
            from numba.cuda.cudadrv.libs import test
            test()
            logger.info("✓ CUDA libraries detected by Numba.")
        except Exception as e:
            logger.error(f"Error accessing GPU info or libraries: {e}", exc_info=True)
    else:
        logger.warning("CUDA not available. GPU acceleration will not be possible.")
    logger.info("=" * 80)


if __name__ == "__main__":
    # --- Main Execution Block ---

    # Configure the root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    check_cuda_environment()

    if not cuda.is_available():
        logger.critical("CUDA is not available. Please install CUDA and ensure your GPU supports it. Exiting.")
        exit(1)

    if not _CUDA_INITIALIZED:
        logger.warning("CUDA context was not initialized on module load. Manual re-initialization may be required.")
        # Depending on the error, you might attempt a manual init here or just exit.

    data_path = HERE.joinpath("data_in")

    results, detailed_results, internal_scenario_results = gpu_acfc_algorithm_complete(
        data_path=data_path,
        nb_accounts=1,
        nb_scenarios=2,
        nb_years=100,
        nb_sc_int=2,
        nb_an_projection_int=100,
        choc_capital=0.35,
        hurdle_rt=0.10,
        log_account_id=1,
        log_scenario=1,
        log_max_years=10,
        log_internal_scenario=1
    )

    logger.info(f"Final Summary Results:\n{results}")
    results.to_csv('test/gpu_results_complete.csv', index=False)

    logger.info("Saving detailed year-by-year results...")
    detailed_results.to_csv('test/gpu_results_detailed.csv', index=False)
    logger.info(f"✓ Detailed results saved to 'test/gpu_results_detailed.csv' ({len(detailed_results)} rows)")

    if len(internal_scenario_results) > 0:
        logger.info("Saving internal scenario detailed results...")
        internal_scenario_results.to_csv('test/gpu_results_internal_scenario.csv', index=False)
        logger.info(f"✓ Internal scenario results saved to 'test/gpu_results_internal_scenario.csv' ({len(internal_scenario_results)} rows)")
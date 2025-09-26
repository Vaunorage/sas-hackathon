import pandas as pd
import numpy as np
from numba import cuda, jit
import numba
from typing import Dict, Tuple, List
import warnings
import logging
import math

from paths import HERE

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


def load_input_data(data_path: str = ".") -> Dict:
    """Load all input data files"""
    try:
        population = pd.read_csv(f"{data_path}/population.csv").head(4)
        rendement = pd.read_csv(f"{data_path}/rendement.csv")
        tx_deces = pd.read_csv(f"{data_path}/tx_deces.csv")
        tx_interet = pd.read_csv(f"{data_path}/tx_interet.csv")
        tx_interet_int = pd.read_csv(f"{data_path}/tx_interet_int.csv")
        tx_retrait = pd.read_csv(f"{data_path}/tx_retrait.csv")

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

    # Mortality lookup table (age -> mortality rate)
    mortality_array = np.zeros(max_age + 1, dtype=np.float64)
    for _, row in data['tx_deces'].iterrows():
        age = int(row['AGE'])
        if age <= max_age:
            mortality_array[age] = float(row['QX'])

    # Lapse lookup table (year -> lapse rate)
    lapse_array = np.zeros(max_year + 1, dtype=np.float64)
    for _, row in data['tx_retrait'].iterrows():
        year = int(row['an_proj'])
        if year <= max_year:
            lapse_array[year] = float(row['WX'])

    # Discount rates external (year -> discount rate)
    discount_ext_array = np.ones(max_year + 1, dtype=np.float64)
    for _, row in data['tx_interet'].iterrows():
        year = int(row['an_proj'])
        if year <= max_year:
            discount_ext_array[year] = float(row['TX_ACTU'])

    # Discount rates internal (year -> discount rate)
    discount_int_array = np.ones(max_year + 1, dtype=np.float64)
    for _, row in data['tx_interet_int'].iterrows():
        year = int(row['an_eval'])
        if year <= max_year:
            discount_int_array[year] = float(row['TX_ACTU_INT'])

    # Returns lookup table (year, scenario, type -> return rate)
    # We'll create separate arrays for EXTERNE and INTERNE
    returns_ext_array = np.zeros((max_year + 1, max_scenarios + 1), dtype=np.float64)
    returns_int_array = np.zeros((max_year + 1, max_scenarios + 1), dtype=np.float64)

    for _, row in data['rendement'].iterrows():
        year = int(row['an_proj'])
        scenario = int(row['scn_proj'])
        if year <= max_year and scenario <= max_scenarios:
            if row['TYPE'] == 'EXTERNE':
            # Store year 0 internal result
            result_idx = combination_idx * 50 + year
            if result_idx < results.shape[0]:
                results[result_idx, 0] = states[combination_idx, STATE_ACCOUNT_ID]
                results[result_idx, 1] = states[combination_idx, STATE_SCENARIO]
                results[result_idx, 2] = year
                results[result_idx, 3] = states[combination_idx, STATE_AGE]
                results[result_idx, 4] = new_MT_VM
                results[result_idx, 5] = initial_data[account_idx, DATA_MT_GAR_DECES]
                results[result_idx, 6] = 1.0
                results[result_idx, 7] = 0.0  # FLUX_NET
                results[result_idx, 8] = 0.0  # VP_FLUX_NET
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


def prepare_gpu_data(data: Dict, nb_accounts: int, nb_scenarios: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepare data for GPU processing"""

    # Create states array: [account_scenario_combinations, state_features]
    total_combinations = min(nb_accounts, len(data['population'])) * nb_scenarios
    states = np.zeros((total_combinations, STATE_SIZE), dtype=np.float64)

    # Create initial data array: [accounts, data_features]
    initial_data = np.zeros((min(nb_accounts, len(data['population'])), DATA_SIZE), dtype=np.float64)

    # Account ID mapping
    account_ids = np.zeros(min(nb_accounts, len(data['population'])), dtype=np.float64)

    combination_idx = 0
    for account_idx in range(min(nb_accounts, len(data['population']))):
        account_data = data['population'].iloc[account_idx]

        # Store account ID
        account_ids[account_idx] = float(account_data['ID_COMPTE'])

        # Store initial data
        initial_data[account_idx, DATA_MT_VM] = float(account_data['MT_VM'])
        initial_data[account_idx, DATA_MT_GAR_DECES] = float(account_data['MT_GAR_DECES'])
        initial_data[account_idx, DATA_AGE_DEB] = int(account_data['age_deb'])
        initial_data[account_idx, DATA_TX_COMM_VENTE] = float(account_data.get('TX_COMM_VENTE', 0.0))
        initial_data[account_idx, DATA_FRAIS_ACQUI] = float(account_data['FRAIS_ACQUI'])
        initial_data[account_idx, DATA_PC_REVENU_FDS] = float(account_data['PC_REVENU_FDS'])
        initial_data[account_idx, DATA_PC_HONORAIRES_GEST] = float(account_data['PC_HONORAIRES_GEST'])
        initial_data[account_idx, DATA_TX_COMM_MAINTIEN] = float(account_data['TX_COMM_MAINTIEN'])
        initial_data[account_idx, DATA_FRAIS_ADMIN] = float(account_data['FRAIS_ADMIN'])
        initial_data[account_idx, DATA_FREQ_RESET_DECES] = float(account_data['FREQ_RESET_DECES'])
        initial_data[account_idx, DATA_MAX_RESET_DECES] = float(account_data['MAX_RESET_DECES'])

        # Initialize states for all scenarios of this account
        for scenario in range(1, nb_scenarios + 1):
            states[combination_idx, STATE_ACCOUNT_ID] = float(account_data['ID_COMPTE'])
            states[combination_idx, STATE_SCENARIO] = float(scenario)
            states[combination_idx, STATE_ACCOUNT_IDX] = float(account_idx)
            states[combination_idx, STATE_AGE_DEB] = float(account_data['age_deb'])
            states[combination_idx, STATE_MT_VM_PROJ] = 0.0
            states[combination_idx, STATE_MT_GAR_DECES_PROJ] = 0.0
            states[combination_idx, STATE_TX_SURVIE] = 0.0
            states[combination_idx, STATE_AGE] = float(account_data['age_deb'])
            states[combination_idx, STATE_IS_TERMINATED] = 0.0
            combination_idx += 1

    return states, initial_data, account_ids


@cuda.jit
def gpu_calculate_year_zero_external(states, initial_data, results, year, combination_idx):
    """GPU kernel for year 0 external calculations"""

    account_idx = int(states[combination_idx, STATE_ACCOUNT_IDX])

    # Initialize variables properly for year 0
    MT_VM_PROJ = initial_data[account_idx, DATA_MT_VM]
    MT_GAR_DECES_PROJ = initial_data[account_idx, DATA_MT_GAR_DECES]
    TX_SURVIE = 1.0
    AGE = initial_data[account_idx, DATA_AGE_DEB]

    COMMISSIONS = -initial_data[account_idx, DATA_TX_COMM_VENTE] * MT_VM_PROJ
    VP_COMMISSIONS = COMMISSIONS

    FRAIS_GEN = -initial_data[account_idx, DATA_FRAIS_ACQUI]
    VP_FRAIS_GEN = FRAIS_GEN

    FLUX_NET = FRAIS_GEN + COMMISSIONS
    VP_FLUX_NET = FLUX_NET

    # Update state
    states[combination_idx, STATE_MT_VM_PROJ] = MT_VM_PROJ
    states[combination_idx, STATE_MT_GAR_DECES_PROJ] = MT_GAR_DECES_PROJ
    states[combination_idx, STATE_TX_SURVIE] = TX_SURVIE
    states[combination_idx, STATE_AGE] = AGE

    # Store results (we'll use a more complex structure in practice)
    result_idx = combination_idx * 50 + year  # Assuming max 50 years
    if result_idx < results.shape[0]:
        results[result_idx, 0] = states[combination_idx, STATE_ACCOUNT_ID]  # account_id
        results[result_idx, 1] = states[combination_idx, STATE_SCENARIO]  # scenario
        results[result_idx, 2] = year  # year
        results[result_idx, 3] = AGE  # age
        results[result_idx, 4] = MT_VM_PROJ  # mt_vm_proj
        results[result_idx, 5] = MT_GAR_DECES_PROJ  # mt_gar_deces_proj
        results[result_idx, 6] = TX_SURVIE  # tx_survie
        results[result_idx, 7] = FLUX_NET  # flux_net
        results[result_idx, 8] = VP_FLUX_NET  # vp_flux_net


@cuda.jit
def gpu_calculate_year_transition(states, initial_data, lookups_mortality, lookups_lapse,
                                  lookups_discount_ext, lookups_discount_int, lookups_returns_ext,
                                  lookups_returns_int, results, year, projection_type, fund_shock, start_year):
    """GPU kernel for year transition calculations"""

    # Get thread index
    combination_idx = cuda.grid(1)

    if combination_idx >= states.shape[0]:
        return

    # Check termination conditions
    if states[combination_idx, STATE_IS_TERMINATED] > 0:
        return

    if states[combination_idx, STATE_TX_SURVIE] == 0 or states[combination_idx, STATE_MT_VM_PROJ] == 0:
        states[combination_idx, STATE_IS_TERMINATED] = 1.0
        return

    account_idx = int(states[combination_idx, STATE_ACCOUNT_IDX])
    scenario = int(states[combination_idx, STATE_SCENARIO])

    # Handle year 0 special cases
    if year == 0:
        if projection_type == 0:  # EXTERNE
            gpu_calculate_year_zero_external(states, initial_data, results, year, combination_idx)
            return
        else:  # INTERNE
            if fund_shock > 0:
                new_MT_VM = initial_data[account_idx, DATA_MT_VM] * (1 - fund_shock)
            else:
                new_MT_VM = initial_data[account_idx, DATA_MT_VM]

            states[combination_idx, STATE_MT_VM_PROJ] = new_MT_VM
            states[combination_idx, STATE_MT_GAR_DECES_PROJ] = initial_data[account_idx, DATA_MT_GAR_DECES]
            states[combination_idx, STATE_TX_SURVIE] = 1.0  # Default TX_SURVIE_DEB
            states[combination_idx, STATE_AGE] = initial_data[account_idx, DATA_AGE_DEB] + start_year

            # Store year 0 internal result
            result_idx = combination_idx * 50 + year
            if result_idx < results.shape[0]:
                results[result_idx, 0] = states[combination_idx, STATE_ACCOUNT_ID]
                results[result_idx, 1] = states[combination_idx, STATE_SCENARIO]
                results[result_idx, 2] = year
                results[result_idx, 3] = states[combination_idx, STATE_AGE]
                results[result_idx, 4] = new_MT_VM
                results[result_idx, 5] = initial_data[account_idx, DATA_MT_GAR_DECES]
                results[result_idx, 6] = 1.0
                results[result_idx, 7] = 0.0  # FLUX_NET
                results[result_idx, 8] = 0.0  # VP_FLUX_NET
            return

    # Regular year calculations
    if projection_type == 1:  # INTERNE
        new_age = int(initial_data[account_idx, DATA_AGE_DEB] + start_year + year)
        an_proj = start_year + year
    else:  # EXTERNE
        new_age = int(initial_data[account_idx, DATA_AGE_DEB] + year)
        an_proj = year

    # Bounds checking
    if new_age >= lookups_mortality.shape[0] or an_proj >= lookups_returns_ext.shape[0]:
        states[combination_idx, STATE_IS_TERMINATED] = 1.0
        return

    # Fund value projection
    MT_VM_DEB = states[combination_idx, STATE_MT_VM_PROJ]

    # Get return rate
    if projection_type == 0:  # EXTERNE
        if scenario < lookups_returns_ext.shape[1]:
            RENDEMENT_rate = lookups_returns_ext[an_proj, scenario]
        else:
            RENDEMENT_rate = 0.0
    else:  # INTERNE
        if scenario < lookups_returns_int.shape[1]:
            RENDEMENT_rate = lookups_returns_int[an_proj, scenario]
        else:
            RENDEMENT_rate = 0.0

    RENDEMENT = MT_VM_DEB * RENDEMENT_rate
    FRAIS = -(MT_VM_DEB + RENDEMENT / 2) * initial_data[account_idx, DATA_PC_REVENU_FDS]
    new_MT_VM_PROJ = max(0.0, states[combination_idx, STATE_MT_VM_PROJ] + RENDEMENT + FRAIS)

    # Death benefit guarantee reset logic
    new_MT_GAR_DECES_PROJ = states[combination_idx, STATE_MT_GAR_DECES_PROJ]
    if (initial_data[account_idx, DATA_FREQ_RESET_DECES] == 1 and
            new_age <= initial_data[account_idx, DATA_MAX_RESET_DECES]):
        new_MT_GAR_DECES_PROJ = max(states[combination_idx, STATE_MT_GAR_DECES_PROJ], new_MT_VM_PROJ)

    # Survival probability calculation
    QX = lookups_mortality[min(new_age, lookups_mortality.shape[0] - 1)]
    WX = lookups_lapse[min(an_proj, lookups_lapse.shape[0] - 1)]

    TX_SURVIE_DEB = states[combination_idx, STATE_TX_SURVIE]
    new_TX_SURVIE = TX_SURVIE_DEB * (1 - QX) * (1 - WX)

    # Cash flow calculations
    REVENUS = -FRAIS * TX_SURVIE_DEB
    FRAIS_GEST = -(MT_VM_DEB + RENDEMENT / 2) * initial_data[account_idx, DATA_PC_HONORAIRES_GEST] * TX_SURVIE_DEB
    COMMISSIONS = -(MT_VM_DEB + RENDEMENT / 2) * initial_data[account_idx, DATA_TX_COMM_MAINTIEN] * TX_SURVIE_DEB
    FRAIS_GEN = -initial_data[account_idx, DATA_FRAIS_ADMIN] * TX_SURVIE_DEB
    PMT_GARANTIE = -max(0.0, new_MT_GAR_DECES_PROJ - new_MT_VM_PROJ) * QX * TX_SURVIE_DEB

    FLUX_NET = REVENUS + FRAIS_GEST + COMMISSIONS + FRAIS_GEN + PMT_GARANTIE

    # Present value calculations
    TX_ACTU = lookups_discount_ext[min(an_proj, lookups_discount_ext.shape[0] - 1)]
    VP_FLUX_NET = FLUX_NET * TX_ACTU

    # Internal scenario adjustment
    if projection_type == 1 and start_year > 0:  # INTERNE
        TX_ACTU_INT = lookups_discount_int[min(start_year, lookups_discount_int.shape[0] - 1)]
        if TX_ACTU_INT != 0:
            VP_FLUX_NET = VP_FLUX_NET / TX_ACTU_INT

    # Internal scenario adjustment
    if projection_type == 1 and start_year > 0:  # INTERNE
        TX_ACTU_INT = lookups_discount_int[min(start_year, lookups_discount_int.shape[0] - 1)]
        if TX_ACTU_INT != 0:
            VP_FLUX_NET = VP_FLUX_NET / TX_ACTU_INT

    # Update state
    states[combination_idx, STATE_MT_VM_PROJ] = new_MT_VM_PROJ
    states[combination_idx, STATE_MT_GAR_DECES_PROJ] = new_MT_GAR_DECES_PROJ
    states[combination_idx, STATE_TX_SURVIE] = new_TX_SURVIE
    states[combination_idx, STATE_AGE] = new_age
    states[combination_idx, STATE_IS_TERMINATED] = 1.0 if (new_TX_SURVIE == 0 or new_MT_VM_PROJ == 0) else 0.0

    # Store results
    result_idx = combination_idx * 50 + year  # Assuming max 50 years
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


@cuda.jit
def gpu_run_internal_projection(account_data, starting_fund_value, starting_death_benefit,
                                starting_survival_prob, starting_age, start_year,
                                lookups_mortality, lookups_lapse, lookups_discount_ext,
                                lookups_discount_int, lookups_returns_int,
                                internal_scenario, nb_an_projection_int, fund_shock):
    """
    GPU kernel to run a single internal projection forward from a given starting state
    Returns the sum of VP_FLUX_NET for this internal scenario
    """

    # Initialize internal projection state
    MT_VM_PROJ = starting_fund_value * (1 - fund_shock) if fund_shock > 0 else starting_fund_value
    MT_GAR_DECES_PROJ = starting_death_benefit
    TX_SURVIE = starting_survival_prob
    AGE = starting_age

    total_vp = 0.0

    # Run internal projection forward
    for year in range(nb_an_projection_int + 1):
        if TX_SURVIE == 0 or MT_VM_PROJ == 0:
            break

        an_proj = start_year + year
        new_age = starting_age + year

        # Bounds checking
        if new_age >= lookups_mortality.shape[0] or an_proj >= lookups_returns_int.shape[0]:
            break

        # Year 0 handling for internal projection
        if year == 0:
            # Store initial state but no cash flows
            continue

        # Fund value projection
        MT_VM_DEB = MT_VM_PROJ

        # Get return rate for internal scenario
        if internal_scenario < lookups_returns_int.shape[1] and an_proj < lookups_returns_int.shape[0]:
            RENDEMENT_rate = lookups_returns_int[an_proj, internal_scenario]
        else:
            RENDEMENT_rate = 0.0

        RENDEMENT = MT_VM_DEB * RENDEMENT_rate
        FRAIS = -(MT_VM_DEB + RENDEMENT / 2) * account_data[DATA_PC_REVENU_FDS]
        new_MT_VM_PROJ = max(0.0, MT_VM_PROJ + RENDEMENT + FRAIS)

        # Death benefit guarantee reset logic
        new_MT_GAR_DECES_PROJ = MT_GAR_DECES_PROJ
        if (account_data[DATA_FREQ_RESET_DECES] == 1 and
                new_age <= account_data[DATA_MAX_RESET_DECES]):
            new_MT_GAR_DECES_PROJ = max(MT_GAR_DECES_PROJ, new_MT_VM_PROJ)

        # Survival probability calculation
        QX = lookups_mortality[min(new_age, lookups_mortality.shape[0] - 1)]
        WX = lookups_lapse[min(an_proj, lookups_lapse.shape[0] - 1)]

        TX_SURVIE_DEB = TX_SURVIE
        new_TX_SURVIE = TX_SURVIE_DEB * (1 - QX) * (1 - WX)

        # Cash flow calculations
        REVENUS = -FRAIS * TX_SURVIE_DEB
        FRAIS_GEST = -(MT_VM_DEB + RENDEMENT / 2) * account_data[DATA_PC_HONORAIRES_GEST] * TX_SURVIE_DEB
        COMMISSIONS = -(MT_VM_DEB + RENDEMENT / 2) * account_data[DATA_TX_COMM_MAINTIEN] * TX_SURVIE_DEB
        FRAIS_GEN = -account_data[DATA_FRAIS_ADMIN] * TX_SURVIE_DEB
        PMT_GARANTIE = -max(0.0, new_MT_GAR_DECES_PROJ - new_MT_VM_PROJ) * QX * TX_SURVIE_DEB

        FLUX_NET = REVENUS + FRAIS_GEST + COMMISSIONS + FRAIS_GEN + PMT_GARANTIE

        # Present value calculations
        TX_ACTU = lookups_discount_ext[min(an_proj, lookups_discount_ext.shape[0] - 1)]
        VP_FLUX_NET = FLUX_NET * TX_ACTU

        # Internal scenario adjustment
        if start_year > 0:
            TX_ACTU_INT = lookups_discount_int[min(start_year, lookups_discount_int.shape[0] - 1)]
            if TX_ACTU_INT != 0:
                VP_FLUX_NET = VP_FLUX_NET / TX_ACTU_INT

        total_vp += VP_FLUX_NET

        # Update state for next year
        MT_VM_PROJ = new_MT_VM_PROJ
        MT_GAR_DECES_PROJ = new_MT_GAR_DECES_PROJ
        TX_SURVIE = new_TX_SURVIE
        AGE = new_age

    return total_vp


@cuda.jit
def gpu_calculate_internal_scenarios(external_results_structured, initial_data, account_mapping,
                                     lookups_mortality, lookups_lapse, lookups_discount_ext,
                                     lookups_discount_int, lookups_returns_int,
                                     internal_results, nb_sc_int, nb_an_projection_int,
                                     calculation_type, choc_capital):
    """
    GPU kernel for calculating internal scenarios for each external result
    calculation_type: 0 = RESERVE, 1 = CAPITAL
    """

    # Get thread indices
    ext_result_idx = cuda.blockIdx.x  # Which external result we're processing
    year_idx = cuda.threadIdx.x  # Which year of that external result

    if (ext_result_idx >= external_results_structured.shape[0] or
            year_idx >= external_results_structured.shape[1]):
        return

    # Get external result data for this year
    ext_data = external_results_structured[ext_result_idx, year_idx]
    account_id = ext_data[0]
    scenario = ext_data[1]
    year = int(ext_data[2])
    age = int(ext_data[3])
    fund_value = ext_data[4]
    death_benefit = ext_data[5]
    survival_prob = ext_data[6]

    # Skip if this is not a valid result or year 0
    if account_id == 0 or year == 0 or survival_prob == 0 or fund_value == 0:
        internal_results[ext_result_idx, year_idx] = 0.0
        return

    # Find account data
    account_data_idx = -1
    for i in range(account_mapping.shape[0]):
        if account_mapping[i] == account_id:
            account_data_idx = i
            break

    if account_data_idx == -1:
        internal_results[ext_result_idx, year_idx] = 0.0
        return

    # Apply shock for capital calculations
    fund_shock = choc_capital if calculation_type == 1 else 0.0

    # Run multiple internal scenarios and sum them
    internal_scenario_sum = 0.0
    valid_scenarios = 0

    for internal_scenario in range(1, min(nb_sc_int + 1, lookups_returns_int.shape[1])):
        scenario_result = gpu_run_internal_projection(
            initial_data[account_data_idx],  # account data
            fund_value,  # starting fund value
            death_benefit,  # starting death benefit
            survival_prob,  # starting survival probability
            age,  # starting age
            year,  # start year
            lookups_mortality, lookups_lapse, lookups_discount_ext,
            lookups_discount_int, lookups_returns_int,
            internal_scenario, nb_an_projection_int, fund_shock
        )

        internal_scenario_sum += scenario_result
        valid_scenarios += 1

    # Calculate mean across internal scenarios
    if valid_scenarios > 0:
        result_value = internal_scenario_sum / valid_scenarios
    else:
        result_value = 0.0

    internal_results[ext_result_idx, year_idx] = result_value


@cuda.jit
def gpu_internal_calculations(external_results, initial_data_array, lookups_mortality, lookups_lapse,
                              lookups_discount_ext, lookups_discount_int, lookups_returns_int,
                              internal_results, nb_sc_int, nb_an_projection_int,
                              calculation_type, choc_capital):
    """GPU kernel for internal scenario calculations"""

    # Get thread indices
    ext_idx = cuda.blockIdx.x
    year = cuda.threadIdx.x

    if ext_idx >= external_results.shape[0] or year >= external_results.shape[1]:
        return

    # Get external result data
    account_id = external_results[ext_idx, year, 0]
    scenario = external_results[ext_idx, year, 1]
    ext_year = external_results[ext_idx, year, 2]
    fund_value = external_results[ext_idx, year, 4]
    death_benefit = external_results[ext_idx, year, 5]
    survival_prob = external_results[ext_idx, year, 6]

    if ext_year == 0 or survival_prob == 0 or fund_value == 0:
        return

    # Apply shock for capital calculations
    fund_shock = choc_capital if calculation_type == 1 else 0.0  # 1 for CAPITAL, 0 for RESERVE

    # Initialize shared memory for internal scenario sums
    internal_sums = cuda.shared.array(1024, dtype=numba.float64)  # Assuming max 1024 internal scenarios

    internal_scenario_sum = 0.0

    # Run internal scenarios (simplified - in practice you'd need more complex logic)
    for internal_scenario in range(1, min(nb_sc_int + 1, 1024)):
        # Create modified initial data for this internal projection
        shocked_fund_value = fund_value * (1 - fund_shock) if fund_shock > 0 else fund_value

        # Simplified internal projection calculation
        # In practice, this would run a full projection like the external one
        internal_value = shocked_fund_value * 0.05  # Simplified calculation

        internal_scenario_sum += internal_value

    # Calculate mean across internal scenarios
    if nb_sc_int > 0:
        result_value = internal_scenario_sum / nb_sc_int
    else:
        result_value = 0.0

    # Store result
    internal_results[ext_idx, year] = result_value


def run_gpu_projection(states, initial_data, lookups, nb_years: int, projection_type: str,
                       fund_shock: float = 0.0, start_year: int = 0) -> np.ndarray:
    """Run projection on GPU"""

    # Convert projection_type to numeric
    proj_type_num = 0 if projection_type == "EXTERNE" else 1

    # Allocate results array
    max_results = states.shape[0] * (nb_years + 1)
    results = np.zeros((max_results, 9), dtype=np.float64)

    # Copy data to GPU
    d_states = cuda.to_device(states)
    d_initial_data = cuda.to_device(initial_data)
    d_results = cuda.to_device(results)

    # Copy lookup tables to GPU
    d_mortality = cuda.to_device(lookups['mortality'])
    d_lapse = cuda.to_device(lookups['lapse'])
    d_discount_ext = cuda.to_device(lookups['discount_ext'])
    d_discount_int = cuda.to_device(lookups['discount_int'])
    d_returns_ext = cuda.to_device(lookups['returns_ext'])
    d_returns_int = cuda.to_device(lookups['returns_int'])

    # Configure GPU grid
    threads_per_block = 256
    blocks_per_grid = (states.shape[0] + threads_per_block - 1) // threads_per_block

    # Run projection for each year
    for year in range(nb_years + 1):
        gpu_calculate_year_transition[blocks_per_grid, threads_per_block](
            d_states, d_initial_data, d_mortality, d_lapse, d_discount_ext, d_discount_int,
            d_returns_ext, d_returns_int, d_results, year, proj_type_num,
            fund_shock, start_year
        )
        cuda.synchronize()

    # Copy results back to CPU
    results = d_results.copy_to_host()
    states = d_states.copy_to_host()

    return results, states


def run_gpu_internal_calculations(external_results, initial_data, lookups, nb_sc_int: int,
                                  nb_an_projection_int: int, calculation_type: str,
                                  choc_capital: float) -> np.ndarray:
    """Run internal calculations on GPU"""

    # Convert calculation_type to numeric (0 for RESERVE, 1 for CAPITAL)
    calc_type_num = 1 if calculation_type == "CAPITAL" else 0

    # Reshape external results for GPU processing
    unique_combinations = np.unique(external_results[:, [0, 1]], axis=0)
    max_years = int(np.max(external_results[:, 2])) + 1

    # Create structured array for external results
    structured_ext_results = np.zeros((len(unique_combinations), max_years, 9), dtype=np.float64)

    for i, (account_id, scenario) in enumerate(unique_combinations):
        mask = (external_results[:, 0] == account_id) & (external_results[:, 1] == scenario)
        account_scenario_data = external_results[mask]

        for row in account_scenario_data:
            year = int(row[2])
            if year < max_years:
                structured_ext_results[i, year] = row

    # Allocate results array
    internal_results = np.zeros((len(unique_combinations), max_years), dtype=np.float64)

    # Copy data to GPU
    d_ext_results = cuda.to_device(structured_ext_results)
    d_initial_data = cuda.to_device(initial_data)
    d_internal_results = cuda.to_device(internal_results)

    # Copy lookup tables to GPU
    d_mortality = cuda.to_device(lookups['mortality'])
    d_lapse = cuda.to_device(lookups['lapse'])
    d_discount_ext = cuda.to_device(lookups['discount_ext'])
    d_discount_int = cuda.to_device(lookups['discount_int'])
    d_returns_int = cuda.to_device(lookups['returns_int'])

    # Configure GPU grid - one block per external result, threads per years
    blocks_per_grid = len(unique_combinations)
    threads_per_block = min(max_years, 256)

    # Run internal calculations
    gpu_internal_calculations[blocks_per_grid, threads_per_block](
        d_ext_results, d_initial_data, d_mortality, d_lapse, d_discount_ext, d_discount_int,
        d_returns_int, d_internal_results, nb_sc_int, nb_an_projection_int,
        calc_type_num, choc_capital
    )
    cuda.synchronize()

    # Copy results back to CPU
    internal_results = d_internal_results.copy_to_host()

    return internal_results, unique_combinations


def gpu_acfc_algorithm(data_path: str = ".", nb_accounts: int = 4, nb_scenarios: int = 10,
                       nb_years: int = 10, nb_sc_int: int = 10, nb_an_projection_int: int = 10,
                       choc_capital: float = 0.35, hurdle_rt: float = 0.10) -> pd.DataFrame:
    """
    GPU-Accelerated ACFC Algorithm using Numba CUDA
    """

    print("Phase 1: Loading input data...")
    data = load_input_data(data_path)

    print("Phase 2: Creating GPU lookup tables...")
    lookups = create_gpu_lookup_tables(data)

    print("Phase 3: Preparing GPU data...")
    states, initial_data, account_ids = prepare_gpu_data(data, nb_accounts, nb_scenarios)

    print("Phase 4: Running GPU external projections...")
    external_results, final_states = run_gpu_projection(
        states, initial_data, lookups, nb_years, 'EXTERNE'
    )

    print("Phase 5: Running GPU internal calculations...")
    # Filter valid external results
    valid_external_results = external_results[external_results[:, 0] != 0]

    # Run reserve calculations on GPU
    print("  - Running reserve calculations...")
    reserve_results, combinations = run_gpu_internal_calculations(
        valid_external_results, initial_data, lookups, nb_sc_int,
        nb_an_projection_int, "RESERVE", choc_capital
    )

    # Run capital calculations on GPU
    print("  - Running capital calculations...")
    capital_raw_results, _ = run_gpu_internal_calculations(
        valid_external_results, initial_data, lookups, nb_sc_int,
        nb_an_projection_int, "CAPITAL", choc_capital
    )

    print("Phase 6: Calculating distributable flows...")
    final_results = []

    for i, (account_id, scenario) in enumerate(combinations):
        print(f"Processing account {account_id} scenario {scenario}...")

        # Get external projection for this account-scenario
        mask = (valid_external_results[:, 0] == account_id) & (valid_external_results[:, 1] == scenario)
        account_scenario_results = valid_external_results[mask]

        if len(account_scenario_results) == 0:
            continue

        # Calculate capital as difference from reserves
        reserve_by_year = {int(year): reserve_results[i, int(year)] for year in range(reserve_results.shape[1])}
        capital_raw_by_year = {int(year): capital_raw_results[i, int(year)] for year in
                               range(capital_raw_results.shape[1])}
        capital_by_year = {year: capital_raw_by_year[year] - reserve_by_year.get(year, 0.0)
                           for year in capital_raw_by_year}

        # Calculate distributable cash flows
        distributable_pvs = []
        prev_reserve = 0.0
        prev_capital = 0.0

        for ext_data in account_scenario_results:
            year = int(ext_data[2])
            external_cf = ext_data[7]  # FLUX_NET

            current_reserve = reserve_by_year.get(year, 0.0)
            current_capital = capital_by_year.get(year, 0.0)

            if year == 0:
                profit = external_cf + current_reserve
                distributable = profit + current_capital
            else:
                profit = external_cf + (current_reserve - prev_reserve)
                distributable = profit + (current_capital - prev_capital)

            # Present value at hurdle rate
            if year > 0:
                pv_distributable = distributable / ((1 + hurdle_rt) ** year)
            else:
                pv_distributable = distributable

            distributable_pvs.append(pv_distributable)

            prev_reserve = current_reserve
            prev_capital = current_capital

        # Sum across all years
        total_pv_distributable = sum(distributable_pvs)

        final_results.append({
            'ID_COMPTE': int(account_id),
            'scn_eval': int(scenario),
            'VP_FLUX_DISTRIBUABLES': total_pv_distributable
        })

    print("Phase 7: Converting to DataFrame...")
    output_df = pd.DataFrame(final_results)

    print(f"GPU algorithm finished. Generated {len(output_df)} results.")
    return output_df


# Example usage
if __name__ == "__main__":
    # Check if CUDA is available
    if not cuda.is_available():
        print("CUDA is not available. Please install CUDA and ensure your GPU supports it.")
        exit(1)

    print(f"CUDA devices available: {cuda.gpus}")

    results = gpu_acfc_algorithm(
        data_path=HERE.joinpath("data_in"),
        nb_accounts=4,
        nb_scenarios=10,
        nb_years=10,
        nb_sc_int=10,
        nb_an_projection_int=10,
        choc_capital=0.35,
        hurdle_rt=0.10
    )

    print("\nFinal Results:")
    print(results)
    results.to_csv(HERE.joinpath('test/gpu_results.csv'))
    print(f"\nMean VP_FLUX_DISTRIBUABLES: {results['VP_FLUX_DISTRIBUABLES'].mean():.2f}")
    print(f"Range: {results['VP_FLUX_DISTRIBUABLES'].min():.2f} to {results['VP_FLUX_DISTRIBUABLES'].max():.2f}")
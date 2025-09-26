import pandas as pd
import numpy as np
from numba import cuda, jit
import numba
from typing import Dict, Tuple, List
import warnings
import logging
import math

# For testing - replace with your actual paths import
# from paths import HERE

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
            states[combination_idx, STATE_MT_VM_PROJ] = float(account_data['MT_VM'])  # Initialize with starting value
            states[combination_idx, STATE_MT_GAR_DECES_PROJ] = float(account_data['MT_GAR_DECES'])  # Initialize
            states[combination_idx, STATE_TX_SURVIE] = 1.0  # Initialize with 1.0
            states[combination_idx, STATE_AGE] = float(account_data['age_deb'])
            states[combination_idx, STATE_IS_TERMINATED] = 0.0
            combination_idx += 1

    return states, initial_data, account_ids


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

    account_idx = int(states[combination_idx, STATE_ACCOUNT_IDX])
    scenario = int(states[combination_idx, STATE_SCENARIO])

    # Handle year 0 special cases
    if year == 0:
        # For year 0, just store initial values
        MT_VM_PROJ = initial_data[account_idx, DATA_MT_VM]
        MT_GAR_DECES_PROJ = initial_data[account_idx, DATA_MT_GAR_DECES]
        TX_SURVIE = 1.0
        AGE = initial_data[account_idx, DATA_AGE_DEB]

        if projection_type == 0:  # EXTERNE
            COMMISSIONS = -initial_data[account_idx, DATA_TX_COMM_VENTE] * MT_VM_PROJ
            FRAIS_GEN = -initial_data[account_idx, DATA_FRAIS_ACQUI]
            FLUX_NET = FRAIS_GEN + COMMISSIONS
            VP_FLUX_NET = FLUX_NET
        else:  # INTERNE
            if fund_shock > 0:
                MT_VM_PROJ = MT_VM_PROJ * (1 - fund_shock)
            FLUX_NET = 0.0
            VP_FLUX_NET = 0.0

        # Update state
        states[combination_idx, STATE_MT_VM_PROJ] = MT_VM_PROJ
        states[combination_idx, STATE_MT_GAR_DECES_PROJ] = MT_GAR_DECES_PROJ
        states[combination_idx, STATE_TX_SURVIE] = TX_SURVIE
        states[combination_idx, STATE_AGE] = AGE

        # Store results
        result_idx = combination_idx * 50 + year  # Assuming max 50 years
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

    # Regular year calculations for year > 0
    if states[combination_idx, STATE_TX_SURVIE] == 0 or states[combination_idx, STATE_MT_VM_PROJ] == 0:
        states[combination_idx, STATE_IS_TERMINATED] = 1.0
        return

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
        if scenario < lookups_returns_ext.shape[1] and an_proj < lookups_returns_ext.shape[0]:
            RENDEMENT_rate = lookups_returns_ext[an_proj, scenario]
        else:
            RENDEMENT_rate = 0.0
    else:  # INTERNE
        if scenario < lookups_returns_int.shape[1] and an_proj < lookups_returns_int.shape[0]:
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

    print("Phase 5: Filtering and processing results...")
    # Filter valid external results - check that account_id is not 0
    valid_mask = external_results[:, 0] != 0
    valid_external_results = external_results[valid_mask]

    print(f"Total results: {len(external_results)}, Valid results: {len(valid_external_results)}")

    if len(valid_external_results) == 0:
        print("WARNING: No valid external results found!")
        # Create dummy results for testing
        final_results = []
        for account_idx in range(min(nb_accounts, len(data['population']))):
            account_id = data['population'].iloc[account_idx]['ID_COMPTE']
            for scenario in range(1, nb_scenarios + 1):
                final_results.append({
                    'ID_COMPTE': int(account_id),
                    'scn_eval': scenario,
                    'VP_FLUX_DISTRIBUABLES': 0.0
                })
        return pd.DataFrame(final_results)

    print("Phase 6: Simplified distributable flows calculation...")
    final_results = []

    # Group by account and scenario
    for account_idx in range(min(nb_accounts, len(data['population']))):
        account_id = data['population'].iloc[account_idx]['ID_COMPTE']

        for scenario in range(1, nb_scenarios + 1):
            # Get external projection for this account-scenario
            mask = (valid_external_results[:, 0] == account_id) & (valid_external_results[:, 1] == scenario)
            account_scenario_results = valid_external_results[mask]

            if len(account_scenario_results) == 0:
                continue

            # Simplified calculation: sum all VP_FLUX_NET
            total_vp_flux = np.sum(account_scenario_results[:, 8])

            final_results.append({
                'ID_COMPTE': int(account_id),
                'scn_eval': scenario,
                'VP_FLUX_DISTRIBUABLES': total_vp_flux
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

    # Replace with your actual data path
    data_path = "data_in"  # Update this path

    results = gpu_acfc_algorithm(
        data_path=data_path,
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

    # Save results
    results.to_csv('gpu_results.csv', index=False)
    print(f"\nMean VP_FLUX_DISTRIBUABLES: {results['VP_FLUX_DISTRIBUABLES'].mean():.2f}")
    print(f"Range: {results['VP_FLUX_DISTRIBUABLES'].min():.2f} to {results['VP_FLUX_DISTRIBUABLES'].max():.2f}")
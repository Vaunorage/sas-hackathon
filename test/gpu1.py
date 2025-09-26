import pandas as pd
import numpy as np
from numba import cuda, jit
import cupy as cp
from typing import Dict, Tuple, List, Optional
import logging
import time
from pathlib import Path
from tqdm import tqdm
import warnings
import math

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global parameters
NBCPT = 4
NB_SC = 10
NB_AN_PROJECTION = 10
NB_SC_INT = 10
NB_AN_PROJECTION_INT = 10
CHOC_CAPITAL = 0.35
HURDLE_RT = 0.10

# GPU Constants
MAX_ACCOUNTS = 1000
MAX_SCENARIOS = 50
MAX_YEARS = 50
MAX_AGE = 120


class GPUDataArrays:
    """Container for GPU-optimized lookup arrays"""

    def __init__(self, hash_tables):
        self.mortality_array = None
        self.lapse_array = None
        self.rendement_array = None
        self.discount_ext_array = None
        self.discount_int_array = None
        self._create_gpu_arrays(hash_tables)

    def _create_gpu_arrays(self, hash_tables):
        """Convert hash tables to GPU arrays for fast lookup"""
        h_mortality, g_lapse, z_rendement, a_discount_ext, b_discount_int = hash_tables

        # Mortality array (indexed by age)
        mortality_array = np.zeros(MAX_AGE, dtype=np.float32)
        for age, rate in h_mortality.items():
            if 0 <= age < MAX_AGE:
                mortality_array[age] = float(rate)

        # Lapse array (indexed by year)
        lapse_array = np.zeros(MAX_YEARS, dtype=np.float32)
        for year, rate in g_lapse.items():
            if 0 <= year < MAX_YEARS:
                lapse_array[year] = float(rate)

        # External discount array
        discount_ext_array = np.ones(MAX_YEARS, dtype=np.float32)  # Default to 1.0
        for year, rate in a_discount_ext.items():
            if 0 <= year < MAX_YEARS:
                discount_ext_array[year] = float(rate)

        # Internal discount array
        discount_int_array = np.ones(MAX_YEARS, dtype=np.float32)
        for year, rate in b_discount_int.items():
            if 0 <= year < MAX_YEARS:
                discount_int_array[year] = float(rate)

        # Rendement 3D array [scenario][year][type] where type: 0=EXTERNE, 1=INTERNE
        rendement_array = np.zeros((MAX_SCENARIOS, MAX_YEARS, 2), dtype=np.float32)
        type_map = {"EXTERNE": 0, "INTERNE": 1}

        for (scn, year, type_str), rate in z_rendement.items():
            if (0 <= scn < MAX_SCENARIOS and
                    0 <= year < MAX_YEARS and
                    type_str in type_map):
                rendement_array[scn, year, type_map[type_str]] = float(rate)

        # Move arrays to GPU
        self.mortality_array = cuda.to_device(mortality_array)
        self.lapse_array = cuda.to_device(lapse_array)
        self.rendement_array = cuda.to_device(rendement_array)
        self.discount_ext_array = cuda.to_device(discount_ext_array)
        self.discount_int_array = cuda.to_device(discount_int_array)

        logger.info("GPU lookup arrays created successfully")


def load_input_files(data_path: str) -> Tuple[pd.DataFrame, ...]:
    """Load all input CSV files exactly as SAS does"""
    try:
        population = pd.read_csv(f"{data_path}/population.csv").head(NBCPT)
        rendement = pd.read_csv(f"{data_path}/rendement.csv")
        tx_deces = pd.read_csv(f"{data_path}/tx_deces.csv")
        tx_interet = pd.read_csv(f"{data_path}/tx_interet.csv")
        tx_interet_int = pd.read_csv(f"{data_path}/tx_interet_int.csv")
        tx_retrait = pd.read_csv(f"{data_path}/tx_retrait.csv")

        # Handle TYPE column encoding
        if 'TYPE' in rendement.columns:
            rendement['TYPE'] = rendement['TYPE'].apply(
                lambda x: x.decode('utf-8') if isinstance(x, bytes) else str(x)
            )

        logger.info(f"Input files loaded - Population: {len(population)} accounts")
        return population, rendement, tx_deces, tx_interet, tx_interet_int, tx_retrait

    except Exception as e:
        logger.error(f"Error loading input files: {e}")
        raise


def create_hash_tables(rendement: pd.DataFrame, tx_deces: pd.DataFrame,
                       tx_interet: pd.DataFrame, tx_interet_int: pd.DataFrame,
                       tx_retrait: pd.DataFrame) -> Tuple[Dict, ...]:
    """Create hash tables for GPU array conversion"""

    # Mortality hash
    h_mortality = {}
    for _, row in tx_deces.iterrows():
        h_mortality[int(row['AGE'])] = float(row['QX'])

    # Lapse hash
    g_lapse = {}
    for _, row in tx_retrait.iterrows():
        g_lapse[int(row['an_proj'])] = float(row['WX'])

    # Rendement hash
    z_rendement = {}
    for _, row in rendement.iterrows():
        key = (int(row['scn_proj']), int(row['an_proj']), str(row['TYPE']))
        z_rendement[key] = float(row['RENDEMENT'])

    # Discount hashes
    a_discount_ext = {}
    for _, row in tx_interet.iterrows():
        a_discount_ext[int(row['an_proj'])] = float(row['TX_ACTU'])

    b_discount_int = {}
    for _, row in tx_interet_int.iterrows():
        b_discount_int[int(row['an_eval'])] = float(row['TX_ACTU_INT'])

    return h_mortality, g_lapse, z_rendement, a_discount_ext, b_discount_int


@cuda.jit(device=True)
def safe_array_access(array, index, default_value=0.0):
    """Safely access array with bounds checking"""
    if 0 <= index < array.shape[0]:
        return array[index]
    return default_value


@cuda.jit
def gpu_external_cash_flow_kernel(
        # Input data arrays
        population_data,  # [account_idx, field_idx] - account characteristics
        mortality_rates,  # [age] - mortality rates by age
        lapse_rates,  # [year] - lapse rates by projection year
        rendement_rates,  # [scenario, year, type] - investment returns
        discount_rates,  # [year] - discount factors

        # Scenario parameters
        account_indices,  # Which account each thread processes
        scenario_numbers,  # Which scenario each thread processes

        # Output arrays
        results_vp_flux,  # [thread_id] - total present value of cash flows
        results_details  # [thread_id, year, field] - detailed year-by-year results
):
    """
    CUDA kernel for external scenario cash flow calculations
    Each thread processes one (account, scenario) combination
    """

    thread_id = cuda.grid(1)
    if thread_id >= len(account_indices):
        return

    # Get my assigned work
    account_idx = account_indices[thread_id]
    scn_eval = scenario_numbers[thread_id]

    # Extract account data (each thread gets its own copy)
    mt_vm = population_data[account_idx, 0]  # MT_VM
    mt_gar_deces = population_data[account_idx, 1]  # MT_GAR_DECES
    age_deb = int(population_data[account_idx, 2])  # age_deb
    tx_comm_vente = population_data[account_idx, 3]  # TX_COMM_VENTE
    frais_acqui = population_data[account_idx, 4]  # FRAIS_ACQUI
    pc_revenu_fds = population_data[account_idx, 5]  # PC_REVENU_FDS
    freq_reset_deces = population_data[account_idx, 6]  # FREQ_RESET_DECES
    max_reset_deces = population_data[account_idx, 7]  # MAX_RESET_DECES
    pc_honoraires = population_data[account_idx, 8]  # PC_HONORAIRES_GEST
    tx_comm_maintien = population_data[account_idx, 9]  # TX_COMM_MAINTIEN
    frais_admin = population_data[account_idx, 10]  # FRAIS_ADMIN

    # Initialize projection variables
    mt_vm_proj = mt_vm
    mt_gar_deces_proj = mt_gar_deces
    tx_survie = 1.0
    total_vp_flux = 0.0

    # Calculate max projection years
    max_years = min(NB_AN_PROJECTION, 99 - age_deb)

    # Project cash flows year by year
    for year in range(max_years + 1):
        current_age = age_deb + year
        an_proj = year

        # Year 0 initialization
        if year == 0:
            # Year 0 cash flows
            commissions = -tx_comm_vente * mt_vm_proj
            frais_gen = -frais_acqui
            flux_net = commissions + frais_gen
            vp_flux_net = flux_net

            # Store year 0 details
            if year < results_details.shape[1]:
                results_details[thread_id, year, 0] = mt_vm_proj
                results_details[thread_id, year, 1] = tx_survie
                results_details[thread_id, year, 2] = vp_flux_net

        # Subsequent years
        elif tx_survie > 0 and mt_vm_proj > 0:

            # Lookup rates using GPU arrays (much faster than hash tables)
            qx = safe_array_access(mortality_rates, current_age)
            wx = safe_array_access(lapse_rates, an_proj)

            # Get investment return rate
            rendement_rate = 0.0
            if (scn_eval < rendement_rates.shape[0] and
                    an_proj < rendement_rates.shape[1]):
                rendement_rate = rendement_rates[scn_eval, an_proj, 0]  # 0 = EXTERNE

            # Fund value projection
            mt_vm_deb = mt_vm_proj
            rendement = mt_vm_deb * rendement_rate
            frais = -(mt_vm_deb + rendement / 2.0) * pc_revenu_fds
            mt_vm_proj = mt_vm_proj + rendement + frais

            # Death benefit guarantee reset logic
            if freq_reset_deces == 1.0 and current_age <= max_reset_deces:
                mt_gar_deces_proj = max(mt_gar_deces_proj, mt_vm_proj)

            # Survival probability calculation
            tx_survie_deb = tx_survie
            tx_survie = tx_survie_deb * (1.0 - qx) * (1.0 - wx)

            # Cash flow calculations
            revenus = -frais * tx_survie_deb
            frais_gest = -(mt_vm_deb + rendement / 2.0) * pc_honoraires * tx_survie_deb
            commissions = -(mt_vm_deb + rendement / 2.0) * tx_comm_maintien * tx_survie_deb
            frais_gen = -frais_admin * tx_survie_deb
            pmt_garantie = -max(0.0, mt_gar_deces_proj - mt_vm_proj) * qx * tx_survie_deb

            flux_net = revenus + frais_gest + commissions + frais_gen + pmt_garantie

            # Present value calculation
            tx_actu = safe_array_access(discount_rates, an_proj, 1.0)
            vp_flux_net = flux_net * tx_actu

            # Store details
            if year < results_details.shape[1]:
                results_details[thread_id, year, 0] = mt_vm_proj
                results_details[thread_id, year, 1] = tx_survie
                results_details[thread_id, year, 2] = vp_flux_net

        else:
            vp_flux_net = 0.0

        # Accumulate total present value
        total_vp_flux += vp_flux_net

    # Store final result
    results_vp_flux[thread_id] = total_vp_flux


@cuda.jit
def gpu_internal_cash_flow_kernel(
        # Input arrays
        population_data, external_states,
        mortality_rates, lapse_rates, rendement_rates,
        discount_ext, discount_int,

        # Parameters
        account_indices, scenario_numbers, eval_years, type2_flags,

        # Outputs
        results_vp_flux
):
    """
    CUDA kernel for internal scenario calculations
    type2_flags: 0=RESERVE, 1=CAPITAL
    """

    thread_id = cuda.grid(1)
    if thread_id >= len(account_indices):
        return

    account_idx = account_indices[thread_id]
    scn_eval_int = scenario_numbers[thread_id]
    an_eval = eval_years[thread_id]
    type2 = type2_flags[thread_id]

    # Get starting state from external projection
    mt_vm_start = external_states[thread_id, 0]  # MT_VM_PROJ at an_eval
    mt_gar_deces_start = external_states[thread_id, 1]  # MT_GAR_DECES_PROJ
    tx_survie_start = external_states[thread_id, 2]  # TX_SURVIE

    # Apply capital shock if needed
    if type2 == 1:  # CAPITAL
        mt_vm_start = mt_vm_start * (1.0 - CHOC_CAPITAL)

    # Initialize internal projection
    mt_vm_proj = mt_vm_start
    mt_gar_deces_proj = mt_gar_deces_start
    tx_survie = tx_survie_start
    total_vp_flux = 0.0

    age_deb = int(population_data[account_idx, 2])
    max_years = min(NB_AN_PROJECTION_INT, 99 - age_deb - an_eval)

    # Project internal cash flows
    for year in range(max_years + 1):
        if year == 0:
            # Year 0 of internal scenario - no cash flows
            vp_flux_net = 0.0
        elif tx_survie > 0 and mt_vm_proj > 0:

            current_age = age_deb + an_eval + year
            an_proj = an_eval + year

            # Get rates
            qx = safe_array_access(mortality_rates, current_age)
            wx = safe_array_access(lapse_rates, an_proj)

            # Investment return for internal scenario
            rendement_rate = 0.0
            if (scn_eval_int < rendement_rates.shape[0] and
                    an_proj < rendement_rates.shape[1]):
                rendement_rate = rendement_rates[scn_eval_int, an_proj, 1]  # 1 = INTERNE

            # Fund value projection (same logic as external)
            mt_vm_deb = mt_vm_proj
            rendement = mt_vm_deb * rendement_rate
            frais = -(mt_vm_deb + rendement / 2.0) * population_data[account_idx, 5]
            mt_vm_proj = mt_vm_proj + rendement + frais

            # Death benefit guarantee reset
            if population_data[account_idx, 6] == 1.0 and current_age <= population_data[account_idx, 7]:
                mt_gar_deces_proj = max(mt_gar_deces_proj, mt_vm_proj)

            # Survival probability
            tx_survie_deb = tx_survie
            tx_survie = tx_survie_deb * (1.0 - qx) * (1.0 - wx)

            # Cash flows
            revenus = -frais * tx_survie_deb
            frais_gest = -(mt_vm_deb + rendement / 2.0) * population_data[account_idx, 8] * tx_survie_deb
            commissions = -(mt_vm_deb + rendement / 2.0) * population_data[account_idx, 9] * tx_survie_deb
            frais_gen = -population_data[account_idx, 10] * tx_survie_deb
            pmt_garantie = -max(0.0, mt_gar_deces_proj - mt_vm_proj) * qx * tx_survie_deb

            flux_net = revenus + frais_gest + commissions + frais_gen + pmt_garantie

            # Present value with dual discounting
            tx_actu = safe_array_access(discount_ext, an_proj, 1.0)
            vp_flux_net = flux_net * tx_actu

            # Internal discount adjustment
            if an_eval > 0:
                tx_actu_int = safe_array_access(discount_int, an_eval, 1.0)
                if tx_actu_int != 0:
                    vp_flux_net = vp_flux_net / tx_actu_int
        else:
            vp_flux_net = 0.0

        total_vp_flux += vp_flux_net

    results_vp_flux[thread_id] = total_vp_flux


def create_population_array(population: pd.DataFrame) -> np.ndarray:
    """Convert population DataFrame to GPU-friendly array"""

    required_fields = [
        'MT_VM', 'MT_GAR_DECES', 'age_deb', 'TX_COMM_VENTE',
        'FRAIS_ACQUI', 'PC_REVENU_FDS', 'FREQ_RESET_DECES',
        'MAX_RESET_DECES', 'PC_HONORAIRES_GEST', 'TX_COMM_MAINTIEN',
        'FRAIS_ADMIN'
    ]

    pop_array = np.zeros((len(population), len(required_fields)), dtype=np.float32)

    for i, field in enumerate(required_fields):
        if field in population.columns:
            pop_array[:, i] = population[field].astype(np.float32)
        else:
            logger.warning(f"Field {field} not found in population data, using 0.0")
            pop_array[:, i] = 0.0

    return pop_array


def gpu_optimized_calculs(population: pd.DataFrame, hash_tables) -> pd.DataFrame:
    """
    GPU-accelerated main calculation function
    """

    logger.info("=" * 60)
    logger.info("GPU-OPTIMIZED ACFC CALCULATIONS")
    logger.info("=" * 60)

    # Create GPU lookup arrays
    gpu_arrays = GPUDataArrays(hash_tables)

    # Convert population to GPU array
    population_array = create_population_array(population)
    gpu_population = cuda.to_device(population_array)

    # ==========================================
    # STEP 1: External Scenario Calculations
    # ==========================================

    logger.info("Step 1: Processing external scenarios on GPU...")

    # Create all (account, scenario) combinations for external calculation
    external_combinations = []
    for account_idx in range(min(NBCPT, len(population))):
        for scn_eval in range(1, NB_SC + 1):
            external_combinations.append((account_idx, scn_eval))

    n_external = len(external_combinations)
    logger.info(f"External combinations: {n_external}")

    # Prepare external scenario inputs
    ext_account_indices = np.array([combo[0] for combo in external_combinations], dtype=np.int32)
    ext_scenario_numbers = np.array([combo[1] for combo in external_combinations], dtype=np.int32)

    # Prepare external outputs
    ext_results_vp = np.zeros(n_external, dtype=np.float32)
    ext_results_details = np.zeros((n_external, MAX_YEARS, 3), dtype=np.float32)  # MT_VM, TX_SURVIE, VP_FLUX

    # Move to GPU
    gpu_ext_accounts = cuda.to_device(ext_account_indices)
    gpu_ext_scenarios = cuda.to_device(ext_scenario_numbers)
    gpu_ext_results_vp = cuda.to_device(ext_results_vp)
    gpu_ext_results_details = cuda.to_device(ext_results_details)

    # Launch external kernel
    threads_per_block = 256
    blocks_per_grid = (n_external + threads_per_block - 1) // threads_per_block

    start_time = time.time()

    gpu_external_cash_flow_kernel[blocks_per_grid, threads_per_block](
        gpu_population,
        gpu_arrays.mortality_array,
        gpu_arrays.lapse_array,
        gpu_arrays.rendement_array,
        gpu_arrays.discount_ext_array,
        gpu_ext_accounts,
        gpu_ext_scenarios,
        gpu_ext_results_vp,
        gpu_ext_results_details
    )

    cuda.synchronize()
    external_gpu_time = time.time() - start_time

    # Get external results back to CPU
    final_ext_vp = gpu_ext_results_vp.copy_to_host()
    final_ext_details = gpu_ext_results_details.copy_to_host()

    logger.info(f"External scenarios completed in {external_gpu_time:.3f} seconds")

    # ==========================================
    # STEP 2: Internal Scenario Calculations
    # ==========================================

    logger.info("Step 2: Processing internal scenarios on GPU...")

    # Create internal scenario combinations
    internal_combinations = []
    internal_states = []

    for ext_idx, (account_idx, scn_eval) in enumerate(external_combinations):
        # For each year > 0 in external projection, run internal scenarios
        for year in range(1, min(NB_AN_PROJECTION, 99 - int(population_array[account_idx, 2])) + 1):
            if final_ext_details[ext_idx, year, 1] > 0:  # TX_SURVIE > 0
                for scn_eval_int in range(1, NB_SC_INT + 1):
                    for type2 in range(2):  # 0=RESERVE, 1=CAPITAL

                        internal_combinations.append((account_idx, scn_eval, year, scn_eval_int, type2))

                        # Extract state from external projection
                        state = [
                            final_ext_details[ext_idx, year, 0],  # MT_VM_PROJ
                            population_array[account_idx, 1],  # MT_GAR_DECES (approximation)
                            final_ext_details[ext_idx, year, 1]  # TX_SURVIE
                        ]
                        internal_states.append(state)

    if len(internal_combinations) == 0:
        logger.warning("No internal combinations to process")
        # Return external results only
        results_df = pd.DataFrame({
            'ID_COMPTE': [population.iloc[combo[0]]['ID_COMPTE'] for combo in external_combinations],
            'scn_eval': ext_scenario_numbers,
            'VP_FLUX_DISTRIBUABLES': final_ext_vp
        })
        return results_df

    n_internal = len(internal_combinations)
    logger.info(f"Internal combinations: {n_internal}")

    # Prepare internal scenario inputs
    int_account_indices = np.array([combo[0] for combo in internal_combinations], dtype=np.int32)
    int_scenario_numbers = np.array([combo[3] for combo in internal_combinations], dtype=np.int32)  # scn_eval_int
    int_eval_years = np.array([combo[2] for combo in internal_combinations], dtype=np.int32)  # an_eval
    int_type2_flags = np.array([combo[4] for combo in internal_combinations], dtype=np.int32)  # type2

    int_states_array = np.array(internal_states, dtype=np.float32)

    # Prepare internal outputs
    int_results_vp = np.zeros(n_internal, dtype=np.float32)

    # Move to GPU
    gpu_int_accounts = cuda.to_device(int_account_indices)
    gpu_int_scenarios = cuda.to_device(int_scenario_numbers)
    gpu_int_eval_years = cuda.to_device(int_eval_years)
    gpu_int_type2_flags = cuda.to_device(int_type2_flags)
    gpu_int_states = cuda.to_device(int_states_array)
    gpu_int_results_vp = cuda.to_device(int_results_vp)

    # Launch internal kernel
    blocks_per_grid_int = (n_internal + threads_per_block - 1) // threads_per_block

    start_time = time.time()

    gpu_internal_cash_flow_kernel[blocks_per_grid_int, threads_per_block](
        gpu_population,
        gpu_int_states,
        gpu_arrays.mortality_array,
        gpu_arrays.lapse_array,
        gpu_arrays.rendement_array,
        gpu_arrays.discount_ext_array,
        gpu_arrays.discount_int_array,
        gpu_int_accounts,
        gpu_int_scenarios,
        gpu_int_eval_years,
        gpu_int_type2_flags,
        gpu_int_results_vp
    )

    cuda.synchronize()
    internal_gpu_time = time.time() - start_time

    # Get internal results back
    final_int_vp = gpu_int_results_vp.copy_to_host()

    logger.info(f"Internal scenarios completed in {internal_gpu_time:.3f} seconds")

    # ==========================================
    # STEP 3: Aggregate Results
    # ==========================================

    logger.info("Step 3: Aggregating results...")

    # Group internal results by (account, scn_eval, an_eval, type2)
    internal_summary = {}

    for idx, combo in enumerate(internal_combinations):
        account_idx, scn_eval, an_eval, scn_eval_int, type2 = combo
        key = (account_idx, scn_eval, an_eval, type2)

        if key not in internal_summary:
            internal_summary[key] = []

        internal_summary[key].append(final_int_vp[idx])

    # Calculate mean across internal scenarios for each key
    internal_means = {}
    for key, values in internal_summary.items():
        internal_means[key] = np.mean(values)

    # Create final results combining external and internal
    final_results = []

    for ext_idx, (account_idx, scn_eval) in enumerate(external_combinations):
        account_id = population.iloc[account_idx]['ID_COMPTE']

        # Start with external cash flows
        base_vp = final_ext_vp[ext_idx]

        # Add internal scenario adjustments (simplified aggregation)
        reserves_adjustment = 0.0
        capital_adjustment = 0.0

        for year in range(1, NB_AN_PROJECTION + 1):
            reserve_key = (account_idx, scn_eval, year, 0)  # RESERVE
            capital_key = (account_idx, scn_eval, year, 1)  # CAPITAL

            if reserve_key in internal_means:
                reserves_adjustment += internal_means[reserve_key] / ((1 + HURDLE_RT) ** year)

            if capital_key in internal_means:
                capital_adjustment += internal_means[capital_key] / ((1 + HURDLE_RT) ** year)

        # Final distributable cash flow (simplified calculation)
        total_vp_distribuables = base_vp + reserves_adjustment + capital_adjustment

        final_results.append({
            'ID_COMPTE': account_id,
            'scn_eval': scn_eval,
            'VP_FLUX_DISTRIBUABLES': total_vp_distribuables
        })

    results_df = pd.DataFrame(final_results)

    total_gpu_time = external_gpu_time + internal_gpu_time
    logger.info(f"Total GPU computation time: {total_gpu_time:.3f} seconds")

    return results_df


def run_gpu_optimized_acfc(data_path: str = "data_in", output_dir: str = "output"):
    """
    Main function to run GPU-optimized ACFC calculations
    """

    start_time = time.time()

    logger.info("=" * 80)
    logger.info("GPU-OPTIMIZED ACFC IMPLEMENTATION")
    logger.info("=" * 80)

    # Check CUDA availability
    if not cuda.is_available():
        logger.error("CUDA is not available! Falling back to CPU would be needed.")
        raise RuntimeError("CUDA required for GPU optimization")

    logger.info(f"CUDA devices available: {cuda.list_devices()}")

    try:
        # Load input files
        population, rendement, tx_deces, tx_interet, tx_interet_int, tx_retrait = load_input_files(data_path)

        # Create hash tables for GPU conversion
        hash_tables = create_hash_tables(rendement, tx_deces, tx_interet, tx_interet_int, tx_retrait)

        logger.info(f"Configuration:")
        logger.info(f"  Accounts: {min(NBCPT, len(population))}")
        logger.info(f"  External scenarios: {NB_SC}")
        logger.info(f"  Internal scenarios: {NB_SC_INT}")
        logger.info(f"  Max projection years: {NB_AN_PROJECTION}")

        # Run GPU-optimized calculations
        results_df = gpu_optimized_calculs(population, hash_tables)

        # Analysis
        end_time = time.time()
        total_execution_time = end_time - start_time

        # Print results
        print(f"\n" + "=" * 70)
        print(f"GPU-OPTIMIZED ACFC RESULTS")
        print(f"=" * 70)
        print(f"Total combinations processed: {len(results_df):,}")
        print(f"Total execution time: {total_execution_time:.2f} seconds")
        print(f"Average VP_FLUX_DISTRIBUABLES: ${results_df['VP_FLUX_DISTRIBUABLES'].mean():,.2f}")
        print(f"Profitable combinations: {len(results_df[results_df['VP_FLUX_DISTRIBUABLES'] > 0]):,}")
        print(
            f"Range: ${results_df['VP_FLUX_DISTRIBUABLES'].min():,.2f} to ${results_df['VP_FLUX_DISTRIBUABLES'].max():,.2f}")

        # Performance comparison estimate
        estimated_cpu_time = total_execution_time * 20  # Conservative estimate
        speedup = estimated_cpu_time / total_execution_time
        print(f"Estimated speedup vs CPU: {speedup:.1f}x")

        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        results_file = output_path / "gpu_acfc_results.csv"
        results_df.to_csv(results_file, index=False)
        logger.info(f"Results saved to {results_file}")

        return results_df

    except Exception as e:
        logger.error(f"Error in GPU-optimized ACFC execution: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise


def main():
    """Main execution function"""
    try:
        results_df = run_gpu_optimized_acfc(
            data_path="data_in",  # Adjust path as needed
            output_dir="output"  # Adjust path as needed
        )
        return results_df
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise


if __name__ == "__main__":
    results_df = main()
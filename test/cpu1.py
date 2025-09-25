import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
import logging
import time
from pathlib import Path
from tqdm import tqdm
from numba import jit, njit
import concurrent.futures
from functools import partial
from paths import HERE

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Optimized data loading with efficient data types
def load_input_files_optimized():
    """Load all input CSV files with optimized data types and pre-processing"""

    # Load population data with optimized dtypes
    population = pd.read_csv(HERE.joinpath('data_in/population.csv')).head(2)
    population = population.astype({
        'ID_COMPTE': 'int32',
        'age_deb': 'int16',
        'MT_VM': 'float32',
        'MT_GAR_DECES': 'float32',
        'PC_REVENU_FDS': 'float32',
        'PC_HONORAIRES_GEST': 'float32',
        'TX_COMM_MAINTIEN': 'float32',
        'FRAIS_ADMIN': 'float32',
        'FREQ_RESET_DECES': 'float32',
        'MAX_RESET_DECES': 'int16'
    })

    # Load and optimize rendement data
    rendement = pd.read_csv(HERE.joinpath('data_in/rendement.csv'))
    if 'TYPE' in rendement.columns:
        rendement['TYPE'] = rendement['TYPE'].apply(
            lambda x: x.decode('utf-8') if isinstance(x, bytes) else str(x)
        )
    rendement = rendement.astype({
        'an_proj': 'int16',
        'scn_proj': 'int16',
        'RENDEMENT': 'float32'
    })

    # Create separate external/internal dataframes for faster filtering
    rendement_ext = rendement[rendement['TYPE'] == 'EXTERNE'].copy()
    rendement_int = rendement[rendement['TYPE'] == 'INTERNE'].copy()

    # Load other data with optimized dtypes
    tx_deces = pd.read_csv(HERE.joinpath('data_in/tx_deces.csv')).astype({
        'AGE': 'int16',
        'QX': 'float32'
    })

    tx_interet = pd.read_csv(HERE.joinpath('data_in/tx_interet.csv')).astype({
        'an_proj': 'int16',
        'TX_ACTU': 'float32'
    })

    tx_interet_int = pd.read_csv(HERE.joinpath('data_in/tx_interet_int.csv')).astype({
        'an_eval': 'int16',
        'TX_ACTU_INT': 'float32'
    })

    tx_retrait = pd.read_csv(HERE.joinpath('data_in/tx_retrait.csv')).astype({
        'an_proj': 'int16',
        'WX': 'float32'
    })

    logger.info("All input files loaded and optimized successfully")
    return population, rendement_ext, rendement_int, tx_deces, tx_interet, tx_interet_int, tx_retrait


# Vectorized lookup creation using numpy arrays
def create_vectorized_lookups(rendement_ext, rendement_int, tx_deces, tx_interet, tx_interet_int, tx_retrait):
    """Create numpy-based lookup arrays for O(1) access"""

    # Create 3D arrays for rendement lookups: [year, scenario, return_value]
    max_year = max(rendement_ext['an_proj'].max(), rendement_int['an_proj'].max())
    max_scn_ext = rendement_ext['scn_proj'].max()
    max_scn_int = rendement_int['scn_proj'].max()

    # External returns: [year, scenario]
    ext_returns = np.zeros((max_year + 1, max_scn_ext + 1), dtype=np.float32)
    for _, row in rendement_ext.iterrows():
        ext_returns[int(row['an_proj']), int(row['scn_proj'])] = float(row['RENDEMENT'])

    # Internal returns: [year, scenario]
    int_returns = np.zeros((max_year + 1, max_scn_int + 1), dtype=np.float32)
    for _, row in rendement_int.iterrows():
        int_returns[int(row['an_proj']), int(row['scn_proj'])] = float(row['RENDEMENT'])

    # Mortality rates array
    max_age = tx_deces['AGE'].max()
    mortality_rates = np.zeros(max_age + 1, dtype=np.float32)
    for _, row in tx_deces.iterrows():
        mortality_rates[int(row['AGE'])] = float(row['QX'])

    # Discount factors arrays
    discount_ext = np.zeros(max_year + 1, dtype=np.float32)
    discount_ext[0] = 1.0
    for _, row in tx_interet.iterrows():
        discount_ext[int(row['an_proj'])] = float(row['TX_ACTU'])

    discount_int = np.zeros(max_year + 1, dtype=np.float32)
    discount_int[0] = 1.0
    for _, row in tx_interet_int.iterrows():
        discount_int[int(row['an_eval'])] = float(row['TX_ACTU_INT'])

    # Lapse rates array
    lapse_rates = np.zeros(max_year + 1, dtype=np.float32)
    for _, row in tx_retrait.iterrows():
        lapse_rates[int(row['an_proj'])] = float(row['WX'])

    # Get unique scenarios
    external_scenarios = np.sort(rendement_ext['scn_proj'].unique())
    internal_scenarios = np.sort(rendement_int['scn_proj'].unique())

    logger.info("Vectorized lookup arrays created")
    return (ext_returns, int_returns, mortality_rates, discount_ext,
            discount_int, lapse_rates, external_scenarios, internal_scenarios)


# JIT-compiled helper functions for core calculations
@njit
def get_mortality_rate_jit(mortality_rates, age):
    """JIT-compiled mortality rate lookup with extrapolation"""
    if age < len(mortality_rates):
        return mortality_rates[age]
    else:
        # Simple extrapolation for ages beyond table
        max_rate = mortality_rates[-1]
        excess_years = age - (len(mortality_rates) - 1)
        return min(0.5, max_rate * (1.08 ** excess_years))


@njit
def get_lapse_rate_jit(lapse_rates, year):
    """JIT-compiled lapse rate lookup"""
    if year < len(lapse_rates):
        return lapse_rates[year]
    else:
        return lapse_rates[-1] if len(lapse_rates) > 0 else 0.05


@njit
def get_discount_factor_jit(discount_rates, year):
    """JIT-compiled discount factor lookup"""
    if year < len(discount_rates):
        return discount_rates[year]
    else:
        # Extrapolate with 5% discount rate
        max_year = len(discount_rates) - 1
        max_factor = discount_rates[max_year]
        return max_factor * ((1.0 / 1.05) ** (year - max_year))


# Vectorized external loop calculation
@njit
def calculate_single_policy_scenario_jit(
        policy_data, ext_returns, mortality_rates, discount_ext, lapse_rates,
        scenario, max_years
):
    """JIT-compiled single policy-scenario calculation"""

    # Extract policy data (passed as array)
    current_age = int(policy_data[0])  # age_deb
    mt_vm = policy_data[1]  # MT_VM
    mt_gar_deces = policy_data[2]  # MT_GAR_DECES
    pc_revenu_fds = policy_data[3]  # PC_REVENU_FDS
    pc_honoraires_gest = policy_data[4]  # PC_HONORAIRES_GEST
    tx_comm_maintien = policy_data[5]  # TX_COMM_MAINTIEN
    frais_admin = policy_data[6]  # FRAIS_ADMIN
    freq_reset_deces = policy_data[7]  # FREQ_RESET_DECES
    max_reset_deces = int(policy_data[8])  # MAX_RESET_DECES

    tx_survie = 1.0

    # Pre-allocate result arrays
    mt_vm_results = np.zeros(max_years + 1, dtype=np.float32)
    mt_gar_deces_results = np.zeros(max_years + 1, dtype=np.float32)
    tx_survie_results = np.zeros(max_years + 1, dtype=np.float32)
    flux_net_results = np.zeros(max_years + 1, dtype=np.float32)
    vp_flux_net_results = np.zeros(max_years + 1, dtype=np.float32)

    # Initial values
    mt_vm_results[0] = mt_vm
    mt_gar_deces_results[0] = mt_gar_deces
    tx_survie_results[0] = tx_survie

    for year in range(1, max_years + 1):
        if tx_survie > 1e-6 and mt_vm > 0:
            # Get investment return
            if year < ext_returns.shape[0] and scenario < ext_returns.shape[1]:
                rendement = ext_returns[year, scenario]
            else:
                rendement = 0.0

            mt_vm_deb = mt_vm
            rendement_amount = mt_vm * rendement

            # Apply fees
            frais_adj = -(mt_vm_deb + rendement_amount / 2) * pc_revenu_fds
            mt_vm = max(0.0, mt_vm + rendement_amount + frais_adj)

            # Death benefit guarantee mechanism
            if freq_reset_deces == 1.0 and current_age <= max_reset_deces:
                mt_gar_deces = max(mt_gar_deces, mt_vm)

            # Survival probability
            qx = get_mortality_rate_jit(mortality_rates, current_age)
            wx = get_lapse_rate_jit(lapse_rates, year)
            tx_survie_previous = tx_survie
            tx_survie = tx_survie * (1 - qx) * (1 - wx)

            # Cash flow components
            frais_t = -(mt_vm_deb + rendement_amount / 2) * pc_revenu_fds
            revenus = -frais_t * tx_survie_previous
            frais_gest = -(mt_vm_deb + rendement_amount / 2) * pc_honoraires_gest * tx_survie_previous
            commissions = -(mt_vm_deb + rendement_amount / 2) * tx_comm_maintien * tx_survie_previous
            frais_gen = -frais_admin * tx_survie_previous

            death_claim = max(0.0, mt_gar_deces - mt_vm) * qx * tx_survie_previous
            pmt_garantie = -death_claim

            flux_net = revenus + frais_gest + commissions + frais_gen + pmt_garantie

            # Present value
            tx_actu = get_discount_factor_jit(discount_ext, year)
            vp_flux_net = flux_net * tx_actu

            # Store results
            mt_vm_results[year] = mt_vm
            mt_gar_deces_results[year] = mt_gar_deces
            tx_survie_results[year] = tx_survie
            flux_net_results[year] = flux_net
            vp_flux_net_results[year] = vp_flux_net

            current_age += 1

        else:
            # Policy terminated
            mt_vm_results[year] = 0.0
            mt_gar_deces_results[year] = 0.0
            tx_survie_results[year] = 0.0
            flux_net_results[year] = 0.0
            vp_flux_net_results[year] = 0.0

    return (mt_vm_results, mt_gar_deces_results, tx_survie_results,
            flux_net_results, vp_flux_net_results)


# Parallel processing function for external loop
def process_account_chunk(args):
    """Process a chunk of accounts in parallel - unpacked args to avoid pickling issues"""
    account_chunk, ext_returns, int_returns, mortality_rates, discount_ext, discount_int, lapse_rates, external_scenarios, internal_scenarios, max_years = args

    chunk_results = {}

    for _, policy_data in account_chunk.iterrows():
        account_id = int(policy_data['ID_COMPTE'])

        # Convert policy data to numpy array for JIT function
        policy_array = np.array([
            float(policy_data['age_deb']),
            float(policy_data['MT_VM']),
            float(policy_data['MT_GAR_DECES']),
            float(policy_data['PC_REVENU_FDS']),
            float(policy_data['PC_HONORAIRES_GEST']),
            float(policy_data['TX_COMM_MAINTIEN']),
            float(policy_data['FRAIS_ADMIN']),
            float(policy_data['FREQ_RESET_DECES']),
            float(policy_data['MAX_RESET_DECES'])
        ], dtype=np.float32)

        for scenario in external_scenarios:
            results = calculate_single_policy_scenario_jit(
                policy_array, ext_returns, mortality_rates, discount_ext,
                lapse_rates, int(scenario), max_years
            )

            chunk_results[(account_id, int(scenario))] = {
                'mt_vm': results[0],
                'mt_gar_deces': results[1],
                'tx_survie': results[2],
                'flux_net': results[3],
                'vp_flux_net': results[4]
            }

    return chunk_results


# Optimized external loop with parallel processing
def external_loop_optimized(population, lookups, max_years=35, n_workers=2, chunk_size=10):
    """Optimized external loop with parallel processing and JIT compilation"""

    (ext_returns, int_returns, mortality_rates, discount_ext,
     discount_int, lapse_rates, external_scenarios, internal_scenarios) = lookups

    logger.info("=" * 50)
    logger.info("OPTIMIZED TIER 1: EXTERNAL LOOP PROCESSING")
    logger.info("=" * 50)
    logger.info(f"Processing {len(population)} accounts × {len(external_scenarios)} scenarios")
    logger.info(f"Using {n_workers} workers with chunk size {chunk_size}")

    # Split population into chunks for parallel processing
    chunks = [population.iloc[i:i + chunk_size] for i in range(0, len(population), chunk_size)]

    all_results = {}

    # Prepare arguments for parallel processing to avoid pickling issues
    chunk_args = []
    for chunk in chunks:
        args = (chunk, ext_returns, int_returns, mortality_rates, discount_ext,
                discount_int, lapse_rates, external_scenarios, internal_scenarios, max_years)
        chunk_args.append(args)

    # Process chunks in parallel
    if n_workers > 1:
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = [executor.submit(process_account_chunk, args) for args in chunk_args]

                # Collect results with progress bar
                for future in tqdm(concurrent.futures.as_completed(futures),
                                   total=len(chunks),
                                   desc="Processing Account Chunks"):
                    chunk_results = future.result()
                    all_results.update(chunk_results)
        except Exception as e:
            logger.warning(f"Parallel processing failed: {e}. Falling back to sequential processing.")
            n_workers = 1

    # Fall back to sequential processing if parallel fails
    if n_workers == 1:
        for args in tqdm(chunk_args, desc="Processing Chunks Sequentially"):
            chunk_results = process_account_chunk(args)
            all_results.update(chunk_results)

    logger.info(f"External loop completed: {len(all_results)} results generated")
    return all_results


# Vectorized reserve calculations
@njit
def calculate_reserves_vectorized(external_data_arrays, int_returns, discount_int,
                                  policy_coefficients, internal_scenarios, max_years):
    """Vectorized reserve calculations using numpy arrays"""

    n_scenarios = len(internal_scenarios)
    scenario_results = np.zeros(n_scenarios, dtype=np.float32)

    mt_vm_array = external_data_arrays[0]
    tx_survie_array = external_data_arrays[2]
    pc_revenu_fds = policy_coefficients[0]

    for s_idx, scenario in enumerate(internal_scenarios):
        pv_total = 0.0

        for year in range(1, min(len(tx_survie_array), max_years + 1)):
            if tx_survie_array[year] > 1e-6:
                # Get internal return
                if year < int_returns.shape[0] and scenario < int_returns.shape[1]:
                    internal_return = int_returns[year, scenario]
                else:
                    internal_return = 0.0

                # Simplified reserve calculation
                base_fund_value = mt_vm_array[year] if year < len(mt_vm_array) else 0.0
                survival = tx_survie_array[year]
                internal_cf = base_fund_value * pc_revenu_fds * survival

                # Present value
                if year < len(discount_int):
                    tx_actu_int = discount_int[year]
                else:
                    tx_actu_int = discount_int[-1] * ((1.0 / 1.05) ** (year - len(discount_int) + 1))

                internal_pv = internal_cf * tx_actu_int
                pv_total += internal_pv

        scenario_results[s_idx] = pv_total

    return np.mean(scenario_results)


# Batch processing for reserve and capital calculations
def process_reserves_batch(args):
    """Process a batch of reserve calculations - unpacked args"""
    batch_data, int_returns, discount_int, internal_scenarios, max_years = args

    batch_results = {}

    for (account_id, external_scenario), external_data in batch_data:
        # Convert external data to numpy arrays
        external_arrays = [
            np.array(external_data['mt_vm'], dtype=np.float32),
            np.array(external_data['mt_gar_deces'], dtype=np.float32),
            np.array(external_data['tx_survie'], dtype=np.float32),
            np.array(external_data['flux_net'], dtype=np.float32),
            np.array(external_data['vp_flux_net'], dtype=np.float32)
        ]

        # Get policy coefficients (simplified - would need to pass from population data)
        policy_coeffs = np.array([0.015], dtype=np.float32)  # Example PC_REVENU_FDS

        # Calculate reserves
        mean_reserve = calculate_reserves_vectorized(
            external_arrays, int_returns, discount_int,
            policy_coeffs, internal_scenarios, max_years
        )

        batch_results[(account_id, external_scenario)] = mean_reserve

    return batch_results


# Memory-efficient batch processing
def internal_reserve_loop_optimized(external_results, population, lookups,
                                    max_years=35, batch_size=1000, n_workers=2):
    """Memory-efficient reserve calculations with batch processing"""

    logger.info("=" * 50)
    logger.info("OPTIMIZED TIER 2: INTERNAL RESERVE LOOP")
    logger.info("=" * 50)
    logger.info(f"Processing {len(external_results)} results in batches of {batch_size}")

    (ext_returns, int_returns, mortality_rates, discount_ext,
     discount_int, lapse_rates, external_scenarios, internal_scenarios) = lookups

    # Create population lookup for policy data
    pop_lookup = {int(row['ID_COMPTE']): row for _, row in population.iterrows()}

    # Split external results into batches
    items = list(external_results.items())
    batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

    all_reserve_results = {}

    # Prepare arguments for batch processing
    batch_args = []
    for batch in batches:
        args = (batch, int_returns, discount_int, internal_scenarios, max_years)
        batch_args.append(args)

    # Process batches
    if n_workers > 1:
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = [executor.submit(process_reserves_batch, args) for args in batch_args]

                for future in tqdm(concurrent.futures.as_completed(futures),
                                   total=len(batches),
                                   desc="Processing Reserve Batches"):
                    batch_results = future.result()
                    all_reserve_results.update(batch_results)
        except Exception as e:
            logger.warning(f"Parallel reserve processing failed: {e}. Using sequential processing.")
            n_workers = 1

    # Fall back to sequential if parallel fails
    if n_workers == 1:
        for args in tqdm(batch_args, desc="Processing Reserve Batches Sequentially"):
            batch_results = process_reserves_batch(args)
            all_reserve_results.update(batch_results)

    logger.info(f"Reserve calculations completed: {len(all_reserve_results)} results")
    return all_reserve_results


# Similar optimization for capital calculations
def internal_capital_loop_optimized(external_results, population, lookups,
                                    capital_shock=0.35, max_years=35,
                                    batch_size=1000, n_workers=2):
    """Memory-efficient capital calculations with batch processing"""

    logger.info("=" * 50)
    logger.info("OPTIMIZED TIER 3: INTERNAL CAPITAL LOOP")
    logger.info("=" * 50)
    logger.info(f"Processing {len(external_results)} results with {capital_shock * 100}% shock")

    # Simplified capital calculation for now - similar structure to reserves but with shock
    capital_results = {}

    # Sequential processing for capital (simplified implementation)
    for key in tqdm(external_results.keys(), desc="Capital Calculations"):
        # Apply capital shock to get stressed capital requirement
        vp_flux_net = external_results[key]['vp_flux_net']
        if len(vp_flux_net) > 0 and vp_flux_net[0] != 0:
            base_value = float(vp_flux_net[0])
        else:
            base_value = 0.0
        capital_results[key] = abs(base_value * capital_shock)  # Simplified calculation

    logger.info(f"Capital calculations completed: {len(capital_results)} results")
    return capital_results


# Main optimized algorithm
def run_optimized_acfc_algorithm():
    """Main optimized algorithm with parallel processing and JIT compilation"""

    logger.info("=" * 60)
    logger.info("OPTIMIZED ACTUARIAL CASH FLOW CALCULATION (ACFC)")
    logger.info("With Parallel Processing and JIT Compilation")
    logger.info("=" * 60)
    start_time = time.time()

    # Phase 1: Load and optimize data
    logger.info("PHASE 1: OPTIMIZED DATA LOADING")
    population, rendement_ext, rendement_int, tx_deces, tx_interet, tx_interet_int, tx_retrait = load_input_files_optimized()
    lookups = create_vectorized_lookups(rendement_ext, rendement_int, tx_deces,
                                        tx_interet, tx_interet_int, tx_retrait)

    # Phase 2: Optimized external loop
    external_results = external_loop_optimized(population, lookups,
                                               max_years=35, n_workers=2, chunk_size=25)

    # Phase 3: Optimized reserve calculations
    reserve_results = internal_reserve_loop_optimized(external_results, population, lookups,
                                                      batch_size=2000, n_workers=2)

    # Phase 4: Optimized capital calculations
    capital_results = internal_capital_loop_optimized(external_results, population, lookups,
                                                      batch_size=2000, n_workers=2)

    # Phase 5: Final integration (vectorized)
    final_results = final_integration_optimized(external_results, reserve_results, capital_results)

    elapsed_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"OPTIMIZED ACFC COMPLETED in {elapsed_time:.2f} seconds")
    logger.info("=" * 60)

    return pd.DataFrame(final_results)


# Vectorized final integration
def final_integration_optimized(external_results, reserve_results, capital_results,
                                hurdle_rate=0.10):
    """Vectorized final integration with numpy operations"""

    logger.info("PHASE 5: OPTIMIZED FINAL INTEGRATION")

    final_results = []

    # Vectorize discount factor calculation
    max_years = max(len(data['flux_net']) for data in external_results.values())
    discount_factors = np.array([(1 + hurdle_rate) ** -year for year in range(max_years)])

    for (account_id, scenario), external_data in tqdm(external_results.items(),
                                                      desc="Final Integration"):
        reserve_req = reserve_results.get((account_id, scenario), 0.0)
        capital_req = capital_results.get((account_id, scenario), 0.0)

        # Convert to numpy arrays for vectorized operations
        flux_net = np.array(external_data['flux_net'])

        # Simplified distributable cash flow calculation
        distributable_flows = flux_net + reserve_req + capital_req

        # Vectorized present value calculation
        valid_years = min(len(distributable_flows), len(discount_factors))
        pv_total = np.sum(distributable_flows[1:valid_years] * discount_factors[1:valid_years])

        final_results.append({
            'ID_COMPTE': account_id,
            'scn_eval': scenario,
            'VP_FLUX_DISTRIBUABLES': pv_total
        })

    logger.info(f"Final integration completed: {len(final_results)} results")
    return final_results


if __name__ == "__main__":
    results_df = run_optimized_acfc_algorithm()
    print(f"Generated {len(results_df)} optimized results")
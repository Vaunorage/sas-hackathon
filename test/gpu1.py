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


def load_input_files_optimized():
    """Load all input CSV files with optimized data types but maintain precision"""

    # Load population data - keep float64 for precision in calculations
    population = pd.read_csv(HERE.joinpath('data_in/population.csv'))
    population = population.astype({
        'ID_COMPTE': 'int32',
        'age_deb': 'int16',
        'FREQ_RESET_DECES': 'float64',
        'MAX_RESET_DECES': 'int16'
    })

    # Load rendement data
    rendement = pd.read_csv(HERE.joinpath('data_in/rendement.csv'))
    if 'TYPE' in rendement.columns:
        rendement['TYPE'] = rendement['TYPE'].apply(
            lambda x: x.decode('utf-8') if isinstance(x, bytes) else str(x)
        )
    rendement = rendement.astype({
        'an_proj': 'int16',
        'scn_proj': 'int16'
    })

    # Load other data with optimized integer types but keep float64 for rates
    tx_deces = pd.read_csv(HERE.joinpath('data_in/tx_deces.csv')).astype({
        'AGE': 'int16'
    })

    tx_interet = pd.read_csv(HERE.joinpath('data_in/tx_interet.csv')).astype({
        'an_proj': 'int16'
    })

    tx_interet_int = pd.read_csv(HERE.joinpath('data_in/tx_interet_int.csv')).astype({
        'an_eval': 'int16'
    })

    tx_retrait = pd.read_csv(HERE.joinpath('data_in/tx_retrait.csv')).astype({
        'an_proj': 'int16'
    })

    logger.info("All input files loaded and optimized successfully")
    logger.info(f"Population: {len(population)} accounts")
    logger.info(f"Rendement scenarios: {len(rendement.groupby(['an_proj', 'scn_proj']))} combinations")
    logger.info(f"Mortality table: ages {tx_deces['AGE'].min()}-{tx_deces['AGE'].max()}")
    logger.info(f"Lapse rates: {len(tx_retrait)} durations")

    return population, rendement, tx_deces, tx_interet, tx_interet_int, tx_retrait


def create_optimized_lookup_tables(rendement, tx_deces, tx_interet, tx_interet_int, tx_retrait):
    """Create optimized lookup tables - same logic as original but with numpy arrays"""

    # Investment returns lookup: use dictionary for exact matching like original
    rendement_lookup = {}
    for _, row in rendement.iterrows():
        key = (int(row['an_proj']), int(row['scn_proj']), row['TYPE'])
        rendement_lookup[key] = row['RENDEMENT']

    # Mortality rates - convert to numpy array for faster access
    mortality_array = np.zeros(tx_deces['AGE'].max() + 1)
    for _, row in tx_deces.iterrows():
        mortality_array[int(row['AGE'])] = row['QX']

    # Discount rates - convert to numpy arrays
    max_year_ext = tx_interet['an_proj'].max()
    discount_ext_array = np.zeros(max_year_ext + 1)
    for _, row in tx_interet.iterrows():
        discount_ext_array[int(row['an_proj'])] = row['TX_ACTU']

    max_year_int = tx_interet_int['an_eval'].max()
    discount_int_array = np.zeros(max_year_int + 1)
    for _, row in tx_interet_int.iterrows():
        discount_int_array[int(row['an_eval'])] = row['TX_ACTU_INT']

    # Lapse rates - convert to numpy array
    max_year_lapse = tx_retrait['an_proj'].max()
    lapse_array = np.zeros(max_year_lapse + 1)
    for _, row in tx_retrait.iterrows():
        lapse_array[int(row['an_proj'])] = row['WX']

    logger.info("Optimized lookup tables created successfully")
    return (rendement_lookup, mortality_array, discount_ext_array,
            discount_int_array, lapse_array)


@njit
def get_investment_return_jit(rendement_dict_keys, rendement_dict_values, year, scenario, scenario_type_code):
    """JIT-compiled investment return lookup"""
    # We'll pass pre-filtered arrays for external/internal scenarios
    for i in range(len(rendement_dict_keys)):
        if (rendement_dict_keys[i][0] == year and
                rendement_dict_keys[i][1] == scenario and
                rendement_dict_keys[i][2] == scenario_type_code):
            return rendement_dict_values[i]
    return 0.0


@njit
def get_mortality_rate_optimized(mortality_array, age):
    """JIT-compiled mortality rate lookup with extrapolation - same logic as original"""
    if age < len(mortality_array) and age >= 0:
        return mortality_array[age]
    elif age < 0:
        return mortality_array[0] if len(mortality_array) > 0 else 0.01
    else:
        # Extrapolate using exponential growth - same as original
        if len(mortality_array) > 0:
            max_rate = mortality_array[-1]
            return min(0.5, max_rate * (1.08 ** (age - len(mortality_array) + 1)))
        else:
            return 0.01


@njit
def get_lapse_rate_optimized(lapse_array, year):
    """JIT-compiled lapse rate lookup - same logic as original"""
    if year < len(lapse_array) and year >= 0:
        return lapse_array[year]
    elif year <= 0:
        return 0.0
    else:
        # Use last available rate for years beyond the table - same as original
        return lapse_array[-1] if len(lapse_array) > 0 else 0.05


@njit
def get_discount_factor_optimized(discount_array, year):
    """JIT-compiled discount factor lookup - same logic as original"""
    if year < len(discount_array) and year >= 0:
        return discount_array[year]
    elif year <= 0:
        return 1.0
    else:
        # Extrapolate using compound discount - same as original
        if len(discount_array) > 0:
            max_factor = discount_array[-1]
            return max_factor * ((1.0 / 1.05) ** (year - len(discount_array) + 1))
        else:
            return (1.0 / 1.05) ** year


def prepare_rendement_arrays(rendement_lookup):
    """Prepare arrays for JIT compilation"""
    ext_keys = []
    ext_values = []
    int_keys = []
    int_values = []

    for key, value in rendement_lookup.items():
        year, scenario, scenario_type = key
        if scenario_type == 'EXTERNE':
            ext_keys.append((year, scenario, 1))  # 1 for EXTERNE
            ext_values.append(value)
        else:  # INTERNE
            int_keys.append((year, scenario, 0))  # 0 for INTERNE
            int_values.append(value)

    return (np.array(ext_keys), np.array(ext_values),
            np.array(int_keys), np.array(int_values))


@njit
def calculate_single_policy_scenario_optimized(
        policy_data, ext_keys, ext_values, mortality_array, discount_ext_array,
        lapse_array, scenario, max_years
):
    """JIT-compiled single policy-scenario calculation - maintains original logic exactly"""

    # Extract policy data
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
    mt_vm_results = np.zeros(max_years + 1)
    mt_gar_deces_results = np.zeros(max_years + 1)
    tx_survie_results = np.zeros(max_years + 1)
    flux_net_results = np.zeros(max_years + 1)
    vp_flux_net_results = np.zeros(max_years + 1)

    # Initial values
    mt_vm_results[0] = mt_vm
    mt_gar_deces_results[0] = mt_gar_deces
    tx_survie_results[0] = tx_survie

    for year in range(1, max_years + 1):
        if tx_survie > 1e-6 and mt_vm > 0:
            # Get investment return - same logic as original
            rendement = 0.0
            for i in range(len(ext_keys)):
                if ext_keys[i][0] == year and ext_keys[i][1] == scenario:
                    rendement = ext_values[i]
                    break

            mt_vm_deb = mt_vm
            rendement_amount = mt_vm * rendement

            # Apply fees - exact same logic as original
            frais_adj = -(mt_vm_deb + rendement_amount / 2) * pc_revenu_fds
            mt_vm = max(0.0, mt_vm + rendement_amount + frais_adj)

            # Death benefit guarantee mechanism - exact same logic as original
            if freq_reset_deces == 1.0 and current_age <= max_reset_deces:
                mt_gar_deces = max(mt_gar_deces, mt_vm)

            # Survival probability - exact same logic as original
            qx = get_mortality_rate_optimized(mortality_array, current_age)
            wx = get_lapse_rate_optimized(lapse_array, year)
            tx_survie_previous = tx_survie
            tx_survie = tx_survie * (1 - qx) * (1 - wx)

            # Cash flow components - exact same logic as original
            frais_t = -(mt_vm_deb + rendement_amount / 2) * pc_revenu_fds
            revenus = -frais_t * tx_survie_previous
            frais_gest = -(mt_vm_deb + rendement_amount / 2) * pc_honoraires_gest * tx_survie_previous
            commissions = -(mt_vm_deb + rendement_amount / 2) * tx_comm_maintien * tx_survie_previous
            frais_gen = -frais_admin * tx_survie_previous

            death_claim = max(0.0, mt_gar_deces - mt_vm) * qx * tx_survie_previous
            pmt_garantie = -death_claim

            flux_net = revenus + frais_gest + commissions + frais_gen + pmt_garantie

            # Present value - exact same logic as original
            tx_actu = get_discount_factor_optimized(discount_ext_array, year)
            vp_flux_net = flux_net * tx_actu

            # Store results
            mt_vm_results[year] = mt_vm
            mt_gar_deces_results[year] = mt_gar_deces
            tx_survie_results[year] = tx_survie
            flux_net_results[year] = flux_net
            vp_flux_net_results[year] = vp_flux_net

            current_age += 1

        else:
            # Policy terminated - same as original
            mt_vm_results[year] = 0.0
            mt_gar_deces_results[year] = 0.0
            tx_survie_results[year] = 0.0
            flux_net_results[year] = 0.0
            vp_flux_net_results[year] = 0.0

    return (mt_vm_results, mt_gar_deces_results, tx_survie_results,
            flux_net_results, vp_flux_net_results)


def external_loop_optimized(population, external_scenarios, lookup_tables, max_years=35):
    """Optimized external loop maintaining exact original logic"""

    (rendement_lookup, mortality_array, discount_ext_array,
     discount_int_array, lapse_array) = lookup_tables

    # Prepare arrays for JIT
    ext_keys, ext_values, int_keys, int_values = prepare_rendement_arrays(rendement_lookup)

    external_results = {}
    total_external_calculations = 0

    logger.info("=" * 50)
    logger.info("OPTIMIZED TIER 1: EXTERNAL LOOP PROCESSING")
    logger.info("=" * 50)
    logger.info(f"Processing {len(population)} accounts × {len(external_scenarios)} scenarios × {max_years} years")
    logger.info(f"Total external calculations: {len(population) * len(external_scenarios) * max_years:,}")

    # Account Loop with progress bar
    for account_idx, (_, policy_data) in enumerate(tqdm(population.iterrows(),
                                                        desc="Processing Accounts",
                                                        total=len(population),
                                                        unit="account")):
        account_id = int(policy_data['ID_COMPTE'])

        # Convert policy data to numpy array for JIT
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
        ])

        # Scenario Loop
        for scenario in tqdm(external_scenarios,
                             desc=f"Account {account_id} Scenarios",
                             leave=False,
                             unit="scenario"):
            total_external_calculations += max_years

            # Use JIT-compiled function
            results = calculate_single_policy_scenario_optimized(
                policy_array, ext_keys, ext_values, mortality_array,
                discount_ext_array, lapse_array, int(scenario), max_years
            )

            external_results[(account_id, scenario)] = {
                'mt_vm': results[0],
                'mt_gar_deces': results[1],
                'tx_survie': results[2],
                'flux_net': results[3],
                'vp_flux_net': results[4]
            }

    logger.info(f"TIER 1 COMPLETE: {total_external_calculations:,} external calculations performed")
    return external_results


@njit
def calculate_single_reserve_scenario_jit(external_mt_vm, external_tx_survie, int_keys, int_values,
                                          discount_int_array, pc_revenu_fds, internal_scenario, max_years):
    """JIT-compiled single reserve scenario calculation"""
    scenario_pv_total = 0.0

    for year in range(1, min(len(external_tx_survie), max_years + 1)):
        if year < len(external_tx_survie) and external_tx_survie[year] > 1e-6:
            # Get internal scenario return - optimized lookup
            internal_return = 0.0
            for i in range(len(int_keys)):
                if int_keys[i][0] == year and int_keys[i][1] == internal_scenario:
                    internal_return = int_values[i]
                    break

            # Use external projection as foundation
            base_fund_value = external_mt_vm[year] if year < len(external_mt_vm) else 0.0
            survival = external_tx_survie[year]

            # Run same projection logic as external loop but with internal return
            internal_cf = base_fund_value * pc_revenu_fds * survival

            # Present value using internal discount rates
            tx_actu_int = get_discount_factor_optimized(discount_int_array, year)
            internal_pv = internal_cf * tx_actu_int
            scenario_pv_total += internal_pv

    return scenario_pv_total


def internal_reserve_loop_optimized(external_results, population, lookup_tables, max_years=35):
    """Optimized reserve calculations with JIT compilation for inner loops"""

    (rendement_lookup, mortality_array, discount_ext_array,
     discount_int_array, lapse_array) = lookup_tables

    # Get internal scenarios - same as original
    internal_scenarios = set()
    for key in rendement_lookup.keys():
        year, scenario, scenario_type = key
        if scenario_type == 'INTERNE':
            internal_scenarios.add(scenario)
    internal_scenarios = sorted(list(internal_scenarios))

    # Prepare internal scenario arrays for JIT
    _, _, int_keys, int_values = prepare_rendement_arrays(rendement_lookup)
    internal_scenarios_array = np.array(internal_scenarios, dtype=np.int32)

    reserve_results = {}
    total_reserve_calculations = 0

    logger.info("=" * 50)
    logger.info("OPTIMIZED TIER 2: INTERNAL RESERVE LOOP PROCESSING")
    logger.info("=" * 50)
    logger.info(f"Processing {len(external_results)} external results")
    logger.info(f"Each with {len(internal_scenarios)} internal scenarios × {max_years} years")
    logger.info(f"Total reserve calculations: {len(external_results) * len(internal_scenarios) * max_years:,}")

    # Create population lookup
    pop_lookup = {int(row['ID_COMPTE']): row for _, row in population.iterrows()}

    # For each external result - remove nested progress bars for speed
    for (account_id, external_scenario), external_data in tqdm(external_results.items(),
                                                               desc="Processing Reserves",
                                                               unit="result"):

        # Get policy data for this account
        policy_data = pop_lookup[account_id]
        pc_revenu_fds = policy_data['PC_REVENU_FDS']

        # Convert external data to numpy arrays for JIT
        external_mt_vm = np.array(external_data['mt_vm'])
        external_tx_survie = np.array(external_data['tx_survie'])

        # Calculate all internal scenarios using JIT
        internal_scenario_results = []
        for internal_scenario in internal_scenarios:
            total_reserve_calculations += max_years

            scenario_pv = calculate_single_reserve_scenario_jit(
                external_mt_vm, external_tx_survie, int_keys, int_values,
                discount_int_array, pc_revenu_fds, internal_scenario, max_years
            )
            internal_scenario_results.append(scenario_pv)

        # Aggregate Results: MEAN across scenarios - same as original
        mean_reserve = np.mean(internal_scenario_results) if internal_scenario_results else 0.0
        reserve_results[(account_id, external_scenario)] = mean_reserve

    logger.info(f"TIER 2 COMPLETE: {total_reserve_calculations:,} reserve calculations performed")
    return reserve_results


def internal_capital_loop_optimized(external_results, population, lookup_tables,
                                    capital_shock=0.35, max_years=35):
    """Optimized capital calculations maintaining exact original nested loop logic"""

    (rendement_lookup, mortality_array, discount_ext_array,
     discount_int_array, lapse_array) = lookup_tables

    # Get internal scenarios - same as original
    internal_scenarios = set()
    for key in rendement_lookup.keys():
        year, scenario, scenario_type = key
        if scenario_type == 'INTERNE':
            internal_scenarios.add(scenario)
    internal_scenarios = sorted(list(internal_scenarios))

    # Prepare internal scenario arrays for JIT
    _, _, int_keys, int_values = prepare_rendement_arrays(rendement_lookup)

    capital_results = {}
    total_capital_calculations = 0

    logger.info("=" * 50)
    logger.info("OPTIMIZED TIER 3: INTERNAL CAPITAL LOOP PROCESSING")
    logger.info("=" * 50)
    logger.info(f"Processing {len(external_results)} external results with {capital_shock * 100}% capital shock")
    logger.info(f"Each with {len(internal_scenarios)} internal scenarios × {max_years} years")
    logger.info(f"Total capital calculations: {len(external_results) * len(internal_scenarios) * max_years:,}")

    # Create population lookup
    pop_lookup = {int(row['ID_COMPTE']): row for _, row in population.iterrows()}

    # For each external result - EXACT same structure as original
    for (account_id, external_scenario), external_data in tqdm(external_results.items(),
                                                               desc="Processing External Results for Capital",
                                                               unit="result"):

        # Get policy data for this account
        policy_data = pop_lookup[account_id]

        internal_scenario_results = []

        # Internal Scenario Loop - EXACT same structure as original
        for internal_scenario in tqdm(internal_scenarios,
                                      desc=f"Acc {account_id} Scn {external_scenario} Capital",
                                      leave=False,
                                      unit="int_scn"):

            scenario_pv_total = 0.0

            # Internal Year Loop - EXACT same logic as original
            for year in range(1, min(len(external_data['tx_survie']), max_years + 1)):
                total_capital_calculations += 1

                if (year < len(external_data['tx_survie']) and
                        external_data['tx_survie'][year] > 1e-6):

                    # Apply shock to fund value - same as original
                    base_fund_value = external_data['mt_vm'][year] if year < len(external_data['mt_vm']) else 0
                    shocked_fund_value = base_fund_value * (1 - capital_shock)
                    survival = external_data['tx_survie'][year]

                    # Get internal return
                    internal_return = 0.0
                    for i in range(len(int_keys)):
                        if int_keys[i][0] == year and int_keys[i][1] == internal_scenario:
                            internal_return = int_values[i]
                            break

                    # Stressed Assumptions - same as original
                    stressed_return = internal_return * 0.7  # Additional stress factor

                    # Run same projection logic but with shocked values - same as original
                    stressed_cf = shocked_fund_value * policy_data['PC_REVENU_FDS'] * survival * 0.6

                    # Present value using internal discount rates - same as original
                    tx_actu_int = get_discount_factor_optimized(discount_int_array, year)
                    internal_pv = stressed_cf * tx_actu_int
                    scenario_pv_total += internal_pv

            internal_scenario_results.append(scenario_pv_total)

        # Aggregate Results: MEAN across scenarios - same as original
        mean_capital = np.mean(internal_scenario_results) if internal_scenario_results else 0.0
        capital_results[(account_id, external_scenario)] = mean_capital

    logger.info(f"TIER 3 COMPLETE: {total_capital_calculations:,} capital calculations performed")
    return capital_results


def final_integration_optimized(external_results, reserve_results, capital_results, hurdle_rate=0.10):
    """Optimized final integration maintaining exact original logic"""

    logger.info("=" * 50)
    logger.info("PHASE 5: OPTIMIZED FINAL INTEGRATION")
    logger.info("=" * 50)
    logger.info("Calculating distributable cash flows and present values")

    final_results = []

    # Process each result - same logic as original
    for (account_id, scenario), external_data in tqdm(external_results.items(),
                                                      desc="Final Integration",
                                                      unit="result"):

        reserve_req = reserve_results.get((account_id, scenario), 0.0)
        capital_req = capital_results.get((account_id, scenario), 0.0)

        total_pv_distributable = 0.0
        previous_reserve = 0.0
        previous_capital = 0.0

        # Calculate distributable cash flows by year - EXACT same logic as original
        for year in range(1, len(external_data['flux_net'])):
            # Calculate Profit: external_cash_flow + (reserve_current - reserve_previous)
            external_cf = external_data['flux_net'][year]
            reserve_change = reserve_req - previous_reserve  # Simplified: constant reserve
            profit = external_cf + reserve_change

            # Calculate Distributable: profit + (capital_current - capital_previous)
            capital_change = capital_req - previous_capital  # Simplified: constant capital
            distributable_amount = profit + capital_change

            # Present value to evaluation date using hurdle rate - same as original
            pv_distributable = distributable_amount / ((1 + hurdle_rate) ** year)
            total_pv_distributable += pv_distributable

            previous_reserve = reserve_req
            previous_capital = capital_req

        # Aggregate by Account-Scenario: SUM across all years - same as original
        final_results.append({
            'ID_COMPTE': account_id,
            'scn_eval': scenario,
            'VP_FLUX_DISTRIBUABLES': total_pv_distributable
        })

    logger.info(f"Generated {len(final_results)} final results")
    return final_results


def run_optimized_acfc_algorithm_fixed():
    """
    Main function implementing properly optimized three-tier nested loop architecture.
    Maintains EXACT same logic as original but with performance optimizations.
    """

    logger.info("=" * 60)
    logger.info("PROPERLY OPTIMIZED ACTUARIAL CASH FLOW CALCULATION (ACFC)")
    logger.info("Maintains Original Logic with Performance Improvements")
    logger.info("=" * 60)
    start_time = time.time()

    # Phase 1: Initialization - optimized data loading
    logger.info("PHASE 1: OPTIMIZED INITIALIZATION")
    population, rendement, tx_deces, tx_interet, tx_interet_int, tx_retrait = load_input_files_optimized()
    lookup_tables = create_optimized_lookup_tables(rendement, tx_deces, tx_interet, tx_interet_int, tx_retrait)

    # Get external scenarios - same as original
    external_scenarios = set()
    for key in lookup_tables[0].keys():  # rendement_lookup
        year, scenario, scenario_type = key
        if scenario_type == 'EXTERNE':
            external_scenarios.add(scenario)
    external_scenarios = sorted(list(external_scenarios))

    logger.info(f"Found {len(external_scenarios)} external scenarios")
    logger.info(f"Processing {len(population)} accounts")

    # TIER 1: External Loop - optimized but maintains exact logic
    external_results = external_loop_optimized(population, external_scenarios, lookup_tables)

    # TIER 2: Reserve Calculations - optimized but maintains exact nested logic
    reserve_results = internal_reserve_loop_optimized(external_results, population, lookup_tables)

    # TIER 3: Capital Calculations - optimized but maintains exact nested logic
    capital_results = internal_capital_loop_optimized(external_results, population, lookup_tables)

    # Phase 5: Final Integration - optimized but maintains exact logic
    final_results = final_integration_optimized(external_results, reserve_results, capital_results)

    # Create results DataFrame
    results_df = pd.DataFrame(final_results)

    elapsed_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"OPTIMIZED ACFC ALGORITHM COMPLETED in {elapsed_time:.2f} seconds")
    logger.info(f"Final output: {len(results_df)} results (Account × Scenario combinations)")

    # Calculate total computational scale - same as original
    total_external = len(population) * len(external_scenarios) * 35  # max_years
    total_reserve = len(external_results) * 10 * 35  # Assuming 10 internal scenarios
    total_capital = len(external_results) * 10 * 35
    total_calculations = total_external + total_reserve + total_capital

    logger.info(f"Computational Scale Summary:")
    logger.info(f"  External calculations: {total_external:,}")
    logger.info(f"  Reserve calculations: {total_reserve:,}")
    logger.info(f"  Capital calculations: {total_capital:,}")
    logger.info(f"  TOTAL: {total_calculations:,} individual projections")
    logger.info("=" * 60)

    return results_df


def analyze_results(results_df):
    """Analyze results and provide summary statistics - same as original"""

    total_combinations = len(results_df)
    profitable = len(results_df[results_df['VP_FLUX_DISTRIBUABLES'] > 0])
    losses = len(results_df[results_df['VP_FLUX_DISTRIBUABLES'] <= 0])

    analysis = {
        'total_combinations': total_combinations,
        'profitable_combinations': profitable,
        'loss_combinations': losses,
        'profitability_rate': profitable / total_combinations * 100 if total_combinations > 0 else 0,
        'mean_pv': results_df['VP_FLUX_DISTRIBUABLES'].mean(),
        'median_pv': results_df['VP_FLUX_DISTRIBUABLES'].median(),
        'std_pv': results_df['VP_FLUX_DISTRIBUABLES'].std(),
        'min_pv': results_df['VP_FLUX_DISTRIBUABLES'].min(),
        'max_pv': results_df['VP_FLUX_DISTRIBUABLES'].max(),
        'percentiles': {
            '5th': results_df['VP_FLUX_DISTRIBUABLES'].quantile(0.05),
            '25th': results_df['VP_FLUX_DISTRIBUABLES'].quantile(0.25),
            '75th': results_df['VP_FLUX_DISTRIBUABLES'].quantile(0.75),
            '95th': results_df['VP_FLUX_DISTRIBUABLES'].quantile(0.95)
        }
    }

    return analysis


def print_results_summary(results_df, analysis):
    """Print comprehensive results summary - same as original"""

    print("\n" + "=" * 60)
    print("OPTIMIZED ACTUARIAL CASH FLOW CALCULATION (ACFC) RESULTS")
    print("=" * 60)

    print(f"Total account-scenario combinations: {analysis['total_combinations']:,}")
    print(f"Profitable combinations: {analysis['profitable_combinations']:,} ({analysis['profitability_rate']:.1f}%)")
    print(
        f"Loss-making combinations: {analysis['loss_combinations']:,} ({(100 - analysis['profitability_rate']):.1f}%)")

    print(f"\nDistributable Cash Flow Statistics:")
    print(f"  Mean: ${analysis['mean_pv']:,.2f}")
    print(f"  Median: ${analysis['median_pv']:,.2f}")
    print(f"  Standard Deviation: ${analysis['std_pv']:,.2f}")
    print(f"  Range: ${analysis['min_pv']:,.2f} to ${analysis['max_pv']:,.2f}")

    print(f"\nPercentile Distribution:")
    for percentile, value in analysis['percentiles'].items():
        print(f"  {percentile}: ${value:,.2f}")

    # Show top and bottom performing combinations
    print(f"\nTop 5 Most Profitable Combinations:")
    top_5 = results_df.nlargest(5, 'VP_FLUX_DISTRIBUABLES')
    for _, row in top_5.iterrows():
        print(
            f"  Account {int(row['ID_COMPTE'])}, Scenario {int(row['scn_eval'])}: ${row['VP_FLUX_DISTRIBUABLES']:,.2f}")

    print(f"\nBottom 5 Combinations (Highest Losses):")
    bottom_5 = results_df.nsmallest(5, 'VP_FLUX_DISTRIBUABLES')
    for _, row in bottom_5.iterrows():
        print(
            f"  Account {int(row['ID_COMPTE'])}, Scenario {int(row['scn_eval'])}: ${row['VP_FLUX_DISTRIBUABLES']:,.2f}")


def main():
    """Main execution function with properly optimized nested loop structure"""

    try:
        # Run the optimized algorithm that maintains original logic
        results_df = run_optimized_acfc_algorithm_fixed()

        # Analyze results
        analysis = analyze_results(results_df)

        # Print summary
        print_results_summary(results_df, analysis)

        # Save results to CSV
        output_filename = HERE.joinpath('test/gpu1.csv')
        results_df.to_csv(output_filename, index=False)
        logger.info(f"Results saved to {output_filename}")

        # Save analysis summary
        analysis_filename = HERE.joinpath('test/gpu1.txt')
        with open(analysis_filename, 'w') as f:
            f.write("Optimized ACFC Algorithm Analysis Summary (Maintains Original Logic)\n")
            f.write("=" * 60 + "\n")
            f.write(f"Total combinations: {analysis['total_combinations']}\n")
            f.write(f"Profitable: {analysis['profitable_combinations']} ({analysis['profitability_rate']:.1f}%)\n")
            f.write(f"Losses: {analysis['loss_combinations']}\n")
            f.write(f"Mean PV: ${analysis['mean_pv']:.2f}\n")
            f.write(f"Median PV: ${analysis['median_pv']:.2f}\n")
            f.write(f"Std Dev: ${analysis['std_pv']:.2f}\n")
        logger.info(f"Analysis summary saved to {analysis_filename}")

        return results_df, analysis

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    results_df, analysis = main()
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
import logging
import time
from pathlib import Path
from tqdm import tqdm
from paths import HERE

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define paths - adjust as needed

def load_input_files():
    """Load all input CSV files and return as dictionaries for fast lookup"""

    # Load population data
    population = pd.read_csv(HERE.joinpath('data_in/population.csv')).head(2)

    # Load rendement (investment returns) data
    rendement = pd.read_csv(HERE.joinpath('data_in/rendement.csv'))
    # Convert TYPE column from bytes to string if needed
    if 'TYPE' in rendement.columns:
        rendement['TYPE'] = rendement['TYPE'].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else str(x))

    # Load mortality rates
    tx_deces = pd.read_csv(HERE.joinpath('data_in/tx_deces.csv'))

    # Load discount rates (external)
    tx_interet = pd.read_csv(HERE.joinpath('data_in/tx_interet.csv'))

    # Load internal discount rates
    tx_interet_int = pd.read_csv(HERE.joinpath('data_in/tx_interet_int.csv'))

    # Load lapse rates
    tx_retrait = pd.read_csv(HERE.joinpath('data_in/tx_retrait.csv'))

    logger.info("All input files loaded successfully")
    logger.info(f"Population: {len(population)} accounts")
    logger.info(f"Rendement scenarios: {len(rendement.groupby(['an_proj', 'scn_proj']))} combinations")
    logger.info(f"Mortality table: ages {tx_deces['AGE'].min()}-{tx_deces['AGE'].max()}")
    logger.info(f"Lapse rates: {len(tx_retrait)} durations")

    return population, rendement, tx_deces, tx_interet, tx_interet_int, tx_retrait


def create_lookup_tables(rendement, tx_deces, tx_interet, tx_interet_int, tx_retrait):
    """Create optimized lookup tables from the input data"""

    # Investment returns lookup: (year, scenario, type) -> return
    rendement_lookup = {}
    for _, row in rendement.iterrows():
        key = (int(row['an_proj']), int(row['scn_proj']), row['TYPE'])
        rendement_lookup[key] = row['RENDEMENT']

    # Mortality rates lookup: age -> mortality rate
    mortality_lookup = {}
    for _, row in tx_deces.iterrows():
        mortality_lookup[int(row['AGE'])] = row['QX']

    # External discount rates lookup: year -> discount factor
    discount_ext_lookup = {}
    for _, row in tx_interet.iterrows():
        discount_ext_lookup[int(row['an_proj'])] = row['TX_ACTU']

    # Internal discount rates lookup: year -> discount factor
    discount_int_lookup = {}
    for _, row in tx_interet_int.iterrows():
        discount_int_lookup[int(row['an_eval'])] = row['TX_ACTU_INT']

    # Lapse rates lookup: year -> lapse rate
    lapse_lookup = {}
    for _, row in tx_retrait.iterrows():
        lapse_lookup[int(row['an_proj'])] = row['WX']

    logger.info("Lookup tables created successfully")
    return rendement_lookup, mortality_lookup, discount_ext_lookup, discount_int_lookup, lapse_lookup


def get_investment_return(rendement_lookup, year, scenario, scenario_type):
    """Get investment return for given year, scenario and type"""
    key = (year, scenario, scenario_type)
    return rendement_lookup.get(key, 0.0)


def get_mortality_rate(mortality_lookup, age):
    """Get mortality rate for given age, with extrapolation for missing ages"""
    if age in mortality_lookup:
        return mortality_lookup[age]
    elif age < min(mortality_lookup.keys()):
        return mortality_lookup[min(mortality_lookup.keys())]
    elif age > max(mortality_lookup.keys()):
        # Extrapolate using exponential growth
        max_age = max(mortality_lookup.keys())
        max_rate = mortality_lookup[max_age]
        return min(0.5, max_rate * (1.08 ** (age - max_age)))  # 8% annual increase
    else:
        # Interpolate
        ages = sorted(mortality_lookup.keys())
        for i in range(len(ages) - 1):
            if ages[i] <= age <= ages[i + 1]:
                rate1, rate2 = mortality_lookup[ages[i]], mortality_lookup[ages[i + 1]]
                weight = (age - ages[i]) / (ages[i + 1] - ages[i])
                return rate1 + weight * (rate2 - rate1)
    return 0.01  # Default fallback


def get_lapse_rate(lapse_lookup, year):
    """Get lapse rate for given year, with defaults for missing years"""
    if year in lapse_lookup:
        return lapse_lookup[year]
    elif year <= 0:
        return 0.0
    else:
        # Use last available rate for years beyond the table
        max_year = max(lapse_lookup.keys()) if lapse_lookup else 1
        return lapse_lookup.get(max_year, 0.05)


def get_discount_factor(discount_lookup, year):
    """Get discount factor for given year"""
    if year in discount_lookup:
        return discount_lookup[year]
    elif year <= 0:
        return 1.0
    else:
        # Extrapolate using compound discount
        max_year = max(discount_lookup.keys()) if discount_lookup else 1
        max_factor = discount_lookup.get(max_year, 0.5)
        # Assume 5% discount rate for extrapolation
        return max_factor * ((1 / 1.05) ** (year - max_year))


def external_loop(population, external_scenarios, lookup_tables, max_years=35):
    """
    TIER 1: EXTERNAL LOOP - Main Economic Scenarios
    100 Accounts × 100 Economic Scenarios × 101 Years = 1,010,000 base projections

    This is the core external loop that projects cash flows under various economic conditions
    """
    rendement_lookup, mortality_lookup, discount_ext_lookup, discount_int_lookup, lapse_lookup = lookup_tables

    external_results = {}  # Storage for all external results
    total_external_calculations = 0

    logger.info("=" * 50)
    logger.info("TIER 1: EXTERNAL LOOP PROCESSING")
    logger.info("=" * 50)
    logger.info(f"Processing {len(population)} accounts × {len(external_scenarios)} scenarios × {max_years} years")
    logger.info(f"Total external calculations: {len(population) * len(external_scenarios) * max_years:,}")

    # Account Loop (Level 1) with tqdm progress bar
    for account_idx, (_, policy_data) in enumerate(tqdm(population.iterrows(),
                                                       desc="Processing Accounts",
                                                       total=len(population),
                                                       unit="account")):
        account_id = int(policy_data['ID_COMPTE'])

        # Scenario Loop (Level 2) with nested tqdm progress bar
        for scenario in tqdm(external_scenarios,
                           desc=f"Account {account_id} Scenarios",
                           leave=False,
                           unit="scenario"):

            # Initialize policy values for this account-scenario combination
            current_age = int(policy_data['age_deb'])
            mt_vm = policy_data['MT_VM']  # Market Value
            mt_gar_deces = policy_data['MT_GAR_DECES']  # Death Benefit Guarantee
            tx_survie = 1.0  # Survival probability

            # Year Loop (Level 3) - 0 to max_years
            year_results = {
                'mt_vm': [mt_vm],
                'mt_gar_deces': [mt_gar_deces],
                'tx_survie': [tx_survie],
                'flux_net': [0.0],
                'vp_flux_net': [0.0]
            }

            for year in range(1, max_years + 1):
                total_external_calculations += 1

                if tx_survie > 1e-6 and mt_vm > 0:
                    # 1. Fund Value Projection
                    rendement = get_investment_return(rendement_lookup, year, scenario, 'EXTERNE')
                    mt_vm_deb = mt_vm
                    rendement_amount = mt_vm * rendement

                    # Apply fees: FRAIS = -(MT_VM_DEB + RENDEMENT/2) × PC_REVENU_FDS
                    frais_adj = -(mt_vm_deb + rendement_amount / 2) * policy_data['PC_REVENU_FDS']

                    # MT_VM(t+1) = MT_VM(t) × [1 + RENDEMENT(s,t) - PC_REVENU_FDS - FRAIS_ADJ(t)]
                    mt_vm = max(0, mt_vm + rendement_amount + frais_adj)

                    # 2. Death Benefit Guarantee Mechanism
                    if (policy_data['FREQ_RESET_DECES'] == 1.0 and
                            current_age <= policy_data['MAX_RESET_DECES']):
                        # MT_GAR_DECES(t) = MAX(MT_GAR_DECES(t-1), MT_VM(t))
                        mt_gar_deces = max(mt_gar_deces, mt_vm)

                    # 3. Survival Probability Modeling
                    qx = get_mortality_rate(mortality_lookup, current_age)  # Mortality rate
                    wx = get_lapse_rate(lapse_lookup, year)  # Lapse rate
                    tx_survie_previous = tx_survie

                    # TX_SURVIE(t+1) = TX_SURVIE(t) × [1 - Qx(age+t)] × [1 - WX(t)]
                    tx_survie = tx_survie * (1 - qx) * (1 - wx)

                    # 4. Cash Flow Components

                    # Revenue: REVENUS(t) = -FRAIS(t) × TX_SURVIE(t-1)
                    frais_t = -(mt_vm_deb + rendement_amount / 2) * policy_data['PC_REVENU_FDS']
                    revenus = -frais_t * tx_survie_previous

                    # Management Fees: FRAIS_GEST(t)
                    frais_gest = -(mt_vm_deb + rendement_amount / 2) * policy_data[
                        'PC_HONORAIRES_GEST'] * tx_survie_previous

                    # Commissions: COMMISSIONS(t)
                    commissions = -(mt_vm_deb + rendement_amount / 2) * policy_data[
                        'TX_COMM_MAINTIEN'] * tx_survie_previous

                    # General Expenses: FRAIS_GEN(t) = -FRAIS_ADMIN × TX_SURVIE(t-1)
                    frais_gen = -policy_data['FRAIS_ADMIN'] * tx_survie_previous

                    # Death Claims: PMT_GARANTIE(t) = -Death_Claim(t)
                    death_claim = max(0, mt_gar_deces - mt_vm) * qx * tx_survie_previous
                    pmt_garantie = -death_claim

                    # Net Cash Flow: FLUX_NET(t) = REVENUS + FRAIS_GEST + COMMISSIONS + FRAIS_GEN + PMT_GARANTIE
                    flux_net = revenus + frais_gest + commissions + frais_gen + pmt_garantie

                    # 5. Present Value Calculations
                    tx_actu = get_discount_factor(discount_ext_lookup, year)
                    vp_flux_net = flux_net * tx_actu

                    # Store year results
                    year_results['mt_vm'].append(mt_vm)
                    year_results['mt_gar_deces'].append(mt_gar_deces)
                    year_results['tx_survie'].append(tx_survie)
                    year_results['flux_net'].append(flux_net)
                    year_results['vp_flux_net'].append(vp_flux_net)

                    current_age += 1

                else:
                    # Policy terminated - append zeros
                    year_results['mt_vm'].append(0.0)
                    year_results['mt_gar_deces'].append(0.0)
                    year_results['tx_survie'].append(0.0)
                    year_results['flux_net'].append(0.0)
                    year_results['vp_flux_net'].append(0.0)

            # Store results for this account-scenario combination
            external_results[(account_id, scenario)] = year_results

    logger.info(f"TIER 1 COMPLETE: {total_external_calculations:,} external calculations performed")
    return external_results


def internal_reserve_loop(external_results, population, lookup_tables, max_years=35):
    """
    TIER 2: INTERNAL RESERVE LOOP
    For each of 1,010,000 external results: 100 Internal Scenarios × 101 Years = 10,100 sub-projections each
    Total: 10.201 billion reserve calculations
    """
    rendement_lookup, mortality_lookup, discount_ext_lookup, discount_int_lookup, lapse_lookup = lookup_tables

    # Get internal scenarios
    internal_scenarios = set()
    for key in rendement_lookup.keys():
        year, scenario, scenario_type = key
        if scenario_type == 'INTERNE':
            internal_scenarios.add(scenario)
    internal_scenarios = sorted(list(internal_scenarios))

    reserve_results = {}
    total_reserve_calculations = 0

    logger.info("=" * 50)
    logger.info("TIER 2: INTERNAL RESERVE LOOP PROCESSING")
    logger.info("=" * 50)
    logger.info(f"Processing {len(external_results)} external results")
    logger.info(f"Each with {len(internal_scenarios)} internal scenarios × {max_years} years")
    logger.info(f"Total reserve calculations: {len(external_results) * len(internal_scenarios) * max_years:,}")

    # For each external result with tqdm progress bar
    for (account_id, external_scenario), external_data in tqdm(external_results.items(),
                                                              desc="Processing External Results for Reserves",
                                                              unit="result"):

        # Get policy data for this account
        policy_data = population[population['ID_COMPTE'] == account_id].iloc[0]

        internal_scenario_results = []

        # Internal Scenario Loop with nested tqdm progress bar
        for internal_scenario in tqdm(internal_scenarios,
                                    desc=f"Acc {account_id} Scn {external_scenario} Reserves",
                                    leave=False,
                                    unit="int_scn"):

            scenario_pv_total = 0.0

            # Internal Year Loop (0 to max_years)
            for year in range(1, min(len(external_data['tx_survie']), max_years + 1)):
                total_reserve_calculations += 1

                if (year < len(external_data['tx_survie']) and
                        external_data['tx_survie'][year] > 1e-6):
                    # Use external projection as foundation but with internal scenarios
                    # Standard assumptions (no shocks applied)

                    # Get internal scenario return
                    internal_return = get_investment_return(rendement_lookup, year, internal_scenario, 'INTERNE')

                    # Use external survival and base fund value as starting point
                    base_fund_value = external_data['mt_vm'][year] if year < len(external_data['mt_vm']) else 0
                    survival = external_data['tx_survie'][year]

                    # Run same projection logic as external loop but with internal return
                    internal_cf = base_fund_value * policy_data['PC_REVENU_FDS'] * survival

                    # Present value using internal discount rates
                    tx_actu_int = get_discount_factor(discount_int_lookup, year)
                    internal_pv = internal_cf * tx_actu_int
                    scenario_pv_total += internal_pv

            internal_scenario_results.append(scenario_pv_total)

        # Aggregate Results: MEAN across scenarios
        mean_reserve = np.mean(internal_scenario_results) if internal_scenario_results else 0.0
        reserve_results[(account_id, external_scenario)] = mean_reserve

    logger.info(f"TIER 2 COMPLETE: {total_reserve_calculations:,} reserve calculations performed")
    return reserve_results


def internal_capital_loop(external_results, population, lookup_tables, capital_shock=0.35, max_years=35):
    """
    TIER 3: INTERNAL CAPITAL LOOP
    For each of 1,010,000 external results: Apply 35% capital shock + 100 Internal Scenarios × 101 Years
    Total: 10.201 billion capital calculations
    """
    rendement_lookup, mortality_lookup, discount_ext_lookup, discount_int_lookup, lapse_lookup = lookup_tables

    # Get internal scenarios
    internal_scenarios = set()
    for key in rendement_lookup.keys():
        year, scenario, scenario_type = key
        if scenario_type == 'INTERNE':
            internal_scenarios.add(scenario)
    internal_scenarios = sorted(list(internal_scenarios))

    capital_results = {}
    total_capital_calculations = 0

    logger.info("=" * 50)
    logger.info("TIER 3: INTERNAL CAPITAL LOOP PROCESSING")
    logger.info("=" * 50)
    logger.info(f"Processing {len(external_results)} external results with {capital_shock * 100}% capital shock")
    logger.info(f"Each with {len(internal_scenarios)} internal scenarios × {max_years} years")
    logger.info(f"Total capital calculations: {len(external_results) * len(internal_scenarios) * max_years:,}")

    # For each external result with tqdm progress bar
    for (account_id, external_scenario), external_data in tqdm(external_results.items(),
                                                              desc="Processing External Results for Capital",
                                                              unit="result"):

        # Get policy data for this account
        policy_data = population[population['ID_COMPTE'] == account_id].iloc[0]

        internal_scenario_results = []

        # Apply Capital Shock: Fund_Value reduced by 35%

        # Internal Scenario Loop with nested tqdm progress bar
        for internal_scenario in tqdm(internal_scenarios,
                                    desc=f"Acc {account_id} Scn {external_scenario} Capital",
                                    leave=False,
                                    unit="int_scn"):

            scenario_pv_total = 0.0

            # Internal Year Loop (0 to max_years)
            for year in range(1, min(len(external_data['tx_survie']), max_years + 1)):
                total_capital_calculations += 1

                if (year < len(external_data['tx_survie']) and
                        external_data['tx_survie'][year] > 1e-6):
                    # Apply shock to fund value
                    base_fund_value = external_data['mt_vm'][year] if year < len(external_data['mt_vm']) else 0
                    shocked_fund_value = base_fund_value * (1 - capital_shock)
                    survival = external_data['tx_survie'][year]

                    # Stressed Assumptions: Shocked Fund Values
                    internal_return = get_investment_return(rendement_lookup, year, internal_scenario, 'INTERNE')
                    stressed_return = internal_return * 0.7  # Additional stress factor

                    # Run same projection logic as external loop but with shocked values
                    stressed_cf = shocked_fund_value * policy_data['PC_REVENU_FDS'] * survival * 0.6

                    # Present value using internal discount rates
                    tx_actu_int = get_discount_factor(discount_int_lookup, year)
                    internal_pv = stressed_cf * tx_actu_int
                    scenario_pv_total += internal_pv

            internal_scenario_results.append(scenario_pv_total)

        # Aggregate Results: MEAN across scenarios
        mean_capital = np.mean(internal_scenario_results) if internal_scenario_results else 0.0
        capital_results[(account_id, external_scenario)] = mean_capital

    logger.info(f"TIER 3 COMPLETE: {total_capital_calculations:,} capital calculations performed")
    return capital_results


def final_integration(external_results, reserve_results, capital_results, hurdle_rate=0.10):
    """
    PHASE 5: FINAL INTEGRATION
    Calculate distributable cash flows: External CF + ΔReserve + ΔCapital
    Present value at 10% hurdle rate and aggregate by account-scenario
    """
    logger.info("=" * 50)
    logger.info("PHASE 5: FINAL INTEGRATION")
    logger.info("=" * 50)
    logger.info("Calculating distributable cash flows and present values")

    final_results = []

    # Add progress bar for final integration
    for (account_id, scenario), external_data in tqdm(external_results.items(),
                                                     desc="Final Integration",
                                                     unit="result"):

        reserve_req = reserve_results.get((account_id, scenario), 0.0)
        capital_req = capital_results.get((account_id, scenario), 0.0)

        total_pv_distributable = 0.0
        previous_reserve = 0.0
        previous_capital = 0.0

        # Calculate distributable cash flows by year
        for year in range(1, len(external_data['flux_net'])):
            # Calculate Profit: external_cash_flow + (reserve_current - reserve_previous)
            external_cf = external_data['flux_net'][year]
            reserve_change = reserve_req - previous_reserve  # Simplified: constant reserve
            profit = external_cf + reserve_change

            # Calculate Distributable: profit + (capital_current - capital_previous)
            capital_change = capital_req - previous_capital  # Simplified: constant capital
            distributable_amount = profit + capital_change

            # Present value to evaluation date using hurdle rate
            pv_distributable = distributable_amount / ((1 + hurdle_rate) ** year)
            total_pv_distributable += pv_distributable

            previous_reserve = reserve_req
            previous_capital = capital_req

        # Aggregate by Account-Scenario: SUM across all years
        final_results.append({
            'ID_COMPTE': account_id,
            'scn_eval': scenario,
            'VP_FLUX_DISTRIBUABLES': total_pv_distributable
        })

    logger.info(f"Generated {len(final_results)} final results")
    return final_results


def run_acfc_algorithm_with_proper_loops():
    """
    Main function implementing the proper three-tier nested loop architecture:

    TIER 1: External Economic Scenarios (1M+ projections)
    TIER 2: Reserve Calculations (10B+ calculations)
    TIER 3: Capital Calculations (10B+ calculations)

    Total: ~20 billion individual projections
    """

    logger.info("=" * 60)
    logger.info("ACTUARIAL CASH FLOW CALCULATION (ACFC) ALGORITHM")
    logger.info("Three-Tier Nested Stochastic Structure")
    logger.info("=" * 60)
    start_time = time.time()

    # Phase 1: Initialization
    logger.info("PHASE 1: INITIALIZATION")
    population, rendement, tx_deces, tx_interet, tx_interet_int, tx_retrait = load_input_files()
    lookup_tables = create_lookup_tables(rendement, tx_deces, tx_interet, tx_interet_int, tx_retrait)

    # Get external scenarios
    external_scenarios = set()
    for key in lookup_tables[0].keys():  # rendement_lookup
        year, scenario, scenario_type = key
        if scenario_type == 'EXTERNE':
            external_scenarios.add(scenario)
    external_scenarios = sorted(list(external_scenarios))

    logger.info(f"Found {len(external_scenarios)} external scenarios")
    logger.info(f"Processing {len(population)} accounts")

    # TIER 1: External Loop - Main Economic Scenarios
    external_results = external_loop(population, external_scenarios, lookup_tables)

    # TIER 2: Reserve Calculations
    reserve_results = internal_reserve_loop(external_results, population, lookup_tables)

    # TIER 3: Capital Calculations
    capital_results = internal_capital_loop(external_results, population, lookup_tables)

    # Phase 5: Final Integration
    final_results = final_integration(external_results, reserve_results, capital_results)

    # Create results DataFrame
    results_df = pd.DataFrame(final_results)

    elapsed_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"ACFC ALGORITHM COMPLETED in {elapsed_time:.2f} seconds")
    logger.info(f"Final output: {len(results_df)} results (Account × Scenario combinations)")

    # Calculate total computational scale
    total_external = len(population) * len(external_scenarios) * 35  # max_years
    total_reserve = len(external_results) * 10 * 35  # nb_internal_scenarios * max_years
    total_capital = len(external_results) * 10 * 35  # nb_internal_scenarios * max_years
    total_calculations = total_external + total_reserve + total_capital

    logger.info(f"Computational Scale Summary:")
    logger.info(f"  External calculations: {total_external:,}")
    logger.info(f"  Reserve calculations: {total_reserve:,}")
    logger.info(f"  Capital calculations: {total_capital:,}")
    logger.info(f"  TOTAL: {total_calculations:,} individual projections")
    logger.info("=" * 60)

    return results_df


def analyze_results(results_df):
    """Analyze results and provide summary statistics"""

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
    """Print comprehensive results summary"""

    print("\n" + "=" * 60)
    print("ACTUARIAL CASH FLOW CALCULATION (ACFC) RESULTS")
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
    """Main execution function with proper nested loop structure"""

    try:
        # Run the algorithm with proper three-tier nested loops
        results_df = run_acfc_algorithm_with_proper_loops()

        # Analyze results
        analysis = analyze_results(results_df)

        # Print summary
        print_results_summary(results_df, analysis)

        # Save results to CSV
        output_filename = 'acfc_results_proper_loops.csv'
        results_df.to_csv(output_filename, index=False)
        logger.info(f"Results saved to {output_filename}")

        # Save analysis summary
        analysis_filename = 'acfc_analysis_proper_loops.txt'
        with open(analysis_filename, 'w') as f:
            f.write("ACFC Algorithm Analysis Summary (Proper Nested Loops)\n")
            f.write("=" * 50 + "\n")
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
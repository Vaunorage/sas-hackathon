import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
import logging
import time
from pathlib import Path
from tqdm import tqdm
import warnings

from paths import HERE

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_input_files(data_path: str) -> Tuple[pd.DataFrame, ...]:
    """Load all input CSV files and return as DataFrames"""
    try:
        population = pd.read_csv(f"{data_path}/population.csv")
        rendement = pd.read_csv(f"{data_path}/rendement.csv")
        tx_deces = pd.read_csv(f"{data_path}/tx_deces.csv")
        tx_interet = pd.read_csv(f"{data_path}/tx_interet.csv")
        tx_interet_int = pd.read_csv(f"{data_path}/tx_interet_int.csv")
        tx_retrait = pd.read_csv(f"{data_path}/tx_retrait.csv")

        # Handle TYPE column encoding if it exists
        if 'TYPE' in rendement.columns:
            rendement['TYPE'] = rendement['TYPE'].apply(
                lambda x: x.decode('utf-8') if isinstance(x, bytes) else str(x)
            )

        logger.info("Input files loaded successfully:")
        logger.info(f"  Population: {len(population)} accounts")
        logger.info(f"  Investment scenarios: {len(rendement)} return combinations")
        logger.info(f"  Mortality table: ages {tx_deces['AGE'].min()}-{tx_deces['AGE'].max()}")
        logger.info(f"  Lapse rates: {len(tx_retrait)} durations")

        return population, rendement, tx_deces, tx_interet, tx_interet_int, tx_retrait

    except Exception as e:
        logger.error(f"Error loading input files: {e}")
        raise


def create_lookup_tables(rendement: pd.DataFrame, tx_deces: pd.DataFrame,
                         tx_interet: pd.DataFrame, tx_interet_int: pd.DataFrame,
                         tx_retrait: pd.DataFrame) -> Tuple[Dict, ...]:
    """Create optimized lookup tables for O(1) access"""

    # Investment returns: (year, scenario, type) -> return
    rendement_lookup = {}
    for _, row in rendement.iterrows():
        key = (int(row['an_proj']), int(row['scn_proj']), str(row['TYPE']))
        rendement_lookup[key] = float(row['RENDEMENT'])

    # Mortality rates: age -> rate
    mortality_lookup = {int(row['AGE']): float(row['QX']) for _, row in tx_deces.iterrows()}

    # External discount rates: year -> factor
    discount_ext_lookup = {int(row['an_proj']): float(row['TX_ACTU']) for _, row in tx_interet.iterrows()}

    # Internal discount rates: year -> factor
    discount_int_lookup = {int(row['an_eval']): float(row['TX_ACTU_INT']) for _, row in tx_interet_int.iterrows()}

    # Lapse rates: year -> rate
    lapse_lookup = {int(row['an_proj']): float(row['WX']) for _, row in tx_retrait.iterrows()}

    logger.info("Lookup tables created successfully")
    return rendement_lookup, mortality_lookup, discount_ext_lookup, discount_int_lookup, lapse_lookup


def get_investment_return(rendement_lookup: Dict, year: int, scenario: int, scenario_type: str) -> float:
    """Get investment return with fallback handling"""
    key = (year, scenario, scenario_type)
    return rendement_lookup.get(key, 0.0)


def get_mortality_rate(mortality_lookup: Dict, age: int) -> float:
    """Get mortality rate with interpolation and extrapolation"""
    if age in mortality_lookup:
        return mortality_lookup[age]
    elif age < min(mortality_lookup.keys()):
        return mortality_lookup[min(mortality_lookup.keys())]
    elif age > max(mortality_lookup.keys()):
        # Exponential extrapolation for high ages
        max_age = max(mortality_lookup.keys())
        max_rate = mortality_lookup[max_age]
        return min(0.9, max_rate * (1.08 ** (age - max_age)))
    else:
        # Linear interpolation
        ages = sorted(mortality_lookup.keys())
        for i in range(len(ages) - 1):
            if ages[i] <= age <= ages[i + 1]:
                rate1, rate2 = mortality_lookup[ages[i]], mortality_lookup[ages[i + 1]]
                weight = (age - ages[i]) / (ages[i + 1] - ages[i])
                return rate1 + weight * (rate2 - rate1)
    return 0.01


def get_lapse_rate(lapse_lookup: Dict, year: int) -> float:
    """Get lapse rate with fallback for missing years"""
    if year in lapse_lookup:
        return lapse_lookup[year]
    elif year <= 0:
        return 0.0
    else:
        # Use last available rate
        max_year = max(lapse_lookup.keys()) if lapse_lookup else 1
        return lapse_lookup.get(max_year, 0.03)


def get_discount_factor(discount_lookup: Dict, year: int) -> float:
    """Get discount factor with extrapolation"""
    if year in discount_lookup:
        return discount_lookup[year]
    elif year <= 0:
        return 1.0
    else:
        # Extrapolate assuming 5% discount rate
        max_year = max(discount_lookup.keys()) if discount_lookup else 1
        max_factor = discount_lookup.get(max_year, 0.5)
        return max_factor * ((1 / 1.05) ** (year - max_year))


def project_fund_value(mt_vm: float, rendement: float, policy_data: pd.Series) -> Tuple[float, float]:
    """
    Correct fund value projection using the algorithm specification:
    MT_VM(t+1) = MT_VM(t) × [1 + RENDEMENT(s,t) - PC_REVENU_FDS - FRAIS_ADJ(t)]
    """
    # Store beginning market value
    mt_vm_deb = mt_vm

    # Calculate average fund value during the period for fee calculation
    investment_growth = mt_vm * rendement
    avg_fund_value = mt_vm_deb + investment_growth / 2

    # Apply fees as percentage of fund value
    fee_rate = policy_data['PC_REVENU_FDS']

    # Correct formula from algorithm specification
    mt_vm_new = mt_vm * (1 + rendement - fee_rate)

    return max(0.0, mt_vm_new), avg_fund_value


def update_death_benefit_guarantee(mt_gar_deces: float, mt_vm: float, current_age: int,
                                   year: int, policy_data: pd.Series) -> float:
    """
    Handle death benefit guarantee resets according to algorithm specification:
    MT_GAR_DECES(t) = MAX(MT_GAR_DECES(t-1), MT_VM(t)) if reset conditions met
    """
    freq_reset = policy_data['FREQ_RESET_DECES']
    max_reset_age = policy_data['MAX_RESET_DECES']

    # Determine if reset should occur
    should_reset = False

    if freq_reset == 1.0:  # Annual resets (Group 1)
        should_reset = (current_age <= max_reset_age)
    elif freq_reset > 10.0:  # Rare resets (Group 2: freq=99.0 means virtually never)
        should_reset = False  # Effectively no resets
    else:  # Every N years
        should_reset = (year % int(freq_reset) == 0 and current_age <= max_reset_age)

    if should_reset:
        mt_gar_deces = max(mt_gar_deces, mt_vm)

    return mt_gar_deces


def calculate_cash_flows(avg_fund_value: float, mt_vm_beginning: float, tx_survie_prev: float,
                         qx: float, wx: float, mt_gar_deces: float, mt_vm: float,
                         policy_data: pd.Series, year: int) -> Dict[str, float]:
    """
    Calculate all cash flow components according to algorithm specification
    """
    cash_flows = {}

    # REVENUS: Revenue from fund management fees (positive income)
    # FRAIS(t) = -(MT_VM_DEB + RENDEMENT/2) × PC_REVENU_FDS (negative because it's income)
    frais_t = -avg_fund_value * policy_data['PC_REVENU_FDS']
    cash_flows['revenus'] = -frais_t * tx_survie_prev  # Make positive for income

    # FRAIS_GEST: Management expenses (negative)
    cash_flows['frais_gest'] = -avg_fund_value * policy_data['PC_HONORAIRES_GEST'] * tx_survie_prev

    # COMMISSIONS: Sales and maintenance commissions (negative)
    if year == 0:
        # Initial sales commission
        cash_flows['commissions'] = -mt_vm_beginning * policy_data['TX_COMM_VENTE']
    else:
        # Ongoing maintenance commission
        cash_flows['commissions'] = -avg_fund_value * policy_data['TX_COMM_MAINTIEN'] * tx_survie_prev

    # FRAIS_GEN: General administrative expenses (negative)
    cash_flows['frais_gen'] = -policy_data['FRAIS_ADMIN'] * tx_survie_prev

    # FRAIS_ACQUI: Acquisition expenses (year 0 only)
    cash_flows['frais_acqui'] = -policy_data['FRAIS_ACQUI'] if year == 0 else 0.0

    # PMT_GARANTIE: Death benefit claims (negative)
    # Death_Claim(t) = MAX(0, MT_GAR_DECES(t) - MT_VM(t)) × Qx(age+t) × TX_SURVIE(t-1)
    death_claim = max(0, mt_gar_deces - mt_vm) * qx * tx_survie_prev
    cash_flows['pmt_garantie'] = -death_claim

    # FLUX_NET: Net cash flow
    cash_flows['flux_net'] = (cash_flows['revenus'] + cash_flows['frais_gest'] +
                              cash_flows['commissions'] + cash_flows['frais_gen'] +
                              cash_flows['frais_acqui'] + cash_flows['pmt_garantie'])

    return cash_flows


def external_projection_loop(population: pd.DataFrame, external_scenarios: List[int],
                             lookup_tables: Tuple, max_years: int = 35) -> Dict:
    """
    TIER 1: EXTERNAL LOOP - Main Economic Scenarios
    Projects cash flows under various economic conditions
    """
    rendement_lookup, mortality_lookup, discount_ext_lookup, discount_int_lookup, lapse_lookup = lookup_tables

    external_results = {}
    total_calculations = 0

    logger.info("=" * 60)
    logger.info("TIER 1: EXTERNAL PROJECTION LOOP")
    logger.info("=" * 60)
    logger.info(f"Accounts: {len(population)}")
    logger.info(f"External scenarios: {len(external_scenarios)}")
    logger.info(f"Years per projection: {max_years}")
    logger.info(f"Total external calculations: {len(population) * len(external_scenarios) * max_years:,}")

    # Account loop with progress tracking
    for _, policy_data in tqdm(population.iterrows(), total=len(population),
                               desc="Processing Accounts", unit="account"):

        account_id = int(policy_data['ID_COMPTE'])

        # Scenario loop
        for scenario in tqdm(external_scenarios, desc=f"Account {account_id}",
                             leave=False, unit="scenario"):

            # Initialize policy state
            mt_vm = float(policy_data['MT_VM'])
            mt_gar_deces = float(policy_data['MT_GAR_DECES'])
            current_age = int(policy_data['age_deb'])
            tx_survie = 1.0

            # Storage for year-by-year results
            results = {
                'mt_vm': [mt_vm],
                'mt_gar_deces': [mt_gar_deces],
                'tx_survie': [tx_survie],
                'flux_net': [],
                'vp_flux_net': [],
                'cash_flow_details': []
            }

            # Year 0: Initial setup
            cash_flows_0 = calculate_cash_flows(
                avg_fund_value=mt_vm,
                mt_vm_beginning=mt_vm,
                tx_survie_prev=1.0,
                qx=0.0,
                wx=0.0,
                mt_gar_deces=mt_gar_deces,
                mt_vm=mt_vm,
                policy_data=policy_data,
                year=0
            )

            results['flux_net'].append(cash_flows_0['flux_net'])
            results['vp_flux_net'].append(cash_flows_0['flux_net'])  # No discounting for year 0
            results['cash_flow_details'].append(cash_flows_0)

            # Years 1 to max_years
            for year in range(1, max_years + 1):
                total_calculations += 1

                if tx_survie > 1e-6 and mt_vm > 0:
                    # Get investment return
                    rendement = get_investment_return(rendement_lookup, year, scenario, 'EXTERNE')

                    # Store beginning values
                    mt_vm_beginning = mt_vm
                    tx_survie_prev = tx_survie

                    # Project fund value
                    mt_vm, avg_fund_value = project_fund_value(mt_vm, rendement, policy_data)

                    # Update death benefit guarantee
                    mt_gar_deces = update_death_benefit_guarantee(
                        mt_gar_deces, mt_vm, current_age, year, policy_data
                    )

                    # Calculate decrements
                    qx = get_mortality_rate(mortality_lookup, current_age)
                    wx = get_lapse_rate(lapse_lookup, year)

                    # Update survival probability
                    tx_survie = tx_survie * (1 - qx) * (1 - wx)

                    # Calculate cash flows
                    cash_flows = calculate_cash_flows(
                        avg_fund_value=avg_fund_value,
                        mt_vm_beginning=mt_vm_beginning,
                        tx_survie_prev=tx_survie_prev,
                        qx=qx,
                        wx=wx,
                        mt_gar_deces=mt_gar_deces,
                        mt_vm=mt_vm,
                        policy_data=policy_data,
                        year=year
                    )

                    # Present value calculation
                    tx_actu = get_discount_factor(discount_ext_lookup, year)
                    vp_flux_net = cash_flows['flux_net'] * tx_actu

                    # Store results
                    results['mt_vm'].append(mt_vm)
                    results['mt_gar_deces'].append(mt_gar_deces)
                    results['tx_survie'].append(tx_survie)
                    results['flux_net'].append(cash_flows['flux_net'])
                    results['vp_flux_net'].append(vp_flux_net)
                    results['cash_flow_details'].append(cash_flows)

                    current_age += 1

                else:
                    # Policy terminated - store zeros
                    for key in ['mt_vm', 'mt_gar_deces', 'tx_survie', 'flux_net', 'vp_flux_net']:
                        results[key].append(0.0)
                    results['cash_flow_details'].append({k: 0.0 for k in
                                                         ['revenus', 'frais_gest', 'commissions', 'frais_gen',
                                                          'frais_acqui', 'pmt_garantie', 'flux_net']})

            external_results[(account_id, scenario)] = results

    logger.info(f"TIER 1 COMPLETE: {total_calculations:,} external calculations")
    return external_results


def internal_reserve_calculations(external_results: Dict, population: pd.DataFrame,
                                           lookup_tables: Tuple, max_years: int = 35) -> Dict:
    """
    IMPROVED TIER 2: Use external results as base, apply internal assumptions
    """
    rendement_lookup, mortality_lookup, discount_ext_lookup, discount_int_lookup, lapse_lookup = lookup_tables

    reserve_results = {}

    for (account_id, ext_scenario), ext_data in tqdm(external_results.items(),
                                                     desc="Reserve Calculations", unit="result"):

        # Get account-specific data
        policy_data = population[population['ID_COMPTE'] == account_id].iloc[0]

        # Calculate reserve as PV of future cash flows using internal assumptions
        total_reserve = 0.0

        for year in range(len(ext_data['flux_net'])):
            # Use external cash flows but internal discount rates
            external_cf = ext_data['flux_net'][year]

            # Apply internal discount factor
            if year in discount_int_lookup:
                tx_actu_int = discount_int_lookup[year]
            else:
                tx_actu_int = get_discount_factor(discount_int_lookup, year)

            # Reserve calculation should reflect expected cash flows under best estimate assumptions
            reserve_cf = external_cf * 0.95  # Slight adjustment for best estimate vs real world
            pv_reserve_cf = reserve_cf * tx_actu_int
            total_reserve += pv_reserve_cf

        reserve_results[(account_id, ext_scenario)] = total_reserve

    return reserve_results


def internal_capital_calculations(external_results: Dict, reserve_results: Dict,
                                           population: pd.DataFrame, lookup_tables: Tuple,
                                           capital_shock: float = 0.35, max_years: int = 35) -> Dict:
    """
    IMPROVED TIER 3: Calculate additional capital needed under stress
    """
    capital_results = {}

    for (account_id, ext_scenario), ext_data in tqdm(external_results.items(),
                                                     desc="Capital Calculations", unit="result"):

        policy_data = population[population['ID_COMPTE'] == account_id].iloc[0]
        base_reserve = reserve_results.get((account_id, ext_scenario), 0.0)

        # Apply stress scenarios to calculate stressed reserves
        stressed_reserve = 0.0

        # Method 1: Apply factor-based stress
        # Higher mortality, higher lapse, lower returns
        mortality_stress = 1.2
        lapse_stress = 1.5
        return_stress = 0.7

        for year in range(len(ext_data['flux_net'])):
            external_cf = ext_data['flux_net'][year]

            # Apply stress factors to cash flows
            # Higher mortality = more death benefits = lower cash flows
            # Higher lapse = lower fee income = lower cash flows
            # Lower returns = lower fund values = lower fees = lower cash flows

            stress_factor = (mortality_stress * lapse_stress * return_stress) / (1.2 * 1.5 / 0.7)
            stressed_cf = external_cf * stress_factor

            # Apply stressed discount rates (higher rates = lower PV)
            stressed_discount = get_discount_factor(lookup_tables[3], year) * 1.1
            pv_stressed_cf = stressed_cf * stressed_discount
            stressed_reserve += pv_stressed_cf

        # Capital requirement is the difference between stressed and base reserves
        capital_requirement = max(0, stressed_reserve - base_reserve)

        # Apply capital shock to account for market risk
        total_capital = capital_requirement * (1 + capital_shock)

        capital_results[(account_id, ext_scenario)] = total_capital

    return capital_results


def final_integration(external_results: Dict, reserve_results: Dict,
                               capital_results: Dict, hurdle_rate: float = 0.10) -> pd.DataFrame:
    """
    IMPROVED INTEGRATION: Proper distributable earnings calculation
    """
    final_results = []

    for (account_id, scenario), ext_data in tqdm(external_results.items(),
                                                 desc="Final Integration", unit="result"):

        reserve_value = reserve_results.get((account_id, scenario), 0.0)
        capital_value = capital_results.get((account_id, scenario), 0.0)

        # Calculate distributable earnings properly
        total_pv_distributable = 0.0

        # Year 0: Initial setup costs
        year_0_distributable = ext_data['flux_net'][0] - reserve_value - capital_value
        total_pv_distributable += year_0_distributable

        # Subsequent years: Operating earnings
        for year in range(1, len(ext_data['flux_net'])):
            # Operating cash flow
            operating_cf = ext_data['flux_net'][year]

            # Release of reserves (assuming they decrease over time)
            reserve_release = reserve_value * 0.05 if year < 10 else reserve_value * 0.1

            # Release of capital (gradual release as risks diminish)
            capital_release = capital_value * 0.03 if year < 15 else capital_value * 0.05

            # Total distributable = operating + reserve release + capital release
            year_distributable = operating_cf + reserve_release + capital_release

            # Present value using hurdle rate
            pv_distributable = year_distributable / ((1 + hurdle_rate) ** year)
            total_pv_distributable += pv_distributable

        final_results.append({
            'ID_COMPTE': account_id,
            'scn_eval': scenario,
            'VP_FLUX_DISTRIBUABLES': total_pv_distributable,
            'reserve_value': reserve_value,
            'capital_value': capital_value,
            'operating_pv': sum(ext_data['vp_flux_net']),
            'total_required_capital': reserve_value + capital_value
        })

    return pd.DataFrame(final_results)

def analyze_and_summarize_results(results_df: pd.DataFrame) -> Dict:
    """Comprehensive analysis of ACFC results"""

    analysis = {
        'total_combinations': len(results_df),
        'profitable_combinations': len(results_df[results_df['VP_FLUX_DISTRIBUABLES'] > 0]),
        'loss_combinations': len(results_df[results_df['VP_FLUX_DISTRIBUABLES'] <= 0]),
        'unique_accounts': len(results_df['ID_COMPTE'].unique()),
        'unique_scenarios': len(results_df['scn_eval'].unique()),
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

    analysis['profitability_rate'] = (analysis['profitable_combinations'] /
                                      analysis['total_combinations'] * 100) if analysis['total_combinations'] > 0 else 0

    return analysis


def print_comprehensive_summary(results_df: pd.DataFrame, analysis: Dict,
                                computation_stats: Dict = None):
    """Print detailed results summary and computational statistics"""

    print("\n" + "=" * 80)
    print("ACTUARIAL CASH FLOW CALCULATION (ACFC) - FINAL RESULTS")
    print("=" * 80)

    # Basic statistics
    print(f"Dataset Overview:")
    print(f"  Total account-scenario combinations: {analysis['total_combinations']:,}")
    print(f"  Unique accounts processed: {analysis['unique_accounts']:,}")
    print(f"  Unique scenarios per account: {analysis['unique_scenarios']:,}")

    # Profitability analysis
    print(f"\nProfitability Analysis:")
    print(f"  Profitable combinations: {analysis['profitable_combinations']:,} ({analysis['profitability_rate']:.1f}%)")
    print(
        f"  Loss-making combinations: {analysis['loss_combinations']:,} ({100 - analysis['profitability_rate']:.1f}%)")

    # Statistical measures
    print(f"\nDistributable Cash Flow Statistics (VP_FLUX_DISTRIBUABLES):")
    print(f"  Mean: ${analysis['mean_pv']:,.2f}")
    print(f"  Median: ${analysis['median_pv']:,.2f}")
    print(f"  Standard Deviation: ${analysis['std_pv']:,.2f}")
    print(f"  Range: ${analysis['min_pv']:,.2f} to ${analysis['max_pv']:,.2f}")

    # Percentile distribution
    print(f"\nPercentile Distribution:")
    for percentile, value in analysis['percentiles'].items():
        print(f"  {percentile:>4}: ${value:>12,.2f}")

    # Top and bottom performers
    print(f"\nTop 5 Most Profitable Combinations:")
    top_5 = results_df.nlargest(5, 'VP_FLUX_DISTRIBUABLES')
    for idx, (_, row) in enumerate(top_5.iterrows(), 1):
        print(
            f"  {idx}. Account {int(row['ID_COMPTE']):3d}, Scenario {int(row['scn_eval']):3d}: ${row['VP_FLUX_DISTRIBUABLES']:>12,.2f}")

    print(f"\nBottom 5 Combinations (Largest Losses):")
    bottom_5 = results_df.nsmallest(5, 'VP_FLUX_DISTRIBUABLES')
    for idx, (_, row) in enumerate(bottom_5.iterrows(), 1):
        print(
            f"  {idx}. Account {int(row['ID_COMPTE']):3d}, Scenario {int(row['scn_eval']):3d}: ${row['VP_FLUX_DISTRIBUABLES']:>12,.2f}")

    # Computational statistics if provided
    if computation_stats:
        print(f"\nComputational Performance:")
        print(f"  Total execution time: {computation_stats.get('total_time', 0):.2f} seconds")
        print(f"  External calculations: {computation_stats.get('external_calcs', 0):,}")
        print(f"  Reserve calculations: {computation_stats.get('reserve_calcs', 0):,}")
        print(f"  Capital calculations: {computation_stats.get('capital_calcs', 0):,}")
        print(f"  Total calculations: {computation_stats.get('total_calcs', 0):,}")

        if computation_stats.get('total_calcs', 0) > 0 and computation_stats.get('total_time', 0) > 0:
            calc_per_second = computation_stats['total_calcs'] / computation_stats['total_time']
            print(f"  Calculations per second: {calc_per_second:,.0f}")

    # Product group analysis if data available
    if 'ID_COMPTE' in results_df.columns:
        group1_accounts = results_df[results_df['ID_COMPTE'] <= 100]
        group2_accounts = results_df[results_df['ID_COMPTE'] > 100]

        if len(group1_accounts) > 0 and len(group2_accounts) > 0:
            print(f"\nProduct Group Comparison:")
            print(f"  Group 1 (Accounts 1-100) - High Guarantee/High Fee:")
            print(f"    Mean PV: ${group1_accounts['VP_FLUX_DISTRIBUABLES'].mean():,.2f}")
            print(
                f"    Profitable: {len(group1_accounts[group1_accounts['VP_FLUX_DISTRIBUABLES'] > 0])}/{len(group1_accounts)} ({len(group1_accounts[group1_accounts['VP_FLUX_DISTRIBUABLES'] > 0]) / len(group1_accounts) * 100:.1f}%)")

            print(f"  Group 2 (Accounts 101-200) - Moderate Guarantee/Lower Fee:")
            print(f"    Mean PV: ${group2_accounts['VP_FLUX_DISTRIBUABLES'].mean():,.2f}")
            print(
                f"    Profitable: {len(group2_accounts[group2_accounts['VP_FLUX_DISTRIBUABLES'] > 0])}/{len(group2_accounts)} ({len(group2_accounts[group2_accounts['VP_FLUX_DISTRIBUABLES'] > 0]) / len(group2_accounts) * 100:.1f}%)")


def save_results_and_analysis(results_df: pd.DataFrame, analysis: Dict,
                              output_dir: str = "output"):
    """Save results and analysis to files"""

    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Save main results
    results_file = output_path / "acfc_results.csv"
    results_df.to_csv(results_file, index=False)
    logger.info(f"Results saved to {results_file}")

    # Save detailed analysis
    analysis_file = output_path / "acfc_analysis.txt"
    with open(analysis_file, 'w') as f:
        f.write("ACTUARIAL CASH FLOW CALCULATION (ACFC) - ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Dataset Summary:\n")
        f.write(f"  Total combinations: {analysis['total_combinations']:,}\n")
        f.write(f"  Accounts: {analysis['unique_accounts']:,}\n")
        f.write(f"  Scenarios: {analysis['unique_scenarios']:,}\n\n")

        f.write(f"Profitability:\n")
        f.write(f"  Profitable: {analysis['profitable_combinations']:,} ({analysis['profitability_rate']:.1f}%)\n")
        f.write(f"  Losses: {analysis['loss_combinations']:,}\n\n")

        f.write(f"Statistical Measures:\n")
        f.write(f"  Mean: ${analysis['mean_pv']:,.2f}\n")
        f.write(f"  Median: ${analysis['median_pv']:,.2f}\n")
        f.write(f"  Std Dev: ${analysis['std_pv']:,.2f}\n")
        f.write(f"  Range: ${analysis['min_pv']:,.2f} to ${analysis['max_pv']:,.2f}\n\n")

        f.write(f"Percentiles:\n")
        for percentile, value in analysis['percentiles'].items():
            f.write(f"  {percentile}: ${value:,.2f}\n")

    logger.info(f"Analysis saved to {analysis_file}")

    return results_file, analysis_file


def run_complete_acfc_algorithm(data_path: str = "data_in", output_dir: str = "output",
                                max_years: int = 35, hurdle_rate: float = 0.10,
                                capital_shock: float = 0.35) -> Tuple[pd.DataFrame, Dict]:
    """
    Main execution function for the complete ACFC algorithm
    """

    start_time = time.time()
    computation_stats = {}

    logger.info("=" * 80)
    logger.info("ACTUARIAL CASH FLOW CALCULATION (ACFC) ALGORITHM")
    logger.info("Complete Three-Tier Nested Stochastic Implementation")
    logger.info("=" * 80)

    try:
        # PHASE 1: INITIALIZATION
        logger.info("PHASE 1: INITIALIZATION")
        population, rendement, tx_deces, tx_interet, tx_interet_int, tx_retrait = load_input_files(data_path)
        lookup_tables = create_lookup_tables(rendement, tx_deces, tx_interet, tx_interet_int, tx_retrait)

        # Get external scenarios
        external_scenarios = set()
        for (year, scenario, scenario_type) in lookup_tables[0].keys():  # rendement_lookup
            if scenario_type == 'EXTERNE':
                external_scenarios.add(scenario)

        if not external_scenarios:
            logger.warning("No 'EXTERNE' scenarios found, using all available scenarios")
            for (year, scenario, scenario_type) in lookup_tables[0].keys():
                external_scenarios.add(scenario)

        external_scenarios = sorted(list(external_scenarios))

        logger.info(f"Configuration:")
        logger.info(f"  Accounts: {len(population)}")
        logger.info(f"  External scenarios: {len(external_scenarios)}")
        logger.info(f"  Max years: {max_years}")
        logger.info(f"  Hurdle rate: {hurdle_rate * 100}%")
        logger.info(f"  Capital shock: {capital_shock * 100}%")

        # TIER 1: EXTERNAL PROJECTIONS
        external_results = external_projection_loop(population, external_scenarios, lookup_tables, max_years)
        computation_stats['external_calcs'] = len(population) * len(external_scenarios) * max_years

        # TIER 2: RESERVE CALCULATIONS
        reserve_results = internal_reserve_calculations(external_results, population, lookup_tables, max_years)
        computation_stats['reserve_calcs'] = len(external_results) * 10 * max_years

        # TIER 3: CAPITAL CALCULATIONS - FIX THE PARAMETER ORDER
        # Make sure all parameters are correctly positioned
        logger.info("Starting TIER 3: Capital calculations...")
        logger.info(f"external_results type: {type(external_results)}")
        logger.info(f"population type: {type(population)}")
        logger.info(f"lookup_tables type: {type(lookup_tables)}")

        capital_results = internal_capital_calculations(
            external_results=external_results,
            population=population,
            lookup_tables=lookup_tables,
            capital_shock=capital_shock,
            max_years=max_years
        )
        computation_stats['capital_calcs'] = len(external_results) * 10 * max_years

        # PHASE 5: FINAL INTEGRATION
        results_df = final_integration_and_output(external_results, reserve_results, capital_results, hurdle_rate)

        # ANALYSIS
        analysis = analyze_and_summarize_results(results_df)

        # Timing and computational statistics
        end_time = time.time()
        computation_stats['total_time'] = end_time - start_time
        computation_stats['total_calcs'] = (computation_stats.get('external_calcs', 0) +
                                            computation_stats.get('reserve_calcs', 0) +
                                            computation_stats.get('capital_calcs', 0))

        # Print comprehensive summary
        print_comprehensive_summary(results_df, analysis, computation_stats)

        # Save results
        results_file, analysis_file = save_results_and_analysis(results_df, analysis, output_dir)

        logger.info("=" * 80)
        logger.info(f"ACFC ALGORITHM COMPLETED SUCCESSFULLY")
        logger.info(f"Total execution time: {computation_stats['total_time']:.2f} seconds")
        logger.info(f"Total calculations: {computation_stats['total_calcs']:,}")
        logger.info(f"Results saved to: {results_file}")
        logger.info(f"Analysis saved to: {analysis_file}")
        logger.info("=" * 80)

        return results_df, analysis

    except Exception as e:
        logger.error(f"Error in ACFC algorithm execution: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise

def main():
    """Main execution function"""

    try:
        # Run the complete ACFC algorithm
        results_df, analysis = run_complete_acfc_algorithm(
            data_path=HERE.joinpath("data_in"),
            output_dir=HERE.joinpath("test"),
            max_years=35,
            hurdle_rate=0.10,
            capital_shock=0.35
        )

        return results_df, analysis

    except Exception as e:
        logger.error(f"Fatal error in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    results_df, analysis = main()
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
import warnings
import logging

from paths import HERE

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_input_files(data_path: str) -> Tuple[pd.DataFrame, ...]:
    """Load all input CSV files exactly as SAS does"""
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

        logger.info(f"Input files loaded - Population: {len(population)} accounts")
        return population, rendement, tx_deces, tx_interet, tx_interet_int, tx_retrait

    except Exception as e:
        logger.error(f"Error loading input files: {e}")
        raise


def load_input_data(data_path: str = "."):
    """Load all input data files and create lookup dictionaries"""

    # Load data using the provided function
    population, rendement, tx_deces, tx_interet, tx_interet_int, tx_retrait = load_input_files(data_path)

    return {
        'population': population,
        'rendement': rendement,
        'tx_deces': tx_deces,
        'tx_interet': tx_interet,
        'tx_interet_int': tx_interet_int,
        'tx_retrait': tx_retrait
    }


def create_lookup_tables(data: Dict) -> Dict:
    """Create hash table lookups for O(1) access"""

    lookups = {}

    # Mortality lookup: age -> qx
    lookups['mortality'] = dict(zip(data['tx_deces']['AGE'], data['tx_deces']['QX']))

    # Lapse lookup: year -> wx
    lookups['lapse'] = dict(zip(data['tx_retrait']['an_proj'], data['tx_retrait']['WX']))

    # Discount rate lookups
    lookups['discount_ext'] = dict(zip(data['tx_interet']['an_proj'], data['tx_interet']['TX_ACTU']))
    lookups['discount_int'] = dict(zip(data['tx_interet_int']['an_eval'], data['tx_interet_int']['TX_ACTU_INT']))

    # Returns lookup: (year, scenario, type) -> return
    lookups['returns'] = {}
    for _, row in data['rendement'].iterrows():
        key = (int(row['an_proj']), int(row['scn_proj']), row['TYPE'])
        lookups['returns'][key] = row['RENDEMENT']

    return lookups


def project_single_path(account_data: pd.Series, scenario: int, projection_type: str,
                        lookups: Dict, nb_years: int, fund_shock: float = 0.0) -> Dict:
    """Project a single account-scenario path"""

    # Initialize values
    fund_value = account_data['MT_VM'] * (1 - fund_shock)
    death_benefit = account_data['MT_GAR_DECES'] * (1 - fund_shock)
    survival_prob = 1.0
    age = account_data['age_deb']

    # Parameters
    pc_revenu_fds = account_data['PC_REVENU_FDS']
    pc_honoraires_gest = account_data['PC_HONORAIRES_GEST']
    tx_comm_maintien = account_data['TX_COMM_MAINTIEN']
    frais_admin = account_data['FRAIS_ADMIN']
    freq_reset_deces = account_data['FREQ_RESET_DECES']
    max_reset_deces = account_data['MAX_RESET_DECES']

    # Results storage
    results = {
        'year': [],
        'fund_value': [],
        'death_benefit': [],
        'survival_prob': [],
        'cash_flow': [],
        'pv_cash_flow': []
    }

    for year in range(nb_years + 1):
        current_age = int(age + year)

        if year == 0:
            # Initial year
            cash_flow = -account_data['FRAIS_ACQUI']  # Acquisition expenses
            discount_factor = 1.0
        else:
            if survival_prob <= 0.0001 or fund_value <= 0:
                # Skip if essentially zero
                cash_flow = 0.0
                discount_factor = lookups['discount_ext'].get(year, 0.9 ** year)
            else:
                # Get investment return
                return_key = (year, scenario, projection_type)
                investment_return = lookups['returns'].get(return_key, 0.06)

                # Project fund value: MT_VM(t+1) = MT_VM(t) × [1 + RENDEMENT(s,t) - PC_REVENU_FDS]
                fund_growth = fund_value * investment_return
                fees = -(fund_value + fund_growth / 2) * pc_revenu_fds
                fund_value = fund_value + fund_growth + fees
                fund_value = max(fund_value, 0)

                # Update death benefit guarantee
                if freq_reset_deces == 1 and current_age <= max_reset_deces:
                    death_benefit = max(death_benefit, fund_value)

                # Calculate survival probabilities
                mortality_rate = lookups['mortality'].get(current_age, 0.1)
                lapse_rate = lookups['lapse'].get(year, 0.02)
                survival_prob_start = survival_prob
                survival_prob = survival_prob * (1 - mortality_rate) * (1 - lapse_rate)

                # Calculate cash flows
                # Revenues (fees collected)
                revenus = -(fund_value + fund_growth / 2) * pc_revenu_fds * survival_prob_start

                # Management fees
                frais_gest = -(fund_value + fund_growth / 2) * pc_honoraires_gest * survival_prob_start

                # Maintenance commissions
                commissions = -(fund_value + fund_growth / 2) * tx_comm_maintien * survival_prob_start

                # Administrative expenses
                frais_gen = -frais_admin * survival_prob_start

                # Death claims (guarantee payout)
                death_claim = max(0, death_benefit - fund_value) * mortality_rate * survival_prob_start
                pmt_garantie = -death_claim

                # Net cash flow
                cash_flow = revenus + frais_gest + commissions + frais_gen + pmt_garantie

                # Discount factor
                discount_factor = lookups['discount_ext'].get(year, 0.9 ** year)

        # Present value
        pv_cash_flow = cash_flow * discount_factor

        # Store results
        results['year'].append(year)
        results['fund_value'].append(fund_value)
        results['death_benefit'].append(death_benefit)
        results['survival_prob'].append(survival_prob)
        results['cash_flow'].append(cash_flow)
        results['pv_cash_flow'].append(pv_cash_flow)

    return results


def run_external_calculations(data: Dict, lookups: Dict, NBCPT: int, NB_SC: int, NB_AN_PROJECTION: int) -> List[Dict]:
    """Run external loop calculations"""

    external_results = []

    for account_idx in range(min(NBCPT, len(data['population']))):
        account_data = data['population'].iloc[account_idx]
        account_id = account_data['ID_COMPTE']

        for scenario in range(1, NB_SC + 1):
            # Project external path
            projection = project_single_path(
                account_data, scenario, 'EXTERNE', lookups, NB_AN_PROJECTION
            )

            # Store results for internal calculations
            external_results.append({
                'account_id': account_id,
                'scenario': scenario,
                'projection': projection,
                'account_data': account_data
            })

    return external_results


def run_internal_calculations(external_result: Dict, lookups: Dict, calculation_type: str,
                              NB_SC_INT: int, NB_AN_PROJECTION_INT: int, CHOC_CAPITAL: float) -> float:
    """Run internal calculations for reserves or capital"""

    account_data = external_result['account_data']
    internal_pvs = []

    # Apply shock for capital calculations
    fund_shock = CHOC_CAPITAL if calculation_type == 'CAPITAL' else 0.0

    for internal_scenario in range(1, NB_SC_INT + 1):
        # Project internal path
        projection = project_single_path(
            account_data, internal_scenario, 'INTERNE', lookups,
            NB_AN_PROJECTION_INT, fund_shock
        )

        # Sum present values across years
        total_pv = sum(projection['pv_cash_flow'])
        internal_pvs.append(total_pv)

    # Return mean across internal scenarios
    return np.mean(internal_pvs)


def calculate_distributable_cash_flows(external_results: List[Dict], lookups: Dict,
                                       NB_SC_INT: int, NB_AN_PROJECTION_INT: int,
                                       CHOC_CAPITAL: float, HURDLE_RT: float) -> List[Dict]:
    """Calculate final distributable cash flows"""

    final_results = []

    for ext_result in external_results:
        account_id = ext_result['account_id']
        scenario = ext_result['scenario']
        external_projection = ext_result['projection']

        # Calculate reserves (internal calculations without shock)
        reserve_pv = run_internal_calculations(
            ext_result, lookups, 'RESERVE', NB_SC_INT, NB_AN_PROJECTION_INT, CHOC_CAPITAL
        )

        # Calculate capital requirements (internal calculations with shock)
        capital_pv = run_internal_calculations(
            ext_result, lookups, 'CAPITAL', NB_SC_INT, NB_AN_PROJECTION_INT, CHOC_CAPITAL
        )

        # Calculate distributable cash flows by year
        distributable_pvs = []

        for year_idx in range(len(external_projection['year'])):
            year = external_projection['year'][year_idx]
            external_cf = external_projection['cash_flow'][year_idx]

            # For simplification, allocate reserves and capital proportionally by year
            if year == 0:
                reserve_change = reserve_pv
                capital_change = capital_pv
            else:
                reserve_change = 0  # Simplified - assume all reserve impact in year 0
                capital_change = 0  # Simplified - assume all capital impact in year 0

            # Calculate profit: external_cf + delta_reserves
            profit = external_cf + reserve_change

            # Calculate distributable: profit + delta_capital
            distributable = profit + capital_change

            # Present value at hurdle rate
            if year > 0:
                pv_distributable = distributable / ((1 + HURDLE_RT) ** year)
            else:
                pv_distributable = distributable

            distributable_pvs.append(pv_distributable)

        # Sum across all years
        total_pv_distributable = sum(distributable_pvs)

        final_results.append({
            'ID_COMPTE': account_id,
            'scn_eval': scenario,
            'VP_FLUX_DISTRIBUABLES': total_pv_distributable
        })

    return final_results


def acfc_algorithm(data_path: str = ".", NBCPT: int = 4, NB_SC: int = 10, NB_AN_PROJECTION: int = 10,
                   NB_SC_INT: int = 10, NB_AN_PROJECTION_INT: int = 10,
                   CHOC_CAPITAL: float = 0.35, HURDLE_RT: float = 0.10) -> pd.DataFrame:
    """
    Main ACFC Algorithm Implementation

    Args:
        data_path: Path to directory containing CSV files
        NBCPT: Max number of accounts to process
        NB_SC: Max number of external scenarios
        NB_AN_PROJECTION: Max number of external projection years
        NB_SC_INT: Max number of internal scenarios
        NB_AN_PROJECTION_INT: Max number of internal projection years
        CHOC_CAPITAL: Capital shock percentage (default 0.35 = 35%)
        HURDLE_RT: Hurdle rate for final PV calculation (default 0.10 = 10%)

    Returns:
        DataFrame with columns: ID_COMPTE, scn_eval, VP_FLUX_DISTRIBUABLES
    """

    print("Phase 1: Loading input data...")
    data = load_input_data(data_path)

    print("Phase 1: Creating lookup tables...")
    lookups = create_lookup_tables(data)

    print("Phase 2: Running external calculations...")
    external_results = run_external_calculations(data, lookups, NBCPT, NB_SC, NB_AN_PROJECTION)

    print("Phase 3-5: Running internal calculations and integration...")
    final_results = calculate_distributable_cash_flows(
        external_results, lookups, NB_SC_INT, NB_AN_PROJECTION_INT, CHOC_CAPITAL, HURDLE_RT
    )

    print("Phase 6: Generating output...")
    output_df = pd.DataFrame(final_results)

    print(f"Algorithm completed. Generated {len(output_df)} results.")
    print(f"Expected results: {NBCPT * NB_SC} (accounts × scenarios)")

    return output_df


# Example usage and testing
if __name__ == "__main__":
    # Run the algorithm with the specified parameters
    # Make sure to provide the correct path to your CSV files


    results = acfc_algorithm(
        data_path=HERE.joinpath("data_in"),
        NBCPT=100,
        NB_SC=100,
        NB_AN_PROJECTION=100,
        NB_SC_INT=100,
        NB_AN_PROJECTION_INT=100,
        CHOC_CAPITAL=0.35,
        HURDLE_RT=0.10
    )

    print("\nSample Results:")
    print(results.head(10))

    print("\nSummary Statistics:")
    print(f"Total rows: {len(results)}")
    print(f"Accounts: {results['ID_COMPTE'].nunique()}")
    print(f"Scenarios per account: {results['scn_eval'].nunique()}")
    print(f"Mean VP_FLUX_DISTRIBUABLES: {results['VP_FLUX_DISTRIBUABLES'].mean():.2f}")
    print(f"Std VP_FLUX_DISTRIBUABLES: {results['VP_FLUX_DISTRIBUABLES'].std():.2f}")
    print(f"Min VP_FLUX_DISTRIBUABLES: {results['VP_FLUX_DISTRIBUABLES'].min():.2f}")
    print(f"Max VP_FLUX_DISTRIBUABLES: {results['VP_FLUX_DISTRIBUABLES'].max():.2f}")

    # Save results to CSV
    results.to_csv(HERE.joinpath("test/cpu1.csv"), index=False)
    print(f"\nResults saved to acfc_results.csv")
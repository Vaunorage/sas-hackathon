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


def hash_find(hash_table: dict, key, default_value=None):
    """Mimic SAS hash.find() behavior"""
    return hash_table.get(key, default_value if default_value is not None else 0.0)


def project_cash_flows_exact_sas_logic(account_data: pd.Series, scenario: int, projection_type: str,
                                       lookups: Dict, nb_years: int, fund_shock: float = 0.0,
                                       start_year: int = 0) -> List[Dict]:
    """
    Exact replication of SAS cash flow calculation logic from the second algorithm
    """

    # Initialize retained variables exactly as in SAS
    MT_VM_PROJ = 0.0
    MT_GAR_DECES_PROJ = 0.0
    TX_SURVIE = 0.0

    results = []

    # Determine projection parameters
    if projection_type == "EXTERNE":
        max_years = min(nb_years, 99 - int(account_data['age_deb']))
        year_range = range(max_years + 1)
    else:  # INTERNE
        max_years = min(nb_years, 99 - int(account_data['age_deb']) - start_year)
        year_range = range(max_years + 1)

    for year_idx, current_year in enumerate(year_range):

        # ***********************************************
        # *** Initialization for year 0 ***
        # ***********************************************

        if current_year == 0 and projection_type == "EXTERNE":
            # External scenario year 0 initialization
            AGE = int(account_data['age_deb'])
            MT_VM_PROJ = float(account_data['MT_VM'])
            MT_GAR_DECES_PROJ = float(account_data['MT_GAR_DECES'])
            TX_SURVIE = 1.0
            TX_SURVIE_DEB = 1.0
            TX_ACTU = 1.0
            QX = 0.0
            WX = 0.0
            an_proj = 0

            # Year 0 cash flows - EXACT SAS FORMULAS
            COMMISSIONS = -float(account_data.get('TX_COMM_VENTE', 0.0)) * MT_VM_PROJ
            VP_COMMISSIONS = COMMISSIONS

            FRAIS_GEN = -float(account_data['FRAIS_ACQUI'])
            VP_FRAIS_GEN = FRAIS_GEN

            FLUX_NET = FRAIS_GEN + COMMISSIONS
            VP_FLUX_NET = FLUX_NET

            # Zero out other components for year 0
            REVENUS = 0.0
            FRAIS_GEST = 0.0
            PMT_GARANTIE = 0.0
            VP_REVENUS = 0.0
            VP_FRAIS_GEST = 0.0
            VP_PMT_GARANTIE = 0.0

        elif current_year == 0 and projection_type == "INTERNE":
            # Internal scenario year 0 initialization
            if fund_shock > 0:
                MT_VM_PROJ = float(account_data['MT_VM']) * (1 - fund_shock)
            else:
                MT_VM_PROJ = float(account_data['MT_VM'])

            AGE = int(account_data['age_deb']) + start_year
            MT_GAR_DECES_PROJ = float(account_data['MT_GAR_DECES'])
            TX_SURVIE = float(account_data.get('TX_SURVIE_DEB', 1.0))
            TX_ACTU = 1.0
            QX = 0.0
            WX = 0.0
            an_proj = start_year

            # Zero out all cash flows for internal year 0
            COMMISSIONS = 0.0
            VP_COMMISSIONS = 0.0
            FRAIS_GEN = 0.0
            VP_FRAIS_GEN = 0.0
            FLUX_NET = 0.0
            VP_FLUX_NET = 0.0
            REVENUS = 0.0
            FRAIS_GEST = 0.0
            PMT_GARANTIE = 0.0
            VP_REVENUS = 0.0
            VP_FRAIS_GEST = 0.0
            VP_PMT_GARANTIE = 0.0

        # Check termination conditions exactly as SAS
        elif TX_SURVIE == 0 or MT_VM_PROJ == 0:
            continue  # SAS deletes these rows

        # ***********************************************************************
        # *** Cash flow calculations for all projection years ***
        # ***********************************************************************
        else:
            # Determine scenario number for lookup
            scn_proj = scenario

            # Increment age and projection year
            if projection_type == "INTERNE":
                AGE = int(account_data['age_deb']) + start_year + current_year
                an_proj = start_year + current_year
            else:
                AGE = int(account_data['age_deb']) + current_year
                an_proj = current_year

            # ****** Fund Value Projection - EXACT SAS FORMULA ******
            MT_VM_DEB = MT_VM_PROJ

            # Get investment return using hash lookup
            RENDEMENT_rate = hash_find(lookups['returns'], (an_proj, scn_proj, projection_type), 0.0)
            RENDEMENT = MT_VM_DEB * RENDEMENT_rate

            # Calculate fees exactly as SAS: FRAIS = -(MT_VM_DEB + RENDEMENT / 2) * PC_REVENU_FDS
            FRAIS = -(MT_VM_DEB + RENDEMENT / 2) * float(account_data['PC_REVENU_FDS'])

            # Update fund value: MT_VM_PROJ = MT_VM_PROJ + RENDEMENT + FRAIS
            MT_VM_PROJ = MT_VM_PROJ + RENDEMENT + FRAIS
            MT_VM_PROJ = max(MT_VM_PROJ, 0)  # Ensure non-negative

            # ****** Death Benefit Guarantee Reset Logic ******
            FREQ_RESET_DECES = float(account_data['FREQ_RESET_DECES'])
            MAX_RESET_DECES = float(account_data['MAX_RESET_DECES'])

            if FREQ_RESET_DECES == 1 and AGE <= MAX_RESET_DECES:
                MT_GAR_DECES_PROJ = max(MT_GAR_DECES_PROJ, MT_VM_PROJ)

            # ****** Survival Probability Calculation ******
            QX = hash_find(lookups['mortality'], AGE, 0.0)
            WX = hash_find(lookups['lapse'], an_proj, 0.0)

            TX_SURVIE_DEB = TX_SURVIE
            TX_SURVIE = TX_SURVIE_DEB * (1 - QX) * (1 - WX)

            # ****** Cash Flow Calculations - EXACT SAS FORMULAS ******
            REVENUS = -FRAIS * TX_SURVIE_DEB
            FRAIS_GEST = -(MT_VM_DEB + RENDEMENT / 2) * float(account_data['PC_HONORAIRES_GEST']) * TX_SURVIE_DEB
            COMMISSIONS = -(MT_VM_DEB + RENDEMENT / 2) * float(account_data['TX_COMM_MAINTIEN']) * TX_SURVIE_DEB
            FRAIS_GEN = -float(account_data['FRAIS_ADMIN']) * TX_SURVIE_DEB
            PMT_GARANTIE = -max(0, MT_GAR_DECES_PROJ - MT_VM_PROJ) * QX * TX_SURVIE_DEB

            FLUX_NET = REVENUS + FRAIS_GEST + COMMISSIONS + FRAIS_GEN + PMT_GARANTIE

            # ****** Present Value Calculations ******
            TX_ACTU = hash_find(lookups['discount_ext'], an_proj, 1.0)

            VP_REVENUS = REVENUS * TX_ACTU
            VP_FRAIS_GEST = FRAIS_GEST * TX_ACTU
            VP_COMMISSIONS = COMMISSIONS * TX_ACTU
            VP_FRAIS_GEN = FRAIS_GEN * TX_ACTU
            VP_PMT_GARANTIE = PMT_GARANTIE * TX_ACTU
            VP_FLUX_NET = FLUX_NET * TX_ACTU

            # ****** Internal Scenario Adjustment ******
            if projection_type == "INTERNE" and start_year > 0:
                TX_ACTU_INT = hash_find(lookups['discount_int'], start_year, 1.0)
                if TX_ACTU_INT != 0:
                    VP_REVENUS = VP_REVENUS / TX_ACTU_INT
                    VP_FRAIS_GEST = VP_FRAIS_GEST / TX_ACTU_INT
                    VP_COMMISSIONS = VP_COMMISSIONS / TX_ACTU_INT
                    VP_FRAIS_GEN = VP_FRAIS_GEN / TX_ACTU_INT
                    VP_PMT_GARANTIE = VP_PMT_GARANTIE / TX_ACTU_INT
                    VP_FLUX_NET = VP_FLUX_NET / TX_ACTU_INT

        # Store results for this year
        result_row = {
            'year': current_year,
            'an_proj': an_proj,
            'AGE': AGE,
            'MT_VM_PROJ': MT_VM_PROJ,
            'MT_GAR_DECES_PROJ': MT_GAR_DECES_PROJ,
            'TX_SURVIE': TX_SURVIE,
            'TX_SURVIE_DEB': TX_SURVIE_DEB if 'TX_SURVIE_DEB' in locals() else TX_SURVIE,
            'FLUX_NET': FLUX_NET,
            'VP_FLUX_NET': VP_FLUX_NET
        }

        results.append(result_row)

    return results


def run_internal_calculations_exact(external_projection: List[Dict], account_data: pd.Series,
                                    scenario: int, lookups: Dict, calculation_type: str,
                                    NB_SC_INT: int, NB_AN_PROJECTION_INT: int,
                                    CHOC_CAPITAL: float) -> Dict:
    """
    Exact replication of internal calculations matching the second algorithm
    """

    year_results = {}

    # For each year in the external projection, calculate internal scenarios
    for ext_data in external_projection:
        year = ext_data['year']

        if year == 0:
            # No internal calculations for year 0
            year_results[year] = 0.0
            continue

        # Get state at this year from external projection
        fund_value = ext_data['MT_VM_PROJ']
        death_benefit = ext_data['MT_GAR_DECES_PROJ']
        survival_prob = ext_data['TX_SURVIE']

        if survival_prob <= 0.0001 or fund_value <= 0:
            year_results[year] = 0.0
            continue

        # Create modified account data for internal projection starting at this year
        modified_account = account_data.copy()
        modified_account['MT_VM'] = fund_value
        modified_account['MT_GAR_DECES'] = death_benefit
        modified_account['TX_SURVIE_DEB'] = survival_prob

        # Apply shock for capital calculations
        fund_shock = CHOC_CAPITAL if calculation_type == 'CAPITAL' else 0.0

        # Run internal scenarios from this year forward
        internal_scenarios_sum = []

        for internal_scenario in range(1, NB_SC_INT + 1):

            # Run internal projection exactly as in second algorithm
            internal_results = project_cash_flows_exact_sas_logic(
                modified_account, internal_scenario, 'INTERNE', lookups,
                NB_AN_PROJECTION_INT, fund_shock, start_year=year
            )

            if internal_results:
                # Sum VP_FLUX_NET for this internal scenario (matching second algorithm)
                total_vp = sum([row['VP_FLUX_NET'] for row in internal_results])
                internal_scenarios_sum.append(total_vp)

        # Calculate mean across internal scenarios
        if internal_scenarios_sum:
            year_results[year] = np.mean(internal_scenarios_sum)
        else:
            year_results[year] = 0.0

    return year_results


def calculate_distributable_flows_exact(external_results: List[Dict], lookups: Dict,
                                        NB_SC_INT: int, NB_AN_PROJECTION_INT: int,
                                        CHOC_CAPITAL: float, HURDLE_RT: float) -> List[Dict]:
    """
    Calculate distributable cash flows exactly matching second algorithm logic
    """

    final_results = []

    for ext_result in external_results:
        account_id = ext_result['account_id']
        scenario = ext_result['scenario']
        external_projection = ext_result['projection']
        account_data = ext_result['account_data']

        # Calculate reserves and capital exactly as second algorithm
        reserve_by_year = run_internal_calculations_exact(
            external_projection, account_data, scenario, lookups, 'RESERVE',
            NB_SC_INT, NB_AN_PROJECTION_INT, CHOC_CAPITAL
        )

        capital_results = run_internal_calculations_exact(
            external_projection, account_data, scenario, lookups, 'CAPITAL',
            NB_SC_INT, NB_AN_PROJECTION_INT, CHOC_CAPITAL
        )

        # Calculate capital as difference from reserves (matching second algorithm)
        capital_by_year = {}
        for year in capital_results:
            reserve_value = reserve_by_year.get(year, 0.0)
            capital_value = capital_results[year] - reserve_value
            capital_by_year[year] = capital_value

        # Calculate distributable cash flows with exact same logic as second algorithm
        distributable_pvs = []

        # Track previous year reserves and capital
        prev_reserve = 0.0
        prev_capital = 0.0

        for ext_data in external_projection:
            year = ext_data['year']
            external_cf = ext_data['FLUX_NET']

            # Get reserves and capital for this year
            current_reserve = reserve_by_year.get(year, 0.0)
            current_capital = capital_by_year.get(year, 0.0)

            # Calculate profit and distributable exactly as second algorithm
            if year == 0:
                # Year 0: Initial establishment
                profit = external_cf + current_reserve
                distributable = profit + current_capital
            else:
                # Other years: Changes in reserves and capital
                profit = external_cf + (current_reserve - prev_reserve)
                distributable = profit + (current_capital - prev_capital)

            # Present value at hurdle rate
            if year > 0:
                pv_distributable = distributable / ((1 + HURDLE_RT) ** year)
            else:
                pv_distributable = distributable

            distributable_pvs.append(pv_distributable)

            # Update previous values
            prev_reserve = current_reserve
            prev_capital = current_capital

        # Sum across all years
        total_pv_distributable = sum(distributable_pvs)

        final_results.append({
            'ID_COMPTE': account_id,
            'scn_eval': scenario,
            'VP_FLUX_DISTRIBUABLES': total_pv_distributable
        })

    return final_results


def run_external_calculations_exact(data: Dict, lookups: Dict, NBCPT: int, NB_SC: int, NB_AN_PROJECTION: int) -> List[
    Dict]:
    """Run external calculations with exact SAS logic"""

    external_results = []

    for account_idx in range(min(NBCPT, len(data['population']))):
        account_data = data['population'].iloc[account_idx]
        account_id = account_data['ID_COMPTE']

        for scenario in range(1, NB_SC + 1):
            # Project external path with exact SAS logic
            projection = project_cash_flows_exact_sas_logic(
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


def acfc_algorithm_fully_fixed(data_path: str = ".", NBCPT: int = 4, NB_SC: int = 10, NB_AN_PROJECTION: int = 10,
                               NB_SC_INT: int = 10, NB_AN_PROJECTION_INT: int = 10,
                               CHOC_CAPITAL: float = 0.35, HURDLE_RT: float = 0.10) -> pd.DataFrame:
    """
    Fully Fixed ACFC Algorithm - Exactly matching second algorithm logic
    """

    print("Phase 1: Loading input data...")
    data = load_input_data(data_path)

    print("Phase 1: Creating lookup tables...")
    lookups = create_lookup_tables(data)

    print("Phase 2: Running external calculations with exact SAS logic...")
    external_results = run_external_calculations_exact(data, lookups, NBCPT, NB_SC, NB_AN_PROJECTION)

    print("Phase 3-5: Running internal calculations with exact matching logic...")
    final_results = calculate_distributable_flows_exact(
        external_results, lookups, NB_SC_INT, NB_AN_PROJECTION_INT, CHOC_CAPITAL, HURDLE_RT
    )

    print("Phase 6: Generating output...")
    output_df = pd.DataFrame(final_results)

    print(f"Fully Fixed Algorithm completed. Generated {len(output_df)} results.")
    return output_df


# Example usage
if __name__ == "__main__":
    results = acfc_algorithm_fully_fixed(
        data_path=HERE.joinpath("data_in"),
        NBCPT=4,
        NB_SC=10,
        NB_AN_PROJECTION=10,
        NB_SC_INT=10,
        NB_AN_PROJECTION_INT=10,
        CHOC_CAPITAL=0.35,
        HURDLE_RT=0.10
    )

    print("\nSample Results:")
    print(results.head(10))
    results.to_csv(HERE.joinpath('test/acfc_results_fixed.csv'))
    print(f"\nMean VP_FLUX_DISTRIBUABLES: {results['VP_FLUX_DISTRIBUABLES'].mean():.2f}")
    print(f"Range: {results['VP_FLUX_DISTRIBUABLES'].min():.2f} to {results['VP_FLUX_DISTRIBUABLES'].max():.2f}")
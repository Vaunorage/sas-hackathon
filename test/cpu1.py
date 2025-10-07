import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
import warnings
import logging
import time
from tqdm import tqdm

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def load_input_data(data_path: str = ".", nb_accounts: int = None) -> Dict:
    """Load all input data files"""
    try:
        population = pd.read_csv(f"{data_path}/population_fixed.csv")
        if nb_accounts is not None:
            population = population.head(nb_accounts)
        rendement = pd.read_csv(f"{data_path}/rendement1.csv")
        tx_deces = pd.read_csv(f"{data_path}/tx_deces_fixed.csv")
        tx_interet = pd.read_csv(f"{data_path}/tx_interet_fixed.csv")
        tx_interet_int = pd.read_csv(f"{data_path}/tx_interet_int_fixed.csv")
        tx_retrait = pd.read_csv(f"{data_path}/tx_retrait_fixed.csv")

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


def create_lookup_tables(data: Dict) -> Dict:
    """Create hash table lookups for O(1) access"""
    lookups = {}

    lookups['mortality'] = dict(zip(data['tx_deces']['AGE'], data['tx_deces']['QX']))
    lookups['lapse'] = dict(zip(data['tx_retrait']['an_proj'], data['tx_retrait']['WX']))
    lookups['discount_ext'] = dict(zip(data['tx_interet']['an_proj'], data['tx_interet']['TX_ACTU']))
    lookups['discount_int'] = dict(zip(data['tx_interet_int']['an_eval'], data['tx_interet_int']['TX_ACTU_INT']))

    lookups['returns'] = {}
    for _, row in data['rendement'].iterrows():
        key = (int(row['an_proj']), int(row['scn_proj']), row['TYPE'])
        lookups['returns'][key] = row['RENDEMENT']

    return lookups


def hash_find(hash_table: dict, key, default_value=None):
    """Mimic SAS hash.find() behavior"""
    return hash_table.get(key, default_value if default_value is not None else 0.0)


def log_external_year_details(results: List[Dict], year: int, account_id: int = None, scenario: int = None):
    """Log detailed information for a specific year (GPU style)"""
    year_data = [r for r in results if r['year'] == year]

    if account_id is not None:
        year_data = [r for r in year_data if r['account_id'] == account_id]
    if scenario is not None:
        year_data = [r for r in year_data if r['scenario'] == scenario]

    if len(year_data) == 0:
        return

    print(f"\n--- Year {year} Details ---")
    for row in year_data:
        acc_id = row['account_id']
        scn = row['scenario']
        age = row['AGE']
        vm = row['MT_VM_PROJ']
        death_ben = row['MT_GAR_DECES_PROJ']
        survie = row['TX_SURVIE']
        flux = row['FLUX_NET']
        vp_flux = row['VP_FLUX_NET']

        print(f"  Account {int(acc_id)}, Scenario {int(scn)}:")
        print(f"    Age: {int(age)}")
        print(f"    Fund Value: {vm:,.2f}")
        print(f"    Death Benefit: {death_ben:,.2f}")
        print(f"    Survival Prob: {survie:.6f}")
        print(f"    Net Cash Flow: {flux:,.2f}")
        print(f"    PV Net Cash Flow: {vp_flux:,.2f}")


def project_cash_flows_exact_sas_logic(account_data: pd.Series, scenario: int, projection_type: str,
                                       lookups: Dict, nb_years: int, fund_shock: float = 0.0,
                                       start_year: int = 0, verbose: bool = False) -> List[Dict]:
    """
    Exact replication of SAS cash flow calculation logic
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

    for current_year in year_range:

        # ***********************************************
        # *** Initialization for year 0 ***
        # ***********************************************

        if current_year == 0 and projection_type == "EXTERNE":
            AGE = int(account_data['age_deb'])
            MT_VM_PROJ = float(account_data['MT_VM'])
            MT_GAR_DECES_PROJ = float(account_data['MT_GAR_DECES'])
            TX_SURVIE = 1.0
            TX_SURVIE_DEB = 1.0
            TX_ACTU = 1.0
            QX = 0.0
            WX = 0.0
            an_proj = 0

            # Year 0 cash flows
            COMMISSIONS = -float(account_data.get('TX_COMM_VENTE', 0.0)) * MT_VM_PROJ
            VP_COMMISSIONS = COMMISSIONS
            FRAIS_GEN = -float(account_data['FRAIS_ACQUI'])
            VP_FRAIS_GEN = FRAIS_GEN
            FLUX_NET = FRAIS_GEN + COMMISSIONS
            VP_FLUX_NET = FLUX_NET

            # Zero out other components
            REVENUS = 0.0
            FRAIS_GEST = 0.0
            PMT_GARANTIE = 0.0
            VP_REVENUS = 0.0
            VP_FRAIS_GEST = 0.0
            VP_PMT_GARANTIE = 0.0

        elif current_year == 0 and projection_type == "INTERNE":
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

            # Zero out all cash flows
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

        elif TX_SURVIE == 0 or MT_VM_PROJ == 0:
            continue

        # ***********************************************************************
        # *** Cash flow calculations for all projection years ***
        # ***********************************************************************
        else:
            scn_proj = scenario

            if projection_type == "INTERNE":
                AGE = int(account_data['age_deb']) + start_year + current_year
                an_proj = start_year + current_year
            else:
                AGE = int(account_data['age_deb']) + current_year
                an_proj = current_year

            # Fund Value Projection
            MT_VM_DEB = MT_VM_PROJ
            RENDEMENT_rate = hash_find(lookups['returns'], (an_proj, scn_proj, projection_type), 0.0)
            RENDEMENT = MT_VM_DEB * RENDEMENT_rate
            FRAIS = -(MT_VM_DEB + RENDEMENT / 2) * float(account_data['PC_REVENU_FDS'])
            MT_VM_PROJ = MT_VM_PROJ + RENDEMENT + FRAIS
            MT_VM_PROJ = max(MT_VM_PROJ, 0)

            # Death Benefit Reset
            FREQ_RESET_DECES = float(account_data['FREQ_RESET_DECES'])
            MAX_RESET_DECES = float(account_data['MAX_RESET_DECES'])

            if FREQ_RESET_DECES == 1 and AGE <= MAX_RESET_DECES:
                MT_GAR_DECES_PROJ = max(MT_GAR_DECES_PROJ, MT_VM_PROJ)

            # Survival Probability
            QX = hash_find(lookups['mortality'], AGE, 0.0)
            WX = hash_find(lookups['lapse'], an_proj, 0.0)
            TX_SURVIE_DEB = TX_SURVIE
            TX_SURVIE = TX_SURVIE_DEB * (1 - QX) * (1 - WX)

            # Cash Flows
            REVENUS = -FRAIS * TX_SURVIE_DEB
            FRAIS_GEST = -(MT_VM_DEB + RENDEMENT / 2) * float(account_data['PC_HONORAIRES_GEST']) * TX_SURVIE_DEB
            COMMISSIONS = -(MT_VM_DEB + RENDEMENT / 2) * float(account_data['TX_COMM_MAINTIEN']) * TX_SURVIE_DEB
            FRAIS_GEN = -float(account_data['FRAIS_ADMIN']) * TX_SURVIE_DEB
            PMT_GARANTIE = -max(0, MT_GAR_DECES_PROJ - MT_VM_PROJ) * QX * TX_SURVIE_DEB
            FLUX_NET = REVENUS + FRAIS_GEST + COMMISSIONS + FRAIS_GEN + PMT_GARANTIE

            # Present Values
            TX_ACTU = hash_find(lookups['discount_ext'], an_proj, 1.0)
            VP_REVENUS = REVENUS * TX_ACTU
            VP_FRAIS_GEST = FRAIS_GEST * TX_ACTU
            VP_COMMISSIONS = COMMISSIONS * TX_ACTU
            VP_FRAIS_GEN = FRAIS_GEN * TX_ACTU
            VP_PMT_GARANTIE = PMT_GARANTIE * TX_ACTU
            VP_FLUX_NET = FLUX_NET * TX_ACTU

            # Internal Adjustment
            if projection_type == "INTERNE" and start_year > 0:
                TX_ACTU_INT = hash_find(lookups['discount_int'], start_year, 1.0)
                if TX_ACTU_INT != 0:
                    VP_REVENUS = VP_REVENUS / TX_ACTU_INT
                    VP_FRAIS_GEST = VP_FRAIS_GEST / TX_ACTU_INT
                    VP_COMMISSIONS = VP_COMMISSIONS / TX_ACTU_INT
                    VP_FRAIS_GEN = VP_FRAIS_GEN / TX_ACTU_INT
                    VP_PMT_GARANTIE = VP_PMT_GARANTIE / TX_ACTU_INT
                    VP_FLUX_NET = VP_FLUX_NET / TX_ACTU_INT

        # Store results
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


def kahan_sum(numbers):
    """Kahan compensated summation for improved numerical accuracy"""
    sum_val = 0.0
    compensation = 0.0
    for x in numbers:
        y = x - compensation
        t = sum_val + y
        compensation = (t - sum_val) - y
        sum_val = t
    return sum_val


def run_internal_calculations_exact(external_projection: List[Dict], account_data: pd.Series,
                                    scenario: int, lookups: Dict, calculation_type: str,
                                    NB_SC_INT: int, NB_AN_PROJECTION_INT: int,
                                    CHOC_CAPITAL: float, verbose: bool = False) -> Dict:
    """
    Internal calculations matching GPU logic
    """
    year_results = {}
    valid_years = [ext_data for ext_data in external_projection if ext_data['year'] > 0]

    for ext_data in valid_years:
        year = ext_data['year']
        fund_value = ext_data['MT_VM_PROJ']
        death_benefit = ext_data['MT_GAR_DECES_PROJ']
        survival_prob = ext_data['TX_SURVIE']

        if survival_prob <= 0.0001 or fund_value <= 0:
            year_results[year] = 0.0
            continue

        modified_account = account_data.copy()
        modified_account['MT_VM'] = fund_value
        modified_account['MT_GAR_DECES'] = death_benefit
        modified_account['TX_SURVIE_DEB'] = survival_prob

        fund_shock = CHOC_CAPITAL if calculation_type == 'CAPITAL' else 0.0

        internal_scenarios_sum = []

        for internal_scenario in range(1, NB_SC_INT + 1):
            internal_results = project_cash_flows_exact_sas_logic(
                modified_account, internal_scenario, 'INTERNE', lookups,
                NB_AN_PROJECTION_INT, fund_shock, start_year=year, verbose=False
            )

            if internal_results:
                total_vp = kahan_sum([row['VP_FLUX_NET'] for row in internal_results])
                internal_scenarios_sum.append(total_vp)

        if internal_scenarios_sum:
            year_results[year] = np.mean(internal_scenarios_sum)
        else:
            year_results[year] = 0.0

    year_results[0] = 0.0
    return year_results


def acfc_algorithm_with_gpu_logging(data_path: str = ".", nb_accounts: int = 4, nb_scenarios: int = 10,
                                    nb_years: int = 10, nb_sc_int: int = 10, nb_an_projection_int: int = 10,
                                    choc_capital: float = 0.35, hurdle_rt: float = 0.10,
                                    verbose: bool = True) -> pd.DataFrame:
    """
    Complete CPU ACFC Algorithm with GPU-style detailed logging
    """

    if verbose:
        print("\n" + "=" * 80)
        print("CPU ACFC ALGORITHM - COMPLETE EXECUTION LOG")
        print("=" * 80)
        print(f"Parameters:")
        print(f"  Accounts: {nb_accounts}")
        print(f"  External Scenarios: {nb_scenarios}")
        print(f"  Projection Years: {nb_years}")
        print(f"  Internal Scenarios: {nb_sc_int}")
        print(f"  Internal Projection Years: {nb_an_projection_int}")
        print(f"  Capital Shock: {choc_capital}")
        print(f"  Hurdle Rate: {hurdle_rt}")

    print("\nPhase 1: Loading input data...")
    data = load_input_data(data_path, nb_accounts)

    print("\nPhase 2: Creating lookup tables...")
    lookups = create_lookup_tables(data)

    print("\nPhase 3: Preparing data...")
    if verbose:
        print("\n" + "=" * 80)
        print("INITIAL DATA PREPARATION - DETAILED LOG")
        print("=" * 80)

        for account_idx in range(min(nb_accounts, len(data['population']))):
            account_data = data['population'].iloc[account_idx]
            account_id = int(account_data['ID_COMPTE'])

            print(f"\n--- Account {account_id} (Index {account_idx}) ---")
            print(f"  MT_VM (Initial Fund Value):       {account_data['MT_VM']:,.2f}")
            print(f"  MT_GAR_DECES (Death Benefit):     {account_data['MT_GAR_DECES']:,.2f}")
            print(f"  AGE_DEB (Starting Age):           {int(account_data['age_deb'])}")
            print(f"  TX_COMM_VENTE (Sales Commission): {account_data.get('TX_COMM_VENTE', 0.0):.4f}")
            print(f"  FRAIS_ACQUI (Acquisition Fee):    {account_data['FRAIS_ACQUI']:.2f}")
            print(f"  PC_REVENU_FDS (Fund Revenue):     {account_data['PC_REVENU_FDS']:.4f}")
            print(f"  PC_HONORAIRES_GEST (Mgmt Fee):    {account_data['PC_HONORAIRES_GEST']:.4f}")
            print(f"  TX_COMM_MAINTIEN (Ongoing Comm):  {account_data['TX_COMM_MAINTIEN']:.4f}")
            print(f"  FRAIS_ADMIN (Admin Fee):          {account_data['FRAIS_ADMIN']:.2f}")
            print(f"  FREQ_RESET_DECES (Reset Freq):    {account_data['FREQ_RESET_DECES']:.0f}")
            print(f"  MAX_RESET_DECES (Max Reset Age):  {account_data['MAX_RESET_DECES']:.0f}")

    print("\nPhase 4: Running external projections...")
    all_external_results = []

    if verbose:
        print(f"\n{'=' * 80}")
        print("RUNNING EXTERNE PROJECTION")
        print(f"{'=' * 80}")

    for account_idx in range(min(nb_accounts, len(data['population']))):
        account_data = data['population'].iloc[account_idx]
        account_id = int(account_data['ID_COMPTE'])

        for scenario in range(1, nb_scenarios + 1):
            if verbose:
                print(f"\nProcessing year 0...")

            projection = project_cash_flows_exact_sas_logic(
                account_data, scenario, 'EXTERNE', lookups, nb_years, verbose=False
            )

            # Add account_id and scenario to each result
            for result in projection:
                result['account_id'] = account_id
                result['scenario'] = scenario

            all_external_results.extend(projection)

            # Log first few years in detail for first scenario
            if verbose and scenario == 1:
                for year in [0, 1, 2]:
                    log_external_year_details(projection, year, account_id, scenario)

    # External results summary
    if verbose:
        print(f"\n{'=' * 80}")
        print("EXTERNAL PROJECTION RESULTS SUMMARY")
        print(f"{'=' * 80}")
        print(f"Total results: {len(all_external_results)}")

        from collections import defaultdict
        by_account = defaultdict(list)
        for r in all_external_results:
            by_account[r['account_id']].append(r)

        for account_id in sorted(by_account.keys()):
            account_results = by_account[account_id]
            scenarios = sorted(set(r['scenario'] for r in account_results))
            print(f"\nAccount {account_id}:")
            print(f"  Total results: {len(account_results)}")
            print(f"  Scenarios: {scenarios}")

            for yr in [0, 1]:
                yr_data = [r for r in account_results if r['year'] == yr and r['scenario'] == 1]
                if yr_data:
                    row = yr_data[0]
                    print(f"  Year {yr} (Scenario 1):")
                    print(f"    Fund Value: {row['MT_VM_PROJ']:,.2f}")
                    print(f"    Death Benefit: {row['MT_GAR_DECES_PROJ']:,.2f}")
                    print(f"    Survival Prob: {row['TX_SURVIE']:.6f}")
                    print(f"    Net Cash Flow: {row['FLUX_NET']:,.2f}")
                    print(f"    PV Cash Flow: {row['VP_FLUX_NET']:,.2f}")

    print("\nPhase 5: Running internal calculations...")

    # Group by account and scenario
    grouped = defaultdict(lambda: defaultdict(list))
    for r in all_external_results:
        grouped[r['account_id']][r['scenario']].append(r)

    final_results = []

    for account_idx in range(min(nb_accounts, len(data['population']))):
        account_data = data['population'].iloc[account_idx]
        account_id = int(account_data['ID_COMPTE'])

        for scenario in range(1, nb_scenarios + 1):
            external_projection = grouped[account_id][scenario]

            if verbose and scenario == 1:
                print(f"\nCalculating reserves (no shock) for Account {account_id}, Scenario {scenario}...")

            # Calculate reserves
            reserve_by_year = run_internal_calculations_exact(
                external_projection, account_data, scenario, lookups, 'RESERVE',
                nb_sc_int, nb_an_projection_int, choc_capital, verbose=False
            )

            if verbose and scenario == 1:
                print(f"Calculating capital (with {choc_capital} shock)...")

            # Calculate capital
            capital_results = run_internal_calculations_exact(
                external_projection, account_data, scenario, lookups, 'CAPITAL',
                nb_sc_int, nb_an_projection_int, choc_capital, verbose=False
            )

            capital_by_year = {}
            for year in capital_results:
                reserve_value = reserve_by_year.get(year, 0.0)
                capital_value = capital_results[year] - reserve_value
                capital_by_year[year] = capital_value

            # Calculate distributable flows
            if verbose and scenario == 1:
                print(f"\n{'=' * 80}")
                print(f"DISTRIBUTABLE FLOWS CALCULATION")
                print(f"{'=' * 80}")
                print(f"Account {account_id}, Scenario {scenario}:")
                print(
                    f"  {'Year':<6} {'Ext CF':<12} {'Reserve':<12} {'Capital':<12} {'Profit':<12} {'Distrib':<12} {'PV Distrib':<12}")
                print(f"  {'-' * 6} {'-' * 12} {'-' * 12} {'-' * 12} {'-' * 12} {'-' * 12} {'-' * 12}")

            distributable_pvs = []
            prev_reserve = 0.0
            prev_capital = 0.0

            for ext_result in external_projection:
                year = ext_result['year']
                external_cf = ext_result['FLUX_NET']

                current_reserve = reserve_by_year.get(year, 0.0)
                current_capital = capital_by_year.get(year, 0.0)

                if year == 0:
                    profit = external_cf + current_reserve
                    distributable = profit + current_capital
                else:
                    profit = external_cf + (current_reserve - prev_reserve)
                    distributable = profit + (current_capital - prev_capital)

                if year > 0:
                    pv_distributable = distributable / ((1 + hurdle_rt) ** year)
                else:
                    pv_distributable = distributable

                distributable_pvs.append(pv_distributable)

                if verbose and scenario == 1 and (year <= 5 or year == len(external_projection) - 1):
                    print(f"  {year:<6} {external_cf:>12,.2f} {current_reserve:>12,.2f} {current_capital:>12,.2f} "
                          f"{profit:>12,.2f} {distributable:>12,.2f} {pv_distributable:>12,.2f}")
                elif verbose and scenario == 1 and year == 6:
                    print(f"  {'...':<6} {'...':<12} {'...':<12} {'...':<12} {'...':<12} {'...':<12} {'...':<12}")

                prev_reserve = current_reserve
                prev_capital = current_capital

            total_pv_distributable = sum(distributable_pvs)

            if verbose and scenario == 1:
                print(f"  {'TOTAL':<6} {'':<12} {'':<12} {'':<12} {'':<12} {'':<12} {total_pv_distributable:>12,.2f}")

            final_results.append({
                'ID_COMPTE': account_id,
                'scn_eval': scenario,
                'VP_FLUX_DISTRIBUABLES': total_pv_distributable
            })

    print("\nPhase 7: Converting to DataFrame...")
    output_df = pd.DataFrame(final_results)

    if verbose:
        print(f"\n{'=' * 80}")
        print("FINAL RESULTS SUMMARY")
        print(f"{'=' * 80}")
        print(f"Total results: {len(output_df)}")
        print(f"\nMean VP_FLUX_DISTRIBUABLES: {output_df['VP_FLUX_DISTRIBUABLES'].mean():,.2f}")
        print(f"Min: {output_df['VP_FLUX_DISTRIBUABLES'].min():,.2f}")
        print(f"Max: {output_df['VP_FLUX_DISTRIBUABLES'].max():,.2f}")
        print(f"\nResults by account:")
        for account_id in sorted(output_df['ID_COMPTE'].unique()):
            account_data = output_df[output_df['ID_COMPTE'] == account_id]
            print(f"  Account {account_id}: Mean = {account_data['VP_FLUX_DISTRIBUABLES'].mean():,.2f}, "
                  f"Scenarios = {len(account_data)}")

    return output_df


if __name__ == "__main__":
    data_path = "data_in"

    results = acfc_algorithm_with_gpu_logging(
        data_path=data_path,
        nb_accounts=2,
        nb_scenarios=2,
        nb_years=100,
        nb_sc_int=2,
        nb_an_projection_int=100,
        choc_capital=0.35,
        hurdle_rt=0.10,
        verbose=True
    )

    print("\nFinal Results:")
    print(results)
    results.to_csv('test/cpu_results_complete.csv', index=False)
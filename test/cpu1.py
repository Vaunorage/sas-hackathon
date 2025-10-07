import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
import warnings
import logging
import time
from tqdm import tqdm

from paths import HERE

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class LoggingConfig:
    """Configuration class for controlling detailed logging"""

    def __init__(self,
                 verbose: bool = True,
                 log_account_id: Optional[int] = None,
                 log_scenario: Optional[int] = None,
                 log_max_years: int = 5,
                 log_internal_scenarios: bool = False):
        """
        Parameters:
        -----------
        verbose : bool
            Enable/disable all logging
        log_account_id : int, optional
            Specific account ID to log (None = log all accounts)
        log_scenario : int, optional
            Specific scenario to log (None = log all scenarios, typically defaults to 1)
        log_max_years : int
            Maximum number of years to show in detailed year-by-year logs
        log_internal_scenarios : bool
            Whether to show details for each internal scenario (can be very verbose)
        """
        self.verbose = verbose
        self.log_account_id = log_account_id
        self.log_scenario = log_scenario if log_scenario is not None else 1  # Default to scenario 1
        self.log_max_years = log_max_years
        self.log_internal_scenarios = log_internal_scenarios

    def should_log_account(self, account_id: int) -> bool:
        """Check if we should log for this account"""
        if not self.verbose:
            return False
        return self.log_account_id is None or account_id == self.log_account_id

    def should_log_scenario(self, scenario: int) -> bool:
        """Check if we should log for this scenario"""
        return self.log_scenario is None or scenario == self.log_scenario

    def should_log_year(self, year: int, total_years: int) -> bool:
        """Check if we should log this year in detail"""
        # Always log year 0 and last year, plus first N years
        return year == 0 or year <= self.log_max_years or year == total_years - 1


def load_input_files(data_path: str, verbose: bool = True) -> Tuple[pd.DataFrame, ...]:
    """Load all input CSV files exactly as SAS does"""
    try:
        if verbose:
            print("\n📁 LOADING INPUT FILES")
            print("-" * 50)

        files_to_load = [
            ("population_fixed.csv", "Population data"),
            ("rendement1.csv", "Returns data"),
            ("tx_deces_fixed.csv", "Mortality rates"),
            ("tx_interet_fixed.csv", "Interest rates"),
            ("tx_interet_int_fixed.csv", "Internal interest rates"),
            ("tx_retrait_fixed.csv", "Lapse rates")
        ]

        loaded_data = []
        for filename, description in files_to_load:
            if verbose:
                print(f"Loading {description}...")
            df = pd.read_csv(f"{data_path}/{filename}")
            loaded_data.append(df)
            if verbose:
                print(f"  ✓ {description}: {len(df):,} rows")
                time.sleep(0.1)

        population, rendement, tx_deces, tx_interet, tx_interet_int, tx_retrait = loaded_data

        if 'TYPE' in rendement.columns:
            if verbose:
                print("🔧 Processing TYPE column encoding...")
            rendement['TYPE'] = rendement['TYPE'].apply(
                lambda x: x.decode('utf-8') if isinstance(x, bytes) else str(x)
            )
            if verbose:
                print(f"  ✓ Processed {len(rendement):,} TYPE entries")

        if verbose:
            print(f"\n✅ All files loaded successfully!")
            print(f"📊 Found {len(population)} accounts for processing")

        return population, rendement, tx_deces, tx_interet, tx_interet_int, tx_retrait

    except Exception as e:
        logger.error(f"Error loading input files: {e}")
        raise


def load_input_data(data_path: str = ".", verbose: bool = True):
    """Load all input data files and create lookup dictionaries"""
    population, rendement, tx_deces, tx_interet, tx_interet_int, tx_retrait = load_input_files(data_path, verbose)

    return {
        'population': population,
        'rendement': rendement,
        'tx_deces': tx_deces,
        'tx_interet': tx_interet,
        'tx_interet_int': tx_interet_int,
        'tx_retrait': tx_retrait
    }


def create_lookup_tables(data: Dict, verbose: bool = True) -> Dict:
    """Create hash table lookups for O(1) access"""
    if verbose:
        print("\n🔍 CREATING LOOKUP TABLES")
        print("-" * 50)

    lookups = {}

    if verbose:
        print("Building mortality lookup table...")
    lookups['mortality'] = dict(zip(data['tx_deces']['AGE'], data['tx_deces']['QX']))
    if verbose:
        print(f"  ✓ {len(lookups['mortality'])} mortality rates loaded")

    if verbose:
        print("Building lapse lookup table...")
    lookups['lapse'] = dict(zip(data['tx_retrait']['an_proj'], data['tx_retrait']['WX']))
    if verbose:
        print(f"  ✓ {len(lookups['lapse'])} lapse rates loaded")

    if verbose:
        print("Building discount rate lookup tables...")
    lookups['discount_ext'] = dict(zip(data['tx_interet']['an_proj'], data['tx_interet']['TX_ACTU']))
    lookups['discount_int'] = dict(zip(data['tx_interet_int']['an_eval'], data['tx_interet_int']['TX_ACTU_INT']))
    if verbose:
        print(f"  ✓ {len(lookups['discount_ext'])} external rates, {len(lookups['discount_int'])} internal rates")

    if verbose:
        print("Building returns lookup table...")
    lookups['returns'] = {}

    if len(data['rendement']) > 5000 and verbose:
        iterator = tqdm(data['rendement'].iterrows(),
                        desc="Processing returns",
                        total=len(data['rendement']),
                        unit="rows")
    else:
        iterator = data['rendement'].iterrows()

    for _, row in iterator:
        key = (int(row['an_proj']), int(row['scn_proj']), row['TYPE'])
        lookups['returns'][key] = row['RENDEMENT']

    if verbose:
        print(f"  ✓ {len(lookups['returns'])} return scenarios loaded")
        print("✅ All lookup tables created successfully!")

    return lookups


def hash_find(hash_table: dict, key, default_value=None):
    """Mimic SAS hash.find() behavior"""
    return hash_table.get(key, default_value if default_value is not None else 0.0)


def project_cash_flows_exact_sas_logic(account_data: pd.Series, scenario: int, projection_type: str,
                                       lookups: Dict, nb_years: int, fund_shock: float = 0.0,
                                       start_year: int = 0, log_config: LoggingConfig = None) -> List[Dict]:
    """
    Exact replication of SAS cash flow calculation logic with configurable logging
    """

    if log_config is None:
        log_config = LoggingConfig(verbose=False)

    # Determine if we should log details for this account and scenario
    account_id = account_data.get('ID_COMPTE')
    should_log = (log_config.should_log_account(account_id) and
                  log_config.should_log_scenario(scenario) and
                  projection_type == "EXTERNE")

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

    if should_log:
        print(f"\n{'=' * 80}")
        print(f"PROJECTION DETAILS - Account {account_id}, Scenario {scenario}")
        print(f"{'=' * 80}")
        print(f"Type: {projection_type}")
        print(f"Starting Age: {int(account_data['age_deb'])}")
        print(f"Initial Fund Value: {account_data['MT_VM']:,.2f}")
        print(f"Initial Death Benefit: {account_data['MT_GAR_DECES']:,.2f}")
        print(f"Max Projection Years: {max_years}")
        print(f"\nYear-by-Year Progression (showing up to {log_config.log_max_years} years):")
        print(
            f"{'Year':<6} {'Age':<5} {'Fund Value':<14} {'Death Ben':<14} {'Survival':<12} {'Net CF':<14} {'PV Net CF':<14}")
        print(f"{'-' * 6} {'-' * 5} {'-' * 14} {'-' * 14} {'-' * 12} {'-' * 14} {'-' * 14}")

    for year_idx, current_year in enumerate(year_range):

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

        # Log detailed year information
        if should_log and log_config.should_log_year(current_year, len(year_range)):
            print(f"{current_year:<6} {AGE:<5} {MT_VM_PROJ:>14,.2f} {MT_GAR_DECES_PROJ:>14,.2f} "
                  f"{TX_SURVIE:>12.6f} {FLUX_NET:>14,.2f} {VP_FLUX_NET:>14,.2f}")
        elif should_log and current_year == log_config.log_max_years + 1:
            print(f"{'...':<6} {'...':<5} {'...':<14} {'...':<14} {'...':<12} {'...':<14} {'...':<14}")

        # Store results
        result_row = {
            'year': current_year,
            'an_proj': an_proj,
            'AGE': AGE,
            'MT_VM_PROJ': MT_VM_PROJ,
            'MT_GAR_DECES_PROJ': MT_GAR_DECES_PROJ,
            'TX_SURVIE': TX_SURVIE,
            'TX_SURVIE_DEB': TX_SURVIE_DEB if 'TX_SURVIE_DEB' in locals() else TX_SURVIE,
            'RENDEMENT': RENDEMENT if 'RENDEMENT' in locals() else 0.0,  # ADD THIS LINE
            'FLUX_NET': FLUX_NET,
            'VP_FLUX_NET': VP_FLUX_NET
        }

        results.append(result_row)

    return results


def kahan_sum(numbers):
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
                                    CHOC_CAPITAL: float, log_config: LoggingConfig = None) -> Dict:
    """
    Internal calculations with configurable logging
    """
    if log_config is None:
        log_config = LoggingConfig(verbose=False)

    account_id = account_data.get('ID_COMPTE')
    should_log = (log_config.should_log_account(account_id) and
                  log_config.should_log_scenario(scenario))

    year_results = {}
    valid_years = [ext_data for ext_data in external_projection if ext_data['year'] > 0]

    if should_log and valid_years:
        print(f"\n{'=' * 80}")
        print(f"{calculation_type} CALCULATION - Account {account_id}, Scenario {scenario}")
        print(f"{'=' * 80}")
        print(f"Processing {len(valid_years)} years × {NB_SC_INT} internal scenarios")
        shock_text = f"with {CHOC_CAPITAL:.1%} shock" if calculation_type == 'CAPITAL' else "no shock"
        print(f"Fund shock: {shock_text}")

    years_logged = 0
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
                NB_AN_PROJECTION_INT, fund_shock, start_year=year, log_config=LoggingConfig(verbose=False)
            )

            if internal_results:
                total_vp = kahan_sum([row['VP_FLUX_NET'] for row in internal_results])
                internal_scenarios_sum.append(total_vp)

        if internal_scenarios_sum:
            year_results[year] = np.mean(internal_scenarios_sum)
            if should_log and years_logged < log_config.log_max_years:
                print(f"  Year {year}: Mean across {len(internal_scenarios_sum)} scenarios = {year_results[year]:,.2f}")
                years_logged += 1
            elif should_log and years_logged == log_config.log_max_years:
                print(f"  ...")
                years_logged += 1
        else:
            year_results[year] = 0.0

    year_results[0] = 0.0

    if should_log and len(year_results) > 0:
        print(
            f"\n{calculation_type} final result for year {max(year_results.keys())}: {year_results[max(year_results.keys())]:,.2f}")

    return year_results


def calculate_distributable_flows_exact(external_results: List[Dict], lookups: Dict,
                                        NB_SC_INT: int, NB_AN_PROJECTION_INT: int,
                                        CHOC_CAPITAL: float, HURDLE_RT: float,
                                        log_config: LoggingConfig = None) -> Tuple[List[Dict], List[Dict]]:
    """
    Calculate distributable cash flows with configurable logging

    Returns:
    --------
    Tuple of (summary_results, detailed_results)
    - summary_results: List of dicts with total VP_FLUX_DISTRIBUABLES per account-scenario
    - detailed_results: List of dicts with year-by-year details
    """
    if log_config is None:
        log_config = LoggingConfig(verbose=False)

    final_results = []
    detailed_results = []

    if log_config.verbose:
        print("\n💰 CALCULATING DISTRIBUTABLE FLOWS")
        print("-" * 50)
        print(f"Processing {len(external_results)} account×scenario combinations...")

    account_groups = {}
    for ext_result in external_results:
        account_id = ext_result['account_id']
        if account_id not in account_groups:
            account_groups[account_id] = []
        account_groups[account_id].append(ext_result)

    account_progress = tqdm(account_groups.items(),
                            desc="Processing accounts",
                            unit="account",
                            disable=not log_config.verbose)

    for account_id, account_scenarios in account_progress:
        if log_config.verbose:
            account_progress.set_postfix({"Account": account_id})

        for scenario_idx, ext_result in enumerate(account_scenarios, 1):
            scenario = ext_result['scenario']
            external_projection = ext_result['projection']
            account_data = ext_result['account_data']

            should_log = (log_config.should_log_account(account_id) and
                          log_config.should_log_scenario(scenario))

            # Calculate reserves and capital
            reserve_by_year = run_internal_calculations_exact(
                external_projection, account_data, scenario, lookups, 'RESERVE',
                NB_SC_INT, NB_AN_PROJECTION_INT, CHOC_CAPITAL, log_config=log_config
            )

            capital_results = run_internal_calculations_exact(
                external_projection, account_data, scenario, lookups, 'CAPITAL',
                NB_SC_INT, NB_AN_PROJECTION_INT, CHOC_CAPITAL, log_config=log_config
            )

            capital_by_year = {}
            for year in capital_results:
                reserve_value = reserve_by_year.get(year, 0.0)
                capital_value = capital_results[year] - reserve_value
                capital_by_year[year] = capital_value

            # Calculate distributable flows
            if should_log:
                print(f"\n{'=' * 80}")
                print(f"DISTRIBUTABLE FLOWS - Account {account_id}, Scenario {scenario}")
                print(f"{'=' * 80}")
                print(f"{'Year':<6} {'Ext CF':<14} {'Reserve':<14} {'ΔReserve':<14} {'Capital':<14} "
                      f"{'ΔCapital':<14} {'Profit':<14} {'Distrib':<14} {'PV Dist':<14}")
                print(
                    f"{'-' * 6} {'-' * 14} {'-' * 14} {'-' * 14} {'-' * 14} {'-' * 14} {'-' * 14} {'-' * 14} {'-' * 14}")

            distributable_pvs = []
            prev_reserve = 0.0
            prev_capital = 0.0
            years_logged = 0

            for ext_data in external_projection:
                year = ext_data['year']
                an_proj = ext_data.get('an_proj', year)
                external_cf = ext_data['FLUX_NET']

                current_reserve = reserve_by_year.get(year, 0.0)
                current_capital = capital_by_year.get(year, 0.0)

                if year == 0:
                    profit = external_cf + current_reserve
                    distributable = profit + current_capital
                    delta_reserve = current_reserve
                    delta_capital = current_capital
                else:
                    delta_reserve = current_reserve - prev_reserve
                    delta_capital = current_capital - prev_capital
                    profit = external_cf + delta_reserve
                    distributable = profit + delta_capital

                if year > 0:
                    pv_distributable = distributable / ((1 + HURDLE_RT) ** year)
                else:
                    pv_distributable = distributable

                distributable_pvs.append(pv_distributable)

                # Store detailed year-by-year results
                detailed_results.append({
                    'ID_COMPTE': account_id,
                    'scn_eval': scenario,
                    'an_proj': an_proj,
                    'TX_SURVIE': ext_data['TX_SURVIE'],  # ADD THIS LINE
                    'RENDEMENT': ext_data.get('RENDEMENT', 0.0),  # ADD THIS LINE
                    'FLUX_NET_EXT': external_cf,
                    'RESERVE': current_reserve,
                    'CAPITAL_REQUIREMENT': current_capital,
                    'PROFIT': profit,
                    'FLUX_DISTRIBUABLE': distributable,
                    'VP_FLUX_DISTRIBUABLE_YEARLY': pv_distributable
                })

                if should_log and years_logged < log_config.log_max_years:
                    print(f"{year:<6} {external_cf:>14,.2f} {current_reserve:>14,.2f} {delta_reserve:>14,.2f} "
                          f"{current_capital:>14,.2f} {delta_capital:>14,.2f} {profit:>14,.2f} "
                          f"{distributable:>14,.2f} {pv_distributable:>14,.2f}")
                    years_logged += 1
                elif should_log and years_logged == log_config.log_max_years and year != len(external_projection) - 1:
                    print(f"{'...':<6} {'...':<14} {'...':<14} {'...':<14} {'...':<14} "
                          f"{'...':<14} {'...':<14} {'...':<14} {'...':<14}")
                    years_logged += 1
                elif should_log and year == len(external_projection) - 1:
                    print(f"{year:<6} {external_cf:>14,.2f} {current_reserve:>14,.2f} {delta_reserve:>14,.2f} "
                          f"{current_capital:>14,.2f} {delta_capital:>14,.2f} {profit:>14,.2f} "
                          f"{distributable:>14,.2f} {pv_distributable:>14,.2f}")

                prev_reserve = current_reserve
                prev_capital = current_capital

            total_pv_distributable = sum(distributable_pvs)

            if should_log:
                print(f"{'TOTAL':<6} {'':<14} {'':<14} {'':<14} {'':<14} {'':<14} "
                      f"{'':<14} {'':<14} {total_pv_distributable:>14,.2f}")

            final_results.append({
                'ID_COMPTE': account_id,
                'scn_eval': scenario,
                'VP_FLUX_DISTRIBUABLES': total_pv_distributable
            })

    if log_config.verbose:
        print(f"\n✅ Completed {len(final_results)} distributable flow calculations")

    return final_results, detailed_results


def run_external_calculations_exact(data: Dict, lookups: Dict, NBCPT: int, NB_SC: int,
                                    NB_AN_PROJECTION: int, log_config: LoggingConfig = None) -> List[Dict]:
    """Run external calculations with configurable logging"""

    if log_config is None:
        log_config = LoggingConfig(verbose=False)

    external_results = []
    total_accounts = min(NBCPT, len(data['population']))

    if log_config.verbose:
        print(f"\n🌍 RUNNING EXTERNAL CALCULATIONS")
        print("-" * 50)
        print(f"Processing {total_accounts} accounts × {NB_SC} scenarios = {total_accounts * NB_SC:,} projections")

    account_progress = tqdm(range(total_accounts), desc="Processing accounts", unit="account",
                            disable=not log_config.verbose)

    for account_idx in account_progress:
        account_data = data['population'].iloc[account_idx]
        account_id = account_data['ID_COMPTE']

        if log_config.verbose:
            account_progress.set_postfix({"Account": account_id})

        should_log_account = log_config.should_log_account(account_id)

        if should_log_account and log_config.should_log_scenario(1):
            print(f"\n{'=' * 80}")
            print(f"ACCOUNT {account_id} - INITIAL DATA")
            print(f"{'=' * 80}")
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

        scenario_progress = tqdm(range(1, NB_SC + 1),
                                 desc=f"    Scenarios for {account_id}",
                                 unit="scenario",
                                 leave=False,
                                 disable=not log_config.verbose or should_log_account)

        for scenario in scenario_progress:
            projection = project_cash_flows_exact_sas_logic(
                account_data, scenario, 'EXTERNE', lookups, NB_AN_PROJECTION,
                log_config=log_config
            )

            external_results.append({
                'account_id': account_id,
                'scenario': scenario,
                'projection': projection,
                'account_data': account_data
            })

    if log_config.verbose:
        print(f"\n✅ Completed {len(external_results)} external projections")

    return external_results


def acfc_algorithm_fully_fixed(data_path: str = ".", NBCPT: int = 4, NB_SC: int = 10, NB_AN_PROJECTION: int = 10,
                               NB_SC_INT: int = 10, NB_AN_PROJECTION_INT: int = 10,
                               CHOC_CAPITAL: float = 0.35, HURDLE_RT: float = 0.10,
                               verbose: bool = True,
                               log_account_id: Optional[int] = None,
                               log_scenario: Optional[int] = 1,
                               log_max_years: int = 5,
                               log_internal_scenarios: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fully Fixed ACFC Algorithm with enhanced logging controls

    Parameters:
    -----------
    verbose : bool
        Enable detailed progress logging
    log_account_id : int, optional
        If specified, only show detailed calculations for this account ID (None = all accounts)
    log_scenario : int, optional
        Which scenario to log in detail (None = all scenarios, default = 1)
    log_max_years : int
        Maximum number of years to show in detailed year-by-year logs (default = 5)
    log_internal_scenarios : bool
        Whether to show details for each internal scenario (can be very verbose)

    Returns:
    --------
    Tuple of (summary_df, detailed_df)
    - summary_df: DataFrame with total VP_FLUX_DISTRIBUABLES per account-scenario
    - detailed_df: DataFrame with year-by-year details including all cash flow components
    """

    start_time = time.time()

    log_config = LoggingConfig(
        verbose=verbose,
        log_account_id=log_account_id,
        log_scenario=log_scenario,
        log_max_years=log_max_years,
        log_internal_scenarios=log_internal_scenarios
    )

    if verbose:
        print("\n" + "=" * 80)
        print("🚀 ACFC ALGORITHM - DETAILED EXECUTION LOG")
        print("=" * 80)
        print(f"📊 Configuration:")
        print(f"   • Accounts to process: {NBCPT}")
        print(f"   • External scenarios: {NB_SC}")
        print(f"   • Projection years: {NB_AN_PROJECTION}")
        print(f"   • Internal scenarios: {NB_SC_INT}")
        print(f"   • Internal projection years: {NB_AN_PROJECTION_INT}")
        print(f"   • Capital shock: {CHOC_CAPITAL:.1%}")
        print(f"   • Hurdle rate: {HURDLE_RT:.1%}")
        print(f"\n📝 Logging Configuration:")
        if log_account_id:
            print(f"   • Detailed logging for Account ID: {log_account_id}")
        else:
            print(f"   • Detailed logging for: All accounts")
        if log_scenario:
            print(f"   • Detailed logging for Scenario: {log_scenario}")
        else:
            print(f"   • Detailed logging for: All scenarios")
        print(f"   • Max years shown in detail: {log_max_years}")
        print(f"   • Show internal scenario details: {log_internal_scenarios}")
        print("=" * 80)

    # Phase 1: Data Loading
    data = load_input_data(data_path, verbose=verbose)
    lookups = create_lookup_tables(data, verbose=verbose)

    # Phase 2: External Calculations
    external_results = run_external_calculations_exact(
        data, lookups, NBCPT, NB_SC, NB_AN_PROJECTION, log_config=log_config
    )

    # Phase 3-5: Internal Calculations and Distributable Flows
    final_results, detailed_results = calculate_distributable_flows_exact(
        external_results, lookups, NB_SC_INT, NB_AN_PROJECTION_INT,
        CHOC_CAPITAL, HURDLE_RT, log_config=log_config
    )

    # Phase 6: Output Generation
    if verbose:
        print(f"\n📄 GENERATING OUTPUT")
        print("-" * 50)

    output_df = pd.DataFrame(final_results)
    detailed_df = pd.DataFrame(detailed_results)

    end_time = time.time()
    total_time = end_time - start_time

    if verbose:
        print(f"\n{'=' * 80}")
        print(f"✅ ALGORITHM COMPLETED SUCCESSFULLY!")
        print(f"{'=' * 80}")
        print(f"📈 Results Summary:")
        print(f"   • Total calculations: {len(output_df):,}")
        print(f"   • Detailed year-by-year records: {len(detailed_df):,}")
        print(f"   • Processing time: {total_time:.1f} seconds")
        print(f"   • Average time per calculation: {total_time / len(output_df):.3f} seconds")

        if len(output_df) > 0:
            print(f"   • Mean VP_FLUX_DISTRIBUABLES: ${output_df['VP_FLUX_DISTRIBUABLES'].mean():,.2f}")
            print(f"   • Range: ${output_df['VP_FLUX_DISTRIBUABLES'].min():,.2f} to "
                  f"${output_df['VP_FLUX_DISTRIBUABLES'].max():,.2f}")
            print(f"\n   Results by Account:")
            for account_id in sorted(output_df['ID_COMPTE'].unique()):
                account_data = output_df[output_df['ID_COMPTE'] == account_id]
                print(f"     Account {account_id}: Mean = ${account_data['VP_FLUX_DISTRIBUABLES'].mean():,.2f}, "
                      f"Scenarios = {len(account_data)}")
        print("=" * 80)

    return output_df, detailed_df


# Example usage
if __name__ == "__main__":
    # Example 1: Log only account 1, scenario 1, showing first 10 years
    summary_results, detailed_results = acfc_algorithm_fully_fixed(
        data_path=HERE.joinpath("data_in"),
        NBCPT=1,
        NB_SC=1,
        NB_AN_PROJECTION=100,
        NB_SC_INT=1,
        NB_AN_PROJECTION_INT=100,
        CHOC_CAPITAL=0.35,
        HURDLE_RT=0.10,
        verbose=True,
        log_account_id=1,  # Only log
        log_scenario=1,  # Only log scenario 1
        log_max_years=100,  # Show first 10 years in detail
        log_internal_scenarios=False  # Don't show each internal scenario
    )

    print(f"\n📋 Sample Summary Results:")
    print(summary_results.head(10))

    print(f"\n📋 Sample Detailed Results:")
    print(detailed_results.head(10))

    # Save both outputs
    summary_results.to_csv(HERE.joinpath('test/acfc_results_summary.csv'), index=False)
    detailed_results.to_csv(HERE.joinpath('test/acfc_results_detailed.csv'), index=False)

    print(f"\n💾 Saved outputs:")
    print(f"   • Summary: test/acfc_results_summary.csv ({len(summary_results)} rows)")
    print(f"   • Detailed: test/acfc_results_detailed.csv ({len(detailed_results)} rows)")
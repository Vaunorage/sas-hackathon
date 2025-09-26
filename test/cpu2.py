import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, NamedTuple
from multiprocessing import Pool
from functools import partial
import warnings
import logging

from collections import defaultdict

from paths import HERE

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# State structure to hold all account-scenario states
class AccountScenarioState(NamedTuple):
    account_id: str
    scenario: int
    account_idx: int
    age_deb: int
    initial_data: dict
    # Current state variables
    MT_VM_PROJ: float
    MT_GAR_DECES_PROJ: float
    TX_SURVIE: float
    AGE: int
    is_terminated: bool = False


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


def create_lookup_tables(data: Dict) -> Dict:
    """Create lookup tables"""
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


def hash_find(hash_table: dict, key, default_value=0.0):
    """Mimic SAS hash.find() behavior"""
    return hash_table.get(key, default_value)


def initialize_all_states(data: Dict, nb_accounts: int, nb_scenarios: int) -> List[AccountScenarioState]:
    """Initialize state for all account-scenario combinations"""
    states = []

    for account_idx in range(min(nb_accounts, len(data['population']))):
        account_data = data['population'].iloc[account_idx]

        initial_data = {
            'MT_VM': float(account_data['MT_VM']),
            'MT_GAR_DECES': float(account_data['MT_GAR_DECES']),
            'age_deb': int(account_data['age_deb']),
            'TX_COMM_VENTE': float(account_data.get('TX_COMM_VENTE', 0.0)),
            'FRAIS_ACQUI': float(account_data['FRAIS_ACQUI']),
            'PC_REVENU_FDS': float(account_data['PC_REVENU_FDS']),
            'PC_HONORAIRES_GEST': float(account_data['PC_HONORAIRES_GEST']),
            'TX_COMM_MAINTIEN': float(account_data['TX_COMM_MAINTIEN']),
            'FRAIS_ADMIN': float(account_data['FRAIS_ADMIN']),
            'FREQ_RESET_DECES': float(account_data['FREQ_RESET_DECES']),
            'MAX_RESET_DECES': float(account_data['MAX_RESET_DECES'])
        }

        for scenario in range(1, nb_scenarios + 1):
            # FIXED: Initialize variables as zeros like sequential version
            state = AccountScenarioState(
                account_id=account_data['ID_COMPTE'],
                scenario=scenario,
                account_idx=account_idx,
                age_deb=int(account_data['age_deb']),
                initial_data=initial_data,
                MT_VM_PROJ=0.0,  # FIXED: Start with 0 like sequential version
                MT_GAR_DECES_PROJ=0.0,  # FIXED: Start with 0 like sequential version
                TX_SURVIE=0.0,  # FIXED: Start with 0 like sequential version
                AGE=int(account_data['age_deb']),
                is_terminated=False
            )
            states.append(state)

    return states


def calculate_year_zero_external(state: AccountScenarioState) -> Dict:
    """Calculate year 0 cash flows for external projection"""
    data = state.initial_data

    # FIXED: Initialize variables properly for year 0
    MT_VM_PROJ = float(data['MT_VM'])
    MT_GAR_DECES_PROJ = float(data['MT_GAR_DECES'])
    TX_SURVIE = 1.0
    AGE = state.age_deb

    COMMISSIONS = -data['TX_COMM_VENTE'] * MT_VM_PROJ
    VP_COMMISSIONS = COMMISSIONS

    FRAIS_GEN = -data['FRAIS_ACQUI']
    VP_FRAIS_GEN = FRAIS_GEN

    FLUX_NET = FRAIS_GEN + COMMISSIONS
    VP_FLUX_NET = FLUX_NET

    result = {
        'account_id': state.account_id,
        'scenario': state.scenario,
        'year': 0,
        'an_proj': 0,
        'AGE': AGE,
        'MT_VM_PROJ': MT_VM_PROJ,
        'MT_GAR_DECES_PROJ': MT_GAR_DECES_PROJ,
        'TX_SURVIE': TX_SURVIE,
        'TX_SURVIE_DEB': 1.0,
        'FLUX_NET': FLUX_NET,
        'VP_FLUX_NET': VP_FLUX_NET
    }

    return result


def calculate_year_transition(args) -> Tuple[AccountScenarioState, Dict]:
    """Calculate one year transition for a single account-scenario combination"""
    state, year, lookups, projection_type, fund_shock, start_year = args

    data = state.initial_data

    # Handle year 0 special cases
    if year == 0 and projection_type == "EXTERNE":
        result = calculate_year_zero_external(state)
        # FIXED: Update state with proper year 0 values
        new_state = state._replace(
            MT_VM_PROJ=result['MT_VM_PROJ'],
            MT_GAR_DECES_PROJ=result['MT_GAR_DECES_PROJ'],
            TX_SURVIE=result['TX_SURVIE'],
            AGE=result['AGE']
        )
        return new_state, result
    elif year == 0 and projection_type == "INTERNE":
        # Internal year 0 - apply shock if needed
        if fund_shock > 0:
            new_MT_VM = float(data['MT_VM']) * (1 - fund_shock)
        else:
            new_MT_VM = float(data['MT_VM'])

        new_state = state._replace(
            MT_VM_PROJ=new_MT_VM,
            MT_GAR_DECES_PROJ=float(data['MT_GAR_DECES']),
            TX_SURVIE=data.get('TX_SURVIE_DEB', 1.0),
            AGE=state.age_deb + start_year
        )

        result = {
            'account_id': state.account_id,
            'scenario': state.scenario,
            'year': 0,
            'an_proj': start_year,
            'AGE': state.age_deb + start_year,
            'MT_VM_PROJ': new_MT_VM,
            'MT_GAR_DECES_PROJ': float(data['MT_GAR_DECES']),
            'TX_SURVIE': data.get('TX_SURVIE_DEB', 1.0),
            'TX_SURVIE_DEB': data.get('TX_SURVIE_DEB', 1.0),
            'FLUX_NET': 0.0,
            'VP_FLUX_NET': 0.0
        }
        return new_state, result

    # FIXED: Check termination conditions like sequential version
    if state.TX_SURVIE == 0 or state.MT_VM_PROJ == 0:
        new_state = state._replace(is_terminated=True)
        return new_state, None

    # Regular year calculations
    if projection_type == "INTERNE":
        new_age = state.age_deb + start_year + year
        an_proj = start_year + year
    else:
        new_age = state.age_deb + year
        an_proj = year

    # Fund value projection
    MT_VM_DEB = state.MT_VM_PROJ

    RENDEMENT_rate = hash_find(lookups['returns'], (an_proj, state.scenario, projection_type), 0.0)
    RENDEMENT = MT_VM_DEB * RENDEMENT_rate

    FRAIS = -(MT_VM_DEB + RENDEMENT / 2) * data['PC_REVENU_FDS']

    new_MT_VM_PROJ = max(0, state.MT_VM_PROJ + RENDEMENT + FRAIS)

    # Death benefit guarantee reset logic
    new_MT_GAR_DECES_PROJ = state.MT_GAR_DECES_PROJ
    if data['FREQ_RESET_DECES'] == 1 and new_age <= data['MAX_RESET_DECES']:
        new_MT_GAR_DECES_PROJ = max(state.MT_GAR_DECES_PROJ, new_MT_VM_PROJ)

    # Survival probability calculation
    QX = hash_find(lookups['mortality'], new_age, 0.0)
    WX = hash_find(lookups['lapse'], an_proj, 0.0)

    TX_SURVIE_DEB = state.TX_SURVIE
    new_TX_SURVIE = TX_SURVIE_DEB * (1 - QX) * (1 - WX)

    # Cash flow calculations
    REVENUS = -FRAIS * TX_SURVIE_DEB
    FRAIS_GEST = -(MT_VM_DEB + RENDEMENT / 2) * data['PC_HONORAIRES_GEST'] * TX_SURVIE_DEB
    COMMISSIONS = -(MT_VM_DEB + RENDEMENT / 2) * data['TX_COMM_MAINTIEN'] * TX_SURVIE_DEB
    FRAIS_GEN = -data['FRAIS_ADMIN'] * TX_SURVIE_DEB
    PMT_GARANTIE = -max(0, new_MT_GAR_DECES_PROJ - new_MT_VM_PROJ) * QX * TX_SURVIE_DEB

    FLUX_NET = REVENUS + FRAIS_GEST + COMMISSIONS + FRAIS_GEN + PMT_GARANTIE

    # Present value calculations
    TX_ACTU = hash_find(lookups['discount_ext'], an_proj, 1.0)
    VP_FLUX_NET = FLUX_NET * TX_ACTU

    # Internal scenario adjustment
    if projection_type == "INTERNE" and start_year > 0:
        TX_ACTU_INT = hash_find(lookups['discount_int'], start_year, 1.0)
        if TX_ACTU_INT != 0:
            VP_FLUX_NET = VP_FLUX_NET / TX_ACTU_INT

    # Create new state
    new_state = state._replace(
        MT_VM_PROJ=new_MT_VM_PROJ,
        MT_GAR_DECES_PROJ=new_MT_GAR_DECES_PROJ,
        TX_SURVIE=new_TX_SURVIE,
        AGE=new_age,
        is_terminated=(new_TX_SURVIE == 0 or new_MT_VM_PROJ == 0)  # FIXED: Use == like sequential
    )

    result = {
        'account_id': state.account_id,
        'scenario': state.scenario,
        'year': year,
        'an_proj': an_proj,
        'AGE': new_age,
        'MT_VM_PROJ': new_MT_VM_PROJ,
        'MT_GAR_DECES_PROJ': new_MT_GAR_DECES_PROJ,
        'TX_SURVIE': new_TX_SURVIE,
        'TX_SURVIE_DEB': TX_SURVIE_DEB,
        'FLUX_NET': FLUX_NET,
        'VP_FLUX_NET': VP_FLUX_NET
    }

    return new_state, result



def run_projection_for_states(states: List[AccountScenarioState], lookups: Dict, nb_years: int,
                              projection_type: str, fund_shock: float = 0.0, start_year: int = 0,
                              use_multiprocessing: bool = True) -> Dict[str, List[Dict]]:
    """Run projection for a list of states and return results grouped by account-scenario"""

    all_results = defaultdict(list)

    for year in range(nb_years + 1):
        # Prepare arguments for parallel processing
        args_list = [(state, year, lookups, projection_type, fund_shock, start_year)
                     for state in states if not state.is_terminated]

        if use_multiprocessing and len(args_list) > 1:
            with Pool() as pool:
                results = pool.map(calculate_year_transition, args_list)
        else:
            results = [calculate_year_transition(args) for args in args_list]

        # Update states and collect results
        new_states = []
        state_idx = 0
        for i, state in enumerate(states):
            if state.is_terminated:
                new_states.append(state)
            else:
                new_state, result = results[state_idx]
                new_states.append(new_state)
                if result is not None:
                    key = f"{result['account_id']}_{result['scenario']}"
                    all_results[key].append(result)
                state_idx += 1

        states = new_states

    return all_results


def run_internal_calculations_for_year(args) -> Tuple[str, int, float]:
    """Run internal calculations for one year of one external account-scenario"""
    account_id, scenario, year, year_data, original_account_data, lookups, nb_sc_int, nb_an_projection_int, calculation_type, choc_capital = args

    if year == 0:
        return account_id, year, 0.0

    # Get state at this year from external projection
    fund_value = year_data['MT_VM_PROJ']
    death_benefit = year_data['MT_GAR_DECES_PROJ']
    survival_prob = year_data['TX_SURVIE']

    if survival_prob == 0 or fund_value == 0:  # FIXED: Use == like sequential
        return account_id, year, 0.0

    # FIXED: Use original account data instead of hardcoded values
    account_data_for_internal = original_account_data.copy()
    account_data_for_internal.update({
        'MT_VM': fund_value,
        'MT_GAR_DECES': death_benefit,
        'TX_SURVIE_DEB': survival_prob
    })

    # Apply shock for capital calculations
    fund_shock = choc_capital if calculation_type == 'CAPITAL' else 0.0

    # Run internal scenarios from this year forward
    internal_scenarios_sum = []

    for internal_scenario in range(1, nb_sc_int + 1):
        # Create initial state for internal projection
        initial_state = AccountScenarioState(
            account_id=account_id,
            scenario=internal_scenario,
            account_idx=0,
            age_deb=int(original_account_data['age_deb']),
            initial_data=account_data_for_internal,
            MT_VM_PROJ=0.0,  # FIXED: Start with 0 like sequential
            MT_GAR_DECES_PROJ=0.0,  # FIXED: Start with 0 like sequential
            TX_SURVIE=0.0,  # FIXED: Start with 0 like sequential
            AGE=int(original_account_data['age_deb']) + year,
            is_terminated=False
        )

        # Run internal projection
        internal_states = [initial_state]
        internal_results = run_projection_for_states(
            internal_states, lookups, nb_an_projection_int, 'INTERNE',
            fund_shock, start_year=year, use_multiprocessing=False
        )

        if internal_results:
            key = f"{account_id}_{internal_scenario}"
            if key in internal_results:
                total_vp = sum([row['VP_FLUX_NET'] for row in internal_results[key]])
                internal_scenarios_sum.append(total_vp)

    # Calculate mean across internal scenarios
    if internal_scenarios_sum:
        result = np.mean(internal_scenarios_sum)
    else:
        result = 0.0

    return account_id, year, result


def calculate_complete_distributable_flows(external_results: Dict[str, List[Dict]], original_data: Dict, lookups: Dict,
                                           nb_sc_int: int, nb_an_projection_int: int,
                                           choc_capital: float, hurdle_rt: float,
                                           use_multiprocessing: bool = True) -> List[Dict]:
    """Calculate complete distributable flows including reserves and capital"""

    final_results = []

    # FIXED: Create mapping of account_id to original account data
    account_data_map = {}
    for _, account_row in original_data['population'].iterrows():
        account_data_map[account_row['ID_COMPTE']] = {
            'age_deb': int(account_row['age_deb']),
            'MT_VM': float(account_row['MT_VM']),
            'MT_GAR_DECES': float(account_row['MT_GAR_DECES']),
            'TX_COMM_VENTE': float(account_row.get('TX_COMM_VENTE', 0.0)),
            'FRAIS_ACQUI': float(account_row['FRAIS_ACQUI']),
            'PC_REVENU_FDS': float(account_row['PC_REVENU_FDS']),
            'PC_HONORAIRES_GEST': float(account_row['PC_HONORAIRES_GEST']),
            'TX_COMM_MAINTIEN': float(account_row['TX_COMM_MAINTIEN']),
            'FRAIS_ADMIN': float(account_row['FRAIS_ADMIN']),
            'FREQ_RESET_DECES': float(account_row['FREQ_RESET_DECES']),
            'MAX_RESET_DECES': float(account_row['MAX_RESET_DECES'])
        }

    for account_scenario_key, external_projection in external_results.items():
        account_id, scenario = account_scenario_key.split('_')
        scenario = int(scenario)

        print(f"Processing {account_id} scenario {scenario}...")

        # FIXED: Get original account data
        original_account_data = account_data_map[float(account_id)]

        # Prepare arguments for parallel internal calculations
        reserve_args = []
        capital_args = []

        for year_data in external_projection:
            year = year_data['year']

            reserve_args.append((account_id, scenario, year, year_data, original_account_data, lookups,
                                 nb_sc_int, nb_an_projection_int, 'RESERVE', choc_capital))
            capital_args.append((account_id, scenario, year, year_data, original_account_data, lookups,
                                 nb_sc_int, nb_an_projection_int, 'CAPITAL', choc_capital))

        # Run reserve calculations
        if use_multiprocessing and len(reserve_args) > 1:
            with Pool() as pool:
                reserve_results = pool.map(run_internal_calculations_for_year, reserve_args)
        else:
            reserve_results = [run_internal_calculations_for_year(args) for args in reserve_args]

        # Run capital calculations
        if use_multiprocessing and len(capital_args) > 1:
            with Pool() as pool:
                capital_results = pool.map(run_internal_calculations_for_year, capital_args)
        else:
            capital_results = [run_internal_calculations_for_year(args) for args in capital_args]

        # Convert to dictionaries
        reserve_by_year = {year: value for _, year, value in reserve_results}
        capital_raw_by_year = {year: value for _, year, value in capital_results}

        # Calculate capital as difference from reserves
        capital_by_year = {}
        for year in capital_raw_by_year:
            reserve_value = reserve_by_year.get(year, 0.0)
            capital_value = capital_raw_by_year[year] - reserve_value
            capital_by_year[year] = capital_value

        # Calculate distributable cash flows
        distributable_pvs = []
        prev_reserve = 0.0
        prev_capital = 0.0

        for ext_data in external_projection:
            year = ext_data['year']
            external_cf = ext_data['FLUX_NET']

            current_reserve = reserve_by_year.get(year, 0.0)
            current_capital = capital_by_year.get(year, 0.0)

            if year == 0:
                profit = external_cf + current_reserve
                distributable = profit + current_capital
            else:
                profit = external_cf + (current_reserve - prev_reserve)
                distributable = profit + (current_capital - prev_capital)

            # Present value at hurdle rate
            if year > 0:
                pv_distributable = distributable / ((1 + hurdle_rt) ** year)
            else:
                pv_distributable = distributable

            distributable_pvs.append(pv_distributable)

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


def parallelized_acfc_algorithm_fixed(data_path: str = ".", nb_accounts: int = 4, nb_scenarios: int = 10,
                                      nb_years: int = 10, nb_sc_int: int = 10, nb_an_projection_int: int = 10,
                                      choc_capital: float = 0.35, hurdle_rt: float = 0.10,
                                      use_multiprocessing: bool = True) -> pd.DataFrame:
    """
    FIXED Parallelized ACFC Algorithm - now matches sequential version exactly
    """

    print("Phase 1: Loading input data...")
    data = load_input_data(data_path)

    print("Phase 2: Creating lookup tables...")
    lookups = create_lookup_tables(data)

    print("Phase 3: Running parallelized external projections...")
    states = initialize_all_states(data, nb_accounts, nb_scenarios)
    external_results = run_projection_for_states(
        states, lookups, nb_years, 'EXTERNE', use_multiprocessing=use_multiprocessing
    )

    print("Phase 4: Running internal calculations and distributable flows...")
    final_results = calculate_complete_distributable_flows(
        external_results, data, lookups, nb_sc_int, nb_an_projection_int,
        choc_capital, hurdle_rt, use_multiprocessing
    )

    print("Phase 5: Converting to DataFrame...")
    output_df = pd.DataFrame(final_results)

    print(f"FIXED parallelized algorithm finished. Generated {len(output_df)} results.")
    return output_df


# Example usage
if __name__ == "__main__":
    results = parallelized_acfc_algorithm_fixed(
        data_path=HERE.joinpath("data_in"),  # Update with your data path
        nb_accounts=30,
        nb_scenarios=50,
        nb_years=50,
        nb_sc_int=50,
        nb_an_projection_int=50,
        choc_capital=0.35,
        hurdle_rt=0.10,
        use_multiprocessing=True
    )

    print("\nFinal Results:")
    print(results)
    results.to_csv(HERE.joinpath('test/cpu2.csv'))
    print(f"\nMean VP_FLUX_DISTRIBUABLES: {results['VP_FLUX_DISTRIBUABLES'].mean():.2f}")
    print(f"Range: {results['VP_FLUX_DISTRIBUABLES'].min():.2f} to {results['VP_FLUX_DISTRIBUABLES'].max():.2f}")
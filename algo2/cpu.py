import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Any
from datetime import datetime
import multiprocessing as mp
from functools import partial
from paths import HERE

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default configuration, can be overridden by arguments in run_projection
CONFIG = {
    'nb_thread_tot': 12,
    'NBCPT': 9999999,
    'NB_SC': 100,
    'NB_AN_PROJECTION': 100,
    'FREQ_EVAL': 12,
    'NO_COMPTE_SORTIE': 6522,
    'NO_SCN_SORTIE': 2,
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def parse_percentage(value):
    """Convert percentage string to float (e.g., '1.5%' -> 0.015, '(0.53%)' -> -0.0053)."""
    if pd.isna(value):
        return 0.0
    if isinstance(value, str):
        # Remove whitespace
        value = value.strip()

        # Check for parentheses (negative values)
        is_negative = False
        if value.startswith('(') and value.endswith(')'):
            is_negative = True
            value = value[1:-1].strip()  # Remove parentheses

        # Check for percentage sign
        if '%' in value:
            value = value.replace('%', '').strip()
            numeric_value = float(value) / 100.0
        else:
            numeric_value = float(value)

        # Apply negative if parentheses were present
        if is_negative:
            numeric_value = -numeric_value

        return numeric_value
    return float(value)


def clean_numeric(df, columns):
    """Clean numeric columns, handling percentage strings."""
    for col in columns:
        if col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].apply(parse_percentage)
    return df


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all column names to uppercase for consistency."""
    df.columns = df.columns.str.upper()
    return df


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_all_data(data_path: Path) -> Dict[str, pd.DataFrame]:
    """Load all CSV files into memory with semicolon delimiter."""
    print("Loading data files...")

    data = {}

    # Load all required tables
    data['population'] = pd.read_csv(data_path.joinpath("POPULATION.csv"), sep=';', encoding='utf-8')
    data['mortalite'] = pd.read_csv(data_path.joinpath("MORTALITE.csv"), sep=';', encoding='utf-8')
    data['rendements'] = pd.read_csv(data_path.joinpath("RENDEMENTS.csv"), sep=';', encoding='utf-8')
    data['depots_futurs'] = pd.read_csv(data_path.joinpath("DEPOTS_FUTURS.csv"), sep=';', encoding='utf-8')
    data['frais_admin'] = pd.read_csv(data_path.joinpath("FRAIS_ADMIN.csv"), sep=';', encoding='utf-8')
    data['min_ferr'] = pd.read_csv(data_path.joinpath("MIN_FERR.csv"), sep=';', encoding='utf-8')
    data['tx_lapse_part'] = pd.read_csv(data_path.joinpath("TX_LAPSE_PART.csv"), sep=';', encoding='utf-8')

    # NEW: Load missing tables
    data['tx_lapse_tot'] = pd.read_csv(data_path.joinpath("TX_LAPSE_TOT.csv"), sep=';', encoding='utf-8')
    data['acquisition'] = pd.read_csv(data_path.joinpath("ACQUISITION.csv"), sep=';', encoding='utf-8')
    data['coussins_escap'] = pd.read_csv(data_path.joinpath("COUSSINS_ESCAP.csv"), sep=';', encoding='utf-8')

    # Normalize all column names to uppercase for consistency
    print("  Normalizing column names...")
    for key in data:
        data[key] = normalize_column_names(data[key])

    # Load POPULATION
    print("  Loading POPULATION...")
    pct_cols = [col for col in data['population'].columns if col.startswith('PC_') or col.startswith('TAUX_')]
    data['population'] = clean_numeric(data['population'], pct_cols)

    # Load MORTALITE
    print("  Loading MORTALITE...")
    data['mortalite'] = clean_numeric(data['mortalite'], ['QX'])

    # Load RENDEMENTS
    print("  Loading RENDEMENTS...")
    rend_cols = ['FORWARD_RATE', 'AJUST_FORWARD_RATE_VM_0', 'RENDDEX_AN', 'RENDMM_AN',
                 'RENDTSX_AN', 'RENDSP500_AN', 'RENDEAFE_AN']
    data['rendements'] = clean_numeric(data['rendements'], rend_cols)

    # Load DEPOTS_FUTURS
    print("  Loading DEPOTS_FUTURS...")
    data['depots_futurs'] = clean_numeric(data['depots_futurs'], ['PC_DEPOT_ANNUEL'])

    # Load FRAIS_ADMIN
    print("  Loading FRAIS_ADMIN...")
    data['frais_admin'] = clean_numeric(data['frais_admin'], ['FRAIS'])

    # Load MIN_FERR
    print("  Loading MIN_FERR...")
    data['min_ferr'] = clean_numeric(data['min_ferr'], ['MIN_FERR'])

    # Load TX_LAPSE_PART
    print("  Loading TX_LAPSE_PART...")
    lapse_cols = ['TX_LAPSE_PART_MIN', 'TX_LAPSE_PART_MAX']
    data['tx_lapse_part'] = clean_numeric(data['tx_lapse_part'], lapse_cols)

    # NEW: Load TX_LAPSE_TOT
    print("  Loading TX_LAPSE_TOT...")
    # TX_LAPSE values are already in decimal format (not percentages)
    # FACT_DIM is also decimal
    # Just ensure they're numeric
    for col in ['TX_LAPSE_TOT_MIN', 'TX_LAPSE_TOT_MAX', 'FACT_DIM']:
        if col in data['tx_lapse_tot'].columns:
            data['tx_lapse_tot'][col] = pd.to_numeric(data['tx_lapse_tot'][col], errors='coerce').fillna(0)

    # NEW: Load ACQUISITION
    print("  Loading ACQUISITION...")
    acq_cols = ['PC_COMMISSION_VENTE_RF', 'PC_COMMISSION_VENTE_AC',
                'PC_COMMISSION_MAINTIEN_RF', 'PC_COMMISSION_MAINTIEN_AC',
                'PC_FRAIS_AN_AC', 'PC_FRAIS_AN_RF']
    data['acquisition'] = clean_numeric(data['acquisition'], acq_cols)

    # NEW: Load COUSSINS_ESCAP
    print("  Loading COUSSINS_ESCAP...")
    # Only TX_ and FACTEUR_ columns need percentage parsing, BASE_ columns are integers
    coussin_cols = [col for col in data['coussins_escap'].columns
                    if col.startswith('TX_') or col.startswith('FACTEUR_')]
    data['coussins_escap'] = clean_numeric(data['coussins_escap'], coussin_cols)
    # BASE columns should be integers
    base_cols = [col for col in data['coussins_escap'].columns if col.startswith('BASE_')]
    for col in base_cols:
        data['coussins_escap'][col] = data['coussins_escap'][col].astype(int)

    # Filter data based on config
    data['rendements'] = data['rendements'][
        (data['rendements']['SCN_EVAL'] <= CONFIG['NB_SC']) &
        (data['rendements']['AN_EVAL'] <= CONFIG['NB_AN_PROJECTION'])
        ]

    data['population'] = data['population'][
        data['population']['ID_COMPTE'] <= CONFIG['NBCPT']
        ]

    print(f"Loaded {len(data['population'])} accounts")
    return data


# =============================================================================
# HASH TABLE / FAST LOOKUP CREATION
# =============================================================================

def create_mortality_lookup(df: pd.DataFrame) -> Dict[Tuple, float]:
    """Create O(1) lookup for mortality rates."""
    lookup = {}
    for _, row in df.iterrows():
        key = (int(row['I_SEXE']), int(row['AGE_MORTALITE']),
               int(row['ANNEE_REELLE']), int(row['I_PRODUIT_REGR']))
        lookup[key] = float(row['QX'])
    return lookup


def create_returns_lookup(df: pd.DataFrame) -> Dict[Tuple, Dict]:
    """Create O(1) lookup for investment returns."""
    lookup = {}
    for _, row in df.iterrows():
        key = (int(row['SCN_EVAL']), int(row['AN_EVAL']), int(row['MOIS_EVAL']))
        lookup[key] = {
            'FORWARD_RATE': float(row['FORWARD_RATE']),
            'AJUST_FORWARD_RATE_VM_0': float(row['AJUST_FORWARD_RATE_VM_0']),
            'RENDDEX_AN': float(row['RENDDEX_AN']),
            'RENDMM_AN': float(row['RENDMM_AN']),
            'RENDTSX_AN': float(row['RENDTSX_AN']),
            'RENDSP500_AN': float(row['RENDSP500_AN']),
            'RENDEAFE_AN': float(row['RENDEAFE_AN']),
        }
    return lookup


def create_min_ferr_lookup(df: pd.DataFrame) -> Dict[int, float]:
    """Create O(1) lookup for minimum RRIF withdrawals."""
    return {int(row['AGE']): float(row['MIN_FERR'])
            for _, row in df.iterrows()}


def create_lapse_part_lookup(df: pd.DataFrame) -> Dict[Tuple, Dict]:
    """Create O(1) lookup for partial lapse rates."""
    lookup = {}
    for _, row in df.iterrows():
        key = (int(row['AGE']), int(row['ID_LAPSE']),
               int(row['I_REGIME_2']), int(row['LAPSE_NIV_PART']))
        lookup[key] = {
            'TX_LAPSE_PART_MIN': float(row['TX_LAPSE_PART_MIN']),
            'TX_LAPSE_PART_MAX': float(row['TX_LAPSE_PART_MAX']),
        }
    return lookup


def create_lapse_tot_lookup(df: pd.DataFrame) -> Dict[Tuple, Dict]:
    """Create O(1) lookup for total lapse rates."""
    lookup = {}
    for _, row in df.iterrows():
        key = (int(row['DUREE_MAX10']), int(row['ID_LAPSE']), int(row['LAPSE_NIV_TOT']))
        lookup[key] = {
            'TX_LAPSE_TOT_MIN': float(row['TX_LAPSE_TOT_MIN']),
            'TX_LAPSE_TOT_MAX': float(row['TX_LAPSE_TOT_MAX']),
            'FACT_DIM': float(row['FACT_DIM']),
        }
    return lookup


def create_fees_lookup(df: pd.DataFrame) -> Dict[Tuple, float]:
    """Create O(1) lookup for fees."""
    return {(int(row['ID_PRODUIT']), int(row['ANNEE_REELLE'])): float(row['FRAIS'])
            for _, row in df.iterrows()}


def create_deposits_lookup(df: pd.DataFrame) -> Dict[Tuple, Dict]:
    """Create O(1) lookup for future deposits."""
    lookup = {}
    for _, row in df.iterrows():
        key = (int(row['DUREE_MAX10']), int(row['ID_DEPOT']))
        lookup[key] = {
            'PC_DEPOT_ANNUEL': float(row['PC_DEPOT_ANNUEL']),
            'VAR_DEPOT_FCT': int(row['VAR_DEPOT_FCT']),
            'AGE_MAX_DEPOT': int(row['AGE_MAX_DEPOT']),
            'I_EVEN_CESSE_DEPOT': int(row['I_EVEN_CESSE_DEPOT']),
        }
    return lookup


def create_acquisition_lookup(df: pd.DataFrame) -> Dict[Tuple, Dict]:
    """Create O(1) lookup for acquisition costs."""
    lookup = {}
    for _, row in df.iterrows():
        key = (int(row['DUREE_MAX10']), int(row['ID_ACQUI']))
        lookup[key] = {
            'PC_COMMISSION_VENTE_RF': float(row['PC_COMMISSION_VENTE_RF']),
            'PC_COMMISSION_VENTE_AC': float(row['PC_COMMISSION_VENTE_AC']),
            'PC_COMMISSION_MAINTIEN_RF': float(row['PC_COMMISSION_MAINTIEN_RF']),
            'PC_COMMISSION_MAINTIEN_AC': float(row['PC_COMMISSION_MAINTIEN_AC']),
            'PC_FRAIS_AN_AC': float(row['PC_FRAIS_AN_AC']),
            'PC_FRAIS_AN_RF': float(row['PC_FRAIS_AN_RF']),
        }
    return lookup


def create_coussins_lookup(df: pd.DataFrame) -> Dict[Tuple, Dict]:
    """Create O(1) lookup for ESCAP cushions."""
    lookup = {}
    for _, row in df.iterrows():
        key = (int(row['CODE_CAT_PRODUIT']), int(row['CAT_COUSSIN_1']), int(row['CAT_COUSSIN_2']))
        lookup[key] = {
            'BASE_PASSIF_REDRESSE': int(row['BASE_PASSIF_REDRESSE']),
            'TX_PASSIF_REDRESSE': float(row['TX_PASSIF_REDRESSE']),
            'BASE_COUSSIN_CREDIT': int(row['BASE_COUSSIN_CREDIT']),
            'TX_COUSSIN_CREDIT': float(row['TX_COUSSIN_CREDIT']),
            'BASE_COUSSIN_MARCHE': int(row['BASE_COUSSIN_MARCHE']),
            'TX_COUSSIN_MARCHE': float(row['TX_COUSSIN_MARCHE']),
            'BASE_COUSSIN_DEPENSE': int(row['BASE_COUSSIN_DEPENSE']),
            'TX_COUSSIN_DEPENSE': float(row['TX_COUSSIN_DEPENSE']),
            'BASE_COUSSIN_DECHEANCE': int(row['BASE_COUSSIN_DECHEANCE']),
            'TX_COUSSIN_DECHEANCE': float(row['TX_COUSSIN_DECHEANCE']),
            'BASE_COUSSIN_MORTALITE': int(row['BASE_COUSSIN_MORTALITE']),
            'TX_COUSSIN_MORTALITE': float(row['TX_COUSSIN_MORTALITE']),
            'BASE_COUSSIN_DEPOT': int(row['BASE_COUSSIN_DEPOT']),
            'TX_COUSSIN_DEPOT': float(row['TX_COUSSIN_DEPOT']),
            'FACTEUR_AGE_80': float(row['FACTEUR_AGE_80']),
            'FACTEUR_AGE_90': float(row['FACTEUR_AGE_90']),
        }
    return lookup


def create_all_lookups(data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Create all hash table equivalents."""
    print("Creating fast lookup structures...")

    lookups = {
        'mortality': create_mortality_lookup(data['mortalite']),
        'returns': create_returns_lookup(data['rendements']),
        'min_ferr': create_min_ferr_lookup(data['min_ferr']),
        'lapse_part': create_lapse_part_lookup(data['tx_lapse_part']),
        'lapse_tot': create_lapse_tot_lookup(data['tx_lapse_tot']),
        'fees': create_fees_lookup(data['frais_admin']),
        'deposits': create_deposits_lookup(data['depots_futurs']),
        'acquisition': create_acquisition_lookup(data['acquisition']),
        'coussins': create_coussins_lookup(data['coussins_escap']),
    }

    print(f"  Created {len(lookups['mortality'])} mortality entries")
    print(f"  Created {len(lookups['returns'])} return scenarios")
    print(f"  Created {len(lookups['lapse_tot'])} total lapse entries")
    print(f"  Created {len(lookups['acquisition'])} acquisition entries")
    print(f"  Created {len(lookups['coussins'])} cushion entries")
    print("Lookup structures created")
    return lookups


# =============================================================================
# HELPER CALCULATION FUNCTIONS
# =============================================================================

def calculate_age(birth_year: int, birth_month: int, current_year: int, current_month: int) -> int:
    """Calculate age at a given point in time."""
    age = current_year - birth_year
    if current_month < birth_month:
        age -= 1
    return max(age, 1)


def calculate_lapse_level_total(vm_vg_ratio: float) -> int:
    """Determine lapse level based on moneyness (returns 1, 2, or 3)."""
    if vm_vg_ratio <= 0.5:
        return 1
    elif vm_vg_ratio <= 0.75:
        return 2
    else:
        return 3


def calculate_lapse_level_partial(vm_vg_ratio: float) -> int:
    """Determine partial lapse level (returns 1, 2, or 3)."""
    if vm_vg_ratio <= 0.5:
        return 1
    elif vm_vg_ratio <= 0.75:
        return 2
    else:
        return 3


# =============================================================================
# LEVEL 2: CREATE EXPANDED TIMELINE
# =============================================================================

def create_expanded_timeline(account: pd.Series, config: Dict) -> pd.DataFrame:
    """
    LEVEL 2: Create the skeleton of scenario × year × month combinations.
    Equivalent to lines 103-125 in SAS code.
    """
    rows = []

    for scn_eval in range(1, config['NB_SC'] + 1):
        for an_eval in range(0, config['NB_AN_PROJECTION'] + 1):
            for mois_simul in range(1, config['FREQ_EVAL'] + 1):

                # Calculate real year and month
                annee_reelle = int(account['ANNEE_EVALUATION_INI']) + an_eval - 1
                mois_eval = mois_simul * 12 // config['FREQ_EVAL']

                # Calculate age
                age = calculate_age(
                    int(account['ANNEE_NAIS']),
                    int(account['MOIS_NAIS']),
                    annee_reelle,
                    mois_eval
                )

                # Keep only relevant periods (line 120 in SAS)
                keep = (
                        age <= account['AGE_FIN_CONTRAT'] and
                        (an_eval > 1 or
                         (an_eval == 1 and mois_eval >= account['MOIS_EVALUATION_INI']) or
                         (an_eval == 0 and mois_eval == 12))
                )

                if keep:
                    rows.append({
                        'ID_COMPTE': int(account['ID_COMPTE']),
                        'SCN_EVAL': scn_eval,
                        'AN_EVAL': an_eval,
                        'MOIS_EVAL': mois_eval,
                        'ANNEE_REELLE': annee_reelle,
                        'AGE': age,
                        'I_SEXE': int(account['I_SEXE']),
                        'I_PRODUIT_REGR': int(account['I_PRODUIT_REGR']),
                        'ID_PRODUIT': int(account['ID_PRODUIT']),
                        'ID_LAPSE': int(account['ID_LAPSE']),
                        'I_REGIME_2': int(account['I_REGIME_2']),
                        'ID_DEPOT': int(account['ID_DEPOT']),
                        'ID_ACQUI': int(account.get('ID_ACQUI', 1)),
                        'AGE_ECH_MIN': int(account['AGE_ECH_MIN']),
                        'AGE_FIN_CONTRAT': int(account['AGE_FIN_CONTRAT']),
                        'MOIS_NAIS': int(account['MOIS_NAIS']),
                        'AGE_DECAISSEMENT': int(account['AGE_DECAISSEMENT']),
                    })

    return pd.DataFrame(rows)


# =============================================================================
# LEVEL 3: PROJECTION CALCULATIONS WITH STATE
# =============================================================================

def initialize_state(account: pd.Series) -> Dict[str, float]:
    """Initialize state variables for projection from account data."""
    return {
        'MT_VM_PROJ': float(account['MT_VM']),
        'MT_GAR_DECES_PROJ': float(account['MT_GAR_DECES']),
        'MT_GAR_ECH_PROJ': float(account['MT_GAR_ECH']),
        'MT_SRG_PROJ': float(account['MT_SRG']),
        'MT_BCB_PROJ': float(account.get('MT_BCB', 0)),
        'MT_DEX_PROJ': float(account['MT_DEX']),
        'MT_MM_PROJ': float(account['MT_MM']),
        'MT_TSX_PROJ': float(account['MT_TSX']),
        'MT_SP500_PROJ': float(account['MT_SP500']),
        'MT_EAFE_PROJ': float(account['MT_EAFE']),
        'MT_BONI_DECES_PROJ': float(account.get('MT_BONI_DECES', 0)),
        'MT_MRV_MRG_MRA_PROJ': float(account.get('MT_MRV_MRG_MRA', 0)),
        'TAUX_MRV_MRG_MRA_PROJ': float(account.get('TAUX_MRV_MRG_MRA', 0)),
        'MT_MIN_FERR_PROJ': 0.0,
        'ANNEE_ECH_PROJ': int(account.get('ANNEE_ECH', 9999)),
        'MOIS_ECH_PROJ': int(account.get('MOIS_ECH', 12)),
        'TX_SURVIE': 1.0,
        'TX_SURVIE_DEB': 1.0,
        'TX_ACTUALISATION': 1.0,
        'PC_HONORAIRES_GEST': float(account['PC_HONORAIRES_GEST']),
        'PC_FRAIS_GARANTIE': float(account['PC_FRAIS_GARANTIE']),
        'PC_GAR_DECES_1': float(account['PC_GAR_DECES_1']),
        'PC_BONI_DECES': float(account.get('PC_BONI_DECES', 0)),
        'PC_RFG': float(account.get('PC_RFG', 0)),
        'PC_REVENU_FDS': float(account.get('PC_REVENU_FDS', 0)),
        'PC_GAR_ECH': float(account.get('PC_GAR_ECH', 0)),
        'PC_GAR_ECH_DEP_FUT': float(account.get('PC_GAR_ECH_DEP_FUT', 0)),
        'AJUSTEMENT_COMMISSION': float(account.get('AJUSTEMENT_COMMISSION', 1.0)),
        'MT_RF': float(account.get('MT_RF', 0)),
        'MT_VM': float(account['MT_VM']),
    }

def lookup_mortality(lookups: Dict, row: Dict, state: Dict) -> float:
    """Lookup mortality rate with fallback."""
    mois_nais = int(row['MOIS_NAIS'])
    mois_eval = int(row['MOIS_EVAL'])

    # Calculate age_MORTALITE same as SAS
    month_diff = mois_nais - mois_eval
    if month_diff <= 0:
        month_diff += 12

    if month_diff <= 6:
        age_mort = int(row['AGE']) + 1
    else:
        age_mort = int(row['AGE'])

    age_mort = min(age_mort, 120)
    key = (row['I_SEXE'], age_mort, int(row['ANNEE_REELLE']), row['I_PRODUIT_REGR'])
    return lookups['mortality'].get(key, 0.001)


def lookup_returns(lookups: Dict, row: Dict, state: Dict) -> Dict[str, float]:
    """Lookup investment returns with fallback to zero returns."""
    key = (int(row['SCN_EVAL']), int(row['AN_EVAL']), int(row['MOIS_EVAL']))
    returns = lookups['returns'].get(key, {
        'FORWARD_RATE': 0.0,
        'AJUST_FORWARD_RATE_VM_0': 0.0,
        'RENDDEX_AN': 0.0,
        'RENDMM_AN': 0.0,
        'RENDTSX_AN': 0.0,
        'RENDSP500_AN': 0.0,
        'RENDEAFE_AN': 0.0,
    })

    # Adjust forward rate if VM is 0 (for RGS that become non-liquid)
    if state['MT_VM_PROJ'] == 0:
        returns['FORWARD_RATE'] += returns['AJUST_FORWARD_RATE_VM_0']

    return returns


def apply_investment_returns(state: Dict, returns: Dict, AJUST_NOUV_AFFAIRES: float) -> Dict:
    """Apply investment returns to each asset class using CONTINUOUS COMPOUNDING."""
    # FIXED: Use exp() for continuous compounding to match SAS
    state['MT_DEX_PROJ'] *= np.exp(returns['RENDDEX_AN'] * AJUST_NOUV_AFFAIRES)
    state['MT_MM_PROJ'] *= np.exp(returns['RENDMM_AN'] * AJUST_NOUV_AFFAIRES)
    state['MT_TSX_PROJ'] *= np.exp(returns['RENDTSX_AN'] * AJUST_NOUV_AFFAIRES)
    state['MT_SP500_PROJ'] *= np.exp(returns['RENDSP500_AN'] * AJUST_NOUV_AFFAIRES)
    state['MT_EAFE_PROJ'] *= np.exp(returns['RENDEAFE_AN'] * AJUST_NOUV_AFFAIRES)

    state['MT_VM_AV_RETRAIT_FRAIS'] = (state['MT_DEX_PROJ'] + state['MT_MM_PROJ'] +
                                       state['MT_TSX_PROJ'] + state['MT_SP500_PROJ'] +
                                       state['MT_EAFE_PROJ'])
    return state


def update_discount_factor(state: Dict, forward_rate: float, AJUST_NOUV_AFFAIRES: float) -> Dict:
    """Update the discount factor using continuous compounding."""
    state['TX_ACTUALISATION'] = state['TX_ACTUALISATION'] * np.exp(-forward_rate * AJUST_NOUV_AFFAIRES)
    return state


def calculate_lapse_rates(state: Dict, lookups: Dict, row: Dict, account: pd.Series, freq: int,
                          AJUST_NOUV_AFFAIRES: float, duree_max10: int) -> Tuple[float, float, float]:
    """
    Calculate total and partial lapse rates with proper interpolation.

    Args:
        duree_max10: Duration from issue date (1-10), calculated from dates not counter
    """
    # Check if VM is 0 for RGS - no lapse
    if state['MT_VM_PROJ'] == 0:
        return 0.0, 0.0, 0.0

    # Calculate VM/VG ratio (moneyness)
    pc_gar_ech = state['PC_GAR_ECH']
    pc_gar_deces = state['PC_GAR_DECES_1']

    # Handle division by zero gracefully
    ratio1 = pc_gar_ech / max(state['MT_GAR_ECH_PROJ'], 0.01) if state['MT_GAR_ECH_PROJ'] > 0 else 9999
    ratio2 = pc_gar_deces / max(state['MT_BONI_DECES_PROJ'] + state['MT_GAR_DECES_PROJ'], 0.01)
    ratio3 = 1 / max(state['MT_SRG_PROJ'], 0.01) if state['MT_SRG_PROJ'] > 0 else 9999

    vm_vg_ratio = min(10, (state['MT_VM_PROJ'] + state['MT_VM_AV_RETRAIT_FRAIS']) / 2 * min(ratio1, ratio2, ratio3))

    # ===== TOTAL LAPSE CALCULATION =====
    lapse_niv_tot = calculate_lapse_level_total(vm_vg_ratio)

    key_tot = (duree_max10, row['ID_LAPSE'], lapse_niv_tot)
    lapse_tot_info = lookups['lapse_tot'].get(key_tot, {
        'TX_LAPSE_TOT_MIN': 0.0, 'TX_LAPSE_TOT_MAX': 0.0, 'FACT_DIM': 1.0
    })

    tx_lapse_tot_min = lapse_tot_info['TX_LAPSE_TOT_MIN']
    tx_lapse_tot_max = lapse_tot_info['TX_LAPSE_TOT_MAX']
    fact_dim = lapse_tot_info['FACT_DIM']

    # Interpolate total lapse
    if tx_lapse_tot_min == tx_lapse_tot_max:
        lapse_tot = tx_lapse_tot_min
    else:
        if lapse_niv_tot == 1:
            interpolation = (vm_vg_ratio - 0.00) / 0.5 if vm_vg_ratio > 0 else 0
        elif lapse_niv_tot == 2:
            interpolation = (vm_vg_ratio - 0.5) / (0.75 - 0.5)
        else:  # lapse_niv_tot == 3
            interpolation = (vm_vg_ratio - 0.75) / (999.99 - 0.75)
        lapse_tot = interpolation * (tx_lapse_tot_max - tx_lapse_tot_min) + tx_lapse_tot_min

    if row['AGE'] >= row['AGE_DECAISSEMENT']:
        lapse_tot *= fact_dim

    # ===== PARTIAL LAPSE CALCULATION =====
    lapse_niv_part = calculate_lapse_level_partial(vm_vg_ratio)
    key_part = (int(row['AGE']), row['ID_LAPSE'], row['I_REGIME_2'], lapse_niv_part)
    lapse_part_info = lookups['lapse_part'].get(key_part, {
        'TX_LAPSE_PART_MIN': 0.0, 'TX_LAPSE_PART_MAX': 0.0
    })

    tx_lapse_part_min = lapse_part_info['TX_LAPSE_PART_MIN']
    tx_lapse_part_max = lapse_part_info['TX_LAPSE_PART_MAX']

    # Interpolate partial lapse
    if tx_lapse_part_min == tx_lapse_part_max:
        lapse_part = tx_lapse_part_min
    else:
        if lapse_niv_part == 1:
            interpolation = (vm_vg_ratio - 0.00) / 0.5 if vm_vg_ratio > 0 else 0
        elif lapse_niv_part == 2:
            interpolation = (vm_vg_ratio - 0.5) / (0.75 - 0.5)
        else:  # lapse_niv_part == 3
            interpolation = (vm_vg_ratio - 0.75) / (999.99 - 0.75)
        lapse_part = interpolation * (tx_lapse_part_max - tx_lapse_part_min) + tx_lapse_part_min

    # Convert annual rates to period rates
    exponent = (1 / freq) * AJUST_NOUV_AFFAIRES
    lapse = 1 - (1 - lapse_tot - lapse_part) ** exponent

    return lapse_tot, lapse_part, lapse

def process_deposits(state: Dict, lookups: Dict, row: Dict, account: pd.Series, freq: int,
                     duree_max10: int) -> Tuple[Dict, float]:
    """
    Add deposits and update guarantees.

    Args:
        duree_max10: Duration from issue date (1-10), calculated from dates not counter
    """
    key = (duree_max10, row['ID_DEPOT'])
    deposit_info = lookups['deposits'].get(key, {})

    pc_depot_annuel = deposit_info.get('PC_DEPOT_ANNUEL', 0.0)
    var_depot_fct = deposit_info.get('VAR_DEPOT_FCT', 0)
    age_max_depot = deposit_info.get('AGE_MAX_DEPOT', 999)
    i_even_cesse_depot = deposit_info.get('I_EVEN_CESSE_DEPOT', 0)

    age_retrait = row['AGE'] + 1
    age_decaissement = row['AGE_DECAISSEMENT']
    mt_tpa_depot = account.get('MT_TPA_DEPOT', 0)

    # Check if deposits should cease
    if (pc_depot_annuel == 0 or
            (i_even_cesse_depot == 1 and age_retrait >= age_decaissement) or
            (age_max_depot < row['AGE']) or
            (state['MT_VM_PROJ'] <= 0 and row['I_PRODUIT_REGR'] == 0)):
        depot_futur = 0.0
    elif mt_tpa_depot > 0:
        depot_futur = mt_tpa_depot
    else:
        # Use VM or normalized death guarantee
        if var_depot_fct == 1:
            base = state['MT_VM_PROJ']
        else:
            base = account['MT_GAR_DECES'] / state['PC_GAR_DECES_1']
        depot_futur = base * pc_depot_annuel

    # Adjust for frequency
    depot_futur = depot_futur / freq

    if depot_futur > 0:
        # Allocate proportionally to current allocation
        total_vm = state['MT_VM_PROJ']
        if total_vm > 0:
            state['MT_DEX_PROJ'] += depot_futur * (state['MT_DEX_PROJ'] / total_vm)
            state['MT_MM_PROJ'] += depot_futur * (state['MT_MM_PROJ'] / total_vm)
            state['MT_TSX_PROJ'] += depot_futur * (state['MT_TSX_PROJ'] / total_vm)
            state['MT_SP500_PROJ'] += depot_futur * (state['MT_SP500_PROJ'] / total_vm)
            state['MT_EAFE_PROJ'] += depot_futur * (state['MT_EAFE_PROJ'] / total_vm)

        # Update guarantees
        state['MT_GAR_DECES_PROJ'] += depot_futur
        state['MT_GAR_ECH_PROJ'] += depot_futur * state['PC_GAR_ECH_DEP_FUT']

        if state['MT_SRG_PROJ'] > 0:
            state['MT_SRG_PROJ'] += depot_futur

    return state, depot_futur

def calculate_mrv_amount(state: Dict, row: Dict, account: pd.Series, freq: int) -> Dict:
    """Calculate MRV/MRG/MRA amount for RGS products."""
    if row['I_PRODUIT_REGR'] != 1:
        return state

    age_retrait = row['AGE'] + 1
    age_mrv_permis = account.get('AGE_MRV_PERMIS', 65)
    table_taux_mrv = account.get('TABLE_TAUX_MRV_MRG_MRA', 1)
    mois_eval = row['MOIS_EVAL']

    # Check if withdrawals should cease
    base_amount = state['MT_SRG_PROJ'] if table_taux_mrv == 1 else state['MT_VM_PROJ']
    if age_retrait < age_mrv_permis and base_amount == 0:
        state['MT_MRV_MRG_MRA_PROJ'] = 0
        return state

    # Only recalculate at end of year
    if mois_eval != 12 / CONFIG['FREQ_EVAL']:
        return state

    # RGS 2.1 logic
    if table_taux_mrv == 2:
        should_reinit = (age_retrait == max(age_mrv_permis, row['AGE_DECAISSEMENT']) or
                         (state['MT_SRG_PROJ'] == state['MT_VM_PROJ'] and state['MT_VM_PROJ'] != 0))

        if should_reinit:
            # Determine rate based on age
            if age_retrait < 60:
                state['TAUX_MRV_MRG_MRA_PROJ'] = 0.03
            elif age_retrait < 65:
                state['TAUX_MRV_MRG_MRA_PROJ'] = 0.035
            elif age_retrait < 70:
                state['TAUX_MRV_MRG_MRA_PROJ'] = 0.04
            elif age_retrait < 75:
                state['TAUX_MRV_MRG_MRA_PROJ'] = 0.0425
            else:
                state['TAUX_MRV_MRG_MRA_PROJ'] = 0.05

            state['MT_MRV_MRG_MRA_PROJ'] = state['TAUX_MRV_MRG_MRA_PROJ'] * state['MT_SRG_PROJ']
        else:
            # Can't decrease
            state['MT_MRV_MRG_MRA_PROJ'] = max(state['MT_MRV_MRG_MRA_PROJ'],
                                               state['TAUX_MRV_MRG_MRA_PROJ'] * state['MT_SRG_PROJ'])
    else:
        # RGS 1 and 2
        if age_retrait == age_mrv_permis:
            state['MT_MRV_MRG_MRA_PROJ'] = state['TAUX_MRV_MRG_MRA_PROJ'] * state['MT_SRG_PROJ']
        else:
            state['MT_MRV_MRG_MRA_PROJ'] = max(state['MT_MRV_MRG_MRA_PROJ'],
                                               state['TAUX_MRV_MRG_MRA_PROJ'] * state['MT_SRG_PROJ'])

    # Handle excess withdrawals (simplified - only at end of first year)
    m_mt_mrv_excedent = account.get('M_MT_MRV_EXCEDENT', 0)
    mois_evaluation_ini = account.get('MOIS_EVALUATION_INI', 1)
    an_eval = row['AN_EVAL']

    if (m_mt_mrv_excedent > 1 and
            mois_evaluation_ini != 12 / CONFIG['FREQ_EVAL'] and
            an_eval == 2 and
            mois_eval == 12 / CONFIG['FREQ_EVAL']):
        state['MT_MRV_MRG_MRA_PROJ'] = min(state['MT_MRV_MRG_MRA_PROJ'],
                                           state['TAUX_MRV_MRG_MRA_PROJ'] * max(state['MT_SRG_PROJ'],
                                                                                state['MT_VM_PROJ']))

    return state


def calculate_withdrawal(state: Dict, lookups: Dict, row: Dict, account: pd.Series, freq: int) -> float:
    """Calculate total withdrawal amount."""
    age_retrait = row['AGE'] + 1
    age_decaissement = row['AGE_DECAISSEMENT']
    mois_nais = row['MOIS_NAIS']
    mois_eval = row['MOIS_EVAL']

    # No withdrawal if age not reached or VM is 0 for regular products
    if (age_retrait < age_decaissement or
            (age_retrait == age_decaissement and mois_eval >= mois_nais) or
            (state['MT_VM_PROJ'] <= 0 and row['I_PRODUIT_REGR'] == 0)):
        return 0.0

    # Minimum FERR withdrawal
    age = int(row['AGE'])
    min_ferr_rate = lookups['min_ferr'].get(age, 0.0)

    # Calculate MIN_FERR_PROJ at start of year or first evaluation month
    if (row['AN_EVAL'] == 1 and mois_eval == account.get('MOIS_EVALUATION_INI', 1)) or mois_eval == 12 / CONFIG[
        'FREQ_EVAL']:
        state['MT_MIN_FERR_PROJ'] = state['MT_VM_PROJ'] * min_ferr_rate

    min_withdrawal = state['MT_MIN_FERR_PROJ']

    # Get withdrawal parameters
    var_retrait_fct = account.get('VAR_RETRAIT_FCT', 1)
    mt_tpa_retrait = account.get('MT_TPA_RETRAIT', 0)
    pc_retrait_age = account.get('PC_RETRAIT_AGE', 1.0)
    mt_retrait_max = account.get('MT_RETRAIT_MAX', 999999999)

    # Calculate retrait based on VAR_RETRAIT_FCT
    if var_retrait_fct == 1:
        retrait = mt_tpa_retrait if mt_tpa_retrait > 0 else state['MT_VM_PROJ'] * pc_retrait_age
    elif var_retrait_fct == 2:
        if mt_tpa_retrait > min_withdrawal:
            retrait = mt_tpa_retrait
        else:
            retrait = min_withdrawal * max(pc_retrait_age, 1)
    elif var_retrait_fct == 3:
        retrait = max(min_withdrawal, state['MT_MRV_MRG_MRA_PROJ']) * pc_retrait_age
    else:
        retrait = 0.0

    # Apply maximum and frequency adjustment
    retrait = min(retrait, mt_retrait_max) / freq

    return retrait


def apply_management_fees(state: Dict, freq: int, AJUST_NOUV_AFFAIRES: float) -> Dict:
    """Apply management fees (RFG)."""
    state['MT_VM_AV_RETRAIT'] = state['MT_VM_AV_RETRAIT_FRAIS'] * np.exp(-state['PC_RFG'] / freq * AJUST_NOUV_AFFAIRES)
    return state


def calculate_guarantee_fees(state: Dict, row: Dict, freq: int, AJUST_NOUV_AFFAIRES: float, tx_survie_deb: float) -> \
Tuple[float, float]:
    """Calculate guarantee fees (primes garanties)."""
    i_frais_sur_srg = row.get('I_FRAIS_SUR_SRG', 0)

    if i_frais_sur_srg == 0:
        primes_garanties = min(state['MT_VM_AV_RETRAIT'] * state['PC_FRAIS_GARANTIE'] / freq * AJUST_NOUV_AFFAIRES,
                               state['MT_VM_AV_RETRAIT']) * tx_survie_deb
    else:
        primes_garanties = min(state['MT_SRG_PROJ'] * state['PC_FRAIS_GARANTIE'] / freq * AJUST_NOUV_AFFAIRES,
                               state['MT_VM_AV_RETRAIT']) * tx_survie_deb

    vp_primes_garanties = primes_garanties * state['TX_ACTUALISATION']

    # Update VM after guarantee fees
    if i_frais_sur_srg == 0:
        state['MT_VM_AV_RETRAIT'] = max(state['MT_VM_AV_RETRAIT'] -
                                        state['MT_VM_AV_RETRAIT'] * state[
                                            'PC_FRAIS_GARANTIE'] / freq * AJUST_NOUV_AFFAIRES, 0)
    else:
        state['MT_VM_AV_RETRAIT'] = max(state['MT_VM_AV_RETRAIT'] -
                                        state['MT_SRG_PROJ'] * state['PC_FRAIS_GARANTIE'] / freq * AJUST_NOUV_AFFAIRES,
                                        0)

    return primes_garanties, vp_primes_garanties


def update_guarantees_for_withdrawal(state: Dict, retrait: float) -> Tuple[Dict, float]:
    """Update guarantees and SRG proportionally to withdrawal."""
    mt_vm_av_retrait = state['MT_VM_AV_RETRAIT']
    mt_srg_av_retrait = state['MT_SRG_PROJ']

    if mt_vm_av_retrait <= retrait:
        state['MT_GAR_ECH_PROJ'] = 0
        state['MT_GAR_DECES_PROJ'] = 0
        state['MT_BONI_DECES_PROJ'] = 0
    else:
        proportion = 1 - retrait / mt_vm_av_retrait
        state['MT_GAR_ECH_PROJ'] *= proportion
        state['MT_GAR_DECES_PROJ'] *= proportion
        state['MT_BONI_DECES_PROJ'] *= proportion
        state['MT_SRG_PROJ'] = max(state['MT_SRG_PROJ'] - retrait, 0)

    # VM after withdrawal
    mt_vm_ap_retrait = max(mt_vm_av_retrait - retrait, 0)

    return state, mt_vm_ap_retrait, mt_srg_av_retrait


def process_deposits_and_update_vm(state: Dict, depot_futur: float, mt_vm_ap_retrait: float) -> Dict:
    """Process deposits and update final VM."""
    if mt_vm_ap_retrait > 0:
        state['MT_VM_PROJ'] = mt_vm_ap_retrait + depot_futur
    else:
        state['MT_VM_PROJ'] = mt_vm_ap_retrait

    return state


def calculate_mrv_benefit(state: Dict, retrait: float, mt_vm_av_retrait: float,
                          row: Dict, tx_survie_deb: float) -> Tuple[float, float]:
    """Calculate MRV benefit for guaranteed income."""
    if row['I_PRODUIT_REGR'] == 1:
        prest_mrv = -max(retrait - mt_vm_av_retrait, 0) * tx_survie_deb
    else:
        prest_mrv = 0

    vp_prest_mrv = prest_mrv * state['TX_ACTUALISATION']
    return prest_mrv, vp_prest_mrv


def calculate_death_benefit(state: Dict, qx: float, tx_survie_deb: float,
                            mt_vm_ap_retrait_depot: float) -> Tuple[float, float]:
    """Calculate death benefit."""
    prest_deces = qx * -max(0, state['MT_GAR_DECES_PROJ'] + state['MT_BONI_DECES_PROJ'] -
                            mt_vm_ap_retrait_depot) * tx_survie_deb
    vp_prest_deces = prest_deces * state['TX_ACTUALISATION']
    return prest_deces, vp_prest_deces


def process_maturity_benefit(state: Dict, row: Dict, account: pd.Series, mt_vm_ap_retrait: float) -> Tuple[
    Dict, float, float]:
    """Process maturity benefit if it occurs."""
    annee_ech_proj = state['ANNEE_ECH_PROJ']
    mois_ech_proj = state['MOIS_ECH_PROJ']
    age_fin_contrat = row['AGE_FIN_CONTRAT']
    mois_nais = row['MOIS_NAIS']
    freq = CONFIG['FREQ_EVAL']

    # Check if maturity occurs
    maturity_occurs = False
    if row['ANNEE_REELLE'] == annee_ech_proj and row['MOIS_EVAL'] == mois_ech_proj:
        maturity_occurs = True
    elif row['AGE'] == age_fin_contrat:
        target_month = 12 if mois_nais == 12 / freq else mois_nais - 12 / freq
        if row['MOIS_EVAL'] == target_month:
            maturity_occurs = True

    if maturity_occurs:
        prest_ech = -max(0, state['MT_GAR_ECH_PROJ'] - mt_vm_ap_retrait) * state['TX_SURVIE']

        # Update maturity parameters
        nb_an_ech = account.get('NB_AN_ECH', 10)
        state['ANNEE_ECH_PROJ'] = annee_ech_proj + nb_an_ech
        state['MOIS_ECH_PROJ'] = row['MOIS_EVAL']

        # Update VM and guarantees
        state['MT_VM_PROJ'] = mt_vm_ap_retrait + max(0, state['MT_GAR_ECH_PROJ'] - mt_vm_ap_retrait)
        state['MT_GAR_ECH_PROJ'] = state['MT_VM_PROJ'] * state['PC_GAR_ECH']

        # Reset death guarantee if applicable
        i_reset_deces_ech = account.get('I_RESET_DECES_ECH', 0)
        if i_reset_deces_ech == 1:
            state['MT_GAR_DECES_PROJ'] = state['MT_VM_PROJ'] * state['PC_GAR_DECES_1']

        # Apply renewal rate
        age_max_renouv_ech = account.get('AGE_MAX_RENOUV_ECH', 999)
        pc_renouv_ech = account.get('PC_RENOUV_ECH', 1.0)
        if row['AGE'] > age_max_renouv_ech:
            pc_renouv_ech = 0
        state['TX_SURVIE'] *= pc_renouv_ech
    else:
        prest_ech = 0

    vp_prest_ech = prest_ech * state['TX_ACTUALISATION']
    return state, prest_ech, vp_prest_ech


def update_death_guarantee_adjustments(state: Dict, row: Dict, account: pd.Series, freq: int) -> Dict:
    """Apply adjustments to death guarantee (CPGIA)."""
    ajustement_mensuel_gar = account.get('AJUSTEMENT_MENSUEL_GAR', 0)
    state['MT_GAR_DECES_PROJ'] = state['MT_GAR_DECES_PROJ'] - ajustement_mensuel_gar * 12 / freq
    return state


def process_srg_bcb_resets(state: Dict, row: Dict, account: pd.Series) -> Dict:
    """Process SRG/BCB resets for RGS products."""
    if row['I_PRODUIT_REGR'] != 1:
        return state

    annee_cotis = account.get('ANNEE_COTIS', account.get('ANNEE_EVALUATION_INI', row['ANNEE_REELLE']))
    mois_cotis = account.get('MOIS_COTIS', 1)
    freq_reset_srg = account.get('FREQ_RESET_SRG', 3)
    max_reset_srg = account.get('MAX_RESET_SRG', 80)

    # Check if SRG reset should occur
    if (row['AGE'] < max_reset_srg and
            state['MT_SRG_PROJ'] < state['MT_VM_PROJ'] and
            row['ANNEE_REELLE'] > annee_cotis):

        years_since_issue = row['ANNEE_REELLE'] - annee_cotis
        if (int(years_since_issue / freq_reset_srg) == years_since_issue / freq_reset_srg and
                row['MOIS_EVAL'] == mois_cotis):
            state['MT_SRG_PROJ'] = state['MT_VM_PROJ']
            state['MT_BCB_PROJ'] = max(state['MT_BCB_PROJ'], state['MT_VM_PROJ'])

    # Bonus to SRG if not yet in decumulation
    age_decaissement = row['AGE_DECAISSEMENT']
    pc_boni_srg = account.get('PC_BONI_SRG', 0)
    if row['AGE'] < age_decaissement and row['MOIS_EVAL'] == 12:
        state['MT_SRG_PROJ'] = state['MT_SRG_PROJ'] + pc_boni_srg * state['MT_BCB_PROJ']

    return state


def process_death_guarantee_resets(state: Dict, row: Dict, account: pd.Series) -> Dict:
    """Process automatic death guarantee resets."""
    annee_cotis = account.get('ANNEE_COTIS', account.get('ANNEE_EVALUATION_INI', row['ANNEE_REELLE']))
    mois_cotis = account.get('MOIS_COTIS', 1)
    freq_reset_deces = account.get('FREQ_RESET_DECES', 3)
    max_reset_deces = account.get('MAX_RESET_DECES', 80)
    mois_nais = row['MOIS_NAIS']
    freq = CONFIG['FREQ_EVAL']

    # Check if death guarantee reset should occur
    should_reset = False
    if (row['AGE'] < max_reset_deces and
            (state['MT_GAR_DECES_PROJ'] + state['MT_BONI_DECES_PROJ']) < (
                    state['MT_VM_PROJ'] * state['PC_GAR_DECES_1']) and
            row['ANNEE_REELLE'] > annee_cotis):

        years_since_issue = row['ANNEE_REELLE'] - annee_cotis

        # Regular reset
        if (int(years_since_issue / freq_reset_deces) == years_since_issue / freq_reset_deces and
                row['MOIS_EVAL'] == mois_cotis):
            should_reset = True

        # Final reset at max age
        target_month = 12 if mois_nais == 12 / freq else mois_nais - 12 / freq
        if row['AGE'] == max_reset_deces - 1 and row['MOIS_EVAL'] == target_month:
            should_reset = True

    if should_reset:
        state['MT_GAR_DECES_PROJ'] = state['MT_VM_PROJ'] * state['PC_GAR_DECES_1']
        state['MT_BONI_DECES_PROJ'] = 0

    return state


def process_facultative_maturity_reset(state: Dict, row: Dict, account: pd.Series) -> Dict:
    """Process facultative maturity guarantee reset (semi-annual)."""
    i_reset_facul_ech = account.get('I_RESET_FACUL_ECH', 0)
    max_reset_facul_ech = account.get('MAX_RESET_FACUL_ECH', 80)
    ratio_vm_vg_reset_ech = account.get('RATIO_VM_VG_RESET_ECH', 1.0)
    nb_an_ech = account.get('NB_AN_ECH', 10)
    age_ech_min = row['AGE_ECH_MIN']
    annee_nais = account.get('ANNEE_NAIS', 1950)
    mois_nais = row['MOIS_NAIS']

    # Only in June and December
    if row['MOIS_EVAL'] not in [6, 12]:
        return state

    # Check conditions for facultative reset
    if (i_reset_facul_ech == 1 and
            row['AGE'] <= max_reset_facul_ech and
            state['MT_GAR_ECH_PROJ'] > 0 and
            (state['MT_VM_PROJ'] * state['PC_GAR_ECH']) >= ratio_vm_vg_reset_ech * state['MT_GAR_ECH_PROJ']):

        state['MT_GAR_ECH_PROJ'] = max(state['MT_GAR_ECH_PROJ'], state['MT_VM_PROJ'] * state['PC_GAR_ECH'])
        state['ANNEE_ECH_PROJ'] = max(row['ANNEE_REELLE'] + nb_an_ech, annee_nais + age_ech_min)

        if state['ANNEE_ECH_PROJ'] == annee_nais + age_ech_min:
            state['MOIS_ECH_PROJ'] = mois_nais
        else:
            state['MOIS_ECH_PROJ'] = row['MOIS_EVAL']

    return state


def process_death_guarantee_age_change(state: Dict, row: Dict, account: pd.Series) -> Dict:
    """Change death guarantee percentage if age threshold reached."""
    age_chang_deces = account.get('AGE_CHANG_DECES', 999)
    pc_gar_deces_2 = account.get('PC_GAR_DECES_2', state['PC_GAR_DECES_1'])
    mois_nais = row['MOIS_NAIS']
    freq = CONFIG['FREQ_EVAL']

    target_month = 12 if mois_nais == 12 / freq else mois_nais - 12 / freq

    if row['AGE'] == age_chang_deces - 1 and row['MOIS_EVAL'] == target_month:
        state['MT_GAR_DECES_PROJ'] = state['MT_GAR_DECES_PROJ'] * pc_gar_deces_2 / state['PC_GAR_DECES_1']
        state['PC_GAR_DECES_1'] = pc_gar_deces_2

    return state


def rebalance_portfolio(state: Dict, account: pd.Series) -> Dict:
    """Rebalance portfolio to original allocation."""
    mt_vm_orig = state['MT_VM']

    if mt_vm_orig > 0 and state['MT_VM_PROJ'] > 0:
        state['MT_SP500_PROJ'] = state['MT_VM_PROJ'] * account['MT_SP500'] / mt_vm_orig
        state['MT_TSX_PROJ'] = state['MT_VM_PROJ'] * account['MT_TSX'] / mt_vm_orig
        state['MT_EAFE_PROJ'] = state['MT_VM_PROJ'] * account['MT_EAFE'] / mt_vm_orig
        state['MT_DEX_PROJ'] = state['MT_VM_PROJ'] * account['MT_DEX'] / mt_vm_orig
        state['MT_MM_PROJ'] = state['MT_VM_PROJ'] * account['MT_MM'] / mt_vm_orig

    return state


def calculate_acquisition_costs(state: Dict, lookups: Dict, row: Dict, account: pd.Series,
                                depot_futur: float, lapse: float, qx: float,
                                mt_vm_ap_retrait: float, tx_survie_deb: float,
                                freq: int, AJUST_NOUV_AFFAIRES: float,
                                duree_max10: int) -> Dict[str, float]:
    """
    Calculate acquisition costs with proper lookups.

    Args:
        duree_max10: Duration from issue date (1-10), calculated from dates not counter
    """
    # Check if VM before withdrawal fees is 0
    if state['MT_VM_AV_RETRAIT_FRAIS'] == 0:
        return {
            'COMM_VENTE': 0.0,
            'VP_COMM_VENTE': 0.0,
            'FRAIS_ACQUIS': 0.0,
            'VP_FRAIS_ACQUIS': 0.0,
            'PC_COMMISSION_MAINTIEN': 0.0,
        }

    # Lookup acquisition parameters
    key = (duree_max10, row['ID_ACQUI'])
    acq_info = lookups['acquisition'].get(key, {
        'PC_COMMISSION_VENTE_RF': 0.0,
        'PC_COMMISSION_VENTE_AC': 0.0,
        'PC_COMMISSION_MAINTIEN_RF': 0.0,
        'PC_COMMISSION_MAINTIEN_AC': 0.0,
        'PC_FRAIS_AN_AC': 0.0,
        'PC_FRAIS_AN_RF': 0.0,
    })

    # Get allocation info
    mt_vm = state['MT_VM']
    mt_rf = state['MT_RF']
    ajustement_commission = state['AJUSTEMENT_COMMISSION']

    if mt_vm > 0:
        # Calculate weighted average rates
        pc_commission_vente = ((acq_info['PC_COMMISSION_VENTE_AC'] * (mt_vm - mt_rf) / mt_vm +
                                acq_info['PC_COMMISSION_VENTE_RF'] * mt_rf / mt_vm) *
                               ajustement_commission)

        pc_commission_maintien = ((acq_info['PC_COMMISSION_MAINTIEN_AC'] * (mt_vm - mt_rf) / mt_vm +
                                   acq_info['PC_COMMISSION_MAINTIEN_RF'] * mt_rf / mt_vm) *
                                  ajustement_commission)

        pc_frais_an = (acq_info['PC_FRAIS_AN_AC'] * (mt_vm - mt_rf) / mt_vm +
                       acq_info['PC_FRAIS_AN_RF'] * mt_rf / mt_vm)
    else:
        pc_commission_vente = 0.0
        pc_commission_maintien = 0.0
        pc_frais_an = 0.0

    comm_vente = -pc_commission_vente * depot_futur * state['TX_SURVIE']
    vp_comm_vente = comm_vente * state['TX_ACTUALISATION']

    # Calculate recovery from lapses
    frais_acquis = pc_frais_an * mt_vm_ap_retrait * lapse * tx_survie_deb * (1 - qx)
    vp_frais_acquis = frais_acquis * state['TX_ACTUALISATION']

    return {
        'COMM_VENTE': comm_vente,
        'VP_COMM_VENTE': vp_comm_vente,
        'FRAIS_ACQUIS': frais_acquis,
        'VP_FRAIS_ACQUIS': vp_frais_acquis,
        'PC_COMMISSION_MAINTIEN': pc_commission_maintien,
    }

def calculate_fees(state: Dict, lookups: Dict, row: Dict, tx_survie_deb: float,
                   freq: int, AJUST_NOUV_AFFAIRES: float, pc_commission_maintien: float) -> Dict[str, float]:
    """Calculate all ongoing fees."""
    # Fixed fees
    fee_key = (row['ID_PRODUIT'], int(row['ANNEE_REELLE']))
    fixed_fee_annual = lookups['fees'].get(fee_key, 0.0)

    # Only charge if VM > 0 or there are MRV benefits
    if state['MT_VM_AV_RETRAIT'] <= 0:
        # Check if there are MRV benefits (would need VP_PREST_MRV, but we can use a proxy)
        # For now, assume no fees if VM is 0
        frais_fixes = 0.0
    else:
        frais_fixes = -fixed_fee_annual / freq * AJUST_NOUV_AFFAIRES * tx_survie_deb

    vp_frais_fixes = frais_fixes * state['TX_ACTUALISATION']

    # Management fees (honoraires)
    hon_gest = -state['MT_VM_AV_RETRAIT_FRAIS'] * (
                np.exp(state['PC_HONORAIRES_GEST'] / freq * AJUST_NOUV_AFFAIRES) - 1) * tx_survie_deb
    vp_hon_gest = hon_gest * state['TX_ACTUALISATION']

    # Maintenance commission
    comm_maintien = -state['MT_VM_AV_RETRAIT_FRAIS'] * (
                np.exp(pc_commission_maintien / freq * AJUST_NOUV_AFFAIRES) - 1) * tx_survie_deb
    vp_comm_maintien = comm_maintien * state['TX_ACTUALISATION']

    # Variable premiums
    primes_variables = (state['MT_VM_AV_RETRAIT_FRAIS'] *
                        np.exp(-(state['PC_RFG'] - state['PC_REVENU_FDS']) / freq * AJUST_NOUV_AFFAIRES) *
                        -(np.exp(-state['PC_REVENU_FDS'] / freq * AJUST_NOUV_AFFAIRES) - 1) *
                        tx_survie_deb)
    vp_primes_variables = primes_variables * state['TX_ACTUALISATION']

    return {
        'FRAIS_FIXES': frais_fixes,
        'VP_FRAIS_FIXES': vp_frais_fixes,
        'HON_GEST': hon_gest,
        'VP_HON_GEST': vp_hon_gest,
        'COMM_MAINTIEN': comm_maintien,
        'VP_COMM_MAINTIEN': vp_comm_maintien,
        'PRIMES_VARIABLES': primes_variables,
        'VP_PRIMES_VARIABLES': vp_primes_variables,
    }


def calculate_escap_cushions(state: Dict, lookups: Dict, row: Dict, account: pd.Series, freq: int,
                             duree_max10: int) -> Dict[str, float]:
    """
    Calculate ESCAP cushions.

    Args:
        duree_max10: Duration from issue date (1-10), calculated from dates not counter
    """
    id_produit = row['ID_PRODUIT']

    # Determine CODE_CAT_PRODUIT
    if id_produit == 22:
        code_cat_produit = 0
    elif id_produit in [12, 13, 14, 15, 16]:
        code_cat_produit = 1
    elif id_produit in [17, 18, 19, 20, 21]:
        code_cat_produit = 2
    elif id_produit == 6:
        code_cat_produit = 3
    elif id_produit in [4, 7]:
        code_cat_produit = 4
    elif id_produit in [5, 8]:
        code_cat_produit = 5
    elif id_produit in [2, 3]:
        code_cat_produit = 6
    else:
        code_cat_produit = 7

    # Determine CAT_COUSSIN_1 (based on % fixed income)
    if state['MT_VM_PROJ'] > 0:
        pct_rf = (state['MT_DEX_PROJ'] + state['MT_MM_PROJ']) / state['MT_VM_PROJ']
    else:
        pct_rf = 0

    if code_cat_produit in [0, 6]:
        cat_coussin_1 = 0
    elif code_cat_produit == 7 and pct_rf < 0.5:
        cat_coussin_1 = 4
    elif code_cat_produit == 7:
        cat_coussin_1 = 5
    elif pct_rf < 1 / 3:
        cat_coussin_1 = 1
    elif pct_rf < 2 / 3:
        cat_coussin_1 = 2
    else:
        cat_coussin_1 = 3

    # Determine CAT_COUSSIN_2 (based on moneyness for RGS, duration for others)
    # Calculate VM/VG ratio
    pc_gar_ech = state['PC_GAR_ECH']
    pc_gar_deces = state['PC_GAR_DECES_1']
    ratio1 = pc_gar_ech / max(state['MT_GAR_ECH_PROJ'], 0.01) if state['MT_GAR_ECH_PROJ'] > 0 else 999
    ratio2 = pc_gar_deces / max(state['MT_BONI_DECES_PROJ'] + state['MT_GAR_DECES_PROJ'], 0.01)
    ratio3 = 1 / max(state['MT_SRG_PROJ'], 0.01) if state['MT_SRG_PROJ'] > 0 else 999
    vm_vg_ratio = min(10, (state['MT_VM_PROJ'] + state['MT_VM_AV_RETRAIT_FRAIS']) / 2 * min(ratio1, ratio2, ratio3))

    if code_cat_produit == 7 and vm_vg_ratio < 0.7:
        cat_coussin_2 = 4
    elif code_cat_produit == 7 and vm_vg_ratio < 0.9:
        cat_coussin_2 = 5
    elif code_cat_produit == 7:
        cat_coussin_2 = 6
    elif duree_max10 <= 3:
        cat_coussin_2 = 1
    elif duree_max10 <= 6:
        cat_coussin_2 = 2
    else:
        cat_coussin_2 = 3

    # Lookup cushion parameters
    key = (code_cat_produit, cat_coussin_1, cat_coussin_2)
    cushion_info = lookups['coussins'].get(key, {
        'BASE_PASSIF_REDRESSE': 0,
        'TX_PASSIF_REDRESSE': 0.0,
        'BASE_COUSSIN_CREDIT': 0,
        'TX_COUSSIN_CREDIT': 0.0,
        'BASE_COUSSIN_MARCHE': 0,
        'TX_COUSSIN_MARCHE': 0.0,
        'BASE_COUSSIN_DEPENSE': 0,
        'TX_COUSSIN_DEPENSE': 0.0,
        'BASE_COUSSIN_DECHEANCE': 0,
        'TX_COUSSIN_DECHEANCE': 0.0,
        'BASE_COUSSIN_MORTALITE': 0,
        'TX_COUSSIN_MORTALITE': 0.0,
        'BASE_COUSSIN_DEPOT': 0,
        'TX_COUSSIN_DEPOT': 0.0,
        'FACTEUR_AGE_80': 1.0,
        'FACTEUR_AGE_90': 1.0,
    })

    # For RGS with VM=0, set certain cushions to 0
    if code_cat_produit == 7 and state['MT_VM_PROJ'] == 0:
        cushion_info['TX_COUSSIN_CREDIT'] = 0
        cushion_info['TX_COUSSIN_MARCHE'] = 0
        cushion_info['TX_COUSSIN_DECHEANCE'] = 0
        cushion_info['TX_COUSSIN_DEPOT'] = 0

    # Determine age factor
    if row['AGE'] < 80:
        age_factor = 1.0
    elif row['AGE'] < 90:
        age_factor = cushion_info['FACTEUR_AGE_80']
    else:
        age_factor = cushion_info['FACTEUR_AGE_90']

    # Calculate base amount for cushions
    max_guarantee = max(state['MT_GAR_ECH_PROJ'],
                        state['MT_GAR_DECES_PROJ'] + state['MT_BONI_DECES_PROJ'],
                        state['MT_SRG_PROJ'])

    # Calculate each cushion
    results = {}
    for cushion_name in ['PASSIF_REDRESSE', 'COUSSIN_CREDIT', 'COUSSIN_MARCHE',
                         'COUSSIN_DEPENSE', 'COUSSIN_DECHEANCE', 'COUSSIN_MORTALITE', 'COUSSIN_DEPOT']:

        base_key = f'BASE_{cushion_name}'
        tx_key = f'TX_{cushion_name}'

        if cushion_info[base_key] == 0:
            base_amount = max_guarantee
        else:
            base_amount = state['MT_VM_PROJ']

        cushion_amount = cushion_info[tx_key] * base_amount * age_factor * state['TX_SURVIE']
        results[cushion_name] = cushion_amount
        results[f'VP_{cushion_name}'] = cushion_amount * state['TX_ACTUALISATION'] / freq

    return results

def update_survival_and_discount(state: Dict, qx: float, lapse: float) -> Dict:
    """Update cumulative survival probability."""
    state['TX_SURVIE'] = state['TX_SURVIE'] * (1 - qx) * (1 - lapse)
    return state


def accumulate_death_bonus(state: Dict, row: Dict, account: pd.Series, freq: int, AJUST_NOUV_AFFAIRES: float) -> Dict:
    """Accumulate death bonus before withdrawals."""
    pc_boni_deces = state['PC_BONI_DECES']
    max_boni_deces = account.get('MAX_BONI_DECES', 999)

    if pc_boni_deces > 0 and row['AGE'] < max_boni_deces:
        state['MT_BONI_DECES_PROJ'] = (state['MT_BONI_DECES_PROJ'] +
                                       state['MT_GAR_DECES_PROJ'] * pc_boni_deces / freq * AJUST_NOUV_AFFAIRES)
    else:
        state['MT_BONI_DECES_PROJ'] = 0

    return state


def apply_fees_and_update_vm(state: Dict, row: Dict, freq: int, AJUST_NOUV_AFFAIRES: float, tx_survie_deb: float) -> \
Tuple[Dict, float, float]:
    """
    NEW: Encapsulates the fee deduction waterfall to ensure correct order of operations.
    This function replicates SAS lines 454-472.

    1. Applies management fees (RFG) continuously.
    2. Calculates guarantee fee cash flow (Primes Garanties).
    3. Deducts guarantee fee amount from VM.
    4. Returns updated state and cash flows.
    """
    # Step 1: Apply management fees (PC_RFG) using continuous compounding.
    # SAS: MT_VM_AV_RETRAIT = MT_VM_AV_RETRAIT_FRAIS * EXP(-PC_RFG / &FREQ_EVAL. * AJUST_NOUV_AFFAIRES);
    mt_vm_av_retrait = state['MT_VM_AV_RETRAIT_FRAIS'] * np.exp(-state['PC_RFG'] / freq * AJUST_NOUV_AFFAIRES)

    # Step 2: Calculate the guarantee fee *amount* (Primes Garanties) before applying survival.
    # The base for the fee is the VM *after* management fees.
    i_frais_sur_srg = row.get('I_FRAIS_SUR_SRG', 0)

    if i_frais_sur_srg == 0:
        # Fee is based on the post-RFG market value
        base_fee_calc = mt_vm_av_retrait
    else:
        # Fee is based on the SRG
        base_fee_calc = state['MT_SRG_PROJ']

    guarantee_fee_amount = base_fee_calc * state['PC_FRAIS_GARANTIE'] / freq * AJUST_NOUV_AFFAIRES

    # The fee cannot exceed the available market value
    guarantee_fee_amount = min(guarantee_fee_amount, mt_vm_av_retrait)

    # Calculate the final cash flow with survival probability
    primes_garanties = guarantee_fee_amount * tx_survie_deb
    vp_primes_garanties = primes_garanties * state['TX_ACTUALISATION']

    # Step 3: Deduct the guarantee fee *amount* from the market value.
    # SAS: MT_VM_AV_RETRAIT = MAX(MT_VM_AV_RETRAIT - ..., 0);
    mt_vm_av_retrait_final = max(mt_vm_av_retrait - guarantee_fee_amount, 0)

    # Step 4: Update the state with the final VM value to be used for withdrawals.
    state['MT_VM_AV_RETRAIT'] = mt_vm_av_retrait_final

    return state, primes_garanties, vp_primes_garanties


def calculate_duree_max10(row: Dict, account: pd.Series) -> int:
    """
    Calculate duration from issue date (matches SAS formula exactly).

    SAS: duree_max10=min(10,int((annee_reelle+mois_eval/12)-(ANNEE_COTIS+MOIS_COTIS/12))+1);
    """
    annee_cotis = account.get('ANNEE_COTIS', account.get('ANNEE_EVALUATION_INI', row['ANNEE_REELLE']))
    mois_cotis = account.get('MOIS_COTIS', 1)

    current_date = row['ANNEE_REELLE'] + row['MOIS_EVAL'] / 12
    issue_date = annee_cotis + mois_cotis / 12

    duree = int(current_date - issue_date) + 1
    return min(duree, 10)


def process_single_row(row: Dict, state: Dict, lookups: Dict, prev_scn: int,
                       original_account: pd.Series) -> Tuple[Dict, Dict]:
    """
    Process a single projection row with all calculations.

    FIXED: Now calculates duree_max10 from dates at each step instead of using counter.
    """
    freq = CONFIG['FREQ_EVAL']
    AJUST_NOUV_AFFAIRES = 1.0

    if row['SCN_EVAL'] != prev_scn:
        state = initialize_state(original_account)

    if state['TX_SURVIE'] == 0 or (state['MT_VM_PROJ'] == 0 and row['I_PRODUIT_REGR'] == 0):
        return None, state

    # CRITICAL FIX: Calculate duration from dates, not from counter
    duree_max10 = calculate_duree_max10(row, original_account)

    tx_survie_deb = state['TX_SURVIE']

    # Step 1: Lookups (Mortality & Returns)
    qx = lookup_mortality(lookups, row, state)
    qx = 1 - (1 - qx) ** (1 / freq * AJUST_NOUV_AFFAIRES)
    returns = lookup_returns(lookups, row, state)

    # Step 2: Update Discount Factor & Apply Investment Returns
    state = update_discount_factor(state, returns['FORWARD_RATE'], AJUST_NOUV_AFFAIRES)
    state = apply_investment_returns(state, returns, AJUST_NOUV_AFFAIRES)

    # Step 3: Calculate Lapse Rates (passing duree_max10)
    lapse_tot, lapse_part, lapse = calculate_lapse_rates(state, lookups, row, original_account, freq,
                                                         AJUST_NOUV_AFFAIRES, duree_max10)

    # Step 4: Update Cumulative Survival Probability
    state = update_survival_and_discount(state, qx, lapse)

    # Step 5: Accumulate Death Bonus
    state = accumulate_death_bonus(state, row, original_account, freq, AJUST_NOUV_AFFAIRES)

    # Step 6: Apply all fees and update the Market Value before withdrawals
    state, primes_garanties, vp_primes_garanties = apply_fees_and_update_vm(
        state, row, freq, AJUST_NOUV_AFFAIRES, tx_survie_deb
    )

    # Step 7: Calculate MRV and Withdrawals
    state = calculate_mrv_amount(state, row, original_account, freq)
    retrait = calculate_withdrawal(state, lookups, row, original_account, freq)

    # Step 8: Calculate Deposits (passing duree_max10)
    _, depot_futur = process_deposits(state, lookups, row, original_account, freq, duree_max10)

    # Step 9: Process Benefits and update VM/Guarantees
    prest_mrv, vp_prest_mrv = calculate_mrv_benefit(state, retrait, state['MT_VM_AV_RETRAIT'], row, tx_survie_deb)
    state, mt_vm_ap_retrait, mt_srg_av_retrait = update_guarantees_for_withdrawal(state, retrait)

    state = process_deposits_and_update_vm(state, depot_futur, mt_vm_ap_retrait)
    mt_vm_ap_retrait_depot = state['MT_VM_PROJ']

    prest_deces, vp_prest_deces = calculate_death_benefit(state, qx, tx_survie_deb, mt_vm_ap_retrait_depot)
    state, prest_ech, vp_prest_ech = process_maturity_benefit(state, row, original_account, mt_vm_ap_retrait)

    # Step 10: Process Resets and Adjustments
    state = update_death_guarantee_adjustments(state, row, original_account, freq)
    state = process_srg_bcb_resets(state, row, original_account)
    state = process_death_guarantee_resets(state, row, original_account)
    state = process_facultative_maturity_reset(state, row, original_account)
    state = process_death_guarantee_age_change(state, row, original_account)
    state = rebalance_portfolio(state, original_account)

    # Step 11: Calculate remaining cash flows (passing duree_max10)
    acquisition = calculate_acquisition_costs(state, lookups, row, original_account,
                                              depot_futur, lapse, qx, mt_vm_ap_retrait,
                                              tx_survie_deb, freq, AJUST_NOUV_AFFAIRES, duree_max10)
    fees = calculate_fees(state, lookups, row, tx_survie_deb, freq, AJUST_NOUV_AFFAIRES,
                          acquisition['PC_COMMISSION_MAINTIEN'])
    cushions = calculate_escap_cushions(state, lookups, row, original_account, freq, duree_max10)

    # Step 12: Final tracking metrics
    valeur_marchande = state['MT_VM_PROJ'] * state['TX_SURVIE']
    vp_valeur_marchande = valeur_marchande * state['TX_ACTUALISATION'] / freq

    # Assemble final output row
    result_row = {
        'ID_COMPTE': row['ID_COMPTE'], 'SCN_EVAL': row['SCN_EVAL'], 'AN_EVAL': row['AN_EVAL'],
        'MOIS_EVAL': row['MOIS_EVAL'],
        'PRIMES_GARANTIES': primes_garanties, 'PREST_DECES': prest_deces, 'PREST_ECH': prest_ech,
        'PREST_MRV': prest_mrv,
        'FRAIS_ACQUIS': acquisition['FRAIS_ACQUIS'], 'COMM_VENTE': acquisition['COMM_VENTE'],
        'PRIMES_VARIABLES': fees['PRIMES_VARIABLES'], 'FRAIS_FIXES': fees['FRAIS_FIXES'], 'HON_GEST': fees['HON_GEST'],
        'COMM_MAINTIEN': fees['COMM_MAINTIEN'],
        'VALEUR_MARCHANDE': valeur_marchande,
        **{k: v for k, v in cushions.items() if not k.startswith('VP_')},
        'VP_FRAIS_ACQUIS': acquisition['VP_FRAIS_ACQUIS'],
        'VP_COMM_VENTE': acquisition['VP_COMM_VENTE'],
        'VP_PRIMES_GARANTIES': vp_primes_garanties,
        'VP_PRIMES_VARIABLES': fees['VP_PRIMES_VARIABLES'],
        'VP_FRAIS_FIXES': fees['VP_FRAIS_FIXES'],
        'VP_HON_GEST': fees['VP_HON_GEST'],
        'VP_COMM_MAINTIEN': fees['VP_COMM_MAINTIEN'],
        'VP_PREST_ECH': vp_prest_ech, 'VP_PREST_MRV': vp_prest_mrv, 'VP_PREST_DECES': vp_prest_deces,
        'VP_VALEUR_MARCHANDE': vp_valeur_marchande,
        **{k: v for k, v in cushions.items() if k.startswith('VP_')},
    }

    return result_row, state

# =============================================================================
# MAIN PROJECTION FUNCTION (COMBINES ALL LEVELS)
# =============================================================================

def project_account_wrapper(account_id: int, population: pd.DataFrame,
                            lookups: Dict, config: Dict, output_path: Path) -> pd.DataFrame:
    """
    Wrapper for parallel processing a single account.

    ENHANCED: Adds logic to save a detailed trace for a specific account/scenario,
    mimicking the SAS `TEST` dataset for easy debugging and comparison.
    """
    account = population[population['ID_COMPTE'] == account_id].iloc[0]

    # --- DEBUGGING FEATURE ---
    # Check if this is the specific account and scenario we want to debug
    create_debug_file = (account_id == config['NO_COMPTE_SORTIE'])
    debug_data = []

    # LEVEL 2: Create expanded timeline
    timeline = create_expanded_timeline(account, config)
    if len(timeline) == 0:
        return pd.DataFrame()

    # LEVEL 3: Process each row with state retention
    results = []
    state = initialize_state(account)
    prev_scn = 1

    for idx, row in timeline.iterrows():
        row_dict = row.to_dict()

        # Process the row
        result_row, new_state = process_single_row(row_dict, state, lookups, prev_scn, account)

        # Update the state for the next iteration
        state = new_state

        # Only append if result is not None (policy still active)
        if result_row is not None:
            results.append(result_row)

            # If this is the debug scenario, save all intermediate variables
            if create_debug_file and row_dict['SCN_EVAL'] == config['NO_SCN_SORTIE']:
                # Combine the original row, the calculated results, and the full state
                full_debug_row = {**row_dict, **result_row, **state}
                debug_data.append(full_debug_row)

        prev_scn = row_dict['SCN_EVAL']

    # If a debug file was created, save it to CSV
    if create_debug_file and debug_data:
        debug_df = pd.DataFrame(debug_data)
        # Ensure output directory exists before saving
        output_path.mkdir(parents=True, exist_ok=True)
        debug_filename = output_path.joinpath("TEST_PY.csv")
        debug_df.to_csv(debug_filename, index=False, sep=';')
        print(f"  ✓ Saved debug file for account {account_id} to {debug_filename}")

    # Return the main results
    if not results:
        return pd.DataFrame()

    final_df = pd.DataFrame(results)
    print(f"  Completed account {account_id}: {len(final_df)} projection rows generated.")
    return final_df


# =============================================================================
# AGGREGATION FUNCTIONS
# =============================================================================

def aggregate_by_scenario(df: pd.DataFrame) -> pd.DataFrame:
    """Average results across scenarios for each account/time period."""
    group_cols = ['ID_COMPTE', 'AN_EVAL', 'MOIS_EVAL']

    value_cols = [col for col in df.columns if col not in group_cols + ['SCN_EVAL']]

    result = df.groupby(group_cols, as_index=False)[value_cols].mean()

    return result


def aggregate_flux_projetes(df: pd.DataFrame) -> pd.DataFrame:
    """Create FLUX_PROJETES: sum by time period across all accounts."""
    group_cols = ['AN_EVAL', 'MOIS_EVAL']

    # Exclude VP columns and ID columns
    value_cols = [col for col in df.columns if not col.startswith('VP_') and
                  col not in group_cols + ['ID_COMPTE', 'SCN_EVAL']]

    result = df.groupby(group_cols, as_index=False)[value_cols].sum()

    return result


def aggregate_vp_flux_compte(df: pd.DataFrame) -> pd.DataFrame:
    """Create VP_FLUX_COMPTE: present values by account with specific column order."""

    # Define the exact column order as requested
    requested_columns = [
        'ID_COMPTE',
        'VP_FRAIS_ACQUIS',
        'VP_COMM_VENTE',
        'VP_PRIMES_GARANTIES',
        'VP_PRIMES_VARIABLES',
        'VP_FRAIS_FIXES',
        'VP_HON_GEST',
        'VP_COMM_MAINTIEN',
        'VP_PREST_ECH',
        'VP_PREST_MRV',
        'VP_PREST_DECES',
        'VP_PASSIF_REDRESSE',
        'VP_COUSSIN_CREDIT',
        'VP_COUSSIN_MARCHE',
        'VP_COUSSIN_DEPENSE',
        'VP_COUSSIN_DECHEANCE',
        'VP_COUSSIN_MORTALITE',
        'VP_COUSSIN_DEPOT',
        'VP_VALEUR_MARCHANDE'
    ]

    # Get VP columns that exist in the dataframe (exclude ID_COMPTE)
    vp_cols = [col for col in requested_columns[1:] if col in df.columns]

    # Group by ID_COMPTE and sum all VP columns
    result = df.groupby('ID_COMPTE', as_index=False)[vp_cols].sum()

    # Reorder to match requested column order
    final_columns = ['ID_COMPTE'] + [col for col in requested_columns[1:] if col in result.columns]
    result = result[final_columns]

    return result


def aggregate_vp_flux_total(df: pd.DataFrame) -> pd.DataFrame:
    """Create VP_FLUX_TOTAL: total present value across all accounts."""

    vp_cols = [
        'VP_FRAIS_ACQUIS', 'VP_COMM_VENTE', 'VP_PRIMES_GARANTIES',
        'VP_PRIMES_VARIABLES', 'VP_FRAIS_FIXES', 'VP_HON_GEST', 'VP_COMM_MAINTIEN',
        'VP_PREST_ECH', 'VP_PREST_MRV', 'VP_PREST_DECES',
        'VP_PASSIF_REDRESSE', 'VP_COUSSIN_CREDIT', 'VP_COUSSIN_MARCHE',
        'VP_COUSSIN_DEPENSE', 'VP_COUSSIN_DECHEANCE', 'VP_COUSSIN_MORTALITE',
        'VP_COUSSIN_DEPOT'
    ]

    # Filter for columns that actually exist in the dataframe
    existing_vp_cols = [col for col in vp_cols if col in df.columns]

    # Calculate total PV of flows per account
    df['VP_FLUX_TOT'] = df[existing_vp_cols].sum(axis=1)

    result = pd.DataFrame({
        'CATEGORIE': ['TOTAL'],
        'VP_FLUX_TOT': [df['VP_FLUX_TOT'].sum()]
    })

    return result

# =============================================================================
# MAIN EXECUTION FUNCTION
# =============================================================================

def run_projection(data_path: Path, output_path: Path, nb_an_projection: int, nb_scenarios: int,
                   use_parallel: bool = False, max_accounts: int = None):
    """
    Main function to run the complete actuarial projection.

    Args:
        data_path: Path to input CSV files.
        output_path: Path for output files.
        nb_an_projection: The number of years to project.
        nb_scenarios: The number of economic scenarios to run.
        use_parallel: Whether to use multiprocessing.
        max_accounts: Maximum number of accounts to process (for testing). None processes all.
    """
    start_time = datetime.now()
    print(f"Starting projection at {start_time}")
    print("=" * 60)

    # --- NEW: Update global config with runtime arguments ---
    CONFIG['NB_AN_PROJECTION'] = nb_an_projection
    CONFIG['NB_SC'] = nb_scenarios
    print(f"Configuration: {nb_an_projection} years, {nb_scenarios} scenarios")
    # --------------------------------------------------------

    # Load data
    data = load_all_data(data_path)

    # Create lookups
    lookups = create_all_lookups(data)

    # Get account list
    account_ids = data['population']['ID_COMPTE'].unique()
    if max_accounts:
        account_ids = account_ids[:max_accounts]

    print(f"\nProcessing {len(account_ids)} accounts...")
    print("=" * 60)

    # Process accounts
    if use_parallel and CONFIG['nb_thread_tot'] > 1:
        print(f"Using {CONFIG['nb_thread_tot']} parallel processes")
        with mp.Pool(processes=CONFIG['nb_thread_tot']) as pool:
            func = partial(project_account_wrapper,
                           population=data['population'],
                           lookups=lookups,
                           config=CONFIG,
                           output_path=output_path)
            results = pool.map(func, account_ids)
    else:
        print("Using sequential processing")
        results = []
        for i, account_id in enumerate(account_ids, 1):
            print(f"[{i}/{len(account_ids)}] Processing account {account_id}...")
            result = project_account_wrapper(account_id, data['population'], lookups, CONFIG, output_path)
            results.append(result)

    # Combine all results
    print("\n" + "=" * 60)
    print("Combining results...")
    all_results = pd.concat([r for r in results if not r.empty], ignore_index=True)

    if all_results.empty:
        print("No results generated. Exiting.")
        return None

    print(f"Total projection rows: {len(all_results):,}")

    # Aggregate results
    print("Aggregating results...")

    # 1. Average across scenarios
    print("  - Averaging across scenarios...")
    calculs_sommaire = aggregate_by_scenario(all_results)

    # 2. Flux projetes (by time period)
    print("  - Creating flux projetes...")
    flux_projetes = aggregate_flux_projetes(calculs_sommaire)

    # 3. VP by account
    print("  - Creating VP by account...")
    vp_flux_compte = aggregate_vp_flux_compte(calculs_sommaire)

    # 4. Total VP
    print("  - Creating total VP...")
    vp_flux_total = aggregate_vp_flux_total(vp_flux_compte)

    # Save outputs
    print("\nSaving outputs...")
    output_path.mkdir(parents=True, exist_ok=True)

    flux_projetes.to_csv(output_path.joinpath("FLUX_PROJETES_PY.csv"), index=False, sep=';')
    vp_flux_compte.to_csv(output_path.joinpath("VP_FLUX_COMPTE_PY.csv"), index=False, sep=';')
    vp_flux_total.to_csv(output_path.joinpath("VP_FLUX_TOTAL_PY.csv"), index=False, sep=';')

    print(f"  ✓ Saved {output_path}/FLUX_PROJETES_PY.csv")
    print(f"  ✓ Saved {output_path}/VP_FLUX_COMPTE_PY.csv")
    print(f"  ✓ Saved {output_path}/VP_FLUX_TOTAL_PY.csv")

    # Print summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print("\n" + "=" * 60)
    print("PROJECTION COMPLETE")
    print("=" * 60)
    print(f"Processing time: {duration:.2f} seconds ({duration / 60:.2f} minutes)")
    print(f"Accounts processed: {len(account_ids)}")
    print(f"Total rows generated: {len(all_results):,}")
    print(f"Rows after averaging: {len(calculs_sommaire):,}")
    print(f"Total PV of flows: ${vp_flux_total['VP_FLUX_TOT'].iloc[0]:,.2f}")
    print("=" * 60)

    return {
        'flux_projetes': flux_projetes,
        'vp_flux_compte': vp_flux_compte,
        'vp_flux_total': vp_flux_total,
        'all_results': all_results,
    }


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Set paths
    DATA_PATH = HERE.joinpath("algo2/data_in")
    OUTPUT_PATH = HERE.joinpath("algo2/data_out")

    # Run projection
    results = run_projection(
        data_path=DATA_PATH,
        output_path=OUTPUT_PATH,
        nb_an_projection=100,  # Set the number of years to project
        nb_scenarios=100,      # Set the number of scenarios to run
        use_parallel=False,   # Set to True for parallel processing
        max_accounts=3        # Process only first 3 accounts for testing, None for all
    )

    if results:
        print("\n" + "=" * 60)
        print("SAMPLE RESULTS")
        print("=" * 60)

        print("\nVP_FLUX_TOTAL:")
        print(results['vp_flux_total'])

        print("\nVP_FLUX_COMPTE (first 5 accounts):")
        print(results['vp_flux_compte'].head())

        print("\nFLUX_PROJETES (first 10 periods):")
        print(results['flux_projetes'].head(10))
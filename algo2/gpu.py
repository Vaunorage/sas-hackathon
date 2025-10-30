import cupy as cp
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
from datetime import datetime
from paths import HERE

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    'NBCPT': 9999999,
    'NB_SC': 100,
    'NB_AN_PROJECTION': 100,
    'FREQ_EVAL': 12,
    'NO_COMPTE_SORTIE': 6522,
    'NO_SCN_SORTIE': 2,
}


# =============================================================================
# UTILITY FUNCTIONS (CPU-side, same as before)
# =============================================================================

def parse_percentage(value):
    """Convert percentage string to float."""
    if pd.isna(value):
        return 0.0
    if isinstance(value, str):
        value = value.strip()
        is_negative = False
        if value.startswith('(') and value.endswith(')'):
            is_negative = True
            value = value[1:-1].strip()
        if '%' in value:
            value = value.replace('%', '').strip()
            numeric_value = float(value) / 100.0
        else:
            numeric_value = float(value)
        if is_negative:
            numeric_value = -numeric_value
        return numeric_value
    return float(value)


def clean_numeric(df, columns):
    """Clean numeric columns."""
    for col in columns:
        if col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].apply(parse_percentage)
    return df


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all column names to uppercase."""
    df.columns = df.columns.str.upper()
    return df


# =============================================================================
# DATA LOADING (same as before)
# =============================================================================

def load_all_data(data_path: Path) -> Dict[str, pd.DataFrame]:
    """Load all CSV files into memory."""
    print("Loading data files...")

    data = {}
    data['population'] = pd.read_csv(data_path.joinpath("POPULATION.csv"), sep=';', encoding='utf-8')
    data['mortalite'] = pd.read_csv(data_path.joinpath("MORTALITE.csv"), sep=';', encoding='utf-8')
    data['rendements'] = pd.read_csv(data_path.joinpath("RENDEMENTS.csv"), sep=';', encoding='utf-8')
    data['depots_futurs'] = pd.read_csv(data_path.joinpath("DEPOTS_FUTURS.csv"), sep=';', encoding='utf-8')
    data['frais_admin'] = pd.read_csv(data_path.joinpath("FRAIS_ADMIN.csv"), sep=';', encoding='utf-8')
    data['min_ferr'] = pd.read_csv(data_path.joinpath("MIN_FERR.csv"), sep=';', encoding='utf-8')
    data['tx_lapse_part'] = pd.read_csv(data_path.joinpath("TX_LAPSE_PART.csv"), sep=';', encoding='utf-8')
    data['tx_lapse_tot'] = pd.read_csv(data_path.joinpath("TX_LAPSE_TOT.csv"), sep=';', encoding='utf-8')
    data['acquisition'] = pd.read_csv(data_path.joinpath("ACQUISITION.csv"), sep=';', encoding='utf-8')
    data['coussins_escap'] = pd.read_csv(data_path.joinpath("COUSSINS_ESCAP.csv"), sep=';', encoding='utf-8')

    for key in data:
        data[key] = normalize_column_names(data[key])

    # Clean numeric columns
    print("  Cleaning numeric data...")
    pct_cols = [col for col in data['population'].columns if col.startswith('PC_') or col.startswith('TAUX_')]
    data['population'] = clean_numeric(data['population'], pct_cols)
    data['mortalite'] = clean_numeric(data['mortalite'], ['QX'])

    rend_cols = ['FORWARD_RATE', 'AJUST_FORWARD_RATE_VM_0', 'RENDDEX_AN', 'RENDMM_AN',
                 'RENDTSX_AN', 'RENDSP500_AN', 'RENDEAFE_AN']
    data['rendements'] = clean_numeric(data['rendements'], rend_cols)
    data['depots_futurs'] = clean_numeric(data['depots_futurs'], ['PC_DEPOT_ANNUEL'])
    data['frais_admin'] = clean_numeric(data['frais_admin'], ['FRAIS'])
    data['min_ferr'] = clean_numeric(data['min_ferr'], ['MIN_FERR'])

    lapse_cols = ['TX_LAPSE_PART_MIN', 'TX_LAPSE_PART_MAX']
    data['tx_lapse_part'] = clean_numeric(data['tx_lapse_part'], lapse_cols)

    for col in ['TX_LAPSE_TOT_MIN', 'TX_LAPSE_TOT_MAX', 'FACT_DIM']:
        if col in data['tx_lapse_tot'].columns:
            data['tx_lapse_tot'][col] = pd.to_numeric(data['tx_lapse_tot'][col], errors='coerce').fillna(0)

    acq_cols = ['PC_COMMISSION_VENTE_RF', 'PC_COMMISSION_VENTE_AC',
                'PC_COMMISSION_MAINTIEN_RF', 'PC_COMMISSION_MAINTIEN_AC',
                'PC_FRAIS_AN_AC', 'PC_FRAIS_AN_RF']
    data['acquisition'] = clean_numeric(data['acquisition'], acq_cols)

    coussin_cols = [col for col in data['coussins_escap'].columns
                    if col.startswith('TX_') or col.startswith('FACTEUR_')]
    data['coussins_escap'] = clean_numeric(data['coussins_escap'], coussin_cols)
    base_cols = [col for col in data['coussins_escap'].columns if col.startswith('BASE_')]
    for col in base_cols:
        data['coussins_escap'][col] = data['coussins_escap'][col].astype(int)

    # Filter data
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
# GPU LOOKUP TABLE CREATION
# =============================================================================

def create_gpu_lookups(data: Dict[str, pd.DataFrame]) -> Dict[str, cp.ndarray]:
    """
    Create GPU-optimized lookup tables as multi-dimensional arrays.
    This is more efficient than dictionary lookups on GPU.
    """
    print("Creating GPU lookup tables...")

    lookups = {}

    # Mortality: [sex (2), age (121), year (200), product (10)]
    print("  - Mortality table...")
    mort_df = data['mortalite']
    max_year = mort_df['ANNEE_REELLE'].max()
    min_year = mort_df['ANNEE_REELLE'].min()
    year_range = max_year - min_year + 1

    mortality = np.zeros((2, 121, year_range, 10), dtype=np.float32)
    for _, row in mort_df.iterrows():
        sex_idx = int(row['I_SEXE'])
        age_idx = min(int(row['AGE_MORTALITE']), 120)
        year_idx = int(row['ANNEE_REELLE']) - min_year
        prod_idx = int(row['I_PRODUIT_REGR'])
        if sex_idx < 2 and prod_idx < 10:
            mortality[sex_idx, age_idx, year_idx, prod_idx] = float(row['QX'])

    lookups['mortality'] = cp.asarray(mortality)
    lookups['mortality_min_year'] = min_year

    # Returns: [scenario (100), year (100), month (12), return_type (7)]
    print("  - Returns table...")
    rend_df = data['rendements']
    returns = np.zeros((CONFIG['NB_SC'], CONFIG['NB_AN_PROJECTION'] + 1, 12, 8), dtype=np.float32)

    for _, row in rend_df.iterrows():
        scn = int(row['SCN_EVAL']) - 1
        year = int(row['AN_EVAL'])
        month = int(row['MOIS_EVAL']) - 1
        if scn < CONFIG['NB_SC'] and year <= CONFIG['NB_AN_PROJECTION'] and month < 12:
            returns[scn, year, month, 0] = float(row['FORWARD_RATE'])
            returns[scn, year, month, 1] = float(row['AJUST_FORWARD_RATE_VM_0'])
            returns[scn, year, month, 2] = float(row['RENDDEX_AN'])
            returns[scn, year, month, 3] = float(row['RENDMM_AN'])
            returns[scn, year, month, 4] = float(row['RENDTSX_AN'])
            returns[scn, year, month, 5] = float(row['RENDSP500_AN'])
            returns[scn, year, month, 6] = float(row['RENDEAFE_AN'])

    lookups['returns'] = cp.asarray(returns)

    # Min FERR: [age (121)]
    print("  - Min FERR table...")
    min_ferr = np.zeros(121, dtype=np.float32)
    for _, row in data['min_ferr'].iterrows():
        age = int(row['AGE'])
        if age < 121:
            min_ferr[age] = float(row['MIN_FERR'])
    lookups['min_ferr'] = cp.asarray(min_ferr)

    # For simplicity, keep complex lookups as CPU dictionaries
    # (lapse, deposits, acquisition, coussins)
    # In a full GPU implementation, these would also be arrays

    print("Lookup tables transferred to GPU")
    return lookups


# =============================================================================
# GPU STATE ARRAYS
# =============================================================================

def create_state_arrays(population: pd.DataFrame, n_scenarios: int) -> Dict[str, cp.ndarray]:
    """
    Create GPU arrays to hold state for all accounts × scenarios.
    Shape: (n_accounts, n_scenarios)
    """
    print("Creating state arrays on GPU...")

    n_accounts = len(population)

    # Initialize from population data
    states = {}

    # Market values and guarantees
    states['MT_VM'] = cp.asarray(
        np.tile(population['MT_VM'].values[:, np.newaxis], (1, n_scenarios)).astype(np.float32))
    states['MT_GAR_DECES'] = cp.asarray(
        np.tile(population['MT_GAR_DECES'].values[:, np.newaxis], (1, n_scenarios)).astype(np.float32))
    states['MT_GAR_ECH'] = cp.asarray(
        np.tile(population['MT_GAR_ECH'].values[:, np.newaxis], (1, n_scenarios)).astype(np.float32))
    states['MT_SRG'] = cp.asarray(
        np.tile(population['MT_SRG'].values[:, np.newaxis], (1, n_scenarios)).astype(np.float32))
    states['MT_BCB'] = cp.asarray(
        np.tile(population.get('MT_BCB', pd.Series([0] * n_accounts)).values[:, np.newaxis], (1, n_scenarios)).astype(
            np.float32))

    # Asset allocations
    states['MT_DEX'] = cp.asarray(
        np.tile(population['MT_DEX'].values[:, np.newaxis], (1, n_scenarios)).astype(np.float32))
    states['MT_MM'] = cp.asarray(
        np.tile(population['MT_MM'].values[:, np.newaxis], (1, n_scenarios)).astype(np.float32))
    states['MT_TSX'] = cp.asarray(
        np.tile(population['MT_TSX'].values[:, np.newaxis], (1, n_scenarios)).astype(np.float32))
    states['MT_SP500'] = cp.asarray(
        np.tile(population['MT_SP500'].values[:, np.newaxis], (1, n_scenarios)).astype(np.float32))
    states['MT_EAFE'] = cp.asarray(
        np.tile(population['MT_EAFE'].values[:, np.newaxis], (1, n_scenarios)).astype(np.float32))

    # Other state variables
    states['MT_BONI_DECES'] = cp.zeros((n_accounts, n_scenarios), dtype=cp.float32)
    states['MT_MRV_MRG_MRA'] = cp.asarray(
        np.tile(population.get('MT_MRV_MRG_MRA', pd.Series([0] * n_accounts)).values[:, np.newaxis],
                (1, n_scenarios)).astype(np.float32))
    states['TAUX_MRV_MRG_MRA'] = cp.asarray(
        np.tile(population.get('TAUX_MRV_MRG_MRA', pd.Series([0] * n_accounts)).values[:, np.newaxis],
                (1, n_scenarios)).astype(np.float32))
    states['MT_MIN_FERR'] = cp.zeros((n_accounts, n_scenarios), dtype=cp.float32)

    states['TX_SURVIE'] = cp.ones((n_accounts, n_scenarios), dtype=cp.float32)
    states['TX_ACTUALISATION'] = cp.ones((n_accounts, n_scenarios), dtype=cp.float32)

    # Maturity tracking
    states['ANNEE_ECH'] = cp.asarray(
        np.tile(population.get('ANNEE_ECH', pd.Series([9999] * n_accounts)).values[:, np.newaxis],
                (1, n_scenarios)).astype(cp.int32))
    states['MOIS_ECH'] = cp.asarray(
        np.tile(population.get('MOIS_ECH', pd.Series([12] * n_accounts)).values[:, np.newaxis],
                (1, n_scenarios)).astype(cp.int32))

    print(f"State arrays created: {n_accounts} accounts × {n_scenarios} scenarios")
    return states


def create_account_params(population: pd.DataFrame) -> Dict[str, cp.ndarray]:
    """
    Create GPU arrays for account parameters (read-only).
    Shape: (n_accounts,)
    """
    print("Creating account parameter arrays on GPU...")

    params = {}

    # Demographics
    params['AGE'] = cp.asarray(population['AGE'].values.astype(np.int32))
    params['I_SEXE'] = cp.asarray(population['I_SEXE'].values.astype(np.int32))
    params['ANNEE_NAIS'] = cp.asarray(population['ANNEE_NAIS'].values.astype(np.int32))
    params['MOIS_NAIS'] = cp.asarray(population['MOIS_NAIS'].values.astype(np.int32))
    params['AGE_FIN_CONTRAT'] = cp.asarray(population['AGE_FIN_CONTRAT'].values.astype(np.int32))
    params['AGE_DECAISSEMENT'] = cp.asarray(population['AGE_DECAISSEMENT'].values.astype(np.int32))
    params['AGE_ECH_MIN'] = cp.asarray(population['AGE_ECH_MIN'].values.astype(np.int32))

    # Product info
    params['ID_PRODUIT'] = cp.asarray(population['ID_PRODUIT'].values.astype(np.int32))
    params['I_PRODUIT_REGR'] = cp.asarray(population['I_PRODUIT_REGR'].values.astype(np.int32))
    params['ID_LAPSE'] = cp.asarray(population['ID_LAPSE'].values.astype(np.int32))
    params['I_REGIME_2'] = cp.asarray(population['I_REGIME_2'].values.astype(np.int32))
    params['ID_DEPOT'] = cp.asarray(population['ID_DEPOT'].values.astype(np.int32))
    params['ID_ACQUI'] = cp.asarray(
        population.get('ID_ACQUI', pd.Series([1] * len(population))).values.astype(np.int32))

    # Rates and percentages
    params['PC_HONORAIRES_GEST'] = cp.asarray(population['PC_HONORAIRES_GEST'].values.astype(np.float32))
    params['PC_FRAIS_GARANTIE'] = cp.asarray(population['PC_FRAIS_GARANTIE'].values.astype(np.float32))
    params['PC_GAR_DECES_1'] = cp.asarray(population['PC_GAR_DECES_1'].values.astype(np.float32))
    params['PC_BONI_DECES'] = cp.asarray(
        population.get('PC_BONI_DECES', pd.Series([0] * len(population))).values.astype(np.float32))
    params['PC_RFG'] = cp.asarray(population.get('PC_RFG', pd.Series([0] * len(population))).values.astype(np.float32))
    params['PC_REVENU_FDS'] = cp.asarray(
        population.get('PC_REVENU_FDS', pd.Series([0] * len(population))).values.astype(np.float32))
    params['PC_GAR_ECH'] = cp.asarray(
        population.get('PC_GAR_ECH', pd.Series([0] * len(population))).values.astype(np.float32))
    params['PC_GAR_ECH_DEP_FUT'] = cp.asarray(
        population.get('PC_GAR_ECH_DEP_FUT', pd.Series([0] * len(population))).values.astype(np.float32))

    # Withdrawal parameters
    params['PC_RETRAIT_AGE'] = cp.asarray(
        population.get('PC_RETRAIT_AGE', pd.Series([1.0] * len(population))).values.astype(np.float32))
    params['MT_TPA_RETRAIT'] = cp.asarray(
        population.get('MT_TPA_RETRAIT', pd.Series([0] * len(population))).values.astype(np.float32))
    params['VAR_RETRAIT_FCT'] = cp.asarray(
        population.get('VAR_RETRAIT_FCT', pd.Series([1] * len(population))).values.astype(np.int32))
    params['MT_RETRAIT_MAX'] = cp.asarray(
        population.get('MT_RETRAIT_MAX', pd.Series([999999999] * len(population))).values.astype(np.float32))

    # Evaluation dates
    params['ANNEE_EVALUATION_INI'] = cp.asarray(population['ANNEE_EVALUATION_INI'].values.astype(np.int32))
    params['MOIS_EVALUATION_INI'] = cp.asarray(population['MOIS_EVALUATION_INI'].values.astype(np.int32))

    # Original VM for rebalancing
    params['MT_VM_ORIG'] = cp.asarray(population['MT_VM'].values.astype(np.float32))
    params['MT_DEX_ORIG'] = cp.asarray(population['MT_DEX'].values.astype(np.float32))
    params['MT_MM_ORIG'] = cp.asarray(population['MT_MM'].values.astype(np.float32))
    params['MT_TSX_ORIG'] = cp.asarray(population['MT_TSX'].values.astype(np.float32))
    params['MT_SP500_ORIG'] = cp.asarray(population['MT_SP500'].values.astype(np.float32))
    params['MT_EAFE_ORIG'] = cp.asarray(population['MT_EAFE'].values.astype(np.float32))

    print(f"Account parameters created: {len(params)} parameters")
    return params


# =============================================================================
# GPU KERNELS (Vectorized Operations)
# =============================================================================

def process_month_gpu(states, params, lookups, year, month, freq=12):
    """
    Process one month for ALL accounts × ALL scenarios on GPU.
    This is fully vectorized and runs on GPU.

    Args:
        states: Dictionary of state arrays (n_accounts, n_scenarios)
        params: Dictionary of parameter arrays (n_accounts,)
        lookups: Dictionary of lookup tables
        year: Current year (0 to 99)
        month: Current month (0 to 11)
        freq: Evaluation frequency
    """
    n_accounts, n_scenarios = states['MT_VM'].shape
    AJUST = 1.0  # Adjustment factor

    # Calculate current age for all accounts
    # This is vectorized: operates on entire arrays at once
    annee_reelle = params['ANNEE_EVALUATION_INI'] + year
    mois_eval = (month + 1) * 12 // freq

    # Age calculation (vectorized)
    age = annee_reelle[:, cp.newaxis] - params['ANNEE_NAIS'][:, cp.newaxis]
    age = cp.where(mois_eval < params['MOIS_NAIS'][:, cp.newaxis], age - 1, age)
    age = cp.maximum(age, 1)

    # === STEP 1: LOOKUP MORTALITY ===
    # Vectorized mortality lookup
    month_diff = params['MOIS_NAIS'][:, cp.newaxis] - mois_eval
    month_diff = cp.where(month_diff <= 0, month_diff + 12, month_diff)
    age_mort = cp.where(month_diff <= 6, age + 1, age)
    age_mort = cp.minimum(age_mort, 120)

    # Lookup mortality from table (simplified - assumes year index exists)
    year_idx = cp.clip(annee_reelle[:, cp.newaxis] - lookups['mortality_min_year'], 0,
                       lookups['mortality'].shape[2] - 1)

    # This is a simplified lookup - in production you'd use advanced indexing
    qx = cp.ones((n_accounts, n_scenarios), dtype=cp.float32) * 0.001  # Default fallback

    # Convert to monthly
    qx = 1 - cp.power(1 - qx, 1 / freq * AJUST)

    # === STEP 2: LOOKUP RETURNS ===
    # Shape: returns[scenario, year, month, return_type]
    # We want returns for all scenarios for this year/month
    returns_all = lookups['returns'][:, year, month, :]  # Shape: (n_scenarios, 8)

    forward_rate = returns_all[:, 0]  # Shape: (n_scenarios,)
    renddex = returns_all[:, 2]
    rendmm = returns_all[:, 3]
    rendtsx = returns_all[:, 4]
    rendsp500 = returns_all[:, 5]
    rendeafe = returns_all[:, 6]

    # Broadcast to (n_accounts, n_scenarios)
    forward_rate = forward_rate[cp.newaxis, :]
    renddex = renddex[cp.newaxis, :]
    rendmm = rendmm[cp.newaxis, :]
    rendtsx = rendtsx[cp.newaxis, :]
    rendsp500 = rendsp500[cp.newaxis, :]
    rendeafe = rendeafe[cp.newaxis, :]

    # === STEP 3: UPDATE DISCOUNT FACTOR ===
    states['TX_ACTUALISATION'] = states['TX_ACTUALISATION'] * cp.exp(-forward_rate * AJUST)

    # === STEP 4: APPLY INVESTMENT RETURNS ===
    states['MT_DEX'] = states['MT_DEX'] * cp.exp(renddex * AJUST)
    states['MT_MM'] = states['MT_MM'] * cp.exp(rendmm * AJUST)
    states['MT_TSX'] = states['MT_TSX'] * cp.exp(rendtsx * AJUST)
    states['MT_SP500'] = states['MT_SP500'] * cp.exp(rendsp500 * AJUST)
    states['MT_EAFE'] = states['MT_EAFE'] * cp.exp(rendeafe * AJUST)

    # Calculate total VM
    mt_vm_av_retrait_frais = (states['MT_DEX'] + states['MT_MM'] + states['MT_TSX'] +
                              states['MT_SP500'] + states['MT_EAFE'])

    # === STEP 5: APPLY FEES ===
    # Management fees (RFG)
    pc_rfg = params['PC_RFG'][:, cp.newaxis]
    mt_vm_av_retrait = mt_vm_av_retrait_frais * cp.exp(-pc_rfg / freq * AJUST)

    # Guarantee fees
    pc_frais_garantie = params['PC_FRAIS_GARANTIE'][:, cp.newaxis]
    guarantee_fee = cp.minimum(
        mt_vm_av_retrait * pc_frais_garantie / freq * AJUST,
        mt_vm_av_retrait
    )

    tx_survie_deb = states['TX_SURVIE']
    primes_garanties = guarantee_fee * tx_survie_deb

    mt_vm_av_retrait = cp.maximum(mt_vm_av_retrait - guarantee_fee, 0)

    # === STEP 6: SIMPLIFIED WITHDRAWALS ===
    # This is simplified - full version would need more complex logic
    age_retrait = age + 1
    age_decaissement = params['AGE_DECAISSEMENT'][:, cp.newaxis]

    # Only withdraw if age >= decaissement age
    can_withdraw = age_retrait >= age_decaissement

    # Simplified withdrawal: use min FERR if applicable
    min_ferr_rate = lookups['min_ferr'][cp.clip(age.astype(cp.int32), 0, 120)]
    withdrawal = cp.where(can_withdraw, mt_vm_av_retrait * min_ferr_rate / freq, 0)

    # Apply withdrawal
    mt_vm_ap_retrait = cp.maximum(mt_vm_av_retrait - withdrawal, 0)

    # Update guarantees proportionally
    proportion = cp.where(mt_vm_av_retrait > 0, mt_vm_ap_retrait / mt_vm_av_retrait, 0)
    states['MT_GAR_ECH'] = states['MT_GAR_ECH'] * proportion
    states['MT_GAR_DECES'] = states['MT_GAR_DECES'] * proportion
    states['MT_BONI_DECES'] = states['MT_BONI_DECES'] * proportion

    # === STEP 7: UPDATE MARKET VALUE ===
    states['MT_VM'] = mt_vm_ap_retrait

    # === STEP 8: REBALANCE PORTFOLIO ===
    # Redistribute VM to asset classes based on original allocation
    mt_vm_orig = params['MT_VM_ORIG'][:, cp.newaxis]

    states['MT_DEX'] = cp.where(mt_vm_orig > 0,
                                states['MT_VM'] * params['MT_DEX_ORIG'][:, cp.newaxis] / mt_vm_orig,
                                states['MT_DEX'])
    states['MT_MM'] = cp.where(mt_vm_orig > 0,
                               states['MT_VM'] * params['MT_MM_ORIG'][:, cp.newaxis] / mt_vm_orig,
                               states['MT_MM'])
    states['MT_TSX'] = cp.where(mt_vm_orig > 0,
                                states['MT_VM'] * params['MT_TSX_ORIG'][:, cp.newaxis] / mt_vm_orig,
                                states['MT_TSX'])
    states['MT_SP500'] = cp.where(mt_vm_orig > 0,
                                  states['MT_VM'] * params['MT_SP500_ORIG'][:, cp.newaxis] / mt_vm_orig,
                                  states['MT_SP500'])
    states['MT_EAFE'] = cp.where(mt_vm_orig > 0,
                                 states['MT_VM'] * params['MT_EAFE_ORIG'][:, cp.newaxis] / mt_vm_orig,
                                 states['MT_EAFE'])

    # === STEP 9: UPDATE SURVIVAL ===
    lapse = cp.zeros_like(qx)  # Simplified - would need lapse calculation
    states['TX_SURVIE'] = states['TX_SURVIE'] * (1 - qx) * (1 - lapse)

    # === STEP 10: CALCULATE CASH FLOWS ===
    # These would be accumulated for reporting
    vp_primes_garanties = primes_garanties * states['TX_ACTUALISATION']

    return {
        'primes_garanties': primes_garanties,
        'vp_primes_garanties': vp_primes_garanties,
        'withdrawal': withdrawal,
    }


def process_year_gpu(states, params, lookups, year, freq=12):
    """
    Process one full year (12 months) for all accounts × scenarios.

    Returns accumulated cash flows for the year.
    """
    # Accumulate cash flows
    year_cashflows = {
        'primes_garanties': cp.zeros_like(states['MT_VM']),
        'vp_primes_garanties': cp.zeros_like(states['MT_VM']),
        'withdrawals': cp.zeros_like(states['MT_VM']),
    }

    # Process each month
    for month in range(freq):
        month_cf = process_month_gpu(states, params, lookups, year, month, freq)

        # Accumulate
        year_cashflows['primes_garanties'] += month_cf['primes_garanties']
        year_cashflows['vp_primes_garanties'] += month_cf['vp_primes_garanties']
        year_cashflows['withdrawals'] += month_cf['withdrawal']

    return year_cashflows


# =============================================================================
# MAIN PROJECTION LOOP
# =============================================================================

def run_projection_gpu(states, params, lookups, n_years=100, freq=12):
    """
    Main projection loop: iterate through years.
    Each year, process all accounts × scenarios in parallel on GPU.
    """
    print("\nStarting GPU projection...")
    print("=" * 60)

    n_accounts, n_scenarios = states['MT_VM'].shape

    # Storage for results (keep on GPU, transfer at end)
    all_cashflows = []

    start_time = datetime.now()

    for year in range(n_years):
        if year % 10 == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            print(f"Processing year {year}/{n_years} (elapsed: {elapsed:.1f}s)")

        # Process this year on GPU (fully parallel)
        year_cf = process_year_gpu(states, params, lookups, year, freq)

        # Store results (keep on GPU for now)
        year_cf['year'] = year
        all_cashflows.append(year_cf)

    total_time = (datetime.now() - start_time).total_seconds()
    print(f"\nGPU projection completed in {total_time:.2f} seconds")
    print("=" * 60)

    return all_cashflows, states


# =============================================================================
# RESULT AGGREGATION
# =============================================================================

def aggregate_results_gpu(cashflows_list, states, params):
    """
    Aggregate results from GPU arrays.
    Transfer to CPU and create pandas DataFrames.
    """
    print("\nAggregating results...")

    n_accounts, n_scenarios = states['MT_VM'].shape

    # Transfer final states to CPU
    final_vm = cp.asnumpy(states['MT_VM'])
    final_survival = cp.asnumpy(states['TX_SURVIE'])

    # Aggregate cash flows across scenarios (average)
    print("  - Averaging across scenarios...")
    total_primes = cp.zeros(n_accounts, dtype=cp.float32)
    total_vp_primes = cp.zeros(n_accounts, dtype=cp.float32)

    for year_cf in cashflows_list:
        # Average across scenarios for each account
        total_primes += cp.mean(year_cf['primes_garanties'], axis=1)
        total_vp_primes += cp.mean(year_cf['vp_primes_garanties'], axis=1)

    # Transfer to CPU
    total_primes_cpu = cp.asnumpy(total_primes)
    total_vp_primes_cpu = cp.asnumpy(total_vp_primes)
    final_vm_avg = np.mean(final_vm, axis=1)

    # Create summary DataFrame
    summary = pd.DataFrame({
        'ID_COMPTE': range(n_accounts),
        'VP_PRIMES_GARANTIES': total_vp_primes_cpu,
        'PRIMES_GARANTIES_TOTAL': total_primes_cpu,
        'VALEUR_MARCHANDE_FINALE': final_vm_avg,
    })

    print("Aggregation complete")
    return summary


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_projection(data_path: str, output_path: str):
    """
    Main function to run GPU-accelerated projection.
    """
    start_time = datetime.now()
    print(f"Starting GPU projection at {start_time}")
    print("=" * 60)

    # Load data (CPU)
    data = load_all_data(Path(data_path))

    # Limit to small sample for testing
    # Remove this line for full run
    data['population'] = data['population'].head(100)  # Test with 100 accounts

    # Create GPU lookup tables
    lookups = create_gpu_lookups(data)

    # Create GPU state arrays
    n_scenarios = min(CONFIG['NB_SC'], 10)  # Start with 10 scenarios for testing
    states = create_state_arrays(data['population'], n_scenarios)
    params = create_account_params(data['population'])

    # Run projection on GPU
    cashflows, final_states = run_projection_gpu(
        states,
        params,
        lookups,
        n_years=CONFIG['NB_AN_PROJECTION'],
        freq=CONFIG['FREQ_EVAL']
    )

    # Aggregate results
    results = aggregate_results_gpu(cashflows, final_states, params)

    # Save outputs
    print("\nSaving outputs...")
    Path(output_path).mkdir(parents=True, exist_ok=True)
    results.to_csv(f"{output_path}/GPU_RESULTS.csv", index=False)
    print(f"  ✓ Saved {output_path}/GPU_RESULTS.csv")

    # Print summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print("\n" + "=" * 60)
    print("PROJECTION COMPLETE")
    print("=" * 60)
    print(f"Processing time: {duration:.2f} seconds")
    print(f"Accounts processed: {len(data['population'])}")
    print(f"Scenarios: {n_scenarios}")
    print(f"Years: {CONFIG['NB_AN_PROJECTION']}")
    print(f"Total computations: {len(data['population']) * n_scenarios * CONFIG['NB_AN_PROJECTION'] * 12:,}")
    print("=" * 60)

    return results


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Set paths
    DATA_PATH = HERE.joinpath("algo2/data_in")
    OUTPUT_PATH = HERE.joinpath("algo2/data_out")

    # Check if GPU is available
    try:
        print(f"GPU Device: {cp.cuda.Device()}")
        print(f"GPU Memory: {cp.cuda.Device().mem_info[1] / 1e9:.1f} GB total")
    except Exception as e:
        print(f"WARNING: GPU not available or CuPy not installed: {e}")
        print("Install CuPy with: pip install cupy-cuda12x")
        exit(1)

    # Run projection
    results = run_projection(
        data_path=DATA_PATH,
        output_path=OUTPUT_PATH
    )

    print("\nSample Results:")
    print(results.head(10))
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
# UTILITY FUNCTIONS (CPU-side)
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
# DATA LOADING
# =============================================================================

def clean_numeric_with_commas(df, columns):
    """Clean numeric columns that may have comma separators."""
    for col in columns:
        if col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].apply(
                    lambda x: float(str(x).replace(',', '')) if pd.notna(x) and str(x).strip() != '' else 0.0
                )
            else:
                # Already numeric, just ensure float
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    return df


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

    # Clean MT_ columns that might have comma separators
    mt_cols = [col for col in data['population'].columns if col.startswith('MT_')]
    data['population'] = clean_numeric_with_commas(data['population'], mt_cols)

    # Also clean other numeric columns
    other_numeric_cols = ['VAR_RETRAIT_FCT', 'VAR_DEPOT_FCT', 'I_EVEN_CESSE_DEPOT',
                          'AGE_MAX_DEPOT', 'FREQ_RESET_SRG', 'MAX_RESET_SRG',
                          'FREQ_RESET_DECES', 'MAX_RESET_DECES', 'AGE_CHANG_DECES',
                          'I_RESET_DECES_ECH', 'I_RESET_FACUL_ECH', 'MAX_RESET_FACUL_ECH',
                          'NB_AN_ECH', 'AGE_MAX_RENOUV_ECH', 'MAX_BONI_DECES',
                          'AGE_MRV_PERMIS', 'TABLE_TAUX_MRV_MRG_MRA', 'I_FRAIS_SUR_SRG',
                          'AJUSTEMENT_MENSUEL_GAR', 'RATIO_VM_VG_RESET_ECH',
                          'AJUSTEMENT_COMMISSION']
    data['population'] = clean_numeric_with_commas(data['population'], other_numeric_cols)

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
    """Create GPU-optimized lookup tables."""
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

    # Returns: [scenario (100), year (101), month (12), return_type (8)]
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

    # Lapse Part: [age (121), id_lapse (10), regime (5), niveau (3), field (2)]
    print("  - Lapse partial table...")
    lapse_part = np.zeros((121, 10, 5, 3, 2), dtype=np.float32)
    for _, row in data['tx_lapse_part'].iterrows():
        age = min(int(row['AGE']), 120)
        id_lapse = min(int(row['ID_LAPSE']), 9)
        regime = min(int(row['I_REGIME_2']), 4)
        niveau = int(row['LAPSE_NIV_PART']) - 1
        if niveau >= 0 and niveau < 3:
            lapse_part[age, id_lapse, regime, niveau, 0] = float(row['TX_LAPSE_PART_MIN'])
            lapse_part[age, id_lapse, regime, niveau, 1] = float(row['TX_LAPSE_PART_MAX'])
    lookups['lapse_part'] = cp.asarray(lapse_part)

    # Lapse Tot: [duree (10), id_lapse (10), niveau (3), field (3)]
    print("  - Lapse total table...")
    lapse_tot = np.zeros((10, 10, 3, 3), dtype=np.float32)
    for _, row in data['tx_lapse_tot'].iterrows():
        duree = min(int(row['DUREE_MAX10']) - 1, 9)
        id_lapse = min(int(row['ID_LAPSE']), 9)
        niveau = int(row['LAPSE_NIV_TOT']) - 1
        if niveau >= 0 and niveau < 3 and duree >= 0:
            lapse_tot[duree, id_lapse, niveau, 0] = float(row['TX_LAPSE_TOT_MIN'])
            lapse_tot[duree, id_lapse, niveau, 1] = float(row['TX_LAPSE_TOT_MAX'])
            lapse_tot[duree, id_lapse, niveau, 2] = float(row['FACT_DIM'])
    lookups['lapse_tot'] = cp.asarray(lapse_tot)

    # Deposits: [duree (10), id_depot (10), field (4)]
    print("  - Deposits table...")
    deposits = np.zeros((10, 10, 4), dtype=np.float32)
    for _, row in data['depots_futurs'].iterrows():
        duree = min(int(row['DUREE_MAX10']) - 1, 9)
        id_depot = min(int(row['ID_DEPOT']), 9)
        if duree >= 0:
            deposits[duree, id_depot, 0] = float(row['PC_DEPOT_ANNUEL'])
            deposits[duree, id_depot, 1] = float(row['VAR_DEPOT_FCT'])
            deposits[duree, id_depot, 2] = float(row['AGE_MAX_DEPOT'])
            deposits[duree, id_depot, 3] = float(row['I_EVEN_CESSE_DEPOT'])
    lookups['deposits'] = cp.asarray(deposits)

    # Acquisition: [duree (10), id_acqui (10), field (6)]
    print("  - Acquisition table...")
    acquisition = np.zeros((10, 10, 6), dtype=np.float32)
    for _, row in data['acquisition'].iterrows():
        duree = min(int(row['DUREE_MAX10']) - 1, 9)
        id_acqui = min(int(row['ID_ACQUI']), 9)
        if duree >= 0:
            acquisition[duree, id_acqui, 0] = float(row['PC_COMMISSION_VENTE_RF'])
            acquisition[duree, id_acqui, 1] = float(row['PC_COMMISSION_VENTE_AC'])
            acquisition[duree, id_acqui, 2] = float(row['PC_COMMISSION_MAINTIEN_RF'])
            acquisition[duree, id_acqui, 3] = float(row['PC_COMMISSION_MAINTIEN_AC'])
            acquisition[duree, id_acqui, 4] = float(row['PC_FRAIS_AN_AC'])
            acquisition[duree, id_acqui, 5] = float(row['PC_FRAIS_AN_RF'])
    lookups['acquisition'] = cp.asarray(acquisition)

    # Coussins: [code_cat (8), cat1 (6), cat2 (7), field (16)]
    print("  - Coussins table...")
    coussins = np.zeros((8, 6, 7, 16), dtype=np.float32)
    for _, row in data['coussins_escap'].iterrows():
        code = min(int(row['CODE_CAT_PRODUIT']), 7)
        cat1 = min(int(row['CAT_COUSSIN_1']), 5)
        cat2 = min(int(row['CAT_COUSSIN_2']), 6)
        coussins[code, cat1, cat2, 0] = float(row['BASE_PASSIF_REDRESSE'])
        coussins[code, cat1, cat2, 1] = float(row['TX_PASSIF_REDRESSE'])
        coussins[code, cat1, cat2, 2] = float(row['BASE_COUSSIN_CREDIT'])
        coussins[code, cat1, cat2, 3] = float(row['TX_COUSSIN_CREDIT'])
        coussins[code, cat1, cat2, 4] = float(row['BASE_COUSSIN_MARCHE'])
        coussins[code, cat1, cat2, 5] = float(row['TX_COUSSIN_MARCHE'])
        coussins[code, cat1, cat2, 6] = float(row['BASE_COUSSIN_DEPENSE'])
        coussins[code, cat1, cat2, 7] = float(row['TX_COUSSIN_DEPENSE'])
        coussins[code, cat1, cat2, 8] = float(row['BASE_COUSSIN_DECHEANCE'])
        coussins[code, cat1, cat2, 9] = float(row['TX_COUSSIN_DECHEANCE'])
        coussins[code, cat1, cat2, 10] = float(row['BASE_COUSSIN_MORTALITE'])
        coussins[code, cat1, cat2, 11] = float(row['TX_COUSSIN_MORTALITE'])
        coussins[code, cat1, cat2, 12] = float(row['BASE_COUSSIN_DEPOT'])
        coussins[code, cat1, cat2, 13] = float(row['TX_COUSSIN_DEPOT'])
        coussins[code, cat1, cat2, 14] = float(row['FACTEUR_AGE_80'])
        coussins[code, cat1, cat2, 15] = float(row['FACTEUR_AGE_90'])
    lookups['coussins'] = cp.asarray(coussins)

    # Fees: [product (30), year (200)]
    print("  - Fees table...")
    fees_df = data['frais_admin']
    max_year_fee = fees_df['ANNEE_REELLE'].max()
    min_year_fee = fees_df['ANNEE_REELLE'].min()
    year_range_fee = max_year_fee - min_year_fee + 1

    fees = np.zeros((30, year_range_fee), dtype=np.float32)
    for _, row in fees_df.iterrows():
        prod = min(int(row['ID_PRODUIT']), 29)
        year_idx = int(row['ANNEE_REELLE']) - min_year_fee
        fees[prod, year_idx] = float(row['FRAIS'])
    lookups['fees'] = cp.asarray(fees)
    lookups['fees_min_year'] = min_year_fee

    print("Lookup tables transferred to GPU")
    return lookups


# =============================================================================
# GPU STATE ARRAYS
# =============================================================================

def create_state_arrays(population: pd.DataFrame, n_scenarios: int) -> Dict[str, cp.ndarray]:
    """Create GPU arrays to hold state for all accounts × scenarios."""
    print("Creating state arrays on GPU...")

    n_accounts = len(population)

    states = {}

    # Market values and guarantees
    states['MT_VM_PROJ'] = cp.asarray(
        np.tile(population['MT_VM'].values[:, np.newaxis], (1, n_scenarios)).astype(np.float32))
    states['MT_GAR_DECES_PROJ'] = cp.asarray(
        np.tile(population['MT_GAR_DECES'].values[:, np.newaxis], (1, n_scenarios)).astype(np.float32))
    states['MT_GAR_ECH_PROJ'] = cp.asarray(
        np.tile(population['MT_GAR_ECH'].values[:, np.newaxis], (1, n_scenarios)).astype(np.float32))
    states['MT_SRG_PROJ'] = cp.asarray(
        np.tile(population['MT_SRG'].values[:, np.newaxis], (1, n_scenarios)).astype(np.float32))
    states['MT_BCB_PROJ'] = cp.asarray(
        np.tile(population.get('MT_BCB', pd.Series([0] * n_accounts)).values[:, np.newaxis],
                (1, n_scenarios)).astype(np.float32))

    # Asset allocations
    states['MT_DEX_PROJ'] = cp.asarray(
        np.tile(population['MT_DEX'].values[:, np.newaxis], (1, n_scenarios)).astype(np.float32))
    states['MT_MM_PROJ'] = cp.asarray(
        np.tile(population['MT_MM'].values[:, np.newaxis], (1, n_scenarios)).astype(np.float32))
    states['MT_TSX_PROJ'] = cp.asarray(
        np.tile(population['MT_TSX'].values[:, np.newaxis], (1, n_scenarios)).astype(np.float32))
    states['MT_SP500_PROJ'] = cp.asarray(
        np.tile(population['MT_SP500'].values[:, np.newaxis], (1, n_scenarios)).astype(np.float32))
    states['MT_EAFE_PROJ'] = cp.asarray(
        np.tile(population['MT_EAFE'].values[:, np.newaxis], (1, n_scenarios)).astype(np.float32))

    # Other state variables
    states['MT_BONI_DECES_PROJ'] = cp.zeros((n_accounts, n_scenarios), dtype=cp.float32)
    states['MT_MRV_MRG_MRA_PROJ'] = cp.asarray(
        np.tile(population.get('MT_MRV_MRG_MRA', pd.Series([0] * n_accounts)).values[:, np.newaxis],
                (1, n_scenarios)).astype(np.float32))
    states['TAUX_MRV_MRG_MRA_PROJ'] = cp.asarray(
        np.tile(population.get('TAUX_MRV_MRG_MRA', pd.Series([0] * n_accounts)).values[:, np.newaxis],
                (1, n_scenarios)).astype(np.float32))
    states['MT_MIN_FERR_PROJ'] = cp.zeros((n_accounts, n_scenarios), dtype=cp.float32)

    states['TX_SURVIE'] = cp.ones((n_accounts, n_scenarios), dtype=cp.float32)
    states['TX_SURVIE_DEB'] = cp.ones((n_accounts, n_scenarios), dtype=cp.float32)
    states['TX_ACTUALISATION'] = cp.ones((n_accounts, n_scenarios), dtype=cp.float32)

    # Maturity tracking
    states['ANNEE_ECH_PROJ'] = cp.asarray(
        np.tile(population.get('ANNEE_ECH', pd.Series([9999] * n_accounts)).values[:, np.newaxis],
                (1, n_scenarios)).astype(cp.int32))
    states['MOIS_ECH_PROJ'] = cp.asarray(
        np.tile(population.get('MOIS_ECH', pd.Series([12] * n_accounts)).values[:, np.newaxis],
                (1, n_scenarios)).astype(cp.int32))

    # Temporary calculation variables
    states['MT_VM_AV_RETRAIT_FRAIS'] = cp.zeros((n_accounts, n_scenarios), dtype=cp.float32)
    states['MT_VM_AV_RETRAIT'] = cp.zeros((n_accounts, n_scenarios), dtype=cp.float32)

    print(f"State arrays created: {n_accounts} accounts × {n_scenarios} scenarios")
    return states


def create_account_params(population: pd.DataFrame) -> Dict[str, cp.ndarray]:
    """Create GPU arrays for account parameters (read-only)."""
    print("Creating account parameter arrays on GPU...")

    params = {}
    n_accounts = len(population)

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
        population.get('ID_ACQUI', pd.Series([1] * n_accounts)).values.astype(np.int32))

    # Rates and percentages - FIX: Use correct column names
    params['PC_HONORAIRES_GEST'] = cp.asarray(population['PC_HONORAIRES_GEST'].values.astype(np.float32))
    params['PC_FRAIS_GARANTIE'] = cp.asarray(population['PC_FRAIS_GARANTIE'].values.astype(np.float32))
    params['PC_GAR_DECES_1'] = cp.asarray(population['PC_GAR_DECES_1'].values.astype(np.float32))
    params['PC_GAR_DECES_2'] = cp.asarray(
        population.get('PC_GAR_DECES_2', population['PC_GAR_DECES_1']).values.astype(np.float32))
    params['PC_BONI_DECES'] = cp.asarray(
        population.get('PC_BONI_DECES', pd.Series([0] * n_accounts)).values.astype(np.float32))
    params['PC_RFG'] = cp.asarray(
        population.get('PC_RFG', pd.Series([0] * n_accounts)).values.astype(np.float32))
    params['PC_REVENU_FDS'] = cp.asarray(
        population.get('PC_REVENU_FDS', pd.Series([0] * n_accounts)).values.astype(np.float32))
    params['PC_GAR_ECH'] = cp.asarray(
        population.get('PC_GAR_ECH', pd.Series([0] * n_accounts)).values.astype(np.float32))
    params['PC_GAR_ECH_DEP_FUT'] = cp.asarray(
        population.get('PC_GAR_ECH_DEP_FUT', pd.Series([0] * n_accounts)).values.astype(np.float32))
    params['PC_BONI_SRG'] = cp.asarray(
        population.get('PC_BONI_SRG', pd.Series([0] * n_accounts)).values.astype(np.float32))

    # Withdrawal parameters
    params['PC_RETRAIT_AGE'] = cp.asarray(
        population.get('PC_RETRAIT_AGE', pd.Series([1.0] * n_accounts)).values.astype(np.float32))
    params['MT_TPA_RETRAIT'] = cp.asarray(
        population.get('MT_TPA_RETRAIT', pd.Series([0] * n_accounts)).values.astype(np.float32))
    params['VAR_RETRAIT_FCT'] = cp.asarray(
        population.get('VAR_RETRAIT_FCT', pd.Series([1] * n_accounts)).values.astype(np.int32))
    params['MT_RETRAIT_MAX'] = cp.asarray(
        population.get('MT_RETRAIT_MAX', pd.Series([999999999] * n_accounts)).values.astype(np.float32))

    # Deposit parameters
    params['MT_TPA_DEPOT'] = cp.asarray(
        population.get('MT_TPA_DEPOT', pd.Series([0] * n_accounts)).values.astype(np.float32))

    # Evaluation dates
    params['ANNEE_EVALUATION_INI'] = cp.asarray(population['ANNEE_EVALUATION_INI'].values.astype(np.int32))
    params['MOIS_EVALUATION_INI'] = cp.asarray(population['MOIS_EVALUATION_INI'].values.astype(np.int32))
    params['ANNEE_COTIS'] = cp.asarray(
        population.get('ANNEE_COTIS', population['ANNEE_EVALUATION_INI']).values.astype(np.int32))
    params['MOIS_COTIS'] = cp.asarray(
        population.get('MOIS_COTIS', pd.Series([1] * n_accounts)).values.astype(np.int32))

    # Reset parameters
    params['FREQ_RESET_SRG'] = cp.asarray(
        population.get('FREQ_RESET_SRG', pd.Series([3] * n_accounts)).values.astype(np.int32))
    params['MAX_RESET_SRG'] = cp.asarray(
        population.get('MAX_RESET_SRG', pd.Series([80] * n_accounts)).values.astype(np.int32))
    params['FREQ_RESET_DECES'] = cp.asarray(
        population.get('FREQ_RESET_DECES', pd.Series([3] * n_accounts)).values.astype(np.int32))
    params['MAX_RESET_DECES'] = cp.asarray(
        population.get('MAX_RESET_DECES', pd.Series([80] * n_accounts)).values.astype(np.int32))
    params['AGE_CHANG_DECES'] = cp.asarray(
        population.get('AGE_CHANG_DECES', pd.Series([999] * n_accounts)).values.astype(np.int32))
    params['I_RESET_DECES_ECH'] = cp.asarray(
        population.get('I_RESET_DECES_ECH', pd.Series([0] * n_accounts)).values.astype(np.int32))
    params['I_RESET_FACUL_ECH'] = cp.asarray(
        population.get('I_RESET_FACUL_ECH', pd.Series([0] * n_accounts)).values.astype(np.int32))
    params['MAX_RESET_FACUL_ECH'] = cp.asarray(
        population.get('MAX_RESET_FACUL_ECH', pd.Series([80] * n_accounts)).values.astype(np.int32))
    params['RATIO_VM_VG_RESET_ECH'] = cp.asarray(
        population.get('RATIO_VM_VG_RESET_ECH', pd.Series([1.0] * n_accounts)).values.astype(np.float32))
    params['NB_AN_ECH'] = cp.asarray(
        population.get('NB_AN_ECH', pd.Series([10] * n_accounts)).values.astype(np.int32))
    params['AGE_MAX_RENOUV_ECH'] = cp.asarray(
        population.get('AGE_MAX_RENOUV_ECH', pd.Series([999] * n_accounts)).values.astype(np.int32))
    params['PC_RENOUV_ECH'] = cp.asarray(
        population.get('PC_RENOUV_ECH', pd.Series([1.0] * n_accounts)).values.astype(np.float32))
    params['MAX_BONI_DECES'] = cp.asarray(
        population.get('MAX_BONI_DECES', pd.Series([999] * n_accounts)).values.astype(np.int32))
    params['AGE_MRV_PERMIS'] = cp.asarray(
        population.get('AGE_MRV_PERMIS', pd.Series([65] * n_accounts)).values.astype(np.int32))
    params['TABLE_TAUX_MRV'] = cp.asarray(
        population.get('TABLE_TAUX_MRV_MRG_MRA', pd.Series([1] * n_accounts)).values.astype(np.int32))
    params['AJUSTEMENT_MENSUEL_GAR'] = cp.asarray(
        population.get('AJUSTEMENT_MENSUEL_GAR', pd.Series([0] * n_accounts)).values.astype(np.float32))
    params['I_FRAIS_SUR_SRG'] = cp.asarray(
        population.get('I_FRAIS_SUR_SRG', pd.Series([0] * n_accounts)).values.astype(np.int32))

    # Original VM for rebalancing
    params['MT_VM_ORIG'] = cp.asarray(population['MT_VM'].values.astype(np.float32))
    params['MT_DEX_ORIG'] = cp.asarray(population['MT_DEX'].values.astype(np.float32))
    params['MT_MM_ORIG'] = cp.asarray(population['MT_MM'].values.astype(np.float32))
    params['MT_TSX_ORIG'] = cp.asarray(population['MT_TSX'].values.astype(np.float32))
    params['MT_SP500_ORIG'] = cp.asarray(population['MT_SP500'].values.astype(np.float32))
    params['MT_EAFE_ORIG'] = cp.asarray(population['MT_EAFE'].values.astype(np.float32))

    # RF amount
    params['MT_RF'] = cp.asarray(
        population.get('MT_RF', pd.Series([0] * n_accounts)).values.astype(np.float32))
    params['AJUSTEMENT_COMMISSION'] = cp.asarray(
        population.get('AJUSTEMENT_COMMISSION', pd.Series([1.0] * n_accounts)).values.astype(np.float32))

    print(f"Account parameters created: {len(params)} parameters")
    return params


# =============================================================================
# GPU CALCULATION FUNCTIONS (Vectorized)
# =============================================================================

def calculate_age_gpu(params, annee_reelle, mois_eval, n_scenarios):
    """Calculate age for all accounts (vectorized)."""
    # Ensure annee_reelle is 1D
    if annee_reelle.ndim > 1:
        annee_reelle = annee_reelle.ravel()

    n_accounts = len(annee_reelle)

    # Ensure params are 1D
    annee_nais = params['ANNEE_NAIS']
    if annee_nais.ndim > 1:
        annee_nais = annee_nais.ravel()

    mois_nais = params['MOIS_NAIS']
    if mois_nais.ndim > 1:
        mois_nais = mois_nais.ravel()

    # Calculate (n_accounts, 1)
    age = annee_reelle[:, cp.newaxis] - annee_nais[:, cp.newaxis]
    age = cp.where(mois_eval < mois_nais[:, cp.newaxis], age - 1, age)
    age = cp.maximum(age, 1)

    # Verify shape
    assert age.shape == (n_accounts, 1), f"age intermediate shape: {age.shape}"

    # Broadcast to all scenarios (n_accounts, n_scenarios)
    age = cp.broadcast_to(age, (n_accounts, n_scenarios))

    # Verify final shape
    assert age.shape == (n_accounts,
                         n_scenarios), f"age final shape: {age.shape}, expected: {(n_accounts, n_scenarios)}"

    return age


def calculate_duree_max10_gpu(params, annee_reelle, mois_eval, n_scenarios):
    """Calculate duration from issue date (vectorized)."""
    # Ensure annee_reelle is 1D
    if annee_reelle.ndim > 1:
        annee_reelle = annee_reelle.ravel()

    n_accounts = len(annee_reelle)

    # Ensure params are 1D
    annee_cotis = params['ANNEE_COTIS']
    if annee_cotis.ndim > 1:
        annee_cotis = annee_cotis.ravel()

    mois_cotis = params['MOIS_COTIS']
    if mois_cotis.ndim > 1:
        mois_cotis = mois_cotis.ravel()

    # Calculate (n_accounts, 1)
    current_date = annee_reelle[:, cp.newaxis] + mois_eval / 12.0
    issue_date = annee_cotis[:, cp.newaxis] + mois_cotis[:, cp.newaxis] / 12.0

    duree = cp.floor(current_date - issue_date).astype(cp.int32) + 1
    duree = cp.clip(duree, 1, 10)

    # Verify shape
    assert duree.shape == (n_accounts, 1), f"duree intermediate shape: {duree.shape}"

    # Broadcast to all scenarios (n_accounts, n_scenarios)
    duree = cp.broadcast_to(duree, (n_accounts, n_scenarios))

    # Verify final shape
    assert duree.shape == (n_accounts,
                           n_scenarios), f"duree final shape: {duree.shape}, expected: {(n_accounts, n_scenarios)}"

    return duree


def lookup_mortality_gpu(lookups, params, age_mort, annee_reelle, n_scenarios):
    """Lookup mortality rates (vectorized)."""
    n_accounts = age_mort.shape[0]

    # Clip indices - broadcast params to match n_scenarios
    sex_idx = cp.clip(params['I_SEXE'], 0, 1)  # Shape: (n_accounts,)
    age_idx = cp.clip(age_mort, 0, 120)  # Shape: (n_accounts, n_scenarios)
    year_idx = cp.clip(annee_reelle - lookups['mortality_min_year'], 0,
                       lookups['mortality'].shape[2] - 1)  # Shape: (n_accounts,)
    prod_idx = cp.clip(params['I_PRODUIT_REGR'], 0, 9)  # Shape: (n_accounts,)

    # Broadcast params to match scenarios
    sex_idx_expanded = sex_idx[:, cp.newaxis].repeat(n_scenarios, axis=1)  # (n_accounts, n_scenarios)
    year_idx_expanded = year_idx[:, cp.newaxis].repeat(n_scenarios, axis=1)
    prod_idx_expanded = prod_idx[:, cp.newaxis].repeat(n_scenarios, axis=1)

    # Flatten for advanced indexing
    sex_flat = sex_idx_expanded.flatten()
    age_flat = age_idx.flatten()
    year_flat = year_idx_expanded.flatten()
    prod_flat = prod_idx_expanded.flatten()

    # Lookup using advanced indexing
    qx_flat = lookups['mortality'][sex_flat, age_flat, year_flat, prod_flat]

    # Reshape back to (n_accounts, n_scenarios)
    qx = qx_flat.reshape(n_accounts, n_scenarios)

    # Fallback for missing values
    qx = cp.where(qx == 0, 0.001, qx)
    return qx


def lookup_returns_gpu(lookups, scn_indices, year, month):
    """Lookup investment returns (vectorized)."""
    # returns shape: [scenario, year, month, return_type]
    n_scenarios = scn_indices.shape[0]

    returns_dict = {}
    for i, name in enumerate(['FORWARD_RATE', 'AJUST_FORWARD_RATE_VM_0', 'RENDDEX_AN',
                              'RENDMM_AN', 'RENDTSX_AN', 'RENDSP500_AN', 'RENDEAFE_AN']):
        # Get returns for all scenarios at this year/month
        returns_dict[name] = lookups['returns'][scn_indices, year, month, i]

    return returns_dict


def calculate_vm_vg_ratio_gpu(states, params):
    """Calculate VM/VG ratio (moneyness) - vectorized."""
    n_accounts, n_scenarios = states['MT_VM_PROJ'].shape

    # Get params as 1D arrays and broadcast to (n_accounts, n_scenarios)
    pc_gar_ech = cp.asarray(params['PC_GAR_ECH']).ravel()[:n_accounts, cp.newaxis]
    pc_gar_deces = cp.asarray(params['PC_GAR_DECES_1']).ravel()[:n_accounts, cp.newaxis]

    # Calculate ratios with protection against division by zero
    ratio1 = cp.where(states['MT_GAR_ECH_PROJ'] > 0,
                      pc_gar_ech / cp.maximum(states['MT_GAR_ECH_PROJ'], 0.01),
                      9999.0)

    ratio2 = pc_gar_deces / cp.maximum(
        states['MT_BONI_DECES_PROJ'] + states['MT_GAR_DECES_PROJ'], 0.01)

    ratio3 = cp.where(states['MT_SRG_PROJ'] > 0,
                      1.0 / cp.maximum(states['MT_SRG_PROJ'], 0.01),
                      9999.0)

    # VM is average of before and after withdrawal
    vm_avg = (states['MT_VM_PROJ'] + states['MT_VM_AV_RETRAIT_FRAIS']) / 2.0

    vm_vg_ratio = cp.minimum(10.0, vm_avg * cp.minimum(cp.minimum(ratio1, ratio2), ratio3))

    return vm_vg_ratio


def calculate_lapse_rates_gpu(states, params, lookups, age, duree_max10, freq, AJUST, vm_vg_ratio):
    """Calculate total and partial lapse rates (vectorized)."""
    n_accounts, n_scenarios = states['MT_VM_PROJ'].shape

    # Check if VM is 0 for RGS
    no_lapse = (states['MT_VM_PROJ'] == 0)

    # Calculate lapse levels based on moneyness
    lapse_niv_tot = cp.where(vm_vg_ratio <= 0.5, 0,
                             cp.where(vm_vg_ratio <= 0.75, 1, 2))
    lapse_niv_part = lapse_niv_tot  # Same logic

    # Lookup total lapse - vectorized
    duree_idx = cp.clip(duree_max10 - 1, 0, 9).astype(cp.int32)
    lapse_id_idx = cp.clip(params['ID_LAPSE'][:, cp.newaxis].repeat(n_scenarios, axis=1), 0, 9).astype(cp.int32)
    niv_tot_idx = cp.clip(lapse_niv_tot, 0, 2).astype(cp.int32)

    # Flatten for lookup
    duree_flat = duree_idx.flatten()
    lapse_id_flat = lapse_id_idx.flatten()
    niv_tot_flat = niv_tot_idx.flatten()

    # Lookup
    tx_lapse_tot_min_flat = lookups['lapse_tot'][duree_flat, lapse_id_flat, niv_tot_flat, 0]
    tx_lapse_tot_max_flat = lookups['lapse_tot'][duree_flat, lapse_id_flat, niv_tot_flat, 1]
    fact_dim_flat = lookups['lapse_tot'][duree_flat, lapse_id_flat, niv_tot_flat, 2]

    # Reshape
    tx_lapse_tot_min = tx_lapse_tot_min_flat.reshape(n_accounts, n_scenarios)
    tx_lapse_tot_max = tx_lapse_tot_max_flat.reshape(n_accounts, n_scenarios)
    fact_dim = fact_dim_flat.reshape(n_accounts, n_scenarios)

    # Interpolate total lapse
    interpolation_tot = cp.where(lapse_niv_tot == 0,
                                 cp.maximum(vm_vg_ratio, 0) / 0.5,
                                 cp.where(lapse_niv_tot == 1,
                                          (vm_vg_ratio - 0.5) / 0.25,
                                          (vm_vg_ratio - 0.75) / 999.24))

    lapse_tot = cp.where(tx_lapse_tot_min == tx_lapse_tot_max,
                         tx_lapse_tot_min,
                         interpolation_tot * (tx_lapse_tot_max - tx_lapse_tot_min) + tx_lapse_tot_min)

    # Apply diminution factor for decumulation
    age_retrait = age + 1
    lapse_tot = cp.where(age_retrait >= params['AGE_DECAISSEMENT'][:, cp.newaxis],
                         lapse_tot * fact_dim,
                         lapse_tot)

    # Lookup partial lapse - vectorized
    age_idx = cp.clip(age, 0, 120).astype(cp.int32)
    regime_idx = cp.clip(params['I_REGIME_2'][:, cp.newaxis].repeat(n_scenarios, axis=1), 0, 4).astype(cp.int32)
    niv_part_idx = cp.clip(lapse_niv_part, 0, 2).astype(cp.int32)

    # Flatten
    age_flat = age_idx.flatten()
    lapse_id_flat = lapse_id_idx.flatten()  # Reuse from above
    regime_flat = regime_idx.flatten()
    niv_part_flat = niv_part_idx.flatten()

    # Lookup
    tx_lapse_part_min_flat = lookups['lapse_part'][age_flat, lapse_id_flat, regime_flat, niv_part_flat, 0]
    tx_lapse_part_max_flat = lookups['lapse_part'][age_flat, lapse_id_flat, regime_flat, niv_part_flat, 1]

    # Reshape
    tx_lapse_part_min = tx_lapse_part_min_flat.reshape(n_accounts, n_scenarios)
    tx_lapse_part_max = tx_lapse_part_max_flat.reshape(n_accounts, n_scenarios)

    # Interpolate partial lapse
    interpolation_part = cp.where(lapse_niv_part == 0,
                                  cp.maximum(vm_vg_ratio, 0) / 0.5,
                                  cp.where(lapse_niv_part == 1,
                                           (vm_vg_ratio - 0.5) / 0.25,
                                           (vm_vg_ratio - 0.75) / 999.24))

    lapse_part = cp.where(tx_lapse_part_min == tx_lapse_part_max,
                          tx_lapse_part_min,
                          interpolation_part * (tx_lapse_part_max - tx_lapse_part_min) + tx_lapse_part_min)

    # Convert annual rates to period rates
    exponent = (1.0 / freq) * AJUST
    lapse = 1.0 - cp.power(1.0 - lapse_tot - lapse_part, exponent)

    # Set to zero where no lapse should occur
    lapse = cp.where(no_lapse, 0.0, lapse)
    lapse_tot = cp.where(no_lapse, 0.0, lapse_tot)
    lapse_part = cp.where(no_lapse, 0.0, lapse_part)

    return lapse_tot, lapse_part, lapse


def process_deposits_gpu(states, params, lookups, age, duree_max10, freq):
    """Process deposits and update guarantees (vectorized)."""
    n_accounts, n_scenarios = states['MT_VM_PROJ'].shape

    # Lookup deposit parameters - vectorized
    duree_idx = cp.clip(duree_max10 - 1, 0, 9).astype(cp.int32)
    depot_id_idx = cp.clip(params['ID_DEPOT'][:, cp.newaxis].repeat(n_scenarios, axis=1), 0, 9).astype(cp.int32)

    # Flatten
    duree_flat = duree_idx.flatten()
    depot_id_flat = depot_id_idx.flatten()

    # Lookup
    pc_depot_annuel_flat = lookups['deposits'][duree_flat, depot_id_flat, 0]
    var_depot_fct_flat = lookups['deposits'][duree_flat, depot_id_flat, 1]
    age_max_depot_flat = lookups['deposits'][duree_flat, depot_id_flat, 2]
    i_even_cesse_depot_flat = lookups['deposits'][duree_flat, depot_id_flat, 3]

    # Reshape
    pc_depot_annuel = pc_depot_annuel_flat.reshape(n_accounts, n_scenarios)
    var_depot_fct = var_depot_fct_flat.reshape(n_accounts, n_scenarios).astype(cp.int32)
    age_max_depot = age_max_depot_flat.reshape(n_accounts, n_scenarios).astype(cp.int32)
    i_even_cesse_depot = i_even_cesse_depot_flat.reshape(n_accounts, n_scenarios).astype(cp.int32)

    age_retrait = age + 1
    age_decaissement = params['AGE_DECAISSEMENT'][:, cp.newaxis]
    mt_tpa_depot = params['MT_TPA_DEPOT'][:, cp.newaxis]

    # Check if deposits should cease
    should_cease = (
            (pc_depot_annuel == 0) |
            ((i_even_cesse_depot == 1) & (age_retrait >= age_decaissement)) |
            (age_max_depot < age) |
            ((states['MT_VM_PROJ'] <= 0) & (params['I_PRODUIT_REGR'][:, cp.newaxis] == 0))
    )

    # Calculate deposit amount
    depot_futur = cp.zeros((n_accounts, n_scenarios), dtype=cp.float32)

    # Use TPA if available
    depot_futur = cp.where(mt_tpa_depot > 0, mt_tpa_depot, depot_futur)

    # Otherwise calculate based on var_depot_fct
    base = cp.where(var_depot_fct == 1,
                    states['MT_VM_PROJ'],
                    params['MT_VM_ORIG'][:, cp.newaxis] * params['PC_GAR_DECES_1'][:, cp.newaxis])

    depot_futur = cp.where((mt_tpa_depot == 0) & ~should_cease,
                           base * pc_depot_annuel,
                           depot_futur)

    # Adjust for frequency
    depot_futur = depot_futur / freq

    # Set to zero where should cease
    depot_futur = cp.where(should_cease, 0.0, depot_futur)

    # Allocate proportionally to asset classes
    total_vm = states['MT_VM_PROJ']
    has_vm = (total_vm > 0)

    states['MT_DEX_PROJ'] = cp.where(has_vm & (depot_futur > 0),
                                     states['MT_DEX_PROJ'] + depot_futur * (states['MT_DEX_PROJ'] / total_vm),
                                     states['MT_DEX_PROJ'])
    states['MT_MM_PROJ'] = cp.where(has_vm & (depot_futur > 0),
                                    states['MT_MM_PROJ'] + depot_futur * (states['MT_MM_PROJ'] / total_vm),
                                    states['MT_MM_PROJ'])
    states['MT_TSX_PROJ'] = cp.where(has_vm & (depot_futur > 0),
                                     states['MT_TSX_PROJ'] + depot_futur * (states['MT_TSX_PROJ'] / total_vm),
                                     states['MT_TSX_PROJ'])
    states['MT_SP500_PROJ'] = cp.where(has_vm & (depot_futur > 0),
                                       states['MT_SP500_PROJ'] + depot_futur * (states['MT_SP500_PROJ'] / total_vm),
                                       states['MT_SP500_PROJ'])
    states['MT_EAFE_PROJ'] = cp.where(has_vm & (depot_futur > 0),
                                      states['MT_EAFE_PROJ'] + depot_futur * (states['MT_EAFE_PROJ'] / total_vm),
                                      states['MT_EAFE_PROJ'])

    # Update guarantees
    states['MT_GAR_DECES_PROJ'] = cp.where(depot_futur > 0,
                                           states['MT_GAR_DECES_PROJ'] + depot_futur,
                                           states['MT_GAR_DECES_PROJ'])
    states['MT_GAR_ECH_PROJ'] = cp.where(depot_futur > 0,
                                         states['MT_GAR_ECH_PROJ'] + depot_futur * params['PC_GAR_ECH_DEP_FUT'][:,
                                                                                   cp.newaxis],
                                         states['MT_GAR_ECH_PROJ'])
    states['MT_SRG_PROJ'] = cp.where((depot_futur > 0) & (states['MT_SRG_PROJ'] > 0),
                                     states['MT_SRG_PROJ'] + depot_futur,
                                     states['MT_SRG_PROJ'])

    return depot_futur


def calculate_mrv_amount_gpu(states, params, age, mois_eval, an_eval, freq):
    """Calculate MRV/MRG/MRA amount for RGS products (vectorized)."""
    is_rgs = (params['I_PRODUIT_REGR'][:, cp.newaxis] == 1)

    age_retrait = age + 1
    age_mrv_permis = params['AGE_MRV_PERMIS'][:, cp.newaxis]
    table_taux_mrv = params['TABLE_TAUX_MRV'][:, cp.newaxis]

    # Check if withdrawals should cease
    base_amount = cp.where(table_taux_mrv == 1, states['MT_SRG_PROJ'], states['MT_VM_PROJ'])
    should_cease = (age_retrait < age_mrv_permis) & (base_amount == 0)

    # Only recalculate at end of year
    is_year_end = (mois_eval == 12 / freq)

    # RGS 2.1 logic
    is_rgs_21 = (table_taux_mrv == 2)

    should_reinit = (
            (age_retrait == cp.maximum(age_mrv_permis, params['AGE_DECAISSEMENT'][:, cp.newaxis])) |
            ((states['MT_SRG_PROJ'] == states['MT_VM_PROJ']) & (states['MT_VM_PROJ'] != 0))
    )

    # Determine rate based on age for RGS 2.1
    new_rate = cp.where(age_retrait < 60, 0.03,
                        cp.where(age_retrait < 65, 0.035,
                                 cp.where(age_retrait < 70, 0.04,
                                          cp.where(age_retrait < 75, 0.0425, 0.05))))

    # Update rate when reinitialization occurs
    states['TAUX_MRV_MRG_MRA_PROJ'] = cp.where(
        is_rgs & is_rgs_21 & is_year_end & should_reinit,
        new_rate,
        states['TAUX_MRV_MRG_MRA_PROJ']
    )

    # Calculate MRV amount for RGS 2.1
    new_mrv_21 = cp.where(
        should_reinit,
        states['TAUX_MRV_MRG_MRA_PROJ'] * states['MT_SRG_PROJ'],
        cp.maximum(states['MT_MRV_MRG_MRA_PROJ'],
                   states['TAUX_MRV_MRG_MRA_PROJ'] * states['MT_SRG_PROJ'])
    )

    # Calculate MRV amount for RGS 1 and 2
    new_mrv_12 = cp.where(
        age_retrait == age_mrv_permis,
        states['TAUX_MRV_MRG_MRA_PROJ'] * states['MT_SRG_PROJ'],
        cp.maximum(states['MT_MRV_MRG_MRA_PROJ'],
                   states['TAUX_MRV_MRG_MRA_PROJ'] * states['MT_SRG_PROJ'])
    )

    # Update MRV amount
    states['MT_MRV_MRG_MRA_PROJ'] = cp.where(
        is_rgs & is_year_end & ~should_cease,
        cp.where(is_rgs_21, new_mrv_21, new_mrv_12),
        states['MT_MRV_MRG_MRA_PROJ']
    )

    # Set to zero where should cease
    states['MT_MRV_MRG_MRA_PROJ'] = cp.where(
        is_rgs & should_cease,
        0.0,
        states['MT_MRV_MRG_MRA_PROJ']
    )


def calculate_withdrawal_gpu(states, params, lookups, age, mois_eval, an_eval, freq):
    """Calculate total withdrawal amount (vectorized)."""
    age_retrait = age + 1
    age_decaissement = params['AGE_DECAISSEMENT'][:, cp.newaxis]
    mois_nais = params['MOIS_NAIS'][:, cp.newaxis]

    # No withdrawal if conditions not met
    no_withdrawal = (
            (age_retrait < age_decaissement) |
            ((age_retrait == age_decaissement) & (mois_eval >= mois_nais)) |
            ((states['MT_VM_PROJ'] <= 0) & (params['I_PRODUIT_REGR'][:, cp.newaxis] == 0))
    )

    # Lookup minimum FERR rate - vectorized
    age_idx = cp.clip(age, 0, 120).astype(cp.int32)

    # Expand to scenarios
    age_idx_expanded = age_idx  # Already (n_accounts, n_scenarios)

    # Flatten and lookup
    age_flat = age_idx_expanded.flatten()
    min_ferr_rate_flat = lookups['min_ferr'][age_flat]
    min_ferr_rate = min_ferr_rate_flat.reshape(age.shape)

    # Calculate MIN_FERR_PROJ at start of year
    is_year_start = (
            ((an_eval == 1) & (mois_eval == params['MOIS_EVALUATION_INI'][:, cp.newaxis])) |
            (mois_eval == 12 / freq)
    )

    states['MT_MIN_FERR_PROJ'] = cp.where(
        is_year_start,
        states['MT_VM_PROJ'] * min_ferr_rate,
        states['MT_MIN_FERR_PROJ']
    )

    min_withdrawal = states['MT_MIN_FERR_PROJ']

    # Get withdrawal parameters
    var_retrait_fct = params['VAR_RETRAIT_FCT'][:, cp.newaxis]
    mt_tpa_retrait = params['MT_TPA_RETRAIT'][:, cp.newaxis]
    pc_retrait_age = params['PC_RETRAIT_AGE'][:, cp.newaxis]
    mt_retrait_max = params['MT_RETRAIT_MAX'][:, cp.newaxis]

    # Calculate retrait based on VAR_RETRAIT_FCT
    # Mode 1: TPA or percentage-based
    retrait_1 = cp.where(mt_tpa_retrait > 0,
                         mt_tpa_retrait,
                         states['MT_VM_PROJ'] * pc_retrait_age)

    # Mode 2: Max of TPA or MIN_FERR
    retrait_2 = cp.where(mt_tpa_retrait > min_withdrawal,
                         mt_tpa_retrait,
                         min_withdrawal * cp.maximum(pc_retrait_age, 1.0))

    # Mode 3: Max of MIN_FERR or MRV
    retrait_3 = cp.maximum(min_withdrawal, states['MT_MRV_MRG_MRA_PROJ']) * pc_retrait_age

    # Select based on var_retrait_fct
    retrait = cp.where(var_retrait_fct == 1, retrait_1,
                       cp.where(var_retrait_fct == 2, retrait_2,
                                cp.where(var_retrait_fct == 3, retrait_3, 0.0)))

    # Apply maximum and frequency adjustment
    retrait = cp.minimum(retrait, mt_retrait_max) / freq

    # Set to zero where no withdrawal
    retrait = cp.where(no_withdrawal, 0.0, retrait)

    return retrait


def accumulate_death_bonus_gpu(states, params, age, freq, AJUST):
    """Accumulate death bonus (vectorized)."""
    pc_boni_deces = params['PC_BONI_DECES'][:, cp.newaxis]
    max_boni_deces = params['MAX_BONI_DECES'][:, cp.newaxis]

    should_accumulate = (pc_boni_deces > 0) & (age < max_boni_deces)

    states['MT_BONI_DECES_PROJ'] = cp.where(
        should_accumulate,
        states['MT_BONI_DECES_PROJ'] + states['MT_GAR_DECES_PROJ'] * pc_boni_deces / freq * AJUST,
        cp.where(pc_boni_deces > 0, 0.0, states['MT_BONI_DECES_PROJ'])
    )


def apply_fees_and_update_vm_gpu(states, params, freq, AJUST, tx_survie_deb):
    """Apply management and guarantee fees (vectorized)."""
    # Step 1: Apply management fees (PC_RFG)
    pc_rfg = params['PC_RFG'][:, cp.newaxis]
    mt_vm_av_retrait = states['MT_VM_AV_RETRAIT_FRAIS'] * cp.exp(-pc_rfg / freq * AJUST)

    # Step 2: Calculate guarantee fee base
    i_frais_sur_srg = params['I_FRAIS_SUR_SRG'][:, cp.newaxis]
    base_fee_calc = cp.where(i_frais_sur_srg == 0,
                             mt_vm_av_retrait,
                             states['MT_SRG_PROJ'])

    # Calculate fee amount
    pc_frais_garantie = params['PC_FRAIS_GARANTIE'][:, cp.newaxis]
    guarantee_fee_amount = base_fee_calc * pc_frais_garantie / freq * AJUST
    guarantee_fee_amount = cp.minimum(guarantee_fee_amount, mt_vm_av_retrait)

    # Calculate cash flow
    primes_garanties = guarantee_fee_amount * tx_survie_deb
    vp_primes_garanties = primes_garanties * states['TX_ACTUALISATION']

    # Step 3: Deduct fee from VM
    mt_vm_av_retrait_final = cp.maximum(mt_vm_av_retrait - guarantee_fee_amount, 0)

    # Update state
    states['MT_VM_AV_RETRAIT'] = mt_vm_av_retrait_final

    return primes_garanties, vp_primes_garanties


def update_guarantees_for_withdrawal_gpu(states, retrait):
    """Update guarantees proportionally to withdrawal (vectorized)."""
    mt_vm_av_retrait = states['MT_VM_AV_RETRAIT']

    # If VM <= withdrawal, set all guarantees to zero
    exceeds_vm = (mt_vm_av_retrait <= retrait)

    # Calculate proportion for partial withdrawal
    proportion = cp.where(mt_vm_av_retrait > 0,
                          1.0 - retrait / mt_vm_av_retrait,
                          0.0)

    # Update guarantees
    states['MT_GAR_ECH_PROJ'] = cp.where(exceeds_vm, 0.0,
                                         states['MT_GAR_ECH_PROJ'] * proportion)
    states['MT_GAR_DECES_PROJ'] = cp.where(exceeds_vm, 0.0,
                                           states['MT_GAR_DECES_PROJ'] * proportion)
    states['MT_BONI_DECES_PROJ'] = cp.where(exceeds_vm, 0.0,
                                            states['MT_BONI_DECES_PROJ'] * proportion)
    states['MT_SRG_PROJ'] = cp.maximum(states['MT_SRG_PROJ'] - retrait, 0.0)

    # VM after withdrawal
    mt_vm_ap_retrait = cp.maximum(mt_vm_av_retrait - retrait, 0.0)

    return mt_vm_ap_retrait


def calculate_death_benefit_gpu(states, params, qx, tx_survie_deb, mt_vm_ap_retrait_depot):
    """Calculate death benefit (vectorized)."""
    prest_deces = qx * -cp.maximum(0.0,
                                   states['MT_GAR_DECES_PROJ'] + states['MT_BONI_DECES_PROJ'] - mt_vm_ap_retrait_depot
                                   ) * tx_survie_deb
    vp_prest_deces = prest_deces * states['TX_ACTUALISATION']

    return prest_deces, vp_prest_deces


def calculate_mrv_benefit_gpu(states, params, retrait, mt_vm_av_retrait, tx_survie_deb):
    """Calculate MRV benefit (vectorized)."""
    is_rgs = (params['I_PRODUIT_REGR'][:, cp.newaxis] == 1)

    prest_mrv = cp.where(is_rgs,
                         -cp.maximum(retrait - mt_vm_av_retrait, 0.0) * tx_survie_deb,
                         0.0)
    vp_prest_mrv = prest_mrv * states['TX_ACTUALISATION']

    return prest_mrv, vp_prest_mrv


def process_maturity_benefit_gpu(states, params, age, annee_reelle, mois_eval, mt_vm_ap_retrait, freq):
    """Process maturity benefit if it occurs (vectorized)."""
    annee_ech_proj = states['ANNEE_ECH_PROJ']
    mois_ech_proj = states['MOIS_ECH_PROJ']
    age_fin_contrat = params['AGE_FIN_CONTRAT'][:, cp.newaxis]
    mois_nais = params['MOIS_NAIS'][:, cp.newaxis]

    # Check if maturity occurs
    maturity_by_date = (annee_reelle[:, cp.newaxis] == annee_ech_proj) & (mois_eval == mois_ech_proj)

    target_month = cp.where(mois_nais == 12 / freq, 12, mois_nais - 12 / freq)
    maturity_by_age = (age == age_fin_contrat) & (mois_eval == target_month)

    maturity_occurs = maturity_by_date | maturity_by_age

    # Calculate maturity benefit
    prest_ech = cp.where(maturity_occurs,
                         -cp.maximum(0.0, states['MT_GAR_ECH_PROJ'] - mt_vm_ap_retrait) * states['TX_SURVIE'],
                         0.0)

    # Update maturity parameters
    nb_an_ech = params['NB_AN_ECH'][:, cp.newaxis]
    states['ANNEE_ECH_PROJ'] = cp.where(maturity_occurs,
                                        annee_ech_proj + nb_an_ech,
                                        annee_ech_proj)
    states['MOIS_ECH_PROJ'] = cp.where(maturity_occurs,
                                       mois_eval,
                                       mois_ech_proj)

    # Update VM and guarantees
    top_up = cp.maximum(0.0, states['MT_GAR_ECH_PROJ'] - mt_vm_ap_retrait)
    states['MT_VM_PROJ'] = cp.where(maturity_occurs,
                                    mt_vm_ap_retrait + top_up,
                                    states['MT_VM_PROJ'])

    pc_gar_ech = params['PC_GAR_ECH'][:, cp.newaxis]
    states['MT_GAR_ECH_PROJ'] = cp.where(maturity_occurs,
                                         states['MT_VM_PROJ'] * pc_gar_ech,
                                         states['MT_GAR_ECH_PROJ'])

    # Reset death guarantee if applicable
    i_reset_deces_ech = params['I_RESET_DECES_ECH'][:, cp.newaxis]
    pc_gar_deces = params['PC_GAR_DECES_1'][:, cp.newaxis]
    states['MT_GAR_DECES_PROJ'] = cp.where(maturity_occurs & (i_reset_deces_ech == 1),
                                           states['MT_VM_PROJ'] * pc_gar_deces,
                                           states['MT_GAR_DECES_PROJ'])

    # Apply renewal rate
    age_max_renouv_ech = params['AGE_MAX_RENOUV_ECH'][:, cp.newaxis]
    pc_renouv_ech = cp.where(age > age_max_renouv_ech, 0.0,
                             params['PC_RENOUV_ECH'][:, cp.newaxis])
    states['TX_SURVIE'] = cp.where(maturity_occurs,
                                   states['TX_SURVIE'] * pc_renouv_ech,
                                   states['TX_SURVIE'])

    vp_prest_ech = prest_ech * states['TX_ACTUALISATION']

    return prest_ech, vp_prest_ech


def update_death_guarantee_adjustments_gpu(states, params, freq):
    """Apply adjustments to death guarantee (vectorized)."""
    ajustement_mensuel_gar = params['AJUSTEMENT_MENSUEL_GAR'][:, cp.newaxis]
    states['MT_GAR_DECES_PROJ'] = states['MT_GAR_DECES_PROJ'] - ajustement_mensuel_gar * 12 / freq


def process_srg_bcb_resets_gpu(states, params, age, annee_reelle, mois_eval):
    """Process SRG/BCB resets for RGS products (vectorized)."""
    is_rgs = (params['I_PRODUIT_REGR'][:, cp.newaxis] == 1)

    annee_cotis = params['ANNEE_COTIS'][:, cp.newaxis]
    mois_cotis = params['MOIS_COTIS'][:, cp.newaxis]
    freq_reset_srg = params['FREQ_RESET_SRG'][:, cp.newaxis]
    max_reset_srg = params['MAX_RESET_SRG'][:, cp.newaxis]

    # Check if SRG reset should occur
    years_since_issue = annee_reelle[:, cp.newaxis] - annee_cotis
    is_reset_year = (years_since_issue % freq_reset_srg == 0) & (years_since_issue > 0)
    is_reset_month = (mois_eval == mois_cotis)

    should_reset = (
            is_rgs &
            (age < max_reset_srg) &
            (states['MT_SRG_PROJ'] < states['MT_VM_PROJ']) &
            is_reset_year &
            is_reset_month
    )

    states['MT_SRG_PROJ'] = cp.where(should_reset,
                                     states['MT_VM_PROJ'],
                                     states['MT_SRG_PROJ'])
    states['MT_BCB_PROJ'] = cp.where(should_reset,
                                     cp.maximum(states['MT_BCB_PROJ'], states['MT_VM_PROJ']),
                                     states['MT_BCB_PROJ'])

    # Bonus to SRG if not yet in decumulation
    age_decaissement = params['AGE_DECAISSEMENT'][:, cp.newaxis]
    pc_boni_srg = params['PC_BONI_SRG'][:, cp.newaxis]

    should_add_bonus = is_rgs & (age < age_decaissement) & (mois_eval == 12)

    states['MT_SRG_PROJ'] = cp.where(should_add_bonus,
                                     states['MT_SRG_PROJ'] + pc_boni_srg * states['MT_BCB_PROJ'],
                                     states['MT_SRG_PROJ'])


def process_death_guarantee_resets_gpu(states, params, age, annee_reelle, mois_eval, freq):
    """Process automatic death guarantee resets (vectorized)."""
    annee_cotis = params['ANNEE_COTIS'][:, cp.newaxis]
    mois_cotis = params['MOIS_COTIS'][:, cp.newaxis]
    freq_reset_deces = params['FREQ_RESET_DECES'][:, cp.newaxis]
    max_reset_deces = params['MAX_RESET_DECES'][:, cp.newaxis]
    mois_nais = params['MOIS_NAIS'][:, cp.newaxis]
    pc_gar_deces = params['PC_GAR_DECES_1'][:, cp.newaxis]

    # Check if death guarantee reset should occur
    years_since_issue = annee_reelle[:, cp.newaxis] - annee_cotis
    is_reset_year = (years_since_issue % freq_reset_deces == 0) & (years_since_issue > 0)
    is_reset_month = (mois_eval == mois_cotis)

    # Regular reset condition
    regular_reset = (
            (age < max_reset_deces) &
            ((states['MT_GAR_DECES_PROJ'] + states['MT_BONI_DECES_PROJ']) <
             (states['MT_VM_PROJ'] * pc_gar_deces)) &
            is_reset_year &
            is_reset_month
    )

    # Final reset at max age
    target_month = cp.where(mois_nais == 12 / freq, 12, mois_nais - 12 / freq)
    final_reset = (
            (age == max_reset_deces - 1) &
            (mois_eval == target_month) &
            ((states['MT_GAR_DECES_PROJ'] + states['MT_BONI_DECES_PROJ']) <
             (states['MT_VM_PROJ'] * pc_gar_deces))
    )

    should_reset = regular_reset | final_reset

    states['MT_GAR_DECES_PROJ'] = cp.where(should_reset,
                                           states['MT_VM_PROJ'] * pc_gar_deces,
                                           states['MT_GAR_DECES_PROJ'])
    states['MT_BONI_DECES_PROJ'] = cp.where(should_reset,
                                            0.0,
                                            states['MT_BONI_DECES_PROJ'])


def process_facultative_maturity_reset_gpu(states, params, age, annee_reelle, mois_eval):
    """Process facultative maturity guarantee reset (vectorized)."""
    # Only in June and December
    is_june_or_dec = (mois_eval == 6) | (mois_eval == 12)

    i_reset_facul_ech = params['I_RESET_FACUL_ECH'][:, cp.newaxis]
    max_reset_facul_ech = params['MAX_RESET_FACUL_ECH'][:, cp.newaxis]
    ratio_vm_vg_reset_ech = params['RATIO_VM_VG_RESET_ECH'][:, cp.newaxis]
    nb_an_ech = params['NB_AN_ECH'][:, cp.newaxis]
    age_ech_min = params['AGE_ECH_MIN'][:, cp.newaxis]
    annee_nais = params['ANNEE_NAIS'][:, cp.newaxis]
    mois_nais = params['MOIS_NAIS'][:, cp.newaxis]
    pc_gar_ech = params['PC_GAR_ECH'][:, cp.newaxis]

    # Check conditions for facultative reset
    should_reset = (
            is_june_or_dec &
            (i_reset_facul_ech == 1) &
            (age <= max_reset_facul_ech) &
            (states['MT_GAR_ECH_PROJ'] > 0) &
            ((states['MT_VM_PROJ'] * pc_gar_ech) >= ratio_vm_vg_reset_ech * states['MT_GAR_ECH_PROJ'])
    )

    states['MT_GAR_ECH_PROJ'] = cp.where(should_reset,
                                         cp.maximum(states['MT_GAR_ECH_PROJ'],
                                                    states['MT_VM_PROJ'] * pc_gar_ech),
                                         states['MT_GAR_ECH_PROJ'])

    new_annee_ech = cp.maximum(annee_reelle[:, cp.newaxis] + nb_an_ech,
                               annee_nais + age_ech_min)
    states['ANNEE_ECH_PROJ'] = cp.where(should_reset,
                                        new_annee_ech,
                                        states['ANNEE_ECH_PROJ'])

    states['MOIS_ECH_PROJ'] = cp.where(should_reset,
                                       cp.where(new_annee_ech == annee_nais + age_ech_min,
                                                mois_nais,
                                                mois_eval),
                                       states['MOIS_ECH_PROJ'])


def process_death_guarantee_age_change_gpu(states, params, age, mois_eval, freq):
    """Change death guarantee percentage if age threshold reached (vectorized)."""
    n_accounts, n_scenarios = states['MT_VM_PROJ'].shape

    age_chang_deces = params['AGE_CHANG_DECES'][:, cp.newaxis]
    pc_gar_deces_2 = params['PC_GAR_DECES_2'][:, cp.newaxis]
    pc_gar_deces_1 = params['PC_GAR_DECES_1'][:, cp.newaxis]
    mois_nais = params['MOIS_NAIS'][:, cp.newaxis]

    target_month = cp.where(mois_nais == 12 / freq, 12, mois_nais - 12 / freq)

    should_change = (age == age_chang_deces - 1) & (mois_eval == target_month)

    states['MT_GAR_DECES_PROJ'] = cp.where(should_change,
                                           states['MT_GAR_DECES_PROJ'] * pc_gar_deces_2 / pc_gar_deces_1,
                                           states['MT_GAR_DECES_PROJ'])


def rebalance_portfolio_gpu(states, params):
    """Rebalance portfolio to original allocation (vectorized)."""
    mt_vm_orig = params['MT_VM_ORIG'][:, cp.newaxis]

    has_vm = (mt_vm_orig > 0) & (states['MT_VM_PROJ'] > 0)

    states['MT_SP500_PROJ'] = cp.where(has_vm,
                                       states['MT_VM_PROJ'] * params['MT_SP500_ORIG'][:, cp.newaxis] / mt_vm_orig,
                                       states['MT_SP500_PROJ'])
    states['MT_TSX_PROJ'] = cp.where(has_vm,
                                     states['MT_VM_PROJ'] * params['MT_TSX_ORIG'][:, cp.newaxis] / mt_vm_orig,
                                     states['MT_TSX_PROJ'])
    states['MT_EAFE_PROJ'] = cp.where(has_vm,
                                      states['MT_VM_PROJ'] * params['MT_EAFE_ORIG'][:, cp.newaxis] / mt_vm_orig,
                                      states['MT_EAFE_PROJ'])
    states['MT_DEX_PROJ'] = cp.where(has_vm,
                                     states['MT_VM_PROJ'] * params['MT_DEX_ORIG'][:, cp.newaxis] / mt_vm_orig,
                                     states['MT_DEX_PROJ'])
    states['MT_MM_PROJ'] = cp.where(has_vm,
                                    states['MT_VM_PROJ'] * params['MT_MM_ORIG'][:, cp.newaxis] / mt_vm_orig,
                                    states['MT_MM_PROJ'])


def calculate_acquisition_costs_gpu(states, params, lookups, duree_max10, depot_futur, lapse,
                                    qx, mt_vm_ap_retrait, tx_survie_deb, freq, AJUST):
    """Calculate acquisition costs (vectorized)."""
    # Debug: check actual shape
    print(f"DEBUG calculate_acquisition_costs_gpu:")
    print(f"  states['MT_VM_PROJ'].shape = {states['MT_VM_PROJ'].shape}")
    print(f"  states['MT_VM_PROJ'].ndim = {states['MT_VM_PROJ'].ndim}")

    # Get shape safely
    vm_shape = states['MT_VM_PROJ'].shape
    if len(vm_shape) == 2:
        n_accounts, n_scenarios = vm_shape
    elif len(vm_shape) == 3:
        # If it's somehow 3D, reshape it
        print(f"  WARNING: MT_VM_PROJ is 3D with shape {vm_shape}, reshaping...")
        n_accounts = vm_shape[0]
        n_scenarios = vm_shape[-1]
        states['MT_VM_PROJ'] = states['MT_VM_PROJ'].reshape(n_accounts, n_scenarios)
    else:
        raise ValueError(f"Unexpected MT_VM_PROJ shape: {vm_shape}")

    print(f"  n_accounts={n_accounts}, n_scenarios={n_scenarios}")

    # Check if VM before fees is 0
    no_acq = (states['MT_VM_AV_RETRAIT_FRAIS'] == 0)

    # Lookup acquisition parameters - vectorized
    duree_idx = cp.clip(duree_max10 - 1, 0, 9).astype(cp.int32)
    acqui_id_idx = cp.clip(params['ID_ACQUI'][:, cp.newaxis].repeat(n_scenarios, axis=1), 0, 9).astype(cp.int32)

    # Flatten
    duree_flat = duree_idx.flatten()
    acqui_id_flat = acqui_id_idx.flatten()

    # Lookup all fields
    pc_comm_vente_rf_flat = lookups['acquisition'][duree_flat, acqui_id_flat, 0]
    pc_comm_vente_ac_flat = lookups['acquisition'][duree_flat, acqui_id_flat, 1]
    pc_comm_maint_rf_flat = lookups['acquisition'][duree_flat, acqui_id_flat, 2]
    pc_comm_maint_ac_flat = lookups['acquisition'][duree_flat, acqui_id_flat, 3]
    pc_frais_an_ac_flat = lookups['acquisition'][duree_flat, acqui_id_flat, 4]
    pc_frais_an_rf_flat = lookups['acquisition'][duree_flat, acqui_id_flat, 5]

    # Reshape
    pc_comm_vente_rf = pc_comm_vente_rf_flat.reshape(n_accounts, n_scenarios)
    pc_comm_vente_ac = pc_comm_vente_ac_flat.reshape(n_accounts, n_scenarios)
    pc_comm_maint_rf = pc_comm_maint_rf_flat.reshape(n_accounts, n_scenarios)
    pc_comm_maint_ac = pc_comm_maint_ac_flat.reshape(n_accounts, n_scenarios)
    pc_frais_an_ac = pc_frais_an_ac_flat.reshape(n_accounts, n_scenarios)
    pc_frais_an_rf = pc_frais_an_rf_flat.reshape(n_accounts, n_scenarios)

    # Calculate weighted rates
    mt_vm_orig = params['MT_VM_ORIG'][:, cp.newaxis]
    mt_rf = params['MT_RF'][:, cp.newaxis]
    ajustement_commission = params['AJUSTEMENT_COMMISSION'][:, cp.newaxis]

    has_vm = (mt_vm_orig > 0)

    pc_commission_vente = cp.where(
        has_vm,
        ((pc_comm_vente_ac * (mt_vm_orig - mt_rf) / mt_vm_orig +
          pc_comm_vente_rf * mt_rf / mt_vm_orig) * ajustement_commission),
        0.0
    )

    pc_commission_maintien = cp.where(
        has_vm,
        ((pc_comm_maint_ac * (mt_vm_orig - mt_rf) / mt_vm_orig +
          pc_comm_maint_rf * mt_rf / mt_vm_orig) * ajustement_commission),
        0.0
    )

    pc_frais_an = cp.where(
        has_vm,
        (pc_frais_an_ac * (mt_vm_orig - mt_rf) / mt_vm_orig +
         pc_frais_an_rf * mt_rf / mt_vm_orig),
        0.0
    )

    # Calculate commissions
    comm_vente = -pc_commission_vente * depot_futur * states['TX_SURVIE']
    vp_comm_vente = comm_vente * states['TX_ACTUALISATION']

    # Calculate recovery from lapses
    frais_acquis = pc_frais_an * mt_vm_ap_retrait * lapse * tx_survie_deb * (1.0 - qx)
    vp_frais_acquis = frais_acquis * states['TX_ACTUALISATION']

    # Set to zero where no acquisition
    comm_vente = cp.where(no_acq, 0.0, comm_vente)
    vp_comm_vente = cp.where(no_acq, 0.0, vp_comm_vente)
    frais_acquis = cp.where(no_acq, 0.0, frais_acquis)
    vp_frais_acquis = cp.where(no_acq, 0.0, vp_frais_acquis)

    return comm_vente, vp_comm_vente, frais_acquis, vp_frais_acquis, pc_commission_maintien


def calculate_fees_gpu(states, params, lookups, annee_reelle, tx_survie_deb, freq, AJUST, pc_commission_maintien):
    """Calculate all ongoing fees (vectorized)."""
    n_accounts, n_scenarios = states['MT_VM_PROJ'].shape

    # Lookup fixed fees - vectorized
    prod_idx = cp.clip(params['ID_PRODUIT'], 0, 29)
    year_idx = cp.clip(annee_reelle - lookups['fees_min_year'], 0, lookups['fees'].shape[1] - 1)

    # Expand to scenarios
    prod_idx_expanded = prod_idx[:, cp.newaxis].repeat(n_scenarios, axis=1)
    year_idx_expanded = year_idx[:, cp.newaxis].repeat(n_scenarios, axis=1)

    # Flatten and lookup
    prod_flat = prod_idx_expanded.flatten()
    year_flat = year_idx_expanded.flatten()

    frais_fixes_annual_flat = lookups['fees'][prod_flat, year_flat]
    frais_fixes_annual = frais_fixes_annual_flat.reshape(n_accounts, n_scenarios)

    # Only charge if VM > 0
    has_vm = (states['MT_VM_AV_RETRAIT'] > 0)
    frais_fixes = cp.where(has_vm,
                           -frais_fixes_annual / freq * AJUST * tx_survie_deb,
                           0.0)
    vp_frais_fixes = frais_fixes * states['TX_ACTUALISATION']

    # Management fees (honoraires)
    pc_honoraires_gest = params['PC_HONORAIRES_GEST'][:, cp.newaxis]
    hon_gest = -states['MT_VM_AV_RETRAIT_FRAIS'] * (
            cp.exp(pc_honoraires_gest / freq * AJUST) - 1.0) * tx_survie_deb
    vp_hon_gest = hon_gest * states['TX_ACTUALISATION']

    # Maintenance commission
    comm_maintien = -states['MT_VM_AV_RETRAIT_FRAIS'] * (
            cp.exp(pc_commission_maintien / freq * AJUST) - 1.0) * tx_survie_deb
    vp_comm_maintien = comm_maintien * states['TX_ACTUALISATION']

    # Variable premiums
    pc_rfg = params['PC_RFG'][:, cp.newaxis]
    pc_revenu_fds = params['PC_REVENU_FDS'][:, cp.newaxis]

    primes_variables = (states['MT_VM_AV_RETRAIT_FRAIS'] *
                        cp.exp(-(pc_rfg - pc_revenu_fds) / freq * AJUST) *
                        -(cp.exp(-pc_revenu_fds / freq * AJUST) - 1.0) *
                        tx_survie_deb)
    vp_primes_variables = primes_variables * states['TX_ACTUALISATION']

    return (frais_fixes, vp_frais_fixes, hon_gest, vp_hon_gest,
            comm_maintien, vp_comm_maintien, primes_variables, vp_primes_variables)


def calculate_escap_cushions_gpu(states, params, lookups, age, duree_max10, freq):
    """Calculate ESCAP cushions (fully vectorized with take)."""
    n_accounts, n_scenarios = states['MT_VM_PROJ'].shape

    # DEBUG: Print input shapes
    print(f"DEBUG calculate_escap_cushions_gpu:")
    print(f"  n_accounts={n_accounts}, n_scenarios={n_scenarios}")
    print(f"  age.shape={age.shape}")
    print(f"  duree_max10.shape={duree_max10.shape}")
    print(f"  states['MT_VM_PROJ'].shape={states['MT_VM_PROJ'].shape}")

    # If duree_max10 has wrong shape, fix it
    if duree_max10.shape != (n_accounts, n_scenarios):
        print(f"  WARNING: duree_max10 has wrong shape {duree_max10.shape}, fixing...")
        # Take only the first 2 dimensions if it's 3D
        if duree_max10.ndim == 3:
            duree_max10 = duree_max10[:, 0, :]  # Take first scenario group
        # Or flatten and reshape
        duree_max10 = duree_max10.reshape(n_accounts, n_scenarios)
        print(f"  Fixed duree_max10.shape={duree_max10.shape}")

    # If age has wrong shape, fix it
    if age.shape != (n_accounts, n_scenarios):
        print(f"  WARNING: age has wrong shape {age.shape}, fixing...")
        if age.ndim == 3:
            age = age[:, 0, :]
        age = age.reshape(n_accounts, n_scenarios)
        print(f"  Fixed age.shape={age.shape}")

    # Determine CODE_CAT_PRODUIT - ensure proper broadcasting
    id_produit = params['ID_PRODUIT'][:, cp.newaxis].repeat(n_scenarios, axis=1)  # (n_accounts, n_scenarios)

    print(f"  id_produit.shape={id_produit.shape}")

    code_cat_produit = cp.where(id_produit == 22, 0,
                                cp.where((id_produit >= 12) & (id_produit <= 16), 1,
                                         cp.where((id_produit >= 17) & (id_produit <= 21), 2,
                                                  cp.where(id_produit == 6, 3,
                                                           cp.where((id_produit == 4) | (id_produit == 7), 4,
                                                                    cp.where((id_produit == 5) | (id_produit == 8), 5,
                                                                             cp.where(
                                                                                 (id_produit == 2) | (id_produit == 3),
                                                                                 6, 7)))))))

    print(f"  code_cat_produit.shape={code_cat_produit.shape}")

    # Determine CAT_COUSSIN_1 (based on % fixed income)
    pct_rf = cp.where(states['MT_VM_PROJ'] > 0,
                      (states['MT_DEX_PROJ'] + states['MT_MM_PROJ']) / states['MT_VM_PROJ'],
                      0.0)

    print(f"  pct_rf.shape={pct_rf.shape}")

    cat_coussin_1 = cp.where((code_cat_produit == 0) | (code_cat_produit == 6), 0,
                             cp.where((code_cat_produit == 7) & (pct_rf < 0.5), 4,
                                      cp.where(code_cat_produit == 7, 5,
                                               cp.where(pct_rf < 1 / 3, 1,
                                                        cp.where(pct_rf < 2 / 3, 2, 3)))))

    print(f"  cat_coussin_1.shape={cat_coussin_1.shape}")

    # Calculate VM/VG ratio for CAT_COUSSIN_2
    vm_vg_ratio = calculate_vm_vg_ratio_gpu(states, params)

    print(f"  vm_vg_ratio.shape={vm_vg_ratio.shape}")

    # Now calculate cat_coussin_2 step by step
    cond1 = (code_cat_produit == 7) & (vm_vg_ratio < 0.7)
    print(f"  cond1.shape={cond1.shape}")

    cond2 = (code_cat_produit == 7) & (vm_vg_ratio < 0.9)
    print(f"  cond2.shape={cond2.shape}")

    cond3 = (code_cat_produit == 7)
    print(f"  cond3.shape={cond3.shape}")

    cond4 = (duree_max10 <= 3)
    print(f"  cond4.shape={cond4.shape} (duree_max10 <= 3)")

    cat_coussin_2 = cp.where(cond1, 4,
                             cp.where(cond2, 5,
                                      cp.where(cond3, 6,
                                               cp.where(cond4, 1,
                                                        cp.where(duree_max10 <= 6, 2, 3)))))

    print(f"  cat_coussin_2.shape={cat_coussin_2.shape}")

    # Verify shapes
    assert code_cat_produit.shape == (n_accounts, n_scenarios), f"code_cat_produit shape: {code_cat_produit.shape}"
    assert cat_coussin_1.shape == (n_accounts, n_scenarios), f"cat_coussin_1 shape: {cat_coussin_1.shape}"
    assert cat_coussin_2.shape == (n_accounts, n_scenarios), f"cat_coussin_2 shape: {cat_coussin_2.shape}"

    # Clip indices
    code_idx = cp.clip(code_cat_produit, 0, 7).astype(cp.int32)
    cat1_idx = cp.clip(cat_coussin_1, 0, 5).astype(cp.int32)
    cat2_idx = cp.clip(cat_coussin_2, 0, 6).astype(cp.int32)

    # Initialize result array
    cushion_params = cp.zeros((n_accounts, n_scenarios, 16), dtype=cp.float32)

    # Calculate raveled indices for the lookup table
    # coussins shape: (8, 6, 7, 16)
    dim1, dim2, dim3, dim4 = lookups['coussins'].shape  # (8, 6, 7, 16)

    # Ravel the lookup table once
    coussins_raveled = lookups['coussins'].ravel()

    # Lookup each field
    for field_idx in range(16):
        # Calculate flat index for each (account, scenario) pair
        # Formula: code*6*7*16 + cat1*7*16 + cat2*16 + field
        flat_indices = (code_idx * (dim2 * dim3 * dim4) +
                        cat1_idx * (dim3 * dim4) +
                        cat2_idx * dim4 +
                        field_idx)

        # Verify flat_indices shape
        assert flat_indices.shape == (n_accounts,
                                      n_scenarios), f"flat_indices shape: {flat_indices.shape}, expected: {(n_accounts, n_scenarios)}"

        # Use take to get values (flatten, take, reshape)
        flat_indices_1d = flat_indices.ravel()

        # Verify 1D shape
        assert flat_indices_1d.shape[
                   0] == n_accounts * n_scenarios, f"flat_indices_1d shape: {flat_indices_1d.shape}, expected: {n_accounts * n_scenarios}"

        values_1d = coussins_raveled.take(flat_indices_1d)

        # Verify values shape before reshape
        assert values_1d.shape[
                   0] == n_accounts * n_scenarios, f"values_1d shape: {values_1d.shape}, expected: {n_accounts * n_scenarios}"

        cushion_params[:, :, field_idx] = values_1d.reshape(n_accounts, n_scenarios)

    # For RGS with VM=0, set certain cushions to 0
    is_rgs_zero_vm = (code_cat_produit == 7) & (states['MT_VM_PROJ'] == 0)
    cushion_params[:, :, 3] = cp.where(is_rgs_zero_vm, 0.0, cushion_params[:, :, 3])  # TX_COUSSIN_CREDIT
    cushion_params[:, :, 5] = cp.where(is_rgs_zero_vm, 0.0, cushion_params[:, :, 5])  # TX_COUSSIN_MARCHE
    cushion_params[:, :, 9] = cp.where(is_rgs_zero_vm, 0.0, cushion_params[:, :, 9])  # TX_COUSSIN_DECHEANCE
    cushion_params[:, :, 13] = cp.where(is_rgs_zero_vm, 0.0, cushion_params[:, :, 13])  # TX_COUSSIN_DEPOT

    # Determine age factor
    age_factor = cp.where(age < 80, 1.0,
                          cp.where(age < 90,
                                   cushion_params[:, :, 14],  # FACTEUR_AGE_80
                                   cushion_params[:, :, 15]))  # FACTEUR_AGE_90

    # Calculate base amount
    max_guarantee = cp.maximum(cp.maximum(states['MT_GAR_ECH_PROJ'],
                                          states['MT_GAR_DECES_PROJ'] + states['MT_BONI_DECES_PROJ']),
                               states['MT_SRG_PROJ'])

    # Initialize results dictionary
    cushion_results = {}

    # Calculate each cushion
    cushion_names = ['PASSIF_REDRESSE', 'COUSSIN_CREDIT', 'COUSSIN_MARCHE',
                     'COUSSIN_DEPENSE', 'COUSSIN_DECHEANCE', 'COUSSIN_MORTALITE', 'COUSSIN_DEPOT']

    for i, cushion_name in enumerate(cushion_names):
        base_idx = i * 2
        base_field = cushion_params[:, :, base_idx]  # BASE_*
        tx_field = cushion_params[:, :, base_idx + 1]  # TX_*

        # Determine base amount: if BASE==0 use max_guarantee, else use VM
        base_amount = cp.where(base_field == 0, max_guarantee, states['MT_VM_PROJ'])

        # Calculate cushion
        cushion_amount = tx_field * base_amount * age_factor * states['TX_SURVIE']
        vp_cushion_amount = cushion_amount * states['TX_ACTUALISATION'] / freq

        cushion_results[cushion_name] = cushion_amount
        cushion_results[f'VP_{cushion_name}'] = vp_cushion_amount

    return cushion_results


# =============================================================================
# MAIN PROJECTION LOOP
# =============================================================================

def process_month_gpu(states, params, lookups, scn_indices, year, month, freq=12):
    """
    Process one month for ALL accounts × specified scenarios on GPU.
    This implements the complete actuarial calculation.
    """
    AJUST = 1.0

    n_accounts, n_scenarios = states['MT_VM_PROJ'].shape
    print(f"\n=== Month {month}, Year {year} START ===")
    print(f"Initial MT_VM_PROJ.shape: {states['MT_VM_PROJ'].shape}")

    # Calculate current age and duration
    annee_reelle = params['ANNEE_EVALUATION_INI'] + year
    mois_eval = (month + 1) * 12 // freq

    age = calculate_age_gpu(params, annee_reelle, mois_eval, n_scenarios)
    duree_max10 = calculate_duree_max10_gpu(params, annee_reelle, mois_eval, n_scenarios)

    # Store survival at start of period
    states['TX_SURVIE_DEB'] = states['TX_SURVIE'].copy()
    tx_survie_deb = states['TX_SURVIE_DEB']

    print(f"After age/duree calc MT_VM_PROJ.shape: {states['MT_VM_PROJ'].shape}")

    # === STEP 1: LOOKUP MORTALITY ===
    month_diff = params['MOIS_NAIS'][:, cp.newaxis] - mois_eval
    month_diff = cp.where(month_diff <= 0, month_diff + 12, month_diff)
    age_mort = cp.where(month_diff <= 6, age + 1, age)
    age_mort = cp.minimum(age_mort, 120)

    qx = lookup_mortality_gpu(lookups, params, age_mort, annee_reelle, n_scenarios)
    qx = 1.0 - cp.power(1.0 - qx, (1.0 / freq) * AJUST)

    print(f"After mortality lookup MT_VM_PROJ.shape: {states['MT_VM_PROJ'].shape}")

    # === STEP 2: LOOKUP RETURNS ===
    returns = lookup_returns_gpu(lookups, scn_indices, year, month)

    forward_rate = returns['FORWARD_RATE'][cp.newaxis, :]

    # Adjust forward rate if VM is 0
    forward_rate = cp.where(states['MT_VM_PROJ'] == 0,
                            forward_rate + returns['AJUST_FORWARD_RATE_VM_0'][cp.newaxis, :],
                            forward_rate)

    # === STEP 3: UPDATE DISCOUNT FACTOR ===
    states['TX_ACTUALISATION'] = states['TX_ACTUALISATION'] * cp.exp(-forward_rate * AJUST)

    # === STEP 4: APPLY INVESTMENT RETURNS ===
    states['MT_DEX_PROJ'] *= cp.exp(returns['RENDDEX_AN'][cp.newaxis, :] * AJUST)
    states['MT_MM_PROJ'] *= cp.exp(returns['RENDMM_AN'][cp.newaxis, :] * AJUST)
    states['MT_TSX_PROJ'] *= cp.exp(returns['RENDTSX_AN'][cp.newaxis, :] * AJUST)
    states['MT_SP500_PROJ'] *= cp.exp(returns['RENDSP500_AN'][cp.newaxis, :] * AJUST)
    states['MT_EAFE_PROJ'] *= cp.exp(returns['RENDEAFE_AN'][cp.newaxis, :] * AJUST)

    states['MT_VM_AV_RETRAIT_FRAIS'] = (states['MT_DEX_PROJ'] + states['MT_MM_PROJ'] +
                                        states['MT_TSX_PROJ'] + states['MT_SP500_PROJ'] +
                                        states['MT_EAFE_PROJ'])

    # === STEP 5: CALCULATE LAPSE RATES ===
    vm_vg_ratio = calculate_vm_vg_ratio_gpu(states, params)
    lapse_tot, lapse_part, lapse = calculate_lapse_rates_gpu(states, params, lookups, age,
                                                             duree_max10, freq, AJUST, vm_vg_ratio)

    # === STEP 6: UPDATE SURVIVAL ===
    states['TX_SURVIE'] = states['TX_SURVIE'] * (1.0 - qx) * (1.0 - lapse)

    # === STEP 7: ACCUMULATE DEATH BONUS ===
    accumulate_death_bonus_gpu(states, params, age, freq, AJUST)

    # === STEP 8: APPLY FEES ===
    primes_garanties, vp_primes_garanties = apply_fees_and_update_vm_gpu(states, params, freq, AJUST, tx_survie_deb)

    # === STEP 9: CALCULATE MRV ===
    calculate_mrv_amount_gpu(states, params, age, mois_eval, year, freq)

    # === STEP 10: CALCULATE WITHDRAWALS ===
    retrait = calculate_withdrawal_gpu(states, params, lookups, age, mois_eval, year, freq)

    # === STEP 11: PROCESS DEPOSITS ===
    depot_futur = process_deposits_gpu(states, params, lookups, age, duree_max10, freq)

    # === STEP 12: PROCESS BENEFITS ===
    prest_mrv, vp_prest_mrv = calculate_mrv_benefit_gpu(states, params, retrait,
                                                        states['MT_VM_AV_RETRAIT'], tx_survie_deb)

    mt_vm_ap_retrait = update_guarantees_for_withdrawal_gpu(states, retrait)

    # Update VM after withdrawal and add deposits
    states['MT_VM_PROJ'] = cp.where(mt_vm_ap_retrait > 0,
                                    mt_vm_ap_retrait + depot_futur,
                                    mt_vm_ap_retrait)

    mt_vm_ap_retrait_depot = states['MT_VM_PROJ']

    prest_deces, vp_prest_deces = calculate_death_benefit_gpu(states, params, qx, tx_survie_deb,
                                                              mt_vm_ap_retrait_depot)

    prest_ech, vp_prest_ech = process_maturity_benefit_gpu(states, params, age, annee_reelle,
                                                           mois_eval, mt_vm_ap_retrait, freq)

    # === STEP 13: PROCESS RESETS ===
    update_death_guarantee_adjustments_gpu(states, params, freq)
    process_srg_bcb_resets_gpu(states, params, age, annee_reelle, mois_eval)
    process_death_guarantee_resets_gpu(states, params, age, annee_reelle, mois_eval, freq)
    process_facultative_maturity_reset_gpu(states, params, age, annee_reelle, mois_eval)
    process_death_guarantee_age_change_gpu(states, params, age, mois_eval, freq)

    # === STEP 14: REBALANCE PORTFOLIO ===
    rebalance_portfolio_gpu(states, params)

    # === STEP 15: CALCULATE ACQUISITION COSTS ===
    print(f"Before acquisition costs MT_VM_PROJ.shape: {states['MT_VM_PROJ'].shape}")
    print(f"  MT_VM_AV_RETRAIT_FRAIS.shape: {states['MT_VM_AV_RETRAIT_FRAIS'].shape}")

    comm_vente, vp_comm_vente, frais_acquis, vp_frais_acquis, pc_commission_maintien = \
        calculate_acquisition_costs_gpu(states, params, lookups, duree_max10, depot_futur, lapse,
                                        qx, mt_vm_ap_retrait, tx_survie_deb, freq, AJUST)
    # === STEP 16: CALCULATE FEES ===
    frais_fixes, vp_frais_fixes, hon_gest, vp_hon_gest, comm_maintien, vp_comm_maintien, \
        primes_variables, vp_primes_variables = calculate_fees_gpu(states, params, lookups,
                                                                   annee_reelle, tx_survie_deb,
                                                                   freq, AJUST, pc_commission_maintien)

    # === STEP 17: CALCULATE ESCAP CUSHIONS ===
    cushions = calculate_escap_cushions_gpu(states, params, lookups, age, duree_max10, freq)

    # === STEP 18: CALCULATE TRACKING METRICS ===
    valeur_marchande = states['MT_VM_PROJ'] * states['TX_SURVIE']
    vp_valeur_marchande = valeur_marchande * states['TX_ACTUALISATION'] / freq

    # Return all cash flows
    return {
        'primes_garanties': primes_garanties,
        'vp_primes_garanties': vp_primes_garanties,
        'prest_deces': prest_deces,
        'vp_prest_deces': vp_prest_deces,
        'prest_ech': prest_ech,
        'vp_prest_ech': vp_prest_ech,
        'prest_mrv': prest_mrv,
        'vp_prest_mrv': vp_prest_mrv,
        'frais_acquis': frais_acquis,
        'vp_frais_acquis': vp_frais_acquis,
        'comm_vente': comm_vente,
        'vp_comm_vente': vp_comm_vente,
        'primes_variables': primes_variables,
        'vp_primes_variables': vp_primes_variables,
        'frais_fixes': frais_fixes,
        'vp_frais_fixes': vp_frais_fixes,
        'hon_gest': hon_gest,
        'vp_hon_gest': vp_hon_gest,
        'comm_maintien': comm_maintien,
        'vp_comm_maintien': vp_comm_maintien,
        'valeur_marchande': valeur_marchande,
        'vp_valeur_marchande': vp_valeur_marchande,
        **cushions
    }


def process_year_gpu(states, params, lookups, scn_indices, year, freq=12):
    """Process one full year (12 months) for all accounts × scenarios."""
    # Accumulate cash flows
    year_cashflows = None

    # Process each month
    for month in range(freq):
        month_cf = process_month_gpu(states, params, lookups, scn_indices, year, month, freq)

        if year_cashflows is None:
            year_cashflows = {k: v.copy() for k, v in month_cf.items()}
        else:
            for k, v in month_cf.items():
                year_cashflows[k] += v

    return year_cashflows


def run_projection_gpu(states, params, lookups, n_years=100, n_scenarios=100, freq=12):
    """
    Main projection loop: iterate through years.
    Each year, process all accounts × scenarios in parallel on GPU.
    """
    print("\nStarting GPU projection...")
    print("=" * 60)

    n_accounts = states['MT_VM_PROJ'].shape[0]

    # Process all scenarios
    scn_indices = cp.arange(n_scenarios)

    # Storage for results
    all_cashflows = []

    start_time = datetime.now()

    for year in range(n_years):
        if year % 10 == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            print(f"Processing year {year}/{n_years} (elapsed: {elapsed:.1f}s)")

        # Process this year on GPU
        year_cf = process_year_gpu(states, params, lookups, scn_indices, year, freq)

        # Store results
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
    """Aggregate results from GPU arrays."""
    print("\nAggregating results...")

    n_accounts, n_scenarios = states['MT_VM_PROJ'].shape

    # Aggregate cash flows across scenarios (average)
    print("  - Averaging across scenarios...")

    # Initialize accumulators
    totals = {}
    for key in cashflows_list[0].keys():
        if key != 'year':
            totals[key] = cp.zeros(n_accounts, dtype=cp.float32)

    # Sum across years
    for year_cf in cashflows_list:
        for key, value in year_cf.items():
            if key != 'year':
                totals[key] += cp.mean(value, axis=1)  # Average across scenarios

    # Transfer to CPU
    results_cpu = {k: cp.asnumpy(v) for k, v in totals.items()}

    # Create summary DataFrame
    summary = pd.DataFrame({
        'ID_COMPTE': range(n_accounts),
        **results_cpu
    })

    print("Aggregation complete")
    return summary


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_projection(data_path: str, output_path: str,
                   nb_accounts: int = None,
                   nb_scenarios: int = None,
                   nb_years: int = None):
    """Main function to run GPU-accelerated projection.

    Args:
        data_path: Path to input data directory
        output_path: Path to output directory
        nb_accounts: Number of accounts to process (None = all accounts)
        nb_scenarios: Number of scenarios to run (None = use CONFIG['NB_SC'])
        nb_years: Number of years to project (None = use CONFIG['NB_AN_PROJECTION'])
    """
    start_time = datetime.now()
    print(f"Starting GPU projection at {start_time}")
    print("=" * 60)

    # Use CONFIG defaults if not specified
    if nb_scenarios is None:
        nb_scenarios = CONFIG['NB_SC']
    if nb_years is None:
        nb_years = CONFIG['NB_AN_PROJECTION']

    # Load data (CPU)
    data = load_all_data(Path(data_path))

    # Limit accounts if requested
    if nb_accounts:
        data['population'] = data['population'].head(nb_accounts)

    # Create GPU lookup tables
    lookups = create_gpu_lookups(data)

    # Create GPU state arrays
    states = create_state_arrays(data['population'], nb_scenarios)
    params = create_account_params(data['population'])

    # Run projection on GPU
    cashflows, final_states = run_projection_gpu(
        states,
        params,
        lookups,
        n_years=nb_years,
        n_scenarios=nb_scenarios,
        freq=CONFIG['FREQ_EVAL']
    )

    # Aggregate results
    results = aggregate_results_gpu(cashflows, final_states, params)

    # Save outputs
    print("\nSaving outputs...")
    Path(output_path).mkdir(parents=True, exist_ok=True)
    results.to_csv(f"{output_path}/GPU_RESULTS.csv", index=False, sep=';')
    print(f"  ✓ Saved {output_path}/GPU_RESULTS.csv")

    # Print summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print("\n" + "=" * 60)
    print("PROJECTION COMPLETE")
    print("=" * 60)
    print(f"Processing time: {duration:.2f} seconds ({duration / 60:.2f} minutes)")
    print(f"Accounts processed: {len(data['population'])}")
    print(f"Scenarios: {nb_scenarios}")
    print(f"Years: {nb_years}")
    print(f"Total computations: {len(data['population']) * nb_scenarios * nb_years * 12:,}")
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
        output_path=OUTPUT_PATH,
        nb_accounts=2,  # Number of accounts to process
        nb_scenarios=100,  # Number of scenarios to run
        nb_years=100  # Number of years to project
    )

    print("\nSample Results:")
    print(results.head())
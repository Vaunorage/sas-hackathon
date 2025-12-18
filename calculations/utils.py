import gc
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import pandas as pd
from numba import cuda
from pyarrow import parquet as pq

CUDF_AVAILABLE = False


class AccountIdx:
    """Constants for account data array indexing in GPU kernels."""
    
    # Account identifiers
    ID_COMPTE = 0
    ANNEE_EVALUATION_INI = 1
    MOIS_EVALUATION_INI = 2
    ANNEE_NAIS = 3
    MOIS_NAIS = 4
    
    # Product information
    I_SEXE = 5
    I_PRODUIT_REGR = 6
    ID_PRODUIT = 7
    ID_LAPSE = 8
    I_REGIME_2 = 9
    ID_DEPOT = 10
    ID_ACQUI = 11
    
    # Age thresholds
    AGE_ECH_MIN = 12
    AGE_FIN_CONTRAT = 13
    AGE_DECAISSEMENT = 14
    
    # Financial amounts
    MT_VM = 15
    MT_GAR_DECES = 16
    MT_GAR_ECH = 17
    MT_SRG = 18
    MT_BCB = 19
    
    # Asset allocations
    MT_DEX = 20
    MT_MM = 21
    MT_TSX = 22
    MT_SP500 = 23
    MT_EAFE = 24
    
    # Additional amounts
    MT_BONI_DECES = 25
    MT_MRV_MRG_MRA = 26
    TAUX_MRV_MRG_MRA = 27
    
    # Dates
    ANNEE_ECH = 28
    MOIS_ECH = 29
    
    # Percentage rates
    PC_HONORAIRES_GEST = 30
    PC_FRAIS_GARANTIE = 31
    PC_GAR_DECES_1 = 32
    PC_BONI_DECES = 33
    PC_RFG = 34
    PC_REVENU_FDS = 35
    PC_GAR_ECH = 36
    PC_GAR_ECH_DEP_FUT = 37
    
    # Additional fields
    AJUSTEMENT_COMMISSION = 38
    MT_RF = 39
    MT_VM_ORIG = 40
    ANNEE_COTIS = 41
    MOIS_COTIS = 42
    MAX_BONI_DECES = 43
    I_FRAIS_SUR_SRG = 44

    AJUSTEMENT_MENSUEL_GAR = 45
    PC_GAR_DECES_2 = 46
    AGE_CHANG_DECES = 47
    FREQ_RESET_DECES = 48
    MAX_RESET_DECES = 49
    I_RESET_DECES_ECH = 50
    NB_AN_ECH = 51
    PC_RENOUV_ECH = 52
    AGE_MAX_RENOUV_ECH = 53
    MAX_RESET_FACUL_ECH = 54
    RATIO_VM_VG_RESET_ECH = 55
    AGE_MRV_PERMIS = 56
    PC_BONI_SRG = 57
    FREQ_RESET_SRG = 58
    MAX_RESET_SRG = 59
    TABLE_TAUX_MRV_MRG_MRA = 60
    MT_TPA_RETRAIT = 61
    M_MT_MRV_EXCEDENT = 62
    MT_TPA_DEPOT = 63
    VAR_RETRAIT_FCT = 64
    PC_RETRAIT_AGE = 65
    MT_RETRAIT_MAX = 66
    I_RESET_FACUL_ECH = 67

    # Total number of fields
    TOTAL_FIELDS = 68


def validate_account_data_structure(account_data_columns):
    """Validate that account data matches expected structure."""
    expected_columns = [
        'ID_COMPTE', 'ANNEE_EVALUATION_INI', 'MOIS_EVALUATION_INI',
        'ANNEE_NAIS', 'MOIS_NAIS', 'I_SEXE', 'I_PRODUIT_REGR',
        'ID_PRODUIT', 'ID_LAPSE', 'I_REGIME_2', 'ID_DEPOT', 'ID_ACQUI',
        'AGE_ECH_MIN', 'AGE_FIN_CONTRAT', 'AGE_DECAISSEMENT',
        'MT_VM', 'MT_GAR_DECES', 'MT_GAR_ECH', 'MT_SRG', 'MT_BCB',
        'MT_DEX', 'MT_MM', 'MT_TSX', 'MT_SP500', 'MT_EAFE',
        'MT_BONI_DECES', 'MT_MRV_MRG_MRA', 'TAUX_MRV_MRG_MRA',
        'ANNEE_ECH', 'MOIS_ECH',
        'PC_HONORAIRES_GEST', 'PC_FRAIS_GARANTIE', 'PC_GAR_DECES_1',
        'PC_BONI_DECES', 'PC_RFG', 'PC_REVENU_FDS', 'PC_GAR_ECH',
        'PC_GAR_ECH_DEP_FUT', 'AJUSTEMENT_COMMISSION', 'MT_RF', 'MT_VM',
        'ANNEE_COTIS', 'MOIS_COTIS', 'MAX_BONI_DECES', 'I_FRAIS_SUR_SRG',
        'AJUSTEMENT_MENSUEL_GAR', 'PC_GAR_DECES_2', 'AGE_CHANG_DECES',
        'FREQ_RESET_DECES', 'MAX_RESET_DECES', 'I_RESET_DECES_ECH',
        'NB_AN_ECH', 'PC_RENOUV_ECH', 'AGE_MAX_RENOUV_ECH',
        'MAX_RESET_FACUL_ECH', 'RATIO_VM_VG_RESET_ECH',
        'AGE_MRV_PERMIS', 'PC_BONI_SRG', 'FREQ_RESET_SRG', 'MAX_RESET_SRG',
        'TABLE_TAUX_MRV_MRG_MRA', 'MT_TPA_RETRAIT', 'M_MT_MRV_EXCEDENT',
        'MT_TPA_DEPOT', 'VAR_RETRAIT_FCT', 'PC_RETRAIT_AGE', 'MT_RETRAIT_MAX',
        'I_RESET_FACUL_ECH'
    ]
    
    if len(account_data_columns) != len(expected_columns):
        raise ValueError(f"Expected {len(expected_columns)} columns, got {len(account_data_columns)}")
    
    for i, (actual, expected) in enumerate(zip(account_data_columns, expected_columns)):
        if actual != expected:
            raise ValueError(f"Column mismatch at index {i}: expected '{expected}', got '{actual}'")
    
    print(f"✓ Account data structure validated: {len(expected_columns)} fields")
    return True


def setup_logger(name='gpu_projection', level=logging.INFO):
    """Setup logger with timestamp formatting for debugging."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear any existing handlers
    logger.handlers.clear()

    # Console handler with timestamp
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Format: [HH:MM:SS.mmm] MESSAGE
    formatter = logging.Formatter(
        '[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


logger = setup_logger()
CONFIG = {
    'NBCPT': 9999999,
    'NB_SC': 100,
    'NB_AN_PROJECTION': 100,
    'FREQ_EVAL': 12,
    'NO_COMPTE_SORTIE': 6522,
    'NO_SCN_SORTIE': 2,
}


def parse_percentage(value):
    """Convert percentage string to float (e.g., '1.5%' -> 0.015, '(0.53%)' -> -0.0053)."""
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
    """Clean numeric columns, handling percentage strings."""
    for col in columns:
        if col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].apply(parse_percentage)
    return df


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all column names to uppercase for consistency."""
    df.columns = [str(col).upper() for col in df.columns]
    return df


def _standardize_rendements_columns(df: pd.DataFrame, internal: bool) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df

    def _find_col(pattern: str):
        for c in df.columns:
            if c.endswith(pattern):
                return c
        for c in df.columns:
            if pattern in c:
                return c
        return None

    mapping = {}

    for target in [
        'FORWARD_RATE',
        'AJUST_FORWARD_RATE_VM_0',
        'RENDDEX_AN',
        'RENDMM_AN',
        'RENDTSX_AN',
        'RENDSP500_AN',
        'RENDEAFE_AN',
    ]:
        src = _find_col(target)
        if src is not None and src != target:
            mapping[src] = target

    if internal:
        for target in ['AN_EVAL_INT', 'SCN_EVAL_INT', 'MOIS_EVAL']:
            src = _find_col(target)
            if src is not None and src != target:
                mapping[src] = target
    else:
        for target in ['AN_EVAL', 'SCN_EVAL', 'MOIS_EVAL']:
            src = _find_col(target)
            if src is not None and src != target:
                mapping[src] = target

    if mapping:
        df = df.rename(columns=mapping)

    required_numeric = [
        'FORWARD_RATE',
        'AJUST_FORWARD_RATE_VM_0',
        'RENDDEX_AN',
        'RENDMM_AN',
        'RENDTSX_AN',
        'RENDSP500_AN',
        'RENDEAFE_AN',
    ]
    for col in required_numeric:
        if col not in df.columns:
            df[col] = 0.0

    if internal:
        for col in ['AN_EVAL_INT', 'SCN_EVAL_INT', 'MOIS_EVAL']:
            if col not in df.columns:
                df[col] = 0
    else:
        for col in ['AN_EVAL', 'SCN_EVAL', 'MOIS_EVAL']:
            if col not in df.columns:
                df[col] = 0

    return df


def load_all_data(data_path: Path,
                  population_path: Optional[Path] = None,
                  mortalite_path: Optional[Path] = None,
                  rendements_path: Optional[Path] = None,
                  depots_futurs_path: Optional[Path] = None,
                  frais_admin_path: Optional[Path] = None,
                  min_ferr_path: Optional[Path] = None,
                  tx_lapse_part_path: Optional[Path] = None,
                  tx_lapse_tot_path: Optional[Path] = None,
                  acquisition_path: Optional[Path] = None,
                  coussins_escap_path: Optional[Path] = None,
                  rendements_int_path: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
    """Load all CSV files into memory with semicolon delimiter.

    Args:
        data_path: Default path for CSV files
        population_path: Optional custom path for POPULATION.csv
        mortalite_path: Optional custom path for MORTALITE.csv
        rendements_path: Optional custom path for RENDEMENTS.csv
        depots_futurs_path: Optional custom path for DEPOTS_FUTURS.csv
        frais_admin_path: Optional custom path for FRAIS_ADMIN.csv
        min_ferr_path: Optional custom path for MIN_FERR.csv
        tx_lapse_part_path: Optional custom path for TX_LAPSE_PART.csv
        tx_lapse_tot_path: Optional custom path for TX_LAPSE_TOT.csv
        acquisition_path: Optional custom path for ACQUISITION.csv
        coussins_escap_path: Optional custom path for COUSSINS_ESCAP.csv
    """
    print("Loading data files...")

    data = {}

    # Load all required tables with optional custom paths
    data['population'] = pd.read_csv(
        population_path or data_path.joinpath("POPULATION.csv"), sep=';', encoding='utf-8')
    data['mortalite'] = pd.read_csv(
        mortalite_path or data_path.joinpath("MORTALITE.csv"), sep=';', encoding='utf-8')
    rendements_file = rendements_path or data_path.joinpath("RENDEMENTS.csv")
    data['rendements'] = pd.read_csv(rendements_file, sep=';', encoding='utf-8')
    if len(data['rendements'].columns) == 1 and ',' in str(data['rendements'].columns[0]):
        data['rendements'] = pd.read_csv(rendements_file, sep=',', encoding='utf-8')
    try:
        data['rendements_int'] = pd.read_csv(
            rendements_int_path or data_path.joinpath("RENDEMENTS_INT.csv"), sep=',', encoding='utf-8')
    except FileNotFoundError:
        data['rendements_int'] = pd.DataFrame()
    data['depots_futurs'] = pd.read_csv(
        depots_futurs_path or data_path.joinpath("DEPOTS_FUTURS.csv"), sep=';', encoding='utf-8')
    data['frais_admin'] = pd.read_csv(
        frais_admin_path or data_path.joinpath("FRAIS_ADMIN.csv"), sep=';', encoding='utf-8')
    data['min_ferr'] = pd.read_csv(
        min_ferr_path or data_path.joinpath("MIN_FERR.csv"), sep=';', encoding='utf-8')
    data['tx_lapse_part'] = pd.read_csv(
        tx_lapse_part_path or data_path.joinpath("TX_LAPSE_PART.csv"), sep=';', encoding='utf-8')
    data['tx_lapse_tot'] = pd.read_csv(
        tx_lapse_tot_path or data_path.joinpath("TX_LAPSE_TOT.csv"), sep=';', encoding='utf-8')
    data['acquisition'] = pd.read_csv(
        acquisition_path or data_path.joinpath("ACQUISITION.csv"), sep=';', encoding='utf-8')
    data['coussins_escap'] = pd.read_csv(
        coussins_escap_path or data_path.joinpath("COUSSINS_ESCAP.csv"), sep=';', encoding='utf-8')

    # Normalize all column names
    print("  Normalizing column names...")
    for key in data:
        data[key] = normalize_column_names(data[key])

    data['rendements'] = _standardize_rendements_columns(data['rendements'], internal=False)
    data['rendements_int'] = _standardize_rendements_columns(data['rendements_int'], internal=True)

    # Clean numeric columns
    print("  Loading POPULATION...")
    pct_cols = [col for col in data['population'].columns if col.startswith('PC_') or col.startswith('TAUX_')]
    data['population'] = clean_numeric(data['population'], pct_cols)

    print("  Loading MORTALITE...")
    data['mortalite'] = clean_numeric(data['mortalite'], ['QX'])

    print("  Loading RENDEMENTS...")
    rend_cols = ['FORWARD_RATE', 'AJUST_FORWARD_RATE_VM_0', 'RENDDEX_AN', 'RENDMM_AN',
                 'RENDTSX_AN', 'RENDSP500_AN', 'RENDEAFE_AN']
    data['rendements'] = clean_numeric(data['rendements'], rend_cols)

    print("  Loading RENDEMENTS_INT...")
    data['rendements_int'] = clean_numeric(data['rendements_int'], rend_cols)

    print("  Loading DEPOTS_FUTURS...")
    data['depots_futurs'] = clean_numeric(data['depots_futurs'], ['PC_DEPOT_ANNUEL'])

    print("  Loading FRAIS_ADMIN...")
    data['frais_admin'] = clean_numeric(data['frais_admin'], ['FRAIS'])

    print("  Loading MIN_FERR...")
    data['min_ferr'] = clean_numeric(data['min_ferr'], ['MIN_FERR'])

    print("  Loading TX_LAPSE_PART...")
    lapse_cols = ['TX_LAPSE_PART_MIN', 'TX_LAPSE_PART_MAX']
    data['tx_lapse_part'] = clean_numeric(data['tx_lapse_part'], lapse_cols)

    print("  Loading TX_LAPSE_TOT...")
    for col in ['TX_LAPSE_TOT_MIN', 'TX_LAPSE_TOT_MAX', 'FACT_DIM']:
        if col in data['tx_lapse_tot'].columns:
            data['tx_lapse_tot'][col] = pd.to_numeric(data['tx_lapse_tot'][col], errors='coerce').fillna(0)

    print("  Loading ACQUISITION...")
    acq_cols = ['PC_COMMISSION_VENTE_RF', 'PC_COMMISSION_VENTE_AC',
                'PC_COMMISSION_MAINTIEN_RF', 'PC_COMMISSION_MAINTIEN_AC',
                'PC_FRAIS_AN_AC', 'PC_FRAIS_AN_RF']
    data['acquisition'] = clean_numeric(data['acquisition'], acq_cols)

    print("  Loading COUSSINS_ESCAP...")
    coussin_cols = [col for col in data['coussins_escap'].columns
                    if col.startswith('TX_') or col.startswith('FACTEUR_')]
    data['coussins_escap'] = clean_numeric(data['coussins_escap'], coussin_cols)
    base_cols = [col for col in data['coussins_escap'].columns if col.startswith('BASE_')]
    for col in base_cols:
        data['coussins_escap'][col] = data['coussins_escap'][col].astype(int)

    # Filter data
    if 'SCN_EVAL' in data['rendements'].columns and 'AN_EVAL' in data['rendements'].columns:
        data['rendements'] = data['rendements'][
            (data['rendements']['SCN_EVAL'] <= CONFIG['NB_SC']) &
            (data['rendements']['AN_EVAL'] <= CONFIG['NB_AN_PROJECTION'])
            ]

    data['population'] = data['population'][
        data['population']['ID_COMPTE'] <= CONFIG['NBCPT']
        ]

    print(f"Loaded {len(data['population'])} accounts")
    return data


def prepare_account_data(population_df: pd.DataFrame):
    """Convert account DataFrame to numpy array for GPU using AccountIdx constants."""
    
    # Define the columns to extract in order - MUST match AccountIdx order exactly
    columns = [
        'ID_COMPTE',              # AccountIdx.ID_COMPTE = 0
        'ANNEE_EVALUATION_INI',   # AccountIdx.ANNEE_EVALUATION_INI = 1
        'MOIS_EVALUATION_INI',    # AccountIdx.MOIS_EVALUATION_INI = 2
        'ANNEE_NAIS',             # AccountIdx.ANNEE_NAIS = 3
        'MOIS_NAIS',              # AccountIdx.MOIS_NAIS = 4
        'I_SEXE',                 # AccountIdx.I_SEXE = 5
        'I_PRODUIT_REGR',         # AccountIdx.I_PRODUIT_REGR = 6
        'ID_PRODUIT',             # AccountIdx.ID_PRODUIT = 7
        'ID_LAPSE',               # AccountIdx.ID_LAPSE = 8
        'I_REGIME_2',             # AccountIdx.I_REGIME_2 = 9
        'ID_DEPOT',               # AccountIdx.ID_DEPOT = 10
        'ID_ACQUI',               # AccountIdx.ID_ACQUI = 11
        'AGE_ECH_MIN',            # AccountIdx.AGE_ECH_MIN = 12
        'AGE_FIN_CONTRAT',        # AccountIdx.AGE_FIN_CONTRAT = 13
        'AGE_DECAISSEMENT',       # AccountIdx.AGE_DECAISSEMENT = 14
        'MT_VM',                  # AccountIdx.MT_VM = 15
        'MT_GAR_DECES',           # AccountIdx.MT_GAR_DECES = 16
        'MT_GAR_ECH',             # AccountIdx.MT_GAR_ECH = 17
        'MT_SRG',                 # AccountIdx.MT_SRG = 18
        'MT_BCB',                 # AccountIdx.MT_BCB = 19
        'MT_DEX',                 # AccountIdx.MT_DEX = 20
        'MT_MM',                  # AccountIdx.MT_MM = 21
        'MT_TSX',                 # AccountIdx.MT_TSX = 22
        'MT_SP500',               # AccountIdx.MT_SP500 = 23
        'MT_EAFE',                # AccountIdx.MT_EAFE = 24
        'MT_BONI_DECES',          # AccountIdx.MT_BONI_DECES = 25
        'MT_MRV_MRG_MRA',         # AccountIdx.MT_MRV_MRG_MRA = 26
        'TAUX_MRV_MRG_MRA',       # AccountIdx.TAUX_MRV_MRG_MRA = 27
        'ANNEE_ECH',              # AccountIdx.ANNEE_ECH = 28
        'MOIS_ECH',               # AccountIdx.MOIS_ECH = 29
        'PC_HONORAIRES_GEST',     # AccountIdx.PC_HONORAIRES_GEST = 30
        'PC_FRAIS_GARANTIE',      # AccountIdx.PC_FRAIS_GARANTIE = 31
        'PC_GAR_DECES_1',         # AccountIdx.PC_GAR_DECES_1 = 32
        'PC_BONI_DECES',          # AccountIdx.PC_BONI_DECES = 33
        'PC_RFG',                 # AccountIdx.PC_RFG = 34
        'PC_REVENU_FDS',          # AccountIdx.PC_REVENU_FDS = 35
        'PC_GAR_ECH',             # AccountIdx.PC_GAR_ECH = 36
        'PC_GAR_ECH_DEP_FUT',     # AccountIdx.PC_GAR_ECH_DEP_FUT = 37
        'AJUSTEMENT_COMMISSION',  # AccountIdx.AJUSTEMENT_COMMISSION = 38
        'MT_RF',                  # AccountIdx.MT_RF = 39
        'MT_VM',                  # AccountIdx.MT_VM_ORIG = 40 (duplicate MT_VM for orig)
        'ANNEE_COTIS',            # AccountIdx.ANNEE_COTIS = 41
        'MOIS_COTIS',             # AccountIdx.MOIS_COTIS = 42
        'MAX_BONI_DECES',         # AccountIdx.MAX_BONI_DECES = 43
        'I_FRAIS_SUR_SRG',        # AccountIdx.I_FRAIS_SUR_SRG = 44
        'AJUSTEMENT_MENSUEL_GAR', # AccountIdx.AJUSTEMENT_MENSUEL_GAR = 45
        'PC_GAR_DECES_2',         # AccountIdx.PC_GAR_DECES_2 = 46
        'AGE_CHANG_DECES',        # AccountIdx.AGE_CHANG_DECES = 47
        'FREQ_RESET_DECES',       # AccountIdx.FREQ_RESET_DECES = 48
        'MAX_RESET_DECES',        # AccountIdx.MAX_RESET_DECES = 49
        'I_RESET_DECES_ECH',      # AccountIdx.I_RESET_DECES_ECH = 50
        'NB_AN_ECH',              # AccountIdx.NB_AN_ECH = 51
        'PC_RENOUV_ECH',          # AccountIdx.PC_RENOUV_ECH = 52
        'AGE_MAX_RENOUV_ECH',     # AccountIdx.AGE_MAX_RENOUV_ECH = 53
        'MAX_RESET_FACUL_ECH',    # AccountIdx.MAX_RESET_FACUL_ECH = 54
        'RATIO_VM_VG_RESET_ECH',  # AccountIdx.RATIO_VM_VG_RESET_ECH = 55
        'AGE_MRV_PERMIS',         # AccountIdx.AGE_MRV_PERMIS = 56
        'PC_BONI_SRG',            # AccountIdx.PC_BONI_SRG = 57
        'FREQ_RESET_SRG',         # AccountIdx.FREQ_RESET_SRG = 58
        'MAX_RESET_SRG',          # AccountIdx.MAX_RESET_SRG = 59
        'TABLE_TAUX_MRV_MRG_MRA', # AccountIdx.TABLE_TAUX_MRV_MRG_MRA = 60
        'MT_TPA_RETRAIT',         # AccountIdx.MT_TPA_RETRAIT = 61
        'M_MT_MRV_EXCEDENT',      # AccountIdx.M_MT_MRV_EXCEDENT = 62
        'MT_TPA_DEPOT',           # AccountIdx.MT_TPA_DEPOT = 63
        'VAR_RETRAIT_FCT',        # AccountIdx.VAR_RETRAIT_FCT = 64
        'PC_RETRAIT_AGE',         # AccountIdx.PC_RETRAIT_AGE = 65
        'MT_RETRAIT_MAX',         # AccountIdx.MT_RETRAIT_MAX = 66
        'I_RESET_FACUL_ECH',      # AccountIdx.I_RESET_FACUL_ECH = 67
    ]

    print(f"  Preparing account data with {len(columns)} fields...")
    
    # Validate that we have the correct number of columns
    if len(columns) != AccountIdx.TOTAL_FIELDS:
        raise ValueError(f"Column count mismatch: {len(columns)} vs {AccountIdx.TOTAL_FIELDS}")

    # Fill missing columns with defaults
    for col in columns:
        if col not in population_df.columns:
            if col in ['MT_BCB', 'MT_BONI_DECES', 'MT_MRV_MRG_MRA', 'TAUX_MRV_MRG_MRA',
                       'PC_BONI_DECES', 'PC_REVENU_FDS', 'MT_RF', 'PC_BONI_SRG',
                       'MT_TPA_RETRAIT', 'M_MT_MRV_EXCEDENT', 'MT_TPA_DEPOT']:
                population_df[col] = 0.0
            elif col in ['ANNEE_ECH', 'MAX_BONI_DECES', 'AGE_CHANG_DECES']:
                population_df[col] = 9999
            elif col in ['MOIS_ECH']:
                population_df[col] = 12
            elif col in ['AJUSTEMENT_COMMISSION', 'PC_RENOUV_ECH', 'PC_RETRAIT_AGE', 'RATIO_VM_VG_RESET_ECH']:
                population_df[col] = 1.0
            elif col in ['ANNEE_COTIS']:
                population_df[col] = population_df.get('ANNEE_EVALUATION_INI', 2020)
            elif col in ['MOIS_COTIS', 'I_FRAIS_SUR_SRG', 'I_RESET_DECES_ECH', 'I_RESET_FACUL_ECH']:
                population_df[col] = 0
            elif col in ['AJUSTEMENT_MENSUEL_GAR']:
                population_df[col] = 0.0
            elif col in ['PC_GAR_DECES_2']:
                population_df[col] = population_df.get('PC_GAR_DECES_1', 0.0)
            elif col in ['FREQ_RESET_DECES']:
                population_df[col] = 3
            elif col in ['MAX_RESET_DECES']:
                population_df[col] = 80
            elif col in ['NB_AN_ECH']:
                population_df[col] = 10
            elif col in ['PC_RENOUV_ECH']:
                population_df[col] = 1.0
            elif col in ['AGE_MAX_RENOUV_ECH']:
                population_df[col] = 999
            elif col in ['MAX_RESET_FACUL_ECH']:
                population_df[col] = 80
            elif col in ['AGE_MRV_PERMIS']:
                population_df[col] = 65
            elif col in ['FREQ_RESET_SRG']:
                population_df[col] = 3
            elif col in ['MAX_RESET_SRG']:
                population_df[col] = 80
            elif col in ['TABLE_TAUX_MRV_MRG_MRA']:
                population_df[col] = 1
            elif col in ['VAR_RETRAIT_FCT']:
                population_df[col] = 1
            elif col in ['MT_RETRAIT_MAX']:
                population_df[col] = 999999999.0

    # Validate the final column structure
    validate_account_data_structure(columns)

    def _coerce_numeric_series(s: pd.Series) -> pd.Series:
        if s.dtype == object:
            s2 = s.astype(str)
            s2 = s2.str.replace('\u00a0', '', regex=False)
            s2 = s2.str.replace(' ', '', regex=False)
            s2 = s2.str.replace("'", '', regex=False)
            has_dot = s2.str.contains('\.', regex=True)
            comma_count = s2.str.count(',', regex=False)

            both = has_dot & (comma_count > 0)
            if both.any():
                s2.loc[both] = s2.loc[both].str.replace(',', '', regex=False)

            multi_comma = comma_count > 1
            if multi_comma.any():
                s2.loc[multi_comma] = s2.loc[multi_comma].str.replace(',', '', regex=False)

            single_comma = comma_count == 1
            if single_comma.any():
                left = s2.loc[single_comma].str.split(',', n=1).str[0]
                right = s2.loc[single_comma].str.split(',', n=1).str[1]
                is_thousands = (right.str.len() == 3) & (left.str.len() > 3)
                if is_thousands.any():
                    idx = s2.loc[single_comma].index[is_thousands]
                    s2.loc[idx] = s2.loc[idx].str.replace(',', '', regex=False)
                is_decimal = ~is_thousands
                if is_decimal.any():
                    idx = s2.loc[single_comma].index[is_decimal]
                    s2.loc[idx] = s2.loc[idx].str.replace(',', '.', regex=False)

            return pd.to_numeric(s2, errors='coerce')
        return pd.to_numeric(s, errors='coerce')

    for col in columns:
        population_df[col] = _coerce_numeric_series(population_df[col])
    population_df['ID_COMPTE'] = _coerce_numeric_series(population_df['ID_COMPTE'])

    population_df[columns] = population_df[columns].fillna(0.0)
    population_df['ID_COMPTE'] = population_df['ID_COMPTE'].fillna(0.0)

    account_data = population_df[columns].values.astype(np.float32)
    account_ids = population_df['ID_COMPTE'].values.astype(np.int32)
    
    print(f"  ✓ Account data prepared: {account_data.shape[0]} accounts × {account_data.shape[1]} fields")
    print(f"  ✓ Field validation: AccountIdx constants match data structure")

    return account_data, account_ids


def log_gpu_memory_debug(step_name: str, verbose: bool = False):
    """Helper function to log GPU memory state for debugging"""
    try:
        free_mem, total_mem = cuda.current_context().get_memory_info()
        used_mem = total_mem - free_mem
        used_pct = (used_mem / total_mem) * 100

        msg = (f"[GPU Memory @ {step_name}] "
               f"Used: {used_mem / 1024**3:.2f} GB / {total_mem / 1024**3:.2f} GB "
               f"({used_pct:.1f}%), Free: {free_mem / 1024**3:.2f} GB")

        if verbose or used_pct > 80:
            if used_pct > 90:
                print(f"⚠️  WARNING: {msg}")
            elif used_pct > 80:
                print(f"⚡ CAUTION: {msg}")
            else:
                print(f"✓  {msg}")

        return free_mem, total_mem, used_mem
    except NotImplementedError:
        if verbose:
            print(f"[GPU Memory @ {step_name}] Memory query not available (using RMM allocator)")
            print(f"  → Use 'nvidia-smi' in another terminal to monitor GPU memory")
        return None, None, None
    except Exception as e:
        if verbose:
            error_type = type(e).__name__
            error_msg = str(e) if str(e) else "Unknown error"
            print(f"[GPU Memory @ {step_name}] Cannot query memory")
            print(f"  → Error type: {error_type}")
            print(f"  → Error message: {error_msg}")
            print(f"  → Use 'nvidia-smi' to monitor GPU memory externally")
        return None, None, None


def force_gpu_memory_cleanup(aggressive=False):
    """Force GPU memory cleanup by running garbage collection

    Args:
        aggressive: If True, performs full generational GC and RMM pool reset
    """
    if aggressive:
        gc.collect(2)  # Full collection (generation 2)
        # Try to reset RMM pool to defragment GPU memory
        if CUDF_AVAILABLE:
            try:
                import rmm
                # Get current pool statistics
                mr = rmm.mr.get_current_device_resource()
                if hasattr(mr, 'release'):
                    mr.release()  # Release unused memory back to OS
            except:
                pass
    else:
        gc.collect()


def write_parquet_async(arrow_table, parquet_path, batch_num, num_rows):
    """Write parquet file asynchronously with optimized settings.

    Args:
        arrow_table: PyArrow table to write
        parquet_path: Path to output file
        batch_num: Batch number for logging
        num_rows: Number of rows for logging

    Returns:
        Tuple of (batch_num, file_size_mb, num_rows, write_time)
    """
    write_start = datetime.now()
    pq.write_table(
        arrow_table,
        parquet_path,
        compression='lz4',  # Faster than snappy
        use_dictionary=False,
        write_statistics=False
    )
    write_time = (datetime.now() - write_start).total_seconds()
    file_size_mb = parquet_path.stat().st_size / 1024**2
    return batch_num, file_size_mb, num_rows, write_time


def write_parquet_async_pandas(df, parquet_path, batch_num, num_rows):
    """Write parquet file from pandas DataFrame asynchronously.

    Args:
        df: Pandas DataFrame to write
        parquet_path: Path to output file
        batch_num: Batch number for logging
        num_rows: Number of rows for logging

    Returns:
        Tuple of (batch_num, file_size_mb, num_rows, write_time)
    """
    write_start = datetime.now()
    df.to_parquet(
        parquet_path,
        engine='pyarrow',
        compression='snappy',
        index=False
    )
    write_time = (datetime.now() - write_start).total_seconds()
    file_size_mb = parquet_path.stat().st_size / 1024**2
    return batch_num, file_size_mb, num_rows, write_time


def write_parquet_async_cudf(gpu_df, parquet_path, batch_num, num_rows):
    """Write parquet file from cuDF DataFrame asynchronously.

    Args:
        gpu_df: cuDF GPU DataFrame to write
        parquet_path: Path to output file
        batch_num: Batch number for logging
        num_rows: Number of rows for logging

    Returns:
        Tuple of (batch_num, file_size_mb, num_rows, write_time)
    """
    write_start = datetime.now()
    gpu_df.to_parquet(
        parquet_path,
        compression='snappy',
        index=False
    )
    write_time = (datetime.now() - write_start).total_seconds()
    file_size_mb = parquet_path.stat().st_size / 1024**2
    return batch_num, file_size_mb, num_rows, write_time

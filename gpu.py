import os
# Set environment variables BEFORE importing numba
os.environ['NUMBA_CUDA_ENABLE_PYNVJITLINK'] = '1'

import pandas as pd
import numpy as np
import gc
import psutil
import logging
from pathlib import Path
from typing import Dict, Tuple, Any, Optional
from datetime import datetime
import math
import duckdb
import tempfile
import pyarrow as pa
import pyarrow.parquet as pq

# Try to import cuDF for GPU-accelerated DataFrame operations
try:
    import cudf
    import cupy as cp
    CUDF_AVAILABLE = False  # DISABLED - cuDF causes GPU memory leaks and slow DataFrame creation
    print("⚠ CuDF disabled - using optimized pandas (CPU) path for stability")
    print("   (cuDF caused 86s DataFrame creation vs 2s with pandas)")
except ImportError:
    CUDF_AVAILABLE = False
    print("⚠ CuDF not available - falling back to pandas (CPU). Install with: pip install cudf-cu12")

# GPU memory cleanup helper
def force_gpu_memory_cleanup():
    """Force GPU memory cleanup by running garbage collection"""
    gc.collect()

# Async parquet writer (Arrow version - kept for compatibility)
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

# Async parquet writer (pandas version - MUCH faster for NumPy arrays)
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
        compression='lz4',
        index=False
    )
    write_time = (datetime.now() - write_start).total_seconds()
    file_size_mb = parquet_path.stat().st_size / 1024**2
    return batch_num, file_size_mb, num_rows, write_time

# Async parquet writer (cuDF version - GPU-accelerated, FASTEST!)
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
        compression='lz4',
        index=False
    )
    write_time = (datetime.now() - write_start).total_seconds()
    file_size_mb = parquet_path.stat().st_size / 1024**2
    return batch_num, file_size_mb, num_rows, write_time

# Import numba for CUDA
from numba import cuda
from paths import HERE
import argparse
from multiprocessing import Pool, cpu_count
import concurrent.futures

# =============================================================================
# LOGGING SETUP
# =============================================================================

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

# Initialize global logger
logger = setup_logger()

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
# UTILITY FUNCTIONS (CPU)
# =============================================================================

def _extract_valid_rows_chunk(args):
    """
    Worker function for parallel extraction of valid rows from a chunk.
    This runs in a separate process to utilize multiple CPU cores.
    
    Args:
        args: Tuple of (chunk_data, chunk_start_idx)
    
    Returns:
        Tuple of (valid_data, chunk_start_idx) or None if no valid rows
    """
    chunk_data, chunk_start_idx = args
    
    # Create boolean mask for this chunk
    valid_mask = chunk_data[:, 0] > 0
    n_valid = np.sum(valid_mask)
    
    if n_valid == 0:
        return None
    
    # Extract valid rows
    valid_data = np.compress(valid_mask, chunk_data, axis=0)
    return valid_data


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
    df.columns = df.columns.str.upper()
    return df


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

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
                  coussins_escap_path: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
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
    data['rendements'] = pd.read_csv(
        rendements_path or data_path.joinpath("RENDEMENTS.csv"), sep=';', encoding='utf-8')
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

def create_gpu_mortality_lookup(df: pd.DataFrame):
    """Create flattened array for mortality lookup on GPU."""
    # Create a 4D array indexed by: [i_sexe, age, year, i_produit_regr]
    max_sexe = 2
    max_age = 121
    max_year = df['ANNEE_REELLE'].max() + 1
    max_produit = df['I_PRODUIT_REGR'].max() + 1

    # Initialize with default value
    lookup = np.full((max_sexe, max_age, max_year, max_produit), 0.001, dtype=np.float32)

    for _, row in df.iterrows():
        i_sexe = int(row['I_SEXE'])
        age = int(row['AGE_MORTALITE'])
        year = int(row['ANNEE_REELLE'])
        i_produit = int(row['I_PRODUIT_REGR'])
        lookup[i_sexe, age, year, i_produit] = float(row['QX'])

    return lookup


def create_gpu_returns_lookup(df: pd.DataFrame):
    """Create flattened arrays for returns lookup on GPU."""
    max_scn = df['SCN_EVAL'].max() + 1
    max_an = df['AN_EVAL'].max() + 1
    max_mois = df['MOIS_EVAL'].max() + 1

    # Create separate arrays for each return type
    forward_rate = np.zeros((max_scn, max_an, max_mois), dtype=np.float32)
    ajust_forward = np.zeros((max_scn, max_an, max_mois), dtype=np.float32)
    rend_dex = np.zeros((max_scn, max_an, max_mois), dtype=np.float32)
    rend_mm = np.zeros((max_scn, max_an, max_mois), dtype=np.float32)
    rend_tsx = np.zeros((max_scn, max_an, max_mois), dtype=np.float32)
    rend_sp500 = np.zeros((max_scn, max_an, max_mois), dtype=np.float32)
    rend_eafe = np.zeros((max_scn, max_an, max_mois), dtype=np.float32)

    for _, row in df.iterrows():
        scn = int(row['SCN_EVAL'])
        an = int(row['AN_EVAL'])
        mois = int(row['MOIS_EVAL'])
        forward_rate[scn, an, mois] = float(row['FORWARD_RATE'])
        ajust_forward[scn, an, mois] = float(row['AJUST_FORWARD_RATE_VM_0'])
        rend_dex[scn, an, mois] = float(row['RENDDEX_AN'])
        rend_mm[scn, an, mois] = float(row['RENDMM_AN'])
        rend_tsx[scn, an, mois] = float(row['RENDTSX_AN'])
        rend_sp500[scn, an, mois] = float(row['RENDSP500_AN'])
        rend_eafe[scn, an, mois] = float(row['RENDEAFE_AN'])

    return (forward_rate, ajust_forward, rend_dex, rend_mm, rend_tsx, rend_sp500, rend_eafe)


def create_gpu_min_ferr_lookup(df: pd.DataFrame):
    """Create array for minimum FERR lookup."""
    max_age = 121
    lookup = np.zeros(max_age, dtype=np.float32)
    for _, row in df.iterrows():
        age = int(row['AGE'])
        lookup[age] = float(row['MIN_FERR'])
    return lookup


def create_gpu_lapse_part_lookup(df: pd.DataFrame):
    """Create arrays for partial lapse lookup."""
    max_age = 121
    max_id_lapse = df['ID_LAPSE'].max() + 1
    max_regime = df['I_REGIME_2'].max() + 1
    max_niv = 4

    tx_min = np.zeros((max_age, max_id_lapse, max_regime, max_niv), dtype=np.float32)
    tx_max = np.zeros((max_age, max_id_lapse, max_regime, max_niv), dtype=np.float32)

    for _, row in df.iterrows():
        age = int(row['AGE'])
        id_lapse = int(row['ID_LAPSE'])
        regime = int(row['I_REGIME_2'])
        niv = int(row['LAPSE_NIV_PART'])
        tx_min[age, id_lapse, regime, niv] = float(row['TX_LAPSE_PART_MIN'])
        tx_max[age, id_lapse, regime, niv] = float(row['TX_LAPSE_PART_MAX'])

    return tx_min, tx_max


def create_gpu_lapse_tot_lookup(df: pd.DataFrame):
    """Create arrays for total lapse lookup."""
    max_duree = 11
    max_id_lapse = df['ID_LAPSE'].max() + 1
    max_niv = 4

    tx_min = np.zeros((max_duree, max_id_lapse, max_niv), dtype=np.float32)
    tx_max = np.zeros((max_duree, max_id_lapse, max_niv), dtype=np.float32)
    fact_dim = np.ones((max_duree, max_id_lapse, max_niv), dtype=np.float32)

    for _, row in df.iterrows():
        duree = int(row['DUREE_MAX10'])
        id_lapse = int(row['ID_LAPSE'])
        niv = int(row['LAPSE_NIV_TOT'])
        tx_min[duree, id_lapse, niv] = float(row['TX_LAPSE_TOT_MIN'])
        tx_max[duree, id_lapse, niv] = float(row['TX_LAPSE_TOT_MAX'])
        fact_dim[duree, id_lapse, niv] = float(row['FACT_DIM'])

    return tx_min, tx_max, fact_dim


def create_gpu_fees_lookup(df: pd.DataFrame):
    """Create array for fees lookup."""
    max_produit = df['ID_PRODUIT'].max() + 1
    max_year = df['ANNEE_REELLE'].max() + 1

    lookup = np.zeros((max_produit, max_year), dtype=np.float32)
    for _, row in df.iterrows():
        produit = int(row['ID_PRODUIT'])
        year = int(row['ANNEE_REELLE'])
        lookup[produit, year] = float(row['FRAIS'])

    return lookup


def create_gpu_deposits_lookup(df: pd.DataFrame):
    """Create arrays for deposits lookup."""
    max_duree = 11
    max_id_depot = df['ID_DEPOT'].max() + 1

    pc_depot = np.zeros((max_duree, max_id_depot), dtype=np.float32)
    var_fct = np.zeros((max_duree, max_id_depot), dtype=np.int32)
    age_max = np.full((max_duree, max_id_depot), 999, dtype=np.int32)
    i_even = np.zeros((max_duree, max_id_depot), dtype=np.int32)

    for _, row in df.iterrows():
        duree = int(row['DUREE_MAX10'])
        id_depot = int(row['ID_DEPOT'])
        pc_depot[duree, id_depot] = float(row['PC_DEPOT_ANNUEL'])
        var_fct[duree, id_depot] = int(row['VAR_DEPOT_FCT'])
        age_max[duree, id_depot] = int(row['AGE_MAX_DEPOT'])
        i_even[duree, id_depot] = int(row['I_EVEN_CESSE_DEPOT'])

    return pc_depot, var_fct, age_max, i_even


def create_gpu_acquisition_lookup(df: pd.DataFrame):
    """Create arrays for acquisition lookup."""
    max_duree = 11
    max_id_acqui = df['ID_ACQUI'].max() + 1

    pc_vente_rf = np.zeros((max_duree, max_id_acqui), dtype=np.float32)
    pc_vente_ac = np.zeros((max_duree, max_id_acqui), dtype=np.float32)
    pc_maintien_rf = np.zeros((max_duree, max_id_acqui), dtype=np.float32)
    pc_maintien_ac = np.zeros((max_duree, max_id_acqui), dtype=np.float32)
    pc_frais_ac = np.zeros((max_duree, max_id_acqui), dtype=np.float32)
    pc_frais_rf = np.zeros((max_duree, max_id_acqui), dtype=np.float32)

    for _, row in df.iterrows():
        duree = int(row['DUREE_MAX10'])
        id_acqui = int(row['ID_ACQUI'])
        pc_vente_rf[duree, id_acqui] = float(row['PC_COMMISSION_VENTE_RF'])
        pc_vente_ac[duree, id_acqui] = float(row['PC_COMMISSION_VENTE_AC'])
        pc_maintien_rf[duree, id_acqui] = float(row['PC_COMMISSION_MAINTIEN_RF'])
        pc_maintien_ac[duree, id_acqui] = float(row['PC_COMMISSION_MAINTIEN_AC'])
        pc_frais_ac[duree, id_acqui] = float(row['PC_FRAIS_AN_AC'])
        pc_frais_rf[duree, id_acqui] = float(row['PC_FRAIS_AN_RF'])

    return pc_vente_rf, pc_vente_ac, pc_maintien_rf, pc_maintien_ac, pc_frais_ac, pc_frais_rf


def create_gpu_coussins_lookup(df: pd.DataFrame):
    """Create arrays for cushions lookup."""
    max_cat_prod = 8
    max_cat1 = 6
    max_cat2 = 7

    # Create arrays for all cushion parameters
    base_passif = np.zeros((max_cat_prod, max_cat1, max_cat2), dtype=np.int32)
    tx_passif = np.zeros((max_cat_prod, max_cat1, max_cat2), dtype=np.float32)
    base_credit = np.zeros((max_cat_prod, max_cat1, max_cat2), dtype=np.int32)
    tx_credit = np.zeros((max_cat_prod, max_cat1, max_cat2), dtype=np.float32)
    base_marche = np.zeros((max_cat_prod, max_cat1, max_cat2), dtype=np.int32)
    tx_marche = np.zeros((max_cat_prod, max_cat1, max_cat2), dtype=np.float32)
    base_depense = np.zeros((max_cat_prod, max_cat1, max_cat2), dtype=np.int32)
    tx_depense = np.zeros((max_cat_prod, max_cat1, max_cat2), dtype=np.float32)
    base_decheance = np.zeros((max_cat_prod, max_cat1, max_cat2), dtype=np.int32)
    tx_decheance = np.zeros((max_cat_prod, max_cat1, max_cat2), dtype=np.float32)
    base_mortalite = np.zeros((max_cat_prod, max_cat1, max_cat2), dtype=np.int32)
    tx_mortalite = np.zeros((max_cat_prod, max_cat1, max_cat2), dtype=np.float32)
    base_depot = np.zeros((max_cat_prod, max_cat1, max_cat2), dtype=np.int32)
    tx_depot = np.zeros((max_cat_prod, max_cat1, max_cat2), dtype=np.float32)
    facteur_80 = np.ones((max_cat_prod, max_cat1, max_cat2), dtype=np.float32)
    facteur_90 = np.ones((max_cat_prod, max_cat1, max_cat2), dtype=np.float32)

    for _, row in df.iterrows():
        cp = int(row['CODE_CAT_PRODUIT'])
        c1 = int(row['CAT_COUSSIN_1'])
        c2 = int(row['CAT_COUSSIN_2'])

        base_passif[cp, c1, c2] = int(row['BASE_PASSIF_REDRESSE'])
        tx_passif[cp, c1, c2] = float(row['TX_PASSIF_REDRESSE'])
        base_credit[cp, c1, c2] = int(row['BASE_COUSSIN_CREDIT'])
        tx_credit[cp, c1, c2] = float(row['TX_COUSSIN_CREDIT'])
        base_marche[cp, c1, c2] = int(row['BASE_COUSSIN_MARCHE'])
        tx_marche[cp, c1, c2] = float(row['TX_COUSSIN_MARCHE'])
        base_depense[cp, c1, c2] = int(row['BASE_COUSSIN_DEPENSE'])
        tx_depense[cp, c1, c2] = float(row['TX_COUSSIN_DEPENSE'])
        base_decheance[cp, c1, c2] = int(row['BASE_COUSSIN_DECHEANCE'])
        tx_decheance[cp, c1, c2] = float(row['TX_COUSSIN_DECHEANCE'])
        base_mortalite[cp, c1, c2] = int(row['BASE_COUSSIN_MORTALITE'])
        tx_mortalite[cp, c1, c2] = float(row['TX_COUSSIN_MORTALITE'])
        base_depot[cp, c1, c2] = int(row['BASE_COUSSIN_DEPOT'])
        tx_depot[cp, c1, c2] = float(row['TX_COUSSIN_DEPOT'])
        facteur_80[cp, c1, c2] = float(row['FACTEUR_AGE_80'])
        facteur_90[cp, c1, c2] = float(row['FACTEUR_AGE_90'])

    return (base_passif, tx_passif, base_credit, tx_credit, base_marche, tx_marche,
            base_depense, tx_depense, base_decheance, tx_decheance, base_mortalite, tx_mortalite,
            base_depot, tx_depot, facteur_80, facteur_90)


# =============================================================================
# GPU KERNEL - MAIN PROJECTION ENGINE
# =============================================================================

@cuda.jit
def projection_kernel(
        # Account data
        account_data,  # Shape: (n_accounts, n_account_fields)
        # Scenario parameters
        n_scenarios,
        n_years,
        freq_eval,
        # Lookup tables - Mortality
        mortality_lookup,
        # Lookup tables - Returns
        forward_rate, ajust_forward, rend_dex, rend_mm, rend_tsx, rend_sp500, rend_eafe,
        # Lookup tables - Lapse
        min_ferr_lookup,
        lapse_part_min, lapse_part_max,
        lapse_tot_min, lapse_tot_max, lapse_tot_fact,
        # Lookup tables - Deposits
        deposits_pc, deposits_var, deposits_age_max, deposits_i_even,
        # Lookup tables - Fees
        fees_lookup,
        # Lookup tables - Acquisition
        acq_vente_rf, acq_vente_ac, acq_maintien_rf, acq_maintien_ac, acq_frais_ac, acq_frais_rf,
        # Lookup tables - Cushions
        cous_base_passif, cous_tx_passif, cous_base_credit, cous_tx_credit,
        cous_base_marche, cous_tx_marche, cous_base_depense, cous_tx_depense,
        cous_base_decheance, cous_tx_decheance, cous_base_mortalite, cous_tx_mortalite,
        cous_base_depot, cous_tx_depot, cous_facteur_80, cous_facteur_90,
        # Output arrays
        output_results  # Shape: (n_accounts, n_scenarios, max_timesteps, n_output_fields)
):
    """
    Main CUDA kernel - processes one account-scenario combination per thread.
    Each thread loops through all timesteps sequentially (state dependency).
    """
    # Get global thread ID
    account_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    scenario_idx = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # Boundary check
    if account_idx >= account_data.shape[0] or scenario_idx >= n_scenarios:
        return

    # Load account data into registers (avoiding repeated global memory access)
    acc = account_data[account_idx]

    # Account static data indices (must match the order in account_data array)
    ID_COMPTE = int(acc[0])
    ANNEE_EVALUATION_INI = int(acc[1])
    MOIS_EVALUATION_INI = int(acc[2])
    ANNEE_NAIS = int(acc[3])
    MOIS_NAIS = int(acc[4])
    I_SEXE = int(acc[5])
    I_PRODUIT_REGR = int(acc[6])
    ID_PRODUIT = int(acc[7])
    ID_LAPSE = int(acc[8])
    I_REGIME_2 = int(acc[9])
    ID_DEPOT = int(acc[10])
    ID_ACQUI = int(acc[11])
    AGE_ECH_MIN = int(acc[12])
    AGE_FIN_CONTRAT = int(acc[13])
    AGE_DECAISSEMENT = int(acc[14])

    # Initialize state variables
    MT_VM_PROJ = acc[15]
    MT_GAR_DECES_PROJ = acc[16]
    MT_GAR_ECH_PROJ = acc[17]
    MT_SRG_PROJ = acc[18]
    MT_BCB_PROJ = acc[19]
    MT_DEX_PROJ = acc[20]
    MT_MM_PROJ = acc[21]
    MT_TSX_PROJ = acc[22]
    MT_SP500_PROJ = acc[23]
    MT_EAFE_PROJ = acc[24]
    MT_BONI_DECES_PROJ = acc[25]
    MT_MRV_MRG_MRA_PROJ = acc[26]
    TAUX_MRV_MRG_MRA_PROJ = acc[27]
    MT_MIN_FERR_PROJ = 0.0
    ANNEE_ECH_PROJ = int(acc[28])
    MOIS_ECH_PROJ = int(acc[29])

    TX_SURVIE = 1.0
    TX_ACTUALISATION = 1.0

    PC_HONORAIRES_GEST = acc[30]
    PC_FRAIS_GARANTIE = acc[31]
    PC_GAR_DECES_1 = acc[32]
    PC_BONI_DECES = acc[33]
    PC_RFG = acc[34]
    PC_REVENU_FDS = acc[35]
    PC_GAR_ECH = acc[36]
    PC_GAR_ECH_DEP_FUT = acc[37]
    AJUSTEMENT_COMMISSION = acc[38]
    MT_RF = acc[39]
    MT_VM_ORIG = acc[40]

    # Additional parameters
    ANNEE_COTIS = int(acc[41]) if acc[41] > 0 else ANNEE_EVALUATION_INI
    MOIS_COTIS = int(acc[42]) if acc[42] > 0 else 1

    # Scenario-specific processing
    scn_eval = scenario_idx + 1  # Scenarios are 1-indexed

    output_idx = 0
    AJUST_NOUV_AFFAIRES = 1.0

    # Loop through years
    for an_eval in range(0, n_years + 1):
        # Loop through months within the year
        for mois_simul in range(1, freq_eval + 1):
            # Calculate real year and month
            annee_reelle = ANNEE_EVALUATION_INI + an_eval - 1
            mois_eval = mois_simul * 12 // freq_eval

            # Calculate age
            age = annee_reelle - ANNEE_NAIS
            if mois_eval < MOIS_NAIS:
                age -= 1
            if age < 1:
                age = 1

            # Check if we should keep this period
            keep = (age <= AGE_FIN_CONTRAT and
                    (an_eval > 1 or
                     (an_eval == 1 and mois_eval >= MOIS_EVALUATION_INI) or
                     (an_eval == 0 and mois_eval == 12)))

            if not keep:
                continue

            # Check if policy is still active
            if TX_SURVIE == 0 or (MT_VM_PROJ == 0 and I_PRODUIT_REGR == 0):
                break

            # Calculate duration from issue date
            current_date = annee_reelle + mois_eval / 12.0
            issue_date = ANNEE_COTIS + MOIS_COTIS / 12.0
            duree = int(current_date - issue_date) + 1
            duree_max10 = min(duree, 10)

            TX_SURVIE_DEB = TX_SURVIE

            # ============= STEP 1: LOOKUP MORTALITY =============
            month_diff = MOIS_NAIS - mois_eval
            if month_diff <= 0:
                month_diff += 12

            if month_diff <= 6:
                age_mort = age + 1
            else:
                age_mort = age

            age_mort = min(age_mort, 120)

            # Boundary checks for mortality lookup
            if (I_SEXE < mortality_lookup.shape[0] and
                    age_mort < mortality_lookup.shape[1] and
                    annee_reelle < mortality_lookup.shape[2] and
                    I_PRODUIT_REGR < mortality_lookup.shape[3]):
                qx = mortality_lookup[I_SEXE, age_mort, annee_reelle, I_PRODUIT_REGR]
            else:
                qx = 0.001

            # Convert annual to period rate
            qx = 1.0 - math.pow(1.0 - qx, (1.0 / freq_eval * AJUST_NOUV_AFFAIRES))

            # ============= STEP 2: LOOKUP RETURNS =============
            if (scn_eval < forward_rate.shape[0] and
                    an_eval < forward_rate.shape[1] and
                    mois_eval < forward_rate.shape[2]):
                fwd_rate = forward_rate[scn_eval, an_eval, mois_eval]
                ajust_fwd = ajust_forward[scn_eval, an_eval, mois_eval]
                r_dex = rend_dex[scn_eval, an_eval, mois_eval]
                r_mm = rend_mm[scn_eval, an_eval, mois_eval]
                r_tsx = rend_tsx[scn_eval, an_eval, mois_eval]
                r_sp500 = rend_sp500[scn_eval, an_eval, mois_eval]
                r_eafe = rend_eafe[scn_eval, an_eval, mois_eval]
            else:
                fwd_rate = 0.0
                ajust_fwd = 0.0
                r_dex = 0.0
                r_mm = 0.0
                r_tsx = 0.0
                r_sp500 = 0.0
                r_eafe = 0.0

            # Adjust forward rate if VM is 0
            if MT_VM_PROJ == 0:
                fwd_rate += ajust_fwd

            # ============= STEP 3: UPDATE DISCOUNT & APPLY RETURNS =============
            TX_ACTUALISATION *= math.exp(-fwd_rate * AJUST_NOUV_AFFAIRES)

            # Apply investment returns using continuous compounding
            MT_DEX_PROJ *= math.exp(r_dex * AJUST_NOUV_AFFAIRES)
            MT_MM_PROJ *= math.exp(r_mm * AJUST_NOUV_AFFAIRES)
            MT_TSX_PROJ *= math.exp(r_tsx * AJUST_NOUV_AFFAIRES)
            MT_SP500_PROJ *= math.exp(r_sp500 * AJUST_NOUV_AFFAIRES)
            MT_EAFE_PROJ *= math.exp(r_eafe * AJUST_NOUV_AFFAIRES)

            MT_VM_AV_RETRAIT_FRAIS = (MT_DEX_PROJ + MT_MM_PROJ + MT_TSX_PROJ +
                                      MT_SP500_PROJ + MT_EAFE_PROJ)

            # ============= STEP 4: CALCULATE LAPSE RATES =============
            lapse_tot = 0.0
            lapse_part = 0.0
            lapse = 0.0

            if MT_VM_PROJ > 0:
                # Calculate VM/VG ratio
                ratio1 = 9999.0
                if MT_GAR_ECH_PROJ > 0.01:
                    ratio1 = PC_GAR_ECH / MT_GAR_ECH_PROJ

                ratio2 = PC_GAR_DECES_1 / max(MT_BONI_DECES_PROJ + MT_GAR_DECES_PROJ, 0.01)

                ratio3 = 9999.0
                if MT_SRG_PROJ > 0.01:
                    ratio3 = 1.0 / MT_SRG_PROJ

                vm_avg = (MT_VM_PROJ + MT_VM_AV_RETRAIT_FRAIS) / 2.0
                vm_vg_ratio = min(10.0, vm_avg * min(ratio1, min(ratio2, ratio3)))

                # Total lapse
                if vm_vg_ratio <= 0.5:
                    lapse_niv_tot = 1
                    interpolation_tot = vm_vg_ratio / 0.5 if vm_vg_ratio > 0 else 0.0
                elif vm_vg_ratio <= 0.75:
                    lapse_niv_tot = 2
                    interpolation_tot = (vm_vg_ratio - 0.5) / 0.25
                else:
                    lapse_niv_tot = 3
                    interpolation_tot = (vm_vg_ratio - 0.75) / 999.24

                # Lookup total lapse parameters
                if (duree_max10 < lapse_tot_min.shape[0] and
                        ID_LAPSE < lapse_tot_min.shape[1] and
                        lapse_niv_tot < lapse_tot_min.shape[2]):
                    tx_lapse_tot_min = lapse_tot_min[duree_max10, ID_LAPSE, lapse_niv_tot]
                    tx_lapse_tot_max = lapse_tot_max[duree_max10, ID_LAPSE, lapse_niv_tot]
                    fact_dim = lapse_tot_fact[duree_max10, ID_LAPSE, lapse_niv_tot]
                else:
                    tx_lapse_tot_min = 0.0
                    tx_lapse_tot_max = 0.0
                    fact_dim = 1.0

                if tx_lapse_tot_min == tx_lapse_tot_max:
                    lapse_tot = tx_lapse_tot_min
                else:
                    lapse_tot = interpolation_tot * (tx_lapse_tot_max - tx_lapse_tot_min) + tx_lapse_tot_min

                if age >= AGE_DECAISSEMENT:
                    lapse_tot *= fact_dim

                # Partial lapse
                if vm_vg_ratio <= 0.5:
                    lapse_niv_part = 1
                    interpolation_part = vm_vg_ratio / 0.5 if vm_vg_ratio > 0 else 0.0
                elif vm_vg_ratio <= 0.75:
                    lapse_niv_part = 2
                    interpolation_part = (vm_vg_ratio - 0.5) / 0.25
                else:
                    lapse_niv_part = 3
                    interpolation_part = (vm_vg_ratio - 0.75) / 999.24

                # Lookup partial lapse parameters
                if (age < lapse_part_min.shape[0] and
                        ID_LAPSE < lapse_part_min.shape[1] and
                        I_REGIME_2 < lapse_part_min.shape[2] and
                        lapse_niv_part < lapse_part_min.shape[3]):
                    tx_lapse_part_min = lapse_part_min[age, ID_LAPSE, I_REGIME_2, lapse_niv_part]
                    tx_lapse_part_max = lapse_part_max[age, ID_LAPSE, I_REGIME_2, lapse_niv_part]
                else:
                    tx_lapse_part_min = 0.0
                    tx_lapse_part_max = 0.0

                if tx_lapse_part_min == tx_lapse_part_max:
                    lapse_part = tx_lapse_part_min
                else:
                    lapse_part = interpolation_part * (tx_lapse_part_max - tx_lapse_part_min) + tx_lapse_part_min

                # Convert to period rates
                exponent = (1.0 / freq_eval) * AJUST_NOUV_AFFAIRES
                lapse = 1.0 - math.pow(1.0 - lapse_tot - lapse_part, exponent)

            # ============= STEP 5: UPDATE SURVIVAL =============
            TX_SURVIE *= (1.0 - qx) * (1.0 - lapse)

            # ============= STEP 6: ACCUMULATE DEATH BONUS =============
            max_boni_deces = int(acc[43]) if len(acc) > 43 else 999
            if PC_BONI_DECES > 0 and age < max_boni_deces:
                MT_BONI_DECES_PROJ += (MT_GAR_DECES_PROJ * PC_BONI_DECES /
                                       freq_eval * AJUST_NOUV_AFFAIRES)
            else:
                MT_BONI_DECES_PROJ = 0.0

            # ============= STEP 7: APPLY FEES =============
            # Apply management fees (RFG)
            MT_VM_AV_RETRAIT = MT_VM_AV_RETRAIT_FRAIS * math.exp(
                -PC_RFG / freq_eval * AJUST_NOUV_AFFAIRES)

            # Calculate guarantee fee
            i_frais_sur_srg = int(acc[44]) if len(acc) > 44 else 0
            if i_frais_sur_srg == 0:
                base_fee_calc = MT_VM_AV_RETRAIT
            else:
                base_fee_calc = MT_SRG_PROJ

            guarantee_fee_amount = (base_fee_calc * PC_FRAIS_GARANTIE /
                                    freq_eval * AJUST_NOUV_AFFAIRES)
            guarantee_fee_amount = min(guarantee_fee_amount, MT_VM_AV_RETRAIT)

            primes_garanties = guarantee_fee_amount * TX_SURVIE_DEB
            vp_primes_garanties = primes_garanties * TX_ACTUALISATION

            # Deduct guarantee fee
            MT_VM_AV_RETRAIT = max(MT_VM_AV_RETRAIT - guarantee_fee_amount, 0.0)

            # ============= STEP 8: CALCULATE WITHDRAWALS =============
            retrait = 0.0
            age_retrait = age + 1

            if (age_retrait >= AGE_DECAISSEMENT and
                    not (age_retrait == AGE_DECAISSEMENT and mois_eval >= MOIS_NAIS) and
                    not (MT_VM_PROJ <= 0 and I_PRODUIT_REGR == 0)):

                # Minimum FERR
                if age < min_ferr_lookup.shape[0]:
                    min_ferr_rate = min_ferr_lookup[age]
                else:
                    min_ferr_rate = 0.0

                # Update MIN_FERR at start of year
                if ((an_eval == 1 and mois_eval == MOIS_EVALUATION_INI) or
                        mois_eval == 12 // freq_eval):
                    MT_MIN_FERR_PROJ = MT_VM_PROJ * min_ferr_rate

                # Calculate withdrawal (simplified - would need more account parameters)
                retrait = MT_MIN_FERR_PROJ / freq_eval

            # ============= STEP 9: PROCESS DEPOSITS =============
            depot_futur = 0.0

            if (duree_max10 < deposits_pc.shape[0] and
                    ID_DEPOT < deposits_pc.shape[1]):
                pc_depot_annuel = deposits_pc[duree_max10, ID_DEPOT]
                var_depot_fct = deposits_var[duree_max10, ID_DEPOT]
                age_max_depot = deposits_age_max[duree_max10, ID_DEPOT]
                i_even_cesse = deposits_i_even[duree_max10, ID_DEPOT]
            else:
                pc_depot_annuel = 0.0
                var_depot_fct = 0
                age_max_depot = 999
                i_even_cesse = 0

            # Check if deposits should cease
            if not (pc_depot_annuel == 0 or
                    (i_even_cesse == 1 and age_retrait >= AGE_DECAISSEMENT) or
                    (age_max_depot < age) or
                    (MT_VM_PROJ <= 0 and I_PRODUIT_REGR == 0)):

                # Calculate deposit base
                if var_depot_fct == 1:
                    base_depot = MT_VM_PROJ
                else:
                    base_depot = MT_GAR_DECES_PROJ / PC_GAR_DECES_1 if PC_GAR_DECES_1 > 0 else 0.0

                depot_futur = base_depot * pc_depot_annuel / freq_eval

            # Allocate deposits proportionally
            if depot_futur > 0 and MT_VM_PROJ > 0:
                MT_DEX_PROJ += depot_futur * (MT_DEX_PROJ / MT_VM_PROJ)
                MT_MM_PROJ += depot_futur * (MT_MM_PROJ / MT_VM_PROJ)
                MT_TSX_PROJ += depot_futur * (MT_TSX_PROJ / MT_VM_PROJ)
                MT_SP500_PROJ += depot_futur * (MT_SP500_PROJ / MT_VM_PROJ)
                MT_EAFE_PROJ += depot_futur * (MT_EAFE_PROJ / MT_VM_PROJ)

                MT_GAR_DECES_PROJ += depot_futur
                MT_GAR_ECH_PROJ += depot_futur * PC_GAR_ECH_DEP_FUT
                if MT_SRG_PROJ > 0:
                    MT_SRG_PROJ += depot_futur

            # ============= STEP 10: UPDATE VM AND GUARANTEES =============
            # Process withdrawal
            mt_vm_av_retrait = MT_VM_AV_RETRAIT

            if mt_vm_av_retrait <= retrait:
                MT_GAR_ECH_PROJ = 0.0
                MT_GAR_DECES_PROJ = 0.0
                MT_BONI_DECES_PROJ = 0.0
            elif mt_vm_av_retrait > 0:
                proportion = 1.0 - retrait / mt_vm_av_retrait
                MT_GAR_ECH_PROJ *= proportion
                MT_GAR_DECES_PROJ *= proportion
                MT_BONI_DECES_PROJ *= proportion
                MT_SRG_PROJ = max(MT_SRG_PROJ - retrait, 0.0)

            mt_vm_ap_retrait = max(mt_vm_av_retrait - retrait, 0.0)

            # Add deposits
            MT_VM_PROJ = mt_vm_ap_retrait + depot_futur

            # ============= STEP 11: CALCULATE BENEFITS =============
            # MRV benefit
            if I_PRODUIT_REGR == 1:
                prest_mrv = -max(retrait - mt_vm_av_retrait, 0.0) * TX_SURVIE_DEB
            else:
                prest_mrv = 0.0
            vp_prest_mrv = prest_mrv * TX_ACTUALISATION

            # Death benefit
            mt_vm_ap_retrait_depot = MT_VM_PROJ
            prest_deces = (qx * -max(0.0, MT_GAR_DECES_PROJ + MT_BONI_DECES_PROJ -
                                     mt_vm_ap_retrait_depot) * TX_SURVIE_DEB)
            vp_prest_deces = prest_deces * TX_ACTUALISATION

            # Maturity benefit (simplified)
            prest_ech = 0.0
            vp_prest_ech = 0.0

            # ============= STEP 12: REBALANCE PORTFOLIO =============
            if MT_VM_ORIG > 0 and MT_VM_PROJ > 0:
                MT_SP500_PROJ = MT_VM_PROJ * acc[23] / MT_VM_ORIG
                MT_TSX_PROJ = MT_VM_PROJ * acc[22] / MT_VM_ORIG
                MT_EAFE_PROJ = MT_VM_PROJ * acc[24] / MT_VM_ORIG
                MT_DEX_PROJ = MT_VM_PROJ * acc[20] / MT_VM_ORIG
                MT_MM_PROJ = MT_VM_PROJ * acc[21] / MT_VM_ORIG

            # ============= STEP 13: CALCULATE ACQUISITION COSTS =============
            comm_vente = 0.0
            vp_comm_vente = 0.0
            frais_acquis = 0.0
            vp_frais_acquis = 0.0
            pc_commission_maintien = 0.0

            if MT_VM_AV_RETRAIT_FRAIS > 0:
                if (duree_max10 < acq_vente_rf.shape[0] and
                        ID_ACQUI < acq_vente_rf.shape[1]):

                    pc_vente_rf = acq_vente_rf[duree_max10, ID_ACQUI]
                    pc_vente_ac = acq_vente_ac[duree_max10, ID_ACQUI]
                    pc_maint_rf = acq_maintien_rf[duree_max10, ID_ACQUI]
                    pc_maint_ac = acq_maintien_ac[duree_max10, ID_ACQUI]
                    pc_fr_ac = acq_frais_ac[duree_max10, ID_ACQUI]
                    pc_fr_rf = acq_frais_rf[duree_max10, ID_ACQUI]

                    if MT_VM_ORIG > 0:
                        pc_commission_vente = ((pc_vente_ac * (MT_VM_ORIG - MT_RF) / MT_VM_ORIG +
                                                pc_vente_rf * MT_RF / MT_VM_ORIG) *
                                               AJUSTEMENT_COMMISSION)
                        pc_commission_maintien = ((pc_maint_ac * (MT_VM_ORIG - MT_RF) / MT_VM_ORIG +
                                                   pc_maint_rf * MT_RF / MT_VM_ORIG) *
                                                  AJUSTEMENT_COMMISSION)
                        pc_frais_an = (pc_fr_ac * (MT_VM_ORIG - MT_RF) / MT_VM_ORIG +
                                       pc_fr_rf * MT_RF / MT_VM_ORIG)

                        comm_vente = -pc_commission_vente * depot_futur * TX_SURVIE
                        vp_comm_vente = comm_vente * TX_ACTUALISATION
                        frais_acquis = (pc_frais_an * mt_vm_ap_retrait * lapse *
                                        TX_SURVIE_DEB * (1.0 - qx))
                        vp_frais_acquis = frais_acquis * TX_ACTUALISATION

            # ============= STEP 14: CALCULATE OTHER FEES =============
            # Fixed fees
            if (ID_PRODUIT < fees_lookup.shape[0] and
                    annee_reelle < fees_lookup.shape[1]):
                fixed_fee_annual = fees_lookup[ID_PRODUIT, annee_reelle]
            else:
                fixed_fee_annual = 0.0

            if MT_VM_AV_RETRAIT > 0:
                frais_fixes = -fixed_fee_annual / freq_eval * AJUST_NOUV_AFFAIRES * TX_SURVIE_DEB
            else:
                frais_fixes = 0.0
            vp_frais_fixes = frais_fixes * TX_ACTUALISATION

            # Management fees (honoraires)
            hon_gest = (-MT_VM_AV_RETRAIT_FRAIS *
                        (math.exp(PC_HONORAIRES_GEST / freq_eval * AJUST_NOUV_AFFAIRES) - 1.0) *
                        TX_SURVIE_DEB)
            vp_hon_gest = hon_gest * TX_ACTUALISATION

            # Maintenance commission
            comm_maintien = (-MT_VM_AV_RETRAIT_FRAIS *
                             (math.exp(pc_commission_maintien / freq_eval * AJUST_NOUV_AFFAIRES) - 1.0) *
                             TX_SURVIE_DEB)
            vp_comm_maintien = comm_maintien * TX_ACTUALISATION

            # Variable premiums
            primes_variables = (MT_VM_AV_RETRAIT_FRAIS *
                                math.exp(-(PC_RFG - PC_REVENU_FDS) / freq_eval * AJUST_NOUV_AFFAIRES) *
                                -(math.exp(-PC_REVENU_FDS / freq_eval * AJUST_NOUV_AFFAIRES) - 1.0) *
                                TX_SURVIE_DEB)
            vp_primes_variables = primes_variables * TX_ACTUALISATION

            # Market value tracking
            valeur_marchande = MT_VM_PROJ * TX_SURVIE
            vp_valeur_marchande = valeur_marchande * TX_ACTUALISATION / freq_eval

            # ============= STEP 16: CALCULATE CUSHIONS =============
            # Determine CODE_CAT_PRODUIT
            if ID_PRODUIT == 22:
                code_cat_produit = 0
            elif ID_PRODUIT in [12, 13, 14, 15, 16]:
                code_cat_produit = 1
            elif ID_PRODUIT in [17, 18, 19, 20, 21]:
                code_cat_produit = 2
            elif ID_PRODUIT == 6:
                code_cat_produit = 3
            elif ID_PRODUIT in [4, 7]:
                code_cat_produit = 4
            elif ID_PRODUIT in [5, 8]:
                code_cat_produit = 5
            elif ID_PRODUIT in [2, 3]:
                code_cat_produit = 6
            else:
                code_cat_produit = 7

            # Determine CAT_COUSSIN_1 (based on % fixed income)
            if MT_VM_PROJ > 0:
                pct_rf = (MT_DEX_PROJ + MT_MM_PROJ) / MT_VM_PROJ
            else:
                pct_rf = 0.0

            if code_cat_produit in [0, 6]:
                cat_coussin_1 = 0
            elif code_cat_produit == 7 and pct_rf < 0.5:
                cat_coussin_1 = 4
            elif code_cat_produit == 7:
                cat_coussin_1 = 5
            elif pct_rf < 1.0 / 3.0:
                cat_coussin_1 = 1
            elif pct_rf < 2.0 / 3.0:
                cat_coussin_1 = 2
            else:
                cat_coussin_1 = 3

            # Determine CAT_COUSSIN_2
            ratio1 = 9999.0
            if MT_GAR_ECH_PROJ > 0.01:
                ratio1 = PC_GAR_ECH / MT_GAR_ECH_PROJ
            ratio2 = PC_GAR_DECES_1 / max(MT_BONI_DECES_PROJ + MT_GAR_DECES_PROJ, 0.01)
            ratio3 = 9999.0
            if MT_SRG_PROJ > 0.01:
                ratio3 = 1.0 / MT_SRG_PROJ
            vm_avg = (MT_VM_PROJ + MT_VM_AV_RETRAIT_FRAIS) / 2.0
            vm_vg_ratio = min(10.0, vm_avg * min(ratio1, min(ratio2, ratio3)))

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
            passif_redresse = 0.0
            coussin_credit = 0.0
            coussin_marche = 0.0
            coussin_depense = 0.0
            coussin_decheance = 0.0
            coussin_mortalite = 0.0
            coussin_depot = 0.0
            vp_passif_redresse = 0.0
            vp_coussin_credit = 0.0
            vp_coussin_marche = 0.0
            vp_coussin_depense = 0.0
            vp_coussin_decheance = 0.0
            vp_coussin_mortalite = 0.0
            vp_coussin_depot = 0.0

            if (code_cat_produit < cous_base_passif.shape[0] and
                    cat_coussin_1 < cous_base_passif.shape[1] and
                    cat_coussin_2 < cous_base_passif.shape[2]):

                # Get all cushion parameters
                base_passif = cous_base_passif[code_cat_produit, cat_coussin_1, cat_coussin_2]
                tx_passif = cous_tx_passif[code_cat_produit, cat_coussin_1, cat_coussin_2]
                base_credit = cous_base_credit[code_cat_produit, cat_coussin_1, cat_coussin_2]
                tx_credit = cous_tx_credit[code_cat_produit, cat_coussin_1, cat_coussin_2]
                base_marche = cous_base_marche[code_cat_produit, cat_coussin_1, cat_coussin_2]
                tx_marche = cous_tx_marche[code_cat_produit, cat_coussin_1, cat_coussin_2]
                base_depense = cous_base_depense[code_cat_produit, cat_coussin_1, cat_coussin_2]
                tx_depense = cous_tx_depense[code_cat_produit, cat_coussin_1, cat_coussin_2]
                base_decheance = cous_base_decheance[code_cat_produit, cat_coussin_1, cat_coussin_2]
                tx_decheance = cous_tx_decheance[code_cat_produit, cat_coussin_1, cat_coussin_2]
                base_mortalite = cous_base_mortalite[code_cat_produit, cat_coussin_1, cat_coussin_2]
                tx_mortalite = cous_tx_mortalite[code_cat_produit, cat_coussin_1, cat_coussin_2]
                base_depot_c = cous_base_depot[code_cat_produit, cat_coussin_1, cat_coussin_2]
                tx_depot = cous_tx_depot[code_cat_produit, cat_coussin_1, cat_coussin_2]
                facteur_80 = cous_facteur_80[code_cat_produit, cat_coussin_1, cat_coussin_2]
                facteur_90 = cous_facteur_90[code_cat_produit, cat_coussin_1, cat_coussin_2]

                # For RGS with VM=0, set certain cushions to 0
                if code_cat_produit == 7 and MT_VM_PROJ == 0:
                    tx_credit = 0.0
                    tx_marche = 0.0
                    tx_decheance = 0.0
                    tx_depot = 0.0

                # Determine age factor
                if age < 80:
                    age_factor = 1.0
                elif age < 90:
                    age_factor = facteur_80
                else:
                    age_factor = facteur_90

                # Calculate base amount
                max_guarantee = max(MT_GAR_ECH_PROJ, max(MT_GAR_DECES_PROJ + MT_BONI_DECES_PROJ, MT_SRG_PROJ))

                # Calculate each cushion
                base_amount_passif = max_guarantee if base_passif == 0 else MT_VM_PROJ
                passif_redresse = tx_passif * base_amount_passif * age_factor * TX_SURVIE
                vp_passif_redresse = passif_redresse * TX_ACTUALISATION / freq_eval

                base_amount_credit = max_guarantee if base_credit == 0 else MT_VM_PROJ
                coussin_credit = tx_credit * base_amount_credit * age_factor * TX_SURVIE
                vp_coussin_credit = coussin_credit * TX_ACTUALISATION / freq_eval

                base_amount_marche = max_guarantee if base_marche == 0 else MT_VM_PROJ
                coussin_marche = tx_marche * base_amount_marche * age_factor * TX_SURVIE
                vp_coussin_marche = coussin_marche * TX_ACTUALISATION / freq_eval

                base_amount_depense = max_guarantee if base_depense == 0 else MT_VM_PROJ
                coussin_depense = tx_depense * base_amount_depense * age_factor * TX_SURVIE
                vp_coussin_depense = coussin_depense * TX_ACTUALISATION / freq_eval

                base_amount_decheance = max_guarantee if base_decheance == 0 else MT_VM_PROJ
                coussin_decheance = tx_decheance * base_amount_decheance * age_factor * TX_SURVIE
                vp_coussin_decheance = coussin_decheance * TX_ACTUALISATION / freq_eval

                base_amount_mortalite = max_guarantee if base_mortalite == 0 else MT_VM_PROJ
                coussin_mortalite = tx_mortalite * base_amount_mortalite * age_factor * TX_SURVIE
                vp_coussin_mortalite = coussin_mortalite * TX_ACTUALISATION / freq_eval

                base_amount_depot_c = max_guarantee if base_depot_c == 0 else MT_VM_PROJ
                coussin_depot = tx_depot * base_amount_depot_c * age_factor * TX_SURVIE
                vp_coussin_depot = coussin_depot * TX_ACTUALISATION / freq_eval

            # ============= STEP 15: STORE RESULTS =============
            # Store results in output array
            if output_idx < output_results.shape[2]:
                output_results[account_idx, scenario_idx, output_idx, 0] = ID_COMPTE
                output_results[account_idx, scenario_idx, output_idx, 1] = scn_eval
                output_results[account_idx, scenario_idx, output_idx, 2] = an_eval
                output_results[account_idx, scenario_idx, output_idx, 3] = mois_eval
                output_results[account_idx, scenario_idx, output_idx, 4] = primes_garanties
                output_results[account_idx, scenario_idx, output_idx, 5] = prest_deces
                output_results[account_idx, scenario_idx, output_idx, 6] = prest_ech
                output_results[account_idx, scenario_idx, output_idx, 7] = prest_mrv
                output_results[account_idx, scenario_idx, output_idx, 8] = frais_acquis
                output_results[account_idx, scenario_idx, output_idx, 9] = comm_vente
                output_results[account_idx, scenario_idx, output_idx, 10] = primes_variables
                output_results[account_idx, scenario_idx, output_idx, 11] = frais_fixes
                output_results[account_idx, scenario_idx, output_idx, 12] = hon_gest
                output_results[account_idx, scenario_idx, output_idx, 13] = comm_maintien
                output_results[account_idx, scenario_idx, output_idx, 14] = valeur_marchande
                # Cushions
                output_results[account_idx, scenario_idx, output_idx, 15] = passif_redresse
                output_results[account_idx, scenario_idx, output_idx, 16] = coussin_credit
                output_results[account_idx, scenario_idx, output_idx, 17] = coussin_marche
                output_results[account_idx, scenario_idx, output_idx, 18] = coussin_depense
                output_results[account_idx, scenario_idx, output_idx, 19] = coussin_decheance
                output_results[account_idx, scenario_idx, output_idx, 20] = coussin_mortalite
                output_results[account_idx, scenario_idx, output_idx, 21] = coussin_depot
                # VP values
                output_results[account_idx, scenario_idx, output_idx, 22] = vp_frais_acquis
                output_results[account_idx, scenario_idx, output_idx, 23] = vp_comm_vente
                output_results[account_idx, scenario_idx, output_idx, 24] = vp_primes_garanties
                output_results[account_idx, scenario_idx, output_idx, 25] = vp_primes_variables
                output_results[account_idx, scenario_idx, output_idx, 26] = vp_frais_fixes
                output_results[account_idx, scenario_idx, output_idx, 27] = vp_hon_gest
                output_results[account_idx, scenario_idx, output_idx, 28] = vp_comm_maintien
                output_results[account_idx, scenario_idx, output_idx, 29] = vp_prest_ech
                output_results[account_idx, scenario_idx, output_idx, 30] = vp_prest_mrv
                output_results[account_idx, scenario_idx, output_idx, 31] = vp_prest_deces
                output_results[account_idx, scenario_idx, output_idx, 32] = vp_valeur_marchande
                # VP Cushions
                output_results[account_idx, scenario_idx, output_idx, 33] = vp_passif_redresse
                output_results[account_idx, scenario_idx, output_idx, 34] = vp_coussin_credit
                output_results[account_idx, scenario_idx, output_idx, 35] = vp_coussin_marche
                output_results[account_idx, scenario_idx, output_idx, 36] = vp_coussin_depense
                output_results[account_idx, scenario_idx, output_idx, 37] = vp_coussin_decheance
                output_results[account_idx, scenario_idx, output_idx, 38] = vp_coussin_mortalite
                output_results[account_idx, scenario_idx, output_idx, 39] = vp_coussin_depot

                output_idx += 1


# =============================================================================
# GPU EXECUTION WRAPPER
# =============================================================================

def prepare_account_data(population_df):
    """Convert account DataFrame to numpy array for GPU."""
    # Define the columns to extract in order
    columns = [
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
        'ANNEE_COTIS', 'MOIS_COTIS', 'MAX_BONI_DECES', 'I_FRAIS_SUR_SRG'
    ]

    # Fill missing columns with defaults
    for col in columns:
        if col not in population_df.columns:
            if col in ['MT_BCB', 'MT_BONI_DECES', 'MT_MRV_MRG_MRA', 'TAUX_MRV_MRG_MRA',
                       'PC_BONI_DECES', 'PC_REVENU_FDS', 'MT_RF']:
                population_df[col] = 0.0
            elif col in ['ANNEE_ECH', 'MAX_BONI_DECES']:
                population_df[col] = 9999
            elif col in ['MOIS_ECH']:
                population_df[col] = 12
            elif col in ['AJUSTEMENT_COMMISSION']:
                population_df[col] = 1.0
            elif col in ['ANNEE_COTIS']:
                population_df[col] = population_df.get('ANNEE_EVALUATION_INI', 2020)
            elif col in ['MOIS_COTIS', 'I_FRAIS_SUR_SRG']:
                population_df[col] = 0

    account_data = population_df[columns].values.astype(np.float32)
    account_ids = population_df['ID_COMPTE'].values.astype(np.int32)

    return account_data, account_ids


# =============================================================================
# GPU EXECUTION WRAPPER (MODIFIED FOR BATCHING - WITH CONTIGUOUS FIX)
# =============================================================================

def run_projection_gpu(data_path: Path, output_path: Path, nb_an_projection: int, nb_scenarios: int,
                       max_accounts: int = None, threads_per_block=(16, 16), use_pinned_memory=True,
                       debug_account: Optional[int] = None, debug_scenario: Optional[int] = None,
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
                       progress_callback: Optional[callable] = None):
    """
    Main function to run GPU-accelerated projection in batches to manage memory.

    Args:
        data_path: Path to input CSV files (default location)
        output_path: Path for output files
        nb_an_projection: Number of years to project
        nb_scenarios: Number of economic scenarios
        max_accounts: Maximum number of accounts (for testing)
        threads_per_block: CUDA block dimensions (accounts, scenarios)
        use_pinned_memory: Use pinned memory for faster transfers (default: True)
        debug_account: Optional account ID for debugging
        debug_scenario: Optional scenario ID for debugging
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
    start_time = datetime.now()
    print(f"Starting GPU projection at {start_time}")
    print("=" * 60)
    
    # Check GPU availability
    try:
        if not cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. Please ensure:\n"
                "  1. NVIDIA GPU is present in the system\n"
                "  2. NVIDIA drivers are installed\n"
                "  3. Docker container is run with '--gpus all' flag\n"
                "  4. NVIDIA Container Toolkit is installed"
            )
        
        # Test GPU access
        gpu = cuda.get_current_device()
        print(f"GPU Device: {gpu.name.decode()}")
        
        # Try to get memory info - may not work with RMM allocator
        try:
            free_mem, total_mem = cuda.current_context().get_memory_info()
            print(f"GPU Memory: {free_mem / 1024**3:.2f} GB free / {total_mem / 1024**3:.2f} GB total")
        except NotImplementedError:
            # RMM allocator doesn't support get_memory_info()
            print(f"GPU Memory: Information not available (using RMM allocator)")
    except Exception as e:
        raise RuntimeError(
            f"Failed to initialize GPU: {e}\n"
            f"Please ensure Docker is run with '--gpus all' flag and NVIDIA drivers are installed."
        )

    # Update config
    CONFIG['NB_AN_PROJECTION'] = nb_an_projection
    CONFIG['NB_SC'] = nb_scenarios
    print(f"Configuration: {nb_an_projection} years, {nb_scenarios} scenarios")
    print(f"Optimization: Pinned memory = {use_pinned_memory}")

    # Load data
    try:
        print("\nLoading data files...")
        data = load_all_data(data_path,
                             population_path=population_path,
                             mortalite_path=mortalite_path,
                             rendements_path=rendements_path,
                             depots_futurs_path=depots_futurs_path,
                             frais_admin_path=frais_admin_path,
                             min_ferr_path=min_ferr_path,
                             tx_lapse_part_path=tx_lapse_part_path,
                             tx_lapse_tot_path=tx_lapse_tot_path,
                             acquisition_path=acquisition_path,
                             coussins_escap_path=coussins_escap_path)
        print("✓ Data loaded successfully")
    except Exception as e:
        raise RuntimeError(f"Failed to load data: {e}")

    if max_accounts:
        data['population'] = data['population'].head(max_accounts)

    n_accounts = len(data['population'])
    print(f"\nPreparing {n_accounts} accounts for GPU processing...")

    # Prepare all account data on CPU first
    try:
        all_account_data, _ = prepare_account_data(data['population'])
        print("✓ Account data prepared")
    except Exception as e:
        raise RuntimeError(f"Failed to prepare account data: {e}")

    # Create GPU lookup tables
    print("\nCreating GPU lookup tables...")
    try:
        mortality_lookup = create_gpu_mortality_lookup(data['mortalite'])
        print("  ✓ Mortality lookup")
        (forward_rate, ajust_forward, rend_dex, rend_mm, rend_tsx,
         rend_sp500, rend_eafe) = create_gpu_returns_lookup(data['rendements'])
        print("  ✓ Returns lookup")
        min_ferr_lookup = create_gpu_min_ferr_lookup(data['min_ferr'])
        print("  ✓ Min FERR lookup")
        lapse_part_min, lapse_part_max = create_gpu_lapse_part_lookup(data['tx_lapse_part'])
        print("  ✓ Lapse partial lookup")
        lapse_tot_min, lapse_tot_max, lapse_tot_fact = create_gpu_lapse_tot_lookup(data['tx_lapse_tot'])
        print("  ✓ Lapse total lookup")
        (deposits_pc, deposits_var, deposits_age_max,
         deposits_i_even) = create_gpu_deposits_lookup(data['depots_futurs'])
        print("  ✓ Deposits lookup")
        fees_lookup = create_gpu_fees_lookup(data['frais_admin'])
        print("  ✓ Fees lookup")
        (acq_vente_rf, acq_vente_ac, acq_maintien_rf, acq_maintien_ac,
         acq_frais_ac, acq_frais_rf) = create_gpu_acquisition_lookup(data['acquisition'])
        print("  ✓ Acquisition lookup")
        (cous_base_passif, cous_tx_passif, cous_base_credit, cous_tx_credit,
         cous_base_marche, cous_tx_marche, cous_base_depense, cous_tx_depense,
         cous_base_decheance, cous_tx_decheance, cous_base_mortalite, cous_tx_mortalite,
         cous_base_depot, cous_tx_depot, cous_facteur_80,
         cous_facteur_90) = create_gpu_coussins_lookup(data['coussins_escap'])
        print("  ✓ Coussins lookup")
        print("✓ All GPU lookup tables created")
    except Exception as e:
        raise RuntimeError(f"Failed to create GPU lookup tables: {e}\n{type(e).__name__}")

    lookup_tables = [
        mortality_lookup, forward_rate, ajust_forward, rend_dex, rend_mm, rend_tsx, rend_sp500, rend_eafe,
        min_ferr_lookup, lapse_part_min, lapse_part_max, lapse_tot_min, lapse_tot_max, lapse_tot_fact,
        deposits_pc, deposits_var, deposits_age_max, deposits_i_even, fees_lookup, acq_vente_rf, acq_vente_ac,
        acq_maintien_rf, acq_maintien_ac, acq_frais_ac, acq_frais_rf, cous_base_passif, cous_tx_passif,
        cous_base_credit, cous_tx_credit, cous_base_marche, cous_tx_marche, cous_base_depense, cous_tx_depense,
        cous_base_decheance, cous_tx_decheance, cous_base_mortalite, cous_tx_mortalite, cous_base_depot,
        cous_tx_depot, cous_facteur_80, cous_facteur_90
    ]

    # --- BATCH SIZE CALCULATION ---
    print("\nCalculating optimal batch size to maximize GPU memory usage...")

    static_mem_usage = sum(table.nbytes for table in lookup_tables)
    gpu = cuda.get_current_device()
    
    # Try to get memory info - may not work with RMM allocator
    try:
        free_mem, total_mem = cuda.current_context().get_memory_info()
        safety_margin = 0.95
        available_mem_for_dynamic_data = (free_mem - static_mem_usage) * safety_margin
        
        print(f"  Total GPU Memory: {total_mem / 1024 ** 3:.2f} GB")
        print(f"  Free GPU Memory: {free_mem / 1024 ** 3:.2f} GB")
        print(f"  Static Lookup Tables Size: {static_mem_usage / 1024 ** 3:.2f} GB")
        print(f"  Available Memory for Batches: {available_mem_for_dynamic_data / 1024 ** 3:.2f} GB")
    except NotImplementedError:
        # RMM allocator doesn't support get_memory_info() - use estimated memory
        # Assume ~16GB total memory for RTX 4000 Ada, use conservative estimate
        estimated_total_mem = 16 * 1024**3  # 16 GB
        estimated_free_mem = estimated_total_mem * 0.8  # Assume 80% available
        safety_margin = 0.8  # More conservative when estimating
        available_mem_for_dynamic_data = (estimated_free_mem - static_mem_usage) * safety_margin
        
        print(f"  GPU Memory: Information not available (using RMM allocator)")
        print(f"  Estimated Total Memory: {estimated_total_mem / 1024 ** 3:.2f} GB")
        print(f"  Static Lookup Tables Size: {static_mem_usage / 1024 ** 3:.2f} GB")
        print(f"  Estimated Available Memory for Batches: {available_mem_for_dynamic_data / 1024 ** 3:.2f} GB")

    max_timesteps = (nb_an_projection + 1) * CONFIG['FREQ_EVAL']
    n_output_fields = 40

    mem_per_account_input = all_account_data.shape[1] * all_account_data.dtype.itemsize
    mem_per_account_output = nb_scenarios * max_timesteps * n_output_fields * np.dtype(np.float32).itemsize
    total_mem_per_account = mem_per_account_input + mem_per_account_output

    print(f"  Memory per account (Input + Output): {total_mem_per_account / 1024 ** 2:.2f} MB")

    if available_mem_for_dynamic_data < total_mem_per_account:
        raise MemoryError(
            f"Not enough GPU memory to process even one account. "
            f"Required: {total_mem_per_account / 1024 ** 2:.2f} MB, "
            f"Available: {available_mem_for_dynamic_data / 1024 ** 2:.2f} MB"
        )

    # Calculate memory-based batch size
    batch_size_memory = int(available_mem_for_dynamic_data // total_mem_per_account)
    
    # Cap batch size to prevent CPU memory allocation bottlenecks
    # Large batches (>4GB) cause memory fragmentation and 100x slower extraction
    # This limits NumPy boolean indexing copy operations to manageable sizes
    max_output_size_gb = 4.0
    max_batch_size_transfer = int((max_output_size_gb * 1024**3) / mem_per_account_output)
    
    # Use the minimum of memory-based and transfer-based limits
    batch_size = min(batch_size_memory, max_batch_size_transfer)
    batch_size = max(1, min(batch_size, n_accounts))
    num_batches = (n_accounts + batch_size - 1) // batch_size

    print(f"  Memory-based max batch size: {batch_size_memory} accounts")
    print(f"  Transfer-optimized max batch size: {max_batch_size_transfer} accounts")
    print(f"  ==> Selected batch size: {batch_size} accounts ({batch_size * mem_per_account_output / 1024**3:.2f} GB per batch)")
    print(f"  ==> Total batches: {num_batches}")

    # --- BATCHED EXECUTION ---

    print("\nCopying static lookup tables to GPU...")
    d_mortality, d_forward_rate, d_ajust_forward, d_rend_dex, d_rend_mm, d_rend_tsx, d_rend_sp500, d_rend_eafe, \
        d_min_ferr, d_lapse_part_min, d_lapse_part_max, d_lapse_tot_min, d_lapse_tot_max, d_lapse_tot_fact, \
        d_deposits_pc, d_deposits_var, d_deposits_age_max, d_deposits_i_even, d_fees, d_acq_vente_rf, d_acq_vente_ac, \
        d_acq_maintien_rf, d_acq_maintien_ac, d_acq_frais_ac, d_acq_frais_rf, d_cous_base_passif, d_cous_tx_passif, \
        d_cous_base_credit, d_cous_tx_credit, d_cous_base_marche, d_cous_tx_marche, d_cous_base_depense, d_cous_tx_depense, \
        d_cous_base_decheance, d_cous_tx_decheance, d_cous_base_mortalite, d_cous_tx_mortalite, d_cous_base_depot, \
        d_cous_tx_depot, d_cous_facteur_80, d_cous_facteur_90 = [cuda.to_device(table) for table in lookup_tables]

    # Create CUDA streams for async pipelined operations
    stream_compute = cuda.stream()  # For GPU compute
    stream_transfer = cuda.stream()  # For data transfers (if needed)
    
    # DuckDB-based batch storage strategy (simpler, more efficient)
    max_possible_rows = n_accounts * nb_scenarios * max_timesteps
    estimated_rows = int(max_possible_rows * 0.6)
    estimated_memory_gb = estimated_rows * n_output_fields * 4 / 1024**3
    
    print(f"\nUsing PARQUET + DUCKDB BATCH STORAGE (optimized for speed)")
    print(f"Estimated: {n_accounts:,} accounts, {estimated_rows:,} rows, ~{estimated_memory_gb:.1f} GB")
    print(f"Each batch will be written to Parquet file (fast columnar format)")
    print(f"Final aggregation will be done with DuckDB SQL reading Parquet files")
    
    # Create temporary Parquet directory
    parquet_dir = Path(output_path) / "_temp_parquet"
    parquet_dir.mkdir(parents=True, exist_ok=True)
    
    # Define column names for the table
    columns = [
        'ID_COMPTE', 'SCN_EVAL', 'AN_EVAL', 'MOIS_EVAL',
        'PRIMES_GARANTIES', 'PREST_DECES', 'PREST_ECH', 'PREST_MRV',
        'FRAIS_ACQUIS', 'COMM_VENTE', 'PRIMES_VARIABLES', 'FRAIS_FIXES',
        'HON_GEST', 'COMM_MAINTIEN', 'VALEUR_MARCHANDE', 'PASSIF_REDRESSE',
        'COUSSIN_CREDIT', 'COUSSIN_MARCHE', 'COUSSIN_DEPENSE', 'COUSSIN_DECHEANCE',
        'COUSSIN_MORTALITE', 'COUSSIN_DEPOT',
        'VP_FRAIS_ACQUIS', 'VP_COMM_VENTE', 'VP_PRIMES_GARANTIES', 'VP_PRIMES_VARIABLES',
        'VP_FRAIS_FIXES', 'VP_HON_GEST', 'VP_COMM_MAINTIEN', 'VP_PREST_ECH',
        'VP_PREST_MRV', 'VP_PREST_DECES', 'VP_VALEUR_MARCHANDE', 'VP_PASSIF_REDRESSE',
        'VP_COUSSIN_CREDIT', 'VP_COUSSIN_MARCHE', 'VP_COUSSIN_DEPENSE', 'VP_COUSSIN_DECHEANCE',
        'VP_COUSSIN_MORTALITE', 'VP_COUSSIN_DEPOT'
    ]
    
    print(f"Parquet directory: {parquet_dir}")
    if CUDF_AVAILABLE:
        print(f"Using cuDF (GPU) filtering + CuPy fast transfer + pandas write (hybrid optimized)")
    else:
        print(f"Using pandas (CPU) DataFrame + LZ4 compression + async I/O (CPU-optimized)")
    
    # Create thread pool for async parquet writes
    parquet_writer_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    write_futures = []
    
    total_kernel_duration = 0
    total_transfer_duration = 0

    for i in range(num_batches):
        batch_start_time = datetime.now()
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_accounts)
        current_batch_size = end_idx - start_idx

        # Memory snapshot at batch start
        process = psutil.Process(os.getpid())
        batch_mem_start = process.memory_info().rss / 1024**3
        
        logger.info(f"\n--- Processing Batch {i + 1}/{num_batches} (Accounts {start_idx} to {end_idx - 1}) ---")
        logger.info(f"  Memory at batch start: {batch_mem_start:.2f} GB")
        
        # Report progress if callback provided
        if progress_callback:
            try:
                progress_callback(i + 1, num_batches)
            except Exception as e:
                print(f"  Warning: Progress callback failed: {e}")

        # 1. Prepare batch-specific data
        batch_account_data = np.ascontiguousarray(all_account_data[start_idx:end_idx])

        # 2. Allocate batch-specific output array on CPU (use pinned memory if enabled)
        if use_pinned_memory:
            h_batch_output = cuda.pinned_array((current_batch_size, nb_scenarios, max_timesteps, n_output_fields), dtype=np.float32)
            h_batch_output[:] = 0  # Initialize to zero
        else:
            h_batch_output = np.zeros((current_batch_size, nb_scenarios, max_timesteps, n_output_fields), dtype=np.float32)
        logger.info(f"  Batch output array size: {h_batch_output.nbytes / 1024 ** 3:.3f} GB")

        # 3. Copy batch data to GPU (async with stream)
        transfer_start = datetime.now()
        logger.info("  Copying batch data to GPU...")
        if use_pinned_memory:
            # Use pinned memory for input too
            h_batch_input_pinned = cuda.pinned_array(batch_account_data.shape, dtype=batch_account_data.dtype)
            h_batch_input_pinned[:] = batch_account_data
            d_batch_account_data = cuda.to_device(h_batch_input_pinned, stream=stream_compute)
        else:
            d_batch_account_data = cuda.to_device(batch_account_data, stream=stream_compute)
        d_batch_output = cuda.to_device(h_batch_output, stream=stream_compute)
        stream_compute.synchronize()
        transfer_end = datetime.now()
        transfer_to_gpu = (transfer_end - transfer_start).total_seconds()
        logger.info(f"  Transfer to GPU: {transfer_to_gpu:.2f} seconds")

        # 4. Calculate grid dimensions for the current batch
        blocks_x = (current_batch_size + threads_per_block[0] - 1) // threads_per_block[0]
        blocks_y = (nb_scenarios + threads_per_block[1] - 1) // threads_per_block[1]
        blocks_per_grid = (blocks_x, blocks_y)

        logger.info(f"  Launching kernel for batch:")
        logger.info(f"    Grid: {blocks_per_grid}, Block: {threads_per_block}")

        # 5. Launch kernel for the batch (using stream)
        kernel_start = datetime.now()
        projection_kernel[blocks_per_grid, threads_per_block, stream_compute](
            # Batch-specific data
            d_batch_account_data,
            # Scenario parameters
            nb_scenarios, nb_an_projection, CONFIG['FREQ_EVAL'],
            # Lookup tables...
            d_mortality, d_forward_rate, d_ajust_forward, d_rend_dex, d_rend_mm, d_rend_tsx, d_rend_sp500, d_rend_eafe,
            d_min_ferr, d_lapse_part_min, d_lapse_part_max, d_lapse_tot_min, d_lapse_tot_max, d_lapse_tot_fact,
            d_deposits_pc, d_deposits_var, d_deposits_age_max, d_deposits_i_even, d_fees,
            d_acq_vente_rf, d_acq_vente_ac, d_acq_maintien_rf, d_acq_maintien_ac, d_acq_frais_ac, d_acq_frais_rf,
            d_cous_base_passif, d_cous_tx_passif, d_cous_base_credit, d_cous_tx_credit, d_cous_base_marche,
            d_cous_tx_marche,
            d_cous_base_depense, d_cous_tx_depense, d_cous_base_decheance, d_cous_tx_decheance,
            d_cous_base_mortalite, d_cous_tx_mortalite, d_cous_base_depot, d_cous_tx_depot,
            d_cous_facteur_80, d_cous_facteur_90,
            # Batch-specific output
            d_batch_output
        )
        stream_compute.synchronize()
        kernel_end = datetime.now()
        kernel_duration = (kernel_end - kernel_start).total_seconds()
        total_kernel_duration += kernel_duration
        logger.info(f"  Kernel execution for batch finished in: {kernel_duration:.2f} seconds")

        # 6. Process results - GPU path (cuDF) or CPU path (pandas)
        cpu_proc_start = datetime.now()
        process = psutil.Process(os.getpid())
        
        if CUDF_AVAILABLE:
            # ===== GPU-ACCELERATED PATH (cuDF) =====
            logger.info("  Processing results on GPU with cuDF...")
            
            # Reshape GPU array directly (stays on GPU!)
            reshape_start = datetime.now()
            total_rows = current_batch_size * nb_scenarios * max_timesteps
            d_batch_reshaped = d_batch_output.reshape(total_rows, n_output_fields)
            reshape_time = (datetime.now() - reshape_start).total_seconds()
            logger.info(f"    Reshape on GPU: {reshape_time:.3f}s")
            
            # Create cuDF DataFrame directly from GPU memory (NO CPU TRANSFER!)
            prep_start = datetime.now()
            logger.info(f"    Creating cuDF DataFrame from GPU array...")
            
            # Convert Numba device array to CuPy array (zero-copy via __cuda_array_interface__)
            # Both Numba and CuPy support the CUDA Array Interface protocol
            cupy_array = cp.asarray(d_batch_reshaped)
            
            # Convert to cuDF DataFrame column by column
            gpu_data = {}
            for col_idx, col_name in enumerate(columns):
                gpu_data[col_name] = cudf.Series(cupy_array[:, col_idx])
            
            gpu_df = cudf.DataFrame(gpu_data)
            df_create_time = (datetime.now() - prep_start).total_seconds()
            logger.info(f"    cuDF DataFrame created on GPU: {df_create_time:.3f}s")
            
            # Filter on GPU (MUCH faster than CPU!)
            filter_start = datetime.now()
            gpu_df = gpu_df[gpu_df['ID_COMPTE'] > 0]
            num_rows = len(gpu_df)
            filter_time = (datetime.now() - filter_start).total_seconds()
            logger.info(f"    Filtered on GPU to {num_rows:,} valid rows: {filter_time:.3f}s")
            
            # Convert ID columns to int32 on GPU
            type_start = datetime.now()
            gpu_df['ID_COMPTE'] = gpu_df['ID_COMPTE'].astype('int32')
            gpu_df['SCN_EVAL'] = gpu_df['SCN_EVAL'].astype('int32')
            gpu_df['AN_EVAL'] = gpu_df['AN_EVAL'].astype('int32')
            gpu_df['MOIS_EVAL'] = gpu_df['MOIS_EVAL'].astype('int32')
            type_time = (datetime.now() - type_start).total_seconds()
            
            prep_time = (datetime.now() - prep_start).total_seconds()
            logger.info(f"    cuDF prep total: {prep_time:.3f}s (create:{df_create_time:.3f}s, filter:{filter_time:.3f}s, types:{type_time:.3f}s)")
            
            if num_rows > 0:
                # Transfer to CPU efficiently using CuPy (MUCH faster than to_pandas)
                # cuDF.to_parquet() needs 3x GPU memory for compression buffers → OOM
                # cuDF.to_pandas() is extremely slow (201s!) → Bad
                # Solution: Use CuPy arrays directly (fast transfer)
                transfer_start = datetime.now()
                logger.info(f"    Transferring filtered data to CPU (CuPy → NumPy)...")
                
                # Convert cuDF columns to CuPy, then to NumPy (much faster than to_pandas)
                cpu_data = {}
                for col in columns:
                    # Get CuPy array from cuDF column, then transfer to CPU
                    cpu_data[col] = cp.asnumpy(gpu_df[col].values)
                
                transfer_time = (datetime.now() - transfer_start).total_seconds()
                logger.info(f"    Transferred {num_rows:,} rows to CPU: {transfer_time:.3f}s")
                
                # Free GPU memory immediately!
                del gpu_df, gpu_data, cupy_array
                
                # Also delete the reshaped GPU array
                del d_batch_reshaped
                
                gc.collect()
                cuda.synchronize()  # Wait for all CUDA operations to complete
                
                # Force RMM memory pool to release unused memory
                try:
                    import rmm
                    rmm.mr.get_current_device_resource().deallocate(0, 0)  # Trigger pool cleanup
                    logger.info(f"    GPU memory freed (RMM pool cleaned)")
                except:
                    logger.info(f"    GPU memory freed (Python GC only)")
                    pass  # If RMM cleanup fails, continue anyway
                
                # Create pandas DataFrame from NumPy arrays (fast on CPU)
                df_start = datetime.now()
                df = pd.DataFrame(cpu_data)
                df_time = (datetime.now() - df_start).total_seconds()
                logger.info(f"    Created pandas DataFrame: {df_time:.3f}s")
                
                del cpu_data
                
                # Write from CPU (pandas is memory-efficient for parquet writes)
                write_start = datetime.now()
                parquet_path = parquet_dir / f"batch_{i:04d}.parquet"
                df.to_parquet(
                    parquet_path,
                    engine='pyarrow',
                    compression='lz4',
                    index=False
                )
                
                write_time = (datetime.now() - write_start).total_seconds()
                file_size_mb = parquet_path.stat().st_size / 1024**2
                logger.info(f"    Written {file_size_mb:.1f} MB in {write_time:.3f}s ({file_size_mb/write_time:.1f} MB/s)")
                
                # Free CPU DataFrame
                del df
                gc.collect()
                
                # Track stats
                write_futures.append((i, file_size_mb, num_rows, transfer_time + write_time))
            else:
                logger.info(f"    No valid rows in batch - skipping write")
                del cupy_array, gpu_data, d_batch_reshaped
                gc.collect()
                cuda.synchronize()
            
            # Write completed, GPU memory freed
            transfer_from_gpu = 0.0  # Filtered data transferred via CuPy
            total_transfer_duration += transfer_to_gpu  # Upload only
            logger.info(f"  ✓ GPU-accelerated filtering + fast CuPy transfer (GPU memory freed)")
            
        else:
            # ===== CPU PATH (pandas) =====
            transfer_back_start = datetime.now()
            logger.info("  Copying batch results from GPU...")
            d_batch_output.copy_to_host(h_batch_output, stream=stream_compute)
            stream_compute.synchronize()
            transfer_back_end = datetime.now()
            transfer_from_gpu = (transfer_back_end - transfer_back_start).total_seconds()
            total_transfer_duration += transfer_to_gpu + transfer_from_gpu
            logger.info(f"  Transfer from GPU: {transfer_from_gpu:.2f} seconds")
            
            logger.info("  Extracting valid results...")
            
            # Reshape directly from pinned memory (no copy needed!)
            reshape_start = datetime.now()
            total_rows = current_batch_size * nb_scenarios * max_timesteps
            reshaped = h_batch_output.reshape(total_rows, n_output_fields)
            reshape_time = (datetime.now() - reshape_start).total_seconds()
            logger.info(f"    Reshape to 2D ({total_rows:,} x {n_output_fields}): {reshape_time:.3f}s (zero-copy from pinned memory)")
            
            # Use pandas DataFrame (optimized for columnar operations)
            extract_start = datetime.now()
            logger.info(f"    Creating valid mask for {total_rows:,} rows...")
            valid_mask = reshaped[:, 0] > 0
            n_valid = np.sum(valid_mask)
            mask_time = (datetime.now() - extract_start).total_seconds()
            logger.info(f"    Found {n_valid:,} valid rows ({n_valid/total_rows*100:.2f}% occupancy) in {mask_time:.3f}s")
            
            if n_valid > 0:
                # Convert to pandas DataFrame
                prep_start = datetime.now()
                logger.info(f"    Creating DataFrame directly from NumPy array...")
                
                # Create DataFrame from full reshaped array (fast - uses views where possible)
                df = pd.DataFrame(reshaped, columns=columns)
                df_create_time = (datetime.now() - prep_start).total_seconds()
                logger.info(f"    DataFrame created: {df_create_time:.3f}s")
                
                # Filter to valid rows (pandas is highly optimized for this)
                filter_start = datetime.now()
                df = df[df['ID_COMPTE'] > 0]
                num_rows = len(df)
                filter_time = (datetime.now() - filter_start).total_seconds()
                logger.info(f"    Filtered to {num_rows:,} valid rows: {filter_time:.3f}s")
                
                # Convert ID columns to int32 for smaller file size
                type_start = datetime.now()
                df['ID_COMPTE'] = df['ID_COMPTE'].astype(np.int32)
                df['SCN_EVAL'] = df['SCN_EVAL'].astype(np.int32)
                df['AN_EVAL'] = df['AN_EVAL'].astype(np.int32)
                df['MOIS_EVAL'] = df['MOIS_EVAL'].astype(np.int32)
                type_time = (datetime.now() - type_start).total_seconds()
                
                prep_time = (datetime.now() - prep_start).total_seconds()
                logger.info(f"    DataFrame prep total: {prep_time:.3f}s (create:{df_create_time:.3f}s, filter:{filter_time:.3f}s, types:{type_time:.3f}s)")
                
                # Free large arrays immediately (before async write)
                del h_batch_output, reshaped, valid_mask
                gc.collect()
                
                # Submit async write to thread pool
                parquet_path = parquet_dir / f"batch_{i:04d}.parquet"
                future = parquet_writer_pool.submit(
                    write_parquet_async_pandas,
                    df,
                    parquet_path,
                    i,
                    num_rows
                )
                write_futures.append(future)
                logger.info(f"    Parquet write submitted to async pool (batch {i})")
                
                # Free DataFrame immediately
                del df
                gc.collect()
            else:
                # No valid data - just clean up
                del h_batch_output, reshaped, valid_mask
                logger.info(f"    No valid rows in batch - skipping write")
        
        cpu_proc_end = datetime.now()
        cpu_proc_time = (cpu_proc_end - cpu_proc_start).total_seconds()
        logger.info(f"    Total CPU processing: {cpu_proc_time:.2f}s")

        # Cleanup remaining GPU/batch memory
        del d_batch_account_data
        del d_batch_output
        # h_batch_output, reshaped, valid_mask, df already deleted above
        if use_pinned_memory and 'h_batch_input_pinned' in locals():
            del h_batch_input_pinned
        
        # Force garbage collection and GPU memory cleanup after EVERY batch
        gc_start = datetime.now()
        force_gpu_memory_cleanup()
        gc_time = (datetime.now() - gc_start).total_seconds()
        batch_mem_end = process.memory_info().rss / 1024**3
        logger.info(f"  [Memory cleanup: GPU+GC in {gc_time:.3f}s, mem at end: {batch_mem_end:.2f} GB, delta: {batch_mem_end - batch_mem_start:+.2f} GB]")

        batch_end_time = datetime.now()
        batch_duration = (batch_end_time - batch_start_time).total_seconds()
        cpu_processing_time = batch_duration - kernel_duration - transfer_to_gpu - transfer_from_gpu
        logger.info(f"  Batch {i + 1} total time: {batch_duration:.2f}s (Kernel: {kernel_duration:.2f}s, Transfer: {transfer_to_gpu + transfer_from_gpu:.2f}s, CPU: {cpu_processing_time:.2f}s)")

    # --- SUMMARIZE WRITES (synchronous with cuDF, async with pandas) ---
    print("\n" + "="*60)
    print("PARQUET WRITE SUMMARY")
    print("="*60)
    
    wait_start = datetime.now()
    total_write_time = 0
    total_file_size_mb = 0
    total_rows_written = 0
    
    # Handle mixed futures (tuples for cuDF sync writes, futures for pandas async)
    if len(write_futures) > 0:
        # Check if first item is a future or tuple
        if hasattr(write_futures[0], 'result'):
            # Async pandas writes - wait for completion
            print(f"\nWaiting for {len(write_futures)} async write operations...")
            for future in concurrent.futures.as_completed(write_futures):
                batch_num, file_size_mb, num_rows, write_time = future.result()
                total_write_time += write_time
                total_file_size_mb += file_size_mb
                total_rows_written += num_rows
        else:
            # Synchronous cuDF writes - already complete, just sum up
            print(f"\nAll {len(write_futures)} writes completed synchronously (cuDF GPU path)")
            for batch_num, file_size_mb, num_rows, write_time in write_futures:
                total_write_time += write_time
                total_file_size_mb += file_size_mb
                total_rows_written += num_rows
    
    # Shutdown thread pool
    parquet_writer_pool.shutdown(wait=True)
    
    wait_duration = (datetime.now() - wait_start).total_seconds()
    avg_write_time = total_write_time / len(write_futures) if write_futures else 0
    
    print(f"\nWrite statistics:")
    print(f"  Processing time: {wait_duration:.2f}s")
    print(f"  Total write time: {total_write_time:.2f}s")
    print(f"  Average write time per batch: {avg_write_time:.3f}s")
    print(f"  Total data written: {total_file_size_mb:.1f} MB ({total_rows_written:,} rows)")
    print(f"  Effective speedup from parallelism: {total_write_time/wait_duration:.1f}x")
    
    # --- FINAL AGGREGATION WITH DUCKDB + PARQUET ---
    print("\n" + "="*60)
    print("FINAL DATA ASSEMBLY (DuckDB reading Parquet)")
    print("="*60)
    
    merge_start = datetime.now()
    
    # Count Parquet files
    parquet_files = list(parquet_dir.glob("batch_*.parquet"))
    print(f"\nFound {len(parquet_files)} Parquet files to aggregate")
    
    # DuckDB can read all Parquet files with wildcard pattern
    parquet_pattern = str(parquet_dir / "batch_*.parquet")
    
    # Perform aggregation using DuckDB SQL (averaging across scenarios)
    # DuckDB reads Parquet files in parallel for fast aggregation
    print("Aggregating across scenarios using DuckDB SQL (parallel Parquet read)...")
    agg_sql = f"""
    SELECT 
        ID_COMPTE,
        AN_EVAL,
        MOIS_EVAL,
        AVG(PRIMES_GARANTIES) AS PRIMES_GARANTIES,
        AVG(PREST_DECES) AS PREST_DECES,
        AVG(PREST_ECH) AS PREST_ECH,
        AVG(PREST_MRV) AS PREST_MRV,
        AVG(FRAIS_ACQUIS) AS FRAIS_ACQUIS,
        AVG(COMM_VENTE) AS COMM_VENTE,
        AVG(PRIMES_VARIABLES) AS PRIMES_VARIABLES,
        AVG(FRAIS_FIXES) AS FRAIS_FIXES,
        AVG(HON_GEST) AS HON_GEST,
        AVG(COMM_MAINTIEN) AS COMM_MAINTIEN,
        AVG(VALEUR_MARCHANDE) AS VALEUR_MARCHANDE,
        AVG(PASSIF_REDRESSE) AS PASSIF_REDRESSE,
        AVG(COUSSIN_CREDIT) AS COUSSIN_CREDIT,
        AVG(COUSSIN_MARCHE) AS COUSSIN_MARCHE,
        AVG(COUSSIN_DEPENSE) AS COUSSIN_DEPENSE,
        AVG(COUSSIN_DECHEANCE) AS COUSSIN_DECHEANCE,
        AVG(COUSSIN_MORTALITE) AS COUSSIN_MORTALITE,
        AVG(COUSSIN_DEPOT) AS COUSSIN_DEPOT,
        AVG(VP_FRAIS_ACQUIS) AS VP_FRAIS_ACQUIS,
        AVG(VP_COMM_VENTE) AS VP_COMM_VENTE,
        AVG(VP_PRIMES_GARANTIES) AS VP_PRIMES_GARANTIES,
        AVG(VP_PRIMES_VARIABLES) AS VP_PRIMES_VARIABLES,
        AVG(VP_FRAIS_FIXES) AS VP_FRAIS_FIXES,
        AVG(VP_HON_GEST) AS VP_HON_GEST,
        AVG(VP_COMM_MAINTIEN) AS VP_COMM_MAINTIEN,
        AVG(VP_PREST_ECH) AS VP_PREST_ECH,
        AVG(VP_PREST_MRV) AS VP_PREST_MRV,
        AVG(VP_PREST_DECES) AS VP_PREST_DECES,
        AVG(VP_VALEUR_MARCHANDE) AS VP_VALEUR_MARCHANDE,
        AVG(VP_PASSIF_REDRESSE) AS VP_PASSIF_REDRESSE,
        AVG(VP_COUSSIN_CREDIT) AS VP_COUSSIN_CREDIT,
        AVG(VP_COUSSIN_MARCHE) AS VP_COUSSIN_MARCHE,
        AVG(VP_COUSSIN_DEPENSE) AS VP_COUSSIN_DEPENSE,
        AVG(VP_COUSSIN_DECHEANCE) AS VP_COUSSIN_DECHEANCE,
        AVG(VP_COUSSIN_MORTALITE) AS VP_COUSSIN_MORTALITE,
        AVG(VP_COUSSIN_DEPOT) AS VP_COUSSIN_DEPOT
    FROM read_parquet('{parquet_pattern}')
    GROUP BY ID_COMPTE, AN_EVAL, MOIS_EVAL
    ORDER BY ID_COMPTE, AN_EVAL, MOIS_EVAL
    """
    
    # Execute aggregation and fetch result as pandas DataFrame
    all_results = duckdb.execute(agg_sql).df()
    
    merge_time = (datetime.now() - merge_start).total_seconds()
    print(f"  Aggregated to {len(all_results):,} rows: {merge_time:.2f}s")
    print(f"  Size: {all_results.memory_usage(deep=True).sum() / 1024**3:.2f} GB")
    
    # Cleanup Parquet files
    print("\nCleaning up Parquet files...")
    try:
        import shutil
        shutil.rmtree(parquet_dir)
        print(f"  Removed temporary Parquet directory: {parquet_dir}")
    except Exception as e:
        print(f"  Warning: Could not remove temporary Parquet files: {e}")

    # Debug logging for specific account
    # Note: With scenario averaging, we show the averaged results (not individual scenarios)
    if debug_account is not None:
        print("\n" + "="*60)
        print(f"DEBUG: DETAILED RESULTS FOR ACCOUNT {debug_account}")
        print(f"(Showing scenario-averaged results across all {nb_scenarios} scenarios)")
        print("="*60)
        
        # Filter for the specific account (scenarios are already averaged)
        debug_mask = all_results['ID_COMPTE'] == debug_account
        debug_data = all_results[debug_mask].copy()
        
        if len(debug_data) > 0:
            # Sort by year and month
            debug_data = debug_data.sort_values(['AN_EVAL', 'MOIS_EVAL'])
            
            print(f"\nFound {len(debug_data):,} timesteps for this account (scenario-averaged)")
            print("\nYear-by-year summary:")
            print("-" * 120)
            
            # Group by year and show key metrics
            for year in sorted(debug_data['AN_EVAL'].unique()):
                year_data = debug_data[debug_data['AN_EVAL'] == year]
                
                # Sum over all months in the year
                primes_tot = year_data['PRIMES_GARANTIES'].sum() + year_data['PRIMES_VARIABLES'].sum()
                prest_tot = year_data['PREST_DECES'].sum() + year_data['PREST_ECH'].sum() + year_data['PREST_MRV'].sum()
                frais_tot = year_data['FRAIS_ACQUIS'].sum() + year_data['FRAIS_FIXES'].sum()
                valeur_march = year_data['VALEUR_MARCHANDE'].iloc[-1]  # End of year value
                passif = year_data['PASSIF_REDRESSE'].iloc[-1]  # End of year value
                
                print(f"Year {year:3d} | Months: {len(year_data):2d} | "
                      f"Premiums: ${primes_tot:12,.2f} | Benefits: ${prest_tot:12,.2f} | "
                      f"Fees: ${frais_tot:10,.2f} | Market Val: ${valeur_march:12,.2f} | "
                      f"Liability: ${passif:12,.2f}")
            
            print("-" * 120)
            
            # Show detailed monthly data for first and last year
            print(f"\nDetailed monthly data for Year 0:")
            year_0 = debug_data[debug_data['AN_EVAL'] == 0]
            print(year_0[['AN_EVAL', 'MOIS_EVAL', 'PRIMES_GARANTIES', 'PRIMES_VARIABLES', 
                         'PREST_DECES', 'VALEUR_MARCHANDE', 'PASSIF_REDRESSE']].to_string(index=False))
            
            last_year = debug_data['AN_EVAL'].max()
            print(f"\nDetailed monthly data for Year {last_year}:")
            year_last = debug_data[debug_data['AN_EVAL'] == last_year]
            print(year_last[['AN_EVAL', 'MOIS_EVAL', 'PRIMES_GARANTIES', 'PRIMES_VARIABLES',
                            'PREST_DECES', 'VALEUR_MARCHANDE', 'PASSIF_REDRESSE']].to_string(index=False))
            
            # Show present values
            print(f"\nPresent Values (discounted to time 0):")
            vp_cols = [col for col in debug_data.columns if col.startswith('VP_')]
            if vp_cols:
                # Sum all PV columns (they're already summed across all timesteps)
                vp_summary = debug_data[vp_cols].iloc[0]  # First row has cumulative PVs
                for col in vp_cols:
                    if abs(vp_summary[col]) > 0.01:  # Only show non-zero values
                        print(f"  {col:30s}: ${vp_summary[col]:15,.2f}")
            
            # Save debug data to CSV
            output_path.mkdir(parents=True, exist_ok=True)
            debug_filename = f"DEBUG_account_{debug_account}_scenario_averaged.csv"
            debug_filepath = output_path.joinpath(debug_filename)
            debug_data.to_csv(debug_filepath, index=False, sep=';')
            print(f"\n✓ Debug data saved to: {debug_filepath}")
            print(f"  Contains {len(debug_data):,} timesteps with all {len(debug_data.columns)} columns")
        else:
            print(f"\n⚠️  WARNING: No data found for Account {debug_account}")
            print(f"    Available account range: {all_results['ID_COMPTE'].min()} to {all_results['ID_COMPTE'].max()}")
        
        print("="*60 + "\n")
    
    # Aggregate and save results
    print("\n" + "="*60)
    print("AGGREGATING RESULTS")
    print("="*60)
    
    from cpu import (aggregate_flux_projetes, aggregate_vp_flux_compte, aggregate_vp_flux_total)

    agg_start = datetime.now()
    
    # Scenario averaging already done by DuckDB SQL
    print("Scenario averaging already completed by DuckDB")
    calculs_sommaire = all_results
    print(f"  → {len(calculs_sommaire):,} rows")
    
    print("Creating VP_FLUX_COMPTE...")
    vp_flux_compte = aggregate_vp_flux_compte(calculs_sommaire)
    print(f"  → {len(vp_flux_compte):,} accounts")
    
    print("Creating VP_FLUX_TOTAL...")
    vp_flux_total = aggregate_vp_flux_total(vp_flux_compte)
    
    print("Creating FLUX_PROJETES...")
    flux_projetes = aggregate_flux_projetes(calculs_sommaire)
    print(f"  → {len(flux_projetes):,} time periods")
    
    agg_time = (datetime.now() - agg_start).total_seconds()
    print(f"\nTotal aggregation time: {agg_time:.2f}s")

    # Save outputs
    print("\nSaving outputs...")
    output_path.mkdir(parents=True, exist_ok=True)
    flux_projetes.to_csv(output_path.joinpath("FLUX_PROJETES_GPU.csv"), index=False, sep=';')
    vp_flux_compte.to_csv(output_path.joinpath("VP_FLUX_COMPTE_GPU.csv"), index=False, sep=';')
    vp_flux_total.to_csv(output_path.joinpath("VP_FLUX_TOTAL_GPU.csv"), index=False, sep=';')
    print(f"  ✓ Saved {output_path}/FLUX_PROJETES_GPU.csv")
    print(f"  ✓ Saved {output_path}/VP_FLUX_COMPTE_GPU.csv")
    print(f"  ✓ Saved {output_path}/VP_FLUX_TOTAL_GPU.csv")

    # Print summary
    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()
    
    # Calculate time breakdowns
    cpu_extraction_time = total_duration - total_kernel_duration - total_transfer_duration - agg_time
    
    print("\n" + "=" * 60)
    print("GPU PROJECTION COMPLETE")
    print("=" * 60)
    print(f"Total processing time: {total_duration:.2f} seconds ({total_duration / 60:.2f} minutes)")
    print(f"\nTime Breakdown:")
    print(f"  GPU Kernel execution:  {total_kernel_duration:8.2f}s ({100*total_kernel_duration/total_duration:5.1f}%)")
    print(f"  GPU↔CPU Transfers:     {total_transfer_duration:8.2f}s ({100*total_transfer_duration/total_duration:5.1f}%)")
    print(f"  CPU Data Extraction:   {cpu_extraction_time:8.2f}s ({100*cpu_extraction_time/total_duration:5.1f}%)")
    print(f"  Aggregation:           {agg_time:8.2f}s ({100*agg_time/total_duration:5.1f}%)")
    print(f"\nProcessing Statistics:")
    print(f"  Accounts processed: {n_accounts}")
    print(f"  Batch size: {batch_size} accounts")
    print(f"  Total batches: {num_batches}")
    print(f"  Scenarios per account: {nb_scenarios}")
    print(f"  Average time per batch: {total_duration/num_batches:.2f}s")
    print(f"  Total rows generated: {len(all_results):,}")
    print(f"\nResults:")
    print(f"  Total PV of flows: ${vp_flux_total['VP_FLUX_TOT'].iloc[0]:,.2f}")
    print("=" * 60)

    return {
        'flux_projetes': flux_projetes,
        'vp_flux_compte': vp_flux_compte,
        'vp_flux_total': vp_flux_total,
    }


# =============================================================================
# SCRIPT ENTRY POINT (MODIFIED)
# =============================================================================

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run GPU-accelerated actuarial projections')
    parser.add_argument('--debug-account', type=int, default=None,
                       help='Account ID to show detailed year-by-year results (for debugging)')
    parser.add_argument('--debug-scenario', type=int, default=None,
                       help='Scenario number (ignored - showing scenario-averaged results)')
    parser.add_argument('--max-accounts', type=int, default=None,
                       help='Maximum number of accounts to process (for testing)')
    args = parser.parse_args()
    
    try:
        if not cuda.is_available():
            print("ERROR: CUDA is not available. Please check your GPU setup.")
            exit(1)

        print(f"CUDA Device: {cuda.get_current_device().name}")
        
        # Show debug mode info
        if args.debug_account is not None:
            print(f"\n🔍 DEBUG MODE ENABLED")
            print(f"   Will show detailed scenario-averaged results for:")
            print(f"   Account {args.debug_account}")
            print(f"   (Note: --debug-scenario parameter is ignored, showing averaged results)")
            print()

        DATA_PATH = HERE.joinpath("data_in")
        OUTPUT_PATH = HERE.joinpath("data_out_gpu")

        results = run_projection_gpu(
            data_path=DATA_PATH,
            output_path=OUTPUT_PATH,
            nb_an_projection=100,
            nb_scenarios=100,
            max_accounts=args.max_accounts,
            threads_per_block=(32, 8),  # (accounts_per_block, scenarios_per_block) - 256 threads per block
            debug_account=args.debug_account,
            debug_scenario=args.debug_scenario
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

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback

        traceback.print_exc()
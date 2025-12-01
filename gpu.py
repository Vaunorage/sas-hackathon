import os
# Set environment variables BEFORE importing numba
os.environ['NUMBA_CUDA_ENABLE_PYNVJITLINK'] = '1'

import pandas as pd
import numpy as np
import polars as pl
import gc
import psutil
import logging
import shutil
from pathlib import Path
from typing import Dict, Tuple, Any, Optional
from datetime import datetime
import math
import duckdb
import tempfile
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
from fastparquet import write as fastparquet_write

# Try to import cuDF for GPU-accelerated DataFrame operations
try:
    import cudf
    import cupy as cp
    CUDF_AVAILABLE = True  # ENABLED for batch-level aggregation before saving
except ImportError:
    CUDF_AVAILABLE = False
    print("⚠ CuDF not available - falling back to pandas (CPU). Install with: pip install cudf-cu12")

# GPU memory cleanup helper
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
        compression='snappy',
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
        compression='snappy',
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
# GPU KERNELS - TWO-PASS NESTED STOCHASTIC ARCHITECTURE
# =============================================================================

# State variables passed between Kernel A and Kernel B
# Index mapping for state tensor
STATE_MT_VM = 0
STATE_MT_GAR_DECES = 1
STATE_MT_GAR_ECH = 2
STATE_MT_SRG = 3
STATE_AGE = 4
STATE_TX_SURVIE = 5
STATE_MT_DEX = 6
STATE_MT_MM = 7
STATE_MT_TSX = 8
STATE_MT_SP500 = 9
STATE_MT_EAFE = 10
STATE_MT_BONI_DECES = 11
STATE_SIZE = 12  # Total number of state variables

@cuda.jit
def external_generator_kernel(
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
        # Output arrays
        output_states,     # Shape: (Batch_Size, n_scenarios, n_years, STATE_SIZE)
        output_cashflows   # Shape: (Batch_Size, n_scenarios, n_years, CF_SIZE)
):
    """
    KERNEL A: EXTERNAL SCENARIO GENERATOR (Tier 1)
    
    Runs the external (real-world) scenarios and saves intermediate states at each timestep.
    These states will be used by Kernel B to perform nested valuations.
    
    Output:
    - State tensor: All policy state variables at each external scenario node
    - Cashflow tensor: Nominal cashflows for reporting (Tier 1 results)
    """
    # Get global thread ID
    account_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    scenario_idx = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # Boundary check
    if account_idx >= account_data.shape[0] or scenario_idx >= n_scenarios:
        return

    # Load account data into registers
    acc = account_data[account_idx]

    # Account static data indices
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

    TX_SURVIE = 1.0

    PC_HONORAIRES_GEST = acc[30]
    PC_FRAIS_GARANTIE = acc[31]
    PC_GAR_DECES_1 = acc[32]
    PC_BONI_DECES = acc[33]
    PC_RFG = acc[34]
    PC_REVENU_FDS = acc[35]
    PC_GAR_ECH = acc[36]
    PC_GAR_ECH_DEP_FUT = acc[37]
    MT_VM_ORIG = acc[40]

    ANNEE_COTIS = int(acc[41]) if acc[41] > 0 else ANNEE_EVALUATION_INI
    MOIS_COTIS = int(acc[42]) if acc[42] > 0 else 1

    # Scenario-specific processing
    scn_eval = scenario_idx + 1

    output_year_idx = 0
    AJUST_NOUV_AFFAIRES = 1.0

    # TIME LOOP - External Projection
    for an_eval in range(0, n_years + 1):
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

            # === PROJECTION LOGIC (SIMPLIFIED FOR BREVITY) ===
            # [Insert full projection logic from existing kernel here]
            # For now, showing key steps:
            
            # 1. Mortality lookup
            month_diff = MOIS_NAIS - mois_eval
            if month_diff <= 0:
                month_diff += 12
            age_mort = age + 1 if month_diff <= 6 else age
            age_mort = min(age_mort, 120)
            
            if (I_SEXE < mortality_lookup.shape[0] and
                    age_mort < mortality_lookup.shape[1] and
                    annee_reelle < mortality_lookup.shape[2] and
                    I_PRODUIT_REGR < mortality_lookup.shape[3]):
                qx = mortality_lookup[I_SEXE, age_mort, annee_reelle, I_PRODUIT_REGR]
            else:
                qx = 0.001
            qx = 1.0 - math.pow(1.0 - qx, (1.0 / freq_eval * AJUST_NOUV_AFFAIRES))

            # 2. Returns lookup
            if (scn_eval < forward_rate.shape[0] and
                    an_eval < forward_rate.shape[1] and
                    mois_eval < forward_rate.shape[2]):
                r_dex = rend_dex[scn_eval, an_eval, mois_eval]
                r_mm = rend_mm[scn_eval, an_eval, mois_eval]
                r_tsx = rend_tsx[scn_eval, an_eval, mois_eval]
                r_sp500 = rend_sp500[scn_eval, an_eval, mois_eval]
                r_eafe = rend_eafe[scn_eval, an_eval, mois_eval]
            else:
                r_dex = r_mm = r_tsx = r_sp500 = r_eafe = 0.0

            # 3. Apply returns
            MT_DEX_PROJ *= math.exp(r_dex * AJUST_NOUV_AFFAIRES)
            MT_MM_PROJ *= math.exp(r_mm * AJUST_NOUV_AFFAIRES)
            MT_TSX_PROJ *= math.exp(r_tsx * AJUST_NOUV_AFFAIRES)
            MT_SP500_PROJ *= math.exp(r_sp500 * AJUST_NOUV_AFFAIRES)
            MT_EAFE_PROJ *= math.exp(r_eafe * AJUST_NOUV_AFFAIRES)

            MT_VM_AV_RETRAIT_FRAIS = (MT_DEX_PROJ + MT_MM_PROJ + MT_TSX_PROJ +
                                      MT_SP500_PROJ + MT_EAFE_PROJ)

            # 4. Calculate lapse (simplified)
            lapse = 0.0
            if MT_VM_PROJ > 0:
                # [Insert full lapse calculation here]
                lapse = 0.01  # Placeholder

            # 5. Update survival
            TX_SURVIE *= (1.0 - qx) * (1.0 - lapse)

            # 6. Apply fees (simplified)
            MT_VM_AV_RETRAIT = MT_VM_AV_RETRAIT_FRAIS * math.exp(-PC_RFG / freq_eval * AJUST_NOUV_AFFAIRES)
            
            # 7. Update VM
            MT_VM_PROJ = MT_VM_AV_RETRAIT

            # 8. Portfolio rebalance
            if MT_VM_ORIG > 0 and MT_VM_PROJ > 0:
                MT_SP500_PROJ = MT_VM_PROJ * acc[23] / MT_VM_ORIG
                MT_TSX_PROJ = MT_VM_PROJ * acc[22] / MT_VM_ORIG
                MT_EAFE_PROJ = MT_VM_PROJ * acc[24] / MT_VM_ORIG
                MT_DEX_PROJ = MT_VM_PROJ * acc[20] / MT_VM_ORIG
                MT_MM_PROJ = MT_VM_PROJ * acc[21] / MT_VM_ORIG

            # === SAVE STATE TO GLOBAL MEMORY ===
            if an_eval < n_years and output_year_idx < output_states.shape[2]:
                output_states[account_idx, scenario_idx, output_year_idx, STATE_MT_VM] = MT_VM_PROJ
                output_states[account_idx, scenario_idx, output_year_idx, STATE_MT_GAR_DECES] = MT_GAR_DECES_PROJ
                output_states[account_idx, scenario_idx, output_year_idx, STATE_MT_GAR_ECH] = MT_GAR_ECH_PROJ
                output_states[account_idx, scenario_idx, output_year_idx, STATE_MT_SRG] = MT_SRG_PROJ
                output_states[account_idx, scenario_idx, output_year_idx, STATE_AGE] = float(age)
                output_states[account_idx, scenario_idx, output_year_idx, STATE_TX_SURVIE] = TX_SURVIE
                output_states[account_idx, scenario_idx, output_year_idx, STATE_MT_DEX] = MT_DEX_PROJ
                output_states[account_idx, scenario_idx, output_year_idx, STATE_MT_MM] = MT_MM_PROJ
                output_states[account_idx, scenario_idx, output_year_idx, STATE_MT_TSX] = MT_TSX_PROJ
                output_states[account_idx, scenario_idx, output_year_idx, STATE_MT_SP500] = MT_SP500_PROJ
                output_states[account_idx, scenario_idx, output_year_idx, STATE_MT_EAFE] = MT_EAFE_PROJ
                output_states[account_idx, scenario_idx, output_year_idx, STATE_MT_BONI_DECES] = MT_BONI_DECES_PROJ

                # Save cashflows for reporting
                flux_net = 0.0  # Calculate actual net cashflow
                output_cashflows[account_idx, scenario_idx, output_year_idx, 0] = flux_net

                output_year_idx += 1


@cuda.jit
def nested_valuation_kernel(
        # Input states from Kernel A
        input_states,        # Shape: (Batch_Size, n_ext_scenarios, n_years, STATE_SIZE)
        account_data,        # Account static data for internal calculations
        # Internal scenario parameters
        n_internal_scenarios,
        n_internal_years,
        shock_capital_pct,   # e.g. 0.30 for 30% shock
        # Risk Neutral / Internal Tables
        rn_forward_rate,     # Risk Neutral Returns
        rn_rend_dex, rn_rend_mm, rn_rend_tsx, rn_rend_sp500, rn_rend_eafe,
        mortality_lookup,    # Reuse mortality table
        # Output arrays
        output_metrics       # Shape: (Batch_Size, n_ext_scenarios, n_years, 2) -> [Reserve, Capital]
):
    """
    KERNEL B: NESTED VALUATOR (Tier 2 & 3)
    
    For each external scenario node, runs internal (risk-neutral) scenarios
    to calculate:
    - Reserve (Best Estimate): PV of future cashflows under risk-neutral scenarios
    - Capital: PV of future cashflows under shocked initial conditions
    
    Each thread processes one external node (Account x Ext_Scenario x Year)
    """
    # Thread setup - one thread per external node
    global_idx = cuda.grid(1)
    
    # Unpack indices from flat index
    total_years = input_states.shape[2]
    total_scens = input_states.shape[1]
    
    year_idx = global_idx % total_years
    rem = global_idx // total_years
    scn_idx = rem % total_scens
    acc_idx = rem // total_scens
    
    if acc_idx >= input_states.shape[0]:
        return

    # Load starting state from Kernel A output
    start_vm = input_states[acc_idx, scn_idx, year_idx, STATE_MT_VM]
    start_gar_deces = input_states[acc_idx, scn_idx, year_idx, STATE_MT_GAR_DECES]
    start_gar_ech = input_states[acc_idx, scn_idx, year_idx, STATE_MT_GAR_ECH]
    start_srg = input_states[acc_idx, scn_idx, year_idx, STATE_MT_SRG]
    start_age = input_states[acc_idx, scn_idx, year_idx, STATE_AGE]
    start_tx_survie = input_states[acc_idx, scn_idx, year_idx, STATE_TX_SURVIE]
    
    # Check if policy is active
    if start_vm <= 0 or start_tx_survie <= 0:
        output_metrics[acc_idx, scn_idx, year_idx, 0] = 0.0
        output_metrics[acc_idx, scn_idx, year_idx, 1] = 0.0
        return

    # Load account parameters
    acc = account_data[acc_idx]
    PC_RFG = acc[34]
    
    # ==================================================
    # PART A: CALCULATE RESERVE (Tier 2 - Best Estimate)
    # ==================================================
    sum_pv_reserve = 0.0
    
    for i_int in range(n_internal_scenarios):
        # Initialize internal state
        curr_vm = start_vm
        curr_age = int(start_age)
        
        pv_path = 0.0
        
        # Internal time loop (project to run-off)
        for t_int in range(n_internal_years):
            if curr_vm <= 0:
                break
            
            # Apply Risk Neutral Return (simplified)
            # In practice, you would sample from risk-neutral distribution
            if (i_int < rn_rend_dex.shape[0] and
                    t_int < rn_rend_dex.shape[1]):
                r_rn = rn_rend_dex[i_int, t_int]  # Simplified single-fund example
            else:
                r_rn = 0.02  # Fallback
            
            curr_vm *= math.exp(r_rn)
            
            # Deduct fees
            fees = curr_vm * PC_RFG
            curr_vm -= fees
            
            # Calculate net flux
            flux = fees  # Simplified
            
            # Discount (using risk-neutral discount factor)
            if (i_int < rn_forward_rate.shape[0] and
                    t_int < rn_forward_rate.shape[1]):
                fwd = rn_forward_rate[i_int, t_int]
            else:
                fwd = 0.02
            
            df = math.exp(-fwd * (t_int + 1))
            pv_path += flux * df
            
            curr_age += 1
        
        sum_pv_reserve += pv_path

    # Average over internal scenarios
    reserve_val = sum_pv_reserve / n_internal_scenarios if n_internal_scenarios > 0 else 0.0
    
    # ==================================================
    # PART B: CALCULATE CAPITAL (Tier 3 - Stressed)
    # ==================================================
    sum_pv_capital = 0.0
    shocked_vm_start = start_vm * (1.0 - shock_capital_pct)
    
    for i_int in range(n_internal_scenarios):
        # Initialize with SHOCK
        curr_vm = shocked_vm_start
        
        pv_path = 0.0
        
        for t_int in range(n_internal_years):
            if curr_vm <= 0:
                break
            
            # Same projection logic as reserve, but with shocked starting point
            if (i_int < rn_rend_dex.shape[0] and
                    t_int < rn_rend_dex.shape[1]):
                r_rn = rn_rend_dex[i_int, t_int]
            else:
                r_rn = 0.02
            
            curr_vm *= math.exp(r_rn)
            fees = curr_vm * PC_RFG
            curr_vm -= fees
            
            flux = fees
            
            if (i_int < rn_forward_rate.shape[0] and
                    t_int < rn_forward_rate.shape[1]):
                fwd = rn_forward_rate[i_int, t_int]
            else:
                fwd = 0.02
            
            df = math.exp(-fwd * (t_int + 1))
            pv_path += flux * df
        
        sum_pv_capital += pv_path

    capital_req_val = sum_pv_capital / n_internal_scenarios if n_internal_scenarios > 0 else 0.0

    # Store results
    output_metrics[acc_idx, scn_idx, year_idx, 0] = reserve_val
    output_metrics[acc_idx, scn_idx, year_idx, 1] = capital_req_val


# =============================================================================
# GPU KERNEL - MAIN PROJECTION ENGINE (ORIGINAL SINGLE-PASS)
# =============================================================================

@cuda.jit
def projection_kernel(
        # Account data
        account_data,  # Shape: (n_accounts, n_account_fields)
        # Scenario parameters
        n_scenarios,
        n_years,
        freq_eval,
        debug_account_id,  # ID of account to output detailed data (-1 = no debug, output VP totals only)
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
        output_results  # Shape: (n_accounts, 1 or max_timesteps, n_output_fields) - 1 row (VP totals) per account unless debug
):
    """
    Main CUDA kernel - processes one account-scenario combination per thread.
    Each thread loops through all timesteps sequentially (state dependency).
    Results are atomically aggregated across scenarios within the kernel.
    
    OUTPUT MODES:
    - Production mode (debug_account_id = -1): Only VP totals (1 row per account)
    - Debug mode (debug_account_id = specific ID): All timesteps for that account only
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

    # Check if this account needs detailed output (for debugging)
    output_all_timesteps = (ID_COMPTE == debug_account_id)
    
    # Accumulators for VP totals (used when output_all_timesteps == False)
    total_vp_frais_acquis = 0.0
    total_vp_comm_vente = 0.0
    total_vp_primes_garanties = 0.0
    total_vp_primes_variables = 0.0
    total_vp_frais_fixes = 0.0
    total_vp_hon_gest = 0.0
    total_vp_comm_maintien = 0.0
    total_vp_prest_ech = 0.0
    total_vp_prest_mrv = 0.0
    total_vp_prest_deces = 0.0
    total_vp_valeur_marchande = 0.0
    total_vp_passif_redresse = 0.0
    total_vp_coussin_credit = 0.0
    total_vp_coussin_marche = 0.0
    total_vp_coussin_depense = 0.0
    total_vp_coussin_decheance = 0.0
    total_vp_coussin_mortalite = 0.0
    total_vp_coussin_depot = 0.0

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
            if output_all_timesteps:
                # DEBUG MODE: Write all timesteps for this account (atomic aggregation across scenarios)
                if output_idx < output_results.shape[1]:
                    # Write ID fields only once (same across all scenarios)
                    if scenario_idx == 0:
                        output_results[account_idx, output_idx, 0] = float(ID_COMPTE)
                        output_results[account_idx, output_idx, 1] = 0.0  # SCN_EVAL removed (aggregated)
                        output_results[account_idx, output_idx, 2] = float(an_eval)
                        output_results[account_idx, output_idx, 3] = float(mois_eval)
                    
                    # Atomically accumulate value fields across scenarios
                    cuda.atomic.add(output_results, (account_idx, output_idx, 4), primes_garanties)
                    cuda.atomic.add(output_results, (account_idx, output_idx, 5), prest_deces)
                    cuda.atomic.add(output_results, (account_idx, output_idx, 6), prest_ech)
                    cuda.atomic.add(output_results, (account_idx, output_idx, 7), prest_mrv)
                    cuda.atomic.add(output_results, (account_idx, output_idx, 8), frais_acquis)
                    cuda.atomic.add(output_results, (account_idx, output_idx, 9), comm_vente)
                    cuda.atomic.add(output_results, (account_idx, output_idx, 10), primes_variables)
                    cuda.atomic.add(output_results, (account_idx, output_idx, 11), frais_fixes)
                    cuda.atomic.add(output_results, (account_idx, output_idx, 12), hon_gest)
                    cuda.atomic.add(output_results, (account_idx, output_idx, 13), comm_maintien)
                    cuda.atomic.add(output_results, (account_idx, output_idx, 14), valeur_marchande)
                    # Cushions
                    cuda.atomic.add(output_results, (account_idx, output_idx, 15), passif_redresse)
                    cuda.atomic.add(output_results, (account_idx, output_idx, 16), coussin_credit)
                    cuda.atomic.add(output_results, (account_idx, output_idx, 17), coussin_marche)
                    cuda.atomic.add(output_results, (account_idx, output_idx, 18), coussin_depense)
                    cuda.atomic.add(output_results, (account_idx, output_idx, 19), coussin_decheance)
                    cuda.atomic.add(output_results, (account_idx, output_idx, 20), coussin_mortalite)
                    cuda.atomic.add(output_results, (account_idx, output_idx, 21), coussin_depot)
                    # VP values
                    cuda.atomic.add(output_results, (account_idx, output_idx, 22), vp_frais_acquis)
                    cuda.atomic.add(output_results, (account_idx, output_idx, 23), vp_comm_vente)
                    cuda.atomic.add(output_results, (account_idx, output_idx, 24), vp_primes_garanties)
                    cuda.atomic.add(output_results, (account_idx, output_idx, 25), vp_primes_variables)
                    cuda.atomic.add(output_results, (account_idx, output_idx, 26), vp_frais_fixes)
                    cuda.atomic.add(output_results, (account_idx, output_idx, 27), vp_hon_gest)
                    cuda.atomic.add(output_results, (account_idx, output_idx, 28), vp_comm_maintien)
                    cuda.atomic.add(output_results, (account_idx, output_idx, 29), vp_prest_ech)
                    cuda.atomic.add(output_results, (account_idx, output_idx, 30), vp_prest_mrv)
                    cuda.atomic.add(output_results, (account_idx, output_idx, 31), vp_prest_deces)
                    cuda.atomic.add(output_results, (account_idx, output_idx, 32), vp_valeur_marchande)
                    # VP Cushions
                    cuda.atomic.add(output_results, (account_idx, output_idx, 33), vp_passif_redresse)
                    cuda.atomic.add(output_results, (account_idx, output_idx, 34), vp_coussin_credit)
                    cuda.atomic.add(output_results, (account_idx, output_idx, 35), vp_coussin_marche)
                    cuda.atomic.add(output_results, (account_idx, output_idx, 36), vp_coussin_depense)
                    cuda.atomic.add(output_results, (account_idx, output_idx, 37), vp_coussin_decheance)
                    cuda.atomic.add(output_results, (account_idx, output_idx, 38), vp_coussin_mortalite)
                    cuda.atomic.add(output_results, (account_idx, output_idx, 39), vp_coussin_depot)

                    output_idx += 1
            else:
                # PRODUCTION MODE: Accumulate VP values only (no timestep detail)
                # Sum up VP values across all timesteps within this thread
                total_vp_frais_acquis += vp_frais_acquis
                total_vp_comm_vente += vp_comm_vente
                total_vp_primes_garanties += vp_primes_garanties
                total_vp_primes_variables += vp_primes_variables
                total_vp_frais_fixes += vp_frais_fixes
                total_vp_hon_gest += vp_hon_gest
                total_vp_comm_maintien += vp_comm_maintien
                total_vp_prest_ech += vp_prest_ech
                total_vp_prest_mrv += vp_prest_mrv
                total_vp_prest_deces += vp_prest_deces
                total_vp_valeur_marchande += vp_valeur_marchande
                total_vp_passif_redresse += vp_passif_redresse
                total_vp_coussin_credit += vp_coussin_credit
                total_vp_coussin_marche += vp_coussin_marche
                total_vp_coussin_depense += vp_coussin_depense
                total_vp_coussin_decheance += vp_coussin_decheance
                total_vp_coussin_mortalite += vp_coussin_mortalite
                total_vp_coussin_depot += vp_coussin_depot
    
    # ============= STEP 16: WRITE FINAL VP TOTALS (PRODUCTION MODE) =============
    # After finishing all timesteps, write one summary row with VP totals
    if not output_all_timesteps:
        # Only write ID fields once (scenario 0)
        if scenario_idx == 0:
            output_results[account_idx, 0, 0] = float(ID_COMPTE)
            output_results[account_idx, 0, 1] = 0.0  # SCN_EVAL not applicable
            output_results[account_idx, 0, 2] = 0.0  # AN_EVAL not applicable (summary)
            output_results[account_idx, 0, 3] = 0.0  # MOIS_EVAL not applicable (summary)
        
        # Atomically write VP totals (aggregated across scenarios)
        # Skip non-VP fields (indices 4-21) - set them to 0 or don't write
        cuda.atomic.add(output_results, (account_idx, 0, 22), total_vp_frais_acquis)
        cuda.atomic.add(output_results, (account_idx, 0, 23), total_vp_comm_vente)
        cuda.atomic.add(output_results, (account_idx, 0, 24), total_vp_primes_garanties)
        cuda.atomic.add(output_results, (account_idx, 0, 25), total_vp_primes_variables)
        cuda.atomic.add(output_results, (account_idx, 0, 26), total_vp_frais_fixes)
        cuda.atomic.add(output_results, (account_idx, 0, 27), total_vp_hon_gest)
        cuda.atomic.add(output_results, (account_idx, 0, 28), total_vp_comm_maintien)
        cuda.atomic.add(output_results, (account_idx, 0, 29), total_vp_prest_ech)
        cuda.atomic.add(output_results, (account_idx, 0, 30), total_vp_prest_mrv)
        cuda.atomic.add(output_results, (account_idx, 0, 31), total_vp_prest_deces)
        cuda.atomic.add(output_results, (account_idx, 0, 32), total_vp_valeur_marchande)
        cuda.atomic.add(output_results, (account_idx, 0, 33), total_vp_passif_redresse)
        cuda.atomic.add(output_results, (account_idx, 0, 34), total_vp_coussin_credit)
        cuda.atomic.add(output_results, (account_idx, 0, 35), total_vp_coussin_marche)
        cuda.atomic.add(output_results, (account_idx, 0, 36), total_vp_coussin_depense)
        cuda.atomic.add(output_results, (account_idx, 0, 37), total_vp_coussin_decheance)
        cuda.atomic.add(output_results, (account_idx, 0, 38), total_vp_coussin_mortalite)
        cuda.atomic.add(output_results, (account_idx, 0, 39), total_vp_coussin_depot)


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
# TWO-PASS NESTED STOCHASTIC ORCHESTRATOR
# =============================================================================

def run_projection_gpu_nested(
        data_path: Path, 
        output_path: Path, 
        nb_an_projection: int,
        nb_ext_scenarios: int,
        nb_int_scenarios: int,
        shock_capital_pct: float = 0.35,
        max_accounts: int = None,
        threads_per_block=(16, 16),
        use_pinned_memory=True,
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
    Run GPU-accelerated nested stochastic projection using Two-Pass architecture.
    
    Architecture:
    - Kernel A (Generator): Runs external scenarios, outputs state tensors to VRAM
    - Kernel B (Valuator): Reads states, runs internal scenarios, outputs reserves & capital
    
    Args:
        data_path: Path to input CSV files
        output_path: Path for output files
        nb_an_projection: Number of years to project (external)
        nb_ext_scenarios: Number of external (real-world) scenarios
        nb_int_scenarios: Number of internal (risk-neutral) scenarios per node
        shock_capital_pct: Capital shock percentage (e.g., 0.35 = 35% shock)
        max_accounts: Maximum number of accounts (for testing)
        threads_per_block: CUDA block dimensions
        use_pinned_memory: Use pinned memory for faster transfers
        ... (other paths for custom data files)
    
    Returns:
        Dictionary with results including reserves and capital requirements
    """
    start_time = datetime.now()
    print(f"Starting NESTED STOCHASTIC GPU projection at {start_time}")
    print("=" * 80)
    print(f"Architecture: Two-Pass (Generator → Valuator)")
    print(f"External scenarios: {nb_ext_scenarios}")
    print(f"Internal scenarios per node: {nb_int_scenarios}")
    print(f"Capital shock: {shock_capital_pct*100:.1f}%")
    print("=" * 80)
    
    # Check GPU availability
    try:
        if not cuda.is_available():
            raise RuntimeError("CUDA is not available")
        
        gpu = cuda.get_current_device()
        print(f"GPU Device: {gpu.name.decode()}")
        
        try:
            free_mem, total_mem = cuda.current_context().get_memory_info()
            print(f"GPU Memory: {free_mem / 1024**3:.2f} GB free / {total_mem / 1024**3:.2f} GB total")
        except NotImplementedError:
            print(f"GPU Memory: Information not available (using RMM allocator)")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize GPU: {e}")

    # Update config
    CONFIG['NB_AN_PROJECTION'] = nb_an_projection
    CONFIG['NB_SC'] = nb_ext_scenarios

    # Load data
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

    if max_accounts:
        data['population'] = data['population'].head(max_accounts)

    n_accounts = len(data['population'])
    print(f"\nPreparing {n_accounts} accounts for GPU processing...")

    # Prepare account data
    all_account_data, _ = prepare_account_data(data['population'])
    print("✓ Account data prepared")

    # Create GPU lookup tables
    print("\nCreating GPU lookup tables...")
    mortality_lookup = create_gpu_mortality_lookup(data['mortalite'])
    (forward_rate, ajust_forward, rend_dex, rend_mm, rend_tsx,
     rend_sp500, rend_eafe) = create_gpu_returns_lookup(data['rendements'])
    min_ferr_lookup = create_gpu_min_ferr_lookup(data['min_ferr'])
    lapse_part_min, lapse_part_max = create_gpu_lapse_part_lookup(data['tx_lapse_part'])
    lapse_tot_min, lapse_tot_max, lapse_tot_fact = create_gpu_lapse_tot_lookup(data['tx_lapse_tot'])
    (deposits_pc, deposits_var, deposits_age_max,
     deposits_i_even) = create_gpu_deposits_lookup(data['depots_futurs'])
    fees_lookup = create_gpu_fees_lookup(data['frais_admin'])
    (acq_vente_rf, acq_vente_ac, acq_maintien_rf, acq_maintien_ac,
     acq_frais_ac, acq_frais_rf) = create_gpu_acquisition_lookup(data['acquisition'])
    print("✓ All GPU lookup tables created")

    # For risk-neutral scenarios, create simplified return tables
    # In practice, these would be calibrated to market prices
    print("\nCreating risk-neutral scenario tables...")
    rn_forward_rate = np.full((nb_int_scenarios, nb_an_projection), 0.03, dtype=np.float32)
    rn_rend_dex = np.full((nb_int_scenarios, nb_an_projection), 0.025, dtype=np.float32)
    rn_rend_mm = np.full((nb_int_scenarios, nb_an_projection), 0.02, dtype=np.float32)
    rn_rend_tsx = np.full((nb_int_scenarios, nb_an_projection), 0.035, dtype=np.float32)
    rn_rend_sp500 = np.full((nb_int_scenarios, nb_an_projection), 0.035, dtype=np.float32)
    rn_rend_eafe = np.full((nb_int_scenarios, nb_an_projection), 0.03, dtype=np.float32)
    print("✓ Risk-neutral tables created")

    # Calculate memory requirements
    print("\nCalculating memory requirements...")
    
    # State tensor: (Batch, Ext_Scenarios, Years, STATE_SIZE)
    state_mem_per_account = nb_ext_scenarios * nb_an_projection * STATE_SIZE * 4  # float32
    
    # Cashflow tensor: (Batch, Ext_Scenarios, Years, 1)
    cf_mem_per_account = nb_ext_scenarios * nb_an_projection * 1 * 4
    
    # Metrics tensor: (Batch, Ext_Scenarios, Years, 2) - Reserve & Capital
    metrics_mem_per_account = nb_ext_scenarios * nb_an_projection * 2 * 4
    
    total_mem_per_account = (state_mem_per_account + cf_mem_per_account + 
                             metrics_mem_per_account + all_account_data.shape[1] * 4)
    
    print(f"  State tensor per account: {state_mem_per_account / 1024**2:.2f} MB")
    print(f"  Total memory per account: {total_mem_per_account / 1024**2:.2f} MB")
    
    # Calculate batch size (conservative for nested scenarios)
    try:
        free_mem, total_mem = cuda.current_context().get_memory_info()
        available_mem = free_mem * 0.7  # Conservative for nested scenarios
    except NotImplementedError:
        available_mem = 12 * 1024**3  # Assume 12 GB available
    
    batch_size = max(1, int(available_mem // total_mem_per_account))
    batch_size = min(batch_size, n_accounts)
    num_batches = (n_accounts + batch_size - 1) // batch_size
    
    print(f"  Batch size: {batch_size} accounts")
    print(f"  Total batches: {num_batches}")

    # Copy lookup tables to GPU
    print("\nCopying lookup tables to GPU...")
    d_mortality = cuda.to_device(mortality_lookup)
    d_forward_rate = cuda.to_device(forward_rate)
    d_ajust_forward = cuda.to_device(ajust_forward)
    d_rend_dex = cuda.to_device(rend_dex)
    d_rend_mm = cuda.to_device(rend_mm)
    d_rend_tsx = cuda.to_device(rend_tsx)
    d_rend_sp500 = cuda.to_device(rend_sp500)
    d_rend_eafe = cuda.to_device(rend_eafe)
    d_min_ferr = cuda.to_device(min_ferr_lookup)
    d_lapse_part_min = cuda.to_device(lapse_part_min)
    d_lapse_part_max = cuda.to_device(lapse_part_max)
    d_lapse_tot_min = cuda.to_device(lapse_tot_min)
    d_lapse_tot_max = cuda.to_device(lapse_tot_max)
    d_lapse_tot_fact = cuda.to_device(lapse_tot_fact)
    d_deposits_pc = cuda.to_device(deposits_pc)
    d_deposits_var = cuda.to_device(deposits_var)
    d_deposits_age_max = cuda.to_device(deposits_age_max)
    d_deposits_i_even = cuda.to_device(deposits_i_even)
    d_fees = cuda.to_device(fees_lookup)
    d_acq_vente_rf = cuda.to_device(acq_vente_rf)
    d_acq_vente_ac = cuda.to_device(acq_vente_ac)
    d_acq_maintien_rf = cuda.to_device(acq_maintien_rf)
    d_acq_maintien_ac = cuda.to_device(acq_maintien_ac)
    d_acq_frais_ac = cuda.to_device(acq_frais_ac)
    d_acq_frais_rf = cuda.to_device(acq_frais_rf)
    
    # Risk-neutral tables
    d_rn_forward_rate = cuda.to_device(rn_forward_rate)
    d_rn_rend_dex = cuda.to_device(rn_rend_dex)
    d_rn_rend_mm = cuda.to_device(rn_rend_mm)
    d_rn_rend_tsx = cuda.to_device(rn_rend_tsx)
    d_rn_rend_sp500 = cuda.to_device(rn_rend_sp500)
    d_rn_rend_eafe = cuda.to_device(rn_rend_eafe)
    print("✓ Lookup tables on GPU")

    # Process batches
    print("\n" + "=" * 80)
    print("RUNNING TWO-PASS NESTED STOCHASTIC PROJECTION")
    print("=" * 80)
    
    all_reserves = []
    all_capital = []
    
    for i in range(num_batches):
        batch_start = datetime.now()
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_accounts)
        current_batch_size = end_idx - start_idx
        
        logger.info(f"\n--- Batch {i+1}/{num_batches} (Accounts {start_idx}-{end_idx-1}) ---")
        
        # Prepare batch data
        batch_account_data = np.ascontiguousarray(all_account_data[start_idx:end_idx])
        d_batch_accounts = cuda.to_device(batch_account_data)
        
        # Allocate state and cashflow tensors (bridge between kernels)
        d_states = cuda.device_array(
            (current_batch_size, nb_ext_scenarios, nb_an_projection, STATE_SIZE),
            dtype=np.float32
        )
        d_cashflows = cuda.device_array(
            (current_batch_size, nb_ext_scenarios, nb_an_projection, 1),
            dtype=np.float32
        )
        d_metrics = cuda.device_array(
            (current_batch_size, nb_ext_scenarios, nb_an_projection, 2),
            dtype=np.float32
        )
        
        # === KERNEL A: EXTERNAL GENERATOR ===
        logger.info("  Launching Kernel A (External Generator)...")
        blocks_x = (current_batch_size + threads_per_block[0] - 1) // threads_per_block[0]
        blocks_y = (nb_ext_scenarios + threads_per_block[1] - 1) // threads_per_block[1]
        grid_A = (blocks_x, blocks_y)
        
        kernel_a_start = datetime.now()
        external_generator_kernel[grid_A, threads_per_block](
            d_batch_accounts,
            nb_ext_scenarios, nb_an_projection, CONFIG['FREQ_EVAL'],
            d_mortality,
            d_forward_rate, d_ajust_forward, d_rend_dex, d_rend_mm, d_rend_tsx, d_rend_sp500, d_rend_eafe,
            d_min_ferr, d_lapse_part_min, d_lapse_part_max,
            d_lapse_tot_min, d_lapse_tot_max, d_lapse_tot_fact,
            d_deposits_pc, d_deposits_var, d_deposits_age_max, d_deposits_i_even,
            d_fees,
            d_acq_vente_rf, d_acq_vente_ac, d_acq_maintien_rf, d_acq_maintien_ac,
            d_acq_frais_ac, d_acq_frais_rf,
            d_states,
            d_cashflows
        )
        cuda.synchronize()
        kernel_a_time = (datetime.now() - kernel_a_start).total_seconds()
        logger.info(f"  Kernel A complete: {kernel_a_time:.2f}s")
        
        # === KERNEL B: NESTED VALUATOR ===
        logger.info("  Launching Kernel B (Nested Valuator)...")
        total_nodes = current_batch_size * nb_ext_scenarios * nb_an_projection
        threads_per_block_B = 256
        blocks_B = (total_nodes + threads_per_block_B - 1) // threads_per_block_B
        
        kernel_b_start = datetime.now()
        nested_valuation_kernel[blocks_B, threads_per_block_B](
            d_states,
            d_batch_accounts,
            nb_int_scenarios,
            nb_an_projection,  # Internal horizon (run-off)
            shock_capital_pct,
            d_rn_forward_rate,
            d_rn_rend_dex, d_rn_rend_mm, d_rn_rend_tsx, d_rn_rend_sp500, d_rn_rend_eafe,
            d_mortality,
            d_metrics
        )
        cuda.synchronize()
        kernel_b_time = (datetime.now() - kernel_b_start).total_seconds()
        logger.info(f"  Kernel B complete: {kernel_b_time:.2f}s")
        
        # Copy results back
        logger.info("  Copying results to CPU...")
        h_metrics = d_metrics.copy_to_host()
        
        # Process metrics (average across scenarios and years for summary)
        # Shape: (current_batch_size, nb_ext_scenarios, nb_an_projection, 2)
        # Average across external scenarios and years to get per-account metrics
        batch_reserves = h_metrics[:, :, :, 0].mean(axis=(1, 2))  # Average over scenarios and years
        batch_capital = h_metrics[:, :, :, 1].mean(axis=(1, 2))
        
        all_reserves.extend(batch_reserves)
        all_capital.extend(batch_capital)
        
        # Cleanup
        del d_batch_accounts, d_states, d_cashflows, d_metrics, h_metrics
        gc.collect()
        cuda.synchronize()
        
        batch_time = (datetime.now() - batch_start).total_seconds()
        logger.info(f"  Batch complete: {batch_time:.2f}s (KernelA: {kernel_a_time:.2f}s, KernelB: {kernel_b_time:.2f}s)")
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'ID_COMPTE': data['population']['ID_COMPTE'].values[:n_accounts],
        'RESERVE_BE': all_reserves,
        'CAPITAL_REQ': all_capital,
        'SCR': [cap - res for res, cap in zip(all_reserves, all_capital)]
    })
    
    # Save results
    output_path.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path / "NESTED_STOCHASTIC_RESULTS.csv", index=False, sep=';')
    
    # Print summary
    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()
    
    print("\n" + "=" * 80)
    print("NESTED STOCHASTIC PROJECTION COMPLETE")
    print("=" * 80)
    print(f"Total time: {total_duration:.2f}s ({total_duration/60:.2f} minutes)")
    print(f"Accounts processed: {n_accounts}")
    print(f"External scenarios: {nb_ext_scenarios}")
    print(f"Internal scenarios per node: {nb_int_scenarios}")
    print(f"Total nested simulations: {n_accounts * nb_ext_scenarios * nb_an_projection * nb_int_scenarios:,}")
    print(f"\nResults Summary:")
    print(f"  Total Best Estimate Reserve: ${results_df['RESERVE_BE'].sum():,.2f}")
    print(f"  Total Capital Requirement:   ${results_df['CAPITAL_REQ'].sum():,.2f}")
    print(f"  Total SCR (Capital - Reserve): ${results_df['SCR'].sum():,.2f}")
    print(f"\n  Average per account:")
    print(f"    Reserve: ${results_df['RESERVE_BE'].mean():,.2f}")
    print(f"    Capital: ${results_df['CAPITAL_REQ'].mean():,.2f}")
    print(f"    SCR:     ${results_df['SCR'].mean():,.2f}")
    print("=" * 80)
    
    return {
        'results': results_df,
        'total_duration': total_duration
    }


# =============================================================================
# SCRIPT ENTRY POINT (MODIFIED)
# =============================================================================

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Run GPU-accelerated actuarial projections',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard projection (Tier 1 - Cashflows & VP)
  python gpu.py --mode standard --max-accounts 10000
  
  # Nested stochastic (Tier 2 & 3 - Reserves & Capital)
  python gpu.py --mode nested --ext-scenarios 100 --int-scenarios 500 --max-accounts 1000
  
  # Debug mode
  python gpu.py --mode standard --debug-account 12345
        """
    )
    
    # Mode selection
    parser.add_argument('--mode', type=str, choices=['standard', 'nested'], default='standard',
                       help='Projection mode: "standard" for cashflows/VP, "nested" for reserves/capital (default: standard)')
    
    # Common parameters
    parser.add_argument('--debug-account', type=int, default=None,
                       help='Account ID to show detailed results (standard mode only)')
    parser.add_argument('--debug-scenario', type=int, default=None,
                       help='Scenario number (ignored - showing scenario-averaged results)')
    parser.add_argument('--max-accounts', type=int, default=None,
                       help='Maximum number of accounts to process (for testing)')
    parser.add_argument('--years', type=int, default=100,
                       help='Number of years to project (default: 100)')
    
    # Standard mode parameters
    parser.add_argument('--scenarios', type=int, default=100,
                       help='Number of scenarios for standard mode (default: 100)')
    
    # Nested mode parameters
    parser.add_argument('--ext-scenarios', type=int, default=100,
                       help='Number of external (real-world) scenarios for nested mode (default: 100)')
    parser.add_argument('--int-scenarios', type=int, default=500,
                       help='Number of internal (risk-neutral) scenarios per node for nested mode (default: 500)')
    parser.add_argument('--shock', type=float, default=0.35,
                       help='Capital shock percentage for nested mode (default: 0.35 = 35%%)')
    
    args = parser.parse_args()
    
    try:
        if not cuda.is_available():
            print("ERROR: CUDA is not available. Please check your GPU setup.")
            exit(1)

        print(f"CUDA Device: {cuda.get_current_device().name}")
        
        DATA_PATH = HERE.joinpath("data_in")
        OUTPUT_PATH = HERE.joinpath("data_out_gpu")

        # =============================================================================
        # NESTED MODE (New - Two Pass)
        # =============================================================================

        print("\n" + "=" * 80)
        print("RUNNING NESTED STOCHASTIC MODE (Tier 2 & 3: Reserves & Capital)")
        print("=" * 80)

        if args.debug_account is not None:
            print("\n⚠️  Warning: --debug-account is not supported in nested mode (ignored)")
            print()

        results = run_projection_gpu_nested(
            data_path=DATA_PATH,
            output_path=OUTPUT_PATH,
            nb_an_projection=args.years,
            nb_ext_scenarios=args.ext_scenarios,
            nb_int_scenarios=args.int_scenarios,
            shock_capital_pct=args.shock,
            max_accounts=args.max_accounts,
            threads_per_block=(16, 16)
        )

        if results:
            print("\n" + "=" * 80)
            print("NESTED STOCHASTIC RESULTS")
            print("=" * 80)
            print("\nTop 10 accounts by SCR:")
            print(results['results'].nlargest(10, 'SCR')[['ID_COMPTE', 'RESERVE_BE', 'CAPITAL_REQ', 'SCR']])

            print("\nSummary Statistics:")
            print(results['results'][['RESERVE_BE', 'CAPITAL_REQ', 'SCR']].describe())

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()
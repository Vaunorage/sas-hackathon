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
        # Debug parameters
        debug_ext_scenario,
        debug_account,
        # Output arrays
        output_states,     # Shape: (Batch_Size, n_scenarios, n_years, STATE_SIZE)
        output_cashflows,  # Shape: (Batch_Size, n_scenarios, n_years, CF_SIZE)
        output_ext_debug   # Shape: (n_years, 12) for debug logging
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
    
    # Check if this is the debug thread for external scenario logging
    is_debug_ext = (ID_COMPTE == debug_account and scenario_idx == debug_ext_scenario)
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
                
                # Save debug info for external scenario if requested
                if is_debug_ext and output_year_idx < output_ext_debug.shape[0]:
                    output_ext_debug[output_year_idx, 0] = float(output_year_idx)  # Year
                    output_ext_debug[output_year_idx, 1] = MT_VM_AV_RETRAIT_FRAIS  # VM before fees
                    output_ext_debug[output_year_idx, 2] = r_dex  # DEX return
                    output_ext_debug[output_year_idx, 3] = MT_VM_PROJ  # VM after fees
                    output_ext_debug[output_year_idx, 4] = PC_RFG  # Fee rate
                    output_ext_debug[output_year_idx, 5] = float(age)  # Age
                    output_ext_debug[output_year_idx, 6] = TX_SURVIE  # Survival probability
                    output_ext_debug[output_year_idx, 7] = qx  # Mortality rate
                    output_ext_debug[output_year_idx, 8] = lapse  # Lapse rate
                    output_ext_debug[output_year_idx, 9] = MT_DEX_PROJ  # DEX value
                    output_ext_debug[output_year_idx, 10] = MT_SP500_PROJ  # SP500 value
                    output_ext_debug[output_year_idx, 11] = flux_net  # Net cashflow

                output_year_idx += 1


@cuda.jit
def nested_valuation_kernel_debug(
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
        # Debug parameters
        debug_int_scenario,
        debug_ext_scenario,
        debug_ext_year,
        # Output arrays
        output_metrics,       # Shape: (Batch_Size, n_ext_scenarios, n_years, 2) -> [Reserve, Capital]
        output_debug          # Shape: (n_internal_years, 10) -> Debug details for specified internal scenario
):
    """
    KERNEL B: NESTED VALUATOR WITH DEBUG (Tier 2 & 3)
    
    Same as nested_valuation_kernel but captures detailed internal scenario calculations
    for debugging purposes.
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
    ID_COMPTE = int(acc[0])
    PC_RFG = acc[34]
    
    # Check if this is the debug node
    is_debug_node = (acc_idx == 0 and scn_idx == debug_ext_scenario and year_idx == debug_ext_year)
    
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
            
            vm_before_return = curr_vm
            
            # Apply Risk Neutral Return (simplified)
            if (i_int < rn_rend_dex.shape[0] and
                    t_int < rn_rend_dex.shape[1]):
                r_rn = rn_rend_dex[i_int, t_int]
            else:
                r_rn = 0.02
            
            curr_vm *= math.exp(r_rn)
            
            # Deduct fees
            fees = curr_vm * PC_RFG
            curr_vm -= fees
            
            # Calculate net flux
            flux = fees
            
            # Discount (using risk-neutral discount factor)
            if (i_int < rn_forward_rate.shape[0] and
                    t_int < rn_forward_rate.shape[1]):
                fwd = rn_forward_rate[i_int, t_int]
            else:
                fwd = 0.02
            
            df = math.exp(-fwd * (t_int + 1))
            pv_flux = flux * df
            pv_path += pv_flux
            
            # Save debug info for specified internal scenario
            if is_debug_node and i_int == debug_int_scenario and t_int < output_debug.shape[0]:
                output_debug[t_int, 0] = float(t_int)  # Year
                output_debug[t_int, 1] = vm_before_return  # VM before return
                output_debug[t_int, 2] = r_rn  # Return rate
                output_debug[t_int, 3] = curr_vm + fees  # VM after return
                output_debug[t_int, 4] = fees  # Fees
                output_debug[t_int, 5] = curr_vm  # VM after fees
                output_debug[t_int, 6] = flux  # Cashflow
                output_debug[t_int, 7] = fwd  # Discount rate
                output_debug[t_int, 8] = df  # Discount factor
                output_debug[t_int, 9] = pv_flux  # PV of cashflow
            
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
        progress_callback: Optional[callable] = None,
        debug_memory: bool = False,
        debug_int_scenario: Optional[int] = None,
        debug_ext_scenario: Optional[int] = None,
        debug_ext_year: Optional[int] = None,
        debug_int_year: Optional[int] = None,
        debug_account: Optional[int] = None):
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
    print(f"Memory debugging: {'ENABLED' if debug_memory else 'DISABLED'}")
    print("=" * 80)
    
    # Check GPU availability
    try:
        if not cuda.is_available():
            raise RuntimeError("CUDA is not available")
        
        gpu = cuda.get_current_device()
        print(f"GPU Device: {gpu.name.decode()}")
        
        # Try to get memory info and provide guidance if not available
        free_mem, total_mem, used_mem = log_gpu_memory_debug("Initialization", verbose=True)
        if free_mem is None:
            print("")
            print("=" * 80)
            print("NOTE: GPU memory cannot be queried programmatically (RMM allocator)")
            print("To monitor GPU memory usage during execution:")
            print("  1. Open another terminal")
            print("  2. Run: watch -n 1 nvidia-smi")
            print("  3. Monitor GPU memory usage in real-time")
            print("=" * 80)
            print("")
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
    log_gpu_memory_debug("After data load", verbose=debug_memory)

    if max_accounts:
        data['population'] = data['population'].head(max_accounts)

    n_accounts = len(data['population'])
    print(f"\nPreparing {n_accounts} accounts for GPU processing...")

    # Prepare account data
    all_account_data, _ = prepare_account_data(data['population'])
    print("✓ Account data prepared")
    log_gpu_memory_debug("After account prep", verbose=debug_memory)

    # Create GPU lookup tables
    print("\nCreating GPU lookup tables...")
    if debug_memory:
        print("  Creating mortality lookup...")
    mortality_lookup = create_gpu_mortality_lookup(data['mortalite'])
    if debug_memory:
        print(f"    Mortality table size: {mortality_lookup.nbytes / 1024**2:.2f} MB")
    if debug_memory:
        print("  Creating returns lookup...")
    (forward_rate, ajust_forward, rend_dex, rend_mm, rend_tsx,
     rend_sp500, rend_eafe) = create_gpu_returns_lookup(data['rendements'])
    if debug_memory:
        returns_size = sum(x.nbytes for x in [forward_rate, ajust_forward, rend_dex, rend_mm, rend_tsx, rend_sp500, rend_eafe])
        print(f"    Returns tables size: {returns_size / 1024**2:.2f} MB")
    min_ferr_lookup = create_gpu_min_ferr_lookup(data['min_ferr'])
    lapse_part_min, lapse_part_max = create_gpu_lapse_part_lookup(data['tx_lapse_part'])
    lapse_tot_min, lapse_tot_max, lapse_tot_fact = create_gpu_lapse_tot_lookup(data['tx_lapse_tot'])
    (deposits_pc, deposits_var, deposits_age_max,
     deposits_i_even) = create_gpu_deposits_lookup(data['depots_futurs'])
    fees_lookup = create_gpu_fees_lookup(data['frais_admin'])
    (acq_vente_rf, acq_vente_ac, acq_maintien_rf, acq_maintien_ac,
     acq_frais_ac, acq_frais_rf) = create_gpu_acquisition_lookup(data['acquisition'])
    
    if debug_memory:
        all_lookups_size = sum(x.nbytes for x in [
            mortality_lookup, forward_rate, ajust_forward, rend_dex, rend_mm, rend_tsx,
            rend_sp500, rend_eafe, min_ferr_lookup, lapse_part_min, lapse_part_max,
            lapse_tot_min, lapse_tot_max, lapse_tot_fact, deposits_pc, deposits_var,
            deposits_age_max, deposits_i_even, fees_lookup, acq_vente_rf, acq_vente_ac,
            acq_maintien_rf, acq_maintien_ac, acq_frais_ac, acq_frais_rf
        ])
        print(f"  Total CPU lookup tables size: {all_lookups_size / 1024**2:.2f} MB")
    
    print("✓ All GPU lookup tables created")
    log_gpu_memory_debug("After lookup table creation", verbose=debug_memory)

    # For risk-neutral scenarios, create simplified return tables
    # In practice, these would be calibrated to market prices
    print("\nCreating risk-neutral scenario tables...")
    rn_forward_rate = np.full((nb_int_scenarios, nb_an_projection), 0.03, dtype=np.float32)
    rn_rend_dex = np.full((nb_int_scenarios, nb_an_projection), 0.025, dtype=np.float32)
    rn_rend_mm = np.full((nb_int_scenarios, nb_an_projection), 0.02, dtype=np.float32)
    rn_rend_tsx = np.full((nb_int_scenarios, nb_an_projection), 0.035, dtype=np.float32)
    rn_rend_sp500 = np.full((nb_int_scenarios, nb_an_projection), 0.035, dtype=np.float32)
    rn_rend_eafe = np.full((nb_int_scenarios, nb_an_projection), 0.03, dtype=np.float32)
    
    if debug_memory:
        rn_size = sum(x.nbytes for x in [rn_forward_rate, rn_rend_dex, rn_rend_mm, rn_rend_tsx, rn_rend_sp500, rn_rend_eafe])
        print(f"  Risk-neutral tables size: {rn_size / 1024**2:.2f} MB")
    
    print("✓ Risk-neutral tables created")
    log_gpu_memory_debug("After RN tables creation", verbose=debug_memory)

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
    
    # Estimate lookup table memory overhead (always resident on GPU)
    lookup_overhead = 0
    # Real-world scenario tables: 6 assets × (nb_ext_scenarios, ~40 years, 12 months) × 4 bytes
    lookup_overhead += 6 * nb_ext_scenarios * nb_an_projection * 12 * 4
    # Risk-neutral tables: 6 assets × (nb_int_scenarios, years) × 4 bytes
    lookup_overhead += 6 * nb_int_scenarios * nb_an_projection * 4
    # Mortality, lapse, deposits, fees, acquisition tables (conservative estimate)
    lookup_overhead += 150 * 1024**2  # ~150 MB
    
    print(f"  State tensor per account: {state_mem_per_account / 1024**2:.2f} MB")
    print(f"  Total memory per account: {total_mem_per_account / 1024**2:.2f} MB")
    print(f"  Lookup table overhead: {lookup_overhead / 1024**2:.2f} MB")
    
    # Calculate batch size (conservative for nested scenarios)
    try:
        free_mem, total_mem = cuda.current_context().get_memory_info()
        print(f"  GPU free memory: {free_mem / 1024**3:.2f} GB")
        print(f"  GPU total memory: {total_mem / 1024**3:.2f} GB")
        # Reserve memory for lookup tables and CUDA overhead
        available_mem = max(0, (free_mem - lookup_overhead) * 0.6)  # More conservative
    except NotImplementedError:
        print("  Warning: Cannot query GPU memory, using conservative estimate")
        available_mem = max(0, 12 * 1024**3 - lookup_overhead)  # Assume 12 GB total
    
    batch_size = max(1, int(available_mem // total_mem_per_account))
    batch_size = min(batch_size, n_accounts)
    num_batches = (n_accounts + batch_size - 1) // batch_size
    
    print(f"  Batch size: {batch_size} accounts")
    print(f"  Total batches: {num_batches}")

    # Copy lookup tables to GPU
    print("\nCopying lookup tables to GPU...")
    if debug_memory:
        print("  This will allocate permanent GPU memory for all lookups")
    log_gpu_memory_debug("Before copying lookups to GPU", verbose=debug_memory)
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
    log_gpu_memory_debug("After copying lookups to GPU", verbose=True)

    # Process batches
    print("\n" + "=" * 80)
    print("RUNNING TWO-PASS NESTED STOCHASTIC PROJECTION")
    print("=" * 80)
    
    all_reserves = []
    all_capital = []
    debug_output = None  # Store internal scenario debug output if requested
    external_debug_output = None  # Store external scenario debug output if requested
    debug_account_data = None  # Store full metrics for debug account if requested
    
    for i in range(num_batches):
        batch_start = datetime.now()
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_accounts)
        current_batch_size = end_idx - start_idx
        
        logger.info(f"\n--- Batch {i+1}/{num_batches} (Accounts {start_idx}-{end_idx-1}) ---")
        log_gpu_memory_debug(f"Batch {i+1} start", verbose=debug_memory)
        
        # Check available memory before allocation
        try:
            free_mem_before, _ = cuda.current_context().get_memory_info()
            estimated_batch_mem = current_batch_size * total_mem_per_account
            logger.info(f"  Free GPU memory: {free_mem_before / 1024**3:.2f} GB")
            logger.info(f"  Estimated batch memory: {estimated_batch_mem / 1024**3:.2f} GB")
            
            if estimated_batch_mem > free_mem_before * 0.9:
                raise RuntimeError(
                    f"Insufficient GPU memory for batch {i+1}. "
                    f"Need {estimated_batch_mem / 1024**3:.2f} GB but only "
                    f"{free_mem_before / 1024**3:.2f} GB available. "
                    f"Try reducing batch size or number of scenarios."
                )
        except NotImplementedError:
            pass  # Cannot check memory, proceed with caution
        
        # Prepare batch data
        batch_account_data = np.ascontiguousarray(all_account_data[start_idx:end_idx])
        if debug_memory:
            print(f"  Batch account data size: {batch_account_data.nbytes / 1024**2:.2f} MB")
        d_batch_accounts = cuda.to_device(batch_account_data)
        log_gpu_memory_debug(f"Batch {i+1} after account data copy", verbose=debug_memory)
        
        # Allocate state and cashflow tensors (bridge between kernels)
        if debug_memory:
            states_size = current_batch_size * nb_ext_scenarios * nb_an_projection * STATE_SIZE * 4
            cf_size = current_batch_size * nb_ext_scenarios * nb_an_projection * 1 * 4
            metrics_size = current_batch_size * nb_ext_scenarios * nb_an_projection * 2 * 4
            print(f"  Allocating state tensor: {states_size / 1024**2:.2f} MB")
            print(f"  Allocating cashflow tensor: {cf_size / 1024**2:.2f} MB")
            print(f"  Allocating metrics tensor: {metrics_size / 1024**2:.2f} MB")
            print(f"  Total batch allocation: {(states_size + cf_size + metrics_size) / 1024**2:.2f} MB")
        
        try:
            d_states = cuda.device_array(
                (current_batch_size, nb_ext_scenarios, nb_an_projection, STATE_SIZE),
                dtype=np.float32
            )
            log_gpu_memory_debug(f"Batch {i+1} after states allocation", verbose=debug_memory)
            
            d_cashflows = cuda.device_array(
                (current_batch_size, nb_ext_scenarios, nb_an_projection, 1),
                dtype=np.float32
            )
            log_gpu_memory_debug(f"Batch {i+1} after cashflows allocation", verbose=debug_memory)
            
            d_metrics = cuda.device_array(
                (current_batch_size, nb_ext_scenarios, nb_an_projection, 2),
                dtype=np.float32
            )
            log_gpu_memory_debug(f"Batch {i+1} after metrics allocation", verbose=debug_memory)
        except Exception as e:
            raise RuntimeError(
                f"Failed to allocate GPU memory for batch {i+1}. "
                f"This typically means GPU is out of memory. "
                f"Try reducing --max-accounts or --ext-scenarios. Original error: {e}"
            )
        
        # Allocate external debug output if requested
        d_ext_debug_output = None
        _debug_ext_scenario = debug_ext_scenario if debug_ext_scenario is not None else -1
        _debug_account = debug_account if debug_account is not None else -1
        
        if debug_account is not None and debug_ext_scenario is not None and i == 0:
            # Check if debug account is in this batch
            batch_start_account = start_idx + 1
            batch_end_account = end_idx
            if batch_start_account <= debug_account <= batch_end_account:
                logger.info(f"  DEBUG MODE: Capturing external scenario {debug_ext_scenario} for account {debug_account}")
                d_ext_debug_output = cuda.device_array((nb_an_projection, 12), dtype=np.float32)
        
        # Use dummy array if not debugging
        if d_ext_debug_output is None:
            d_ext_debug_output = cuda.device_array((1, 12), dtype=np.float32)
        
        # === KERNEL A: EXTERNAL GENERATOR ===
        logger.info("  Launching Kernel A (External Generator)...")
        log_gpu_memory_debug(f"Batch {i+1} before Kernel A", verbose=debug_memory)
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
            _debug_ext_scenario,
            _debug_account,
            d_states,
            d_cashflows,
            d_ext_debug_output
        )
        cuda.synchronize()
        kernel_a_time = (datetime.now() - kernel_a_start).total_seconds()
        logger.info(f"  Kernel A complete: {kernel_a_time:.2f}s")
        log_gpu_memory_debug(f"Batch {i+1} after Kernel A", verbose=debug_memory)
        
        # Capture external debug output if requested
        if debug_account is not None and debug_ext_scenario is not None and i == 0:
            batch_start_account = start_idx + 1
            batch_end_account = end_idx
            if batch_start_account <= debug_account <= batch_end_account:
                external_debug_output = d_ext_debug_output.copy_to_host()
                logger.info(f"  Captured external scenario debug data for account {debug_account}, scenario {debug_ext_scenario}")
        
        # === KERNEL B: NESTED VALUATOR ===
        logger.info("  Launching Kernel B (Nested Valuator)...")
        total_nodes = current_batch_size * nb_ext_scenarios * nb_an_projection
        if debug_memory:
            print(f"  Kernel B will process {total_nodes:,} nodes with {nb_int_scenarios} internal scenarios each")
        log_gpu_memory_debug(f"Batch {i+1} before Kernel B", verbose=debug_memory)
        threads_per_block_B = 256
        blocks_B = (total_nodes + threads_per_block_B - 1) // threads_per_block_B
        
        kernel_b_start = datetime.now()
        
        # Check if we're debugging an internal scenario
        if debug_int_scenario is not None and i == 0:  # Only in first batch
            # Set defaults if not provided
            _debug_ext_scenario = debug_ext_scenario if debug_ext_scenario is not None else 0
            _debug_ext_year = debug_ext_year if debug_ext_year is not None else 0
            
            logger.info(f"  DEBUG MODE: Capturing internal scenario {debug_int_scenario} "
                       f"(ext_scenario={_debug_ext_scenario}, year={_debug_ext_year})")
            
            # Allocate debug output array
            d_debug_output = cuda.device_array((nb_an_projection, 10), dtype=np.float32)
            
            nested_valuation_kernel_debug[blocks_B, threads_per_block_B](
                d_states,
                d_batch_accounts,
                nb_int_scenarios,
                nb_an_projection,  # Internal horizon (run-off)
                shock_capital_pct,
                d_rn_forward_rate,
                d_rn_rend_dex, d_rn_rend_mm, d_rn_rend_tsx, d_rn_rend_sp500, d_rn_rend_eafe,
                d_mortality,
                debug_int_scenario,
                _debug_ext_scenario,
                _debug_ext_year,
                d_metrics,
                d_debug_output
            )
            cuda.synchronize()
            
            # Copy debug output back to host
            h_debug_output = d_debug_output.copy_to_host()
            debug_output = h_debug_output  # Store for return
            
        else:
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
            h_debug_output = None
        
        kernel_b_time = (datetime.now() - kernel_b_start).total_seconds()
        logger.info(f"  Kernel B complete: {kernel_b_time:.2f}s")
        log_gpu_memory_debug(f"Batch {i+1} after Kernel B", verbose=debug_memory)
        
        # Copy results back
        logger.info("  Copying results to CPU...")
        copy_start = datetime.now()
        h_metrics = d_metrics.copy_to_host()
        copy_time = (datetime.now() - copy_start).total_seconds()
        if debug_memory:
            print(f"  Copy to CPU took {copy_time:.2f}s ({h_metrics.nbytes / 1024**2:.2f} MB)")
        log_gpu_memory_debug(f"Batch {i+1} after copy to CPU", verbose=debug_memory)
        
        # Process metrics (average across scenarios and years for summary)
        # Shape: (current_batch_size, nb_ext_scenarios, nb_an_projection, 2)
        # Average across external scenarios and years to get per-account metrics
        batch_reserves = h_metrics[:, :, :, 0].mean(axis=(1, 2))  # Average over scenarios and years
        batch_capital = h_metrics[:, :, :, 1].mean(axis=(1, 2))
        
        all_reserves.extend(batch_reserves)
        all_capital.extend(batch_capital)
        
        # Capture detailed metrics for debug account if requested
        if debug_account is not None:
            batch_start_account = start_idx + 1
            batch_end_account = end_idx
            if batch_start_account <= debug_account <= batch_end_account:
                account_batch_idx = debug_account - batch_start_account
                # Extract full metrics for this account: (nb_ext_scenarios, nb_an_projection, 2)
                debug_account_data = h_metrics[account_batch_idx]
                logger.info(f"  Captured debug data for account {debug_account} (batch index {account_batch_idx})")
        
        # Explicit cleanup to free GPU memory
        if debug_memory:
            print(f"  Starting memory cleanup for batch {i+1}...")
        log_gpu_memory_debug(f"Batch {i+1} before cleanup", verbose=debug_memory)
        
        del d_batch_accounts, d_states, d_cashflows, d_metrics
        cuda.synchronize()
        del h_metrics
        gc.collect()
        
        # Force memory pool cleanup if using RMM
        try:
            import rmm
            rmm.mr.get_current_device_resource().deallocate(0, 0)
            if debug_memory:
                print("  RMM memory pool cleanup successful")
        except (ImportError, AttributeError):
            pass  # RMM not available or no manual cleanup needed
        
        log_gpu_memory_debug(f"Batch {i+1} after cleanup", verbose=True)
        
        batch_time = (datetime.now() - batch_start).total_seconds()
        logger.info(f"  Batch complete: {batch_time:.2f}s (KernelA: {kernel_a_time:.2f}s, KernelB: {kernel_b_time:.2f}s)")
        
        if debug_memory:
            print(f"  Memory should be back to baseline after cleanup")
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'ID_COMPTE': data['population']['ID_COMPTE'].values[:n_accounts],
        'RESERVE_BE': all_reserves,
        'CAPITAL_REQ': all_capital,
        'SCR': [cap - res for res, cap in zip(all_reserves, all_capital)]
    })
    
    # Save results
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save main results CSV
    main_results_path = output_path / "NESTED_STOCHASTIC_RESULTS.csv"
    results_df.to_csv(main_results_path, index=False, sep=';')
    print(f"\n✓ Saved main results: {main_results_path}")
    
    # Save debug account data if requested
    if debug_account is not None and debug_account_data is not None:
        print(f"\nSaving detailed debug data for account {debug_account}...")
        
        # Create detailed breakdown: EXT_SCENARIO, YEAR, RESERVE, CAPITAL
        debug_rows = []
        for ext_scn in range(debug_account_data.shape[0]):
            for year in range(debug_account_data.shape[1]):
                debug_rows.append({
                    'ID_COMPTE': debug_account,
                    'EXT_SCENARIO': ext_scn,
                    'YEAR': year,
                    'RESERVE_BE': debug_account_data[ext_scn, year, 0],
                    'CAPITAL_REQ': debug_account_data[ext_scn, year, 1],
                    'SCR': debug_account_data[ext_scn, year, 1] - debug_account_data[ext_scn, year, 0]
                })
        
        debug_account_df = pd.DataFrame(debug_rows)
        debug_account_path = output_path / f"DEBUG_ACCOUNT_{debug_account}_NESTED_DETAILS.csv"
        debug_account_df.to_csv(debug_account_path, index=False, sep=';')
        print(f"✓ Saved account debug file: {debug_account_path}")
        print(f"  Contains {len(debug_account_df)} rows ({nb_ext_scenarios} scenarios × {nb_an_projection} years)")
        
        # Show summary for this account
        print(f"\nAccount {debug_account} Summary:")
        print(f"  Average Reserve (BE): ${debug_account_data[:, :, 0].mean():,.2f}")
        print(f"  Average Capital Req:  ${debug_account_data[:, :, 1].mean():,.2f}")
        print(f"  Average SCR:          ${(debug_account_data[:, :, 1] - debug_account_data[:, :, 0]).mean():,.2f}")
    
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
    
    # Display external scenario debug output if captured
    if external_debug_output is not None:
        print("\n" + "=" * 80)
        _debug_ext_scenario = debug_ext_scenario if debug_ext_scenario is not None else 0
        
        print(f"DEBUG: External Scenario {_debug_ext_scenario} Details")
        print(f"  Account: {debug_account}")
        print("=" * 80)
        
        # Create debug DataFrame
        ext_debug_df = pd.DataFrame(external_debug_output, columns=[
            'Year', 'VM_Before_Fees', 'DEX_Return', 'VM_After_Fees',
            'Fee_Rate', 'Age', 'Survival_Prob', 'Mortality_Rate',
            'Lapse_Rate', 'DEX_Value', 'SP500_Value', 'Net_Cashflow'
        ])
        
        # Filter out zero rows (policy terminated)
        ext_debug_df = ext_debug_df[ext_debug_df['VM_Before_Fees'] > 0]
        
        if len(ext_debug_df) > 0:
            print("\nExternal Scenario Projection:")
            print(ext_debug_df.to_string(index=False, float_format=lambda x: f'{x:,.4f}'))
            
            print(f"\nFinal VM: ${ext_debug_df['VM_After_Fees'].iloc[-1]:,.2f}")
            print(f"Final Age: {ext_debug_df['Age'].iloc[-1]:.0f}")
            print(f"Final Survival: {ext_debug_df['Survival_Prob'].iloc[-1]:.4f}")
            
            # Save external scenario debug output to CSV
            ext_debug_filename = f"DEBUG_EXT_SCENARIO_{_debug_ext_scenario}_ACCOUNT_{debug_account}.csv"
            ext_debug_csv_path = output_path / ext_debug_filename
            ext_debug_df.to_csv(ext_debug_csv_path, index=False, sep=';')
            print(f"\n✓ External scenario debug output saved: {ext_debug_csv_path}")
            print(f"  Contains {len(ext_debug_df)} rows with external projection details")
        else:
            print("\n(No data - policy may have terminated)")
        
        print("=" * 80)
    
    # Display internal scenario debug output if captured
    if debug_output is not None:
        print("\n" + "=" * 80)
        _debug_ext_scenario = debug_ext_scenario if debug_ext_scenario is not None else 0
        _debug_ext_year = debug_ext_year if debug_ext_year is not None else 0
        
        print(f"DEBUG: Internal Scenario {debug_int_scenario} Details")
        print(f"  External Scenario: {_debug_ext_scenario}")
        print(f"  External Year: {_debug_ext_year}")
        print("=" * 80)
        
        # Create debug DataFrame
        debug_df = pd.DataFrame(debug_output, columns=[
            'Year', 'VM_Before_Return', 'Return_Rate', 'VM_After_Return',
            'Fees', 'VM_After_Fees', 'Cashflow', 'Discount_Rate',
            'Discount_Factor', 'PV_Cashflow'
        ])
        
        # Filter out zero rows (policy terminated)
        debug_df = debug_df[debug_df['VM_Before_Return'] > 0]
        
        # Filter to specific internal year if requested
        if debug_int_year is not None:
            debug_df = debug_df[debug_df['Year'] == debug_int_year]
            if len(debug_df) == 0:
                print(f"\n(No data for internal year {debug_int_year} - policy may have terminated or invalid year)")
        
        if len(debug_df) > 0:
            print("\nInternal Scenario Projection:")
            print(debug_df.to_string(index=False, float_format=lambda x: f'{x:,.4f}'))
            
            print(f"\nTotal PV of Cashflows (Reserve): ${debug_df['PV_Cashflow'].sum():,.2f}")
            
            # Save debug output to CSV with comprehensive naming
            debug_filename = f"DEBUG_INT_SCENARIO_{debug_int_scenario}"
            if debug_ext_scenario is not None:
                debug_filename += f"_EXT_SCENARIO_{_debug_ext_scenario}"
            if debug_ext_year is not None:
                debug_filename += f"_EXT_YEAR_{_debug_ext_year}"
            if debug_int_year is not None:
                debug_filename += f"_INT_YEAR_{debug_int_year}"
            debug_filename += ".csv"
            
            debug_csv_path = output_path / debug_filename
            debug_df.to_csv(debug_csv_path, index=False, sep=';')
            print(f"\n✓ Debug output saved: {debug_csv_path}")
            print(f"  Contains {len(debug_df)} rows with internal projection details")
        else:
            print("\n(No data - policy may have terminated)")
        
        print("=" * 80)
    
    return {
        'results': results_df,
        'total_duration': total_duration,
        'debug_output': debug_output,
        'external_debug_output': external_debug_output
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
  
  # Debug external scenario (show detailed external scenario projection)
  python gpu.py --mode nested --ext-scenarios 10 --int-scenarios 100 --max-accounts 10 \\
      --debug-account 1 --debug-ext-scenario 0
  
  # Debug internal scenario (show detailed calculations for a specific internal scenario)
  python gpu.py --mode nested --ext-scenarios 10 --int-scenarios 100 --max-accounts 10 \\
      --debug-int-scenario 0 --debug-ext-scenario 0 --debug-ext-year 0
  
  # Debug with internal year filter (show only year 5 of the internal projection)
  python gpu.py --mode nested --ext-scenarios 10 --int-scenarios 100 --max-accounts 10 \\
      --debug-int-scenario 0 --debug-ext-scenario 0 --debug-ext-year 0 --debug-int-year 5
  
  # Debug mode (standard)
  python gpu.py --mode standard --debug-account 12345
        """
    )

    parser.add_argument('--max-accounts', type=int, default=2000,
                        help='Maximum number of accounts to process (for testing)')
    parser.add_argument('--years', type=int, default=100,
                        help='Number of years to project (default: 100)')
    parser.add_argument('--scenarios', type=int, default=100,
                        help='Number of scenarios for standard mode (default: 100)')

    # Nested mode parameters
    parser.add_argument('--ext-scenarios', type=int, default=100,
                        help='Number of external (real-world) scenarios for nested mode (default: 100)')
    parser.add_argument('--int-scenarios', type=int, default=100,
                        help='Number of internal (risk-neutral) scenarios per node for nested mode (default: 500)')
    parser.add_argument('--shock', type=float, default=0.35,
                        help='Capital shock percentage for nested mode (default: 0.35 = 35%%)')

    parser.add_argument('--debug-memory', action='store_true',
                        help='Enable detailed GPU memory debugging output')
    parser.add_argument('--debug-account', type=int, default=None,
                        help='Account ID to show detailed results (both modes). In nested mode, use with --debug-ext-scenario to log external scenario projection')
    parser.add_argument('--debug-ext-scenario', type=int, default=None,
                        help='External scenario number for debug (nested mode only, use with --debug-account for external projection or --debug-int-scenario for internal projection)')
    parser.add_argument('--debug-ext-year', type=int, default=None,
                        help='External year index for debug (nested mode only, used with --debug-int-scenario)')
    parser.add_argument('--debug-int-scenario', type=int, default=None,
                        help='Internal scenario number to show detailed calculations (nested mode only, e.g., 0 for first internal scenario)')
    parser.add_argument('--debug-int-year', type=int, default=None,
                        help='Specific internal year to display (nested mode only, requires --debug-int-scenario, default: None = show all years)')

    args = parser.parse_args()
    
    # Validate debug arguments
    # 1. If ext scenario/year are specified, int scenario is required
    if (args.debug_ext_scenario is not None or args.debug_ext_year is not None):
        if args.debug_int_scenario is None:
            parser.error("--debug-int-scenario is required when --debug-ext-scenario or --debug-ext-year are specified")
    
    # 2. If int year is specified, int scenario is required (can't filter without enabling debug)
    if args.debug_int_year is not None:
        if args.debug_int_scenario is None:
            parser.error("--debug-int-scenario is required when --debug-int-year is specified")
    
    # Set defaults for debug arguments if they're being used
    if args.debug_int_scenario is not None:
        if args.debug_ext_scenario is None:
            args.debug_ext_scenario = 0
        if args.debug_ext_year is None:
            args.debug_ext_year = 0
    
    try:
        if not cuda.is_available():
            print("ERROR: CUDA is not available. Please check your GPU setup.")
            exit(1)

        print(f"CUDA Device: {cuda.get_current_device().name}")
        
        DATA_PATH = HERE.joinpath("data_in")
        OUTPUT_PATH = HERE.joinpath("data_out_gpu")

        print("\n" + "=" * 80)
        print("RUNNING NESTED STOCHASTIC MODE (Tier 2 & 3: Reserves & Capital)")
        print("=" * 80)

        if args.debug_account is not None:
            print(f"\n🔍 DEBUG MODE: Will save detailed breakdown for account {args.debug_account}")
            print()

        results = run_projection_gpu_nested(
            data_path=DATA_PATH,
            output_path=OUTPUT_PATH,
            nb_an_projection=args.years,
            nb_ext_scenarios=args.ext_scenarios,
            nb_int_scenarios=args.int_scenarios,
            shock_capital_pct=args.shock,
            max_accounts=args.max_accounts,
            threads_per_block=(16, 16),
            debug_memory=args.debug_memory,
            debug_int_scenario=args.debug_int_scenario,
            debug_ext_scenario=args.debug_ext_scenario,
            debug_ext_year=args.debug_ext_year,
            debug_int_year=args.debug_int_year,
            debug_account=args.debug_account
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
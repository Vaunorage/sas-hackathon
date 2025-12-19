import csv
import os

from calculations.kernels import (
    external_generator_kernel, nested_valuation_kernel_five_chocs, STATE_SIZE,
    EXT_DEBUG_SIZE, INT_DEBUG_SIZE,
)
from calculations.constants import (
    MAX_SEXE, MAX_AGE, MAX_LAPSE_LEVELS, MAX_DURATION, DEFAULT_AGE_MAX_DEPOSIT,
    RN_DEFAULT_FORWARD_RATE, RN_DEFAULT_REND_DEX, RN_DEFAULT_REND_MM,
    RN_DEFAULT_REND_TSX, RN_DEFAULT_REND_SP500, RN_DEFAULT_REND_EAFE,
    NUM_CHOCS, CHOC_NAMES, METRICS_RESERVE_IDX, METRICS_CAPITAL_IDX, METRICS_OUTPUT_SIZE,
    FLUX_COMP_IDX_PRIMES_GARANTIES,
    FLUX_COMP_IDX_PREST_DECES,
    FLUX_COMP_IDX_PREST_ECH,
    FLUX_COMP_IDX_PREST_MRV,
    FLUX_COMP_IDX_FRAIS_ACQUIS,
    FLUX_COMP_IDX_COMM_VENTE,
    FLUX_COMP_IDX_PRIMES_VARIABLES,
    FLUX_COMP_IDX_FRAIS_FIXES,
    FLUX_COMP_IDX_HON_GEST,
    FLUX_COMP_IDX_COMM_MAINTIEN,
    FLUX_COMP_IDX_VALEUR_MARCHANDE,
    FLUX_COMP_IDX_PASSIF_REDRESSE,
    FLUX_COMP_IDX_COUSSIN_CREDIT,
    FLUX_COMP_IDX_COUSSIN_MARCHE,
    FLUX_COMP_IDX_COUSSIN_DEPENSE,
    FLUX_COMP_IDX_COUSSIN_DECHEANCE,
    FLUX_COMP_IDX_COUSSIN_MORTALITE,
    FLUX_COMP_IDX_COUSSIN_DEPOT,
    FLUX_COMP_IDX_MT_VM,
    FLUX_COMP_IDX_MT_VM_AV_RETRAIT,
    FLUX_COMP_IDX_MT_VM_AP_RETRAIT,
    FLUX_COMP_IDX_AGE,
    FLUX_COMP_IDX_QX,
    FLUX_COMP_IDX_LAPSE_TOT,
    FLUX_COMP_IDX_LAPSE_PART,
    FLUX_COMP_IDX_TX_SURVIE,
    FLUX_COMP_IDX_RETRAIT,
    FLUX_COMP_IDX_DEPOT_FUTUR,
    FLUX_COMP_IDX_MT_GAR_DECES,
    FLUX_COMP_IDX_MT_GAR_ECH,
    FLUX_COMP_IDX_MT_SRG,
    FLUX_COMP_IDX_REND_SP500,
    FLUX_COMP_IDX_REND_TSX,
    FLUX_COMP_IDX_REND_EAFE,
    FLUX_COMP_IDX_REND_DEX,
    FLUX_COMP_IDX_REND_MM,
    FLUX_COMP_IDX_MT_SP500,
    FLUX_COMP_IDX_MT_TSX,
    FLUX_COMP_IDX_MT_EAFE,
    FLUX_COMP_IDX_MT_DEX,
    FLUX_COMP_IDX_MT_MM,
    FLUX_COMP_IDX_SIZE,
    INT_TS_DEBUG_IDX_CURR_VM,
    INT_TS_DEBUG_IDX_FEES,
    INT_TS_DEBUG_IDX_PV_PATH,
    INT_TS_DEBUG_IDX_R_PORTFOLIO,
    INT_TS_DEBUG_IDX_FWD_RATE,
    INT_TS_DEBUG_IDX_DF,
    INT_TS_DEBUG_IDX_SIZE,
    LOOKUP_TABLE_OVERHEAD_MB, DEFAULT_GPU_MEMORY_GB, MEMORY_SAFETY_FACTOR, MEMORY_BATCH_THRESHOLD,
    DEFAULT_THREADS_PER_BLOCK_1D,
)

os.environ['NUMBA_CUDA_ENABLE_PYNVJITLINK'] = '1'

from calculations.utils import logger, CONFIG, load_all_data, prepare_account_data
from numba import cuda
from paths import HERE
import argparse
import pandas as pd
import numpy as np
import polars as pl
import gc
import sys
from pathlib import Path
from typing import Optional, List, TypedDict, Dict
from dataclasses import dataclass
from datetime import datetime
from fastparquet import write as fastparquet_write

# Try to import cuDF for GPU-accelerated DataFrame operations
try:
    import cudf
    import cupy as cp
except ImportError:
    print("⚠ CuDF not available - falling back to pandas (CPU). Install with: pip install cudf-cu12")


class KernelIncompatibilityError(RuntimeError):
    pass


EXPECTED_EXTERNAL_GENERATOR_ARGCOUNT = 18
EXPECTED_NESTED_VALUATION_FIVE_CHOCS_ARGCOUNT = 18


def validate_kernel_compatibility():
    kernel_a_argcount = external_generator_kernel.py_func.__code__.co_argcount
    if kernel_a_argcount != EXPECTED_EXTERNAL_GENERATOR_ARGCOUNT:
        raise KernelIncompatibilityError(
            "Kernel not compatible with the running methods: "
            f"external_generator_kernel has {kernel_a_argcount} parameters but calculations.gpu expects {EXPECTED_EXTERNAL_GENERATOR_ARGCOUNT}. "
            "Update calculations/kernels.py (or the service code) so the kernel signature matches."
        )

    kernel_b_argcount = nested_valuation_kernel_five_chocs.py_func.__code__.co_argcount
    if kernel_b_argcount != EXPECTED_NESTED_VALUATION_FIVE_CHOCS_ARGCOUNT:
        raise KernelIncompatibilityError(
            "Kernel not compatible with the running methods: "
            f"nested_valuation_kernel_five_chocs has {kernel_b_argcount} parameters but calculations.gpu expects {EXPECTED_NESTED_VALUATION_FIVE_CHOCS_ARGCOUNT}. "
            "Update calculations/kernels.py (or the service code) so the kernel signature matches."
        )


@dataclass
class ProjectionResult:
    """Result of run_projection_gpu_nested containing all output DataFrames."""
    results: pd.DataFrame
    results_5chocs: Optional[pd.DataFrame]
    sensitivities: Optional[pd.DataFrame]
    total_duration: float
    vp_flux_total: pd.DataFrame
    chocs_summary: Optional[pd.DataFrame]
    ext_debug_df: Optional[pd.DataFrame]
    int_debug_df: Optional[pd.DataFrame]
    int_debug_ts_df: Optional[pd.DataFrame]
    flux_projetes_df: Optional[pd.DataFrame]
    saved_files: List[str]


def create_gpu_mortality_lookup(df: pd.DataFrame):
    """Create flattened array for mortality lookup on GPU."""
    from calculations.constants import DEFAULT_MORTALITY_RATE
    # Create a 4D array indexed by: [i_sexe, age, year, i_produit_regr]
    max_year = df['ANNEE_REELLE'].max() + 1
    max_produit = df['I_PRODUIT_REGR'].max() + 1

    # Initialize with default value
    lookup = np.full((MAX_SEXE, MAX_AGE, max_year, max_produit), DEFAULT_MORTALITY_RATE, dtype=np.float32)

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


def create_gpu_rn_returns_lookup(df: pd.DataFrame, nb_int_scenarios: int, nb_an_projection: int):
    """Create 2D arrays for risk-neutral/internal returns lookup on GPU.

    Expected columns after normalization:
    - FORWARD_RATE, AJUST_FORWARD_RATE_VM_0, RENDDEX_AN, RENDMM_AN, RENDTSX_AN, RENDSP500_AN, RENDEAFE_AN
    - AN_EVAL_INT (or AN_EVAL)
    - SCN_EVAL_INT (or SCN_EVAL)
    - MOIS_EVAL (optional; if present we keep only month=12)

    Returns 7 arrays shaped (nb_int_scenarios, nb_an_projection) with scenario indices 1..N mapped to 0..N-1.
    Order: (forward_rate, ajust_forward, rend_dex, rend_mm, rend_tsx, rend_sp500, rend_eafe)
    """
    rn_forward_rate = np.full((nb_int_scenarios, nb_an_projection), RN_DEFAULT_FORWARD_RATE, dtype=np.float32)
    rn_ajust_forward = np.zeros((nb_int_scenarios, nb_an_projection), dtype=np.float32)
    rn_rend_dex = np.full((nb_int_scenarios, nb_an_projection), RN_DEFAULT_REND_DEX, dtype=np.float32)
    rn_rend_mm = np.full((nb_int_scenarios, nb_an_projection), RN_DEFAULT_REND_MM, dtype=np.float32)
    rn_rend_tsx = np.full((nb_int_scenarios, nb_an_projection), RN_DEFAULT_REND_TSX, dtype=np.float32)
    rn_rend_sp500 = np.full((nb_int_scenarios, nb_an_projection), RN_DEFAULT_REND_SP500, dtype=np.float32)
    rn_rend_eafe = np.full((nb_int_scenarios, nb_an_projection), RN_DEFAULT_REND_EAFE, dtype=np.float32)

    if df is None or len(df) == 0 or nb_int_scenarios <= 0 or nb_an_projection <= 0:
        return (rn_forward_rate, rn_ajust_forward, rn_rend_dex, rn_rend_mm, rn_rend_tsx, rn_rend_sp500, rn_rend_eafe)

    scn_col = 'SCN_EVAL_INT' if 'SCN_EVAL_INT' in df.columns else ('SCN_EVAL' if 'SCN_EVAL' in df.columns else None)
    an_col = 'AN_EVAL_INT' if 'AN_EVAL_INT' in df.columns else ('AN_EVAL' if 'AN_EVAL' in df.columns else None)
    mois_col = 'MOIS_EVAL' if 'MOIS_EVAL' in df.columns else None
    if scn_col is None or an_col is None:
        return (rn_forward_rate, rn_ajust_forward, rn_rend_dex, rn_rend_mm, rn_rend_tsx, rn_rend_sp500, rn_rend_eafe)

    df_iter = df
    if mois_col is not None:
        try:
            df_iter = df_iter[df_iter[mois_col] == 12]
        except Exception:
            df_iter = df

    # Pre-check which columns exist in the DataFrame
    has_forward_rate = 'FORWARD_RATE' in df_iter.columns
    has_ajust_forward = 'AJUST_FORWARD_RATE_VM_0' in df_iter.columns
    has_rend_dex = 'RENDDEX_AN' in df_iter.columns
    has_rend_mm = 'RENDMM_AN' in df_iter.columns
    has_rend_tsx = 'RENDTSX_AN' in df_iter.columns
    has_rend_sp500 = 'RENDSP500_AN' in df_iter.columns
    has_rend_eafe = 'RENDEAFE_AN' in df_iter.columns

    for _, row in df_iter.iterrows():
        try:
            scn_raw = int(row[scn_col])
            an = int(row[an_col])
        except Exception:
            continue

        scn = scn_raw - 1
        if scn < 0 or scn >= nb_int_scenarios or an < 0 or an >= nb_an_projection:
            continue

        if has_forward_rate:
            rn_forward_rate[scn, an] = float(row['FORWARD_RATE'])
        if has_ajust_forward:
            rn_ajust_forward[scn, an] = float(row['AJUST_FORWARD_RATE_VM_0'])
        if has_rend_dex:
            rn_rend_dex[scn, an] = float(row['RENDDEX_AN'])
        if has_rend_mm:
            rn_rend_mm[scn, an] = float(row['RENDMM_AN'])
        if has_rend_tsx:
            rn_rend_tsx[scn, an] = float(row['RENDTSX_AN'])
        if has_rend_sp500:
            rn_rend_sp500[scn, an] = float(row['RENDSP500_AN'])
        if has_rend_eafe:
            rn_rend_eafe[scn, an] = float(row['RENDEAFE_AN'])

    return (rn_forward_rate, rn_ajust_forward, rn_rend_dex, rn_rend_mm, rn_rend_tsx, rn_rend_sp500, rn_rend_eafe)


def create_gpu_min_ferr_lookup(df: pd.DataFrame):
    """Create array for minimum FERR lookup."""
    lookup = np.zeros(MAX_AGE, dtype=np.float32)
    for _, row in df.iterrows():
        age = int(row['AGE'])
        lookup[age] = float(row['MIN_FERR'])
    return lookup


def create_gpu_lapse_part_lookup(df: pd.DataFrame):
    """Create arrays for partial lapse lookup."""
    max_id_lapse = df['ID_LAPSE'].max() + 1
    max_regime = df['I_REGIME_2'].max() + 1

    tx_min = np.zeros((MAX_AGE, max_id_lapse, max_regime, MAX_LAPSE_LEVELS), dtype=np.float32)
    tx_max = np.zeros((MAX_AGE, max_id_lapse, max_regime, MAX_LAPSE_LEVELS), dtype=np.float32)

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
    max_id_lapse = df['ID_LAPSE'].max() + 1

    tx_min = np.zeros((MAX_DURATION, max_id_lapse, MAX_LAPSE_LEVELS), dtype=np.float32)
    tx_max = np.zeros((MAX_DURATION, max_id_lapse, MAX_LAPSE_LEVELS), dtype=np.float32)
    fact_dim = np.ones((MAX_DURATION, max_id_lapse, MAX_LAPSE_LEVELS), dtype=np.float32)

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
    max_id_depot = df['ID_DEPOT'].max() + 1

    pc_depot = np.zeros((MAX_DURATION, max_id_depot), dtype=np.float32)
    var_fct = np.zeros((MAX_DURATION, max_id_depot), dtype=np.int32)
    age_max = np.full((MAX_DURATION, max_id_depot), DEFAULT_AGE_MAX_DEPOSIT, dtype=np.int32)
    i_even = np.zeros((MAX_DURATION, max_id_depot), dtype=np.int32)

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
    max_id_acqui = df['ID_ACQUI'].max() + 1

    pc_vente_rf = np.zeros((MAX_DURATION, max_id_acqui), dtype=np.float32)
    pc_vente_ac = np.zeros((MAX_DURATION, max_id_acqui), dtype=np.float32)
    pc_maintien_rf = np.zeros((MAX_DURATION, max_id_acqui), dtype=np.float32)
    pc_maintien_ac = np.zeros((MAX_DURATION, max_id_acqui), dtype=np.float32)
    pc_frais_ac = np.zeros((MAX_DURATION, max_id_acqui), dtype=np.float32)
    pc_frais_rf = np.zeros((MAX_DURATION, max_id_acqui), dtype=np.float32)

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
    max_code_cat = int(df['CODE_CAT_PRODUIT'].max()) + 1 if len(df) > 0 else 8
    max_cat1 = int(df['CAT_COUSSIN_1'].max()) + 1 if len(df) > 0 else 6
    max_cat2 = int(df['CAT_COUSSIN_2'].max()) + 1 if len(df) > 0 else 7

    max_code_cat = max(max_code_cat, 8)
    max_cat1 = max(max_cat1, 6)
    max_cat2 = max(max_cat2, 7)

    shape = (max_code_cat, max_cat1, max_cat2)

    base_passif = np.zeros(shape, dtype=np.int32)
    tx_passif = np.zeros(shape, dtype=np.float32)
    base_credit = np.zeros(shape, dtype=np.int32)
    tx_credit = np.zeros(shape, dtype=np.float32)
    base_marche = np.zeros(shape, dtype=np.int32)
    tx_marche = np.zeros(shape, dtype=np.float32)
    base_depense = np.zeros(shape, dtype=np.int32)
    tx_depense = np.zeros(shape, dtype=np.float32)
    base_decheance = np.zeros(shape, dtype=np.int32)
    tx_decheance = np.zeros(shape, dtype=np.float32)
    base_mortalite = np.zeros(shape, dtype=np.int32)
    tx_mortalite = np.zeros(shape, dtype=np.float32)
    base_depot = np.zeros(shape, dtype=np.int32)
    tx_depot = np.zeros(shape, dtype=np.float32)
    facteur_age_80 = np.ones(shape, dtype=np.float32)
    facteur_age_90 = np.ones(shape, dtype=np.float32)

    for _, row in df.iterrows():
        code = int(row['CODE_CAT_PRODUIT'])
        c1 = int(row['CAT_COUSSIN_1'])
        c2 = int(row['CAT_COUSSIN_2'])
        if code < 0 or c1 < 0 or c2 < 0:
            continue
        if code >= shape[0] or c1 >= shape[1] or c2 >= shape[2]:
            continue
        base_passif[code, c1, c2] = int(row.get('BASE_PASSIF_REDRESSE', 0))
        tx_passif[code, c1, c2] = float(row.get('TX_PASSIF_REDRESSE', 0.0))
        base_credit[code, c1, c2] = int(row.get('BASE_COUSSIN_CREDIT', 0))
        tx_credit[code, c1, c2] = float(row.get('TX_COUSSIN_CREDIT', 0.0))
        base_marche[code, c1, c2] = int(row.get('BASE_COUSSIN_MARCHE', 0))
        tx_marche[code, c1, c2] = float(row.get('TX_COUSSIN_MARCHE', 0.0))
        base_depense[code, c1, c2] = int(row.get('BASE_COUSSIN_DEPENSE', 0))
        tx_depense[code, c1, c2] = float(row.get('TX_COUSSIN_DEPENSE', 0.0))
        base_decheance[code, c1, c2] = int(row.get('BASE_COUSSIN_DECHEANCE', 0))
        tx_decheance[code, c1, c2] = float(row.get('TX_COUSSIN_DECHEANCE', 0.0))
        base_mortalite[code, c1, c2] = int(row.get('BASE_COUSSIN_MORTALITE', 0))
        tx_mortalite[code, c1, c2] = float(row.get('TX_COUSSIN_MORTALITE', 0.0))
        base_depot[code, c1, c2] = int(row.get('BASE_COUSSIN_DEPOT', 0))
        tx_depot[code, c1, c2] = float(row.get('TX_COUSSIN_DEPOT', 0.0))
        facteur_age_80[code, c1, c2] = float(row.get('FACTEUR_AGE_80', 1.0))
        facteur_age_90[code, c1, c2] = float(row.get('FACTEUR_AGE_90', 1.0))

    return (
        base_passif, tx_passif,
        base_credit, tx_credit,
        base_marche, tx_marche,
        base_depense, tx_depense,
        base_decheance, tx_decheance,
        base_mortalite, tx_mortalite,
        base_depot, tx_depot,
        facteur_age_80, facteur_age_90,
    )


def initialize_gpu():
    """
    Initialize GPU and check availability.
    
    Returns:
        Tuple of (gpu_device, free_mem, total_mem) or raises RuntimeError
    """
    try:
        if not cuda.is_available():
            raise RuntimeError("CUDA is not available")
        
        gpu = cuda.get_current_device()
        print(f"GPU Device: {gpu.name.decode()}")
        
        try:
            free_mem, total_mem = cuda.current_context().get_memory_info()
            print(f"GPU Memory: {free_mem / 1024**3:.2f} GB free / {total_mem / 1024**3:.2f} GB total")
        except NotImplementedError:
            free_mem, total_mem = None, None
        
        return gpu, free_mem, total_mem
    except Exception as e:
        raise RuntimeError(f"Failed to initialize GPU: {e}")


def calculate_batch_size(n_accounts: int, nb_ext_scenarios: int, nb_an_projection: int, 
                         nb_int_scenarios: int, account_data_cols: int):
    """
    Calculate optimal batch size based on memory requirements.
    
    Returns:
        Tuple of (batch_size, num_batches, total_mem_per_account, lookup_overhead)
    """
    print("\nCalculating memory requirements...")
    
    # State tensor: (Batch, Ext_Scenarios, Years, STATE_SIZE)
    state_mem_per_account = nb_ext_scenarios * nb_an_projection * STATE_SIZE * 4  # float32
    
    # Cashflow tensor: (Batch, Ext_Scenarios, Years, 1)
    cf_mem_per_account = nb_ext_scenarios * nb_an_projection * 1 * 4
    
    # Metrics tensor: (Batch, Ext_Scenarios, Years, NUM_CHOCS, METRICS_OUTPUT_SIZE) - chocs × (Reserve & Capital)
    metrics_mem_per_account = nb_ext_scenarios * nb_an_projection * NUM_CHOCS * METRICS_OUTPUT_SIZE * 4
    
    total_mem_per_account = (state_mem_per_account + cf_mem_per_account + 
                             metrics_mem_per_account + account_data_cols * 4)
    
    # Estimate lookup table memory overhead (always resident on GPU)
    lookup_overhead = 0
    lookup_overhead += 6 * nb_ext_scenarios * nb_an_projection * 12 * 4
    lookup_overhead += 6 * nb_int_scenarios * nb_an_projection * 4
    lookup_overhead += LOOKUP_TABLE_OVERHEAD_MB * 1024**2
    
    print(f"  State tensor per account: {state_mem_per_account / 1024**2:.2f} MB")
    print(f"  Total memory per account: {total_mem_per_account / 1024**2:.2f} MB")
    print(f"  Lookup table overhead: {lookup_overhead / 1024**2:.2f} MB")
    
    # Calculate batch size (conservative for nested scenarios)
    try:
        free_mem, total_mem = cuda.current_context().get_memory_info()
        print(f"  GPU free memory: {free_mem / 1024**3:.2f} GB")
        print(f"  GPU total memory: {total_mem / 1024**3:.2f} GB")
        available_mem = max(0, (free_mem - lookup_overhead) * MEMORY_SAFETY_FACTOR)
    except NotImplementedError:
        print("  Warning: Cannot query GPU memory, using conservative estimate")
        available_mem = max(0, DEFAULT_GPU_MEMORY_GB * 1024**3 - lookup_overhead)
    
    batch_size = max(1, int(available_mem // total_mem_per_account))
    batch_size = min(batch_size, n_accounts)
    num_batches = (n_accounts + batch_size - 1) // batch_size
    
    print(f"  Batch size: {batch_size} accounts")
    print(f"  Total batches: {num_batches}")
    
    return batch_size, num_batches, total_mem_per_account, lookup_overhead


class ProcessBatchResult(TypedDict):
    """Typed result from process_batch function."""
    batch_reserves: np.ndarray
    batch_capital: np.ndarray
    batch_reserves_5chocs: np.ndarray
    batch_capital_5chocs: np.ndarray
    ext_debug: Optional[np.ndarray]  # Debug output from external kernel
    int_debug: Optional[np.ndarray]  # Debug output from internal kernel
    int_debug_ts: Optional[np.ndarray]  # Debug time series output from internal kernel


def check_gpu_memory(batch_size: int, mem_per_account: float, batch_idx: int = 0):
    """Log GPU memory status and raise if insufficient for batch."""
    try:
        free_mem, _ = cuda.current_context().get_memory_info()
        estimated_mem = batch_size * mem_per_account
        logger.info(f"  Free GPU memory: {free_mem / 1024 ** 3:.2f} GB")
        logger.info(f"  Estimated batch memory: {estimated_mem / 1024 ** 3:.2f} GB")
        
        if estimated_mem > free_mem * MEMORY_BATCH_THRESHOLD:
            raise RuntimeError(
                f"Insufficient GPU memory for batch {batch_idx + 1}. "
                f"Need {estimated_mem / 1024**3:.2f} GB but only "
                f"{free_mem / 1024**3:.2f} GB available. "
                f"Try reducing batch size or number of scenarios."
            )
    except NotImplementedError:
        pass


def process_batch(
    batch_account_data: np.ndarray,
    nb_ext_scenarios: int,
    nb_an_projection: int,
    nb_int_scenarios: int,
    shock_capital_pct: float,
    total_mem_per_account: float,
    threads_per_block: tuple,
    gpu_lookups: dict,
    batch_idx: int = 0,
    num_batches: int = 1,
    debug_account: int = -1,
    debug_scenario: int = -1,
    debug_year: int = -1,
    debug_month: int = -1,
    debug_int_scenario: int = -1,
    debug_int_year: int = -1,
) -> ProcessBatchResult:
    """
    Process a single batch through both kernels.
    
    Args:
        batch_account_data: 2D array of account data for this batch (n_batch_accounts, n_features)
        nb_ext_scenarios: Number of external scenarios
        nb_an_projection: Number of projection years
        nb_int_scenarios: Number of internal scenarios
        total_mem_per_account: Estimated GPU memory per account (bytes)
        threads_per_block: CUDA thread block dimensions for Kernel A
        gpu_lookups: Dictionary of GPU device arrays with lookup tables
        batch_idx: Index of current batch (for logging)
        num_batches: Total number of batches (for logging)
        debug_account: Account index to debug (-1 = disabled)
        debug_scenario: External scenario index to debug (-1 = disabled)
        debug_year: Year (an_eval) to debug (-1 = disabled)
        debug_month: Month (mois_eval) to debug (-1 = disabled)
        debug_int_scenario: Internal scenario to debug (-1 = disabled)
        debug_int_year: Internal year to debug (-1 = disabled)
    
    Returns:
        ProcessBatchResult with batch results
    """
    batch_start = datetime.now()
    current_batch_size = len(batch_account_data)
    
    logger.info(f"\n--- Batch {batch_idx + 1}/{num_batches} ({current_batch_size} accounts) ---")
    check_gpu_memory(current_batch_size, total_mem_per_account, batch_idx)
    
    # Prepare batch data
    batch_account_data_contiguous = np.ascontiguousarray(batch_account_data)
    d_batch_accounts = _to_device_contiguous(batch_account_data_contiguous)

    # Allocate tensors using cupy to avoid numba-cuda device_pointer bug
    try:
        d_states = _device_array_cupy(
            (current_batch_size, nb_ext_scenarios, nb_an_projection, STATE_SIZE)
        )
        d_cashflows = _device_array_cupy(
            (current_batch_size, nb_ext_scenarios, nb_an_projection, 1)
        )
        d_metrics = _device_array_cupy(
            (current_batch_size, nb_ext_scenarios, nb_an_projection, NUM_CHOCS, METRICS_OUTPUT_SIZE)
        )
        
        # Allocate debug arrays (always allocate, use -1 flags to disable)
        enable_ext_debug = debug_account >= 0 or debug_scenario >= 0 or debug_year >= 0 or debug_month >= 0
        enable_int_debug = enable_ext_debug  # Internal debug only if external debug is enabled
        
        if enable_ext_debug:
            logger.info(f"  Debug mode: account={debug_account}, scenario={debug_scenario}, year={debug_year}, month={debug_month}")
        if enable_int_debug:
            logger.info(f"  Internal debug: int_scenario={debug_int_scenario}, int_year={debug_int_year}")
        
        # Always allocate debug arrays (kernel uses -1 flags to skip writing)
        d_ext_debug = _device_array_cupy((EXT_DEBUG_SIZE,))
        d_int_debug = _device_array_cupy((NUM_CHOCS, INT_DEBUG_SIZE))
        
        # Allocate debug flux array for single account/scenario flux capture
        # Shape: (n_years+1, freq_eval, FLUX_COMP_IDX_SIZE)
        freq_eval_int = int(CONFIG['FREQ_EVAL'])
        if enable_ext_debug:
            d_debug_flux = _to_device_contiguous(
                np.zeros((nb_an_projection + 1, freq_eval_int + 1, FLUX_COMP_IDX_SIZE), dtype=np.float32)
            )
        else:
            # Minimal array when debug is disabled
            d_debug_flux = _to_device_contiguous(np.zeros((1, 1, FLUX_COMP_IDX_SIZE), dtype=np.float32))
        
        enable_int_debug_ts = enable_int_debug and debug_int_scenario >= 0
        if enable_int_debug_ts:
            d_int_debug_ts = _to_device_contiguous(
                np.zeros((NUM_CHOCS, nb_an_projection, INT_TS_DEBUG_IDX_SIZE), dtype=np.float32)
            )
        else:
            # Always pass an array to the kernel to keep the CUDA signature stable.
            d_int_debug_ts = _to_device_contiguous(np.zeros((1, 1, INT_TS_DEBUG_IDX_SIZE), dtype=np.float32))
    except Exception as e:
        raise RuntimeError(
            f"Failed to allocate GPU memory for batch {batch_idx+1}. "
            f"Try reducing --max-accounts or --ext-scenarios. Original error: {e}"
        )
    
    # === KERNEL A: EXTERNAL GENERATOR ===
    logger.info("  Launching Kernel A (External Generator)...")
    validate_kernel_compatibility()
    blocks_x = (current_batch_size + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_y = (nb_ext_scenarios + threads_per_block[1] - 1) // threads_per_block[1]
    grid_A = (blocks_x, blocks_y)
    
    kernel_a_start = datetime.now()
    if 'coussins' not in gpu_lookups:
        raise RuntimeError(
            "Kernel A expects 'coussins' lookup but gpu_lookups['coussins'] is missing. "
            "Ensure create_gpu_coussins_lookup() is called and the lookup is transferred to device."
        )
    external_generator_kernel[grid_A, threads_per_block](
        d_batch_accounts,
        nb_ext_scenarios, nb_an_projection,
        CONFIG['FREQ_EVAL'],
        gpu_lookups['mortality'],
        gpu_lookups['returns'],
        gpu_lookups['lapse'],
        gpu_lookups['policy'],
        gpu_lookups['commission'],
        gpu_lookups['coussins'],
        d_states,
        d_cashflows,
        d_ext_debug,
        d_debug_flux,
        debug_account,
        debug_scenario,
        debug_year,
        debug_month,
    )
    cuda.synchronize()
    kernel_a_time = (datetime.now() - kernel_a_start).total_seconds()
    logger.info(f"  Kernel A complete: {kernel_a_time:.2f}s")
    
    # === KERNEL B: NESTED VALUATOR WITH 5 CHOCS ===
    logger.info(f"  Launching Kernel B (Five Chocs Nested Valuator)...")
    total_nodes = current_batch_size * nb_ext_scenarios * nb_an_projection
    threads_per_block_B = DEFAULT_THREADS_PER_BLOCK_1D
    blocks_B = (total_nodes + threads_per_block_B - 1) // threads_per_block_B
    
    kernel_b_start = datetime.now()
    
    nested_valuation_kernel_five_chocs[blocks_B, threads_per_block_B](
        d_states,
        d_batch_accounts,
        nb_int_scenarios,
        nb_an_projection,
        gpu_lookups['rn_returns'],
        gpu_lookups['mortality'],
        gpu_lookups['lapse'],
        gpu_lookups['policy'],
        gpu_lookups['commission'],
        d_metrics,
        d_int_debug,
        d_int_debug_ts,
        debug_int_scenario,
        debug_int_year,
        debug_account,
        debug_scenario,
        debug_year,
        float(shock_capital_pct),
    )
    cuda.synchronize()
    
    kernel_b_time = (datetime.now() - kernel_b_start).total_seconds()
    logger.info(f"  Kernel B complete: {kernel_b_time:.2f}s")
    
    # Copy results back
    logger.info("  Copying results to CPU...")
    h_metrics = d_metrics.copy_to_host()
    
    # Copy debug arrays if enabled
    h_ext_debug = None
    h_int_debug = None
    h_int_debug_ts = None
    h_debug_flux = None
    if enable_ext_debug:
        logger.info("  Copying external debug output to CPU...")
        h_ext_debug = d_ext_debug.copy_to_host()
        h_debug_flux = d_debug_flux.copy_to_host()
    if enable_int_debug:
        logger.info("  Copying internal debug output to CPU...")
        h_int_debug = d_int_debug.copy_to_host()
        if enable_int_debug_ts:
            h_int_debug_ts = d_int_debug_ts.copy_to_host()
    
    # Process metrics
    batch_reserves_5chocs = h_metrics[:, :, :, :, METRICS_RESERVE_IDX].mean(axis=(1, 2))
    batch_capital_5chocs = h_metrics[:, :, :, :, METRICS_CAPITAL_IDX].mean(axis=(1, 2))
    batch_reserves = batch_reserves_5chocs[:, 0]
    batch_capital = batch_capital_5chocs[:, 0]
    
    # Cleanup
    del d_batch_accounts, d_states, d_cashflows, d_metrics, d_ext_debug, d_int_debug, d_int_debug_ts, d_debug_flux
    cuda.synchronize()
    del h_metrics
    gc.collect()
    
    try:
        import rmm
        rmm.mr.get_current_device_resource().deallocate(0, 0)
    except (ImportError, AttributeError):
        pass
    
    batch_time = (datetime.now() - batch_start).total_seconds()
    logger.info(f"  Batch complete: {batch_time:.2f}s (KernelA: {kernel_a_time:.2f}s, KernelB: {kernel_b_time:.2f}s)")
    
    return {
        'batch_reserves': batch_reserves,
        'batch_capital': batch_capital,
        'batch_reserves_5chocs': batch_reserves_5chocs,
        'batch_capital_5chocs': batch_capital_5chocs,
        'ext_debug': h_ext_debug,
        'int_debug': h_int_debug,
        'int_debug_ts': h_int_debug_ts,
        'debug_flux': h_debug_flux,
    }


def create_results_dataframes(
    population_ids: np.ndarray,
    all_reserves: list,
    all_capital: list,
    all_reserves_5chocs: list,
    all_capital_5chocs: list,
    n_accounts: int
):
    """
    Create results DataFrames from accumulated batch results.
    
    Returns:
        Tuple of (results_df, results_5chocs_df, sensitivities_df)
    """
    results_df = pd.DataFrame({
        'ID_COMPTE': population_ids[:n_accounts],
        'RESERVE_BE': all_reserves,
        'CAPITAL_REQ': all_capital,
        'SCR': [cap - res for res, cap in zip(all_reserves, all_capital)]
    })
    
    results_5chocs_df = None
    sensitivities_df = None
    
    if all_reserves_5chocs:
        all_reserves_5chocs_array = np.array(all_reserves_5chocs)
        all_capital_5chocs_array = np.array(all_capital_5chocs)
        choc_rows = []
        
        for acc_idx in range(n_accounts):
            account_id = population_ids[acc_idx]
            for choc_idx, choc_name in enumerate(CHOC_NAMES):
                reserve = all_reserves_5chocs_array[acc_idx, choc_idx]
                capital = all_capital_5chocs_array[acc_idx, choc_idx]
                choc_rows.append({
                    'ID_COMPTE': account_id,
                    'CHOC_TYPE': choc_name,
                    'CHOC_IDX': choc_idx,
                    'RESERVE_BE': reserve,
                    'CAPITAL_REQ': capital,
                    'SCR': capital - reserve
                })
        
        results_5chocs_df = pd.DataFrame(choc_rows)
        
        base_reserves = all_reserves_5chocs_array[:, 0]
        base_capital = all_capital_5chocs_array[:, 0]
        
        sensitivities_df = pd.DataFrame({
            'ID_COMPTE': population_ids[:n_accounts],
            'DELTA_SP500_RESERVE': all_reserves_5chocs_array[:, 1] - base_reserves,
            'DELTA_TSX_RESERVE': all_reserves_5chocs_array[:, 2] - base_reserves,
            'DELTA_EAFE_RESERVE': all_reserves_5chocs_array[:, 3] - base_reserves,
            'DELTA_DEX_RESERVE': all_reserves_5chocs_array[:, 4] - base_reserves,
            'DELTA_SP500_CAPITAL': all_capital_5chocs_array[:, 1] - base_capital,
            'DELTA_TSX_CAPITAL': all_capital_5chocs_array[:, 2] - base_capital,
            'DELTA_EAFE_CAPITAL': all_capital_5chocs_array[:, 3] - base_capital,
            'DELTA_DEX_CAPITAL': all_capital_5chocs_array[:, 4] - base_capital,
        })
        
        logger.info(f"Created 5 chocs results with {len(results_5chocs_df)} rows and sensitivities for {len(sensitivities_df)} accounts")
    
    return results_df, results_5chocs_df, sensitivities_df


def save_results(
    output_path: Path,
    results_df: pd.DataFrame,
    results_5chocs_df: Optional[pd.DataFrame],
    sensitivities_df: Optional[pd.DataFrame],
    n_accounts: int,
    ext_debug: Optional[np.ndarray] = None,
    int_debug: Optional[np.ndarray] = None,
    int_debug_ts_df: Optional[pd.DataFrame] = None,
    debug_params: Optional[dict] = None,
    population_ids: Optional[np.ndarray] = None,
    debug_flux: Optional[np.ndarray] = None,
    population_df: Optional[pd.DataFrame] = None,
    lookup_data: Optional[Dict[str, pd.DataFrame]] = None,
):
    """
    Save all results (final simulation results and debug output) to CSV files.
    
    Args:
        output_path: Directory to save CSV files
        results_df: Main results DataFrame with reserves/capital per account
        results_5chocs_df: Optional DataFrame with 5 chocs results
        sensitivities_df: Optional DataFrame with sensitivities/Greeks
        n_accounts: Number of accounts processed
        ext_debug: Optional external kernel debug array (EXT_DEBUG_SIZE,) - single row
        int_debug: Optional internal kernel debug array (NUM_CHOCS, INT_DEBUG_SIZE) - one row per choc
        debug_params: Optional dictionary with debug filter parameters for context
        population_ids: Optional array of real account IDs (ID_COMPTE) for mapping debug account index
    
    Returns:
        Dictionary containing all created DataFrames:
        - 'saved_files': List of saved file names
        - 'vp_flux_total': Portfolio totals DataFrame
        - 'chocs_summary': 5 chocs summary DataFrame (if results_5chocs_df provided)
        - 'ext_debug_df': External kernel debug DataFrame (if ext_debug provided)
        - 'int_debug_df': Internal kernel debug DataFrame (if int_debug provided)
    """
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Column names for debug CSV files
    EXT_DEBUG_COLUMNS = [
        'VM', 'AGE', 'QX', 'LAPSE_TOT', 'LAPSE_PART', 'TX_SURVIE',
        'FORWARD_RATE', 'REND_SP500', 'REND_TSX', 'REND_EAFE', 'REND_DEX',
        'RETRAIT', 'PREST_DECES', 'PRIMES_GARANTIES', 'VM_VG_RATIO'
    ]
    
    INT_DEBUG_COLUMNS = [
        'START_VM', 'VM_CHOC', 'AVG_PV_FLUX', 'RESERVE', 'CAPITAL',
        'START_TX_SURVIE', 'START_AGE',
        # Values captured at specific internal scenario/year
        'INT_CURR_VM', 'INT_FEES', 'INT_PV_PATH', 'INT_R_PORTFOLIO', 'INT_FWD_RATE'
    ]
    
    print("\n" + "=" * 80)
    print("SAVING OUTPUT FILES")
    print("=" * 80)
    
    saved_files = []
    chocs_summary_df = None
    ext_debug_df = None
    int_debug_df = None
    int_debug_ts_saved_df = None
    flux_projetes_df = None
    
    # ===========================================
    # 1. FINAL SIMULATION RESULTS
    # ===========================================
    
    # 1a. VP_FLUX_TOTAL_GPU.csv - Portfolio totals
    vp_flux_total_path = output_path / "VP_FLUX_TOTAL_GPU.csv"
    vp_flux_total_df = pd.DataFrame({
        'CATEGORIE': ['TOTAL'],
        'VP_RESERVE_BE': [results_df['RESERVE_BE'].sum()],
        'VP_CAPITAL_REQ': [results_df['CAPITAL_REQ'].sum()],
        'VP_SCR': [results_df['SCR'].sum()],
        'AVG_RESERVE_BE': [results_df['RESERVE_BE'].mean()],
        'AVG_CAPITAL_REQ': [results_df['CAPITAL_REQ'].mean()],
        'AVG_SCR': [results_df['SCR'].mean()],
        'N_ACCOUNTS': [len(results_df)]
    })
    vp_flux_total_df.to_csv(vp_flux_total_path, index=False, sep=';')
    print(f"✓ Saved VP_FLUX_TOTAL_GPU.csv")
    print(f"  Total Reserve (BE): ${vp_flux_total_df['VP_RESERVE_BE'].iloc[0]:,.2f}")
    print(f"  Total Capital Req:  ${vp_flux_total_df['VP_CAPITAL_REQ'].iloc[0]:,.2f}")
    print(f"  Total SCR:          ${vp_flux_total_df['VP_SCR'].iloc[0]:,.2f}")
    saved_files.append("VP_FLUX_TOTAL_GPU.csv (portfolio totals)")

    # Save debug flux for single account/scenario if available
    if debug_flux is not None and debug_params is not None:
        debug_account_idx = debug_params.get('account', -1)
        debug_scenario_idx = debug_params.get('scenario', -1)
        
        # Get ID_COMPTE for the debug account
        id_compte = -1
        if population_ids is not None and debug_account_idx >= 0 and debug_account_idx < len(population_ids):
            id_compte = int(population_ids[debug_account_idx])
        
        # Build flux DataFrame from debug_flux array (n_years+1, freq_eval+1, FLUX_COMP_IDX_SIZE)
        rows = []
        n_years = debug_flux.shape[0]
        n_months = debug_flux.shape[1]
        
        for an_eval in range(n_years):
            for mois_eval in range(n_months):
                # Skip if all values are zero (no data for this period)
                flux_row = debug_flux[an_eval, mois_eval, :]
                if np.all(flux_row == 0):
                    continue
                rows.append({
                    'ID_COMPTE': id_compte,
                    'DEBUG_ACCOUNT_IDX': debug_account_idx,
                    'DEBUG_SCENARIO': debug_scenario_idx,
                    'AN_EVAL': an_eval,
                    'MOIS_EVAL': mois_eval,
                    # Cashflow components
                    'PRIMES_GARANTIES': float(flux_row[FLUX_COMP_IDX_PRIMES_GARANTIES]),
                    'PREST_DECES': float(flux_row[FLUX_COMP_IDX_PREST_DECES]),
                    'PREST_ECH': float(flux_row[FLUX_COMP_IDX_PREST_ECH]),
                    'PREST_MRV': float(flux_row[FLUX_COMP_IDX_PREST_MRV]),
                    'FRAIS_ACQUIS': float(flux_row[FLUX_COMP_IDX_FRAIS_ACQUIS]),
                    'COMM_VENTE': float(flux_row[FLUX_COMP_IDX_COMM_VENTE]),
                    'PRIMES_VARIABLES': float(flux_row[FLUX_COMP_IDX_PRIMES_VARIABLES]),
                    'FRAIS_FIXES': float(flux_row[FLUX_COMP_IDX_FRAIS_FIXES]),
                    'HON_GEST': float(flux_row[FLUX_COMP_IDX_HON_GEST]),
                    'COMM_MAINTIEN': float(flux_row[FLUX_COMP_IDX_COMM_MAINTIEN]),
                    'VALEUR_MARCHANDE': float(flux_row[FLUX_COMP_IDX_VALEUR_MARCHANDE]),
                    'PASSIF_REDRESSE': float(flux_row[FLUX_COMP_IDX_PASSIF_REDRESSE]),
                    'COUSSIN_CREDIT': float(flux_row[FLUX_COMP_IDX_COUSSIN_CREDIT]),
                    'COUSSIN_MARCHE': float(flux_row[FLUX_COMP_IDX_COUSSIN_MARCHE]),
                    'COUSSIN_DEPENSE': float(flux_row[FLUX_COMP_IDX_COUSSIN_DEPENSE]),
                    'COUSSIN_DECHEANCE': float(flux_row[FLUX_COMP_IDX_COUSSIN_DECHEANCE]),
                    'COUSSIN_MORTALITE': float(flux_row[FLUX_COMP_IDX_COUSSIN_MORTALITE]),
                    'COUSSIN_DEPOT': float(flux_row[FLUX_COMP_IDX_COUSSIN_DEPOT]),
                    # Detailed calculation fields
                    'MT_VM': float(flux_row[FLUX_COMP_IDX_MT_VM]),
                    'MT_VM_AV_RETRAIT': float(flux_row[FLUX_COMP_IDX_MT_VM_AV_RETRAIT]),
                    'MT_VM_AP_RETRAIT': float(flux_row[FLUX_COMP_IDX_MT_VM_AP_RETRAIT]),
                    'AGE': float(flux_row[FLUX_COMP_IDX_AGE]),
                    'QX': float(flux_row[FLUX_COMP_IDX_QX]),
                    'LAPSE_TOT': float(flux_row[FLUX_COMP_IDX_LAPSE_TOT]),
                    'LAPSE_PART': float(flux_row[FLUX_COMP_IDX_LAPSE_PART]),
                    'TX_SURVIE': float(flux_row[FLUX_COMP_IDX_TX_SURVIE]),
                    'RETRAIT': float(flux_row[FLUX_COMP_IDX_RETRAIT]),
                    'DEPOT_FUTUR': float(flux_row[FLUX_COMP_IDX_DEPOT_FUTUR]),
                    'MT_GAR_DECES': float(flux_row[FLUX_COMP_IDX_MT_GAR_DECES]),
                    'MT_GAR_ECH': float(flux_row[FLUX_COMP_IDX_MT_GAR_ECH]),
                    'MT_SRG': float(flux_row[FLUX_COMP_IDX_MT_SRG]),
                    'REND_SP500': float(flux_row[FLUX_COMP_IDX_REND_SP500]),
                    'REND_TSX': float(flux_row[FLUX_COMP_IDX_REND_TSX]),
                    'REND_EAFE': float(flux_row[FLUX_COMP_IDX_REND_EAFE]),
                    'REND_DEX': float(flux_row[FLUX_COMP_IDX_REND_DEX]),
                    'REND_MM': float(flux_row[FLUX_COMP_IDX_REND_MM]),
                    'MT_SP500': float(flux_row[FLUX_COMP_IDX_MT_SP500]),
                    'MT_TSX': float(flux_row[FLUX_COMP_IDX_MT_TSX]),
                    'MT_EAFE': float(flux_row[FLUX_COMP_IDX_MT_EAFE]),
                    'MT_DEX': float(flux_row[FLUX_COMP_IDX_MT_DEX]),
                    'MT_MM': float(flux_row[FLUX_COMP_IDX_MT_MM]),
                })
        
        if rows:
            flux_projetes_df = pd.DataFrame(rows)
            flux_projetes_path = output_path / "FLUX_PROJETES_GPU.csv"
            flux_projetes_df.to_csv(flux_projetes_path, index=False, sep=';')
            print(f"✓ Saved FLUX_PROJETES_GPU.csv (debug: account={debug_account_idx}, scenario={debug_scenario_idx}, ID_COMPTE={id_compte})")
            saved_files.append("FLUX_PROJETES_GPU.csv (single account/scenario flux)")

            example_header = None
            example_path = Path(__file__).resolve().parents[1] / "output_example.csv"
            try:
                with example_path.open('r', newline='') as f:
                    reader = csv.reader(f)
                    example_header = next(reader)
            except Exception:
                example_header = None

            if example_header:
                acc_row = None
                if population_df is not None and 0 <= debug_account_idx < len(population_df):
                    acc_row = population_df.iloc[debug_account_idx]

                wide_rows = []
                # Match output_example.csv convention: include an initial evaluation row (t0)
                # with TX_SURVIE = 1 and projected values equal to initial account values.
                if acc_row is not None:
                    w0 = {c: np.nan for c in example_header}
                    w0['ID_COMPTE'] = int(acc_row.get('ID_COMPTE', -1))
                    w0['scn_eval'] = int(debug_scenario_idx) + 1 if debug_scenario_idx is not None and debug_scenario_idx >= 0 else np.nan
                    w0['an_eval'] = 0
                    w0['mois_eval'] = 12
                    w0['mois_eval_ext'] = 12
                    w0['TX_SURVIE'] = 1.0
                    w0['TX_SURVIE_DEB'] = 1.0
                    w0['MT_VM_PROJ'] = acc_row.get('MT_VM', np.nan)
                    w0['MT_SP500_PROJ'] = acc_row.get('MT_SP500', np.nan)
                    w0['MT_TSX_PROJ'] = acc_row.get('MT_TSX', np.nan)
                    w0['MT_EAFE_PROJ'] = acc_row.get('MT_EAFE', np.nan)
                    w0['MT_DEX_PROJ'] = acc_row.get('MT_DEX', np.nan)
                    w0['MT_MM_PROJ'] = acc_row.get('MT_MM', np.nan)
                    w0['MT_GAR_DECES_PROJ'] = acc_row.get('MT_GAR_DECES', np.nan)
                    w0['MT_GAR_ECH_PROJ'] = acc_row.get('MT_GAR_ECH', np.nan)
                    w0['MT_SRG_PROJ'] = acc_row.get('MT_SRG', np.nan)
                    skip_cols = {'mois_eval', 'mois_eval_ext', 'an_eval', 'TX_SURVIE', 'TX_SURVIE_DEB'}
                    for c in example_header:
                        if c in skip_cols:
                            continue
                        if pd.isna(w0.get(c, np.nan)) and c in acc_row.index:
                            w0[c] = acc_row[c]
                    wide_rows.append(w0)

                prev_tx_survie = 1.0
                for r in rows:
                    w = {c: np.nan for c in example_header}

                    w['ID_COMPTE'] = r.get('ID_COMPTE', -1)
                    w['scn_eval'] = int(debug_scenario_idx) + 1 if debug_scenario_idx is not None and debug_scenario_idx >= 0 else np.nan
                    w['an_eval'] = r.get('AN_EVAL', np.nan)
                    w['mois_eval'] = r.get('MOIS_EVAL', np.nan)
                    w['mois_eval_ext'] = r.get('MOIS_EVAL', np.nan)

                    w['Qx'] = r.get('QX', np.nan)
                    w['TX_SURVIE'] = r.get('TX_SURVIE', np.nan)
                    w['TX_SURVIE_DEB'] = prev_tx_survie
                    w['LAPSE_TOT'] = r.get('LAPSE_TOT', np.nan)
                    w['LAPSE_PART'] = r.get('LAPSE_PART', np.nan)
                    w['RETRAIT'] = r.get('RETRAIT', np.nan)
                    w['DEPOT_FUTUR'] = r.get('DEPOT_FUTUR', np.nan)

                    try:
                        curr_survie = float(w['TX_SURVIE'])
                        if not np.isnan(curr_survie):
                            prev_tx_survie = curr_survie
                    except Exception:
                        pass

                    w['rendSP500_an'] = r.get('REND_SP500', np.nan)
                    w['rendTSX_an'] = r.get('REND_TSX', np.nan)
                    w['rendEAFE_an'] = r.get('REND_EAFE', np.nan)
                    w['rendDEX_an'] = r.get('REND_DEX', np.nan)
                    w['rendMM_an'] = r.get('REND_MM', np.nan)

                    w['MT_VM_PROJ'] = r.get('MT_VM', np.nan)
                    w['MT_VM_AV_RETRAIT'] = r.get('MT_VM_AV_RETRAIT', np.nan)
                    w['MT_VM_AP_RETRAIT'] = r.get('MT_VM_AP_RETRAIT', np.nan)
                    w['MT_SP500_PROJ'] = r.get('MT_SP500', np.nan)
                    w['MT_TSX_PROJ'] = r.get('MT_TSX', np.nan)
                    w['MT_EAFE_PROJ'] = r.get('MT_EAFE', np.nan)
                    w['MT_DEX_PROJ'] = r.get('MT_DEX', np.nan)
                    w['MT_MM_PROJ'] = r.get('MT_MM', np.nan)
                    if acc_row is not None:
                        w['MT_VM'] = acc_row.get('MT_VM', np.nan)
                        w['MT_SP500'] = acc_row.get('MT_SP500', np.nan)
                        w['MT_TSX'] = acc_row.get('MT_TSX', np.nan)
                        w['MT_EAFE'] = acc_row.get('MT_EAFE', np.nan)
                        w['MT_DEX'] = acc_row.get('MT_DEX', np.nan)
                        w['MT_MM'] = acc_row.get('MT_MM', np.nan)
                    w['MT_GAR_DECES_PROJ'] = r.get('MT_GAR_DECES', np.nan)
                    w['MT_GAR_ECH_PROJ'] = r.get('MT_GAR_ECH', np.nan)
                    w['MT_SRG_PROJ'] = r.get('MT_SRG', np.nan)
                    if acc_row is not None:
                        w['MT_GAR_DECES'] = acc_row.get('MT_GAR_DECES', np.nan)
                        w['MT_GAR_ECH'] = acc_row.get('MT_GAR_ECH', np.nan)
                        w['MT_SRG'] = acc_row.get('MT_SRG', np.nan)
                    w['AGE'] = r.get('AGE', np.nan)

                    w['PRIMES_GARANTIES'] = r.get('PRIMES_GARANTIES', np.nan)
                    w['PREST_DECES'] = r.get('PREST_DECES', np.nan)
                    w['PREST_ECH'] = r.get('PREST_ECH', np.nan)
                    w['PREST_MRV'] = r.get('PREST_MRV', np.nan)
                    w['FRAIS_ACQUIS'] = r.get('FRAIS_ACQUIS', np.nan)
                    w['COMM_VENTE'] = r.get('COMM_VENTE', np.nan)
                    w['PRIMES_VARIABLES'] = r.get('PRIMES_VARIABLES', np.nan)
                    w['FRAIS_FIXES'] = r.get('FRAIS_FIXES', np.nan)
                    w['HON_GEST'] = r.get('HON_GEST', np.nan)
                    w['COMM_MAINTIEN'] = r.get('COMM_MAINTIEN', np.nan)
                    
                    # Additional computed/derived columns
                    w['VALEUR_MARCHANDE'] = r.get('VALEUR_MARCHANDE', np.nan)
                    w['PASSIF_REDRESSE'] = r.get('PASSIF_REDRESSE', np.nan)
                    w['COUSSIN_CREDIT'] = r.get('COUSSIN_CREDIT', np.nan)
                    w['COUSSIN_MARCHE'] = r.get('COUSSIN_MARCHE', np.nan)
                    w['COUSSIN_DEPENSE'] = r.get('COUSSIN_DEPENSE', np.nan)
                    w['COUSSIN_DECHEANCE'] = r.get('COUSSIN_DECHEANCE', np.nan)
                    w['COUSSIN_MORTALITE'] = r.get('COUSSIN_MORTALITE', np.nan)
                    w['COUSSIN_DEPOT'] = r.get('COUSSIN_DEPOT', np.nan)
                    
                    # Computed fields from acc_row and current state
                    if acc_row is not None:
                        an_eval = r.get('AN_EVAL', 0)
                        mois_eval = r.get('MOIS_EVAL', 0)
                        age = r.get('AGE', 0)
                        
                        # Age-related columns
                        w['age_MORTALITE'] = age
                        w['AGE_RETRAIT'] = age
                        
                        # Year calculations
                        annee_eval_ini = acc_row.get('ANNEE_EVALUATION_INI', 2024)
                        w['annee_reelle'] = annee_eval_ini + an_eval if pd.notna(an_eval) else np.nan
                        
                        # Duration calculation
                        annee_cotis = acc_row.get('ANNEE_COTIS', 2024)
                        if pd.notna(an_eval) and pd.notna(annee_eval_ini) and pd.notna(annee_cotis):
                            duree = (annee_eval_ini + an_eval) - annee_cotis
                            w['duree_max10'] = min(duree, 10) if duree >= 0 else 0
                        
                        # VM/VG ratio
                        mt_vm_proj = r.get('MT_VM', 0)
                        mt_gar_deces = r.get('MT_GAR_DECES', 0)
                        mt_gar_ech = r.get('MT_GAR_ECH', 0)
                        vg = max(mt_gar_deces, mt_gar_ech) if pd.notna(mt_gar_deces) and pd.notna(mt_gar_ech) else 0
                        if vg > 0 and pd.notna(mt_vm_proj) and mt_vm_proj > 0:
                            w['VM_VG_RATIO'] = mt_vm_proj / vg
                        else:
                            w['VM_VG_RATIO'] = 0.0
                        
                        # Lapse levels (simplified - would need full lookup logic)
                        w['LAPSE_NIV_TOT'] = 1  # Default level
                        w['LAPSE_NIV_PART'] = 1  # Default level
                        w['LAPSE'] = r.get('LAPSE_TOT', 0)  # Total lapse
                        
                        # Projected guarantee columns
                        w['MT_BONI_DECES_PROJ'] = acc_row.get('MT_BONI_DECES', 0)
                        w['MT_BCB_PROJ'] = acc_row.get('MT_BCB', 0)
                        w['MT_MRV_MRG_MRA_PROJ'] = acc_row.get('MT_MRV_MRG_MRA', 0)
                        w['TAUX_MRV_MRG_MRA_PROJ'] = acc_row.get('TAUX_MRV_MRG_MRA', 0)
                        
                        # Echeance projections
                        w['ANNEE_ECH_PROJ'] = acc_row.get('ANNEE_ECH', np.nan)
                        w['MOIS_ECH_PROJ'] = acc_row.get('MOIS_ECH', np.nan)
                        
                        # Age factors (simplified)
                        w['FACTEUR_AGE_80'] = 1.0 if age < 80 else 0.0
                        w['FACTEUR_AGE_90'] = 1.0 if age < 90 else 0.0
                        
                        # MIN_FERR_PROJ (would need lookup)
                        w['MT_MIN_FERR_PROJ'] = 0.0
                        
                        # Actualization rates (simplified)
                        w['TX_ACTUALISATION'] = 0.0
                        w['TX_ACTUALISATION_DEB'] = 0.0
                        
                        # Internal scenario fields
                        w['scn_eval_int'] = np.nan
                        w['an_eval_int'] = np.nan
                        
                        # Adjustment fields
                        w['rc'] = 0.0
                        w['AJUST_NOUV_AFFAIRES'] = 0.0
                        w['MT_VM_AV_RETRAIT_FRAIS'] = r.get('MT_VM_AV_RETRAIT', np.nan)
                        
                        # Present value columns (VP_*) - set to 0 as placeholders
                        w['VP_PRIMES_GARANTIES'] = 0.0
                        w['VP_PREST_MRV'] = 0.0
                        w['VP_PREST_DECES'] = 0.0
                        w['VP_PREST_ECH'] = 0.0
                        w['VP_COMM_VENTE'] = 0.0
                        w['VP_FRAIS_ACQUIS'] = 0.0
                        w['VP_FRAIS_FIXES'] = 0.0
                        w['VP_HON_GEST'] = 0.0
                        w['VP_COMM_MAINTIEN'] = 0.0
                        w['VP_PRIMES_VARIABLES'] = 0.0
                        w['VP_FLUX_TOT'] = 0.0
                        w['VP_VALEUR_MARCHANDE'] = 0.0
                        w['VP_COUSSIN_DEPENSE'] = 0.0
                        w['VP_COUSSIN_DECHEANCE'] = 0.0
                        w['VP_COUSSIN_MORTALITE'] = 0.0
                        w['VP_COUSSIN_DEPOT'] = 0.0
                        w['VP_PASSIF_REDRESSE'] = 0.0
                        w['VP_COUSSIN_CREDIT'] = 0.0
                        w['VP_COUSSIN_MARCHE'] = 0.0
                        
                        # Additional computed fields
                        w['MT_SRG_AV_RETRAIT'] = r.get('MT_SRG', np.nan)
                        w['MT_VM_AP_RETRAIT_DEPOT'] = r.get('MT_VM_AP_RETRAIT', np.nan)
                        w['DEPOT_FUTUR_SURVIE'] = r.get('DEPOT_FUTUR', 0) * w.get('TX_SURVIE', 1.0) if pd.notna(r.get('DEPOT_FUTUR')) else 0.0
                        
                        # Commission/fee percentages from acquisition table (will be filled from lookup)
                        w['PC_COMMISSION_MAINTIEN'] = 0.0
                        w['PC_COMMISSION_VENTE'] = 0.0
                        w['PC_FRAIS_AN'] = 0.0
                        
                        # Category and coverage fields
                        w['UNITE_COUVERTURE'] = 1.0
                        w['VALEUR_GARANTIE'] = max(mt_gar_deces, mt_gar_ech) if pd.notna(mt_gar_deces) and pd.notna(mt_gar_ech) else 0.0
                        w['REM_COMP_INV'] = 0.0
                        w['CODE_CAT_PRODUIT'] = acc_row.get('ID_PRODUIT', 0)
                        w['CAT_COUSSIN_1'] = 0
                        w['CAT_COUSSIN_2'] = 0
                        
                        # Internal scenario fields (set to scenario index if available)
                        w['scn_eval_int'] = debug_scenario_idx + 1 if debug_scenario_idx is not None and debug_scenario_idx >= 0 else np.nan
                        w['an_eval_int'] = an_eval

                    if acc_row is not None:
                        for c in example_header:
                            if pd.isna(w.get(c, np.nan)) and c in acc_row.index:
                                w[c] = acc_row[c]

                    wide_rows.append(w)

                # Convert to DataFrame and fill NA columns from lookup tables
                output_df = pd.DataFrame(wide_rows, columns=example_header)
                
                if lookup_data is not None and acc_row is not None:
                    # Get account keys for lookups
                    id_lapse = acc_row.get('ID_LAPSE', 0)
                    id_acqui = acc_row.get('ID_ACQUI', 0)
                    id_depot = acc_row.get('ID_DEPOT', 0)
                    i_regime_2 = acc_row.get('I_REGIME_2', 0)
                    
                    # Fill MIN_FERR from min_ferr table (keyed by AGE)
                    if 'min_ferr' in lookup_data and 'MIN_FERR' in output_df.columns:
                        min_ferr_df = lookup_data['min_ferr']
                        if 'AGE' in min_ferr_df.columns and 'MIN_FERR' in min_ferr_df.columns:
                            age_to_minferr = dict(zip(min_ferr_df['AGE'], min_ferr_df['MIN_FERR']))
                            output_df['MIN_FERR'] = output_df['AGE'].map(age_to_minferr)
                    
                    # Fill TX_LAPSE_TOT columns from tx_lapse_tot table
                    if 'tx_lapse_tot' in lookup_data:
                        lapse_tot_df = lookup_data['tx_lapse_tot']
                        if 'ID_LAPSE' in lapse_tot_df.columns:
                            lapse_tot_row = lapse_tot_df[lapse_tot_df['ID_LAPSE'] == id_lapse]
                            if len(lapse_tot_row) > 0:
                                for col in ['TX_LAPSE_TOT_MIN', 'TX_LAPSE_TOT_MAX', 'FACT_DIM']:
                                    if col in lapse_tot_row.columns and col in output_df.columns:
                                        output_df[col] = lapse_tot_row[col].iloc[0]
                    
                    # Fill TX_LAPSE_PART columns from tx_lapse_part table
                    if 'tx_lapse_part' in lookup_data:
                        lapse_part_df = lookup_data['tx_lapse_part']
                        if 'ID_LAPSE' in lapse_part_df.columns:
                            lapse_part_row = lapse_part_df[lapse_part_df['ID_LAPSE'] == id_lapse]
                            if len(lapse_part_row) > 0:
                                for col in ['TX_LAPSE_PART_MIN', 'TX_LAPSE_PART_MAX']:
                                    if col in lapse_part_row.columns and col in output_df.columns:
                                        output_df[col] = lapse_part_row[col].iloc[0]
                    
                    # Fill ACQUISITION columns (PC_COMMISSION_*, PC_FRAIS_AN_*)
                    if 'acquisition' in lookup_data:
                        acq_df = lookup_data['acquisition']
                        if 'ID_ACQUI' in acq_df.columns:
                            acq_row = acq_df[acq_df['ID_ACQUI'] == id_acqui]
                            if len(acq_row) > 0:
                                for col in ['PC_COMMISSION_VENTE_RF', 'PC_COMMISSION_VENTE_AC', 
                                           'PC_COMMISSION_MAINTIEN_RF', 'PC_COMMISSION_MAINTIEN_AC',
                                           'PC_FRAIS_AN_AC', 'PC_FRAIS_AN_RF']:
                                    if col in acq_row.columns and col in output_df.columns:
                                        output_df[col] = acq_row[col].iloc[0]
                    
                    # Fill DEPOTS_FUTURS columns
                    if 'depots_futurs' in lookup_data:
                        depot_df = lookup_data['depots_futurs']
                        if 'ID_DEPOT' in depot_df.columns:
                            depot_row = depot_df[depot_df['ID_DEPOT'] == id_depot]
                            if len(depot_row) > 0:
                                for col in ['PC_DEPOT_ANNUEL', 'VAR_DEPOT_FCT', 'AGE_MAX_DEPOT', 'I_EVEN_CESSE_DEPOT']:
                                    if col in depot_row.columns and col in output_df.columns:
                                        output_df[col] = depot_row[col].iloc[0]
                    
                    # Fill FORWARD_RATE and AJUST_FORWARD_RATE_VM_0 from rendements
                    if 'rendements' in lookup_data:
                        rend_df = lookup_data['rendements']
                        if 'AN_EVAL' in rend_df.columns or 'an_eval' in rend_df.columns:
                            an_col = 'AN_EVAL' if 'AN_EVAL' in rend_df.columns else 'an_eval'
                            mois_col = 'MOIS_EVAL' if 'MOIS_EVAL' in rend_df.columns else 'mois_eval'
                            for idx, row in output_df.iterrows():
                                an_val = row.get('an_eval', np.nan)
                                mois_val = row.get('mois_eval', np.nan)
                                if pd.notna(an_val) and pd.notna(mois_val):
                                    rend_match = rend_df[(rend_df[an_col] == an_val) & (rend_df[mois_col] == mois_val)]
                                    if len(rend_match) > 0:
                                        if 'FORWARD_RATE' in rend_match.columns and pd.isna(output_df.loc[idx, 'FORWARD_RATE']):
                                            output_df.loc[idx, 'FORWARD_RATE'] = rend_match['FORWARD_RATE'].iloc[0]
                                        if 'AJUST_FORWARD_RATE_VM_0' in rend_match.columns and pd.isna(output_df.loc[idx, 'AJUST_FORWARD_RATE_VM_0']):
                                            output_df.loc[idx, 'AJUST_FORWARD_RATE_VM_0'] = rend_match['AJUST_FORWARD_RATE_VM_0'].iloc[0]
                    
                    # Fill FRAIS from frais_admin
                    if 'frais_admin' in lookup_data:
                        frais_df = lookup_data['frais_admin']
                        if 'FRAIS' in frais_df.columns and 'FRAIS' in output_df.columns:
                            output_df['FRAIS'] = frais_df['FRAIS'].iloc[0] if len(frais_df) > 0 else np.nan
                    
                    # Fill COUSSINS_ESCAP columns
                    if 'coussins_escap' in lookup_data:
                        coussin_df = lookup_data['coussins_escap']
                        coussin_cols = [c for c in coussin_df.columns if c.startswith('BASE_') or c.startswith('TX_')]
                        for col in coussin_cols:
                            if col in output_df.columns:
                                output_df[col] = coussin_df[col].iloc[0] if len(coussin_df) > 0 else np.nan

                output_example_gpu_path = output_path / "OUTPUT_EXAMPLE_GPU.csv"
                output_df.to_csv(output_example_gpu_path, index=False)
                print(f"✓ Saved OUTPUT_EXAMPLE_GPU.csv (matches output_example.csv schema; debug: account={debug_account_idx}, scenario={debug_scenario_idx})")
                saved_files.append("OUTPUT_EXAMPLE_GPU.csv (output_example.csv schema; partial fill)")
    
    # 1b. Five Chocs Results
    if results_5chocs_df is not None:
        print(f"\n✓ [FIVE_CHOCS] Saving five chocs results...")
        
        chocs_detailed_path = output_path / "VP_FLUX_5CHOCS_DETAILED_GPU.csv"
        results_5chocs_df.to_csv(chocs_detailed_path, index=False, sep=';')
        print(f"  Saved VP_FLUX_5CHOCS_DETAILED_GPU.csv")
        print(f"  Contains {len(results_5chocs_df)} rows (5 chocs × {n_accounts} accounts)")
        
        sensitivities_path = output_path / "VP_FLUX_SENSITIVITIES_GPU.csv"
        sensitivities_df.to_csv(sensitivities_path, index=False, sep=';')
        print(f"  Saved VP_FLUX_SENSITIVITIES_GPU.csv")
        print(f"  Contains {len(sensitivities_df)} rows with Greeks/Deltas")
        
        chocs_summary_df = results_5chocs_df.groupby('CHOC_TYPE').agg({
            'RESERVE_BE': ['sum', 'mean'],
            'CAPITAL_REQ': ['sum', 'mean'], 
            'SCR': ['sum', 'mean']
        }).round(2)
        chocs_summary_df.columns = ['_'.join(col).strip() for col in chocs_summary_df.columns]
        chocs_summary_df = chocs_summary_df.reset_index()
        
        chocs_summary_path = output_path / "VP_FLUX_5CHOCS_SUMMARY_GPU.csv"
        chocs_summary_df.to_csv(chocs_summary_path, index=False, sep=';')
        print(f"  Saved VP_FLUX_5CHOCS_SUMMARY_GPU.csv")
        
        print(f"\n  Key Portfolio Sensitivities (Total):")
        total_sensitivities = sensitivities_df.sum()
        print(f"    SP500 Delta (Reserve): ${total_sensitivities['DELTA_SP500_RESERVE']:,.2f}")
        print(f"    TSX Delta (Reserve):   ${total_sensitivities['DELTA_TSX_RESERVE']:,.2f}")
        print(f"    EAFE Delta (Reserve):  ${total_sensitivities['DELTA_EAFE_RESERVE']:,.2f}")
        print(f"    DEX Delta (Reserve):   ${total_sensitivities['DELTA_DEX_RESERVE']:,.2f}")
        
        saved_files.extend([
            "VP_FLUX_5CHOCS_DETAILED_GPU.csv (5 chocs × accounts)",
            "VP_FLUX_SENSITIVITIES_GPU.csv (Greeks/Deltas per account)",
            "VP_FLUX_5CHOCS_SUMMARY_GPU.csv (aggregated by choc type)"
        ])
    
    # ===========================================
    # 2. DEBUG OUTPUT (if enabled)
    # ===========================================
    
    if ext_debug is not None:
        print(f"\n✓ [DEBUG] Saving external kernel debug output...")
        
        # Create single-row DataFrame with debug filter context
        row = {}
        if debug_params:
            debug_account_idx = debug_params.get('account', -1)
            row['DEBUG_ACCOUNT_IDX'] = debug_account_idx
            # Map account index to real ID_COMPTE if available
            if population_ids is not None and debug_account_idx >= 0 and debug_account_idx < len(population_ids):
                row['ID_COMPTE'] = int(population_ids[debug_account_idx])
            else:
                row['ID_COMPTE'] = -1
            row['DEBUG_SCENARIO'] = debug_params.get('scenario', -1)
            row['DEBUG_YEAR'] = debug_params.get('year', -1)
            row['DEBUG_MONTH'] = debug_params.get('month', -1)
        
        for col_idx, col_name in enumerate(EXT_DEBUG_COLUMNS):
            row[col_name] = ext_debug[col_idx]
        
        ext_debug_df = pd.DataFrame([row])
        ext_debug_path = output_path / "DEBUG_EXTERNAL_KERNEL.csv"
        ext_debug_df.to_csv(ext_debug_path, index=False, sep=';')
        print(f"  Saved DEBUG_EXTERNAL_KERNEL.csv (1 row)")
        if debug_params:
            print(f"  Filter: account={debug_params.get('account', -1)}, scenario={debug_params.get('scenario', -1)}, year={debug_params.get('year', -1)}, month={debug_params.get('month', -1)}")
        saved_files.append("DEBUG_EXTERNAL_KERNEL.csv (external kernel debug)")
    
    if int_debug is not None:
        print(f"\n✓ [DEBUG] Saving internal kernel debug output...")
        n_chocs = int_debug.shape[0]
        
        rows = []
        for choc_idx in range(n_chocs):
            choc_name = CHOC_NAMES[choc_idx] if choc_idx < len(CHOC_NAMES) else f"CHOC_{choc_idx}"
            row = {
                'CHOC_IDX': choc_idx,
                'CHOC_NAME': choc_name,
            }
            if debug_params:
                debug_account_idx = debug_params.get('account', -1)
                row['DEBUG_ACCOUNT_IDX'] = debug_account_idx
                # Map account index to real ID_COMPTE if available
                if population_ids is not None and debug_account_idx >= 0 and debug_account_idx < len(population_ids):
                    row['ID_COMPTE'] = int(population_ids[debug_account_idx])
                else:
                    row['ID_COMPTE'] = -1
                row['DEBUG_INT_SCENARIO'] = debug_params.get('int_scenario', -1)
                row['DEBUG_INT_YEAR'] = debug_params.get('int_year', -1)
            
            for col_idx, col_name in enumerate(INT_DEBUG_COLUMNS):
                row[col_name] = int_debug[choc_idx, col_idx]
            rows.append(row)
        
        int_debug_df = pd.DataFrame(rows)
        int_debug_path = output_path / "DEBUG_INTERNAL_KERNEL.csv"
        int_debug_df.to_csv(int_debug_path, index=False, sep=';')
        print(f"  Saved DEBUG_INTERNAL_KERNEL.csv ({len(int_debug_df)} rows - one per choc)")
        if debug_params:
            print(f"  Filter: int_scenario={debug_params.get('int_scenario', -1)}, int_year={debug_params.get('int_year', -1)}")
        saved_files.append("DEBUG_INTERNAL_KERNEL.csv (internal kernel debug)")

    if int_debug_ts_df is not None and len(int_debug_ts_df) > 0:
        int_debug_ts_saved_df = int_debug_ts_df.copy()
        int_debug_ts_path = output_path / "DEBUG_INTERNAL_LOOP_TS.csv"
        int_debug_ts_saved_df.to_csv(int_debug_ts_path, index=False, sep=';')
        print(f"  Saved DEBUG_INTERNAL_LOOP_TS.csv ({len(int_debug_ts_saved_df)} rows)")
        saved_files.append("DEBUG_INTERNAL_LOOP_TS.csv (internal loop time series debug)")
    
    # ===========================================
    # 3. SUMMARY
    # ===========================================
    
    print("\n" + "=" * 80)
    print("FILE SAVING SUMMARY")
    print("=" * 80)
    for idx, file_name in enumerate(saved_files, 1):
        print(f"  {idx}. {file_name}")
    print("=" * 80)
    
    # Build return dictionary with all created DataFrames
    result = {
        'saved_files': saved_files,
        'vp_flux_total': vp_flux_total_df,
        'chocs_summary': chocs_summary_df if results_5chocs_df is not None else None,
        'ext_debug_df': ext_debug_df if ext_debug is not None else None,
        'int_debug_df': int_debug_df if int_debug is not None else None,
        'int_debug_ts_df': int_debug_ts_saved_df,
        'flux_projetes_df': flux_projetes_df,
    }
    
    return result


def create_all_lookup_tables(data: dict, nb_int_scenarios: int, nb_an_projection: int):
    """
    Create all CPU lookup tables from loaded data.
    
    Args:
        data: Dictionary containing loaded DataFrames (mortalite, rendements, etc.)
        nb_int_scenarios: Number of internal scenarios for risk-neutral tables
        nb_an_projection: Number of projection years
    
    Returns:
        Dictionary containing all lookup tables (CPU numpy arrays)
    """
    print("\nCreating CPU lookup tables...")
    
    lookups = {}
    
    lookups['mortality'] = create_gpu_mortality_lookup(data['mortalite'])
    (lookups['forward_rate'], lookups['ajust_forward'], lookups['rend_dex'], 
     lookups['rend_mm'], lookups['rend_tsx'], lookups['rend_sp500'], 
     lookups['rend_eafe']) = create_gpu_returns_lookup(data['rendements'])
    
    lookups['min_ferr'] = create_gpu_min_ferr_lookup(data['min_ferr'])
    lookups['lapse_part_min'], lookups['lapse_part_max'] = create_gpu_lapse_part_lookup(data['tx_lapse_part'])
    lookups['lapse_tot_min'], lookups['lapse_tot_max'], lookups['lapse_tot_fact'] = create_gpu_lapse_tot_lookup(data['tx_lapse_tot'])
    (lookups['deposits_pc'], lookups['deposits_var'], lookups['deposits_age_max'],
     lookups['deposits_i_even']) = create_gpu_deposits_lookup(data['depots_futurs'])
    lookups['fees'] = create_gpu_fees_lookup(data['frais_admin'])
    (lookups['acq_vente_rf'], lookups['acq_vente_ac'], lookups['acq_maintien_rf'], 
     lookups['acq_maintien_ac'], lookups['acq_frais_ac'], 
     lookups['acq_frais_rf']) = create_gpu_acquisition_lookup(data['acquisition'])

    lookups['coussins'] = create_gpu_coussins_lookup(data['coussins_escap'])
    
    print("✓ All CPU lookup tables created")
    
    # Create risk-neutral scenario tables
    print("\nCreating risk-neutral scenario tables...")
    if 'rendements_int' in data and data['rendements_int'] is not None and len(data['rendements_int']) > 0:
        (lookups['rn_forward_rate'], lookups['rn_ajust_forward'], lookups['rn_rend_dex'], lookups['rn_rend_mm'],
         lookups['rn_rend_tsx'], lookups['rn_rend_sp500'], lookups['rn_rend_eafe']) = create_gpu_rn_returns_lookup(
            data['rendements_int'], nb_int_scenarios, nb_an_projection
        )
    else:
        lookups['rn_forward_rate'] = np.full((nb_int_scenarios, nb_an_projection), RN_DEFAULT_FORWARD_RATE, dtype=np.float32)
        lookups['rn_ajust_forward'] = np.zeros((nb_int_scenarios, nb_an_projection), dtype=np.float32)
        lookups['rn_rend_dex'] = np.full((nb_int_scenarios, nb_an_projection), RN_DEFAULT_REND_DEX, dtype=np.float32)
        lookups['rn_rend_mm'] = np.full((nb_int_scenarios, nb_an_projection), RN_DEFAULT_REND_MM, dtype=np.float32)
        lookups['rn_rend_tsx'] = np.full((nb_int_scenarios, nb_an_projection), RN_DEFAULT_REND_TSX, dtype=np.float32)
        lookups['rn_rend_sp500'] = np.full((nb_int_scenarios, nb_an_projection), RN_DEFAULT_REND_SP500, dtype=np.float32)
        lookups['rn_rend_eafe'] = np.full((nb_int_scenarios, nb_an_projection), RN_DEFAULT_REND_EAFE, dtype=np.float32)
    
    print("✓ Risk-neutral tables created")
    
    return lookups


# Global list to keep cupy arrays alive (prevents garbage collection)
_cupy_array_refs = []


def _clear_cupy_refs():
    """Clear cupy array references to free GPU memory."""
    global _cupy_array_refs
    _cupy_array_refs.clear()


def _device_array_cupy(shape, dtype=np.float32):
    """Allocate an uninitialized device array using cupy.
    
    Uses cupy for allocation to work around numba-cuda bug
    where device_pointer_value incorrectly parses bytes as int.
    """
    import cupy as cp
    
    # Allocate empty array on GPU using cupy
    cp_arr = cp.empty(shape, dtype=dtype)
    
    # Keep reference to prevent garbage collection
    _cupy_array_refs.append(cp_arr)
    
    # Convert cupy array to numba device array using __cuda_array_interface__
    return cuda.as_cuda_array(cp_arr)


def _to_device_contiguous(arr):
    """Ensure array is C-contiguous before copying to GPU.
    
    Uses cupy for array transfer to work around numba-cuda bug
    where device_pointer_value incorrectly parses bytes as int.
    """
    import cupy as cp
    
    # Ensure array is C-contiguous and float32
    host_arr = np.ascontiguousarray(arr, dtype=np.float32)
    
    # Use cupy to transfer to GPU, then get numba device array view
    cp_arr = cp.asarray(host_arr)
    
    # Keep reference to prevent garbage collection
    _cupy_array_refs.append(cp_arr)
    
    # Convert cupy array to numba device array using __cuda_array_interface__
    return cuda.as_cuda_array(cp_arr)


def copy_lookups_to_gpu(lookups: dict):
    """
    Copy all CPU lookup tables to GPU memory.
    
    Args:
        lookups: Dictionary of CPU numpy arrays from create_all_lookup_tables()
    
    Returns:
        Dictionary containing grouped GPU device arrays as tuples
    """
    print("\nCopying lookup tables to GPU...")
    
    # Clear previous cupy references to free GPU memory
    _clear_cupy_refs()
    
    gpu_lookups = {}
    
    # Mortality table
    gpu_lookups['mortality'] = _to_device_contiguous(lookups['mortality'])
    
    # Returns lookups (7 arrays): forward_rate, ajust_forward, rend_dex, rend_mm, rend_tsx, rend_sp500, rend_eafe
    gpu_lookups['returns'] = (
        _to_device_contiguous(lookups['forward_rate']),
        _to_device_contiguous(lookups['ajust_forward']),
        _to_device_contiguous(lookups['rend_dex']),
        _to_device_contiguous(lookups['rend_mm']),
        _to_device_contiguous(lookups['rend_tsx']),
        _to_device_contiguous(lookups['rend_sp500']),
        _to_device_contiguous(lookups['rend_eafe']),
    )
    
    # Lapse lookups (6 arrays): min_ferr, lapse_part_min, lapse_part_max, lapse_tot_min, lapse_tot_max, lapse_tot_fact
    gpu_lookups['lapse'] = (
        _to_device_contiguous(lookups['min_ferr']),
        _to_device_contiguous(lookups['lapse_part_min']),
        _to_device_contiguous(lookups['lapse_part_max']),
        _to_device_contiguous(lookups['lapse_tot_min']),
        _to_device_contiguous(lookups['lapse_tot_max']),
        _to_device_contiguous(lookups['lapse_tot_fact']),
    )
    
    # Policy lookups (5 arrays): deposits_pc, deposits_var, deposits_age_max, deposits_i_even, fees
    gpu_lookups['policy'] = (
        _to_device_contiguous(lookups['deposits_pc']),
        _to_device_contiguous(lookups['deposits_var']),
        _to_device_contiguous(lookups['deposits_age_max']),
        _to_device_contiguous(lookups['deposits_i_even']),
        _to_device_contiguous(lookups['fees']),
    )
    
    # Commission lookups (6 arrays): acq_vente_rf, acq_vente_ac, acq_maintien_rf, acq_maintien_ac, acq_frais_ac, acq_frais_rf
    gpu_lookups['commission'] = (
        _to_device_contiguous(lookups['acq_vente_rf']),
        _to_device_contiguous(lookups['acq_vente_ac']),
        _to_device_contiguous(lookups['acq_maintien_rf']),
        _to_device_contiguous(lookups['acq_maintien_ac']),
        _to_device_contiguous(lookups['acq_frais_ac']),
        _to_device_contiguous(lookups['acq_frais_rf']),
    )

    gpu_lookups['coussins'] = tuple(_to_device_contiguous(arr) for arr in lookups['coussins'])
    
    # Risk-neutral returns (7 arrays): rn_forward_rate, rn_ajust_forward, rn_rend_dex, rn_rend_mm, rn_rend_tsx, rn_rend_sp500, rn_rend_eafe
    gpu_lookups['rn_returns'] = (
        _to_device_contiguous(lookups['rn_forward_rate']),
        _to_device_contiguous(lookups['rn_ajust_forward']),
        _to_device_contiguous(lookups['rn_rend_dex']),
        _to_device_contiguous(lookups['rn_rend_mm']),
        _to_device_contiguous(lookups['rn_rend_tsx']),
        _to_device_contiguous(lookups['rn_rend_sp500']),
        _to_device_contiguous(lookups['rn_rend_eafe']),
    )
    
    print("✓ Lookup tables on GPU")
    
    return gpu_lookups


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
        rendements_int_path: Optional[Path] = None,
        depots_futurs_path: Optional[Path] = None,
        frais_admin_path: Optional[Path] = None,
        min_ferr_path: Optional[Path] = None,
        tx_lapse_part_path: Optional[Path] = None,
        tx_lapse_tot_path: Optional[Path] = None,
        acquisition_path: Optional[Path] = None,
        coussins_escap_path: Optional[Path] = None,
        progress_callback: Optional[callable] = None,
        debug_account_id: Optional[int] = None,
        debug_account: int = -1,
        debug_scenario: int = -1,
        debug_year: int = -1,
        debug_month: int = -1,
        debug_int_scenario: int = -1,
        debug_int_year: int = -1,
        debug_only: bool = False):
    """
    Run GPU-accelerated nested stochastic projection using Two-Pass architecture.
    
    Architecture:
    - Kernel A (Generator): Runs external scenarios, outputs state tensors to VRAM
    - Kernel B (Valuator): Reads states, runs internal scenarios with 5 chocs, outputs reserves & capital
    
    Args:
        debug_only: If True and debug_account >= 0, only process the single account 
                   specified by debug_account (filters population to that account only).
    """
    start_time = datetime.now()
    print(f"Starting NESTED STOCHASTIC GPU projection at {start_time}")
    print("=" * 80)
    print(f"Architecture: Two-Pass (Generator → Valuator with 5 Chocs)")
    print(f"External scenarios: {nb_ext_scenarios}")
    print(f"Internal scenarios per node: {nb_int_scenarios}")
    sys.stdout.flush()
    print(f"Capital shock: {shock_capital_pct*100:.1f}%")
    sys.stdout.flush()
    enable_debug = debug_account >= 0 or debug_scenario >= 0 or debug_year >= 0 or debug_month >= 0
    if enable_debug:
        print(f"Debug mode: ENABLED (account={debug_account}, scenario={debug_scenario}, year={debug_year}, month={debug_month})")
        print(f"  Internal debug: int_scenario={debug_int_scenario}, int_year={debug_int_year}")
        if debug_only and debug_account >= 0:
            print(f"  DEBUG_ONLY: Will process ONLY account {debug_account}")
    else:
        print(f"Debug mode: disabled")
    print("=" * 80)
    sys.stdout.flush()
    
    # Initialize GPU
    print("Initializing GPU...")
    sys.stdout.flush()
    initialize_gpu()

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
                         coussins_escap_path=coussins_escap_path,
                         rendements_int_path=rendements_int_path)
    print("✓ Data loaded successfully")

    # Filter to single account if debug_only mode
    if debug_only and debug_account >= 0:
        # Find the account by ID (assuming there's an ID column like 'NO_COMPTE' or index)
        pop_df = data['population']
        if 'ID_COMPTE' in pop_df.columns:
            filtered = pop_df[pop_df['ID_COMPTE'] == debug_account]
        elif 'NO_COMPTE' in pop_df.columns:
            filtered = pop_df[pop_df['NO_COMPTE'] == debug_account]
        else:
            # Fall back to using index/row position
            if debug_account < len(pop_df):
                filtered = pop_df.iloc[[debug_account]]
            else:
                raise ValueError(f"debug_account {debug_account} is out of range (max: {len(pop_df)-1})")
        
        if len(filtered) == 0:
            raise ValueError(f"Account {debug_account} not found in population data")
        
        data['population'] = filtered.reset_index(drop=True)
        print(f"⚠️  DEBUG_ONLY: Filtered to single account {debug_account}")
        # Override max_accounts since we're only doing one
        max_accounts = None
    elif max_accounts:
        data['population'] = data['population'].head(max_accounts)

    if debug_account_id is not None:
        if 'ID_COMPTE' not in data['population'].columns:
            raise ValueError("Population data does not contain ID_COMPTE; cannot use debug_account_id")
        matches = np.where(data['population']['ID_COMPTE'].values == debug_account_id)[0]
        if len(matches) == 0:
            raise ValueError(
                f"debug_account_id={debug_account_id} not found in loaded population (after max_accounts/debug_only filtering). "
                f"Available ID_COMPTE examples: {data['population']['ID_COMPTE'].head(10).tolist()}"
            )
        debug_account = int(matches[0])
        enable_debug = True
        print(f"Debug account resolved: debug_account_id={debug_account_id} -> debug_account_index={debug_account} (0-based)")

    n_accounts = len(data['population'])
    print(f"\nPreparing {n_accounts} accounts for GPU processing...")

    # Prepare account data
    all_account_data, _ = prepare_account_data(data['population'])
    print("✓ Account data prepared")

    # Create all CPU lookup tables
    lookups = create_all_lookup_tables(data, nb_int_scenarios, nb_an_projection)

    # Calculate batch size
    batch_size, num_batches, total_mem_per_account, _ = calculate_batch_size(
        n_accounts, nb_ext_scenarios, nb_an_projection, 
        nb_int_scenarios, all_account_data.shape[1]
    )

    # Copy lookup tables to GPU
    gpu_lookups = copy_lookups_to_gpu(lookups)

    # Process batches
    print("\n" + "=" * 80)
    print("RUNNING TWO-PASS NESTED STOCHASTIC PROJECTION")
    print("=" * 80)
    
    all_reserves = []
    all_capital = []
    all_reserves_5chocs = []
    all_capital_5chocs = []
    ext_debug_result = None
    int_debug_result = None
    int_debug_ts_result = None
    debug_flux_result = None
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_accounts)
        batch_account_data = all_account_data[start_idx:end_idx]
        
        # Adjust debug_account for batch offset (only debug if account is in this batch)
        batch_debug_account = -1
        if debug_account >= 0:
            if start_idx <= debug_account < end_idx:
                batch_debug_account = debug_account - start_idx
        
        batch_result = process_batch(
            batch_account_data=batch_account_data,
            nb_ext_scenarios=nb_ext_scenarios,
            nb_an_projection=nb_an_projection,
            nb_int_scenarios=nb_int_scenarios,
            shock_capital_pct=shock_capital_pct,
            total_mem_per_account=total_mem_per_account,
            threads_per_block=threads_per_block,
            gpu_lookups=gpu_lookups,
            batch_idx=i,
            num_batches=num_batches,
            debug_account=batch_debug_account,
            debug_scenario=debug_scenario,
            debug_year=debug_year,
            debug_month=debug_month,
            debug_int_scenario=debug_int_scenario,
            debug_int_year=debug_int_year,
        )
        
        # Accumulate results
        all_reserves.extend(batch_result['batch_reserves'])
        all_capital.extend(batch_result['batch_capital'])
        all_reserves_5chocs.extend(batch_result['batch_reserves_5chocs'])
        all_capital_5chocs.extend(batch_result['batch_capital_5chocs'])
        
        # Store debug output (only one batch will have it if account filter is used)
        if batch_result['ext_debug'] is not None:
            ext_debug_result = batch_result['ext_debug']
        if batch_result['int_debug'] is not None:
            int_debug_result = batch_result['int_debug']
        if batch_result.get('int_debug_ts') is not None:
            int_debug_ts_result = batch_result['int_debug_ts']
        if batch_result.get('debug_flux') is not None:
            debug_flux_result = batch_result['debug_flux']
        
        # Call progress callback if provided
        if progress_callback is not None:
            progress_callback(i + 1, num_batches)
    
    # Create results DataFrames
    population_ids = data['population']['ID_COMPTE'].values
    results_df, results_5chocs_df, sensitivities_df = create_results_dataframes(
        population_ids=population_ids,
        all_reserves=all_reserves,
        all_capital=all_capital,
        all_reserves_5chocs=all_reserves_5chocs,
        all_capital_5chocs=all_capital_5chocs,
        n_accounts=n_accounts
    )
    
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
    
    # Build debug params if debug is enabled
    debug_params = None
    if enable_debug:
        debug_params = {
            'account': debug_account,
            'scenario': debug_scenario,
            'year': debug_year,
            'month': debug_month,
            'int_scenario': debug_int_scenario,
            'int_year': debug_int_year,
        }

    int_debug_ts_df = None
    if enable_debug and int_debug_ts_result is not None:
        rows = []
        max_years = int_debug_ts_result.shape[1]
        years_iter = [debug_int_year] if debug_int_year is not None and debug_int_year >= 0 else range(max_years)
        for choc_idx in range(int_debug_ts_result.shape[0]):
            choc_name = CHOC_NAMES[choc_idx] if choc_idx < len(CHOC_NAMES) else f"CHOC_{choc_idx}"
            for t_int in years_iter:
                if t_int < 0 or t_int >= max_years:
                    continue
                # Map account index to real ID_COMPTE
                id_compte = int(population_ids[debug_account]) if population_ids is not None and debug_account >= 0 and debug_account < len(population_ids) else -1
                rows.append({
                    'CHOC_IDX': choc_idx,
                    'CHOC_NAME': choc_name,
                    'T_INT': int(t_int),
                    'DEBUG_ACCOUNT_IDX': debug_account,
                    'ID_COMPTE': id_compte,
                    'DEBUG_SCENARIO': debug_scenario,
                    'DEBUG_YEAR': debug_year,
                    'DEBUG_INT_SCENARIO': debug_int_scenario,
                    'CURR_VM': float(int_debug_ts_result[choc_idx, t_int, INT_TS_DEBUG_IDX_CURR_VM]),
                    'FEES': float(int_debug_ts_result[choc_idx, t_int, INT_TS_DEBUG_IDX_FEES]),
                    'PV_PATH': float(int_debug_ts_result[choc_idx, t_int, INT_TS_DEBUG_IDX_PV_PATH]),
                    'R_PORTFOLIO': float(int_debug_ts_result[choc_idx, t_int, INT_TS_DEBUG_IDX_R_PORTFOLIO]),
                    'FWD_RATE': float(int_debug_ts_result[choc_idx, t_int, INT_TS_DEBUG_IDX_FWD_RATE]),
                    'DF': float(int_debug_ts_result[choc_idx, t_int, INT_TS_DEBUG_IDX_DF]),
                })
        if rows:
            int_debug_ts_df = pd.DataFrame(rows)
    
    # Save all results (including debug output if enabled)
    save_result = save_results(
        output_path=output_path,
        results_df=results_df,
        results_5chocs_df=results_5chocs_df,
        sensitivities_df=sensitivities_df,
        n_accounts=n_accounts,
        ext_debug=ext_debug_result,
        int_debug=int_debug_result,
        int_debug_ts_df=int_debug_ts_df,
        debug_params=debug_params,
        population_ids=population_ids,
        debug_flux=debug_flux_result,
        population_df=data.get('population'),
        lookup_data=data,
    )
    
    return ProjectionResult(
        results=results_df,
        results_5chocs=results_5chocs_df,
        sensitivities=sensitivities_df if results_5chocs_df is not None else None,
        total_duration=total_duration,
        vp_flux_total=save_result['vp_flux_total'],
        chocs_summary=save_result['chocs_summary'],
        ext_debug_df=save_result['ext_debug_df'],
        int_debug_df=save_result['int_debug_df'],
        int_debug_ts_df=save_result['int_debug_ts_df'],
        flux_projetes_df=save_result['flux_projetes_df'],
        saved_files=save_result['saved_files'],
    )


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Run GPU-accelerated actuarial projections',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic nested stochastic
  python gpu.py --ext-scenarios 100 --int-scenarios 500 --max-accounts 1000
  
  # Full production run
  python gpu.py --ext-scenarios 1000 --int-scenarios 1000 --years 100
  
  # Debug specific account/scenario/year/month
  python gpu.py --max-accounts 10 --debug-account 0 --debug-scenario 0 --debug-year 5 --debug-month 12
        """
    )

    parser.add_argument('--max-accounts', type=int, default=1,
                        help='Maximum number of accounts to process (for testing)')
    parser.add_argument('--years', type=int, default=3,
                        help='Number of years to project (default: 100)')

    # Nested mode parameters
    parser.add_argument('--ext-scenarios', type=int, default=1,
                        help='Number of external (real-world) scenarios for nested mode (default: 100)')
    parser.add_argument('--int-scenarios', type=int, default=5,
                        help='Number of internal (risk-neutral) scenarios per node for nested mode (default: 100)')
    parser.add_argument('--shock', type=float, default=0.35,
                        help='Capital shock percentage for nested mode (default: 0.35 = 35%%)')
    
    # Debug filter parameters
    parser.add_argument('--debug-account', type=int, default=0,
                        help='Account index (0-based row index) to debug (-1 = disabled)')
    parser.add_argument('--debug-account-id', type=int, default=None,
                        help='Account ID_COMPTE to debug (overrides --debug-account when provided)')
    parser.add_argument('--debug-scenario', type=int, default=0,
                        help='External scenario index to debug (-1 = disabled)')
    parser.add_argument('--debug-year', type=int, default=-1,
                        help='Year (an_eval) to debug (-1 = disabled)')
    parser.add_argument('--debug-month', type=int, default=-1,
                        help='Month (mois_eval) to debug (-1 = disabled)')
    parser.add_argument('--debug-int-scenario', type=int, default=0,
                        help='Internal scenario to debug (-1 = disabled)')
    parser.add_argument('--debug-int-year', type=int, default=1,
                        help='Internal year to debug (-1 = disabled)')

    args = parser.parse_args()
    
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

        results = run_projection_gpu_nested(
            data_path=DATA_PATH,
            output_path=OUTPUT_PATH,
            nb_an_projection=args.years,
            nb_ext_scenarios=args.ext_scenarios,
            nb_int_scenarios=args.int_scenarios,
            shock_capital_pct=args.shock,
            max_accounts=args.max_accounts,
            threads_per_block=(16, 16),
            debug_account_id=args.debug_account_id,
            debug_account=args.debug_account,
            debug_scenario=args.debug_scenario,
            debug_year=args.debug_year,
            debug_month=args.debug_month,
            debug_int_scenario=args.debug_int_scenario,
            debug_int_year=args.debug_int_year,
        )

        if results:
            print("\n" + "=" * 80)
            print("NESTED STOCHASTIC RESULTS")
            print("=" * 80)
            print("\nTop 10 accounts by SCR:")
            print(results.results.nlargest(10, 'SCR')[['ID_COMPTE', 'RESERVE_BE', 'CAPITAL_REQ', 'SCR']])

            print("\nSummary Statistics:")
            print(results.results[['RESERVE_BE', 'CAPITAL_REQ', 'SCR']].describe())

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()
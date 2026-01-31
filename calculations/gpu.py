import csv
import os
import math

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
    FLUX_COMP_IDX_CAT_COUSSIN_1,
    FLUX_COMP_IDX_CAT_COUSSIN_2,
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
    # Cashflow output tensor indices (matching SAS output)
    CF_OUT_IDX_FRAIS_ACQUIS, CF_OUT_IDX_COMM_VENTE, CF_OUT_IDX_PRIMES_GARANTIES,
    CF_OUT_IDX_PRIMES_VARIABLES, CF_OUT_IDX_FRAIS_FIXES, CF_OUT_IDX_HON_GEST,
    CF_OUT_IDX_COMM_MAINTIEN, CF_OUT_IDX_PREST_ECH, CF_OUT_IDX_PREST_MRV, CF_OUT_IDX_PREST_DECES,
    CF_OUT_IDX_VP_FRAIS_ACQUIS, CF_OUT_IDX_VP_COMM_VENTE, CF_OUT_IDX_VP_PRIMES_GARANTIES,
    CF_OUT_IDX_VP_PRIMES_VARIABLES, CF_OUT_IDX_VP_FRAIS_FIXES, CF_OUT_IDX_VP_HON_GEST,
    CF_OUT_IDX_VP_COMM_MAINTIEN, CF_OUT_IDX_VP_PREST_ECH, CF_OUT_IDX_VP_PREST_MRV,
    CF_OUT_IDX_VP_PREST_DECES, CF_OUT_IDX_VP_VALEUR_MARCHANDE,
    CF_OUT_IDX_UNITE_COUVERTURE, CF_OUT_IDX_DEPOT_FUTUR, CF_OUT_IDX_REM_COMP_INV,
    CF_OUT_IDX_VALEUR_MARCHANDE, CF_OUT_IDX_VALEUR_GARANTIE, CF_OUT_IDX_DEPOT_FUTUR_SURVIE,
    CF_OUT_IDX_PASSIF_REDRESSE, CF_OUT_IDX_COUSSIN_CREDIT, CF_OUT_IDX_COUSSIN_MARCHE,
    CF_OUT_IDX_COUSSIN_DEPENSE, CF_OUT_IDX_COUSSIN_DECHEANCE, CF_OUT_IDX_COUSSIN_MORTALITE,
    CF_OUT_IDX_COUSSIN_DEPOT,
    CF_OUT_IDX_VP_PASSIF_REDRESSE, CF_OUT_IDX_VP_COUSSIN_CREDIT, CF_OUT_IDX_VP_COUSSIN_MARCHE,
    CF_OUT_IDX_VP_COUSSIN_DEPENSE, CF_OUT_IDX_VP_COUSSIN_DECHEANCE, CF_OUT_IDX_VP_COUSSIN_MORTALITE,
    CF_OUT_IDX_VP_COUSSIN_DEPOT,
    CF_OUT_IDX_SIZE,
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
        
        # Configure RMM to use CUDA memory resource (no pooling) for predictable memory release
        try:
            import rmm
            rmm.reinitialize(managed_memory=False, pool_allocator=False)
        except (ImportError, Exception):
            pass
        
        # Configure CuPy to not use memory pool
        try:
            import cupy as cp
            cp.cuda.set_allocator(None)  # Use default CUDA allocator
            cp.cuda.set_pinned_memory_allocator(None)
        except (ImportError, Exception):
            pass
        
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
    
    # Cashflow tensor: (Batch, Ext_Scenarios, Years*12, CF_OUT_IDX_SIZE) - monthly data
    cf_mem_per_account = nb_ext_scenarios * nb_an_projection * 12 * CF_OUT_IDX_SIZE * 4
    
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
        # Use more conservative factor (55% instead of 60%) to account for memory fragmentation
        available_mem = max(0, (free_mem - lookup_overhead) * 0.55)
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
    batch_cashflows: Optional[np.ndarray]  # Full cashflow tensor (batch, scenarios, years, CF_OUT_IDX_SIZE)
    ext_debug: Optional[np.ndarray]  # Debug output from external kernel
    int_debug: Optional[np.ndarray]  # Debug output from internal kernel
    int_debug_ts: Optional[np.ndarray]  # Debug time series output from internal kernel


def check_gpu_memory(batch_size: int, mem_per_account: float, batch_idx: int = 0):
    """Log GPU memory status and raise if insufficient for batch."""
    try:
        # Force aggressive cleanup before checking memory
        if batch_idx > 0:
            cuda.synchronize()
            gc.collect()
            try:
                import rmm
                rmm.mr.get_current_device_resource().deallocate(0, 0)
            except (ImportError, AttributeError):
                pass
            cuda.synchronize()
        
        free_mem, _ = cuda.current_context().get_memory_info()
        estimated_mem = batch_size * mem_per_account
        logger.info(f"  Free GPU memory: {free_mem / 1024 ** 3:.2f} GB")
        logger.info(f"  Estimated batch memory: {estimated_mem / 1024 ** 3:.2f} GB")
        
        # Check if we have enough memory (simple check: need < available)
        if estimated_mem > free_mem:
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
    run_nested_valuation: bool = True,
) -> ProcessBatchResult:
    """
    Process a single batch through both kernels (or just Kernel A if run_nested_valuation=False).
    
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
        run_nested_valuation: If True, run Kernel B (nested valuation). If False, only run Kernel A (outer loop)
    
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
            (current_batch_size, nb_ext_scenarios, nb_an_projection * 12, CF_OUT_IDX_SIZE)
        )
        # Only allocate metrics tensor if running nested valuation
        if run_nested_valuation:
            d_metrics = _device_array_cupy(
                (current_batch_size, nb_ext_scenarios, nb_an_projection, NUM_CHOCS, METRICS_OUTPUT_SIZE)
            )
        else:
            d_metrics = None
        
        # Allocate debug arrays (always allocate, use -1 flags to disable)
        enable_ext_debug = debug_account >= 0 or debug_scenario >= 0 or debug_year >= 0 or debug_month >= 0
        enable_int_debug = enable_ext_debug and run_nested_valuation  # Internal debug only if external debug is enabled AND running nested valuation
        
        if enable_ext_debug:
            logger.info(f"  Debug mode: account={debug_account}, scenario={debug_scenario}, year={debug_year}, month={debug_month}")
        if enable_int_debug:
            logger.info(f"  Internal debug: int_scenario={debug_int_scenario}, int_year={debug_int_year}")
        
        # Always allocate debug arrays (kernel uses -1 flags to skip writing)
        d_ext_debug = _device_array_cupy((EXT_DEBUG_SIZE,))
        if run_nested_valuation:
            d_int_debug = _device_array_cupy((NUM_CHOCS, INT_DEBUG_SIZE))
        else:
            d_int_debug = None
        
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
        elif run_nested_valuation:
            # Always pass an array to the kernel to keep the CUDA signature stable.
            d_int_debug_ts = _to_device_contiguous(np.zeros((1, 1, INT_TS_DEBUG_IDX_SIZE), dtype=np.float32))
        else:
            d_int_debug_ts = None
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
    
    # Free states tensor immediately if not running nested valuation
    if not run_nested_valuation:
        del d_states
        cuda.synchronize()
    
    # === KERNEL B: NESTED VALUATOR WITH 5 CHOCS ===
    kernel_b_time = 0.0
    if run_nested_valuation:
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
    else:
        logger.info(f"  Kernel B skipped (run_nested_valuation=False - outer loop only)")
    
    # Copy results back
    logger.info("  Copying results to CPU...")
    
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
    
    # ==========================================================================
    # GPU-BASED AGGREGATIONS (SAS-compatible outputs)
    # Compute aggregations on GPU before copying to minimize data transfer
    # ==========================================================================
    import time as _time
    _t0 = _time.time()
    cuda.synchronize()  # Ensure GPU work is done before timing
    _t1 = _time.time()
    logger.info(f"  GPU sync before aggregations: {_t1 - _t0:.2f}s")
    
    # Convert d_cashflows to CuPy array for GPU aggregations
    # d_cashflows shape: (batch, scenarios, years, CF_OUT_IDX_SIZE)
    try:
        import cupy as cp
        d_cf_cupy = cp.asarray(d_cashflows)
        
        # --- VP_FLUX_COMPTE: Mean across scenarios, then sum across all months ---
        # Result: (batch, CF_OUT_IDX_SIZE) - one row per account with summed VP values
        _t_vp_start = _time.time()
        # Step 1: Mean across scenarios (axis=1)
        cf_mean_scenarios = cp.mean(d_cf_cupy, axis=1)  # (batch, months, CF_OUT_IDX_SIZE) where months=years*12
        # Step 2: Sum across all months (axis=1) to get total VP per account
        vp_flux_compte_gpu = cp.sum(cf_mean_scenarios, axis=1)  # (batch, CF_OUT_IDX_SIZE)
        h_vp_flux_compte = cp.asnumpy(vp_flux_compte_gpu)  # Copy small result to host
        _t_vp_end = _time.time()
        logger.info(f"  VP_FLUX_COMPTE GPU aggregation: {_t_vp_end - _t_vp_start:.4f}s")
        
        # --- FLUX_PROJETE: Mean across scenarios, sum across accounts ---
        # Result: (months, CF_OUT_IDX_SIZE) - one row per month with summed values across all accounts
        _t_flux_start = _time.time()
        # Step 1: Mean across scenarios (already computed above as cf_mean_scenarios)
        # Step 2: Sum across accounts (axis=0)
        flux_projete_gpu = cp.sum(cf_mean_scenarios, axis=0)  # (months, CF_OUT_IDX_SIZE) where months=years*12
        h_flux_projete = cp.asnumpy(flux_projete_gpu)  # Copy small result to host
        _t_flux_end = _time.time()
        logger.info(f"  FLUX_PROJETE GPU aggregation: {_t_flux_end - _t_flux_start:.4f}s")
        
        # Clean up intermediate GPU arrays
        del cf_mean_scenarios, vp_flux_compte_gpu, flux_projete_gpu, d_cf_cupy
        
    except Exception as e:
        logger.warning(f"  GPU aggregation failed, falling back to CPU: {e}")
        h_vp_flux_compte = None
        h_flux_projete = None
    
    # Copy full cashflows to host (still needed for FLUX_PROJETE_GPU.csv detailed output)
    _t2 = _time.time()
    h_cashflows = d_cashflows.copy_to_host()
    _t3 = _time.time()
    cf_size_gb = h_cashflows.nbytes / (1024**3)
    logger.info(f"  Cashflow copy_to_host: {_t3 - _t2:.2f}s for {cf_size_gb:.2f} GB ({cf_size_gb / (_t3 - _t2 + 0.001):.2f} GB/s)")
    
    # Process metrics (only if nested valuation was run)
    if run_nested_valuation:
        h_metrics = d_metrics.copy_to_host()
        batch_reserves_5chocs = h_metrics[:, :, :, :, METRICS_RESERVE_IDX].mean(axis=(1, 2))
        batch_capital_5chocs = h_metrics[:, :, :, :, METRICS_CAPITAL_IDX].mean(axis=(1, 2))
        batch_reserves = batch_reserves_5chocs[:, 0]
        batch_capital = batch_capital_5chocs[:, 0]
    else:
        # Outer loop only - compute simple PV-based reserves from Kernel A cashflows
        logger.info("  Computing simple PV-based reserves from external scenarios...")
        h_metrics = None
        
        del d_cashflows  # Free GPU memory immediately
        cuda.synchronize()
        
        # Compute simple reserve estimate by summing VP cashflows across years, avg over scenarios
        # h_cashflows shape: (batch, scenarios, years, CF_OUT_IDX_SIZE)
        # Sum VP cashflows: VP_FRAIS_ACQUIS + VP_COMM_VENTE + VP_PRIMES_GARANTIES + ... + VP_PREST_DECES
        vp_total = (
            h_cashflows[:, :, :, CF_OUT_IDX_VP_FRAIS_ACQUIS] +
            h_cashflows[:, :, :, CF_OUT_IDX_VP_COMM_VENTE] +
            h_cashflows[:, :, :, CF_OUT_IDX_VP_PRIMES_GARANTIES] +
            h_cashflows[:, :, :, CF_OUT_IDX_VP_PRIMES_VARIABLES] +
            h_cashflows[:, :, :, CF_OUT_IDX_VP_FRAIS_FIXES] +
            h_cashflows[:, :, :, CF_OUT_IDX_VP_HON_GEST] +
            h_cashflows[:, :, :, CF_OUT_IDX_VP_COMM_MAINTIEN] +
            h_cashflows[:, :, :, CF_OUT_IDX_VP_PREST_ECH] +
            h_cashflows[:, :, :, CF_OUT_IDX_VP_PREST_MRV] +
            h_cashflows[:, :, :, CF_OUT_IDX_VP_PREST_DECES]
        )
        batch_reserves = vp_total.sum(axis=2).mean(axis=1)  # Sum over years, avg over scenarios
        
        # No capital calculation without nested valuation
        batch_capital = np.zeros(current_batch_size, dtype=np.float32)
        batch_reserves_5chocs = np.zeros((current_batch_size, NUM_CHOCS), dtype=np.float32)
        batch_capital_5chocs = np.zeros((current_batch_size, NUM_CHOCS), dtype=np.float32)
        
        # Store base reserves in first choc position for consistency
        batch_reserves_5chocs[:, 0] = batch_reserves
        
        gc.collect()
    
    # Cleanup
    del d_batch_accounts, d_ext_debug, d_debug_flux
    if not run_nested_valuation:
        # d_states and d_cashflows already deleted in outer-loop-only branch
        pass
    else:
        del d_states, d_cashflows
    if d_metrics is not None:
        del d_metrics
    if d_int_debug is not None:
        del d_int_debug
    if d_int_debug_ts is not None:
        del d_int_debug_ts
    cuda.synchronize()
    if h_metrics is not None:
        del h_metrics
    gc.collect()
    
    # Clear CuPy array references to allow garbage collection
    _clear_cupy_refs()
    gc.collect()
    
    # Force RMM/CuPy to release memory back to CUDA
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except (ImportError, AttributeError):
        pass
    
    try:
        import rmm
        rmm.mr.get_current_device_resource().deallocate(0, 0)
    except (ImportError, AttributeError):
        pass
    
    cuda.synchronize()
    gc.collect()
    
    batch_time = (datetime.now() - batch_start).total_seconds()
    logger.info(f"  Batch complete: {batch_time:.2f}s (KernelA: {kernel_a_time:.2f}s, KernelB: {kernel_b_time:.2f}s)")
    
    return {
        'batch_reserves': batch_reserves,
        'batch_capital': batch_capital,
        'batch_reserves_5chocs': batch_reserves_5chocs,
        'batch_capital_5chocs': batch_capital_5chocs,
        'batch_cashflows': h_cashflows,
        'batch_vp_flux_compte': h_vp_flux_compte,  # GPU-aggregated VP by account
        'batch_flux_projete': h_flux_projete,      # GPU-aggregated flux by year
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


def write_cashflows_batch(
    output_path: Path,
    batch_cashflows: np.ndarray,
    population_ids: np.ndarray,
    start_idx: int,
    is_first_batch: bool = False,
):
    """
    Write a batch of cashflows to FLUX_PROJETE_GPU.csv incrementally.
    
    Args:
        output_path: Directory to save CSV file
        batch_cashflows: Cashflow tensor (batch_size, n_scenarios, n_years, CF_OUT_IDX_SIZE)
        population_ids: Array of account IDs
        start_idx: Starting index in population_ids for this batch
        is_first_batch: If True, write header; otherwise append
    """
    # Take mean across scenarios (axis=1) to match SAS proc summary
    # Result shape: (batch_size, n_months, CF_OUT_IDX_SIZE) where n_months = n_years * 12
    mean_cashflows = batch_cashflows.mean(axis=1)
    
    batch_size = mean_cashflows.shape[0]
    n_months_out = mean_cashflows.shape[1]
    
    flux_rows = []
    for batch_idx in range(batch_size):
        acc_idx = start_idx + batch_idx
        if acc_idx >= len(population_ids):
            break
        id_compte = int(population_ids[acc_idx])
        
        for month_idx in range(n_months_out):
            # Calculate year and month from monthly index
            an_eval = (month_idx // 12) + 1  # Year starts at 1
            mois_eval = (month_idx % 12) + 1  # Month 1-12
            cf = mean_cashflows[batch_idx, month_idx, :]
            
            # Skip if all zeros (no data)
            if np.all(cf == 0):
                continue
            
            flux_rows.append({
                'ID_COMPTE': id_compte,
                'AN_EVAL': an_eval,
                'MOIS_EVAL': mois_eval,  # Monthly output (mois_eval=1-12)
                # Non-discounted cashflows
                'FRAIS_ACQUIS': float(cf[CF_OUT_IDX_FRAIS_ACQUIS]),
                'COMM_VENTE': float(cf[CF_OUT_IDX_COMM_VENTE]),
                'PRIMES_GARANTIES': float(cf[CF_OUT_IDX_PRIMES_GARANTIES]),
                'PRIMES_VARIABLES': float(cf[CF_OUT_IDX_PRIMES_VARIABLES]),
                'FRAIS_FIXES': float(cf[CF_OUT_IDX_FRAIS_FIXES]),
                'HON_GEST': float(cf[CF_OUT_IDX_HON_GEST]),
                'COMM_MAINTIEN': float(cf[CF_OUT_IDX_COMM_MAINTIEN]),
                'PREST_ECH': float(cf[CF_OUT_IDX_PREST_ECH]),
                'PREST_MRV': float(cf[CF_OUT_IDX_PREST_MRV]),
                'PREST_DECES': float(cf[CF_OUT_IDX_PREST_DECES]),
                # Present value cashflows
                'VP_FRAIS_ACQUIS': float(cf[CF_OUT_IDX_VP_FRAIS_ACQUIS]),
                'VP_COMM_VENTE': float(cf[CF_OUT_IDX_VP_COMM_VENTE]),
                'VP_PRIMES_GARANTIES': float(cf[CF_OUT_IDX_VP_PRIMES_GARANTIES]),
                'VP_PRIMES_VARIABLES': float(cf[CF_OUT_IDX_VP_PRIMES_VARIABLES]),
                'VP_FRAIS_FIXES': float(cf[CF_OUT_IDX_VP_FRAIS_FIXES]),
                'VP_HON_GEST': float(cf[CF_OUT_IDX_VP_HON_GEST]),
                'VP_COMM_MAINTIEN': float(cf[CF_OUT_IDX_VP_COMM_MAINTIEN]),
                'VP_PREST_ECH': float(cf[CF_OUT_IDX_VP_PREST_ECH]),
                'VP_PREST_MRV': float(cf[CF_OUT_IDX_VP_PREST_MRV]),
                'VP_PREST_DECES': float(cf[CF_OUT_IDX_VP_PREST_DECES]),
                'VP_VALEUR_MARCHANDE': float(cf[CF_OUT_IDX_VP_VALEUR_MARCHANDE]),
                # Coverage and values
                'UNITE_COUVERTURE': float(cf[CF_OUT_IDX_UNITE_COUVERTURE]),
                'DEPOT_FUTUR': float(cf[CF_OUT_IDX_DEPOT_FUTUR]),
                'REM_COMP_INV': float(cf[CF_OUT_IDX_REM_COMP_INV]),
                'VALEUR_MARCHANDE': float(cf[CF_OUT_IDX_VALEUR_MARCHANDE]),
                'VALEUR_GARANTIE': float(cf[CF_OUT_IDX_VALEUR_GARANTIE]),
                'DEPOT_FUTUR_SURVIE': float(cf[CF_OUT_IDX_DEPOT_FUTUR_SURVIE]),
                # Cushions (non-discounted)
                'PASSIF_REDRESSE': float(cf[CF_OUT_IDX_PASSIF_REDRESSE]),
                'COUSSIN_CREDIT': float(cf[CF_OUT_IDX_COUSSIN_CREDIT]),
                'COUSSIN_MARCHE': float(cf[CF_OUT_IDX_COUSSIN_MARCHE]),
                'COUSSIN_DEPENSE': float(cf[CF_OUT_IDX_COUSSIN_DEPENSE]),
                'COUSSIN_DECHEANCE': float(cf[CF_OUT_IDX_COUSSIN_DECHEANCE]),
                'COUSSIN_MORTALITE': float(cf[CF_OUT_IDX_COUSSIN_MORTALITE]),
                'COUSSIN_DEPOT': float(cf[CF_OUT_IDX_COUSSIN_DEPOT]),
                # Cushions (present value)
                'VP_PASSIF_REDRESSE': float(cf[CF_OUT_IDX_VP_PASSIF_REDRESSE]),
                'VP_COUSSIN_CREDIT': float(cf[CF_OUT_IDX_VP_COUSSIN_CREDIT]),
                'VP_COUSSIN_MARCHE': float(cf[CF_OUT_IDX_VP_COUSSIN_MARCHE]),
                'VP_COUSSIN_DEPENSE': float(cf[CF_OUT_IDX_VP_COUSSIN_DEPENSE]),
                'VP_COUSSIN_DECHEANCE': float(cf[CF_OUT_IDX_VP_COUSSIN_DECHEANCE]),
                'VP_COUSSIN_MORTALITE': float(cf[CF_OUT_IDX_VP_COUSSIN_MORTALITE]),
                'VP_COUSSIN_DEPOT': float(cf[CF_OUT_IDX_VP_COUSSIN_DEPOT]),
            })
    
    if flux_rows:
        df = pd.DataFrame(flux_rows)
        flux_path = output_path / "FLUX_PROJETE_GPU.csv"
        
        if is_first_batch:
            # Write with header
            df.to_csv(flux_path, index=False, sep=';', mode='w')
        else:
            # Append without header
            df.to_csv(flux_path, index=False, sep=';', mode='a', header=False)
        
        return len(flux_rows)
    return 0


def write_vp_flux_compte_batch(
    output_path: Path,
    batch_vp_flux_compte: np.ndarray,
    population_ids: np.ndarray,
    start_idx: int,
    is_first_batch: bool = False,
):
    """
    Write a batch of VP_FLUX_COMPTE (VP by account) to CSV incrementally.
    
    This matches SAS: PROC SUMMARY ... CLASS ID_COMPTE; VAR VP_*; OUTPUT SUM=
    
    Args:
        output_path: Directory to save CSV file
        batch_vp_flux_compte: VP aggregates per account (batch_size, CF_OUT_IDX_SIZE)
        population_ids: Array of account IDs
        start_idx: Starting index in population_ids for this batch
        is_first_batch: If True, write header; otherwise append
    """
    if batch_vp_flux_compte is None:
        return 0
    
    batch_size = batch_vp_flux_compte.shape[0]
    
    rows = []
    for batch_idx in range(batch_size):
        acc_idx = start_idx + batch_idx
        if acc_idx >= len(population_ids):
            break
        id_compte = int(population_ids[acc_idx])
        vp = batch_vp_flux_compte[batch_idx, :]
        
        rows.append({
            'ID_COMPTE': id_compte,
            # VP columns (matching SAS VP_FLUX_COMPTE output)
            'VP_FRAIS_ACQUIS': float(vp[CF_OUT_IDX_VP_FRAIS_ACQUIS]),
            'VP_COMM_VENTE': float(vp[CF_OUT_IDX_VP_COMM_VENTE]),
            'VP_PRIMES_GARANTIES': float(vp[CF_OUT_IDX_VP_PRIMES_GARANTIES]),
            'VP_PRIMES_VARIABLES': float(vp[CF_OUT_IDX_VP_PRIMES_VARIABLES]),
            'VP_FRAIS_FIXES': float(vp[CF_OUT_IDX_VP_FRAIS_FIXES]),
            'VP_HON_GEST': float(vp[CF_OUT_IDX_VP_HON_GEST]),
            'VP_COMM_MAINTIEN': float(vp[CF_OUT_IDX_VP_COMM_MAINTIEN]),
            'VP_PREST_ECH': float(vp[CF_OUT_IDX_VP_PREST_ECH]),
            'VP_PREST_MRV': float(vp[CF_OUT_IDX_VP_PREST_MRV]),
            'VP_PREST_DECES': float(vp[CF_OUT_IDX_VP_PREST_DECES]),
            'VP_PASSIF_REDRESSE': float(vp[CF_OUT_IDX_VP_PASSIF_REDRESSE]),
            'VP_COUSSIN_CREDIT': float(vp[CF_OUT_IDX_VP_COUSSIN_CREDIT]),
            'VP_COUSSIN_MARCHE': float(vp[CF_OUT_IDX_VP_COUSSIN_MARCHE]),
            'VP_COUSSIN_DEPENSE': float(vp[CF_OUT_IDX_VP_COUSSIN_DEPENSE]),
            'VP_COUSSIN_DECHEANCE': float(vp[CF_OUT_IDX_VP_COUSSIN_DECHEANCE]),
            'VP_COUSSIN_MORTALITE': float(vp[CF_OUT_IDX_VP_COUSSIN_MORTALITE]),
            'VP_COUSSIN_DEPOT': float(vp[CF_OUT_IDX_VP_COUSSIN_DEPOT]),
            'VP_VALEUR_MARCHANDE': float(vp[CF_OUT_IDX_VP_VALEUR_MARCHANDE]),
        })
    
    if rows:
        df = pd.DataFrame(rows)
        vp_path = output_path / "VP_FLUX_COMPTE_GPU.csv"
        
        if is_first_batch:
            df.to_csv(vp_path, index=False, sep=';', mode='w')
        else:
            df.to_csv(vp_path, index=False, sep=';', mode='a', header=False)
        
        return len(rows)
    return 0


def accumulate_flux_projete(
    accumulated: Optional[np.ndarray],
    batch_flux_projete: np.ndarray,
) -> np.ndarray:
    """
    Accumulate FLUX_PROJETE across batches (sum by year).
    
    This matches SAS: PROC SUMMARY ... CLASS AN_EVAL MOIS_EVAL; OUTPUT SUM=
    
    Args:
        accumulated: Previously accumulated flux (years, CF_OUT_IDX_SIZE) or None
        batch_flux_projete: This batch's flux by year (years, CF_OUT_IDX_SIZE)
    
    Returns:
        Updated accumulated flux
    """
    if batch_flux_projete is None:
        return accumulated
    
    if accumulated is None:
        return batch_flux_projete.copy()
    else:
        return accumulated + batch_flux_projete


def write_flux_projete(
    output_path: Path,
    flux_projete: np.ndarray,
    nb_an_projection: int,
):
    """
    Write final FLUX_PROJETE (flux by year/month) to CSV.
    
    This matches SAS: PROC SUMMARY ... CLASS AN_EVAL MOIS_EVAL; OUTPUT SUM=
    
    Args:
        output_path: Directory to save CSV file
        flux_projete: Aggregated flux by month (months, CF_OUT_IDX_SIZE) where months = years * 12
        nb_an_projection: Number of projection years
    """
    if flux_projete is None:
        return 0
    
    rows = []
    n_months = min(flux_projete.shape[0], nb_an_projection * 12)
    
    for month_idx in range(n_months):
        # Calculate year and month from monthly index
        an_eval = (month_idx // 12) + 1  # Year starts at 1
        mois_eval = (month_idx % 12) + 1  # Month 1-12
        cf = flux_projete[month_idx, :]
        
        # Skip if all zeros
        if np.all(cf == 0):
            continue
        
        rows.append({
            'AN_EVAL': an_eval,
            'MOIS_EVAL': mois_eval,  # Monthly output (1-12)
            # Non-discounted cashflows
            'FRAIS_ACQUIS': float(cf[CF_OUT_IDX_FRAIS_ACQUIS]),
            'COMM_VENTE': float(cf[CF_OUT_IDX_COMM_VENTE]),
            'PRIMES_GARANTIES': float(cf[CF_OUT_IDX_PRIMES_GARANTIES]),
            'PRIMES_VARIABLES': float(cf[CF_OUT_IDX_PRIMES_VARIABLES]),
            'FRAIS_FIXES': float(cf[CF_OUT_IDX_FRAIS_FIXES]),
            'HON_GEST': float(cf[CF_OUT_IDX_HON_GEST]),
            'COMM_MAINTIEN': float(cf[CF_OUT_IDX_COMM_MAINTIEN]),
            'PREST_ECH': float(cf[CF_OUT_IDX_PREST_ECH]),
            'PREST_MRV': float(cf[CF_OUT_IDX_PREST_MRV]),
            'PREST_DECES': float(cf[CF_OUT_IDX_PREST_DECES]),
            'UNITE_COUVERTURE': float(cf[CF_OUT_IDX_UNITE_COUVERTURE]),
            'DEPOT_FUTUR': float(cf[CF_OUT_IDX_DEPOT_FUTUR]),
            'REM_COMP_INV': float(cf[CF_OUT_IDX_REM_COMP_INV]),
            'VALEUR_MARCHANDE': float(cf[CF_OUT_IDX_VALEUR_MARCHANDE]),
            'VALEUR_GARANTIE': float(cf[CF_OUT_IDX_VALEUR_GARANTIE]),
            'DEPOT_FUTUR_SURVIE': float(cf[CF_OUT_IDX_DEPOT_FUTUR_SURVIE]),
            'PASSIF_REDRESSE': float(cf[CF_OUT_IDX_PASSIF_REDRESSE]),
            'COUSSIN_CREDIT': float(cf[CF_OUT_IDX_COUSSIN_CREDIT]),
            'COUSSIN_MARCHE': float(cf[CF_OUT_IDX_COUSSIN_MARCHE]),
            'COUSSIN_DEPENSE': float(cf[CF_OUT_IDX_COUSSIN_DEPENSE]),
            'COUSSIN_DECHEANCE': float(cf[CF_OUT_IDX_COUSSIN_DECHEANCE]),
            'COUSSIN_MORTALITE': float(cf[CF_OUT_IDX_COUSSIN_MORTALITE]),
            'COUSSIN_DEPOT': float(cf[CF_OUT_IDX_COUSSIN_DEPOT]),
        })
    
    if rows:
        df = pd.DataFrame(rows)
        flux_path = output_path / "FLUX_PROJETES_GPU.csv"
        df.to_csv(flux_path, index=False, sep=';', mode='w')
        return len(rows)
    return 0


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
    total_cashflow_rows: int = 0,
    total_vp_flux_compte_rows: int = 0,
    total_flux_projete_rows: int = 0,
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

    # 1b. FLUX_PROJETE_GPU.csv - already written incrementally during batch processing
    if total_cashflow_rows > 0:
        print(f"\n✓ [FLUX_PROJETE] FLUX_PROJETE_GPU.csv written incrementally during batch processing")
        print(f"  Contains {total_cashflow_rows} rows (SAS DONNEES_COMPTE equivalent - MEAN by account/year)")
        saved_files.append("FLUX_PROJETE_GPU.csv (SAS DONNEES_COMPTE - cashflows by account/year)")
    
    # 1c. VP_FLUX_COMPTE_GPU.csv - already written incrementally during batch processing
    if total_vp_flux_compte_rows > 0:
        print(f"\n✓ [VP_FLUX_COMPTE] VP_FLUX_COMPTE_GPU.csv written incrementally during batch processing")
        print(f"  Contains {total_vp_flux_compte_rows} rows (SAS VP_FLUX_COMPTE equivalent - SUM VP by account)")
        saved_files.append("VP_FLUX_COMPTE_GPU.csv (SAS VP_FLUX_COMPTE - VP sums by account)")
    
    # 1d. FLUX_PROJETES_GPU.csv - already written after batch processing
    if total_flux_projete_rows > 0:
        print(f"\n✓ [FLUX_PROJETES] FLUX_PROJETES_GPU.csv written after batch processing")
        print(f"  Contains {total_flux_projete_rows} rows (SAS FLUX_PROJETE equivalent - SUM by year)")
        saved_files.append("FLUX_PROJETES_GPU.csv (SAS FLUX_PROJETE - totals by year)")

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
                    'CAT_COUSSIN_1': int(flux_row[FLUX_COMP_IDX_CAT_COUSSIN_1]),
                    'CAT_COUSSIN_2': int(flux_row[FLUX_COMP_IDX_CAT_COUSSIN_2]),
                })
        
        if rows:
            flux_projetes_df = pd.DataFrame(rows)
            flux_projetes_debug_path = output_path / "FLUX_PROJETES_GPU_DEBUG.csv"
            flux_projetes_df.to_csv(flux_projetes_debug_path, index=False, sep=';')
            print(f"✓ Saved FLUX_PROJETES_GPU_DEBUG.csv (debug: account={debug_account_idx}, scenario={debug_scenario_idx}, ID_COMPTE={id_compte})")
            saved_files.append("FLUX_PROJETES_GPU_DEBUG.csv (single account/scenario flux)")

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
                # Add initialization row (an_eval=0, mois_eval=12) like ground truth
                # Ground truth first row has many NaN fields - only specific fields are populated
                if acc_row is not None:
                    init_row = {c: np.nan for c in example_header}
                    # Only these fields have values in ground truth init row:
                    init_row['FORWARD_RATE'] = 0.0
                    # Projected values (from account initial state)
                    init_row['MT_SP500_PROJ'] = acc_row.get('MT_SP500', np.nan)
                    init_row['MT_TSX_PROJ'] = acc_row.get('MT_TSX', np.nan)
                    init_row['MT_EAFE_PROJ'] = acc_row.get('MT_EAFE', np.nan)
                    init_row['MT_DEX_PROJ'] = acc_row.get('MT_DEX', np.nan)
                    init_row['MT_MM_PROJ'] = acc_row.get('MT_MM', 0)
                    init_row['MT_VM_PROJ'] = acc_row.get('MT_VM', np.nan)
                    init_row['MT_GAR_ECH_PROJ'] = acc_row.get('MT_GAR_ECH', np.nan)
                    init_row['MT_GAR_DECES_PROJ'] = acc_row.get('MT_GAR_DECES', np.nan)
                    init_row['MT_BONI_DECES_PROJ'] = acc_row.get('MT_BONI_DECES', 0)
                    init_row['ANNEE_ECH_PROJ'] = acc_row.get('ANNEE_ECH', np.nan)
                    init_row['MOIS_ECH_PROJ'] = acc_row.get('MOIS_ECH', np.nan)
                    init_row['TX_SURVIE'] = 1.0
                    init_row['MT_SRG_PROJ'] = acc_row.get('MT_SRG', 0)
                    init_row['MT_BCB_PROJ'] = acc_row.get('MT_BCB', 0)
                    init_row['MT_MRV_MRG_MRA_PROJ'] = acc_row.get('MT_MRV_MRG_MRA', 0)
                    init_row['TAUX_MRV_MRG_MRA_PROJ'] = acc_row.get('TAUX_MRV_MRG_MRA', 0)
                    init_row['MT_MIN_FERR_PROJ'] = 0.0
                    init_row['TX_ACTUALISATION'] = 1.0
                    # Initial market values (same as projected for init row)
                    init_row['MT_VM'] = acc_row.get('MT_VM', np.nan)
                    init_row['MT_DEX'] = acc_row.get('MT_DEX', np.nan)
                    init_row['MT_SP500'] = acc_row.get('MT_SP500', np.nan)
                    init_row['MT_TSX'] = acc_row.get('MT_TSX', np.nan)
                    init_row['MT_EAFE'] = acc_row.get('MT_EAFE', np.nan)
                    init_row['MT_MM'] = acc_row.get('MT_MM', 0)
                    init_row['AJUSTEMENT_MENSUEL_GAR'] = 0
                    init_row['MT_RF'] = acc_row.get('MT_RF', 0)
                    # Product parameters
                    init_row['PC_REVENU_FDS'] = acc_row.get('PC_REVENU_FDS', np.nan)
                    init_row['PC_RFG'] = acc_row.get('PC_RFG', np.nan)
                    init_row['PC_FRAIS_GARANTIE'] = acc_row.get('PC_FRAIS_GARANTIE', 0)
                    init_row['PC_HONORAIRES_GEST'] = acc_row.get('PC_HONORAIRES_GEST', np.nan)
                    init_row['MT_GAR_ECH'] = acc_row.get('MT_GAR_ECH', np.nan)
                    init_row['MT_GAR_DECES'] = acc_row.get('MT_GAR_DECES', np.nan)
                    init_row['MT_BCB'] = acc_row.get('MT_BCB', 0)
                    init_row['MT_SRG'] = acc_row.get('MT_SRG', 0)
                    init_row['MT_MRV_MRG_MRA'] = acc_row.get('MT_MRV_MRG_MRA', 0)
                    init_row['M_MT_MRV_EXCEDENT'] = acc_row.get('M_MT_MRV_EXCEDENT', 0)
                    init_row['TAUX_MRV_MRG_MRA'] = acc_row.get('TAUX_MRV_MRG_MRA', 0)
                    init_row['MT_BONI_DECES'] = acc_row.get('MT_BONI_DECES', 0)
                    # Product info fields
                    init_row['ID_PRODUIT'] = acc_row.get('ID_PRODUIT', np.nan)
                    init_row['I_PRODUIT_REGR'] = acc_row.get('I_PRODUIT_REGR', 0)
                    init_row['I_PRODUIT_HEDGE'] = acc_row.get('I_PRODUIT_HEDGE', 0)
                    init_row['PC_GAR_ECH'] = acc_row.get('PC_GAR_ECH', np.nan)
                    init_row['PC_GAR_ECH_DEP_FUT'] = acc_row.get('PC_GAR_ECH_DEP_FUT', np.nan)
                    init_row['MAX_RESET_FACUL_ECH'] = acc_row.get('MAX_RESET_FACUL_ECH', 0)
                    init_row['RATIO_VM_VG_RESET_ECH'] = acc_row.get('RATIO_VM_VG_RESET_ECH', 0)
                    init_row['AGE_FIN_CONTRAT'] = acc_row.get('AGE_FIN_CONTRAT', np.nan)
                    init_row['PC_RENOUV_ECH'] = acc_row.get('PC_RENOUV_ECH', np.nan)
                    init_row['AGE_MAX_RENOUV_ECH'] = acc_row.get('AGE_MAX_RENOUV_ECH', np.nan)
                    init_row['NB_AN_ECH'] = acc_row.get('NB_AN_ECH', np.nan)
                    init_row['AGE_ECH_MIN'] = acc_row.get('AGE_ECH_MIN', 0)
                    init_row['PC_GAR_DECES_1'] = acc_row.get('PC_GAR_DECES_1', np.nan)
                    init_row['PC_GAR_DECES_2'] = acc_row.get('PC_GAR_DECES_2', 0)
                    init_row['AGE_CHANG_DECES'] = acc_row.get('AGE_CHANG_DECES', np.nan)
                    init_row['FREQ_RESET_DECES'] = acc_row.get('FREQ_RESET_DECES', 0)
                    init_row['MAX_RESET_DECES'] = acc_row.get('MAX_RESET_DECES', 0)
                    init_row['I_RESET_DECES_ECH'] = acc_row.get('I_RESET_DECES_ECH', 0)
                    init_row['PC_BONI_DECES'] = acc_row.get('PC_BONI_DECES', 0)
                    init_row['MAX_BONI_DECES'] = acc_row.get('MAX_BONI_DECES', np.nan)
                    init_row['AGE_MRV_PERMIS'] = acc_row.get('AGE_MRV_PERMIS', 0)
                    init_row['PC_BONI_SRG'] = acc_row.get('PC_BONI_SRG', np.nan)
                    init_row['FREQ_RESET_SRG'] = acc_row.get('FREQ_RESET_SRG', 0)
                    init_row['MAX_RESET_SRG'] = acc_row.get('MAX_RESET_SRG', 0)
                    init_row['I_FRAIS_SUR_SRG'] = acc_row.get('I_FRAIS_SUR_SRG', 0)
                    init_row['TABLE_TAUX_MRV_MRG_MRA'] = acc_row.get('TABLE_TAUX_MRV_MRG_MRA', 0)
                    init_row['MT_TPA_RETRAIT'] = acc_row.get('MT_TPA_RETRAIT', 0)
                    init_row['MT_TPA_DEPOT'] = acc_row.get('MT_TPA_DEPOT', 0)
                    init_row['AJUSTEMENT_COMMISSION'] = acc_row.get('AJUSTEMENT_COMMISSION', 1)
                    init_row['MOIS_EVALUATION_INI'] = acc_row.get('MOIS_EVALUATION_INI', np.nan)
                    init_row['ANNEE_NAIS'] = acc_row.get('ANNEE_NAIS', np.nan)
                    init_row['MOIS_NAIS'] = acc_row.get('MOIS_NAIS', np.nan)
                    init_row['ANNEE_COTIS'] = acc_row.get('ANNEE_COTIS', np.nan)
                    init_row['MOIS_COTIS'] = acc_row.get('MOIS_COTIS', np.nan)
                    init_row['ANNEE_ECH'] = acc_row.get('ANNEE_ECH', np.nan)
                    init_row['MOIS_ECH'] = acc_row.get('MOIS_ECH', np.nan)
                    init_row['I_REGIME'] = acc_row.get('I_REGIME', 0)
                    init_row['I_REGIME_2'] = acc_row.get('I_REGIME_2', 0)
                    init_row['AGE_DECAISSEMENT'] = acc_row.get('AGE_DECAISSEMENT', np.nan)
                    init_row['I_SEXE'] = acc_row.get('I_SEXE', 0)
                    init_row['ID_LAPSE'] = acc_row.get('ID_LAPSE', np.nan)
                    init_row['ID_ACQUI'] = acc_row.get('ID_ACQUI', np.nan)
                    init_row['ID_DEPOT'] = acc_row.get('ID_DEPOT', np.nan)
                    init_row['VAR_RETRAIT_FCT'] = acc_row.get('VAR_RETRAIT_FCT', 1)
                    init_row['PC_RETRAIT_AGE'] = acc_row.get('PC_RETRAIT_AGE', 0)
                    init_row['MT_RETRAIT_MAX'] = acc_row.get('MT_RETRAIT_MAX', 0)
                    init_row['I_RESET_FACUL_ECH'] = acc_row.get('I_RESET_FACUL_ECH', 0)
                    # Key identifiers
                    init_row['ID_COMPTE'] = id_compte
                    init_row['scn_eval'] = int(debug_scenario_idx) + 1 if debug_scenario_idx is not None and debug_scenario_idx >= 0 else np.nan
                    init_row['an_eval'] = 0
                    init_row['mois_eval'] = 12
                    # These should be NaN in init row per ground truth
                    init_row['scn_eval_int'] = np.nan
                    init_row['an_eval_int'] = np.nan
                    init_row['mois_eval_ext'] = np.nan
                    init_row['TX_SURVIE_DEB'] = 1.0
                    init_row['TX_ACTUALISATION_DEB'] = 1.0
                    init_row['VALEUR_MARCHANDE'] = acc_row.get('MT_VM', np.nan)
                    # These calculation fields should be NaN in init row
                    init_row['rc'] = np.nan
                    init_row['AJUST_NOUV_AFFAIRES'] = np.nan
                    init_row['MT_VM_AV_RETRAIT_FRAIS'] = np.nan
                    init_row['duree_max10'] = np.nan
                    init_row['VM_VG_RATIO'] = np.nan
                    init_row['LAPSE_NIV_PART'] = np.nan
                    init_row['LAPSE_NIV_TOT'] = np.nan
                    init_row['LAPSE_TOT'] = np.nan
                    init_row['LAPSE_PART'] = np.nan
                    init_row['LAPSE'] = np.nan
                    annee_eval_ini = acc_row.get('ANNEE_EVALUATION_INI', 2024)
                    annee_nais = acc_row.get('ANNEE_NAIS', 2000)
                    mois_nais = acc_row.get('MOIS_NAIS', 1)
                    # SAS line 903: annee_reelle = ANNEE_EVALUATION_INI + an_eval - 1
                    # For init row: an_eval=0, so annee_reelle = ANNEE_EVALUATION_INI - 1
                    init_row['annee_reelle'] = annee_eval_ini - 1
                    # SAS line 909: age = MAX(INT(YRDIF(MDY(MOIS_NAIS,01,ANNEE_NAIS),MDY(mois_eval,01,annee_reelle),'AGE')),1)
                    # For init row: mois_eval=12, annee_reelle=annee_eval_ini-1
                    # YRDIF calculates years between two dates
                    annee_reelle_init = annee_eval_ini - 1
                    # Age = years between (MOIS_NAIS/1/ANNEE_NAIS) and (12/1/annee_reelle_init)
                    age_years = annee_reelle_init - annee_nais
                    # If birth month (mois_nais) > eval month (12), subtract 1 year
                    if mois_nais > 12:
                        age_years -= 1
                    init_row['AGE'] = max(age_years, 1)
                    # These should be NaN in init row
                    init_row['AGE_RETRAIT'] = np.nan
                    init_row['age_MORTALITE'] = np.nan
                    init_row['RETRAIT'] = np.nan
                    init_row['DEPOT_FUTUR'] = np.nan
                    init_row['MT_VM_AV_RETRAIT'] = np.nan
                    init_row['PRIMES_GARANTIES'] = np.nan
                    init_row['VP_PRIMES_GARANTIES'] = np.nan
                    init_row['MT_SRG_AV_RETRAIT'] = np.nan
                    init_row['PREST_MRV'] = np.nan
                    init_row['VP_PREST_MRV'] = np.nan
                    init_row['MT_VM_AP_RETRAIT'] = np.nan
                    init_row['MT_VM_AP_RETRAIT_DEPOT'] = np.nan
                    init_row['PREST_DECES'] = np.nan
                    init_row['VP_PREST_DECES'] = np.nan
                    init_row['PREST_ECH'] = np.nan
                    init_row['VP_PREST_ECH'] = np.nan
                    # All other calculation/output fields should be NaN
                    wide_rows.append(init_row)
                
                prev_tx_survie = 1.0
                prev_tx_actual = 1.0
                prev_mt_min_ferr_proj = 0.0  # Retained value per SAS line 212
                prev_mt_vm_proj = acc_row.get('MT_VM', 0) if acc_row is not None else 0  # Retained MT_VM_PROJ
                for r in rows:
                    w = {c: np.nan for c in example_header}

                    w['ID_COMPTE'] = r.get('ID_COMPTE', -1)
                    w['scn_eval'] = int(debug_scenario_idx) + 1 if debug_scenario_idx is not None and debug_scenario_idx >= 0 else np.nan
                    w['an_eval'] = r.get('AN_EVAL', np.nan)
                    w['mois_eval'] = r.get('MOIS_EVAL', np.nan)
                    # mois_eval_ext should be NaN per ground truth
                    w['mois_eval_ext'] = np.nan

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
                        
                        # Age-related columns (SAS lines 421-422, 442)
                        # AGE_RETRAIT = AGE + 1 (SAS line 442)
                        w['AGE_RETRAIT'] = age + 1
                        
                        # age_MORTALITE calculation (SAS lines 421-422)
                        # if IFN((mois_nais - mois_eval) <=0,(mois_nais - mois_eval)+12,(mois_nais - mois_eval)) <= 6 then age_MORTALITE = age +1
                        mois_nais = acc_row.get('MOIS_NAIS', 1) if acc_row is not None else 1
                        month_diff = mois_nais - mois_eval
                        if month_diff <= 0:
                            month_diff = month_diff + 12
                        if month_diff <= 6:
                            w['age_MORTALITE'] = age + 1
                        else:
                            w['age_MORTALITE'] = age
                        
                        # Year calculations - SAS line 903: annee_reelle = ANNEE_EVALUATION_INI + an_eval - 1
                        annee_eval_ini = acc_row.get('ANNEE_EVALUATION_INI', 2024)
                        w['annee_reelle'] = annee_eval_ini + an_eval - 1 if pd.notna(an_eval) else np.nan
                        
                        # Duration calculation - SAS line 350: duree_max10=min(10,int((annee_reelle+mois_eval/12)-(ANNEE_COTIS+MOIS_COTIS/12))+1)
                        annee_cotis = acc_row.get('ANNEE_COTIS', 2024)
                        mois_cotis = acc_row.get('MOIS_COTIS', 1)
                        mois_eval_val = r.get('MOIS_EVAL', 12)
                        if pd.notna(an_eval) and pd.notna(annee_eval_ini) and pd.notna(annee_cotis):
                            annee_reelle = annee_eval_ini + an_eval - 1
                            duree = int((annee_reelle + mois_eval_val/12) - (annee_cotis + mois_cotis/12)) + 1
                            w['duree_max10'] = min(duree, 10) if duree >= 0 else 0
                        
                        # VM/VG ratio - SAS line 355:
                        # VM_VG_RATIO=MIN(10,(MT_VM_PROJ+MT_VM_AV_RETRAIT_FRAIS)/2 * MIN(PC_GAR_ECH/MAX(MT_GAR_ECH_PROJ,0.01),PC_GAR_DECES_1/MAX(MT_BONI_DECES_PROJ + MT_GAR_DECES_PROJ,0.01),1/MAX(MT_SRG_PROJ,0.01)))
                        mt_vm_proj = r.get('MT_VM', 0) if pd.notna(r.get('MT_VM')) else 0
                        mt_vm_av_retrait_frais = r.get('MT_VM_AV_RETRAIT', 0) if pd.notna(r.get('MT_VM_AV_RETRAIT')) else 0
                        mt_gar_ech_proj = r.get('MT_GAR_ECH', 0) if pd.notna(r.get('MT_GAR_ECH')) else 0
                        mt_gar_deces_proj = r.get('MT_GAR_DECES', 0) if pd.notna(r.get('MT_GAR_DECES')) else 0
                        mt_boni_deces_proj = acc_row.get('MT_BONI_DECES', 0) if pd.notna(acc_row.get('MT_BONI_DECES')) else 0
                        mt_srg_proj = r.get('MT_SRG', 0) if pd.notna(r.get('MT_SRG')) else 0
                        pc_gar_ech = acc_row.get('PC_GAR_ECH', 1.0) if pd.notna(acc_row.get('PC_GAR_ECH')) else 1.0
                        pc_gar_deces_1 = acc_row.get('PC_GAR_DECES_1', 1.0) if pd.notna(acc_row.get('PC_GAR_DECES_1')) else 1.0
                        
                        vm_avg = (mt_vm_proj + mt_vm_av_retrait_frais) / 2.0
                        ratio1 = pc_gar_ech / max(mt_gar_ech_proj, 0.01)
                        ratio2 = pc_gar_deces_1 / max(mt_boni_deces_proj + mt_gar_deces_proj, 0.01)
                        ratio3 = 1.0 / max(mt_srg_proj, 0.01)
                        vm_vg_ratio = min(10.0, vm_avg * min(ratio1, ratio2, ratio3))
                        w['VM_VG_RATIO'] = vm_vg_ratio
                        
                        # Lapse levels - SAS lines 363-365 and 392-394
                        # if vm_vg_ratio <= 0.5 then LAPSE_NIV_TOT=1; else if vm_vg_ratio <= 0.75 then LAPSE_NIV_TOT=2; else LAPSE_NIV_TOT=3;
                        if vm_vg_ratio <= 0.5:
                            lapse_niv_tot = 1
                            lapse_niv_part = 1
                        elif vm_vg_ratio <= 0.75:
                            lapse_niv_tot = 2
                            lapse_niv_part = 2
                        else:
                            lapse_niv_tot = 3
                            lapse_niv_part = 3
                        w['LAPSE_NIV_TOT'] = lapse_niv_tot
                        w['LAPSE_NIV_PART'] = lapse_niv_part
                        
                        # Calculate LAPSE from LAPSE_TOT and LAPSE_PART using SAS formula (line 412)
                        # LAPSE = (1-(1 - LAPSE_TOT - LAPSE_PART)**(1/FREQ_EVAL * AJUST_NOUV_AFFAIRES))
                        lapse_tot = r.get('LAPSE_TOT', 0) if pd.notna(r.get('LAPSE_TOT')) else 0
                        lapse_part = r.get('LAPSE_PART', 0) if pd.notna(r.get('LAPSE_PART')) else 0
                        freq_eval = 12.0  # Monthly frequency
                        ajust_nouv_affaires = 1.0
                        w['LAPSE'] = 1.0 - math.pow(1.0 - lapse_tot - lapse_part, 1.0 / freq_eval * ajust_nouv_affaires)
                        
                        # Projected guarantee columns
                        w['MT_BONI_DECES_PROJ'] = mt_boni_deces_proj
                        w['MT_BCB_PROJ'] = acc_row.get('MT_BCB', 0)
                        w['MT_MRV_MRG_MRA_PROJ'] = acc_row.get('MT_MRV_MRG_MRA', 0)
                        w['TAUX_MRV_MRG_MRA_PROJ'] = acc_row.get('TAUX_MRV_MRG_MRA', 0)
                        w['MT_SRG_PROJ'] = mt_srg_proj
                        
                        # Echeance projections
                        w['ANNEE_ECH_PROJ'] = acc_row.get('ANNEE_ECH', np.nan)
                        w['MOIS_ECH_PROJ'] = acc_row.get('MOIS_ECH', np.nan)
                        
                        # Age factors - these are looked up from COUSSINS_ESCAP table, not calculated
                        # Will be filled later from lookup_data
                        w['FACTEUR_AGE_80'] = np.nan
                        w['FACTEUR_AGE_90'] = np.nan
                        
                        # MIN_FERR_PROJ - lookup from min_ferr table by age
                        # SAS line 449: if (an_eval = 1 and mois_eval = MOIS_EVALUATION_INI) or mois_eval = 12/&FREQ_EVAL. then MT_MIN_FERR_PROJ = MT_VM_PROJ * MIN_FERR;
                        # With FREQ_EVAL=12 (monthly), 12/FREQ_EVAL = 1, so update at mois_eval=1 (January) or first month of year 1
                        mois_eval_ini = acc_row.get('MOIS_EVALUATION_INI', 1) if acc_row is not None else 1
                        freq_eval = 12  # Monthly evaluation
                        should_update_min_ferr = (an_eval == 1 and mois_eval == mois_eval_ini) or mois_eval == (12 // freq_eval)
                        
                        if should_update_min_ferr:
                            mt_min_ferr = 0.0
                            if lookup_data is not None and 'min_ferr' in lookup_data:
                                min_ferr_df = lookup_data['min_ferr']
                                if 'AGE' in min_ferr_df.columns and 'MIN_FERR' in min_ferr_df.columns:
                                    age_match = min_ferr_df[min_ferr_df['AGE'] == int(age)]
                                    if len(age_match) > 0:
                                        # Use previous period's MT_VM_PROJ (retained value) per SAS logic
                                        mt_min_ferr = float(age_match['MIN_FERR'].iloc[0]) * prev_mt_vm_proj
                            w['MT_MIN_FERR_PROJ'] = mt_min_ferr
                            prev_mt_min_ferr_proj = mt_min_ferr  # Update retained value
                        else:
                            # Retain previous value
                            w['MT_MIN_FERR_PROJ'] = prev_mt_min_ferr_proj
                        
                        # Update retained MT_VM_PROJ for next iteration
                        prev_mt_vm_proj = mt_vm_proj if pd.notna(mt_vm_proj) else prev_mt_vm_proj
                        
                        # Actualization rates - calculate from forward rate
                        # SAS: TX_ACTUALISATION = TX_ACTUALISATION * EXP(-FORWARD_RATE * AJUST_NOUV_AFFAIRES)
                        # TX_ACTUALISATION_DEB is the previous period's discount factor
                        w['TX_ACTUALISATION_DEB'] = prev_tx_actual
                        # TX_ACTUALISATION is current discount factor (compounded)
                        forward_rate = 0.0
                        ajust_nouv_affaires = 1.0  # Default adjustment
                        if lookup_data is not None and 'rendements' in lookup_data:
                            rend_df = lookup_data['rendements']
                            an_col = 'AN_EVAL' if 'AN_EVAL' in rend_df.columns else 'an_eval'
                            mois_col = 'MOIS_EVAL' if 'MOIS_EVAL' in rend_df.columns else 'mois_eval'
                            rend_match = rend_df[(rend_df[an_col] == an_eval) & (rend_df[mois_col] == mois_eval)]
                            if len(rend_match) > 0 and 'FORWARD_RATE' in rend_match.columns:
                                forward_rate = float(rend_match['FORWARD_RATE'].iloc[0])
                        # SAS formula: TX_ACTUALISATION = TX_ACTUALISATION * EXP(-FORWARD_RATE * AJUST_NOUV_AFFAIRES)
                        curr_tx_actual = prev_tx_actual * math.exp(-forward_rate * ajust_nouv_affaires)
                        w['TX_ACTUALISATION'] = curr_tx_actual
                        w['AJUST_NOUV_AFFAIRES'] = ajust_nouv_affaires
                        prev_tx_actual = curr_tx_actual
                        
                        # Internal scenario fields - should be NaN per ground truth
                        w['scn_eval_int'] = np.nan
                        w['an_eval_int'] = np.nan
                        
                        # Adjustment fields
                        w['rc'] = 0.0
                        # AJUST_NOUV_AFFAIRES already set above from discount calculation
                        # MT_VM_AV_RETRAIT_FRAIS = sum of all asset classes (before fees)
                        mt_sp500 = r.get('MT_SP500', 0) if pd.notna(r.get('MT_SP500')) else 0
                        mt_tsx = r.get('MT_TSX', 0) if pd.notna(r.get('MT_TSX')) else 0
                        mt_eafe = r.get('MT_EAFE', 0) if pd.notna(r.get('MT_EAFE')) else 0
                        mt_dex = r.get('MT_DEX', 0) if pd.notna(r.get('MT_DEX')) else 0
                        mt_mm = r.get('MT_MM', 0) if pd.notna(r.get('MT_MM')) else 0
                        w['MT_VM_AV_RETRAIT_FRAIS'] = mt_sp500 + mt_tsx + mt_eafe + mt_dex + mt_mm
                        
                        # Present value columns (VP_*) - calculated as base * TX_ACTUALISATION per SAS lines 736-786
                        tx_actual = w.get('TX_ACTUALISATION', 1.0)
                        if pd.isna(tx_actual):
                            tx_actual = 1.0
                        
                        primes_garanties = r.get('PRIMES_GARANTIES', 0) if pd.notna(r.get('PRIMES_GARANTIES')) else 0
                        w['VP_PRIMES_GARANTIES'] = primes_garanties * tx_actual
                        
                        prest_mrv = r.get('PREST_MRV', 0) if pd.notna(r.get('PREST_MRV')) else 0
                        w['VP_PREST_MRV'] = prest_mrv * tx_actual
                        
                        prest_deces = r.get('PREST_DECES', 0) if pd.notna(r.get('PREST_DECES')) else 0
                        w['VP_PREST_DECES'] = prest_deces * tx_actual
                        
                        prest_ech = r.get('PREST_ECH', 0) if pd.notna(r.get('PREST_ECH')) else 0
                        w['VP_PREST_ECH'] = prest_ech * tx_actual
                        
                        comm_vente = r.get('COMM_VENTE', 0) if pd.notna(r.get('COMM_VENTE')) else 0
                        w['VP_COMM_VENTE'] = comm_vente * tx_actual
                        
                        frais_acquis = r.get('FRAIS_ACQUIS', 0) if pd.notna(r.get('FRAIS_ACQUIS')) else 0
                        w['VP_FRAIS_ACQUIS'] = frais_acquis * tx_actual
                        
                        frais_fixes = r.get('FRAIS_FIXES', 0) if pd.notna(r.get('FRAIS_FIXES')) else 0
                        w['VP_FRAIS_FIXES'] = frais_fixes * tx_actual
                        
                        hon_gest = r.get('HON_GEST', 0) if pd.notna(r.get('HON_GEST')) else 0
                        w['VP_HON_GEST'] = hon_gest * tx_actual
                        
                        comm_maintien = r.get('COMM_MAINTIEN', 0) if pd.notna(r.get('COMM_MAINTIEN')) else 0
                        w['VP_COMM_MAINTIEN'] = comm_maintien * tx_actual
                        
                        primes_variables = r.get('PRIMES_VARIABLES', 0) if pd.notna(r.get('PRIMES_VARIABLES')) else 0
                        w['VP_PRIMES_VARIABLES'] = primes_variables * tx_actual
                        
                        # VP_FLUX_TOT is sum of all VP_* cashflows (SAS line 757)
                        w['VP_FLUX_TOT'] = (w['VP_PRIMES_GARANTIES'] + w['VP_PREST_MRV'] + w['VP_PREST_DECES'] + 
                                           w['VP_PREST_ECH'] + w['VP_COMM_VENTE'] + w['VP_FRAIS_ACQUIS'] +
                                           w['VP_FRAIS_FIXES'] + w['VP_HON_GEST'] + w['VP_COMM_MAINTIEN'] +
                                           w['VP_PRIMES_VARIABLES'])
                        
                        # VP_VALEUR_MARCHANDE = VALEUR_MARCHANDE * TX_ACTUALISATION / FREQ_EVAL (SAS line 785)
                        valeur_marchande = r.get('VALEUR_MARCHANDE', 0) if pd.notna(r.get('VALEUR_MARCHANDE')) else 0
                        w['VP_VALEUR_MARCHANDE'] = valeur_marchande * tx_actual / 12.0  # FREQ_EVAL=12
                        
                        # Cushion VP columns (SAS lines 868-873)
                        coussin_depense = r.get('COUSSIN_DEPENSE', 0) if pd.notna(r.get('COUSSIN_DEPENSE')) else 0
                        w['VP_COUSSIN_DEPENSE'] = coussin_depense * tx_actual / 12.0
                        
                        coussin_decheance = r.get('COUSSIN_DECHEANCE', 0) if pd.notna(r.get('COUSSIN_DECHEANCE')) else 0
                        w['VP_COUSSIN_DECHEANCE'] = coussin_decheance * tx_actual / 12.0
                        
                        coussin_mortalite = r.get('COUSSIN_MORTALITE', 0) if pd.notna(r.get('COUSSIN_MORTALITE')) else 0
                        w['VP_COUSSIN_MORTALITE'] = coussin_mortalite * tx_actual / 12.0
                        
                        coussin_depot = r.get('COUSSIN_DEPOT', 0) if pd.notna(r.get('COUSSIN_DEPOT')) else 0
                        w['VP_COUSSIN_DEPOT'] = coussin_depot * tx_actual / 12.0
                        
                        passif_redresse = r.get('PASSIF_REDRESSE', 0) if pd.notna(r.get('PASSIF_REDRESSE')) else 0
                        w['VP_PASSIF_REDRESSE'] = passif_redresse * tx_actual / 12.0
                        
                        coussin_credit = r.get('COUSSIN_CREDIT', 0) if pd.notna(r.get('COUSSIN_CREDIT')) else 0
                        w['VP_COUSSIN_CREDIT'] = coussin_credit * tx_actual / 12.0
                        
                        coussin_marche = r.get('COUSSIN_MARCHE', 0) if pd.notna(r.get('COUSSIN_MARCHE')) else 0
                        w['VP_COUSSIN_MARCHE'] = coussin_marche * tx_actual / 12.0
                        
                        # Additional computed fields
                        w['MT_SRG_AV_RETRAIT'] = r.get('MT_SRG', np.nan)
                        w['MT_VM_AP_RETRAIT_DEPOT'] = r.get('MT_VM_AP_RETRAIT', np.nan)
                        w['DEPOT_FUTUR_SURVIE'] = r.get('DEPOT_FUTUR', 0) * w.get('TX_SURVIE', 1.0) if pd.notna(r.get('DEPOT_FUTUR')) else 0.0
                        
                        # Commission/fee percentages from acquisition table (will be filled from lookup)
                        w['PC_COMMISSION_MAINTIEN'] = 0.0
                        w['PC_COMMISSION_VENTE'] = 0.0
                        w['PC_FRAIS_AN'] = 0.0
                        
                        # Category and coverage fields - SAS formulas:
                        # VALEUR_GARANTIE = MT_GAR_DECES_PROJ * TX_SURVIE (line 780)
                        # UNITE_COUVERTURE = MAX(MT_VM_PROJ, MT_GAR_DECES_PROJ + MT_BONI_DECES_PROJ, RETRAIT) * TX_SURVIE (line 768)
                        tx_survie = w.get('TX_SURVIE', 1.0)
                        if pd.isna(tx_survie):
                            tx_survie = 1.0
                        mt_boni_deces_local = acc_row.get('MT_BONI_DECES', 0) if acc_row is not None else 0
                        mt_gar_deces_local = r.get('MT_GAR_DECES', 0) if pd.notna(r.get('MT_GAR_DECES')) else 0
                        mt_vm_local = r.get('MT_VM', 0) if pd.notna(r.get('MT_VM')) else 0
                        retrait = r.get('RETRAIT', 0) if pd.notna(r.get('RETRAIT')) else 0
                        # VALEUR_GARANTIE = MT_GAR_DECES_PROJ * TX_SURVIE
                        valeur_garantie = mt_gar_deces_local * tx_survie
                        # UNITE_COUVERTURE = MAX(MT_VM_PROJ, MT_GAR_DECES_PROJ + MT_BONI_DECES_PROJ, RETRAIT) * TX_SURVIE
                        unite_couverture = max(mt_vm_local,
                                              mt_gar_deces_local + mt_boni_deces_local,
                                              retrait) * tx_survie
                        w['UNITE_COUVERTURE'] = unite_couverture
                        w['VALEUR_GARANTIE'] = valeur_garantie
                        
                        # REM_COMP_INV calculation per SAS line 791
                        # REM_COMP_INV = ((RETRAIT - PREST_MRV) + MT_VM_AP_RETRAIT_DEPOT * (1 - TX_SURVIE/TX_SURVIE_DEB)) * TX_SURVIE_DEB
                        retrait_val = r.get('RETRAIT', 0) if pd.notna(r.get('RETRAIT')) else 0
                        prest_mrv_val = r.get('PREST_MRV', 0) if pd.notna(r.get('PREST_MRV')) else 0
                        mt_vm_ap_retrait_depot = r.get('MT_VM_AP_RETRAIT', 0) if pd.notna(r.get('MT_VM_AP_RETRAIT')) else 0
                        tx_survie_deb = w.get('TX_SURVIE_DEB', 1.0) if pd.notna(w.get('TX_SURVIE_DEB')) else 1.0
                        tx_survie_val = w.get('TX_SURVIE', 1.0) if pd.notna(w.get('TX_SURVIE')) else 1.0
                        if tx_survie_deb > 0:
                            rem_comp_inv = ((retrait_val - prest_mrv_val) + mt_vm_ap_retrait_depot * (1 - tx_survie_val / tx_survie_deb)) * tx_survie_deb
                        else:
                            rem_comp_inv = 0.0
                        w['REM_COMP_INV'] = rem_comp_inv
                        
                        # CODE_CAT_PRODUIT should be 1 (not ID_PRODUIT)
                        w['CODE_CAT_PRODUIT'] = 1
                        # Extract CAT_COUSSIN values from flux data
                        w['CAT_COUSSIN_1'] = int(r.get('CAT_COUSSIN_1', 0)) if pd.notna(r.get('CAT_COUSSIN_1')) else 0
                        w['CAT_COUSSIN_2'] = int(r.get('CAT_COUSSIN_2', 0)) if pd.notna(r.get('CAT_COUSSIN_2')) else 0

                    if acc_row is not None:
                        for c in example_header:
                            if pd.isna(w.get(c, np.nan)) and c in acc_row.index:
                                w[c] = acc_row[c]

                    wide_rows.append(w)

                # Convert to DataFrame and fill NA columns from lookup tables
                output_df = pd.DataFrame(wide_rows, columns=example_header)
                
                # Note: Row 0 is the init row (an_eval=0, mois_eval=12) which should NOT have lookup values
                # Only fill lookup values for data rows (row 1 onwards)
                if lookup_data is not None and acc_row is not None:
                    # Get account keys for lookups
                    id_lapse = acc_row.get('ID_LAPSE', 0)
                    id_acqui = acc_row.get('ID_ACQUI', 0)
                    id_depot = acc_row.get('ID_DEPOT', 0)
                    i_regime_2 = acc_row.get('I_REGIME_2', 0)
                    
                    # Fill MIN_FERR from min_ferr table (keyed by AGE) - skip init row
                    if 'min_ferr' in lookup_data and 'MIN_FERR' in output_df.columns:
                        min_ferr_df = lookup_data['min_ferr']
                        if 'AGE' in min_ferr_df.columns and 'MIN_FERR' in min_ferr_df.columns:
                            age_to_minferr = dict(zip(min_ferr_df['AGE'], min_ferr_df['MIN_FERR']))
                            # Only fill for data rows (skip row 0)
                            output_df.loc[1:, 'MIN_FERR'] = output_df.loc[1:, 'AGE'].map(age_to_minferr)
                    
                    # Fill TX_LAPSE_TOT columns from tx_lapse_tot table - keyed by ID_LAPSE, DUREE_MAX10, LAPSE_NIV_TOT
                    if 'tx_lapse_tot' in lookup_data:
                        lapse_tot_df = lookup_data['tx_lapse_tot']
                        if all(c in lapse_tot_df.columns for c in ['ID_LAPSE', 'DUREE_MAX10', 'LAPSE_NIV_TOT']):
                            # For each data row, look up using composite key
                            for idx in range(1, len(output_df)):
                                duree = output_df.loc[idx, 'duree_max10']
                                lapse_niv = output_df.loc[idx, 'LAPSE_NIV_TOT']
                                if pd.notna(duree) and pd.notna(lapse_niv):
                                    match = lapse_tot_df[
                                        (lapse_tot_df['ID_LAPSE'] == id_lapse) &
                                        (lapse_tot_df['DUREE_MAX10'] == int(duree)) &
                                        (lapse_tot_df['LAPSE_NIV_TOT'] == int(lapse_niv))
                                    ]
                                    if len(match) > 0:
                                        for col in ['TX_LAPSE_TOT_MIN', 'TX_LAPSE_TOT_MAX', 'FACT_DIM']:
                                            if col in match.columns and col in output_df.columns:
                                                output_df.loc[idx, col] = match[col].iloc[0]
                    
                    # Fill TX_LAPSE_PART columns from tx_lapse_part table - keyed by ID_LAPSE, AGE, LAPSE_NIV_PART, I_REGIME_2
                    if 'tx_lapse_part' in lookup_data:
                        lapse_part_df = lookup_data['tx_lapse_part']
                        if all(c in lapse_part_df.columns for c in ['ID_LAPSE', 'AGE', 'LAPSE_NIV_PART', 'I_REGIME_2']):
                            for idx in range(1, len(output_df)):
                                age_val = output_df.loc[idx, 'AGE']
                                lapse_niv = output_df.loc[idx, 'LAPSE_NIV_PART']
                                if pd.notna(age_val) and pd.notna(lapse_niv):
                                    match = lapse_part_df[
                                        (lapse_part_df['ID_LAPSE'] == id_lapse) &
                                        (lapse_part_df['AGE'] == int(age_val)) &
                                        (lapse_part_df['LAPSE_NIV_PART'] == int(lapse_niv)) &
                                        (lapse_part_df['I_REGIME_2'] == i_regime_2)
                                    ]
                                    if len(match) > 0:
                                        for col in ['TX_LAPSE_PART_MIN', 'TX_LAPSE_PART_MAX']:
                                            if col in match.columns and col in output_df.columns:
                                                output_df.loc[idx, col] = match[col].iloc[0]
                    
                    # Fill ACQUISITION columns (PC_COMMISSION_*, PC_FRAIS_AN_*) - keyed by DUREE_MAX10, ID_ACQUI
                    if 'acquisition' in lookup_data:
                        acq_df = lookup_data['acquisition']
                        if all(c in acq_df.columns for c in ['DUREE_MAX10', 'ID_ACQUI']):
                            for idx in range(1, len(output_df)):
                                duree = output_df.loc[idx, 'duree_max10']
                                if pd.notna(duree):
                                    match = acq_df[
                                        (acq_df['DUREE_MAX10'] == int(duree)) &
                                        (acq_df['ID_ACQUI'] == id_acqui)
                                    ]
                                    if len(match) > 0:
                                        for col in ['PC_COMMISSION_VENTE_RF', 'PC_COMMISSION_VENTE_AC', 
                                                   'PC_COMMISSION_MAINTIEN_RF', 'PC_COMMISSION_MAINTIEN_AC',
                                                   'PC_FRAIS_AN_AC', 'PC_FRAIS_AN_RF']:
                                            if col in match.columns and col in output_df.columns:
                                                val = match[col].iloc[0]
                                                # Handle percentage strings
                                                if isinstance(val, str) and '%' in val:
                                                    try:
                                                        val = float(val.replace('%', '').strip()) / 100.0
                                                    except:
                                                        pass
                                                output_df.loc[idx, col] = val
                    
                    # Fill DEPOTS_FUTURS columns - keyed by ID_DEPOT, DUREE_MAX10
                    if 'depots_futurs' in lookup_data:
                        depot_df = lookup_data['depots_futurs']
                        if all(c in depot_df.columns for c in ['ID_DEPOT', 'DUREE_MAX10']):
                            for idx in range(1, len(output_df)):
                                duree = output_df.loc[idx, 'duree_max10']
                                if pd.notna(duree):
                                    match = depot_df[
                                        (depot_df['ID_DEPOT'] == id_depot) &
                                        (depot_df['DUREE_MAX10'] == int(duree))
                                    ]
                                    if len(match) > 0:
                                        for col in ['PC_DEPOT_ANNUEL', 'VAR_DEPOT_FCT', 'AGE_MAX_DEPOT', 'I_EVEN_CESSE_DEPOT']:
                                            if col in match.columns and col in output_df.columns:
                                                val = match[col].iloc[0]
                                                # Handle percentage strings like "8.7%"
                                                if isinstance(val, str) and '%' in val:
                                                    try:
                                                        val = float(val.replace('%', '').strip()) / 100.0
                                                    except:
                                                        pass
                                                output_df.loc[idx, col] = val
                    
                    # Fill FORWARD_RATE and AJUST_FORWARD_RATE_VM_0 from rendements - skip init row
                    if 'rendements' in lookup_data:
                        rend_df = lookup_data['rendements']
                        if 'AN_EVAL' in rend_df.columns or 'an_eval' in rend_df.columns:
                            an_col = 'AN_EVAL' if 'AN_EVAL' in rend_df.columns else 'an_eval'
                            mois_col = 'MOIS_EVAL' if 'MOIS_EVAL' in rend_df.columns else 'mois_eval'
                            # Skip row 0 (init row)
                            for idx, row in output_df.iloc[1:].iterrows():
                                an_val = row.get('an_eval', np.nan)
                                mois_val = row.get('mois_eval', np.nan)
                                if pd.notna(an_val) and pd.notna(mois_val):
                                    rend_match = rend_df[(rend_df[an_col] == an_val) & (rend_df[mois_col] == mois_val)]
                                    if len(rend_match) > 0:
                                        if 'FORWARD_RATE' in rend_match.columns and pd.isna(output_df.loc[idx, 'FORWARD_RATE']):
                                            output_df.loc[idx, 'FORWARD_RATE'] = rend_match['FORWARD_RATE'].iloc[0]
                                        if 'AJUST_FORWARD_RATE_VM_0' in rend_match.columns and pd.isna(output_df.loc[idx, 'AJUST_FORWARD_RATE_VM_0']):
                                            output_df.loc[idx, 'AJUST_FORWARD_RATE_VM_0'] = rend_match['AJUST_FORWARD_RATE_VM_0'].iloc[0]
                    
                    # Fill FRAIS from frais_admin - keyed by ANNEE_REELLE, ID_PRODUIT
                    if 'frais_admin' in lookup_data:
                        frais_df = lookup_data['frais_admin']
                        id_produit = acc_row.get('ID_PRODUIT', 0)
                        if all(c in frais_df.columns for c in ['ANNEE_REELLE', 'ID_PRODUIT', 'FRAIS']) and 'FRAIS' in output_df.columns:
                            for idx in range(1, len(output_df)):
                                annee_reelle = output_df.loc[idx, 'annee_reelle']
                                if pd.notna(annee_reelle):
                                    match = frais_df[
                                        (frais_df['ANNEE_REELLE'] == int(annee_reelle)) &
                                        (frais_df['ID_PRODUIT'] == id_produit)
                                    ]
                                    if len(match) > 0:
                                        output_df.loc[idx, 'FRAIS'] = match['FRAIS'].iloc[0]
                    
                    # Fill COUSSINS_ESCAP columns - keyed by CODE_CAT_PRODUIT, CAT_COUSSIN_1, CAT_COUSSIN_2
                    # SAS lines 801-824 define these categories
                    if 'coussins_escap' in lookup_data:
                        coussin_df = lookup_data['coussins_escap']
                        id_produit = acc_row.get('ID_PRODUIT', 0)
                        
                        # CODE_CAT_PRODUIT from ID_PRODUIT (SAS lines 801-808)
                        if id_produit == 22:
                            code_cat_produit = 0  # CPG IA
                        elif id_produit in [12, 13, 14, 15, 16]:
                            code_cat_produit = 1  # CFB
                        elif id_produit in [17, 18, 19, 20, 21]:
                            code_cat_produit = 2  # Courtage
                        elif id_produit == 6:
                            code_cat_produit = 3  # R12
                        elif id_produit in [4, 7]:
                            code_cat_produit = 4  # E12 et N75
                        elif id_produit in [5, 8]:
                            code_cat_produit = 5  # O12 et N10
                        elif id_produit in [2, 3]:
                            code_cat_produit = 6  # CIG Boursier A et B
                        else:
                            code_cat_produit = 7  # RGS
                        
                        coussin_cols = ['BASE_PASSIF_REDRESSE', 'TX_PASSIF_REDRESSE', 'BASE_COUSSIN_CREDIT', 
                                       'TX_COUSSIN_CREDIT', 'BASE_COUSSIN_MARCHE', 'TX_COUSSIN_MARCHE',
                                       'BASE_COUSSIN_DEPENSE', 'TX_COUSSIN_DEPENSE', 'BASE_COUSSIN_DECHEANCE',
                                       'TX_COUSSIN_DECHEANCE', 'BASE_COUSSIN_MORTALITE', 'TX_COUSSIN_MORTALITE',
                                       'BASE_COUSSIN_DEPOT', 'TX_COUSSIN_DEPOT', 'FACTEUR_AGE_80', 'FACTEUR_AGE_90']
                        
                        if all(c in coussin_df.columns for c in ['CODE_CAT_PRODUIT', 'CAT_COUSSIN_1', 'CAT_COUSSIN_2']):
                            for idx in range(1, len(output_df)):
                                duree = output_df.loc[idx, 'duree_max10']
                                vm_vg = output_df.loc[idx, 'VM_VG_RATIO']
                                mt_dex_proj = output_df.loc[idx, 'MT_DEX_PROJ'] if pd.notna(output_df.loc[idx, 'MT_DEX_PROJ']) else 0
                                mt_mm_proj = output_df.loc[idx, 'MT_MM_PROJ'] if pd.notna(output_df.loc[idx, 'MT_MM_PROJ']) else 0
                                mt_vm_proj = output_df.loc[idx, 'MT_VM_PROJ'] if pd.notna(output_df.loc[idx, 'MT_VM_PROJ']) else 0.01
                                rf_ratio = (mt_dex_proj + mt_mm_proj) / max(mt_vm_proj, 0.01)
                                
                                # CAT_COUSSIN_1 (SAS lines 811-816)
                                if code_cat_produit in [0, 6]:
                                    cat_coussin_1 = 0  # CPG IA and CIG Boursier
                                elif code_cat_produit == 7 and rf_ratio < 0.5:
                                    cat_coussin_1 = 4  # RGS < 50%
                                elif code_cat_produit == 7:
                                    cat_coussin_1 = 5  # RGS >= 50%
                                elif rf_ratio < 1/3:
                                    cat_coussin_1 = 1  # < 1/3
                                elif rf_ratio < 2/3:
                                    cat_coussin_1 = 2  # < 2/3
                                else:
                                    cat_coussin_1 = 3  # >= 2/3
                                
                                # CAT_COUSSIN_2 (SAS lines 819-824)
                                if code_cat_produit == 7 and pd.notna(vm_vg) and vm_vg < 0.7:
                                    cat_coussin_2 = 4  # RGS < 70%
                                elif code_cat_produit == 7 and pd.notna(vm_vg) and vm_vg < 0.9:
                                    cat_coussin_2 = 5  # RGS < 90%
                                elif code_cat_produit == 7:
                                    cat_coussin_2 = 6  # RGS >= 90%
                                elif pd.notna(duree) and duree <= 3:
                                    cat_coussin_2 = 1  # année police 0-3
                                elif pd.notna(duree) and duree <= 6:
                                    cat_coussin_2 = 2  # année police 4-6
                                else:
                                    cat_coussin_2 = 3  # année police 7+
                                
                                match = coussin_df[
                                    (coussin_df['CODE_CAT_PRODUIT'] == code_cat_produit) &
                                    (coussin_df['CAT_COUSSIN_1'] == cat_coussin_1) &
                                    (coussin_df['CAT_COUSSIN_2'] == cat_coussin_2)
                                ]
                                if len(match) > 0:
                                    for col in coussin_cols:
                                        if col in match.columns and col in output_df.columns:
                                            val = match[col].iloc[0]
                                            # Handle percentage strings like "87.15%"
                                            if isinstance(val, str) and '%' in val:
                                                try:
                                                    val = float(val.replace('%', '').replace('(', '').replace(')', '').strip()) / 100.0
                                                except:
                                                    pass
                                            output_df.loc[idx, col] = val
                    
                    # Calculate PC_COMMISSION_MAINTIEN, PC_COMMISSION_VENTE, PC_FRAIS_AN per SAS lines 718-720
                    # PC_COMMISSION_MAINTIEN = (PC_COMMISSION_MAINTIEN_AC * (MT_VM-MT_RF)/MT_VM + PC_COMMISSION_MAINTIEN_RF * (MT_RF)/MT_VM) * Ajustement_commission
                    ajustement_commission = acc_row.get('AJUSTEMENT_COMMISSION', 1.0) if pd.notna(acc_row.get('AJUSTEMENT_COMMISSION')) else 1.0
                    mt_rf = acc_row.get('MT_RF', 0) if pd.notna(acc_row.get('MT_RF')) else 0
                    mt_vm_init = acc_row.get('MT_VM', 0.01) if pd.notna(acc_row.get('MT_VM')) else 0.01
                    
                    for idx in range(1, len(output_df)):
                        pc_maintien_rf = output_df.loc[idx, 'PC_COMMISSION_MAINTIEN_RF'] if pd.notna(output_df.loc[idx, 'PC_COMMISSION_MAINTIEN_RF']) else 0
                        pc_maintien_ac = output_df.loc[idx, 'PC_COMMISSION_MAINTIEN_AC'] if pd.notna(output_df.loc[idx, 'PC_COMMISSION_MAINTIEN_AC']) else 0
                        pc_vente_rf = output_df.loc[idx, 'PC_COMMISSION_VENTE_RF'] if pd.notna(output_df.loc[idx, 'PC_COMMISSION_VENTE_RF']) else 0
                        pc_vente_ac = output_df.loc[idx, 'PC_COMMISSION_VENTE_AC'] if pd.notna(output_df.loc[idx, 'PC_COMMISSION_VENTE_AC']) else 0
                        pc_frais_rf = output_df.loc[idx, 'PC_FRAIS_AN_RF'] if pd.notna(output_df.loc[idx, 'PC_FRAIS_AN_RF']) else 0
                        pc_frais_ac = output_df.loc[idx, 'PC_FRAIS_AN_AC'] if pd.notna(output_df.loc[idx, 'PC_FRAIS_AN_AC']) else 0
                        
                        if mt_vm_init > 0:
                            rf_ratio = mt_rf / mt_vm_init
                            ac_ratio = (mt_vm_init - mt_rf) / mt_vm_init
                            output_df.loc[idx, 'PC_COMMISSION_MAINTIEN'] = (pc_maintien_ac * ac_ratio + pc_maintien_rf * rf_ratio) * ajustement_commission
                            output_df.loc[idx, 'PC_COMMISSION_VENTE'] = (pc_vente_ac * ac_ratio + pc_vente_rf * rf_ratio) * ajustement_commission
                            output_df.loc[idx, 'PC_FRAIS_AN'] = pc_frais_ac * ac_ratio + pc_frais_rf * rf_ratio

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
    
    # Note: EXT_DEBUG_GPU.csv removed - redundant with FLUX_PROJETES_GPU_DEBUG.csv
    
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

    # Note: INT_DEBUG_TS_GPU.csv removed - rarely needed, use INT_DEBUG_GPU.csv instead
    
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
        'ext_debug_df': None,  # Removed - redundant with FLUX_PROJETES_GPU_DEBUG.csv
        'int_debug_df': int_debug_df if int_debug is not None else None,
        'int_debug_ts_df': None,  # Removed - rarely needed
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
        debug_only: bool = False,
        run_nested_valuation: bool = True):
    """
    Run GPU-accelerated nested stochastic projection using Two-Pass architecture.
    
    Architecture:
    - Kernel A (Generator): Runs external scenarios, outputs state tensors to VRAM
    - Kernel B (Valuator): Reads states, runs internal scenarios with 5 chocs, outputs reserves & capital
    
    Args:
        debug_only: If True and debug_account >= 0, only process the single account 
                   specified by debug_account (filters population to that account only).
        run_nested_valuation: If True (default), run both Kernel A and Kernel B (full nested valuation).
                             If False, run only Kernel A (outer loop only - no nested valuation).
    """
    start_time = datetime.now()
    print(f"Starting {'NESTED STOCHASTIC' if run_nested_valuation else 'OUTER LOOP ONLY'} GPU projection at {start_time}")
    print("=" * 80)
    if run_nested_valuation:
        print(f"Architecture: Two-Pass (Generator → Valuator with 5 Chocs)")
    else:
        print(f"Architecture: Single-Pass (Generator Only - Outer Loop)")
    print(f"External scenarios: {nb_ext_scenarios}")
    if run_nested_valuation:
        print(f"Internal scenarios per node: {nb_int_scenarios}")
    sys.stdout.flush()
    if run_nested_valuation:
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
    if run_nested_valuation:
        print("RUNNING TWO-PASS NESTED STOCHASTIC PROJECTION")
    else:
        print("RUNNING SINGLE-PASS OUTER LOOP PROJECTION (KERNEL A ONLY)")
    print("=" * 80)
    
    all_reserves = []
    all_capital = []
    all_reserves_5chocs = []
    all_capital_5chocs = []
    ext_debug_result = None
    int_debug_result = None
    int_debug_ts_result = None
    debug_flux_result = None
    total_cashflow_rows = 0  # Track rows written to FLUX_PROJETE_GPU.csv
    total_vp_flux_compte_rows = 0  # Track rows written to VP_FLUX_COMPTE_GPU.csv
    accumulated_flux_projete = None  # Accumulate FLUX_PROJETE across batches
    
    # Extract population IDs early for incremental cashflow writing
    population_ids = data['population']['ID_COMPTE'].values
    
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
            run_nested_valuation=run_nested_valuation,
        )
        
        # Accumulate results
        all_reserves.extend(batch_result['batch_reserves'])
        all_capital.extend(batch_result['batch_capital'])
        all_reserves_5chocs.extend(batch_result['batch_reserves_5chocs'])
        all_capital_5chocs.extend(batch_result['batch_capital_5chocs'])
        
        # Write cashflows incrementally to avoid memory issues
        if batch_result.get('batch_cashflows') is not None:
            rows_written = write_cashflows_batch(
                output_path=output_path,
                batch_cashflows=batch_result['batch_cashflows'],
                population_ids=population_ids,
                start_idx=start_idx,
                is_first_batch=(i == 0),
            )
            total_cashflow_rows += rows_written
            # Free memory immediately
            del batch_result['batch_cashflows']
        
        # Write VP_FLUX_COMPTE incrementally (GPU-aggregated VP by account)
        if batch_result.get('batch_vp_flux_compte') is not None:
            vp_rows_written = write_vp_flux_compte_batch(
                output_path=output_path,
                batch_vp_flux_compte=batch_result['batch_vp_flux_compte'],
                population_ids=population_ids,
                start_idx=start_idx,
                is_first_batch=(i == 0),
            )
            total_vp_flux_compte_rows += vp_rows_written
            del batch_result['batch_vp_flux_compte']
        
        # Accumulate FLUX_PROJETE across batches (GPU-aggregated flux by year)
        if batch_result.get('batch_flux_projete') is not None:
            accumulated_flux_projete = accumulate_flux_projete(
                accumulated_flux_projete,
                batch_result['batch_flux_projete']
            )
            del batch_result['batch_flux_projete']
        
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
    
    # Log cashflow file status
    if total_cashflow_rows > 0:
        logger.info(f"  Written {total_cashflow_rows} rows to FLUX_PROJETE_GPU.csv")
    
    # Log VP_FLUX_COMPTE file status
    if total_vp_flux_compte_rows > 0:
        logger.info(f"  Written {total_vp_flux_compte_rows} rows to VP_FLUX_COMPTE_GPU.csv")
    
    # Write final FLUX_PROJETE (accumulated across all batches)
    total_flux_projete_rows = 0
    if accumulated_flux_projete is not None:
        total_flux_projete_rows = write_flux_projete(
            output_path=output_path,
            flux_projete=accumulated_flux_projete,
            nb_an_projection=nb_an_projection,
        )
        logger.info(f"  Written {total_flux_projete_rows} rows to FLUX_PROJETES_GPU.csv")
    
    # Create results DataFrames
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
    if run_nested_valuation:
        print("NESTED STOCHASTIC PROJECTION COMPLETE")
    else:
        print("OUTER LOOP PROJECTION COMPLETE (NO NESTED VALUATION)")
    print("=" * 80)
    print(f"Total time: {total_duration:.2f}s ({total_duration/60:.2f} minutes)")
    print(f"Accounts processed: {n_accounts}")
    print(f"External scenarios: {nb_ext_scenarios}")
    if run_nested_valuation:
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
    else:
        print(f"Total external simulations: {n_accounts * nb_ext_scenarios * nb_an_projection:,}")
        print(f"\n📊 Reserves Summary (Simple PV from External Scenarios):")
        print(f"  Total Reserve Estimate: ${results_df['RESERVE_BE'].sum():,.2f}")
        print(f"  Average per account:    ${results_df['RESERVE_BE'].mean():,.2f}")
        print(f"\n⚠️  Note: Capital is zero (requires nested valuation)")
        print(f"⚠️  Reserves are approximate (real-world PV, not risk-neutral)")
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
        total_cashflow_rows=total_cashflow_rows,
        total_vp_flux_compte_rows=total_vp_flux_compte_rows,
        total_flux_projete_rows=total_flux_projete_rows,
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

    parser.add_argument('--max-accounts', type=int, default=200,
                        help='Maximum number of accounts to process (for testing)')
    parser.add_argument('--years', type=int, default=100,
                        help='Number of years to project (default: 100)')

    # Nested mode parameters
    parser.add_argument('--ext-scenarios', type=int, default=100,
                        help='Number of external (real-world) scenarios for nested mode (default: 100)')
    parser.add_argument('--int-scenarios', type=int, default=100,
                        help='Number of internal (risk-neutral) scenarios per node for nested mode (default: 100)')
    parser.add_argument('--shock', type=float, default=0.35,
                        help='Capital shock percentage for nested mode (default: 0.35 = 35%%)')
    parser.add_argument('--outer-loop-only', action='store_true',
                        help='Run only outer loop (Kernel A) without nested valuation (Kernel B). Faster but no reserves/capital.')
    
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
            run_nested_valuation=not args.outer_loop_only,
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
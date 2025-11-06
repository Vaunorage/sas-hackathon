import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Any
from datetime import datetime
import math
from numba import cuda

from algo2.gpu2 import prepare_account_data
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
# UTILITY FUNCTIONS (CPU)
# =============================================================================

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
    data['tx_lapse_tot'] = pd.read_csv(data_path.joinpath("TX_LAPSE_TOT.csv"), sep=';', encoding='utf-8')
    data['acquisition'] = pd.read_csv(data_path.joinpath("ACQUISITION.csv"), sep=';', encoding='utf-8')
    data['coussins_escap'] = pd.read_csv(data_path.joinpath("COUSSINS_ESCAP.csv"), sep=';', encoding='utf-8')

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
    account_ids,
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
    # Output arrays - COMPACT FORMAT
    output_results,  # Shape: (max_total_rows, n_output_fields)
    output_counter   # Shape: (1,) - atomic counter for next available row
):
    """
    Main CUDA kernel - processes one account-scenario combination per thread.
    Each thread loops through all timesteps sequentially (state dependency).
    Uses atomic operations to write only valid rows to a compact output buffer.
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

            # ============= STEP 15: STORE RESULTS ATOMICALLY =============
            # Get next available row index atomically
            row_idx = cuda.atomic.add(output_counter, 0, 1)
            
            # Write to compact output buffer (no sparse array needed)
            if row_idx < output_results.shape[0]:
                output_results[row_idx, 0] = ID_COMPTE
                output_results[row_idx, 1] = scn_eval
                output_results[row_idx, 2] = an_eval
                output_results[row_idx, 3] = mois_eval
                output_results[row_idx, 4] = primes_garanties
                output_results[row_idx, 5] = prest_deces
                output_results[row_idx, 6] = prest_ech
                output_results[row_idx, 7] = prest_mrv
                output_results[row_idx, 8] = frais_acquis
                output_results[row_idx, 9] = comm_vente
                output_results[row_idx, 10] = primes_variables
                output_results[row_idx, 11] = frais_fixes
                output_results[row_idx, 12] = hon_gest
                output_results[row_idx, 13] = comm_maintien
                output_results[row_idx, 14] = valeur_marchande
                # Cushions
                output_results[row_idx, 15] = passif_redresse
                output_results[row_idx, 16] = coussin_credit
                output_results[row_idx, 17] = coussin_marche
                output_results[row_idx, 18] = coussin_depense
                output_results[row_idx, 19] = coussin_decheance
                output_results[row_idx, 20] = coussin_mortalite
                output_results[row_idx, 21] = coussin_depot
                # VP values
                output_results[row_idx, 22] = vp_frais_acquis
                output_results[row_idx, 23] = vp_comm_vente
                output_results[row_idx, 24] = vp_primes_garanties
                output_results[row_idx, 25] = vp_primes_variables
                output_results[row_idx, 26] = vp_frais_fixes
                output_results[row_idx, 27] = vp_hon_gest
                output_results[row_idx, 28] = vp_comm_maintien
                output_results[row_idx, 29] = vp_prest_ech
                output_results[row_idx, 30] = vp_prest_mrv
                output_results[row_idx, 31] = vp_prest_deces
                output_results[row_idx, 32] = vp_valeur_marchande
                # VP Cushions
                output_results[row_idx, 33] = vp_passif_redresse
                output_results[row_idx, 34] = vp_coussin_credit
                output_results[row_idx, 35] = vp_coussin_marche
                output_results[row_idx, 36] = vp_coussin_depense
                output_results[row_idx, 37] = vp_coussin_decheance
                output_results[row_idx, 38] = vp_coussin_mortalite
                output_results[row_idx, 39] = vp_coussin_depot

            output_idx += 1

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

    # Calculate total PV of cash flows (EXCLUDE VP_VALEUR_MARCHANDE)
    # VP_VALEUR_MARCHANDE is a stock variable, not a cash flow
    cash_flow_cols = [col for col in vp_cols if col != 'VP_VALEUR_MARCHANDE']
    result['VP_FLUX_TOT'] = result[cash_flow_cols].sum(axis=1)

    # Reorder to match requested column order
    final_columns = ['ID_COMPTE'] + [col for col in requested_columns[1:] if col in result.columns] + ['VP_FLUX_TOT']
    result = result[final_columns]

    return result


def aggregate_vp_flux_total(df: pd.DataFrame) -> pd.DataFrame:
    """Create VP_FLUX_TOTAL: total present value across all accounts."""

    # Cash flow columns (exclude VP_VALEUR_MARCHANDE as it's a stock variable)
    vp_cols = [
        'VP_FRAIS_ACQUIS', 'VP_COMM_VENTE', 'VP_PRIMES_GARANTIES',
        'VP_PRIMES_VARIABLES', 'VP_FRAIS_FIXES', 'VP_HON_GEST', 'VP_COMM_MAINTIEN',
        'VP_PREST_ECH', 'VP_PREST_MRV', 'VP_PREST_DECES',
        'VP_PASSIF_REDRESSE', 'VP_COUSSIN_CREDIT', 'VP_COUSSIN_MARCHE',
        'VP_COUSSIN_DEPENSE', 'VP_COUSSIN_DECHEANCE', 'VP_COUSSIN_MORTALITE',
        'VP_COUSSIN_DEPOT'
        # NOTE: VP_VALEUR_MARCHANDE is NOT included - it's a balance, not a flow
    ]

    # If VP_FLUX_TOT already exists and is correctly calculated, use it
    if 'VP_FLUX_TOT' in df.columns:
        total_vp = df['VP_FLUX_TOT'].sum()
    else:
        # Otherwise, sum the cash flow columns
        existing_vp_cols = [col for col in vp_cols if col in df.columns]
        total_vp = df[existing_vp_cols].sum().sum()

    result = pd.DataFrame({
        'CATEGORIE': ['TOTAL'],
        'VP_FLUX_TOT': [total_vp]
    })

    return result


def get_gpu_memory_info():
    """Get available GPU memory in bytes."""
    try:
        cuda.select_device(0)
        gpu = cuda.get_current_device()

        # Get total memory
        total_memory = cuda.current_context().get_memory_info()[1]
        free_memory = cuda.current_context().get_memory_info()[0]

        return {
            'total_gb': total_memory / 1024**3,
            'free_gb': free_memory / 1024**3,
            'used_gb': (total_memory - free_memory) / 1024**3
        }
    except Exception as e:
        print(f"Warning: Could not get GPU memory info: {e}")
        return {'total_gb': 16, 'free_gb': 12, 'used_gb': 4}  # Conservative defaults


def calculate_optimal_batch_size(n_accounts, n_scenarios, n_years, freq_eval,
                                 n_output_fields, memory_margin=0.8):
    """
    Calculate optimal batch size based on available GPU memory.

    Args:
        n_accounts: Total number of accounts
        n_scenarios: Number of scenarios
        n_years: Number of years to project
        freq_eval: Evaluation frequency
        n_output_fields: Number of output fields per timestep
        memory_margin: Use only this fraction of available memory (default 0.8)

    Returns:
        int: Optimal batch size (number of accounts per batch)
    """
    mem_info = get_gpu_memory_info()
    available_gb = mem_info['free_gb'] * memory_margin

    print(f"\nGPU Memory:")
    print(f"  Total: {mem_info['total_gb']:.2f} GB")
    print(f"  Free: {mem_info['free_gb']:.2f} GB")
    print(f"  Using: {available_gb:.2f} GB (with {memory_margin:.0%} margin)")

    # Calculate memory per account (in GB)
    max_timesteps = (n_years + 1) * freq_eval
    bytes_per_account = n_scenarios * max_timesteps * n_output_fields * 4  # float32
    gb_per_account = bytes_per_account / 1024**3

    # Also account for lookup tables (approximately)
    lookup_table_gb = 0.5  # Conservative estimate

    # Calculate batch size
    batch_size = int((available_gb - lookup_table_gb) / gb_per_account)

    # Ensure batch size is reasonable
    batch_size = max(1, min(batch_size, n_accounts))

    # Round down to a nice number
    if batch_size > 100:
        batch_size = (batch_size // 100) * 100
    elif batch_size > 10:
        batch_size = (batch_size // 10) * 10

    print(f"\nBatch Configuration:")
    print(f"  Memory per account: {gb_per_account*1024:.1f} MB")
    print(f"  Optimal batch size: {batch_size} accounts")
    print(f"  Number of batches: {(n_accounts + batch_size - 1) // batch_size}")
    print(f"  Memory per batch: {batch_size * gb_per_account:.2f} GB")

    return batch_size


def run_projection_gpu_batched(data_path, output_path, nb_an_projection, nb_scenarios,
                               max_accounts=None, threads_per_block=(16, 8),
                               batch_size=None, memory_margin=0.8):
    """
    Run GPU projection with automatic batching.

    Args:
        data_path: Path to input CSV files
        output_path: Path for output files
        nb_an_projection: Number of years to project
        nb_scenarios: Number of economic scenarios
        max_accounts: Maximum number of accounts (None = all)
        threads_per_block: CUDA block dimensions
        batch_size: Manual batch size (None = auto-calculate)
        memory_margin: Fraction of GPU memory to use (default 0.8)
    """
    start_time = datetime.now()
    print(f"Starting batched GPU projection at {start_time}")
    print("=" * 60)

    # Update config
    CONFIG['NB_AN_PROJECTION'] = nb_an_projection
    CONFIG['NB_SC'] = nb_scenarios
    print(f"Configuration: {nb_an_projection} years, {nb_scenarios} scenarios")

    # Load data
    data = load_all_data(data_path)

    if max_accounts:
        data['population'] = data['population'].head(max_accounts)

    n_accounts_total = len(data['population'])
    print(f"\nTotal accounts to process: {n_accounts_total:,}")

    # Calculate batch size if not provided
    if batch_size is None:
        batch_size = calculate_optimal_batch_size(
            n_accounts_total, nb_scenarios, nb_an_projection,
            CONFIG['FREQ_EVAL'], 40, memory_margin
        )
    else:
        print(f"\nUsing manual batch size: {batch_size}")

    # Create GPU lookup tables (shared across all batches)
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
    (cous_base_passif, cous_tx_passif, cous_base_credit, cous_tx_credit,
     cous_base_marche, cous_tx_marche, cous_base_depense, cous_tx_depense,
     cous_base_decheance, cous_tx_decheance, cous_base_mortalite, cous_tx_mortalite,
     cous_base_depot, cous_tx_depot, cous_facteur_80,
     cous_facteur_90) = create_gpu_coussins_lookup(data['coussins_escap'])

    print("Lookup tables created")

    # Copy lookup tables to GPU (once, reused across batches)
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
    d_cous_base_passif = cuda.to_device(cous_base_passif)
    d_cous_tx_passif = cuda.to_device(cous_tx_passif)
    d_cous_base_credit = cuda.to_device(cous_base_credit)
    d_cous_tx_credit = cuda.to_device(cous_tx_credit)
    d_cous_base_marche = cuda.to_device(cous_base_marche)
    d_cous_tx_marche = cuda.to_device(cous_tx_marche)
    d_cous_base_depense = cuda.to_device(cous_base_depense)
    d_cous_tx_depense = cuda.to_device(cous_tx_depense)
    d_cous_base_decheance = cuda.to_device(cous_base_decheance)
    d_cous_tx_decheance = cuda.to_device(cous_tx_decheance)
    d_cous_base_mortalite = cuda.to_device(cous_base_mortalite)
    d_cous_tx_mortalite = cuda.to_device(cous_tx_mortalite)
    d_cous_base_depot = cuda.to_device(cous_base_depot)
    d_cous_tx_depot = cuda.to_device(cous_tx_depot)
    d_cous_facteur_80 = cuda.to_device(cous_facteur_80)
    d_cous_facteur_90 = cuda.to_device(cous_facteur_90)
    print("Lookup tables on GPU")

    # Process in batches
    n_batches = (n_accounts_total + batch_size - 1) // batch_size
    all_results = []
    total_kernel_time = 0

    print(f"\n{'='*60}")
    print(f"PROCESSING {n_batches} BATCHES")
    print(f"{'='*60}")

    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, n_accounts_total)
        batch_n_accounts = batch_end - batch_start

        print(f"\nBatch {batch_idx + 1}/{n_batches}: Accounts {batch_start + 1} to {batch_end}")

        # Get batch data
        batch_population = data['population'].iloc[batch_start:batch_end]
        batch_account_data, batch_account_ids = prepare_account_data(batch_population)

        # Allocate COMPACT output array for this batch
        # Estimate: ~120 timesteps per account-scenario (conservative)
        max_timesteps = (nb_an_projection + 1) * CONFIG['FREQ_EVAL']
        estimated_rows_per_account_scenario = min(120, max_timesteps)
        max_total_rows = batch_n_accounts * nb_scenarios * estimated_rows_per_account_scenario
        n_output_fields = 40
        batch_output_results = np.zeros((max_total_rows, n_output_fields), dtype=np.float32)
        batch_output_counter = np.zeros(1, dtype=np.int32)

        old_memory = batch_n_accounts * nb_scenarios * max_timesteps * n_output_fields * 4 / 1024**3
        new_memory = batch_output_results.nbytes / 1024**3
        print(f"  Compact output array: {batch_output_results.shape} (max {estimated_rows_per_account_scenario} rows/scenario)")
        print(f"  Memory saved: {old_memory:.2f} GB → {new_memory:.2f} GB ({(1-new_memory/old_memory)*100:.1f}% reduction)")

        # Copy batch data to GPU
        print(f"  Copying batch data to GPU...")
        d_account_data = cuda.to_device(batch_account_data)
        d_account_ids = cuda.to_device(batch_account_ids)
        d_output = cuda.to_device(batch_output_results)
        d_output_counter = cuda.to_device(batch_output_counter)

        # Calculate grid dimensions
        blocks_x = (batch_n_accounts + threads_per_block[0] - 1) // threads_per_block[0]
        blocks_y = (nb_scenarios + threads_per_block[1] - 1) // threads_per_block[1]
        blocks_per_grid = (blocks_x, blocks_y)

        print(f"  Launching kernel: Grid={blocks_per_grid}, Block={threads_per_block}")

        # Launch kernel
        batch_kernel_start = datetime.now()
        projection_kernel[blocks_per_grid, threads_per_block](
            d_account_data, d_account_ids, nb_scenarios, nb_an_projection, CONFIG['FREQ_EVAL'],
            d_mortality,
            d_forward_rate, d_ajust_forward, d_rend_dex, d_rend_mm, d_rend_tsx, d_rend_sp500, d_rend_eafe,
            d_min_ferr,
            d_lapse_part_min, d_lapse_part_max,
            d_lapse_tot_min, d_lapse_tot_max, d_lapse_tot_fact,
            d_deposits_pc, d_deposits_var, d_deposits_age_max, d_deposits_i_even,
            d_fees,
            d_acq_vente_rf, d_acq_vente_ac, d_acq_maintien_rf, d_acq_maintien_ac,
            d_acq_frais_ac, d_acq_frais_rf,
            d_cous_base_passif, d_cous_tx_passif, d_cous_base_credit, d_cous_tx_credit,
            d_cous_base_marche, d_cous_tx_marche, d_cous_base_depense, d_cous_tx_depense,
            d_cous_base_decheance, d_cous_tx_decheance, d_cous_base_mortalite, d_cous_tx_mortalite,
            d_cous_base_depot, d_cous_tx_depot, d_cous_facteur_80, d_cous_facteur_90,
            d_output,
            d_output_counter
        )

        cuda.synchronize()
        batch_kernel_end = datetime.now()
        batch_kernel_time = (batch_kernel_end - batch_kernel_start).total_seconds()
        total_kernel_time += batch_kernel_time

        print(f"  Kernel time: {batch_kernel_time:.2f}s")

        # Copy results back
        print(f"  Copying results from GPU...", end='', flush=True)
        copy_start = datetime.now()
        batch_output_results = d_output.copy_to_host()
        batch_output_counter_host = d_output_counter.copy_to_host()
        actual_rows = batch_output_counter_host[0]
        copy_time = (datetime.now() - copy_start).total_seconds()
        print(f" {copy_time:.2f}s ({actual_rows:,} rows)")

        # Process batch results into DataFrame (SUPER FAST - NO FILTERING NEEDED)
        print(f"  Processing batch results...", end='', flush=True)
        process_start = datetime.now()

        # Take only the actual rows written (no zeros, no filtering!)
        valid_rows = batch_output_results[:actual_rows]

        # Create DataFrame directly from compact numpy array
        batch_df = pd.DataFrame(valid_rows, columns=[
            'ID_COMPTE', 'SCN_EVAL', 'AN_EVAL', 'MOIS_EVAL',
            'PRIMES_GARANTIES', 'PREST_DECES', 'PREST_ECH', 'PREST_MRV',
            'FRAIS_ACQUIS', 'COMM_VENTE', 'PRIMES_VARIABLES', 'FRAIS_FIXES',
            'HON_GEST', 'COMM_MAINTIEN', 'VALEUR_MARCHANDE',
            'PASSIF_REDRESSE', 'COUSSIN_CREDIT', 'COUSSIN_MARCHE',
            'COUSSIN_DEPENSE', 'COUSSIN_DECHEANCE', 'COUSSIN_MORTALITE', 'COUSSIN_DEPOT',
            'VP_FRAIS_ACQUIS', 'VP_COMM_VENTE', 'VP_PRIMES_GARANTIES',
            'VP_PRIMES_VARIABLES', 'VP_FRAIS_FIXES', 'VP_HON_GEST', 'VP_COMM_MAINTIEN',
            'VP_PREST_ECH', 'VP_PREST_MRV', 'VP_PREST_DECES', 'VP_VALEUR_MARCHANDE',
            'VP_PASSIF_REDRESSE', 'VP_COUSSIN_CREDIT', 'VP_COUSSIN_MARCHE',
            'VP_COUSSIN_DEPENSE', 'VP_COUSSIN_DECHEANCE', 'VP_COUSSIN_MORTALITE', 'VP_COUSSIN_DEPOT'
        ])

        # Convert ID columns to int
        batch_df['ID_COMPTE'] = batch_df['ID_COMPTE'].astype(int)
        batch_df['SCN_EVAL'] = batch_df['SCN_EVAL'].astype(int)
        batch_df['AN_EVAL'] = batch_df['AN_EVAL'].astype(int)
        batch_df['MOIS_EVAL'] = batch_df['MOIS_EVAL'].astype(int)

        process_time = (datetime.now() - process_start).total_seconds()
        print(f" {process_time:.2f}s")

        all_results.append(batch_df)

        print(f"  ✓ Batch complete: {len(batch_df):,} rows (copy: {copy_time:.1f}s, process: {process_time:.1f}s)")
        print(f"    Buffer utilization: {actual_rows}/{max_total_rows} ({actual_rows/max_total_rows*100:.1f}%)")

        # Clear GPU memory for this batch
        del d_account_data, d_account_ids, d_output, d_output_counter, batch_output_results
        cuda.current_context().deallocations.clear()

    # Combine all batches
    print(f"\n{'='*60}")
    print("Combining all batches...")
    all_results_df = pd.concat(all_results, ignore_index=True)
    print(f"Total projection rows: {len(all_results_df):,}")

    # Aggregate results
    print("Aggregating results...")
    calculs_sommaire = aggregate_by_scenario(all_results_df)
    vp_flux_compte = aggregate_vp_flux_compte(calculs_sommaire)
    vp_flux_total = aggregate_vp_flux_total(vp_flux_compte)
    flux_projetes = aggregate_flux_projetes(calculs_sommaire)

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

    print("\n" + "=" * 60)
    print("BATCHED GPU PROJECTION COMPLETE")
    print("=" * 60)
    print(f"Total processing time: {total_duration:.2f} seconds ({total_duration / 60:.2f} minutes)")
    print(f"Total kernel time: {total_kernel_time:.2f} seconds")
    print(f"Overhead time: {total_duration - total_kernel_time:.2f} seconds")
    print(f"Accounts processed: {n_accounts_total:,}")
    print(f"Scenarios per account: {nb_scenarios}")
    print(f"Number of batches: {n_batches}")
    print(f"Average batch time: {total_duration / n_batches:.2f} seconds")
    print(f"Total rows generated: {len(all_results_df):,}")
    print(f"Total PV of flows: ${vp_flux_total['VP_FLUX_TOT'].iloc[0]:,.2f}")
    print("=" * 60)

    return {
        'flux_projetes': flux_projetes,
        'vp_flux_compte': vp_flux_compte,
        'vp_flux_total': vp_flux_total,
        'all_results': all_results_df,
        'n_batches': n_batches,
        'batch_size': batch_size,
        'total_kernel_time': total_kernel_time,
        'total_time': total_duration
    }




if __name__ == "__main__":
    # Check CUDA availability
    if not cuda.is_available():
        print("ERROR: CUDA is not available. Please check your GPU setup.")
        exit(1)

    print(f"CUDA Device: {cuda.get_current_device().name}")

    # Set paths
    DATA_PATH = Path("algo2/data_in")
    OUTPUT_PATH = Path("algo2/data_out_gpu")

    # Run batched GPU projection
    results = run_projection_gpu_batched(
        data_path=DATA_PATH,
        output_path=OUTPUT_PATH,
        nb_an_projection=100,
        nb_scenarios=100,
        max_accounts=None,  # None = all accounts (200,000)
        threads_per_block=(16, 8),
        batch_size=None,  # None = auto-calculate based on GPU memory
        memory_margin=0.85  # Use 85% of free GPU memory
    )

    if results:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Processed {results['n_batches']} batches")
        print(f"Average accounts per batch: {results['batch_size']}")
        print(f"Kernel efficiency: {results['total_kernel_time']/results['total_time']:.1%}")
        print("\nVP_FLUX_TOTAL:")
        print(results['vp_flux_total'])
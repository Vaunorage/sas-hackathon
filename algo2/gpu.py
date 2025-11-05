import numpy as np
import pandas as pd
from pathlib import Path
from numba import cuda, float32, int32
import math
from typing import Dict, Tuple
from datetime import datetime
from paths import HERE
import argparse


# =============================================================================
# CONFIGURATION
# =============================================================================

def create_config(nb_scenarios=100, nb_years=100, max_accounts=None, freq_eval=12, threads_per_block=256):
    """Create configuration dictionary."""
    return {
        'NBCPT': max_accounts if max_accounts else 9999999,
        'NB_SC': nb_scenarios,
        'NB_AN_PROJECTION': nb_years,
        'FREQ_EVAL': freq_eval,
        'THREADS_PER_BLOCK': threads_per_block,
    }


# =============================================================================
# CUDA DEVICE FUNCTIONS
# =============================================================================

@cuda.jit(device=True)
def calculate_age_gpu(birth_year, birth_month, current_year, current_month):
    age = current_year - birth_year
    if current_month < birth_month:
        age -= 1
    return max(age, 1)


@cuda.jit(device=True)
def calculate_mortality_age_gpu(age, birth_month, current_month):
    month_diff = birth_month - current_month
    if month_diff <= 0: month_diff += 12
    return age + 1 if month_diff <= 6 else age


@cuda.jit(device=True)
def lookup_fees_gpu(fees_table, annee_reelle):
    """Simplified lookup for fixed fees by year."""
    if annee_reelle < len(fees_table):
        return fees_table[annee_reelle]
    return 0.0


@cuda.jit(device=True)
def lookup_deposits_gpu(deposits_table, duree_max10, id_depot, deposits_shape):
    """Lookup deposit percentage."""
    idx = (duree_max10 - 1) * deposits_shape[1] + id_depot
    if idx < len(deposits_table):
        return deposits_table[idx]
    return 0.0


@cuda.jit(device=True)
def lookup_acquisition_gpu(acq_table, duree_max10, id_acqui, acq_shape):
    """Lookup acquisition/maintenance commission rates."""
    idx = (duree_max10 - 1) * acq_shape[1] + id_acqui
    if idx < len(acq_table):
        return acq_table[idx]  # Assuming table stores PC_COMMISSION_MAINTIEN
    return 0.0


@cuda.jit(device=True)
def lookup_mortality_gpu(mortality_table, i_sexe, age, annee_reelle, i_produit_regr, mortality_shape):
    age = min(age, 120)
    idx = (i_sexe * mortality_shape[1] * mortality_shape[2] * mortality_shape[3] +
           age * mortality_shape[2] * mortality_shape[3] +
           annee_reelle * mortality_shape[3] + i_produit_regr)
    if idx < len(mortality_table): return mortality_table[idx]
    return 0.001


@cuda.jit(device=True)
def lookup_returns_gpu(returns_table, scn_eval, an_eval, mois_eval, returns_shape):
    idx = (scn_eval * returns_shape[1] * returns_shape[2] + an_eval * returns_shape[2] + mois_eval)
    if idx * 7 < len(returns_table):
        base_idx = idx * 7
        return (returns_table[base_idx], returns_table[base_idx + 1], returns_table[base_idx + 2],
                returns_table[base_idx + 3], returns_table[base_idx + 4], returns_table[base_idx + 5],
                returns_table[base_idx + 6])
    return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0


@cuda.jit(device=True)
def lookup_min_ferr_gpu(min_ferr_table, age):
    if age < len(min_ferr_table): return min_ferr_table[age]
    return 0.0


@cuda.jit(device=True)
def calculate_lapse_level(vm_vg_ratio):
    if vm_vg_ratio <= 0.5:
        return 1
    elif vm_vg_ratio <= 0.75:
        return 2
    else:
        return 3


@cuda.jit(device=True)
def lookup_lapse_rates_gpu(lapse_tot_table, lapse_part_table, duree_max10, age, id_lapse, i_regime_2,
                           vm_vg_ratio, age_decaissement, lapse_tot_shape, lapse_part_shape):
    lapse_niv_tot = calculate_lapse_level(vm_vg_ratio)
    idx_tot = ((duree_max10 - 1) * lapse_tot_shape[1] * lapse_tot_shape[2] +
               id_lapse * lapse_tot_shape[2] + (lapse_niv_tot - 1))
    if idx_tot * 3 < len(lapse_tot_table):
        base_idx = idx_tot * 3
        tx_min, tx_max, fact_dim = lapse_tot_table[base_idx], lapse_tot_table[base_idx + 1], lapse_tot_table[
            base_idx + 2]
    else:
        tx_min, tx_max, fact_dim = 0.0, 0.0, 1.0
    if tx_min == tx_max:
        lapse_tot = tx_min
    else:
        if lapse_niv_tot == 1:
            interpolation = (vm_vg_ratio - 0.0) / 0.5 if vm_vg_ratio > 0 else 0
        elif lapse_niv_tot == 2:
            interpolation = (vm_vg_ratio - 0.5) / 0.25
        else:
            interpolation = (vm_vg_ratio - 0.75) / 999.24
        lapse_tot = interpolation * (tx_max - tx_min) + tx_min
    if age >= age_decaissement: lapse_tot *= fact_dim

    lapse_niv_part = calculate_lapse_level(vm_vg_ratio)
    idx_part = (age * lapse_part_shape[1] * lapse_part_shape[2] * lapse_part_shape[3] +
                id_lapse * lapse_part_shape[2] * lapse_part_shape[3] +
                i_regime_2 * lapse_part_shape[3] + (lapse_niv_part - 1))
    if idx_part * 2 < len(lapse_part_table):
        base_idx = idx_part * 2
        tx_part_min, tx_part_max = lapse_part_table[base_idx], lapse_part_table[base_idx + 1]
    else:
        tx_part_min, tx_part_max = 0.0, 0.0
    if tx_part_min == tx_part_max:
        lapse_part = tx_part_min
    else:
        if lapse_niv_part == 1:
            interpolation = (vm_vg_ratio - 0.0) / 0.5 if vm_vg_ratio > 0 else 0
        elif lapse_niv_part == 2:
            interpolation = (vm_vg_ratio - 0.5) / 0.25
        else:
            interpolation = (vm_vg_ratio - 0.75) / 999.24
        lapse_part = interpolation * (tx_part_max - tx_part_min) + tx_part_min
    return lapse_tot, lapse_part


# =============================================================================
# MAIN GPU KERNEL
# =============================================================================
@cuda.jit
def project_account_scenario_kernel(
        accounts, account_features, n_accounts, n_scenarios, n_years, freq_eval,
        mortality_table, mortality_shape,
        returns_table, returns_shape,
        min_ferr_table,
        lapse_tot_table, lapse_tot_shape,
        lapse_part_table, lapse_part_shape,
        fees_table,
        deposits_table, deposits_shape,
        acquisition_table, acquisition_shape,
        output_cashflows, output_pvs, output_vm
):
    thread_id = cuda.grid(1)
    if thread_id >= n_accounts * n_scenarios: return

    account_idx = thread_id // n_scenarios
    scenario_idx = thread_id % n_scenarios

    # Load account data
    annee_eval_ini = int32(accounts[account_idx, 1])
    mois_eval_ini = int32(accounts[account_idx, 2])
    annee_nais = int32(accounts[account_idx, 3])
    mois_nais = int32(accounts[account_idx, 4])
    i_sexe = int32(accounts[account_idx, 5])
    i_produit_regr = int32(accounts[account_idx, 6])
    id_produit = int32(accounts[account_idx, 7])
    id_lapse = int32(accounts[account_idx, 8])
    i_regime_2 = int32(accounts[account_idx, 9])
    id_depot = int32(accounts[account_idx, 10])
    id_acqui = int32(accounts[account_idx, 11])
    age_fin_contrat = int32(accounts[account_idx, 12])
    age_decaissement = int32(accounts[account_idx, 13])

    # Initialize state
    mt_vm = accounts[account_idx, 14]
    mt_gar_deces = accounts[account_idx, 15]
    mt_gar_ech = accounts[account_idx, 16]
    mt_srg = accounts[account_idx, 17]
    mt_dex, mt_mm, mt_tsx, mt_sp500, mt_eafe = (
        accounts[account_idx, 18], accounts[account_idx, 19], accounts[account_idx, 20],
        accounts[account_idx, 21], accounts[account_idx, 22]
    )
    pc_honoraires_gest = accounts[account_idx, 23]
    pc_frais_garantie = accounts[account_idx, 24]
    pc_gar_deces_1 = accounts[account_idx, 25]
    pc_gar_ech = accounts[account_idx, 26]
    pc_rfg = accounts[account_idx, 27]
    mt_boni_deces = accounts[account_idx, 28]
    pc_boni_deces = accounts[account_idx, 29]
    mt_gar_deces_orig = accounts[account_idx, 15]

    tx_survie = 1.0
    tx_actualisation = 1.0
    mt_min_ferr_proj = 0.0
    mt_vm_orig = mt_vm
    time_idx = 0

    for an_eval in range(n_years + 1):
        for mois_simul in range(1, freq_eval + 1):
            annee_reelle = annee_eval_ini + an_eval - 1
            mois_eval = mois_simul * 12 // freq_eval
            age = calculate_age_gpu(annee_nais, mois_nais, annee_reelle, mois_eval)

            if age > age_fin_contrat: break
            if an_eval == 0 and mois_eval < mois_eval_ini: continue
            if an_eval == 0 and mois_eval != 12: continue
            if tx_survie <= 0.0001: break
            if mt_vm <= 0 and i_produit_regr == 0: break

            current_date = float32(annee_reelle) + float32(mois_eval) / 12.0
            issue_date = float32(annee_eval_ini) + float32(mois_eval_ini) / 12.0
            duree = int32(current_date - issue_date) + 1
            duree_max10 = min(duree, 10)

            tx_survie_deb = tx_survie
            ajust_nouv_affaires = 1.0

            # === STEP 1: DEPOSITS ===
            depot_futur = 0.0
            pc_depot_annuel = lookup_deposits_gpu(deposits_table, duree_max10, id_depot, deposits_shape)
            if pc_depot_annuel > 0 and age < age_decaissement and pc_gar_deces_1 > 0:
                base = mt_gar_deces_orig / pc_gar_deces_1
                depot_futur = (base * pc_depot_annuel) / freq_eval

            if depot_futur > 0 and mt_vm > 0:
                proportion = depot_futur / mt_vm
                mt_dex += mt_dex * proportion
                mt_mm += mt_mm * proportion
                mt_tsx += mt_tsx * proportion
                mt_sp500 += mt_sp500 * proportion
                mt_eafe += mt_eafe * proportion
                mt_gar_deces += depot_futur
                mt_gar_ech += depot_futur
                if mt_srg > 0: mt_srg += depot_futur

            # === STEP 2: LOOKUPS & RETURNS ===
            age_mort = calculate_mortality_age_gpu(age, mois_nais, mois_eval)
            qx = lookup_mortality_gpu(mortality_table, i_sexe, age_mort, annee_reelle, i_produit_regr, mortality_shape)
            qx = 1.0 - math.pow(1.0 - qx, 1.0 / freq_eval * ajust_nouv_affaires)

            forward_rate, _, renddex, rendmm, rendtsx, rendsp500, rendeafe = \
                lookup_returns_gpu(returns_table, scenario_idx + 1, an_eval, mois_eval, returns_shape)

            tx_actualisation *= math.exp(-forward_rate * ajust_nouv_affaires)

            mt_dex *= math.exp(renddex * ajust_nouv_affaires)
            mt_mm *= math.exp(rendmm * ajust_nouv_affaires)
            mt_tsx *= math.exp(rendtsx * ajust_nouv_affaires)
            mt_sp500 *= math.exp(rendsp500 * ajust_nouv_affaires)
            mt_eafe *= math.exp(rendeafe * ajust_nouv_affaires)
            mt_vm_av_retrait_frais = mt_dex + mt_mm + mt_tsx + mt_sp500 + mt_eafe

            # === STEP 3: LAPSE & SURVIVAL ===
            ratio_base = mt_gar_deces + mt_boni_deces
            ratio1 = pc_gar_ech / max(mt_gar_ech, 0.01) if mt_gar_ech > 0 else 9999.0
            ratio2 = pc_gar_deces_1 / max(ratio_base, 0.01) if ratio_base > 0 else 9999.0
            ratio3 = 1.0 / max(mt_srg, 0.01) if mt_srg > 0 else 9999.0
            vm_vg_ratio = min(10.0, (mt_vm + mt_vm_av_retrait_frais) / 2.0 * min(ratio1, ratio2, ratio3))

            lapse_tot, lapse_part = lookup_lapse_rates_gpu(lapse_tot_table, lapse_part_table, duree_max10, age,
                                                           id_lapse,
                                                           i_regime_2, vm_vg_ratio, age_decaissement, lapse_tot_shape,
                                                           lapse_part_shape)
            lapse = 1.0 - math.pow(1.0 - lapse_tot - lapse_part, 1.0 / freq_eval * ajust_nouv_affaires)

            tx_survie *= (1.0 - qx) * (1.0 - lapse)
            if pc_boni_deces > 0: mt_boni_deces += mt_gar_deces * pc_boni_deces / freq_eval

            # === STEP 4: FEES & COMMISSIONS (CALCULATE ALL CASH FLOWS) ===
            hon_gest = -mt_vm_av_retrait_frais * (math.exp(pc_honoraires_gest / freq_eval) - 1) * tx_survie_deb

            fixed_fee_annual = lookup_fees_gpu(fees_table, annee_reelle)
            frais_fixes = -fixed_fee_annual / freq_eval * tx_survie_deb

            pc_comm_maintien = lookup_acquisition_gpu(acquisition_table, duree_max10, id_acqui, acquisition_shape)
            comm_maintien = -mt_vm_av_retrait_frais * (math.exp(pc_comm_maintien / freq_eval) - 1) * tx_survie_deb

            # === STEP 5: APPLY DEDUCTIONS TO MARKET VALUE ===
            # The order of deductions matters. RFG -> Guarantee -> Other Fees
            mt_vm_av_retrait = mt_vm_av_retrait_frais * math.exp(-pc_rfg / freq_eval)

            guarantee_fee_amount = min(mt_vm_av_retrait * pc_frais_garantie / freq_eval, mt_vm_av_retrait)
            primes_garanties = guarantee_fee_amount * tx_survie_deb
            mt_vm_av_retrait = max(mt_vm_av_retrait - guarantee_fee_amount, 0.0)

            # Deduct other fees from VM
            mt_vm_av_retrait = max(mt_vm_av_retrait - abs(hon_gest / tx_survie_deb), 0.0)
            mt_vm_av_retrait = max(mt_vm_av_retrait - abs(frais_fixes / tx_survie_deb), 0.0)
            mt_vm_av_retrait = max(mt_vm_av_retrait - abs(comm_maintien / tx_survie_deb), 0.0)

            # === STEP 6: WITHDRAWALS & BENEFITS ===
            retrait = 0.0
            if age + 1 >= age_decaissement and mt_vm_av_retrait > 0:
                min_ferr_rate = lookup_min_ferr_gpu(min_ferr_table, age)
                if mois_eval == 12 // freq_eval: mt_min_ferr_proj = mt_vm * min_ferr_rate
                retrait = mt_min_ferr_proj / freq_eval

            if mt_vm_av_retrait <= retrait:
                mt_gar_ech, mt_gar_deces, mt_boni_deces = 0.0, 0.0, 0.0
            else:
                proportion = 1.0 - retrait / mt_vm_av_retrait
                mt_gar_ech *= proportion;
                mt_gar_deces *= proportion;
                mt_boni_deces *= proportion
                mt_srg = max(mt_srg - retrait, 0.0)

            mt_vm_ap_retrait = max(mt_vm_av_retrait - retrait, 0.0)

            prest_deces = qx * -max(0.0, (mt_gar_deces + mt_boni_deces) - mt_vm_ap_retrait) * tx_survie_deb
            mt_vm = mt_vm_ap_retrait

            if mt_vm_orig > 0 and mt_vm > 0:
                ratio = mt_vm / mt_vm_orig
                mt_dex, mt_mm, mt_tsx, mt_sp500, mt_eafe = (accounts[account_idx, 18] * ratio,
                                                            accounts[account_idx, 19] * ratio,
                                                            accounts[account_idx, 20] * ratio,
                                                            accounts[account_idx, 21] * ratio,
                                                            accounts[account_idx, 22] * ratio)

            # === STEP 7: WRITE OUTPUTS ===
            # Cashflows: [primes_gar, prest_deces, val_march, hon_gest, frais_fixes, comm_maintien]
            output_cashflows[thread_id, time_idx, 0] = primes_garanties
            output_cashflows[thread_id, time_idx, 1] = prest_deces
            output_cashflows[thread_id, time_idx, 2] = mt_vm * tx_survie
            output_cashflows[thread_id, time_idx, 3] = hon_gest
            output_cashflows[thread_id, time_idx, 4] = frais_fixes
            output_cashflows[thread_id, time_idx, 5] = comm_maintien

            # PVs
            output_pvs[thread_id, time_idx, 0] = primes_garanties * tx_actualisation
            output_pvs[thread_id, time_idx, 1] = prest_deces * tx_actualisation
            output_pvs[thread_id, time_idx, 2] = (mt_vm * tx_survie) * tx_actualisation / freq_eval
            output_pvs[thread_id, time_idx, 3] = hon_gest * tx_actualisation
            output_pvs[thread_id, time_idx, 4] = frais_fixes * tx_actualisation
            output_pvs[thread_id, time_idx, 5] = comm_maintien * tx_actualisation

            output_vm[thread_id, time_idx] = mt_vm

            time_idx += 1
            if time_idx >= output_cashflows.shape[1]: break
        if time_idx >= output_cashflows.shape[1]: break


# =============================================================================
# DATA PREPARATION
# =============================================================================
def parse_percentage(value):
    if pd.isna(value): return 0.0
    if isinstance(value, str):
        value = value.strip()
        is_negative = value.startswith('(') and value.endswith(')')
        if is_negative: value = value[1:-1]
        numeric_value = float(value.replace('%', '')) / 100.0 if '%' in value else float(value)
        return -numeric_value if is_negative else numeric_value
    return float(value)


def clean_numeric(df, columns):
    for col in columns:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].apply(parse_percentage)
    return df


def prepare_gpu_data(data_path: Path, config: Dict) -> Dict:
    print("Loading and preparing data for GPU...")

    # Load all tables
    dfs = {name: pd.read_csv(data_path.joinpath(f"{name.upper()}.csv"), sep=';', encoding='utf-8')
           for name in ['population', 'mortalite', 'rendements', 'min_ferr', 'tx_lapse_part',
                        'tx_lapse_tot', 'frais_admin', 'depots_futurs', 'acquisition']}

    # Normalize columns and clean
    for name, df in dfs.items():
        df.columns = df.columns.str.upper()

    clean_numeric(dfs['population'],
                  [c for c in dfs['population'].columns if c.startswith('PC_') or c.startswith('TAUX_')])
    clean_numeric(dfs['mortalite'], ['QX'])
    clean_numeric(dfs['rendements'],
                  ['FORWARD_RATE', 'AJUST_FORWARD_RATE_VM_0', 'RENDDEX_AN', 'RENDMM_AN', 'RENDTSX_AN', 'RENDSP500_AN',
                   'RENDEAFE_AN'])
    clean_numeric(dfs['min_ferr'], ['MIN_FERR'])
    clean_numeric(dfs['tx_lapse_part'], ['TX_LAPSE_PART_MIN', 'TX_LAPSE_PART_MAX'])
    for col in ['TX_LAPSE_TOT_MIN', 'TX_LAPSE_TOT_MAX', 'FACT_DIM']:
        dfs['tx_lapse_tot'][col] = pd.to_numeric(dfs['tx_lapse_tot'][col], errors='coerce').fillna(0)
    clean_numeric(dfs['frais_admin'], ['FRAIS'])
    clean_numeric(dfs['depots_futurs'], ['PC_DEPOT_ANNUEL'])
    clean_numeric(dfs['acquisition'], ['PC_COMMISSION_MAINTIEN_AC', 'PC_COMMISSION_MAINTIEN_RF'])

    # Filter data
    dfs['population'] = dfs['population'][dfs['population']['ID_COMPTE'] <= config['NBCPT']]
    dfs['rendements'] = dfs['rendements'][(dfs['rendements']['SCN_EVAL'] <= config['NB_SC']) & (
                dfs['rendements']['AN_EVAL'] <= config['NB_AN_PROJECTION'])]

    # Prepare accounts array
    account_cols = ['ID_COMPTE', 'ANNEE_EVALUATION_INI', 'MOIS_EVALUATION_INI', 'ANNEE_NAIS', 'MOIS_NAIS',
                    'I_SEXE', 'I_PRODUIT_REGR', 'ID_PRODUIT', 'ID_LAPSE', 'I_REGIME_2', 'ID_DEPOT', 'ID_ACQUI',
                    'AGE_FIN_CONTRAT', 'AGE_DECAISSEMENT', 'MT_VM', 'MT_GAR_DECES', 'MT_GAR_ECH', 'MT_SRG',
                    'MT_DEX', 'MT_MM', 'MT_TSX', 'MT_SP500', 'MT_EAFE', 'PC_HONORAIRES_GEST',
                    'PC_FRAIS_GARANTIE', 'PC_GAR_DECES_1', 'PC_GAR_ECH', 'PC_RFG', 'MT_BONI_DECES', 'PC_BONI_DECES']
    for col in account_cols:
        if col not in dfs['population'].columns: dfs['population'][col] = 0.0
    accounts_array = np.ascontiguousarray(dfs['population'][account_cols].values.astype(np.float32))

    # Prepare lookup tables (Mortality, Returns, Min FERR, Lapse)
    mortalite = dfs['mortalite']
    mortality_shape = (mortalite['I_SEXE'].max() + 1, 121, mortalite['ANNEE_REELLE'].max() + 1,
                       mortalite['I_PRODUIT_REGR'].max() + 1)
    mortality_table = np.zeros(mortality_shape, dtype=np.float32).flatten()
    for _, r in mortalite.iterrows():
        idx = (int(r['I_SEXE']) * mortality_shape[1] * mortality_shape[2] * mortality_shape[3] + int(
            r['AGE_MORTALITE']) * mortality_shape[2] * mortality_shape[3] + int(r['ANNEE_REELLE']) * mortality_shape[
                   3] + int(r['I_PRODUIT_REGR']))
        if idx < len(mortality_table): mortality_table[idx] = r['QX']

    # ... (Other table preps are similar and can be condensed)
    rendements = dfs['rendements']
    returns_shape = (rendements['SCN_EVAL'].max() + 1, rendements['AN_EVAL'].max() + 1,
                     rendements['MOIS_EVAL'].max() + 1)
    returns_table = np.zeros(returns_shape + (7,), dtype=np.float32).flatten()
    for _, r in rendements.iterrows():
        idx = (int(r['SCN_EVAL']) * returns_shape[1] * returns_shape[2] + int(r['AN_EVAL']) * returns_shape[2] + int(
            r['MOIS_EVAL']))
        base = idx * 7
        if base + 6 < len(returns_table): returns_table[base:base + 7] = r[
            ['FORWARD_RATE', 'AJUST_FORWARD_RATE_VM_0', 'RENDDEX_AN', 'RENDMM_AN', 'RENDTSX_AN', 'RENDSP500_AN',
             'RENDEAFE_AN']]

    min_ferr_table = np.zeros(121, dtype=np.float32)
    for _, r in dfs['min_ferr'].iterrows():
        if int(r['AGE']) < 121: min_ferr_table[int(r['AGE'])] = r['MIN_FERR']

    lapse_tot, lapse_part = dfs['tx_lapse_tot'], dfs['tx_lapse_part']
    lapse_tot_shape = (10, lapse_tot['ID_LAPSE'].max() + 1, 3)
    lapse_tot_table = np.zeros(lapse_tot_shape + (3,), dtype=np.float32).flatten()
    for _, r in lapse_tot.iterrows():
        idx = ((int(r['DUREE_MAX10']) - 1) * lapse_tot_shape[1] * lapse_tot_shape[2] + int(r['ID_LAPSE']) *
               lapse_tot_shape[2] + (int(r['LAPSE_NIV_TOT']) - 1))
        base = idx * 3
        if base + 2 < len(lapse_tot_table): lapse_tot_table[base:base + 3] = r[
            ['TX_LAPSE_TOT_MIN', 'TX_LAPSE_TOT_MAX', 'FACT_DIM']]

    lapse_part_shape = (121, lapse_part['ID_LAPSE'].max() + 1, lapse_part['I_REGIME_2'].max() + 1, 3)
    lapse_part_table = np.zeros(lapse_part_shape + (2,), dtype=np.float32).flatten()
    for _, r in lapse_part.iterrows():
        idx = (int(r['AGE']) * lapse_part_shape[1] * lapse_part_shape[2] * lapse_part_shape[3] + int(r['ID_LAPSE']) *
               lapse_part_shape[2] * lapse_part_shape[3] + int(r['I_REGIME_2']) * lapse_part_shape[3] + (
                           int(r['LAPSE_NIV_PART']) - 1))
        base = idx * 2
        if base + 1 < len(lapse_part_table): lapse_part_table[base:base + 2] = r[
            ['TX_LAPSE_PART_MIN', 'TX_LAPSE_PART_MAX']]

    # Prepare new tables
    fees_agg = dfs['frais_admin'].groupby('ANNEE_REELLE')['FRAIS'].mean()
    fees_table = np.ascontiguousarray(
        fees_agg.reindex(range(fees_agg.index.max() + 1), fill_value=0).astype(np.float32).values)

    acqui = dfs['acquisition']
    acqui['PC_COMM_MAINTIEN'] = (acqui['PC_COMMISSION_MAINTIEN_AC'] + acqui[
        'PC_COMMISSION_MAINTIEN_RF']) / 2  # Simplification
    acquisition_shape = (10, acqui['ID_ACQUI'].max() + 1)
    acquisition_table = np.zeros(acquisition_shape, dtype=np.float32).flatten()
    for _, r in acqui.iterrows():
        idx = (int(r['DUREE_MAX10']) - 1) * acquisition_shape[1] + int(r['ID_ACQUI'])
        if idx < len(acquisition_table): acquisition_table[idx] = r['PC_COMM_MAINTIEN']

    deposits = dfs['depots_futurs']
    deposits_shape = (10, deposits['ID_DEPOT'].max() + 1)
    deposits_table = np.zeros(deposits_shape, dtype=np.float32).flatten()
    for _, r in deposits.iterrows():
        idx = (int(r['DUREE_MAX10']) - 1) * deposits_shape[1] + int(r['ID_DEPOT'])
        if idx < len(deposits_table): deposits_table[idx] = r['PC_DEPOT_ANNUEL']

    return {
        'accounts': accounts_array, 'mortality_table': np.ascontiguousarray(mortality_table),
        'mortality_shape': np.array(mortality_shape, dtype=np.int32),
        'returns_table': np.ascontiguousarray(returns_table), 'returns_shape': np.array(returns_shape, dtype=np.int32),
        'min_ferr_table': np.ascontiguousarray(min_ferr_table),
        'lapse_tot_table': np.ascontiguousarray(lapse_tot_table),
        'lapse_tot_shape': np.array(lapse_tot_shape, dtype=np.int32),
        'lapse_part_table': np.ascontiguousarray(lapse_part_table),
        'lapse_part_shape': np.array(lapse_part_shape, dtype=np.int32), 'fees_table': fees_table,
        'deposits_table': np.ascontiguousarray(deposits_table),
        'deposits_shape': np.array(deposits_shape, dtype=np.int32),
        'acquisition_table': np.ascontiguousarray(acquisition_table),
        'acquisition_shape': np.array(acquisition_shape, dtype=np.int32)
    }


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def run_gpu_projection(data_path: Path, output_path: Path, max_accounts: int = None,
                       nb_scenarios: int = 100, nb_years: int = 100):
    start_time = datetime.now()
    print("=" * 80 + "\nGPU ACTUARIAL PROJECTION\n" + "=" * 80)
    config = create_config(nb_scenarios=nb_scenarios, nb_years=nb_years, max_accounts=max_accounts)
    gpu_data = prepare_gpu_data(data_path, config)

    n_accounts, n_scenarios, n_years, freq_eval = len(gpu_data['accounts']), config['NB_SC'], config[
        'NB_AN_PROJECTION'], config['FREQ_EVAL']
    max_timesteps = (n_years + 1) * freq_eval
    total_paths = n_accounts * n_scenarios
    print(
        f"\nConfiguration: {n_accounts} accounts, {n_scenarios} scenarios, {n_years} years -> {total_paths:,} total paths")

    # Allocate wider output arrays for more cash flows
    output_cashflows = np.ascontiguousarray(np.zeros((total_paths, max_timesteps, 6), dtype=np.float32))
    output_pvs = np.ascontiguousarray(np.zeros((total_paths, max_timesteps, 6), dtype=np.float32))
    output_vm = np.ascontiguousarray(np.zeros((total_paths, max_timesteps), dtype=np.float32))

    print("\nTransferring data to GPU...")
    d_data = {k: cuda.to_device(v) for k, v in gpu_data.items()}
    d_output_cashflows = cuda.to_device(output_cashflows)
    d_output_pvs = cuda.to_device(output_pvs)
    d_output_vm = cuda.to_device(output_vm)

    threads_per_block = config['THREADS_PER_BLOCK']
    blocks_per_grid = (total_paths + threads_per_block - 1) // threads_per_block
    print(f"\nLaunching GPU kernel ({blocks_per_grid} blocks, {threads_per_block} threads)...")

    kernel_start = datetime.now()
    project_account_scenario_kernel[blocks_per_grid, threads_per_block](
        d_data['accounts'], gpu_data['accounts'].shape[1], n_accounts, n_scenarios, n_years, freq_eval,
        d_data['mortality_table'], d_data['mortality_shape'], d_data['returns_table'], d_data['returns_shape'],
        d_data['min_ferr_table'], d_data['lapse_tot_table'], d_data['lapse_tot_shape'],
        d_data['lapse_part_table'], d_data['lapse_part_shape'], d_data['fees_table'],
        d_data['deposits_table'], d_data['deposits_shape'], d_data['acquisition_table'], d_data['acquisition_shape'],
        d_output_cashflows, d_output_pvs, d_output_vm
    )
    cuda.synchronize()
    kernel_end = datetime.now()
    print(f"  ✓ Kernel execution time: {(kernel_end - kernel_start).total_seconds():.2f} seconds")

    print("\nTransferring results from GPU and aggregating...")
    output_pvs = d_output_pvs.copy_to_host()

    pvs_avg = output_pvs.reshape(n_accounts, n_scenarios, max_timesteps, 6).mean(axis=1)

    vp_flux_compte = pd.DataFrame({
        'ID_COMPTE': gpu_data['accounts'][:, 0].astype(int),
        'VP_PRIMES_GARANTIES': pvs_avg[:, :, 0].sum(axis=1),
        'VP_PREST_DECES': pvs_avg[:, :, 1].sum(axis=1),
        'VP_VALEUR_MARCHANDE': pvs_avg[:, :, 2].sum(axis=1),
        'VP_HON_GEST': pvs_avg[:, :, 3].sum(axis=1),
        'VP_FRAIS_FIXES': pvs_avg[:, :, 4].sum(axis=1),
        'VP_COMM_MAINTIEN': pvs_avg[:, :, 5].sum(axis=1),
    })

    vp_flux_compte_cols = ['ID_COMPTE', 'VP_FRAIS_ACQUIS', 'VP_COMM_VENTE', 'VP_PRIMES_GARANTIES',
                           'VP_PRIMES_VARIABLES',
                           'VP_FRAIS_FIXES', 'VP_HON_GEST', 'VP_COMM_MAINTIEN', 'VP_PREST_ECH', 'VP_PREST_MRV',
                           'VP_PREST_DECES',
                           'VP_PASSIF_REDRESSE', 'VP_COUSSIN_CREDIT', 'VP_COUSSIN_MARCHE', 'VP_COUSSIN_DEPENSE',
                           'VP_COUSSIN_DECHEANCE', 'VP_COUSSIN_MORTALITE', 'VP_COUSSIN_DEPOT', 'VP_VALEUR_MARCHANDE']
    for col in vp_flux_compte_cols:
        if col not in vp_flux_compte.columns: vp_flux_compte[col] = 0.0
    vp_flux_compte = vp_flux_compte[vp_flux_compte_cols]

    vp_cols_to_sum = [c for c in vp_flux_compte.columns if c.startswith('VP_') and c != 'VP_VALEUR_MARCHANDE']
    vp_flux_total = pd.DataFrame({'CATEGORIE': ['TOTAL'], 'VP_FLUX_TOT': [vp_flux_compte[vp_cols_to_sum].sum().sum()]})

    print("\nSaving outputs...")
    output_path.mkdir(parents=True, exist_ok=True)
    vp_flux_compte.to_csv(output_path / "VP_FLUX_COMPTE_GPU.csv", index=False, sep=';')
    vp_flux_total.to_csv(output_path / "VP_FLUX_TOTAL_GPU.csv", index=False, sep=';')
    print(f"  ✓ Saved outputs to {output_path}")

    total_time = (datetime.now() - start_time).total_seconds()
    print("\n" + "=" * 80 + "\nPROJECTION COMPLETE\n" + "=" * 80)
    print(f"Total time: {total_time:.2f}s | Kernel time: {(kernel_end - kernel_start).total_seconds():.2f}s")
    print(f"Paths per second (kernel): {total_paths / (kernel_end - kernel_start).total_seconds():,.0f}")
    print(f"Total PV: ${vp_flux_total['VP_FLUX_TOT'].iloc[0]:,.2f}")
    print("=" * 80)

    return {'vp_flux_compte': vp_flux_compte, 'vp_flux_total': vp_flux_total}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GPU Actuarial Projection')
    parser.add_argument('--accounts', type=int, default=3, help='Number of accounts to process (default: all)')
    parser.add_argument('--scenarios', type=int, default=100, help='Number of scenarios (default: 100)')
    parser.add_argument('--years', type=int, default=100, help='Number of projection years (default: 100)')
    parser.add_argument('--data-path', type=str, default=None, help='Path to input data directory')
    parser.add_argument('--output-path', type=str, default=None, help='Path to output directory')
    args = parser.parse_args()

    DATA_PATH = Path(args.data_path) if args.data_path else HERE.joinpath("algo2/data_in")
    OUTPUT_PATH = Path(args.output_path) if args.output_path else HERE.joinpath("algo2/data_out")

    print(
        f"\nRunning GPU projection with:\n  Accounts: {args.accounts or 'ALL'}, Scenarios: {args.scenarios}, Years: {args.years}")

    results = run_gpu_projection(data_path=DATA_PATH, output_path=OUTPUT_PATH, max_accounts=args.accounts,
                                 nb_scenarios=args.scenarios, nb_years=args.years)

    print("\n" + "=" * 80 + "\nSAMPLE RESULTS\n" + "=" * 80)
    print("\nVP_FLUX_TOTAL:\n", results['vp_flux_total'])
    print("\nVP_FLUX_COMPTE (first 5 accounts):\n", results['vp_flux_compte'].head(5).to_string())
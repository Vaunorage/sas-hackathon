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
def create_config(nb_scenarios=100, nb_years=100, max_accounts=None, freq_eval=12, threads_per_block=128):
    return {'NBCPT': max_accounts or 9999999, 'NB_SC': nb_scenarios, 'NB_AN_PROJECTION': nb_years,
            'FREQ_EVAL': freq_eval, 'THREADS_PER_BLOCK': threads_per_block}


# =============================================================================
# UTILITY FUNCTIONS
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


# =============================================================================
# CUDA DEVICE FUNCTIONS
# =============================================================================
@cuda.jit(device=True)
def calculate_age_gpu(birth_year, birth_month, current_year, current_month):
    age = current_year - birth_year
    if current_month < birth_month: age -= 1
    return max(age, 1)


@cuda.jit(device=True)
def calculate_mortality_age_gpu(age, birth_month, current_month):
    month_diff = birth_month - current_month
    if month_diff <= 0: month_diff += 12
    return age + 1 if month_diff <= 6 else age


@cuda.jit(device=True)
def lookup_fees_gpu(tbl, shape, year, id_prod):
    if year < shape[0] and id_prod < shape[1]:
        idx = year * shape[1] + id_prod
        if idx < len(tbl):
            return tbl[idx]
    return 0.0


@cuda.jit(device=True)
def lookup_deposits_gpu(tbl, duree, id_depot, shape):
    idx = (duree - 1) * shape[1] + id_depot
    if idx * 4 < len(tbl):
        base = idx * 4
        return tbl[base], tbl[base + 1], tbl[base + 2], tbl[base + 3]
    return 0.0, 0, 999, 0


@cuda.jit(device=True)
def lookup_acquisition_gpu(tbl, duree, id_acqui, shape):
    idx = (duree - 1) * shape[1] + id_acqui
    if idx * 6 < len(tbl):
        base = idx * 6
        return tbl[base], tbl[base + 1], tbl[base + 2], tbl[base + 3], tbl[base + 4], tbl[base + 5]
    return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0


@cuda.jit(device=True)
def lookup_coussins_gpu(tbl, code_cat_prod, cat1, cat2, shape):
    idx = (code_cat_prod * shape[1] * shape[2] + cat1 * shape[2] + cat2) * 16
    if idx + 15 < len(tbl):
        return (tbl[idx], tbl[idx + 1], tbl[idx + 2], tbl[idx + 3], tbl[idx + 4], tbl[idx + 5], tbl[idx + 6],
                tbl[idx + 7],
                tbl[idx + 8], tbl[idx + 9], tbl[idx + 10], tbl[idx + 11], tbl[idx + 12], tbl[idx + 13], tbl[idx + 14],
                tbl[idx + 15])
    return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 1.0


@cuda.jit(device=True)
def lookup_mortality_gpu(tbl, i_sexe, age, year, prod, shape):
    age = min(age, 120)
    idx = (i_sexe * shape[1] * shape[2] * shape[3] + age * shape[2] * shape[3] + year * shape[3] + prod)
    if idx < len(tbl): return tbl[idx]
    return 0.001


@cuda.jit(device=True)
def lookup_returns_gpu(tbl, scn, year, month, shape):
    idx = (scn * shape[1] * shape[2] + year * shape[2] + month) * 7
    if idx + 6 < len(tbl):
        return tbl[idx], tbl[idx + 1], tbl[idx + 2], tbl[idx + 3], tbl[idx + 4], tbl[idx + 5], tbl[idx + 6]
    return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0


@cuda.jit(device=True)
def lookup_min_ferr_gpu(tbl, age):
    if age < len(tbl): return tbl[age]
    return 0.0


@cuda.jit(device=True)
def calculate_lapse_level_gpu(vm_vg_ratio):
    if vm_vg_ratio <= 0.5:
        return 1
    elif vm_vg_ratio <= 0.75:
        return 2
    else:
        return 3


@cuda.jit(device=True)
def calculate_lapse_rates_gpu(lapse_tot_tbl, lapse_part_tbl, duree, age, id_lapse, i_regime_2, vm_vg_ratio,
                              age_decaissement, tot_shape, part_shape):
    niv_tot = calculate_lapse_level_gpu(vm_vg_ratio)
    idx_tot = ((duree - 1) * tot_shape[1] * tot_shape[2] + id_lapse * tot_shape[2] + (niv_tot - 1)) * 3
    if idx_tot + 2 < len(lapse_tot_tbl):
        tx_min, tx_max, fact_dim = lapse_tot_tbl[idx_tot], lapse_tot_tbl[idx_tot + 1], lapse_tot_tbl[idx_tot + 2]
    else:
        tx_min, tx_max, fact_dim = 0.0, 0.0, 1.0
    if tx_min == tx_max:
        lapse_tot = tx_min
    else:
        if niv_tot == 1:
            interp = (vm_vg_ratio - 0.0) / 0.5 if vm_vg_ratio > 0 else 0
        elif niv_tot == 2:
            interp = (vm_vg_ratio - 0.5) / 0.25
        else:
            interp = (vm_vg_ratio - 0.75) / 999.24
        lapse_tot = interp * (tx_max - tx_min) + tx_min
    if age >= age_decaissement: lapse_tot *= fact_dim
    niv_part = calculate_lapse_level_gpu(vm_vg_ratio)
    idx_part = (age * part_shape[1] * part_shape[2] * part_shape[3] + id_lapse * part_shape[2] * part_shape[
        3] + i_regime_2 * part_shape[3] + (niv_part - 1)) * 2
    if idx_part + 1 < len(lapse_part_tbl):
        tx_p_min, tx_p_max = lapse_part_tbl[idx_part], lapse_part_tbl[idx_part + 1]
    else:
        tx_p_min, tx_p_max = 0.0, 0.0
    if tx_p_min == tx_p_max:
        lapse_part = tx_p_min
    else:
        if niv_part == 1:
            interp = (vm_vg_ratio - 0.0) / 0.5 if vm_vg_ratio > 0 else 0
        elif niv_part == 2:
            interp = (vm_vg_ratio - 0.5) / 0.25
        else:
            interp = (vm_vg_ratio - 0.75) / 999.24
        lapse_part = interp * (tx_p_max - tx_p_min) + tx_p_min
    return lapse_tot, lapse_part


@cuda.jit(device=True)
def calculate_coussins_gpu(c_tbl, c_shape, id_prod, mt_dex, mt_mm, mt_vm, mt_gar_ech, mt_gar_deces, mt_boni_deces,
                           mt_srg, pc_gar_ech, pc_gar_deces_1, mt_vm_av, duree, age, tx_survie):
    # Determine CODE_CAT_PRODUIT
    code_cat_prod = 7
    if id_prod == 22:
        code_cat_prod = 0
    elif 12 <= id_prod <= 16:
        code_cat_prod = 1
    elif 17 <= id_prod <= 21:
        code_cat_prod = 2
    elif id_prod == 6:
        code_cat_prod = 3
    elif id_prod == 4 or id_prod == 7:
        code_cat_prod = 4
    elif id_prod == 5 or id_prod == 8:
        code_cat_prod = 5
    elif id_prod == 2 or id_prod == 3:
        code_cat_prod = 6

    # Determine CAT_COUSSIN_1
    pct_rf = (mt_dex + mt_mm) / mt_vm if mt_vm > 0 else 0.0
    cat1 = 3
    if code_cat_prod == 0 or code_cat_prod == 6:
        cat1 = 0
    elif code_cat_prod == 7 and pct_rf < 0.5:
        cat1 = 4
    elif code_cat_prod == 7:
        cat1 = 5
    elif pct_rf < 1.0 / 3.0:
        cat1 = 1
    elif pct_rf < 2.0 / 3.0:
        cat1 = 2

    # Determine CAT_COUSSIN_2
    ratio_base = mt_gar_deces + mt_boni_deces
    r1 = pc_gar_ech / max(mt_gar_ech, 1e-2) if mt_gar_ech > 0 else 9999.
    r2 = pc_gar_deces_1 / max(ratio_base, 1e-2) if ratio_base > 0 else 9999.
    r3 = 1. / max(mt_srg, 1e-2) if mt_srg > 0 else 9999.
    vm_vg_ratio = min(10., (mt_vm + mt_vm_av) / 2. * min(min(r1, r2), r3))

    cat2 = 3
    if code_cat_prod == 7 and vm_vg_ratio < 0.7:
        cat2 = 4
    elif code_cat_prod == 7 and vm_vg_ratio < 0.9:
        cat2 = 5
    elif code_cat_prod == 7:
        cat2 = 6
    elif duree <= 3:
        cat2 = 1
    elif duree <= 6:
        cat2 = 2

    b_pr, t_pr, b_cr, t_cr, b_ma, t_ma, b_de, t_de, b_dc, t_dc, b_mo, t_mo, b_dp, t_dp, f80, f90 = lookup_coussins_gpu(
        c_tbl, code_cat_prod, cat1, cat2, c_shape)

    if code_cat_prod == 7 and mt_vm == 0:
        t_cr, t_ma, t_dc, t_dp = 0.0, 0.0, 0.0, 0.0

    age_factor = 1.0
    if age >= 90:
        age_factor = f90
    elif age >= 80:
        age_factor = f80

    max_guar = max(max(mt_gar_ech, ratio_base), mt_srg)

    p_red = t_pr * (max_guar if b_pr == 0 else mt_vm) * age_factor * tx_survie
    c_cred = t_cr * (max_guar if b_cr == 0 else mt_vm) * age_factor * tx_survie
    c_march = t_ma * (max_guar if b_ma == 0 else mt_vm) * age_factor * tx_survie
    c_dep = t_de * (max_guar if b_de == 0 else mt_vm) * age_factor * tx_survie
    c_dech = t_dc * (max_guar if b_dc == 0 else mt_vm) * age_factor * tx_survie
    c_mort = t_mo * (max_guar if b_mo == 0 else mt_vm) * age_factor * tx_survie
    c_depot = t_dp * (max_guar if b_dp == 0 else mt_vm) * age_factor * tx_survie

    return p_red, c_cred, c_march, c_dep, c_dech, c_mort, c_depot


# =============================================================================
# MAIN GPU KERNEL
# =============================================================================
# MODIFIED: Kernel signature updated to accept debug parameters
@cuda.jit
def project_account_scenario_kernel(
        accounts, n_accounts, n_scenarios, n_years, freq_eval,
        mortality_table, mortality_shape, returns_table, returns_shape, min_ferr_table,
        lapse_tot_table, lapse_tot_shape, lapse_part_table, lapse_part_shape,
        fees_table, fees_shape,
        deposits_table, deposits_shape, acquisition_table, acquisition_shape, coussins_table, coussins_shape,
        output_cashflows, output_pvs, output_vm,
        # NEW: Debugging arguments
        debug_output, debug_account_id, debug_scenario_id
):
    thread_id = cuda.grid(1)
    if thread_id >= n_accounts * n_scenarios: return

    account_idx, scenario_idx = thread_id // n_scenarios, thread_id % n_scenarios

    # NEW: Check if this thread should write debug info
    is_debug_thread = (accounts[account_idx, 0] == debug_account_id and scenario_idx + 1 == debug_scenario_id)

    # --- ACCOUNT DATA INITIALIZATION (Unchanged) ---
    annee_eval_ini, mois_eval_ini = int32(accounts[account_idx, 1]), int32(accounts[account_idx, 2])
    annee_nais, mois_nais = int32(accounts[account_idx, 3]), int32(accounts[account_idx, 4])
    i_sexe, i_prod_regr, id_prod = int32(accounts[account_idx, 5]), int32(accounts[account_idx, 6]), int32(
        accounts[account_idx, 7])
    id_lapse, i_regime_2, id_depot = int32(accounts[account_idx, 8]), int32(accounts[account_idx, 9]), int32(
        accounts[account_idx, 10])
    id_acqui, age_fin_contrat = int32(accounts[account_idx, 11]), int32(accounts[account_idx, 12])
    age_decaissement = int32(accounts[account_idx, 13])
    var_retrait_fct, mt_tpa_retrait, pc_retrait_age = int32(accounts[account_idx, 30]), accounts[account_idx, 31], \
        accounts[account_idx, 32]
    nb_an_ech, age_ech_min = int32(accounts[account_idx, 33]), int32(accounts[account_idx, 34])
    mt_vm = accounts[account_idx, 14]
    mt_gar_deces, mt_gar_ech, mt_srg = accounts[account_idx, 15], accounts[account_idx, 16], accounts[account_idx, 17]
    mt_dex, mt_mm, mt_tsx, mt_sp500, mt_eafe = accounts[account_idx, 18], accounts[account_idx, 19], accounts[
        account_idx, 20], accounts[account_idx, 21], accounts[account_idx, 22]
    pc_hon_gest, pc_frais_gar = accounts[account_idx, 23], accounts[account_idx, 24]
    pc_gar_deces_1, pc_gar_ech, pc_rfg = accounts[account_idx, 25], accounts[account_idx, 26], accounts[account_idx, 27]
    mt_boni_deces, pc_boni_deces = accounts[account_idx, 28], accounts[account_idx, 29]
    mt_gar_deces_orig, pc_revenu_fds, mt_rf = accounts[account_idx, 15], accounts[account_idx, 35], accounts[
        account_idx, 36]
    mt_vm_orig = mt_vm

    # --- STATE VARIABLES (Unchanged) ---
    tx_survie, tx_actualisation = 1.0, 1.0;
    mt_min_ferr_proj, mt_mrv_proj = 0.0, 0.0
    annee_ech_proj, mois_ech_proj = float32(annee_eval_ini + nb_an_ech), float32(mois_eval_ini)
    time_idx = 0
    f_eval = float32(freq_eval)

    # --- MAIN PROJECTION LOOP ---
    for an_eval in range(n_years + 1):
        for mois_simul in range(1, freq_eval + 1):
            annee_reelle, mois_eval = annee_eval_ini + an_eval - 1, mois_simul * 12 // freq_eval
            age = calculate_age_gpu(annee_nais, mois_nais, annee_reelle, mois_eval)

            keep = (age <= age_fin_contrat and
                    (an_eval > 1 or
                     (an_eval == 1 and mois_eval >= mois_eval_ini) or
                     (an_eval == 0 and mois_eval == 12)))
            if not keep:
                continue

            if tx_survie <= 1e-4 or (mt_vm <= 0 and i_prod_regr == 0): break

            duree_max10 = min(int32(annee_reelle + mois_eval / 12. - (annee_eval_ini + mois_eval_ini / 12.)) + 1, 10)
            tx_survie_deb = tx_survie

            fwd, _, r_dex, r_mm, r_tsx, r_sp500, r_eafe = lookup_returns_gpu(returns_table, scenario_idx + 1, an_eval,
                                                                             mois_eval, returns_shape)
            tx_actualisation *= math.exp(-fwd)
            mt_dex *= math.exp(r_dex);
            mt_mm *= math.exp(r_mm);
            mt_tsx *= math.exp(r_tsx);
            mt_sp500 *= math.exp(r_sp500);
            mt_eafe *= math.exp(r_eafe)
            mt_vm_av_frais = mt_dex + mt_mm + mt_tsx + mt_sp500 + mt_eafe

            age_mort = calculate_mortality_age_gpu(age, mois_nais, mois_eval)
            qx = 1. - math.pow(1. - lookup_mortality_gpu(mortality_table, i_sexe, age_mort, annee_reelle, i_prod_regr,
                                                         mortality_shape), 1. / f_eval)

            ratio_base = mt_gar_deces + mt_boni_deces;
            r1 = pc_gar_ech / max(mt_gar_ech, 1e-2) if mt_gar_ech > 0 else 9999.;
            r2 = pc_gar_deces_1 / max(ratio_base, 1e-2) if ratio_base > 0 else 9999.;
            r3 = 1. / max(mt_srg, 1e-2) if mt_srg > 0 else 9999.
            vm_vg_ratio = min(10., (mt_vm + mt_vm_av_frais) / 2. * min(min(r1, r2), r3))
            lapse_t, lapse_p = calculate_lapse_rates_gpu(lapse_tot_table, lapse_part_table, duree_max10, age, id_lapse,
                                                         i_regime_2, vm_vg_ratio, age_decaissement, lapse_tot_shape,
                                                         lapse_part_shape)
            lapse = 1. - math.pow(1. - lapse_t - lapse_p, 1. / f_eval)

            tx_survie *= (1. - qx) * (1. - lapse)

            if pc_boni_deces > 0: mt_boni_deces += mt_gar_deces * pc_boni_deces / f_eval

            mt_vm_av_retrait = mt_vm_av_frais * math.exp(-pc_rfg / f_eval)
            guarantee_fee_amount = min(mt_vm_av_retrait * pc_frais_gar / f_eval, mt_vm_av_retrait)
            primes_garanties = guarantee_fee_amount * tx_survie_deb
            mt_vm_av_retrait = max(mt_vm_av_retrait - guarantee_fee_amount, 0.0)

            if mois_eval == 12 // freq_eval: mt_min_ferr_proj = mt_vm * lookup_min_ferr_gpu(min_ferr_table, age)
            retrait = 0.0
            if age + 1 >= age_decaissement and mt_vm_av_retrait > 0:
                if var_retrait_fct == 1:
                    retrait = mt_tpa_retrait if mt_tpa_retrait > 0 else mt_vm_av_retrait * pc_retrait_age
                elif var_retrait_fct == 2:
                    retrait = max(mt_tpa_retrait, mt_min_ferr_proj * max(pc_retrait_age, 1.0))
                elif var_retrait_fct == 3:
                    retrait = max(mt_min_ferr_proj, mt_mrv_proj) * pc_retrait_age
                retrait /= f_eval

            prest_mrv = -max(retrait - mt_vm_av_retrait, 0) * tx_survie_deb if i_prod_regr == 1 else 0.0

            mt_vm_ap_retrait = max(mt_vm_av_retrait - retrait, 0.)
            if mt_vm_av_retrait <= retrait:
                mt_gar_ech, mt_gar_deces, mt_boni_deces, mt_srg = 0., 0., 0., 0.
            else:
                prop = 1. - retrait / mt_vm_av_retrait
                mt_gar_ech *= prop;
                mt_gar_deces *= prop;
                mt_boni_deces *= prop
                mt_srg = max(mt_srg - retrait, 0.)

            pc_depot, var_depot, age_max_depot, i_even_cesse = lookup_deposits_gpu(deposits_table, duree_max10,
                                                                                   id_depot, deposits_shape)
            depot_futur = 0.0
            if pc_depot > 0 and age < age_max_depot and not (i_even_cesse == 1 and age + 1 >= age_decaissement):
                base = mt_gar_deces_orig / pc_gar_deces_1 if (var_depot != 1 and pc_gar_deces_1 > 0) else mt_vm
                depot_futur = (base * pc_depot) / f_eval

            mt_vm = mt_vm_ap_retrait + depot_futur

            if depot_futur > 0:
                mt_gar_deces += depot_futur
                mt_gar_ech += depot_futur
                if mt_srg > 0: mt_srg += depot_futur

            prest_deces = qx * -max(0., (mt_gar_deces + mt_boni_deces) - mt_vm) * tx_survie_deb

            prest_ech = 0.0
            if (annee_reelle == annee_ech_proj and mois_eval == mois_ech_proj) or (
                    age == age_fin_contrat and mois_eval == (
                    mois_nais - 12 / f_eval if mois_nais > 12 / f_eval else 12)):
                prest_ech = -max(0., mt_gar_ech - mt_vm) * tx_survie
                if tx_survie > 0: mt_vm += abs(prest_ech / tx_survie)
                mt_gar_ech = mt_vm * pc_gar_ech
                annee_ech_proj += nb_an_ech
                if annee_ech_proj > annee_nais + age_ech_min:
                    mois_ech_proj = mois_eval
                else:
                    annee_ech_proj = annee_nais + age_ech_min;
                    mois_ech_proj = mois_nais

            pc_v_rf, pc_v_ac, pc_m_rf, pc_m_ac, pc_f_ac, pc_f_rf = lookup_acquisition_gpu(acquisition_table,
                                                                                          duree_max10, id_acqui,
                                                                                          acquisition_shape)
            mt_rf_current = mt_rf * (mt_vm / mt_vm_orig) if mt_vm_orig > 0 else 0

            pc_comm_vente = (
                    pc_v_ac * (mt_vm - mt_rf_current) / mt_vm + pc_v_rf * mt_rf_current / mt_vm) if mt_vm > 0 else 0
            pc_comm_maintien = (
                    pc_m_ac * (mt_vm - mt_rf_current) / mt_vm + pc_m_rf * mt_rf_current / mt_vm) if mt_vm > 0 else 0
            pc_frais_an = (
                    pc_f_ac * (mt_vm - mt_rf_current) / mt_vm + pc_f_rf * mt_rf_current / mt_vm) if mt_vm > 0 else 0

            comm_vente = -pc_comm_vente * depot_futur * tx_survie_deb
            hon_gest = -mt_vm_av_frais * (math.exp(pc_hon_gest / f_eval) - 1) * tx_survie_deb
            comm_maintien = -mt_vm_av_frais * (math.exp(pc_comm_maintien / f_eval) - 1) * tx_survie_deb
            primes_variables = mt_vm_av_frais * math.exp(-(pc_rfg - pc_revenu_fds) / f_eval) * -(
                    math.exp(-pc_revenu_fds / f_eval) - 1) * tx_survie_deb

            frais_fixes = -lookup_fees_gpu(fees_table, fees_shape, annee_reelle,
                                           id_prod) / f_eval * tx_survie_deb
            frais_acquis = pc_frais_an * mt_vm_ap_retrait * lapse * tx_survie_deb * (1 - qx)

            p_red, c_cred, c_march, c_dep, c_dech, c_mort, c_depot = calculate_coussins_gpu(coussins_table,
                                                                                            coussins_shape, id_prod,
                                                                                            mt_dex, mt_mm, mt_vm,
                                                                                            mt_gar_ech, mt_gar_deces,
                                                                                            mt_boni_deces, mt_srg,
                                                                                            pc_gar_ech, pc_gar_deces_1,
                                                                                            mt_vm_av_frais, duree_max10,
                                                                                            age, tx_survie)
            valeur_marchande = mt_vm * tx_survie

            if mt_vm_orig > 0 and mt_vm > 0:
                ratio = mt_vm / mt_vm_orig
                mt_dex, mt_mm, mt_tsx, mt_sp500, mt_eafe = accounts[account_idx, 18] * ratio, accounts[
                    account_idx, 19] * ratio, accounts[account_idx, 20] * ratio, accounts[account_idx, 21] * ratio, \
                                                           accounts[account_idx, 22] * ratio
            elif mt_vm <= 0:
                mt_dex, mt_mm, mt_tsx, mt_sp500, mt_eafe = 0, 0, 0, 0, 0

            cf = cuda.local.array(18, dtype=float32)
            cf[0] = primes_garanties;
            cf[1] = prest_deces;
            cf[2] = prest_ech;
            cf[3] = prest_mrv;
            cf[4] = frais_acquis;
            cf[5] = comm_vente
            cf[6] = primes_variables;
            cf[7] = frais_fixes;
            cf[8] = hon_gest;
            cf[9] = comm_maintien;
            cf[10] = p_red;
            cf[11] = c_cred
            cf[12] = c_march;
            cf[13] = c_dep;
            cf[14] = c_dech;
            cf[15] = c_mort;
            cf[16] = c_depot;
            cf[17] = valeur_marchande

            for i in range(18): output_cashflows[thread_id, time_idx, i] = cf[i]
            for i in range(18): output_pvs[thread_id, time_idx, i] = cf[i] * tx_actualisation

            for i in range(10, 18):
                output_pvs[thread_id, time_idx, i] /= f_eval

            output_vm[thread_id, time_idx] = mt_vm

            # NEW: If this is the debug thread, write all state variables to the debug output array
            if is_debug_thread:
                if time_idx < debug_output.shape[0]:
                    debug_output[time_idx, 0] = an_eval
                    debug_output[time_idx, 1] = mois_eval
                    debug_output[time_idx, 2] = age
                    debug_output[time_idx, 3] = mt_vm
                    debug_output[time_idx, 4] = mt_gar_deces
                    debug_output[time_idx, 5] = mt_gar_ech
                    debug_output[time_idx, 6] = mt_srg
                    debug_output[time_idx, 7] = tx_survie
                    debug_output[time_idx, 8] = tx_actualisation
                    debug_output[time_idx, 9] = qx
                    debug_output[time_idx, 10] = lapse
                    debug_output[time_idx, 11] = depot_futur
                    debug_output[time_idx, 12] = retrait
                    debug_output[time_idx, 13] = vm_vg_ratio

            time_idx += 1
            if time_idx >= output_cashflows.shape[1]: break

        if tx_survie <= 1e-4 or (mt_vm <= 0 and i_prod_regr == 0): break
        if time_idx >= output_cashflows.shape[1]: break


# =============================================================================
# DATA PREPARATION & EXECUTION
# =============================================================================
def prepare_gpu_data(data_path: Path, config: Dict) -> Dict:
    """
    Loads, cleans, and transforms all necessary data into flattened NumPy arrays
    suitable for GPU transfer.
    """
    print("Loading and preparing all data for GPU...")
    table_names = [
        'population', 'mortalite', 'rendements', 'min_ferr', 'tx_lapse_part',
        'tx_lapse_tot', 'frais_admin', 'depots_futurs', 'acquisition', 'coussins_escap'
    ]
    dfs = {name: pd.read_csv(data_path.joinpath(f"{name.upper()}.csv"), sep=';', encoding='utf-8') for name in
           table_names}
    for df in dfs.values(): df.columns = df.columns.str.upper()
    clean_numeric(dfs['population'], [c for c in dfs['population'].columns if c.startswith(('PC_', 'TAUX_'))])
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
    clean_numeric(dfs['acquisition'], ['PC_COMMISSION_VENTE_RF', 'PC_COMMISSION_VENTE_AC', 'PC_COMMISSION_MAINTIEN_RF',
                                       'PC_COMMISSION_MAINTIEN_AC', 'PC_FRAIS_AN_AC', 'PC_FRAIS_AN_RF'])
    clean_numeric(dfs['coussins_escap'],
                  [c for c in dfs['coussins_escap'].columns if c.startswith(('TX_', 'FACTEUR_'))])
    dfs['population'] = dfs['population'][dfs['population']['ID_COMPTE'] <= config['NBCPT']].copy()
    dfs['rendements'] = dfs['rendements'][(dfs['rendements']['SCN_EVAL'] <= config['NB_SC']) & (
            dfs['rendements']['AN_EVAL'] <= config['NB_AN_PROJECTION'])].copy()
    data_out = {}
    account_cols = [
        'ID_COMPTE', 'ANNEE_EVALUATION_INI', 'MOIS_EVALUATION_INI', 'ANNEE_NAIS',
        'MOIS_NAIS', 'I_SEXE', 'I_PRODUIT_REGR', 'ID_PRODUIT', 'ID_LAPSE',
        'I_REGIME_2', 'ID_DEPOT', 'ID_ACQUI', 'AGE_FIN_CONTRAT', 'AGE_DECAISSEMENT',
        'MT_VM', 'MT_GAR_DECES', 'MT_GAR_ECH', 'MT_SRG', 'MT_DEX', 'MT_MM',
        'MT_TSX', 'MT_SP500', 'MT_EAFE', 'PC_HONORAIRES_GEST', 'PC_FRAIS_GARANTIE',
        'PC_GAR_DECES_1', 'PC_GAR_ECH', 'PC_RFG', 'MT_BONI_DECES', 'PC_BONI_DECES',
        'VAR_RETRAIT_FCT', 'MT_TPA_RETRAIT', 'PC_RETRAIT_AGE', 'NB_AN_ECH',
        'AGE_ECH_MIN', 'PC_REVENU_FDS', 'MT_RF'
    ]
    for col in account_cols:
        if col not in dfs['population'].columns: dfs['population'][col] = 0.0
    data_out['accounts'] = np.ascontiguousarray(dfs['population'][account_cols].values.astype(np.float32))
    df = dfs['mortalite']
    shape = (df['I_SEXE'].max() + 1, 121, df['ANNEE_REELLE'].max() + 1, df['I_PRODUIT_REGR'].max() + 1)
    flat_array = np.zeros(np.prod(shape), dtype=np.float32)
    for _, r in df.iterrows():
        idx = (int(r['I_SEXE']) * shape[1] * shape[2] * shape[3] +
               int(r['AGE_MORTALITE']) * shape[2] * shape[3] +
               int(r['ANNEE_REELLE']) * shape[3] +
               int(r['I_PRODUIT_REGR']))
        if idx < len(flat_array): flat_array[idx] = r['QX']
    data_out.update({'mortality_table': flat_array, 'mortality_shape': np.array(shape, dtype=np.int32)})
    df = dfs['rendements']
    cols = ['FORWARD_RATE', 'AJUST_FORWARD_RATE_VM_0', 'RENDDEX_AN', 'RENDMM_AN', 'RENDTSX_AN', 'RENDSP500_AN',
            'RENDEAFE_AN']
    num_cols = len(cols)
    shape = (df['SCN_EVAL'].max() + 1, df['AN_EVAL'].max() + 1, df['MOIS_EVAL'].max() + 1)
    flat_array = np.zeros(np.prod(shape) * num_cols, dtype=np.float32)
    for _, r in df.iterrows():
        base_idx = (int(r['SCN_EVAL']) * shape[1] * shape[2] + int(r['AN_EVAL']) * shape[2] + int(
            r['MOIS_EVAL'])) * num_cols
        if base_idx + num_cols - 1 < len(flat_array): flat_array[base_idx: base_idx + num_cols] = r[cols].values
    data_out.update({'returns_table': flat_array, 'returns_shape': np.array(shape, dtype=np.int32)})
    flat_array = np.zeros(121, dtype=np.float32)
    for _, r in dfs['min_ferr'].iterrows():
        if int(r['AGE']) < 121: flat_array[int(r['AGE'])] = r['MIN_FERR']
    data_out['min_ferr_table'] = flat_array
    df_tot = dfs['tx_lapse_tot']
    cols_tot = ['TX_LAPSE_TOT_MIN', 'TX_LAPSE_TOT_MAX', 'FACT_DIM']
    num_cols_tot = len(cols_tot)
    shape_tot = (10, df_tot['ID_LAPSE'].max() + 1, 3)
    flat_array_tot = np.zeros(np.prod(shape_tot) * num_cols_tot, dtype=np.float32)
    for _, r in df_tot.iterrows():
        base_idx = ((int(r['DUREE_MAX10']) - 1) * shape_tot[1] * shape_tot[2] + int(r['ID_LAPSE']) * shape_tot[2] + (
                int(r['LAPSE_NIV_TOT']) - 1)) * num_cols_tot
        if base_idx + num_cols_tot - 1 < len(flat_array_tot): flat_array_tot[base_idx: base_idx + num_cols_tot] = r[
            cols_tot].values
    df_part = dfs['tx_lapse_part']
    cols_part = ['TX_LAPSE_PART_MIN', 'TX_LAPSE_PART_MAX']
    num_cols_part = len(cols_part)
    shape_part = (121, df_part['ID_LAPSE'].max() + 1, df_part['I_REGIME_2'].max() + 1, 3)
    flat_array_part = np.zeros(np.prod(shape_part) * num_cols_part, dtype=np.float32)
    for _, r in df_part.iterrows():
        base_idx = (int(r['AGE']) * shape_part[1] * shape_part[2] * shape_part[3] + int(r['ID_LAPSE']) * shape_part[2] *
                    shape_part[3] + int(r['I_REGIME_2']) * shape_part[3] + (
                            int(r['LAPSE_NIV_PART']) - 1)) * num_cols_part
        if base_idx + num_cols_part - 1 < len(flat_array_part): flat_array_part[base_idx: base_idx + num_cols_part] = r[
            cols_part].values
    data_out.update({
        'lapse_tot_table': flat_array_tot, 'lapse_tot_shape': np.array(shape_tot, dtype=np.int32),
        'lapse_part_table': flat_array_part, 'lapse_part_shape': np.array(shape_part, dtype=np.int32)
    })
    df = dfs['frais_admin']
    shape = (df['ANNEE_REELLE'].max() + 1, df['ID_PRODUIT'].max() + 1)
    fees_2d_array = np.zeros(shape, dtype=np.float32)
    for _, r in df.iterrows():
        year, prod_id = int(r['ANNEE_REELLE']), int(r['ID_PRODUIT'])
        if year < shape[0] and prod_id < shape[1]: fees_2d_array[year, prod_id] = r['FRAIS']
    flat_array = np.ascontiguousarray(fees_2d_array.flatten())
    data_out.update({'fees_table': flat_array, 'fees_shape': np.array(shape, dtype=np.int32)})
    df = dfs['depots_futurs']
    cols = ['PC_DEPOT_ANNUEL', 'VAR_DEPOT_FCT', 'AGE_MAX_DEPOT', 'I_EVEN_CESSE_DEPOT']
    num_cols = len(cols)
    shape = (10, df['ID_DEPOT'].max() + 1)
    flat_array = np.zeros(np.prod(shape) * num_cols, dtype=np.float32)
    for _, r in df.iterrows():
        base_idx = ((int(r['DUREE_MAX10']) - 1) * shape[1] + int(r['ID_DEPOT'])) * num_cols
        if base_idx + num_cols - 1 < len(flat_array): flat_array[base_idx: base_idx + num_cols] = r[cols].values
    data_out.update({'deposits_table': flat_array, 'deposits_shape': np.array(shape, dtype=np.int32)})
    df = dfs['acquisition']
    cols = ['PC_COMMISSION_VENTE_RF', 'PC_COMMISSION_VENTE_AC', 'PC_COMMISSION_MAINTIEN_RF',
            'PC_COMMISSION_MAINTIEN_AC', 'PC_FRAIS_AN_AC', 'PC_FRAIS_AN_RF']
    num_cols = len(cols)
    shape = (10, df['ID_ACQUI'].max() + 1)
    flat_array = np.zeros(np.prod(shape) * num_cols, dtype=np.float32)
    for _, r in df.iterrows():
        base_idx = ((int(r['DUREE_MAX10']) - 1) * shape[1] + int(r['ID_ACQUI'])) * num_cols
        if base_idx + num_cols - 1 < len(flat_array): flat_array[base_idx: base_idx + num_cols] = r[cols].values
    data_out.update({'acquisition_table': flat_array, 'acquisition_shape': np.array(shape, dtype=np.int32)})
    df = dfs['coussins_escap']
    cols = [
        'BASE_PASSIF_REDRESSE', 'TX_PASSIF_REDRESSE', 'BASE_COUSSIN_CREDIT', 'TX_COUSSIN_CREDIT',
        'BASE_COUSSIN_MARCHE', 'TX_COUSSIN_MARCHE', 'BASE_COUSSIN_DEPENSE', 'TX_COUSSIN_DEPENSE',
        'BASE_COUSSIN_DECHEANCE', 'TX_COUSSIN_DECHEANCE', 'BASE_COUSSIN_MORTALITE', 'TX_COUSSIN_MORTALITE',
        'BASE_COUSSIN_DEPOT', 'TX_COUSSIN_DEPOT', 'FACTEUR_AGE_80', 'FACTEUR_AGE_90'
    ]
    num_cols = len(cols)
    shape = (df['CODE_CAT_PRODUIT'].max() + 1, df['CAT_COUSSIN_1'].max() + 1, df['CAT_COUSSIN_2'].max() + 1)
    flat_array = np.zeros(np.prod(shape) * num_cols, dtype=np.float32)
    for _, r in df.iterrows():
        base_idx = (int(r['CODE_CAT_PRODUIT']) * shape[1] * shape[2] + int(r['CAT_COUSSIN_1']) * shape[2] + int(
            r['CAT_COUSSIN_2'])) * num_cols
        if base_idx + num_cols - 1 < len(flat_array): flat_array[base_idx: base_idx + num_cols] = r[cols].values
    data_out.update({'coussins_table': flat_array, 'coussins_shape': np.array(shape, dtype=np.int32)})
    print("✓ Data preparation complete.")
    return data_out


# MODIFIED: Function signature updated to accept new arguments
def run_gpu_projection(data_path: Path, output_path: Path, max_accounts: int = None, nb_scenarios: int = 100,
                       nb_years: int = 100, debug_account_id: int = -1, debug_scenario_id: int = -1,
                       start_year_out: int = None, end_year_out: int = None):
    start_time = datetime.now()
    print("=" * 80 + "\nGPU ACTUARIAL PROJECTION\n" + "=" * 80)
    config = create_config(nb_scenarios=nb_scenarios, nb_years=nb_years, max_accounts=max_accounts)
    gpu_data = prepare_gpu_data(data_path, config)

    n_accounts, n_scenarios, n_years, freq_eval = len(gpu_data['accounts']), config['NB_SC'], config[
        'NB_AN_PROJECTION'], config['FREQ_EVAL']
    max_timesteps, total_paths = (n_years + 1) * freq_eval, n_accounts * n_scenarios
    print(
        f"\nConfiguration: {n_accounts} accounts, {n_scenarios} scenarios, {n_years} years -> {total_paths:,} total paths")

    # --- Output Array Initialization ---
    output_cashflows = np.zeros((total_paths, max_timesteps, 18), dtype=np.float32)
    output_pvs = np.zeros((total_paths, max_timesteps, 18), dtype=np.float32)
    output_vm = np.zeros((total_paths, max_timesteps), dtype=np.float32)

    # NEW: Initialize debug array if needed
    debug_output = np.zeros((max_timesteps, 14), dtype=np.float32) if debug_account_id > 0 else None

    print("\nTransferring data to GPU...")
    d_data = {k: cuda.to_device(v) for k, v in gpu_data.items()}
    d_output_cashflows, d_output_pvs, d_output_vm = cuda.to_device(output_cashflows), cuda.to_device(
        output_pvs), cuda.to_device(output_vm)
    d_debug_output = cuda.to_device(debug_output) if debug_output is not None else cuda.device_array((0, 0))

    threads_per_block = config['THREADS_PER_BLOCK']
    blocks_per_grid = (total_paths + threads_per_block - 1) // threads_per_block
    print(f"\nLaunching GPU kernel ({blocks_per_grid} blocks, {threads_per_block} threads)...")

    kernel_start = datetime.now()
    # MODIFIED: Kernel call updated with new debug arguments
    project_account_scenario_kernel[blocks_per_grid, threads_per_block](
        d_data['accounts'], n_accounts, n_scenarios, n_years, freq_eval,
        d_data['mortality_table'], d_data['mortality_shape'],
        d_data['returns_table'], d_data['returns_shape'], d_data['min_ferr_table'],
        d_data['lapse_tot_table'], d_data['lapse_tot_shape'],
        d_data['lapse_part_table'], d_data['lapse_part_shape'],
        d_data['fees_table'], d_data['fees_shape'],
        d_data['deposits_table'], d_data['deposits_shape'],
        d_data['acquisition_table'], d_data['acquisition_shape'],
        d_data['coussins_table'], d_data['coussins_shape'],
        d_output_cashflows, d_output_pvs, d_output_vm,
        d_debug_output, float32(debug_account_id), int32(debug_scenario_id)  # NEW
    )
    cuda.synchronize()
    kernel_end = datetime.now()
    print(f"  ✓ Kernel execution time: {(kernel_end - kernel_start).total_seconds():.2f} seconds")

    print("\nTransferring results from GPU and aggregating...")
    cashflows_host = d_output_cashflows.copy_to_host()
    pvs_avg = d_output_pvs.copy_to_host().reshape(n_accounts, n_scenarios, max_timesteps, 18).mean(axis=1)

    # --- AGGREGATION FOR VP FILES (UNCHANGED) ---
    vp_flux_compte = pd.DataFrame({
        'ID_COMPTE': gpu_data['accounts'][:, 0].astype(int), 'VP_PRIMES_GARANTIES': pvs_avg[:, :, 0].sum(axis=1),
        'VP_PREST_DECES': pvs_avg[:, :, 1].sum(axis=1), 'VP_PREST_ECH': pvs_avg[:, :, 2].sum(axis=1),
        'VP_PREST_MRV': pvs_avg[:, :, 3].sum(axis=1), 'VP_FRAIS_ACQUIS': pvs_avg[:, :, 4].sum(axis=1),
        'VP_COMM_VENTE': pvs_avg[:, :, 5].sum(axis=1), 'VP_PRIMES_VARIABLES': pvs_avg[:, :, 6].sum(axis=1),
        'VP_FRAIS_FIXES': pvs_avg[:, :, 7].sum(axis=1), 'VP_HON_GEST': pvs_avg[:, :, 8].sum(axis=1),
        'VP_COMM_MAINTIEN': pvs_avg[:, :, 9].sum(axis=1), 'VP_PASSIF_REDRESSE': pvs_avg[:, :, 10].sum(axis=1),
        'VP_COUSSIN_CREDIT': pvs_avg[:, :, 11].sum(axis=1), 'VP_COUSSIN_MARCHE': pvs_avg[:, :, 12].sum(axis=1),
        'VP_COUSSIN_DEPENSE': pvs_avg[:, :, 13].sum(axis=1), 'VP_COUSSIN_DECHEANCE': pvs_avg[:, :, 14].sum(axis=1),
        'VP_COUSSIN_MORTALITE': pvs_avg[:, :, 15].sum(axis=1), 'VP_COUSSIN_DEPOT': pvs_avg[:, :, 16].sum(axis=1),
        'VP_VALEUR_MARCHANDE': pvs_avg[:, :, 17].sum(axis=1),
    })
    vp_cols_to_sum = [c for c in vp_flux_compte.columns if c.startswith('VP_') and c != 'VP_VALEUR_MARCHANDE']
    vp_flux_total = pd.DataFrame({'CATEGORIE': ['TOTAL'], 'VP_FLUX_TOT': [vp_flux_compte[vp_cols_to_sum].sum().sum()]})

    # --- NEW: AGGREGATION FOR FLUX_PROJETES ---
    print("  - Aggregating flux projetes...")
    cashflows_avg_acct = cashflows_host.reshape(n_accounts, n_scenarios, max_timesteps, 18).mean(axis=1)
    flux_projetes_agg = cashflows_avg_acct.sum(axis=0)

    # Create time index columns (AN_EVAL, MOIS_EVAL)
    timesteps = np.arange(max_timesteps)
    an_eval = timesteps // freq_eval
    mois_simul = (timesteps % freq_eval) + 1
    mois_eval = mois_simul * 12 // freq_eval

    cf_cols = ['PRIMES_GARANTIES', 'PREST_DECES', 'PREST_ECH', 'PREST_MRV', 'FRAIS_ACQUIS', 'COMM_VENTE',
               'PRIMES_VARIABLES', 'FRAIS_FIXES', 'HON_GEST', 'COMM_MAINTIEN', 'PASSIF_REDRESSE', 'COUSSIN_CREDIT',
               'COUSSIN_MARCHE', 'COUSSIN_DEPENSE', 'COUSSIN_DECHEANCE', 'COUSSIN_MORTALITE', 'COUSSIN_DEPOT',
               'VALEUR_MARCHANDE']

    flux_projetes_df = pd.DataFrame(flux_projetes_agg, columns=cf_cols)
    flux_projetes_df.insert(0, 'AN_EVAL', an_eval)
    flux_projetes_df.insert(1, 'MOIS_EVAL', mois_eval)

    # Filter out empty future periods
    flux_projetes_df = flux_projetes_df[flux_projetes_df[cf_cols].abs().sum(axis=1) > 1e-9]

    # NEW: Apply year filtering
    if start_year_out is not None:
        flux_projetes_df = flux_projetes_df[flux_projetes_df['AN_EVAL'] >= start_year_out]
    if end_year_out is not None:
        flux_projetes_df = flux_projetes_df[flux_projetes_df['AN_EVAL'] <= end_year_out]

    print("\nSaving outputs...");
    output_path.mkdir(parents=True, exist_ok=True)
    vp_flux_compte.to_csv(output_path / "VP_FLUX_COMPTE_GPU.csv", index=False, sep=';')
    vp_flux_total.to_csv(output_path / "VP_FLUX_TOTAL_GPU.csv", index=False, sep=';')
    flux_projetes_df.to_csv(output_path / "FLUX_PROJETES_GPU.csv", index=False, sep=';')  # NEW
    print(f"  ✓ Saved outputs to {output_path}")

    # NEW: Save debug file if it was generated
    if debug_output is not None:
        debug_host = d_debug_output.copy_to_host()
        debug_cols = ['AN_EVAL', 'MOIS_EVAL', 'AGE', 'MT_VM', 'MT_GAR_DECES', 'MT_GAR_ECH', 'MT_SRG',
                      'TX_SURVIE', 'TX_ACTUALISATION', 'QX', 'LAPSE', 'DEPOT_FUTUR', 'RETRAIT', 'VM_VG_RATIO']
        debug_df = pd.DataFrame(debug_host, columns=debug_cols)
        debug_df = debug_df[debug_df['AN_EVAL'] > 0]  # Filter out padding
        debug_df.to_csv(output_path / "TEST_GPU.csv", index=False, sep=';')
        print(f"  ✓ Saved debug trace to {output_path / 'TEST_GPU.csv'}")

    total_time, kernel_time = (datetime.now() - start_time).total_seconds(), (kernel_end - kernel_start).total_seconds()
    print("\n" + "=" * 80 + "\nPROJECTION COMPLETE\n" + "=" * 80)
    print(
        f"Total time: {total_time:.2f}s | Kernel time: {kernel_time:.2f}s | Overhead: {total_time - kernel_time:.2f}s")
    if kernel_time > 0: print(f"Paths per second (kernel): {total_paths / kernel_time:,.0f}")
    print(f"Total PV: ${vp_flux_total['VP_FLUX_TOT'].iloc[0]:,.2f}")
    print("=" * 80)

    return {'vp_flux_compte': vp_flux_compte, 'vp_flux_total': vp_flux_total, 'flux_projetes': flux_projetes_df}


def main():
    # MODIFIED: Updated argument parser
    parser = argparse.ArgumentParser(description='GPU Actuarial Projection')
    parser.add_argument('--accounts', type=int, default=3, help='Number of accounts to process.')
    parser.add_argument('--scenarios', type=int, default=100, help='Number of economic scenarios.')
    parser.add_argument('--years', type=int, default=100, help='Number of projection years.')
    parser.add_argument('--data-path', type=str, default=None, help='Path to input data directory.')
    parser.add_argument('--output-path', type=str, default=None, help='Path for output files.')
    # NEW: Debug and filtering arguments
    parser.add_argument('--debug-account', type=int, default=1, help='Account ID to trace for debugging.')
    parser.add_argument('--debug-scenario', type=int, default=2, help='Scenario ID to trace for debugging.')
    parser.add_argument('--start-year', type=int, default=1, help='Start year for FLUX_PROJETES output.')
    parser.add_argument('--end-year', type=int, default=10, help='End year for FLUX_PROJETES output.')

    args = parser.parse_args()

    DATA_PATH = Path(args.data_path) if args.data_path else HERE.joinpath("algo2/data_in")
    OUTPUT_PATH = Path(args.output_path) if args.output_path else HERE.joinpath("algo2/data_out")

    print(
        f"\nRunning GPU projection with:\n  Accounts: {args.accounts or 'ALL'}, Scenarios: {args.scenarios}, Years: {args.years}")
    if args.debug_account and args.debug_scenario:
        print(f"  DEBUG MODE ON for Account: {args.debug_account}, Scenario: {args.debug_scenario}")
    if args.start_year or args.end_year:
        print(f"  Output filtering for years: {args.start_year or 'start'} to {args.end_year or 'end'}")

    # MODIFIED: Pass all arguments to the main function
    results = run_gpu_projection(
        data_path=DATA_PATH,
        output_path=OUTPUT_PATH,
        max_accounts=args.accounts,
        nb_scenarios=args.scenarios,
        nb_years=args.years,
        debug_account_id=args.debug_account,
        debug_scenario_id=args.debug_scenario,
        start_year_out=args.start_year,
        end_year_out=args.end_year
    )

    print("\n" + "=" * 80 + "\nSAMPLE RESULTS\n" + "=" * 80)
    print("\nVP_FLUX_TOTAL:\n", results['vp_flux_total'])
    print("\nVP_FLUX_COMPTE (first 5 accounts):\n", results['vp_flux_compte'].head(5).to_string())
    print("\nFLUX_PROJETES (first 10 periods):\n", results['flux_projetes'].head(10).to_string())


if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
import logging
import time
from pathlib import Path
from tqdm import tqdm
import warnings

# Attempt to import CuPy and set the backend
try:
    import cupy as cp

    GPU_ENABLED = True
    logger = logging.getLogger(__name__)
    logger.info("CuPy found. Running on GPU.")
except ImportError:
    cp = np
    GPU_ENABLED = False
    logger = logging.getLogger(__name__)
    logger.warning("CuPy not found. Running on CPU with NumPy.")

from paths import HERE

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)

# --- FIX 1: Synchronized Global Parameters ---
# These parameters should be identical to the CPU version for a valid comparison.
# Using smaller values for faster testing and development.
NBCPT = 4
NB_SC = 10
NB_AN_PROJECTION = 10
NB_SC_INT = 10
NB_AN_PROJECTION_INT = 10
CHOC_CAPITAL = 0.35
HURDLE_RT = 0.10


def load_input_files(data_path: str) -> Tuple[pd.DataFrame, ...]:
    """Load all input CSV files exactly as SAS does"""
    try:
        population = pd.read_csv(f"{data_path}/population.csv").head(NBCPT)
        rendement = pd.read_csv(f"{data_path}/rendement.csv")
        tx_deces = pd.read_csv(f"{data_path}/tx_deces.csv")
        tx_interet = pd.read_csv(f"{data_path}/tx_interet.csv")
        tx_interet_int = pd.read_csv(f"{data_path}/tx_interet_int.csv")
        tx_retrait = pd.read_csv(f"{data_path}/tx_retrait.csv")

        if 'TYPE' in rendement.columns:
            rendement['TYPE'] = rendement['TYPE'].apply(
                lambda x: x.decode('utf-8') if isinstance(x, bytes) else str(x)
            )

        logger.info(f"Input files loaded - Population: {len(population)} accounts")
        return population, rendement, tx_deces, tx_interet, tx_interet_int, tx_retrait
    except Exception as e:
        logger.error(f"Error loading input files: {e}")
        raise


def prepare_gpu_data(rendement: pd.DataFrame, tx_deces: pd.DataFrame,
                     tx_interet: pd.DataFrame, tx_interet_int: pd.DataFrame, tx_retrait: pd.DataFrame,
                     xp) -> Dict:
    """
    Converts all Pandas DataFrames and lookup tables into GPU-ready arrays.
    """
    max_age = int(tx_deces['AGE'].max()) + 1 if not tx_deces.empty else 150
    h_mortality = np.zeros(max_age, dtype=np.float64)
    if not tx_deces.empty:
        h_mortality[tx_deces['AGE'].astype(int)] = tx_deces['QX']

    max_proj_years = max(NB_AN_PROJECTION, int(tx_retrait['an_proj'].max() if not tx_retrait.empty else 0)) + 1
    g_lapse = np.zeros(max_proj_years, dtype=np.float64)
    if not tx_retrait.empty:
        g_lapse[tx_retrait['an_proj'].astype(int)] = tx_retrait['WX']

    max_sc = max(NB_SC, NB_SC_INT, int(rendement['scn_proj'].max() if not rendement.empty else 0)) + 1
    max_an = max(NB_AN_PROJECTION, NB_AN_PROJECTION_INT,
                 int(rendement['an_proj'].max() if not rendement.empty else 0)) + 1
    z_rendement = np.zeros((max_sc, max_an, 2), dtype=np.float64)

    type_map = {'EXTERNE': 0, 'INTERNE': 1}
    for _, row in rendement.iterrows():
        scn = int(row['scn_proj'])
        an = int(row['an_proj'])
        type_idx = type_map.get(str(row['TYPE']), -1)
        if type_idx != -1 and scn < max_sc and an < max_an:
            z_rendement[scn, an, type_idx] = float(row['RENDEMENT'])

    max_discount_years = max(NB_AN_PROJECTION, int(tx_interet['an_proj'].max() if not tx_interet.empty else 0)) + 1
    a_discount_ext = np.ones(max_discount_years, dtype=np.float64)
    if not tx_interet.empty:
        a_discount_ext[tx_interet['an_proj'].astype(int)] = tx_interet['TX_ACTU']

    max_int_discount_years = max(NB_AN_PROJECTION_INT,
                                 int(tx_interet_int['an_eval'].max() if not tx_interet_int.empty else 0)) + 1
    b_discount_int = np.ones(max_int_discount_years, dtype=np.float64)
    if not tx_interet_int.empty:
        b_discount_int[tx_interet_int['an_eval'].astype(int)] = tx_interet_int['TX_ACTU_INT']

    data = {
        'h_mortality': xp.asarray(h_mortality),
        'g_lapse': xp.asarray(g_lapse),
        'z_rendement': xp.asarray(z_rendement),
        'a_discount_ext': xp.asarray(a_discount_ext),
        'b_discount_int': xp.asarray(b_discount_int)
    }

    logger.info("Lookup tables converted to arrays and moved to target device.")
    return data


def vectorized_cash_flow_calculation(initial_state: Dict, lookup_data: Dict, params: Dict, xp) -> pd.DataFrame:
    """
    Vectorized implementation of the cash flow calculation for a batch of simulations.
    """
    batch_size = params['batch_size']
    max_years = params['max_years']
    scenario_type = params['scenario_type']

    h_mortality, g_lapse, z_rendement = lookup_data['h_mortality'], lookup_data['g_lapse'], lookup_data['z_rendement']
    a_discount_ext, b_discount_int = lookup_data['a_discount_ext'], lookup_data['b_discount_int']

    mt_vm_proj = xp.copy(initial_state['mt_vm'])
    mt_gar_deces_proj = xp.copy(initial_state['mt_gar_deces'])
    tx_survie = xp.copy(initial_state['tx_survie_deb'])

    age_deb = initial_state['age_deb']
    pc_revenu_fds = initial_state['pc_revenu_fds']
    pc_honoraires_gest = initial_state['pc_honoraires_gest']
    tx_comm_maintien = initial_state['tx_comm_maintien']
    frais_admin = initial_state['frais_admin']
    freq_reset_deces = initial_state['freq_reset_deces']
    max_reset_deces = initial_state['max_reset_deces']

    num_outputs = 13
    results_batch = xp.zeros((max_years + 1, batch_size, num_outputs), dtype=xp.float64)

    for current_year in range(max_years + 1):
        an_proj_relative = current_year

        if scenario_type == "INTERNE":
            an_proj = initial_state['an_eval_start'] + an_proj_relative
            age = age_deb + an_proj
        else:  # EXTERNE
            # --- FIX: Ensure an_proj is always a broadcastable array ---
            # Instead of being a scalar `current_year`, it's now an array of that value.
            an_proj = xp.full(batch_size, an_proj_relative, dtype=xp.int32)
            age = age_deb + an_proj

        if current_year == 0:
            if scenario_type == "EXTERNE":
                tx_survie_deb = xp.ones(batch_size, dtype=xp.float64)
                commissions = -initial_state['tx_comm_vente'] * mt_vm_proj
                frais_gen = -initial_state['frais_acqui']
                flux_net = frais_gen + commissions
                revenus, frais_gest, pmt_garantie = 0.0, 0.0, 0.0
            else:  # INTERNE
                tx_survie_deb = xp.copy(tx_survie)
                commissions, frais_gen, flux_net, revenus, frais_gest, pmt_garantie = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        else:
            active_mask = (tx_survie > 0) & (mt_vm_proj > 0) & (age < 100)

            mt_vm_deb = mt_vm_proj
            tx_survie_deb = tx_survie

            rendement_rate = z_rendement[initial_state['scn_proj'], an_proj, params['type_idx']]
            rendement = mt_vm_deb * rendement_rate
            frais = -(mt_vm_deb + rendement / 2) * pc_revenu_fds
            mt_vm_proj = mt_vm_proj + rendement + frais

            reset_mask = (freq_reset_deces == 1) & (age <= max_reset_deces)
            mt_gar_deces_proj[reset_mask] = xp.maximum(mt_gar_deces_proj[reset_mask], mt_vm_proj[reset_mask])

            qx = h_mortality[age.astype(xp.int32)]
            wx = g_lapse[an_proj.astype(xp.int32)]
            tx_survie = tx_survie_deb * (1 - qx) * (1 - wx)

            revenus = -frais * tx_survie_deb
            frais_gest = -(mt_vm_deb + rendement / 2) * pc_honoraires_gest * tx_survie_deb
            commissions = -(mt_vm_deb + rendement / 2) * tx_comm_maintien * tx_survie_deb
            frais_gen = -frais_admin * tx_survie_deb
            pmt_garantie = -xp.maximum(0, mt_gar_deces_proj - mt_vm_proj) * qx * tx_survie_deb
            flux_net = revenus + frais_gest + commissions + frais_gen + pmt_garantie

            for cf in [revenus, frais_gest, commissions, frais_gen, pmt_garantie, flux_net, mt_vm_proj,
                       mt_gar_deces_proj, tx_survie]:
                cf[~active_mask] = 0.0

        tx_actu = a_discount_ext[an_proj.astype(xp.int32)]

        vp_flux_net = flux_net * tx_actu

        if scenario_type == "INTERNE":
            an_eval = initial_state['an_eval_start']
            if xp.any(an_eval > 0):
                tx_actu_int = b_discount_int[an_eval.astype(xp.int32)]
                inv_tx_actu_int = xp.where(tx_actu_int != 0, 1.0 / tx_actu_int, 0)
                vp_flux_net *= inv_tx_actu_int

        results_batch[current_year, :, 0] = an_proj
        results_batch[current_year, :, 1] = age
        results_batch[current_year, :, 2] = mt_vm_proj
        results_batch[current_year, :, 3] = mt_gar_deces_proj
        results_batch[current_year, :, 4] = tx_survie
        results_batch[current_year, :, 5] = tx_survie_deb
        results_batch[current_year, :, 6] = revenus
        results_batch[current_year, :, 7] = frais_gest
        results_batch[current_year, :, 8] = commissions
        results_batch[current_year, :, 9] = frais_gen
        results_batch[current_year, :, 10] = pmt_garantie
        results_batch[current_year, :, 11] = flux_net
        results_batch[current_year, :, 12] = vp_flux_net

    results_cpu = cp.asnumpy(results_batch) if GPU_ENABLED else results_batch
    results_flat = results_cpu.reshape(-1, num_outputs)

    df = pd.DataFrame(results_flat, columns=[
        'an_proj', 'AGE', 'MT_VM_PROJ', 'MT_GAR_DECES_PROJ', 'TX_SURVIE', 'TX_SURVIE_DEB',
        'REVENUS', 'FRAIS_GEST', 'COMMISSIONS', 'FRAIS_GEN', 'PMT_GARANTIE', 'FLUX_NET', 'VP_FLUX_NET'
    ])

    return df

def gpu_calculs_macro(population: pd.DataFrame, lookup_data: Dict, xp):
    """
    Main vectorized orchestrator replacing the nested loops.
    """
    num_accounts = len(population)
    logger.info("Starting vectorized external scenario calculations...")
    batch_size_ext = num_accounts * NB_SC

    id_compte_vec = np.repeat(population['ID_COMPTE'].values, NB_SC)
    scn_eval_vec = np.tile(np.arange(1, NB_SC + 1), num_accounts)

    initial_state_ext = {
        'mt_vm': xp.asarray(np.repeat(population['MT_VM'].values, NB_SC)),
        'mt_gar_deces': xp.asarray(np.repeat(population['MT_GAR_DECES'].values, NB_SC)),
        'tx_survie_deb': xp.ones(batch_size_ext, dtype=xp.float64),
        'age_deb': xp.asarray(np.repeat(population['age_deb'].values, NB_SC)),
        'pc_revenu_fds': xp.asarray(np.repeat(population['PC_REVENU_FDS'].values, NB_SC)),
        'pc_honoraires_gest': xp.asarray(np.repeat(population['PC_HONORAIRES_GEST'].values, NB_SC)),
        'tx_comm_maintien': xp.asarray(np.repeat(population['TX_COMM_MAINTIEN'].values, NB_SC)),
        'frais_admin': xp.asarray(np.repeat(population['FRAIS_ADMIN'].values, NB_SC)),
        'freq_reset_deces': xp.asarray(np.repeat(population['FREQ_RESET_DECES'].values, NB_SC)),
        'max_reset_deces': xp.asarray(np.repeat(population['MAX_RESET_DECES'].values, NB_SC)),
        'tx_comm_vente': xp.asarray(np.repeat(population['TX_COMM_VENTE'].values, NB_SC)),
        'frais_acqui': xp.asarray(np.repeat(population['FRAIS_ACQUI'].values, NB_SC)),
        'scn_proj': xp.asarray(scn_eval_vec)
    }

    params_ext = {'batch_size': batch_size_ext, 'max_years': NB_AN_PROJECTION, 'scenario_type': 'EXTERNE',
                  'type_idx': 0}
    external_results = vectorized_cash_flow_calculation(initial_state_ext, lookup_data, params_ext, xp)

    # --- FIX 3 (Part 1): Add static policy data to external_results for safe transfer ---
    # This ensures that when we filter by `an_eval`, we retain all necessary policy-level data.
    num_years_proj = NB_AN_PROJECTION + 1
    external_results['ID_COMPTE'] = np.repeat(id_compte_vec, num_years_proj)
    external_results['scn_eval'] = np.repeat(scn_eval_vec, num_years_proj)
    external_results['an_eval'] = external_results['an_proj']

    # Carry forward static parameters needed for the internal calculations
    for key in ['age_deb', 'pc_revenu_fds', 'pc_honoraires_gest', 'tx_comm_maintien', 'frais_admin', 'freq_reset_deces',
                'max_reset_deces']:
        external_results[key] = np.repeat(cp.asnumpy(initial_state_ext[key]) if GPU_ENABLED else initial_state_ext[key],
                                          num_years_proj)

    logger.info(f"External calculations complete. Shape: {external_results.shape}")

    logger.info("Starting vectorized internal scenario calculations...")
    reserve_capital_results = []
    for an_eval in tqdm(range(1, NB_AN_PROJECTION + 1), desc="Internal Calcs (Years)"):
        prev_state = external_results[external_results['an_eval'] == an_eval]
        if prev_state.empty: continue

        num_paths_this_year = len(prev_state)
        repeats = NB_SC_INT * 2
        batch_size_int = num_paths_this_year * repeats

        mt_vm_int = np.repeat(prev_state['MT_VM_PROJ'].values, repeats)
        type2_vec = np.tile(np.repeat(['RESERVE', 'CAPITAL'], NB_SC_INT), num_paths_this_year)
        shock_mask = (type2_vec == 'CAPITAL')
        mt_vm_int[shock_mask] *= (1 - CHOC_CAPITAL)

        # --- FIX 3 (Part 2): Source all data from `prev_state` for guaranteed alignment ---
        # Instead of rebuilding arrays from the original `population` or `initial_state_ext`,
        # we use the already-aligned data from the `prev_state` DataFrame.
        initial_state_int = {
            'mt_vm': xp.asarray(mt_vm_int),
            'mt_gar_deces': xp.asarray(np.repeat(prev_state['MT_GAR_DECES_PROJ'].values, repeats)),
            'tx_survie_deb': xp.asarray(np.repeat(prev_state['TX_SURVIE'].values, repeats)),
            'age_deb': xp.asarray(np.repeat(prev_state['age_deb'].values, repeats)),
            'scn_proj': xp.asarray(np.tile(np.arange(1, NB_SC_INT + 1), num_paths_this_year * 2)),
            'an_eval_start': xp.asarray(np.repeat(an_eval, batch_size_int)),
        }
        for key in ['pc_revenu_fds', 'pc_honoraires_gest', 'tx_comm_maintien', 'frais_admin', 'freq_reset_deces',
                    'max_reset_deces']:
            initial_state_int[key] = xp.asarray(np.repeat(prev_state[key].values, repeats))

        # The loop runs for the max projection years, the internal mask handles early termination
        params_int = {'batch_size': batch_size_int, 'max_years': NB_AN_PROJECTION_INT, 'scenario_type': 'INTERNE',
                      'type_idx': 1}
        internal_results_df = vectorized_cash_flow_calculation(initial_state_int, lookup_data, params_int, xp)

        max_years_int_proj = NB_AN_PROJECTION_INT + 1
        internal_results_df['ID_COMPTE'] = np.repeat(prev_state['ID_COMPTE'].values, repeats * max_years_int_proj)
        internal_results_df['scn_eval'] = np.repeat(prev_state['scn_eval'].values, repeats * max_years_int_proj)
        internal_results_df['scn_eval_int'] = np.tile(np.arange(1, NB_SC_INT + 1),
                                                      num_paths_this_year * 2 * max_years_int_proj)
        internal_results_df['TYPE2'] = np.repeat(type2_vec, max_years_int_proj)
        internal_results_df['an_eval'] = an_eval

        agg = internal_results_df.groupby(['ID_COMPTE', 'scn_eval', 'an_eval', 'TYPE2', 'scn_eval_int'])[
            'VP_FLUX_NET'].sum().reset_index()
        mean_agg = agg.groupby(['ID_COMPTE', 'scn_eval', 'an_eval', 'TYPE2'])['VP_FLUX_NET'].mean().reset_index()
        reserve_capital_results.append(mean_agg)

    logger.info("Merging results and performing final calculations...")
    if not reserve_capital_results:
        logger.warning("No internal results generated. Skipping final merge.")
        enhanced_external = external_results.copy()
        enhanced_external['RESERVE'] = 0.0
        enhanced_external['CAPITAL'] = 0.0
    else:
        rc_df = pd.concat(reserve_capital_results)
        rc_pivot = rc_df.pivot_table(index=['ID_COMPTE', 'scn_eval', 'an_eval'], columns='TYPE2',
                                     values='VP_FLUX_NET').reset_index()
        rc_pivot.rename(columns={'RESERVE': 'RESERVE_val', 'CAPITAL': 'CAPITAL_base'}, inplace=True)
        enhanced_external = pd.merge(external_results, rc_pivot, on=['ID_COMPTE', 'scn_eval', 'an_eval'], how='left')
        enhanced_external[['RESERVE_val', 'CAPITAL_base']] = enhanced_external[['RESERVE_val', 'CAPITAL_base']].fillna(
            0)
        enhanced_external['RESERVE'] = enhanced_external['RESERVE_val']
        enhanced_external['CAPITAL'] = enhanced_external['CAPITAL_base'] - enhanced_external['RESERVE']

    enhanced_external = enhanced_external.sort_values(['ID_COMPTE', 'scn_eval', 'an_eval']).reset_index(drop=True)
    enhanced_external['reserve_prec'] = enhanced_external.groupby(['ID_COMPTE', 'scn_eval'])['RESERVE'].shift(1,
                                                                                                              fill_value=0)
    enhanced_external['capital_prec'] = enhanced_external.groupby(['ID_COMPTE', 'scn_eval'])['CAPITAL'].shift(1,
                                                                                                              fill_value=0)

    is_year_0 = enhanced_external['an_eval'] == 0
    enhanced_external.loc[is_year_0, 'PROFIT'] = enhanced_external['FLUX_NET'] + enhanced_external['RESERVE']
    enhanced_external.loc[is_year_0, 'FLUX_DISTRIBUABLES'] = enhanced_external['PROFIT'] + enhanced_external['CAPITAL']

    enhanced_external.loc[~is_year_0, 'PROFIT'] = enhanced_external['FLUX_NET'] + enhanced_external['RESERVE'] - \
                                                  enhanced_external['reserve_prec']
    enhanced_external.loc[~is_year_0, 'FLUX_DISTRIBUABLES'] = enhanced_external['PROFIT'] + enhanced_external[
        'CAPITAL'] - enhanced_external['capital_prec']

    enhanced_external['VP_FLUX_DISTRIBUABLES'] = enhanced_external['FLUX_DISTRIBUABLES'] / (
                (1 + HURDLE_RT) ** enhanced_external['an_eval'])

    calculs_sommaire = enhanced_external.groupby(['ID_COMPTE', 'scn_eval'])['VP_FLUX_DISTRIBUABLES'].sum().reset_index()
    logger.info(f"Vectorized calculations complete. Results: {len(calculs_sommaire)} combinations")
    return calculs_sommaire


def run_gpu_acfc(data_path: str = "data_in", output_dir: str = "output"):
    """Main function to run the vectorized/GPU ACFC implementation."""
    start_time = time.time()
    xp = cp if GPU_ENABLED else np

    device = "GPU" if GPU_ENABLED else "CPU (NumPy)"
    logger.info("=" * 80)
    logger.info(f"VECTORIZED SAS-TO-PYTHON ACFC IMPLEMENTATION (RUNNING ON {device})")
    logger.info("=" * 80)

    try:
        population, rendement, tx_deces, tx_interet, tx_interet_int, tx_retrait = load_input_files(data_path)
        lookup_data = prepare_gpu_data(rendement, tx_deces, tx_interet, tx_interet_int, tx_retrait, xp)

        logger.info(f"Configuration:")
        logger.info(f"  Accounts: {min(NBCPT, len(population))}")
        logger.info(f"  External scenarios: {NB_SC}")
        logger.info(f"  Internal scenarios: {NB_SC_INT}")
        logger.info(f"  Max projection years: {NB_AN_PROJECTION}")

        results_df = gpu_calculs_macro(population, lookup_data, xp)

        end_time = time.time()
        execution_time = end_time - start_time

        print(f"\n" + "=" * 60)
        print(f"VECTORIZED ACFC RESULTS (ON {device})")
        print(f"=" * 60)
        if not results_df.empty:
            print(f"Total combinations: {len(results_df):,}")
            print(f"Execution time: {execution_time:.2f} seconds")
            print(f"Mean VP_FLUX_DISTRIBUABLES: ${results_df['VP_FLUX_DISTRIBUABLES'].mean():,.2f}")
            print(f"Profitable combinations: {len(results_df[results_df['VP_FLUX_DISTRIBUABLES'] > 0]):,}")
            print(
                f"Range: ${results_df['VP_FLUX_DISTRIBUABLES'].min():,.2f} to ${results_df['VP_FLUX_DISTRIBUABLES'].max():,.2f}")
        else:
            print("No results were generated.")

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        results_file = output_path / "acfc_results_gpu.csv"
        results_df.to_csv(results_file, index=False)
        logger.info(f"Results saved to {results_file}")

        return results_df

    except Exception as e:
        logger.error(f"Error in vectorized ACFC execution: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise


def main():
    """Main execution"""
    try:
        results_df = run_gpu_acfc(
            data_path=HERE.joinpath("data_in"),
            output_dir=HERE.joinpath("test"),
        )
        return results_df
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise


if __name__ == "__main__":
    results_df = main()
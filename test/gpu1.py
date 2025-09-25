import pandas as pd
import cupy as cp  # Replaces numpy with cupy
import logging
import time
from pathlib import Path
from tqdm import tqdm

from paths import HERE

# from paths import HERE # Assuming HERE is defined

# --- Assume paths are configured ---
# Example setup if paths.py is not available
# Make sure your data_in folder is in the same directory or adjust the path
DATA_PATH = HERE.joinpath('data_in')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_input_files():
    """Load all input CSV files using Pandas (runs on CPU)"""
    population = pd.read_csv(DATA_PATH.joinpath('population.csv'))
    rendement = pd.read_csv(DATA_PATH.joinpath('rendement.csv'))
    if 'TYPE' in rendement.columns and isinstance(rendement['TYPE'].iloc[0], bytes):
        rendement['TYPE'] = rendement['TYPE'].str.decode('utf-8')
    tx_deces = pd.read_csv(DATA_PATH.joinpath('tx_deces.csv'))
    tx_interet = pd.read_csv(DATA_PATH.joinpath('tx_interet.csv'))
    tx_interet_int = pd.read_csv(DATA_PATH.joinpath('tx_interet_int.csv'))
    tx_retrait = pd.read_csv(DATA_PATH.joinpath('tx_retrait.csv'))
    logger.info("All input files loaded successfully from disk.")
    return population, rendement, tx_deces, tx_interet, tx_interet_int, tx_retrait


def prepare_gpu_data(population_df, rendement_df, tx_deces_df, tx_interet_df, tx_interet_int_df, tx_retrait_df,
                     max_years=35):
    """
    Convert Pandas DataFrames to CuPy arrays and create efficient GPU lookup tables.
    This is a critical step for performance.
    """
    logger.info("Preparing data for GPU...")

    # 1. Convert population data to a dictionary of CuPy arrays
    population_gpu = {col: cp.array(population_df[col].values) for col in population_df.columns}

    # 2. Create dense, multi-dimensional lookup tables on the GPU

    # Rendement: Create a 3D array [type, year, scenario] for instant lookup
    # Map types 'EXTERNE', 'INTERNE' to indices 0, 1
    type_map = {'EXTERNE': 0, 'INTERNE': 1}
    max_scn = int(rendement_df['scn_proj'].max())
    rendement_lookup = cp.zeros((2, max_years + 1, max_scn + 1), dtype=cp.float32)
    rendement_lookup[
        rendement_df['TYPE'].map(type_map).values,
        rendement_df['an_proj'].values.astype(int),
        rendement_df['scn_proj'].values.astype(int)
    ] = rendement_df['RENDEMENT'].values

    # Mortality: Create a dense 1D array. Pre-calculate interpolation/extrapolation.
    max_age = 200  # A reasonable upper bound for age
    mortality_lookup = cp.zeros(max_age + 1, dtype=cp.float32)
    ages = tx_deces_df['AGE'].values
    rates = tx_deces_df['QX'].values
    mortality_lookup[ages] = rates
    # Simple forward fill for missing values
    for age in range(1, max_age + 1):
        if mortality_lookup[age] == 0:
            mortality_lookup[age] = mortality_lookup[age - 1]

    # Discount Rates & Lapse Rates: Create dense 1D arrays
    def create_dense_lookup(df, col_name, max_idx):
        lookup = cp.zeros(max_idx + 1, dtype=cp.float32)
        idx = df.iloc[:, 0].values.astype(int)
        vals = df[col_name].values
        lookup[idx] = vals
        # Forward fill for missing years
        for i in range(1, max_idx + 1):
            if lookup[i] == 0: lookup[i] = lookup[i - 1]
        return lookup

    discount_ext_lookup = create_dense_lookup(tx_interet_df, 'TX_ACTU', max_years)
    discount_int_lookup = create_dense_lookup(tx_interet_int_df, 'TX_ACTU_INT', max_years)
    lapse_lookup = create_dense_lookup(tx_retrait_df, 'WX', max_years)

    # Get scenario lists
    external_scenarios = cp.unique(cp.array(rendement_df[rendement_df['TYPE'] == 'EXTERNE']['scn_proj'].values))
    internal_scenarios = cp.unique(cp.array(rendement_df[rendement_df['TYPE'] == 'INTERNE']['scn_proj'].values))

    logger.info("GPU data prepared successfully.")
    return population_gpu, rendement_lookup, mortality_lookup, discount_ext_lookup, discount_int_lookup, lapse_lookup, external_scenarios, internal_scenarios


def external_loop_gpu(population_gpu, lookups, external_scenarios, max_years=35):
    """
    TIER 1: Vectorized external loop running entirely on the GPU.
    """
    logger.info("=" * 50)
    logger.info("TIER 1: EXTERNAL LOOP PROCESSING (GPU)")
    logger.info("=" * 50)

    rendement_lookup, mortality_lookup, _, _, lapse_lookup, discount_ext_lookup = lookups
    num_accounts = len(population_gpu['ID_COMPTE'])
    num_scenarios = len(external_scenarios)

    # --- Initialize state arrays on GPU ---
    # Shape of all state arrays: (num_accounts, num_scenarios)
    mt_vm = cp.full((num_accounts, num_scenarios), population_gpu['MT_VM'][:, None])
    mt_gar_deces = cp.full((num_accounts, num_scenarios), population_gpu['MT_GAR_DECES'][:, None])
    current_age = cp.full((num_accounts, num_scenarios), population_gpu['age_deb'][:, None])
    tx_survie = cp.ones((num_accounts, num_scenarios), dtype=cp.float32)

    # --- Initialize result storage arrays on GPU ---
    # Shape: (max_years + 1, num_accounts, num_scenarios)
    all_flux_net = cp.zeros((max_years + 1, num_accounts, num_scenarios), dtype=cp.float32)
    all_vp_flux_net = cp.zeros((max_years + 1, num_accounts, num_scenarios), dtype=cp.float32)
    all_mt_vm = cp.zeros((max_years + 1, num_accounts, num_scenarios), dtype=cp.float32)
    all_tx_survie = cp.zeros((max_years + 1, num_accounts, num_scenarios), dtype=cp.float32)

    all_mt_vm[0], all_tx_survie[0] = mt_vm, tx_survie

    # The only loop is over time. All accounts/scenarios are processed in parallel.
    for year in tqdm(range(1, max_years + 1), desc="GPU External Loop (Years)"):
        # Create a mask for active policies to avoid unnecessary calculations
        active_mask = (tx_survie > 1e-6) & (mt_vm > 0)

        # --- Perform vectorized lookups for ALL simulations at once ---
        rendement = rendement_lookup[0, year, external_scenarios]  # Shape: (num_scenarios,)
        qx = mortality_lookup[current_age.astype(cp.int32)]  # Shape: (num_accounts, num_scenarios)
        wx = lapse_lookup[year]  # Scalar

        # 1. Fund Value Projection (vectorized)
        mt_vm_deb = mt_vm.copy()
        rendement_amount = mt_vm * rendement  # Broadcasting applies scenario return
        frais_adj = -(mt_vm_deb + rendement_amount / 2) * population_gpu['PC_REVENU_FDS'][:, None]
        mt_vm = cp.maximum(0, mt_vm + rendement_amount + frais_adj)

        # 2. Death Benefit Guarantee (vectorized)
        reset_mask = (population_gpu['FREQ_RESET_DECES'][:, None] == 1.0) & \
                     (current_age <= population_gpu['MAX_RESET_DECES'][:, None])
        mt_gar_deces = cp.where(reset_mask, cp.maximum(mt_gar_deces, mt_vm), mt_gar_deces)

        # 3. Survival Probability (vectorized)
        tx_survie_previous = tx_survie.copy()
        tx_survie = tx_survie * (1 - qx) * (1 - wx)

        # 4. Cash Flow Components (vectorized)
        frais_t = -(mt_vm_deb + rendement_amount / 2) * population_gpu['PC_REVENU_FDS'][:, None]
        revenus = -frais_t * tx_survie_previous
        frais_gest = -(mt_vm_deb + rendement_amount / 2) * population_gpu['PC_HONORAIRES_GEST'][:,
                                                           None] * tx_survie_previous
        commissions = -(mt_vm_deb + rendement_amount / 2) * population_gpu['TX_COMM_MAINTIEN'][:,
                                                            None] * tx_survie_previous
        frais_gen = -population_gpu['FRAIS_ADMIN'][:, None] * tx_survie_previous
        death_claim = cp.maximum(0, mt_gar_deces - mt_vm) * qx * tx_survie_previous
        pmt_garantie = -death_claim
        flux_net = revenus + frais_gest + commissions + frais_gen + pmt_garantie

        # 5. Present Value (vectorized)
        tx_actu = discount_ext_lookup[year]
        vp_flux_net = flux_net * tx_actu

        # Update states only for active policies
        mt_vm = cp.where(active_mask, mt_vm, 0)
        tx_survie = cp.where(active_mask, tx_survie, 0)

        # Store results
        all_flux_net[year], all_vp_flux_net[year] = flux_net, vp_flux_net
        all_mt_vm[year], all_tx_survie[year] = mt_vm, tx_survie

        current_age += 1

    logger.info("TIER 1 COMPLETE")
    return {
        "flux_net": all_flux_net,
        "vp_flux_net": all_vp_flux_net,
        "mt_vm": all_mt_vm,
        "tx_survie": all_tx_survie
    }


def internal_projection_gpu(base_mt_vm_ts, base_tx_survie_ts, policy_data, lookups, internal_scenarios,
                            capital_shock=0.0, max_years=35):
    """
    Reusable GPU function to project all internal scenarios for a SINGLE external result.
    This is the core of Tier 2 and 3.
    """
    rendement_lookup, _, discount_int_lookup, _, _, _ = lookups
    num_internal_scenarios = len(internal_scenarios)

    # State arrays are now 1D for internal scenarios. Shape: (num_internal_scenarios,)
    scenario_pv_total = cp.zeros(num_internal_scenarios, dtype=cp.float32)

    for year in range(1, max_years + 1):
        # Starting point from the external projection for this year
        base_fund_value = base_mt_vm_ts[year]
        survival = base_tx_survie_ts[year]

        if survival > 1e-6:
            # Apply shock for capital calculations (Tier 3)
            shocked_fund_value = base_fund_value * (1 - capital_shock)

            # Vectorized calculation for all internal scenarios
            internal_cf = shocked_fund_value * policy_data['PC_REVENU_FDS'] * survival

            # Present value using internal discount rates
            tx_actu_int = discount_int_lookup[year]
            internal_pv = internal_cf * tx_actu_int
            scenario_pv_total += internal_pv

    # Aggregate result: MEAN across all internal scenarios
    return cp.mean(scenario_pv_total)


def run_internal_loops_gpu(external_results, population_gpu, lookups, internal_scenarios, max_years=35):
    """
    Orchestrates Tier 2 (Reserve) and Tier 3 (Capital) calculations.
    It iterates on the CPU but launches fast GPU kernels for each projection.
    """
    logger.info("=" * 50)
    logger.info("TIER 2 & 3: INTERNAL LOOPS PROCESSING (GPU)")
    logger.info("=" * 50)

    num_accounts = external_results['mt_vm'].shape[1]
    num_ext_scenarios = external_results['mt_vm'].shape[2]

    reserve_results = cp.zeros((num_accounts, num_ext_scenarios), dtype=cp.float32)
    capital_results = cp.zeros((num_accounts, num_ext_scenarios), dtype=cp.float32)

    # Unpack lookups needed for the internal kernel
    internal_lookups = (lookups[0], lookups[2], lookups[3], lookups[4], lookups[5], lookups[1])

    # This loop is on the CPU, but each iteration is a fast, parallel GPU computation.
    total_iterations = num_accounts * num_ext_scenarios
    with tqdm(total=total_iterations, desc="GPU Internal Loops (Reserves & Capital)") as pbar:
        for acc_idx in range(num_accounts):
            # Create a dictionary for the single policy's data
            policy_data = {key: val[acc_idx] for key, val in population_gpu.items()}
            for scn_idx in range(num_ext_scenarios):
                # Get the time series data for this specific external run
                base_mt_vm_ts = external_results['mt_vm'][:, acc_idx, scn_idx]
                base_tx_survie_ts = external_results['tx_survie'][:, acc_idx, scn_idx]

                # TIER 2: Reserve Calculation
                reserve_results[acc_idx, scn_idx] = internal_projection_gpu(
                    base_mt_vm_ts, base_tx_survie_ts, policy_data, internal_lookups, internal_scenarios,
                    capital_shock=0.0, max_years=max_years
                )

                # TIER 3: Capital Calculation
                capital_results[acc_idx, scn_idx] = internal_projection_gpu(
                    base_mt_vm_ts, base_tx_survie_ts, policy_data, internal_lookups, internal_scenarios,
                    capital_shock=0.35, max_years=max_years
                )
                pbar.update(1)

    logger.info("TIER 2 & 3 COMPLETE")
    return reserve_results, capital_results


def final_integration_gpu(external_results, reserve_results, capital_results, hurdle_rate=0.10, max_years=35):
    """
    PHASE 5: Vectorized final integration on the GPU.
    """
    logger.info("=" * 50)
    logger.info("PHASE 5: FINAL INTEGRATION (GPU)")
    logger.info("=" * 50)

    external_cf = external_results['flux_net']

    # In this simplified model, reserve and capital are constant over the projection horizon for a given scenario
    # delta_reserve is non-zero only at year 1. For simplicity, we add the total reserve.
    # This matches the logic of the original code.
    profit = external_cf[1:] + reserve_results[None, :, :]
    distributable_amount = profit + capital_results[None, :, :]

    # Create a discount vector on the GPU
    years = cp.arange(1, max_years + 1)
    discount_factors = 1 / ((1 + hurdle_rate) ** years)

    # Calculate Present Value by multiplying and summing over the time axis (axis=0)
    pv_distributable = (distributable_amount * discount_factors[:, None, None]).sum(axis=0)

    return pv_distributable


def run_acfc_gpu(max_years=35):
    """Main function to run the entire GPU-accelerated algorithm."""
    logger.info("=" * 60)
    logger.info("ACTUARIAL CASH FLOW CALCULATION (ACFC) ALGORITHM - GPU VERSION")
    logger.info("=" * 60)
    start_time = time.time()

    # Phase 1: Load data on CPU, then prepare it for GPU
    data_frames = load_input_files()
    population_gpu, *lookups, external_scenarios, internal_scenarios = prepare_gpu_data(*data_frames,
                                                                                        max_years=max_years)

    # TIER 1: External Loop
    external_results = external_loop_gpu(population_gpu, lookups, external_scenarios, max_years=max_years)

    # TIER 2 & 3: Internal Loops
    reserve_results, capital_results = run_internal_loops_gpu(external_results, population_gpu, lookups,
                                                              internal_scenarios, max_years=max_years)

    # Phase 5: Final Integration
    final_pv_results = final_integration_gpu(external_results, reserve_results, capital_results, max_years=max_years)

    # --- Post-processing: Transfer final results back to CPU to create a DataFrame ---
    logger.info("Transferring final results from GPU to CPU...")
    # Create meshgrid of account IDs and scenario numbers on the GPU
    account_ids_grid, scenario_grid = cp.meshgrid(population_gpu['ID_COMPTE'], external_scenarios, indexing='ij')

    # Flatten all results and bring them to the host (CPU)
    final_df = pd.DataFrame({
        'ID_COMPTE': cp.asnumpy(account_ids_grid.flatten()),
        'scn_eval': cp.asnumpy(scenario_grid.flatten()),
        'VP_FLUX_DISTRIBUABLES': cp.asnumpy(final_pv_results.flatten())
    })

    elapsed_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"GPU ACFC ALGORITHM COMPLETED in {elapsed_time:.2f} seconds")
    logger.info("=" * 60)

    return final_df


if __name__ == "__main__":
    try:
        results_df_gpu = run_acfc_gpu()

        # You can reuse the analysis and summary functions from the original script
        # from original_script import analyze_results, print_results_summary

        # Example of how to use them (assuming they are in the same file or imported)
        # analysis = analyze_results(results_df_gpu)
        # print_results_summary(results_df_gpu, analysis)

        output_filename = HERE.joinpath('test/gpu1.csv')
        results_df_gpu.to_csv(output_filename, index=False)
        logger.info(f"GPU results saved to {output_filename}")

    except Exception as e:
        logger.error(f"An error occurred during the GPU run: {e}")
        logger.error("Please ensure you have a compatible NVIDIA GPU, CUDA, and CuPy installed.")
        raise
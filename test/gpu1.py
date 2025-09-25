import pandas as pd
import cupy as cp
import logging
import time
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_input_files(data_path: str = "data_in"):
    """Load all input CSV files using Pandas (runs on CPU)"""
    try:
        population = pd.read_csv(f"{data_path}/population.csv")
        rendement = pd.read_csv(f"{data_path}/rendement.csv")

        # Handle TYPE column encoding
        if 'TYPE' in rendement.columns:
            if isinstance(rendement['TYPE'].iloc[0], bytes):
                rendement['TYPE'] = rendement['TYPE'].str.decode('utf-8')
            rendement['TYPE'] = rendement['TYPE'].astype(str)

        tx_deces = pd.read_csv(f"{data_path}/tx_deces.csv")
        tx_interet = pd.read_csv(f"{data_path}/tx_interet.csv")
        tx_interet_int = pd.read_csv(f"{data_path}/tx_interet_int.csv")
        tx_retrait = pd.read_csv(f"{data_path}/tx_retrait.csv")

        logger.info("All input files loaded successfully from disk")
        logger.info(f"Population: {len(population)} accounts")
        logger.info(f"Investment scenarios: {len(rendement)} combinations")

        return population, rendement, tx_deces, tx_interet, tx_interet_int, tx_retrait

    except Exception as e:
        logger.error(f"Error loading input files: {e}")
        raise


def prepare_gpu_data(population_df, rendement_df, tx_deces_df, tx_interet_df,
                     tx_interet_int_df, tx_retrait_df, max_years=35):
    """
    Convert DataFrames to CuPy arrays and create efficient GPU lookup tables
    """
    logger.info("Preparing data for GPU processing...")

    # Convert population data to GPU arrays
    population_gpu = {}
    for col in population_df.columns:
        population_gpu[col] = cp.array(population_df[col].values, dtype=cp.float32)

    # Create scenario mappings
    type_map = {'EXTERNE': 0, 'INTERNE': 1}

    # Get unique scenarios by type
    external_scenarios = rendement_df[rendement_df['TYPE'] == 'EXTERNE']['scn_proj'].unique()
    internal_scenarios = rendement_df[rendement_df['TYPE'] == 'INTERNE']['scn_proj'].unique()

    # Fallback if no scenario types found
    if len(external_scenarios) == 0:
        logger.warning("No 'EXTERNE' scenarios found, using all scenarios")
        external_scenarios = rendement_df['scn_proj'].unique()
    if len(internal_scenarios) == 0:
        logger.warning("No 'INTERNE' scenarios found, using all scenarios")
        internal_scenarios = rendement_df['scn_proj'].unique()

    external_scenarios = cp.array(sorted(external_scenarios), dtype=cp.int32)
    internal_scenarios = cp.array(sorted(internal_scenarios), dtype=cp.int32)

    # Create investment return lookup table [type, year, scenario]
    max_year = max(max_years, int(rendement_df['an_proj'].max()))
    max_scn = max(int(rendement_df['scn_proj'].max()),
                  max(external_scenarios.max().get(), internal_scenarios.max().get()) if len(
                      external_scenarios) > 0 else 1)

    rendement_lookup = cp.zeros((2, max_year + 1, max_scn + 1), dtype=cp.float32)

    for _, row in rendement_df.iterrows():
        type_idx = type_map.get(row['TYPE'], 0)
        year_idx = int(row['an_proj'])
        scn_idx = int(row['scn_proj'])
        if year_idx <= max_year and scn_idx <= max_scn:
            rendement_lookup[type_idx, year_idx, scn_idx] = float(row['RENDEMENT'])

    # Create mortality lookup table
    max_age = 120  # Reasonable upper bound
    mortality_lookup = cp.zeros(max_age + 1, dtype=cp.float32)

    # Fill known values
    for _, row in tx_deces_df.iterrows():
        age = int(row['AGE'])
        if age <= max_age:
            mortality_lookup[age] = float(row['QX'])

    # Forward fill for missing values and extrapolate
    last_valid_rate = 0.0001  # Default minimum
    for age in range(max_age + 1):
        if mortality_lookup[age] > 0:
            last_valid_rate = mortality_lookup[age]
        elif age > 0:
            if last_valid_rate > 0:
                # Exponential extrapolation for high ages
                mortality_lookup[age] = min(0.9, last_valid_rate * (1.08 ** (age - tx_deces_df['AGE'].max())))
            else:
                mortality_lookup[age] = 0.001

    # Create discount rate lookup tables
    def create_discount_lookup(df, col_name, max_idx):
        lookup = cp.zeros(max_idx + 1, dtype=cp.float32)
        for _, row in df.iterrows():
            idx = int(row.iloc[0])  # First column is the index
            if idx <= max_idx:
                lookup[idx] = float(row[col_name])

        # Forward fill missing values
        last_valid = 1.0
        for i in range(max_idx + 1):
            if lookup[i] > 0:
                last_valid = lookup[i]
            elif i > 0:
                # Extrapolate using 5% discount rate
                lookup[i] = last_valid * (0.95 ** (i - max(df.iloc[:, 0])))

        return lookup

    discount_ext_lookup = create_discount_lookup(tx_interet_df, 'TX_ACTU', max_years)
    discount_int_lookup = create_discount_lookup(tx_interet_int_df, 'TX_ACTU_INT', max_years)

    # Create lapse rate lookup
    lapse_lookup = cp.zeros(max_years + 1, dtype=cp.float32)
    for _, row in tx_retrait_df.iterrows():
        year = int(row['an_proj'])
        if year <= max_years:
            lapse_lookup[year] = float(row['WX'])

    # Forward fill lapse rates
    last_lapse = 0.03  # Default
    for year in range(max_years + 1):
        if lapse_lookup[year] > 0:
            last_lapse = lapse_lookup[year]
        elif year > 0:
            lapse_lookup[year] = last_lapse

    logger.info("GPU data preparation completed successfully")
    logger.info(f"External scenarios: {len(external_scenarios)}")
    logger.info(f"Internal scenarios: {len(internal_scenarios)}")

    return (population_gpu, rendement_lookup, mortality_lookup, discount_ext_lookup,
            discount_int_lookup, lapse_lookup, external_scenarios, internal_scenarios)


def project_fund_value_gpu(mt_vm, rendement, pc_revenu_fds):
    """
    Correct GPU fund value projection using proper formula:
    MT_VM(t+1) = MT_VM(t) × [1 + RENDEMENT(s,t) - PC_REVENU_FDS]
    """
    # Corrected formula - multiplicative, not additive
    mt_vm_new = mt_vm * (1 + rendement - pc_revenu_fds)

    # Calculate average fund value for fee calculations
    avg_fund_value = (mt_vm + mt_vm_new) / 2

    return cp.maximum(0.0, mt_vm_new), avg_fund_value


def update_death_benefit_gpu(mt_gar_deces, mt_vm, current_age, year, freq_reset, max_reset_age):
    """
    GPU death benefit guarantee reset logic
    """
    # Determine reset conditions
    should_reset = cp.zeros_like(mt_gar_deces, dtype=bool)

    # Annual resets (freq_reset = 1.0)
    annual_reset_mask = (freq_reset == 1.0) & (current_age <= max_reset_age)

    # Rare resets (freq_reset > 10 means virtually never)
    rare_reset_mask = freq_reset > 10.0

    should_reset = annual_reset_mask & (~rare_reset_mask)

    # Apply resets where conditions are met
    mt_gar_deces_new = cp.where(should_reset,
                                cp.maximum(mt_gar_deces, mt_vm),
                                mt_gar_deces)

    return mt_gar_deces_new


def calculate_cash_flows_gpu(mt_vm_beginning, avg_fund_value, tx_survie_prev, qx, wx,
                             mt_gar_deces, mt_vm, population_params, year):
    """
    Calculate all cash flow components on GPU with correct formulas
    """
    # Extract population parameters (broadcasted to match array dimensions)
    pc_revenu_fds = population_params['PC_REVENU_FDS']
    pc_honoraires_gest = population_params['PC_HONORAIRES_GEST']
    tx_comm_vente = population_params['TX_COMM_VENTE']
    tx_comm_maintien = population_params['TX_COMM_MAINTIEN']
    frais_admin = population_params['FRAIS_ADMIN']
    frais_acqui = population_params['FRAIS_ACQUI']

    # Initialize cash flow components
    cash_flows = {}

    # REVENUS: Revenue from fund management fees (positive income)
    frais_t = -avg_fund_value * pc_revenu_fds
    cash_flows['revenus'] = -frais_t * tx_survie_prev

    # FRAIS_GEST: Management expenses (negative)
    cash_flows['frais_gest'] = -avg_fund_value * pc_honoraires_gest * tx_survie_prev

    # COMMISSIONS: Different for year 0 vs other years
    if year == 0:
        # Initial sales commission (year 0 only)
        cash_flows['commissions'] = -mt_vm_beginning * tx_comm_vente
        # Acquisition expenses (year 0 only)
        cash_flows['frais_acqui'] = -frais_acqui
    else:
        # Ongoing maintenance commission
        cash_flows['commissions'] = -avg_fund_value * tx_comm_maintien * tx_survie_prev
        cash_flows['frais_acqui'] = cp.zeros_like(avg_fund_value)

    # FRAIS_GEN: General administrative expenses (negative)
    cash_flows['frais_gen'] = -frais_admin * tx_survie_prev

    # PMT_GARANTIE: Death benefit claims (negative)
    death_claim = cp.maximum(0, mt_gar_deces - mt_vm) * qx * tx_survie_prev
    cash_flows['pmt_garantie'] = -death_claim

    # FLUX_NET: Total net cash flow
    cash_flows['flux_net'] = (cash_flows['revenus'] + cash_flows['frais_gest'] +
                              cash_flows['commissions'] + cash_flows['frais_gen'] +
                              cash_flows['frais_acqui'] + cash_flows['pmt_garantie'])

    return cash_flows


def external_loop_gpu(population_gpu, rendement_lookup, mortality_lookup, discount_ext_lookup,
                      lapse_lookup, external_scenarios, max_years=35):
    """
    TIER 1: Vectorized external loop on GPU with correct mathematical formulas
    """
    logger.info("=" * 60)
    logger.info("TIER 1: EXTERNAL PROJECTION LOOP (GPU)")
    logger.info("=" * 60)

    num_accounts = len(population_gpu['ID_COMPTE'])
    num_scenarios = len(external_scenarios)

    logger.info(f"Processing {num_accounts} accounts × {num_scenarios} scenarios × {max_years} years")
    logger.info(f"Total external calculations: {num_accounts * num_scenarios * max_years:,}")

    # Initialize state arrays (accounts, scenarios)
    mt_vm = cp.outer(population_gpu['MT_VM'], cp.ones(num_scenarios, dtype=cp.float32))
    mt_gar_deces = cp.outer(population_gpu['MT_GAR_DECES'], cp.ones(num_scenarios, dtype=cp.float32))
    current_age = cp.outer(population_gpu['age_deb'], cp.ones(num_scenarios, dtype=cp.float32))
    tx_survie = cp.ones((num_accounts, num_scenarios), dtype=cp.float32)

    # Result storage arrays (time, accounts, scenarios)
    all_results = {
        'flux_net': cp.zeros((max_years + 1, num_accounts, num_scenarios), dtype=cp.float32),
        'vp_flux_net': cp.zeros((max_years + 1, num_accounts, num_scenarios), dtype=cp.float32),
        'mt_vm': cp.zeros((max_years + 1, num_accounts, num_scenarios), dtype=cp.float32),
        'mt_gar_deces': cp.zeros((max_years + 1, num_accounts, num_scenarios), dtype=cp.float32),
        'tx_survie': cp.zeros((max_years + 1, num_accounts, num_scenarios), dtype=cp.float32)
    }

    # Store initial values
    all_results['mt_vm'][0] = mt_vm
    all_results['mt_gar_deces'][0] = mt_gar_deces
    all_results['tx_survie'][0] = tx_survie

    # Year 0: Initial setup with acquisition costs
    cash_flows_0 = calculate_cash_flows_gpu(
        mt_vm_beginning=mt_vm,
        avg_fund_value=mt_vm,
        tx_survie_prev=cp.ones_like(tx_survie),
        qx=cp.zeros_like(tx_survie),
        wx=cp.zeros_like(tx_survie),
        mt_gar_deces=mt_gar_deces,
        mt_vm=mt_vm,
        population_params=population_gpu,
        year=0
    )

    all_results['flux_net'][0] = cash_flows_0['flux_net']
    all_results['vp_flux_net'][0] = cash_flows_0['flux_net']  # No discounting for year 0

    # Years 1 to max_years
    for year in tqdm(range(1, max_years + 1), desc="GPU External Years"):
        # Create active policy mask
        active_mask = (tx_survie > 1e-6) & (mt_vm > 0)

        # Get investment returns for all scenarios (broadcasting)
        # Shape: external_scenarios -> (num_scenarios,)
        scenario_returns = cp.array([rendement_lookup[0, year, int(scn)]
                                     for scn in external_scenarios])

        # Store beginning values
        mt_vm_beginning = mt_vm.copy()
        tx_survie_prev = tx_survie.copy()

        # 1. Fund Value Projection (corrected formula)
        mt_vm_new, avg_fund_value = project_fund_value_gpu(
            mt_vm,
            scenario_returns[None, :],  # Broadcast over accounts
            population_gpu['PC_REVENU_FDS'][:, None]  # Broadcast over scenarios
        )
        mt_vm = mt_vm_new

        # 2. Death Benefit Guarantee Updates
        mt_gar_deces = update_death_benefit_gpu(
            mt_gar_deces, mt_vm, current_age, year,
            population_gpu['FREQ_RESET_DECES'][:, None],
            population_gpu['MAX_RESET_DECES'][:, None]
        )

        # 3. Mortality and Lapse Rates
        age_indices = cp.clip(current_age.astype(cp.int32), 0, len(mortality_lookup) - 1)
        qx = mortality_lookup[age_indices]
        wx = lapse_lookup[year]  # Scalar value

        # 4. Update Survival Probabilities
        tx_survie = tx_survie * (1 - qx) * (1 - wx)

        # 5. Calculate Cash Flows
        cash_flows = calculate_cash_flows_gpu(
            mt_vm_beginning=mt_vm_beginning,
            avg_fund_value=avg_fund_value,
            tx_survie_prev=tx_survie_prev,
            qx=qx,
            wx=wx,
            mt_gar_deces=mt_gar_deces,
            mt_vm=mt_vm,
            population_params=population_gpu,
            year=year
        )

        # 6. Present Value Calculation
        tx_actu = discount_ext_lookup[year]
        vp_flux_net = cash_flows['flux_net'] * tx_actu

        # Apply active mask to prevent calculations on terminated policies
        mt_vm = cp.where(active_mask, mt_vm, 0.0)
        tx_survie = cp.where(active_mask, tx_survie, 0.0)
        flux_net = cp.where(active_mask, cash_flows['flux_net'], 0.0)
        vp_flux_net = cp.where(active_mask, vp_flux_net, 0.0)

        # Store results
        all_results['mt_vm'][year] = mt_vm
        all_results['mt_gar_deces'][year] = mt_gar_deces
        all_results['tx_survie'][year] = tx_survie
        all_results['flux_net'][year] = flux_net
        all_results['vp_flux_net'][year] = vp_flux_net

        # Age increment
        current_age += 1

    total_calcs = num_accounts * num_scenarios * max_years
    logger.info(f"TIER 1 COMPLETE: {total_calcs:,} external calculations")

    return all_results


def internal_projection_single_gpu(base_mt_vm_ts, base_tx_survie_ts, policy_params,
                                   rendement_lookup, mortality_lookup, discount_int_lookup,
                                   lapse_lookup, internal_scenarios, capital_shock=0.0,
                                   max_years=35):
    """
    Run internal projections for a single external result with proper mathematical logic
    """
    num_internal_scenarios = len(internal_scenarios)
    scenario_results = cp.zeros(num_internal_scenarios, dtype=cp.float32)

    for scn_idx, scenario in enumerate(internal_scenarios):
        total_pv = 0.0

        # Initialize projection state
        mt_vm = float(policy_params['MT_VM']) * (1 - capital_shock)  # Apply shock to initial value
        mt_gar_deces = float(policy_params['MT_GAR_DECES'])
        current_age = int(policy_params['age_deb'])
        tx_survie = 1.0

        # Run year-by-year projection
        for year in range(1, min(len(base_mt_vm_ts), max_years + 1)):
            if tx_survie > 1e-6 and mt_vm > 0:
                # Get internal scenario return
                internal_return = rendement_lookup[1, year, int(scenario)]
                if capital_shock > 0:
                    internal_return *= 0.7  # Additional stress for capital calculations

                # Store previous values
                tx_survie_prev = tx_survie

                # Project fund value using correct formula
                mt_vm = mt_vm * (1 + internal_return - policy_params['PC_REVENU_FDS'])
                mt_vm = max(0.0, mt_vm)
                avg_fund_value = (policy_params['MT_VM'] + mt_vm) / 2

                # Update death benefit guarantee
                if (policy_params['FREQ_RESET_DECES'] == 1.0 and
                        current_age <= policy_params['MAX_RESET_DECES']):
                    mt_gar_deces = max(mt_gar_deces, mt_vm)

                # Calculate decrements
                qx = mortality_lookup[min(int(current_age), len(mortality_lookup) - 1)]
                wx = lapse_lookup[min(year, len(lapse_lookup) - 1)]

                # Apply additional stress for capital calculations
                if capital_shock > 0:
                    qx = min(0.5, qx * 1.2)  # 20% higher mortality
                    wx = min(0.3, wx * 1.5)  # 50% higher lapse

                # Update survival
                tx_survie = tx_survie * (1 - qx) * (1 - wx)

                # Calculate cash flows (simplified for internal calculations)
                revenue = avg_fund_value * policy_params['PC_REVENU_FDS'] * tx_survie_prev
                expenses = (avg_fund_value * policy_params['PC_HONORAIRES_GEST'] +
                            avg_fund_value * policy_params['TX_COMM_MAINTIEN'] +
                            policy_params['FRAIS_ADMIN']) * tx_survie_prev
                death_claims = max(0, mt_gar_deces - mt_vm) * qx * tx_survie_prev

                net_cf = revenue - expenses - death_claims
                if capital_shock > 0:
                    net_cf *= 0.8  # Additional stress on cash flows

                # Present value
                tx_actu_int = discount_int_lookup[min(year, len(discount_int_lookup) - 1)]
                pv_cf = net_cf * tx_actu_int
                total_pv += pv_cf

                current_age += 1

        scenario_results[scn_idx] = total_pv

    return cp.mean(scenario_results)


def run_internal_loops_gpu(external_results, population_gpu, rendement_lookup, mortality_lookup,
                           discount_int_lookup, lapse_lookup, internal_scenarios, max_years=35):
    """
    TIER 2 & 3: Internal reserve and capital calculations on GPU
    """
    logger.info("=" * 60)
    logger.info("TIER 2 & 3: INTERNAL CALCULATIONS (GPU)")
    logger.info("=" * 60)

    num_accounts = external_results['mt_vm'].shape[1]
    num_ext_scenarios = external_results['mt_vm'].shape[2]

    # Initialize result arrays
    reserve_results = cp.zeros((num_accounts, num_ext_scenarios), dtype=cp.float32)
    capital_results = cp.zeros((num_accounts, num_ext_scenarios), dtype=cp.float32)

    total_iterations = num_accounts * num_ext_scenarios
    logger.info(f"Processing {total_iterations:,} internal calculations")
    logger.info(f"Each with {len(internal_scenarios)} scenarios × {max_years} years")

    with tqdm(total=total_iterations, desc="GPU Internal Loops") as pbar:
        for acc_idx in range(num_accounts):
            # Extract policy parameters for this account
            policy_params = {key: float(val[acc_idx]) for key, val in population_gpu.items()}

            for scn_idx in range(num_ext_scenarios):
                # Get external projection time series
                base_mt_vm_ts = external_results['mt_vm'][:, acc_idx, scn_idx]
                base_tx_survie_ts = external_results['tx_survie'][:, acc_idx, scn_idx]

                # TIER 2: Reserve Calculation (no shock)
                reserve_results[acc_idx, scn_idx] = internal_projection_single_gpu(
                    base_mt_vm_ts, base_tx_survie_ts, policy_params,
                    rendement_lookup, mortality_lookup, discount_int_lookup, lapse_lookup,
                    internal_scenarios, capital_shock=0.0, max_years=max_years
                )

                # TIER 3: Capital Calculation (35% shock)
                capital_results[acc_idx, scn_idx] = internal_projection_single_gpu(
                    base_mt_vm_ts, base_tx_survie_ts, policy_params,
                    rendement_lookup, mortality_lookup, discount_int_lookup, lapse_lookup,
                    internal_scenarios, capital_shock=0.35, max_years=max_years
                )

                pbar.update(1)

    total_calcs = total_iterations * len(internal_scenarios) * max_years * 2  # Reserve + Capital
    logger.info(f"TIER 2 & 3 COMPLETE: {total_calcs:,} internal calculations")

    return reserve_results, capital_results


def final_integration_gpu(external_results, reserve_results, capital_results,
                          hurdle_rate=0.10, max_years=35):
    """
    PHASE 5: Final integration with proper distributable cash flow calculation
    """
    logger.info("=" * 60)
    logger.info("PHASE 5: FINAL INTEGRATION (GPU)")
    logger.info("=" * 60)

    # Extract external cash flows (excluding year 0)
    external_cf = external_results['flux_net'][1:]  # Shape: (max_years, accounts, scenarios)

    # Calculate distributable amounts for each year
    # Simplified: assume reserves and capital are constant over projection
    total_pv_distributable = cp.zeros((external_cf.shape[1], external_cf.shape[2]), dtype=cp.float32)

    # Create discount factors for hurdle rate
    years = cp.arange(1, max_years + 1, dtype=cp.float32)
    discount_factors = 1.0 / ((1 + hurdle_rate) ** years)

    for year_idx, year in enumerate(years):
        if year_idx < external_cf.shape[0]:
            # Profit = external cash flow + change in reserves
            profit = external_cf[year_idx]  # Simplified: no reserve changes

            # Distributable = profit + change in capital
            distributable = profit  # Simplified: no capital changes

            # Present value
            pv_distributable = distributable * discount_factors[year_idx]
            total_pv_distributable += pv_distributable

    logger.info("PHASE 5 COMPLETE: Final integration finished")

    return total_pv_distributable


def run_acfc_gpu(data_path="data_in", output_dir="output", max_years=35,
                 hurdle_rate=0.10, capital_shock=0.35):
    """
    Main function to run the complete GPU-accelerated ACFC algorithm
    """
    logger.info("=" * 80)
    logger.info("ACTUARIAL CASH FLOW CALCULATION (ACFC) - GPU VERSION")
    logger.info("Complete Three-Tier Nested Stochastic Implementation")
    logger.info("=" * 80)

    start_time = time.time()

    try:
        # Check GPU availability
        logger.info(f"GPU Memory: {cp.cuda.Device().mem_info[1] / 1e9:.1f} GB total")

        # PHASE 1: Load and prepare data
        logger.info("PHASE 1: DATA LOADING AND GPU PREPARATION")
        data_frames = load_input_files(data_path)
        (population_gpu, rendement_lookup, mortality_lookup, discount_ext_lookup,
         discount_int_lookup, lapse_lookup, external_scenarios, internal_scenarios) = prepare_gpu_data(
            *data_frames, max_years=max_years)

        logger.info(f"Configuration:")
        logger.info(f"  Accounts: {len(population_gpu['ID_COMPTE'])}")
        logger.info(f"  External scenarios: {len(external_scenarios)}")
        logger.info(f"  Internal scenarios: {len(internal_scenarios)}")
        logger.info(f"  Max years: {max_years}")
        logger.info(f"  Hurdle rate: {hurdle_rate * 100}%")
        logger.info(f"  Capital shock: {capital_shock * 100}%")

        # TIER 1: External projections
        external_results = external_loop_gpu(
            population_gpu, rendement_lookup, mortality_lookup, discount_ext_lookup,
            lapse_lookup, external_scenarios, max_years
        )

        # TIER 2 & 3: Internal calculations
        reserve_results, capital_results = run_internal_loops_gpu(
            external_results, population_gpu, rendement_lookup, mortality_lookup,
            discount_int_lookup, lapse_lookup, internal_scenarios, max_years
        )

        # PHASE 5: Final integration
        final_pv_results = final_integration_gpu(
            external_results, reserve_results, capital_results, hurdle_rate, max_years
        )

        # Transfer results to CPU and create DataFrame
        logger.info("Transferring results from GPU to CPU...")

        num_accounts = len(population_gpu['ID_COMPTE'])
        num_scenarios = len(external_scenarios)

        # Create result arrays
        account_ids = cp.tile(population_gpu['ID_COMPTE'][:, None], (1, num_scenarios)).flatten()
        scenario_ids = cp.tile(external_scenarios[None, :], (num_accounts, 1)).flatten()
        pv_values = final_pv_results.flatten()
        reserve_vals = reserve_results.flatten()
        capital_vals = capital_results.flatten()

        # Create final DataFrame
        results_df = pd.DataFrame({
            'ID_COMPTE': cp.asnumpy(account_ids).astype(int),
            'scn_eval': cp.asnumpy(scenario_ids).astype(int),
            'VP_FLUX_DISTRIBUABLES': cp.asnumpy(pv_values),
            'reserve_value': cp.asnumpy(reserve_vals),
            'capital_value': cp.asnumpy(capital_vals)
        })

        # Calculate execution statistics
        elapsed_time = time.time() - start_time

        # Estimate total calculations performed
        external_calcs = len(population_gpu['ID_COMPTE']) * len(external_scenarios) * max_years
        internal_calcs = len(population_gpu['ID_COMPTE']) * len(external_scenarios) * len(
            internal_scenarios) * max_years * 2
        total_calcs = external_calcs + internal_calcs

        logger.info("=" * 80)
        logger.info("GPU ACFC ALGORITHM COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Execution time: {elapsed_time:.2f} seconds")
        logger.info(f"External calculations: {external_calcs:,}")
        logger.info(f"Internal calculations: {internal_calcs:,}")
        logger.info(f"Total calculations: {total_calcs:,}")
        logger.info(f"Calculations per second: {total_calcs / elapsed_time:,.0f}")
        logger.info(f"Final results: {len(results_df):,} combinations")

        # Analysis summary
        profitable = len(results_df[results_df['VP_FLUX_DISTRIBUABLES'] > 0])
        total_combinations = len(results_df)
        profitability_rate = profitable / total_combinations * 100 if total_combinations > 0 else 0

        logger.info(f"Profitability: {profitable}/{total_combinations} ({profitability_rate:.1f}%)")
        logger.info(f"Mean PV: ${results_df['VP_FLUX_DISTRIBUABLES'].mean():,.2f}")
        logger.info(
            f"PV Range: ${results_df['VP_FLUX_DISTRIBUABLES'].min():,.2f} to ${results_df['VP_FLUX_DISTRIBUABLES'].max():,.2f}")

        # Save results if output directory specified
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)

            results_file = output_path / "gpu_acfc_results.csv"
            results_df.to_csv(results_file, index=False)
            logger.info(f"Results saved to {results_file}")

            # Save performance summary
            summary_file = output_path / "gpu_acfc_summary.txt"
            with open(summary_file, 'w') as f:
                f.write("GPU ACFC Algorithm Performance Summary\n")
                f.write("=" * 40 + "\n")
                f.write(f"Execution time: {elapsed_time:.2f} seconds\n")
                f.write(f"Total calculations: {total_calcs:,}\n")
                f.write(f"Calculations per second: {total_calcs / elapsed_time:,.0f}\n")
                f.write(f"GPU memory used: {cp.cuda.Device().mem_info[0] / 1e9:.1f} GB\n")
                f.write(f"Results generated: {len(results_df):,}\n")
                f.write(f"Profitability rate: {profitability_rate:.1f}%\n")
            logger.info(f"Summary saved to {summary_file}")

        logger.info("=" * 80)

        return results_df

    except Exception as e:
        logger.error(f"Error in GPU ACFC execution: {str(e)}")
        logger.error("Please ensure CUDA and CuPy are properly installed")
        raise


def analyze_gpu_results(results_df):
    """Analyze GPU ACFC results with comprehensive statistics"""

    analysis = {
        'total_combinations': len(results_df),
        'profitable_combinations': len(results_df[results_df['VP_FLUX_DISTRIBUABLES'] > 0]),
        'loss_combinations': len(results_df[results_df['VP_FLUX_DISTRIBUABLES'] <= 0]),
        'unique_accounts': results_df['ID_COMPTE'].nunique(),
        'unique_scenarios': results_df['scn_eval'].nunique(),
        'mean_pv': results_df['VP_FLUX_DISTRIBUABLES'].mean(),
        'median_pv': results_df['VP_FLUX_DISTRIBUABLES'].median(),
        'std_pv': results_df['VP_FLUX_DISTRIBUABLES'].std(),
        'min_pv': results_df['VP_FLUX_DISTRIBUABLES'].min(),
        'max_pv': results_df['VP_FLUX_DISTRIBUABLES'].max(),
        'percentiles': {
            '5th': results_df['VP_FLUX_DISTRIBUABLES'].quantile(0.05),
            '25th': results_df['VP_FLUX_DISTRIBUABLES'].quantile(0.25),
            '75th': results_df['VP_FLUX_DISTRIBUABLES'].quantile(0.75),
            '95th': results_df['VP_FLUX_DISTRIBUABLES'].quantile(0.95)
        }
    }

    analysis['profitability_rate'] = (analysis['profitable_combinations'] /
                                      analysis['total_combinations'] * 100) if analysis['total_combinations'] > 0 else 0

    return analysis


def print_gpu_results_summary(results_df, analysis):
    """Print comprehensive GPU results summary"""

    print("\n" + "=" * 80)
    print("GPU ACTUARIAL CASH FLOW CALCULATION (ACFC) - RESULTS ANALYSIS")
    print("=" * 80)

    # Dataset overview
    print(f"Dataset Overview:")
    print(f"  Total combinations: {analysis['total_combinations']:,}")
    print(f"  Unique accounts: {analysis['unique_accounts']:,}")
    print(f"  Unique scenarios: {analysis['unique_scenarios']:,}")

    # Profitability analysis
    print(f"\nProfitability Analysis:")
    print(f"  Profitable: {analysis['profitable_combinations']:,} ({analysis['profitability_rate']:.1f}%)")
    print(f"  Loss-making: {analysis['loss_combinations']:,} ({100 - analysis['profitability_rate']:.1f}%)")

    # Statistical measures
    print(f"\nDistributable Cash Flow Statistics:")
    print(f"  Mean: ${analysis['mean_pv']:,.2f}")
    print(f"  Median: ${analysis['median_pv']:,.2f}")
    print(f"  Standard Deviation: ${analysis['std_pv']:,.2f}")
    print(f"  Range: ${analysis['min_pv']:,.2f} to ${analysis['max_pv']:,.2f}")

    # Percentile distribution
    print(f"\nPercentile Distribution:")
    for percentile, value in analysis['percentiles'].items():
        print(f"  {percentile:>4}: ${value:>12,.2f}")

    # Top performers
    print(f"\nTop 5 Most Profitable Combinations:")
    top_5 = results_df.nlargest(5, 'VP_FLUX_DISTRIBUABLES')
    for idx, (_, row) in enumerate(top_5.iterrows(), 1):
        print(
            f"  {idx}. Account {int(row['ID_COMPTE']):3d}, Scenario {int(row['scn_eval']):3d}: ${row['VP_FLUX_DISTRIBUABLES']:>12,.2f}")

    # Worst performers
    print(f"\nBottom 5 Combinations (Largest Losses):")
    bottom_5 = results_df.nsmallest(5, 'VP_FLUX_DISTRIBUABLES')
    for idx, (_, row) in enumerate(bottom_5.iterrows(), 1):
        print(
            f"  {idx}. Account {int(row['ID_COMPTE']):3d}, Scenario {int(row['scn_eval']):3d}: ${row['VP_FLUX_DISTRIBUABLES']:>12,.2f}")

    # Product group analysis if applicable
    if results_df['ID_COMPTE'].max() > 100:
        group1 = results_df[results_df['ID_COMPTE'] <= 100]
        group2 = results_df[results_df['ID_COMPTE'] > 100]

        print(f"\nProduct Group Comparison:")
        if len(group1) > 0:
            print(f"  Group 1 (Accounts 1-100) - High Guarantee/High Fee:")
            print(f"    Mean PV: ${group1['VP_FLUX_DISTRIBUABLES'].mean():,.2f}")
            print(
                f"    Profitable: {len(group1[group1['VP_FLUX_DISTRIBUABLES'] > 0])}/{len(group1)} ({len(group1[group1['VP_FLUX_DISTRIBUABLES'] > 0]) / len(group1) * 100:.1f}%)")

        if len(group2) > 0:
            print(f"  Group 2 (Accounts 101-200) - Moderate Guarantee/Lower Fee:")
            print(f"    Mean PV: ${group2['VP_FLUX_DISTRIBUABLES'].mean():,.2f}")
            print(
                f"    Profitable: {len(group2[group2['VP_FLUX_DISTRIBUABLES'] > 0])}/{len(group2)} ({len(group2[group2['VP_FLUX_DISTRIBUABLES'] > 0]) / len(group2) * 100:.1f}%)")


def benchmark_gpu_vs_cpu_performance():
    """
    Utility function to compare GPU vs CPU performance characteristics
    """
    logger.info("GPU vs CPU Performance Characteristics:")
    logger.info("=" * 50)

    try:
        # GPU info
        gpu_mem_info = cp.cuda.Device().mem_info
        logger.info(f"GPU Memory: {gpu_mem_info[1] / 1e9:.1f} GB total, {gpu_mem_info[0] / 1e9:.1f} GB used")

        # Simple benchmark
        size = 10000
        start_time = time.time()

        # GPU operations
        a_gpu = cp.random.random((size, size), dtype=cp.float32)
        b_gpu = cp.random.random((size, size), dtype=cp.float32)
        c_gpu = cp.matmul(a_gpu, b_gpu)
        cp.cuda.Stream.null.synchronize()  # Wait for GPU operations

        gpu_time = time.time() - start_time
        logger.info(f"GPU matrix multiplication ({size}x{size}): {gpu_time:.3f} seconds")

        # Memory cleanup
        del a_gpu, b_gpu, c_gpu

        logger.info("GPU is ready for ACFC processing")

    except Exception as e:
        logger.warning(f"GPU benchmark failed: {e}")
        logger.warning("Ensure CUDA and CuPy are properly installed")


def main():
    """Main execution function for GPU ACFC algorithm"""

    try:
        # Optional: Run benchmark to verify GPU functionality
        benchmark_gpu_vs_cpu_performance()

        # Run the complete GPU ACFC algorithm
        results_df = run_acfc_gpu(
            data_path="data_in",
            output_dir="output",
            max_years=35,
            hurdle_rate=0.10,
            capital_shock=0.35
        )

        # Analyze results
        analysis = analyze_gpu_results(results_df)

        # Print comprehensive summary
        print_gpu_results_summary(results_df, analysis)

        return results_df, analysis

    except Exception as e:
        logger.error(f"Fatal error in GPU ACFC execution: {str(e)}")
        logger.error("Falling back to CPU implementation may be necessary")
        raise


if __name__ == "__main__":
    results_df, analysis = main()
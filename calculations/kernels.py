import math

from numba import cuda

from calculations.constants import (
    DEFAULT_MORTALITY_RATE, DEFAULT_RETURN_RATE, DEFAULT_FORWARD_RATE,
    DEFAULT_LAPSE_RATE_TOT, DEFAULT_LAPSE_RATE_PART, DEFAULT_LAPSE_FACT_DIM,
    DEFAULT_FERR_MIN_RATE, DEFAULT_COMMISSION_MAINTIEN,
    SHOCK_FACTOR, NUM_CHOCS,
    VM_VG_RATIO_MAX, VM_VG_RATIO_LEVEL1_THRESHOLD, VM_VG_RATIO_LEVEL2_THRESHOLD, VM_VG_RATIO_LEVEL3_DIVISOR,
    LAPSE_LEVEL_1, LAPSE_LEVEL_2, LAPSE_LEVEL_3,
    MORTALITY_AGE_ADJUSTMENT_THRESHOLD, MAX_WITHDRAWAL, MIN_GUARANTEE_VALUE,
    DEFAULT_FREQ_EVAL, METRICS_RESERVE_IDX, METRICS_CAPITAL_IDX,
    # Account data array indices (module-level constants for Numba)
    ACCOUNT_IDX_ANNEE_EVALUATION_INI, ACCOUNT_IDX_MOIS_EVALUATION_INI,
    ACCOUNT_IDX_ANNEE_NAIS, ACCOUNT_IDX_MOIS_NAIS,
    ACCOUNT_IDX_I_SEXE, ACCOUNT_IDX_I_PRODUIT_REGR, ACCOUNT_IDX_ID_PRODUIT,
    ACCOUNT_IDX_ID_LAPSE, ACCOUNT_IDX_I_REGIME_2, ACCOUNT_IDX_ID_DEPOT, ACCOUNT_IDX_ID_ACQUI,
    ACCOUNT_IDX_AGE_ECH_MIN, ACCOUNT_IDX_AGE_FIN_CONTRAT, ACCOUNT_IDX_AGE_DECAISSEMENT,
    ACCOUNT_IDX_MT_VM, ACCOUNT_IDX_MT_GAR_DECES, ACCOUNT_IDX_MT_GAR_ECH,
    ACCOUNT_IDX_MT_SRG, ACCOUNT_IDX_MT_BCB,
    ACCOUNT_IDX_MT_DEX, ACCOUNT_IDX_MT_MM, ACCOUNT_IDX_MT_TSX,
    ACCOUNT_IDX_MT_SP500, ACCOUNT_IDX_MT_EAFE,
    ACCOUNT_IDX_MT_BONI_DECES, ACCOUNT_IDX_MT_MRV_MRG_MRA, ACCOUNT_IDX_TAUX_MRV_MRG_MRA,
    ACCOUNT_IDX_ANNEE_ECH, ACCOUNT_IDX_MOIS_ECH,
    ACCOUNT_IDX_PC_HONORAIRES_GEST, ACCOUNT_IDX_PC_FRAIS_GARANTIE,
    ACCOUNT_IDX_PC_GAR_DECES_1, ACCOUNT_IDX_PC_BONI_DECES,
    ACCOUNT_IDX_PC_RFG, ACCOUNT_IDX_PC_REVENU_FDS,
    ACCOUNT_IDX_PC_GAR_ECH, ACCOUNT_IDX_PC_GAR_ECH_DEP_FUT,
    ACCOUNT_IDX_AJUSTEMENT_COMMISSION, ACCOUNT_IDX_MT_RF, ACCOUNT_IDX_I_FRAIS_SUR_SRG,
    ACCOUNT_IDX_MT_VM_ORIG, ACCOUNT_IDX_ANNEE_COTIS, ACCOUNT_IDX_MOIS_COTIS,
    # State tensor indices (module-level constants for Numba)
    STATE_IDX_MT_VM, STATE_IDX_MT_GAR_DECES, STATE_IDX_MT_GAR_ECH,
    STATE_IDX_MT_SRG, STATE_IDX_AGE, STATE_IDX_TX_SURVIE,
    STATE_IDX_MT_DEX, STATE_IDX_MT_MM, STATE_IDX_MT_TSX,
    STATE_IDX_MT_SP500, STATE_IDX_MT_EAFE, STATE_IDX_MT_BONI_DECES,
    STATE_IDX_SIZE,
    # External debug indices (module-level constants for Numba)
    EXT_DEBUG_IDX_VM, EXT_DEBUG_IDX_AGE, EXT_DEBUG_IDX_QX,
    EXT_DEBUG_IDX_LAPSE_TOT, EXT_DEBUG_IDX_LAPSE_PART, EXT_DEBUG_IDX_TX_SURVIE,
    EXT_DEBUG_IDX_FORWARD_RATE, EXT_DEBUG_IDX_REND_SP500, EXT_DEBUG_IDX_REND_TSX,
    EXT_DEBUG_IDX_REND_EAFE, EXT_DEBUG_IDX_REND_DEX, EXT_DEBUG_IDX_RETRAIT,
    EXT_DEBUG_IDX_PREST_DECES, EXT_DEBUG_IDX_PRIMES_GARANTIES, EXT_DEBUG_IDX_VM_VG_RATIO,
    EXT_DEBUG_IDX_SIZE,
    # Internal debug indices (module-level constants for Numba)
    INT_DEBUG_IDX_START_VM, INT_DEBUG_IDX_VM_CHOC, INT_DEBUG_IDX_AVG_PV_FLUX,
    INT_DEBUG_IDX_RESERVE, INT_DEBUG_IDX_CAPITAL, INT_DEBUG_IDX_START_TX_SURVIE,
    INT_DEBUG_IDX_START_AGE, INT_DEBUG_IDX_CURR_VM, INT_DEBUG_IDX_FEES,
    INT_DEBUG_IDX_PV_PATH, INT_DEBUG_IDX_R_PORTFOLIO, INT_DEBUG_IDX_FWD_RATE,
    INT_DEBUG_IDX_SIZE,
    INT_TS_DEBUG_IDX_CURR_VM, INT_TS_DEBUG_IDX_FEES, INT_TS_DEBUG_IDX_PV_PATH,
    INT_TS_DEBUG_IDX_R_PORTFOLIO, INT_TS_DEBUG_IDX_FWD_RATE, INT_TS_DEBUG_IDX_DF,
    INT_TS_DEBUG_IDX_SIZE,
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
)

@cuda.jit
def external_generator_kernel(
        account_data,        # AccountData: (n_accounts, n_fields)
        n_scenarios,         # int
        n_years,             # int
        freq_eval,           # float
        mortality_lookup,    # MortalityLookup: (sex, age, year, product)
        returns_lookups,     # ReturnsLookups: 7 arrays
        lapse_lookups,       # LapseLookups: 6 arrays
        policy_lookups,      # PolicyLookups: 5 arrays
        commission_lookups,  # CommissionLookups: 6 arrays
        coussins_lookups,    # CoussinsLookups: 16 arrays
        output_states,       # StatesTensor: (batch, scenarios, years, STATE_SIZE)
        output_cashflows,    # CashflowsTensor: (batch, scenarios, years, 1)
        output_flux_agg,     # FluxAggTensor: (n_years+1, 13, FLUX_COMP_IDX_SIZE)
        debug_output=None,   # Optional: (EXT_DEBUG_SIZE,) - single row for filtered debug
        debug_account=-1,    # Account index to debug (-1 = disabled)
        debug_scenario=-1,   # Scenario index to debug (-1 = disabled)
        debug_year=-1,       # Year (an_eval) to debug (-1 = disabled)
        debug_month=-1,      # Month (mois_eval) to debug (-1 = disabled)
):
    """
    KERNEL A: EXTERNAL SCENARIO GENERATOR (Tier 1)

    Runs the external (real-world) scenarios and saves intermediate states at each timestep.
    These states will be used by Kernel B to perform nested valuations.

    Args:
        account_data: Account attributes array
        n_scenarios: Number of external scenarios
        n_years: Projection horizon
        freq_eval: Evaluation frequency per year
        mortality_lookup: Mortality rates by (sex, age, year, product)
        returns_lookups: (forward_rate, ajust_forward, rend_dex, rend_mm, rend_tsx, rend_sp500, rend_eafe)
        lapse_lookups: (min_ferr, lapse_part_min, lapse_part_max, lapse_tot_min, lapse_tot_max, lapse_tot_fact)
        policy_lookups: (deposits_pc, deposits_var, deposits_age_max, deposits_i_even, fees)
        commission_lookups: (acq_vente_rf, acq_vente_ac, acq_maintien_rf, acq_maintien_ac, acq_frais_ac, acq_frais_rf)
        output_states: Output state tensor
        output_cashflows: Output cashflow tensor
    """
    # Unpack lookup tuples
    forward_rate, ajust_forward, rend_dex, rend_mm, rend_tsx, rend_sp500, rend_eafe = returns_lookups
    min_ferr_lookup, lapse_part_min, lapse_part_max, lapse_tot_min, lapse_tot_max, lapse_tot_fact = lapse_lookups
    deposits_pc, deposits_var, deposits_age_max, deposits_i_even, fees_lookup = policy_lookups
    acq_vente_rf, acq_vente_ac, acq_maintien_rf, acq_maintien_ac, acq_frais_ac, acq_frais_rf = commission_lookups
    (base_passif, tx_passif,
     base_credit, tx_credit,
     base_marche, tx_marche,
     base_depense, tx_depense,
     base_decheance, tx_decheance,
     base_mortalite, tx_mortalite,
     base_depot, tx_depot,
     facteur_age_80, facteur_age_90) = coussins_lookups

    # Get global thread ID
    account_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    scenario_idx = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # Boundary check
    if account_idx >= account_data.shape[0] or scenario_idx >= n_scenarios:
        return

    # Load account data into registers
    acc = account_data[account_idx]

    # Account static data using module-level constants (Numba-compatible)
    ANNEE_EVALUATION_INI = int(acc[ACCOUNT_IDX_ANNEE_EVALUATION_INI])
    MOIS_EVALUATION_INI = int(acc[ACCOUNT_IDX_MOIS_EVALUATION_INI])
    ANNEE_NAIS = int(acc[ACCOUNT_IDX_ANNEE_NAIS])
    MOIS_NAIS = int(acc[ACCOUNT_IDX_MOIS_NAIS])
    I_SEXE = int(acc[ACCOUNT_IDX_I_SEXE])
    I_PRODUIT_REGR = int(acc[ACCOUNT_IDX_I_PRODUIT_REGR])
    ID_PRODUIT = int(acc[ACCOUNT_IDX_ID_PRODUIT])
    ID_LAPSE = int(acc[ACCOUNT_IDX_ID_LAPSE])
    I_REGIME_2 = int(acc[ACCOUNT_IDX_I_REGIME_2])
    ID_DEPOT = int(acc[ACCOUNT_IDX_ID_DEPOT])
    ID_ACQUI = int(acc[ACCOUNT_IDX_ID_ACQUI])
    AGE_ECH_MIN = int(acc[ACCOUNT_IDX_AGE_ECH_MIN])
    AGE_FIN_CONTRAT = int(acc[ACCOUNT_IDX_AGE_FIN_CONTRAT])
    AGE_DECAISSEMENT = int(acc[ACCOUNT_IDX_AGE_DECAISSEMENT])

    # Initialize state variables using module-level constants
    MT_VM_PROJ = acc[ACCOUNT_IDX_MT_VM]
    MT_GAR_DECES_PROJ = acc[ACCOUNT_IDX_MT_GAR_DECES]
    MT_GAR_ECH_PROJ = acc[ACCOUNT_IDX_MT_GAR_ECH]
    MT_SRG_PROJ = acc[ACCOUNT_IDX_MT_SRG]
    MT_BCB_PROJ = acc[ACCOUNT_IDX_MT_BCB]
    MT_DEX_PROJ = acc[ACCOUNT_IDX_MT_DEX]
    MT_MM_PROJ = acc[ACCOUNT_IDX_MT_MM]
    MT_TSX_PROJ = acc[ACCOUNT_IDX_MT_TSX]
    MT_SP500_PROJ = acc[ACCOUNT_IDX_MT_SP500]
    MT_EAFE_PROJ = acc[ACCOUNT_IDX_MT_EAFE]
    MT_BONI_DECES_PROJ = acc[ACCOUNT_IDX_MT_BONI_DECES]
    MT_MRV_MRG_MRA_PROJ = acc[ACCOUNT_IDX_MT_MRV_MRG_MRA]
    TAUX_MRV_MRG_MRA_PROJ = acc[ACCOUNT_IDX_TAUX_MRV_MRG_MRA]
    MT_MIN_FERR_PROJ = 0.0

    TX_SURVIE = 1.0

    PC_HONORAIRES_GEST = acc[ACCOUNT_IDX_PC_HONORAIRES_GEST]
    PC_FRAIS_GARANTIE = acc[ACCOUNT_IDX_PC_FRAIS_GARANTIE]
    PC_GAR_DECES_1 = acc[ACCOUNT_IDX_PC_GAR_DECES_1]
    PC_BONI_DECES = acc[ACCOUNT_IDX_PC_BONI_DECES]
    PC_RFG = acc[ACCOUNT_IDX_PC_RFG]
    PC_REVENU_FDS = acc[ACCOUNT_IDX_PC_REVENU_FDS]
    PC_GAR_ECH = acc[ACCOUNT_IDX_PC_GAR_ECH]
    PC_GAR_ECH_DEP_FUT = acc[ACCOUNT_IDX_PC_GAR_ECH_DEP_FUT]
    MT_VM_ORIG = acc[ACCOUNT_IDX_MT_VM_ORIG]

    AJUSTEMENT_COMMISSION = acc[ACCOUNT_IDX_AJUSTEMENT_COMMISSION]
    MT_RF = acc[ACCOUNT_IDX_MT_RF]
    I_FRAIS_SUR_SRG = int(acc[ACCOUNT_IDX_I_FRAIS_SUR_SRG])

    ANNEE_COTIS = int(acc[ACCOUNT_IDX_ANNEE_COTIS]) if acc[ACCOUNT_IDX_ANNEE_COTIS] > 0 else ANNEE_EVALUATION_INI
    MOIS_COTIS = int(acc[ACCOUNT_IDX_MOIS_COTIS]) if acc[ACCOUNT_IDX_MOIS_COTIS] > 0 else 1

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

            # === COMPLETE SAS PROJECTION LOGIC ===

            # Initialize year 0 (SAS lines 241-265)
            if an_eval == 0:
                MT_SP500_PROJ = acc[ACCOUNT_IDX_MT_SP500]
                MT_TSX_PROJ = acc[ACCOUNT_IDX_MT_TSX]
                MT_EAFE_PROJ = acc[ACCOUNT_IDX_MT_EAFE]
                MT_DEX_PROJ = acc[ACCOUNT_IDX_MT_DEX]
                MT_MM_PROJ = acc[ACCOUNT_IDX_MT_MM]
                MT_VM_PROJ = MT_SP500_PROJ + MT_TSX_PROJ + MT_EAFE_PROJ + MT_DEX_PROJ + MT_MM_PROJ
                TX_ACTUALISATION = 1.0
                continue

            # Mortality lookup (SAS lines 420-429)
            month_diff = MOIS_NAIS - mois_eval
            if month_diff <= 0:
                month_diff += 12
            age_mort = age + 1 if month_diff <= 6 else age
            age_mort = min(age_mort, 120)  # Cap at max mortality table age

            if (I_SEXE < mortality_lookup.shape[0] and
                    age_mort < mortality_lookup.shape[1] and
                    annee_reelle < mortality_lookup.shape[2] and
                    I_PRODUIT_REGR < mortality_lookup.shape[3]):
                qx = mortality_lookup[I_SEXE, age_mort, annee_reelle, I_PRODUIT_REGR]
            else:
                qx = DEFAULT_MORTALITY_RATE
            qx = 1.0 - math.pow(1.0 - qx, (1.0 / freq_eval * AJUST_NOUV_AFFAIRES))

            # Returns lookup and application (SAS lines 305-327)
            if (scn_eval < forward_rate.shape[0] and
                    an_eval < forward_rate.shape[1] and
                    mois_eval < forward_rate.shape[2]):
                FORWARD_RATE = forward_rate[scn_eval, an_eval, mois_eval]
                AJUST_FORWARD_RATE_VM_0 = ajust_forward[scn_eval, an_eval, mois_eval]
                r_dex = rend_dex[scn_eval, an_eval, mois_eval]
                r_mm = rend_mm[scn_eval, an_eval, mois_eval]
                r_tsx = rend_tsx[scn_eval, an_eval, mois_eval]
                r_sp500 = rend_sp500[scn_eval, an_eval, mois_eval]
                r_eafe = rend_eafe[scn_eval, an_eval, mois_eval]
            else:
                FORWARD_RATE = DEFAULT_FORWARD_RATE
                AJUST_FORWARD_RATE_VM_0 = 0.0
                r_dex = r_mm = r_tsx = r_sp500 = r_eafe = DEFAULT_RETURN_RATE

            # Adjust forward rate for VM=0 case (SAS line 309)
            if MT_VM_PROJ == 0:
                FORWARD_RATE += AJUST_FORWARD_RATE_VM_0

            # Apply returns to each asset (SAS lines 316-321)
            MT_SP500_PROJ *= math.exp(r_sp500 * AJUST_NOUV_AFFAIRES)
            MT_TSX_PROJ *= math.exp(r_tsx * AJUST_NOUV_AFFAIRES)
            MT_EAFE_PROJ *= math.exp(r_eafe * AJUST_NOUV_AFFAIRES)
            MT_DEX_PROJ *= math.exp(r_dex * AJUST_NOUV_AFFAIRES)
            MT_MM_PROJ *= math.exp(r_mm * AJUST_NOUV_AFFAIRES)

            MT_VM_AV_RETRAIT_FRAIS = (MT_SP500_PROJ + MT_TSX_PROJ + MT_EAFE_PROJ +
                                      MT_DEX_PROJ + MT_MM_PROJ)

            # Update discount factor (SAS lines 326-327)
            TX_ACTUALISATION *= math.exp(-FORWARD_RATE * AJUST_NOUV_AFFAIRES)

            # Complete lapse calculation (SAS lines 330-413)
            lapse = 0.0
            if MT_VM_PROJ == 0:
                # No lapse for zero VM RGS products
                VM_VG_RATIO = 0.0
                LAPSE_NIV_TOT = 0
                LAPSE_NIV_PART = 0
                LAPSE_TOT = 0.0
                LAPSE_PART = 0.0
            else:
                # Calculate VM/VG ratio (SAS lines 350-355)
                vm_mid_period = (MT_VM_PROJ + MT_VM_AV_RETRAIT_FRAIS) / 2.0
                pc_gar_ech_ratio = PC_GAR_ECH / max(MT_GAR_ECH_PROJ, MIN_GUARANTEE_VALUE) if MT_GAR_ECH_PROJ > 0 else 1.0
                pc_gar_deces_ratio = PC_GAR_DECES_1 / max(MT_BONI_DECES_PROJ + MT_GAR_DECES_PROJ, MIN_GUARANTEE_VALUE) if (MT_BONI_DECES_PROJ + MT_GAR_DECES_PROJ) > 0 else 1.0
                srg_ratio = 1.0 / max(MT_SRG_PROJ, MIN_GUARANTEE_VALUE) if MT_SRG_PROJ > 0 else 1.0

                min_ratio = min(pc_gar_ech_ratio, pc_gar_deces_ratio, srg_ratio)
                VM_VG_RATIO = min(VM_VG_RATIO_MAX, vm_mid_period * min_ratio)

                # Determine lapse levels (SAS lines 363-365, 392-394)
                if VM_VG_RATIO <= VM_VG_RATIO_LEVEL1_THRESHOLD:
                    LAPSE_NIV_TOT = LAPSE_LEVEL_1
                    LAPSE_NIV_PART = LAPSE_LEVEL_1
                elif VM_VG_RATIO <= VM_VG_RATIO_LEVEL2_THRESHOLD:
                    LAPSE_NIV_TOT = LAPSE_LEVEL_2
                    LAPSE_NIV_PART = LAPSE_LEVEL_2
                else:
                    LAPSE_NIV_TOT = LAPSE_LEVEL_3
                    LAPSE_NIV_PART = LAPSE_LEVEL_3

                # Lookup lapse rates with bounds checking
                if (duree_max10 < lapse_tot_min.shape[0] and ID_LAPSE < lapse_tot_min.shape[1] and
                        LAPSE_NIV_TOT < lapse_tot_min.shape[2]):
                    tx_lapse_tot_min = lapse_tot_min[duree_max10, ID_LAPSE, LAPSE_NIV_TOT]
                    tx_lapse_tot_max = lapse_tot_max[duree_max10, ID_LAPSE, LAPSE_NIV_TOT]
                    fact_dim = lapse_tot_fact[duree_max10, ID_LAPSE, LAPSE_NIV_TOT]
                else:
                    tx_lapse_tot_min = tx_lapse_tot_max = DEFAULT_LAPSE_RATE_TOT
                    fact_dim = DEFAULT_LAPSE_FACT_DIM

                # Calculate total lapse (SAS lines 374-383)
                if tx_lapse_tot_min == tx_lapse_tot_max:
                    LAPSE_TOT = tx_lapse_tot_min
                else:
                    if LAPSE_NIV_TOT == LAPSE_LEVEL_1:
                        lapse_interp = (VM_VG_RATIO - 0.0) / VM_VG_RATIO_LEVEL1_THRESHOLD
                    elif LAPSE_NIV_TOT == LAPSE_LEVEL_2:
                        lapse_interp = (VM_VG_RATIO - VM_VG_RATIO_LEVEL1_THRESHOLD) / (VM_VG_RATIO_LEVEL2_THRESHOLD - VM_VG_RATIO_LEVEL1_THRESHOLD)
                    else:
                        lapse_interp = (VM_VG_RATIO - VM_VG_RATIO_LEVEL2_THRESHOLD) / VM_VG_RATIO_LEVEL3_DIVISOR
                    LAPSE_TOT = lapse_interp * (tx_lapse_tot_max - tx_lapse_tot_min) + tx_lapse_tot_min

                # Age adjustment for FERR
                age_factor = fact_dim if age >= AGE_DECAISSEMENT else 1.0
                LAPSE_TOT *= age_factor

                # Partial lapse calculation (similar logic)
                if (age < lapse_part_min.shape[0] and ID_LAPSE < lapse_part_min.shape[1] and
                        I_REGIME_2 < lapse_part_min.shape[2] and LAPSE_NIV_PART < lapse_part_min.shape[3]):
                    tx_lapse_part_min = lapse_part_min[age, ID_LAPSE, I_REGIME_2, LAPSE_NIV_PART]
                    tx_lapse_part_max = lapse_part_max[age, ID_LAPSE, I_REGIME_2, LAPSE_NIV_PART]
                else:
                    tx_lapse_part_min = tx_lapse_part_max = DEFAULT_LAPSE_RATE_PART

                if tx_lapse_part_min == tx_lapse_part_max:
                    LAPSE_PART = tx_lapse_part_min
                else:
                    if LAPSE_NIV_PART == LAPSE_LEVEL_1:
                        lapse_interp = (VM_VG_RATIO - 0.0) / VM_VG_RATIO_LEVEL1_THRESHOLD
                    elif LAPSE_NIV_PART == LAPSE_LEVEL_2:
                        lapse_interp = (VM_VG_RATIO - VM_VG_RATIO_LEVEL1_THRESHOLD) / (VM_VG_RATIO_LEVEL2_THRESHOLD - VM_VG_RATIO_LEVEL1_THRESHOLD)
                    else:
                        lapse_interp = (VM_VG_RATIO - VM_VG_RATIO_LEVEL2_THRESHOLD) / VM_VG_RATIO_LEVEL3_DIVISOR
                    LAPSE_PART = lapse_interp * (tx_lapse_part_max - tx_lapse_part_min) + tx_lapse_part_min

            # Convert annual rates to frequency basis (SAS line 412)
            lapse = 1.0 - math.pow(1.0 - LAPSE_TOT - LAPSE_PART, 1.0 / freq_eval * AJUST_NOUV_AFFAIRES)

            # Update survival probability (SAS lines 432-433)
            TX_SURVIE *= (1.0 - qx) * (1.0 - lapse)

            # Apply management fees (SAS line 537)
            MT_VM_AV_RETRAIT = MT_VM_AV_RETRAIT_FRAIS * math.exp(-PC_RFG / freq_eval * AJUST_NOUV_AFFAIRES)

            # Calculate guarantee fees (SAS lines 540-551)
            guarantee_fee_amount = 0.0
            if PC_FRAIS_GARANTIE > 0:
                base_fee_calc = MT_VM_AV_RETRAIT if I_FRAIS_SUR_SRG == 0 else MT_SRG_PROJ
                guarantee_fee_amount = base_fee_calc * PC_FRAIS_GARANTIE / freq_eval * AJUST_NOUV_AFFAIRES
                if guarantee_fee_amount > MT_VM_AV_RETRAIT:
                    guarantee_fee_amount = MT_VM_AV_RETRAIT
                MT_VM_AV_RETRAIT = max(MT_VM_AV_RETRAIT - guarantee_fee_amount, 0.0)

            PRIMES_GARANTIES = guarantee_fee_amount * TX_SURVIE_DEB

            # FERR minimum calculation (SAS lines 444-449)
            if age < min_ferr_lookup.shape[0]:
                min_ferr_rate = min_ferr_lookup[age]
            else:
                min_ferr_rate = DEFAULT_FERR_MIN_RATE

            if (an_eval == 1 and mois_eval == MOIS_EVALUATION_INI) or mois_eval == 12 // freq_eval:
                MT_MIN_FERR_PROJ = MT_VM_PROJ * min_ferr_rate

            # Calculate retirement age (SAS line 442)
            AGE_RETRAIT = age + 1

            # Withdrawals calculation (SAS lines 498-507)
            RETRAIT = 0.0
            if (AGE_RETRAIT >= AGE_DECAISSEMENT and
                not (AGE_RETRAIT == AGE_DECAISSEMENT and mois_eval >= MOIS_NAIS) and
                not (MT_VM_PROJ <= 0 and I_PRODUIT_REGR == 0)):

                # Simplified withdrawal calculation - use FERR minimum
                RETRAIT = MT_MIN_FERR_PROJ
                RETRAIT = min(RETRAIT, MAX_WITHDRAWAL) / freq_eval  # Monthly basis

            # Apply withdrawals to guarantees and VM (SAS lines 564-577)
            if MT_VM_AV_RETRAIT <= RETRAIT:
                MT_GAR_ECH_PROJ = 0.0
                MT_GAR_DECES_PROJ = 0.0
                MT_BONI_DECES_PROJ = 0.0
                MT_SRG_PROJ = 0.0
            else:
                withdrawal_factor = 1.0 - RETRAIT / MT_VM_AV_RETRAIT
                MT_GAR_ECH_PROJ *= withdrawal_factor
                MT_GAR_DECES_PROJ *= withdrawal_factor
                MT_BONI_DECES_PROJ *= withdrawal_factor
                MT_SRG_PROJ = max(MT_SRG_PROJ - RETRAIT, 0.0)

            # VM after withdrawals (SAS line 577)
            MT_VM_AP_RETRAIT = max(MT_VM_AV_RETRAIT - RETRAIT, 0.0)

            MT_VM_PROJ = MT_VM_AP_RETRAIT

            PREST_MRV = 0.0
            if I_PRODUIT_REGR == 1:
                diff = RETRAIT - MT_VM_AV_RETRAIT
                PREST_MRV = -diff * TX_SURVIE_DEB if diff > 0 else 0.0

            depot_futur = 0.0
            if (duree_max10 < deposits_pc.shape[0] and ID_DEPOT < deposits_pc.shape[1]):
                pc_depot_annuel = deposits_pc[duree_max10, ID_DEPOT]
                var_depot_fct = int(deposits_var[duree_max10, ID_DEPOT])
                age_max_depot = int(deposits_age_max[duree_max10, ID_DEPOT])
                i_even_cesse_depot = int(deposits_i_even[duree_max10, ID_DEPOT])
            else:
                pc_depot_annuel = 0.0
                var_depot_fct = 0
                age_max_depot = 999
                i_even_cesse_depot = 0

            age_retrait = age + 1
            if (pc_depot_annuel == 0.0 or
                (i_even_cesse_depot == 1 and age_retrait >= AGE_DECAISSEMENT) or
                (age_max_depot < age) or
                (MT_VM_PROJ <= 0 and I_PRODUIT_REGR == 0)):
                depot_futur = 0.0
            else:
                base_depot_calc = MT_VM_PROJ if var_depot_fct == 1 else (acc[ACCOUNT_IDX_MT_GAR_DECES] / max(PC_GAR_DECES_1, MIN_GUARANTEE_VALUE))
                depot_futur = base_depot_calc * pc_depot_annuel / freq_eval

            if depot_futur > 0.0 and MT_VM_PROJ > 0.0:
                MT_DEX_PROJ += depot_futur * (MT_DEX_PROJ / MT_VM_PROJ)
                MT_MM_PROJ += depot_futur * (MT_MM_PROJ / MT_VM_PROJ)
                MT_TSX_PROJ += depot_futur * (MT_TSX_PROJ / MT_VM_PROJ)
                MT_SP500_PROJ += depot_futur * (MT_SP500_PROJ / MT_VM_PROJ)
                MT_EAFE_PROJ += depot_futur * (MT_EAFE_PROJ / MT_VM_PROJ)
                MT_GAR_DECES_PROJ += depot_futur
                MT_GAR_ECH_PROJ += depot_futur * PC_GAR_ECH_DEP_FUT
                if MT_SRG_PROJ > 0.0:
                    MT_SRG_PROJ += depot_futur

            if MT_VM_AP_RETRAIT > 0.0:
                MT_VM_PROJ = MT_VM_AP_RETRAIT + depot_futur
            else:
                MT_VM_PROJ = MT_VM_AP_RETRAIT

            # Death benefits calculation (SAS line 598)
            claim = MT_GAR_DECES_PROJ + MT_BONI_DECES_PROJ - MT_VM_PROJ
            PREST_DECES = -qx * claim * TX_SURVIE_DEB if claim > 0 else 0.0

            PREST_ECH = 0.0
            maturity_occurs = False
            if annee_reelle == int(acc[ACCOUNT_IDX_ANNEE_ECH]) and mois_eval == int(acc[ACCOUNT_IDX_MOIS_ECH]):
                maturity_occurs = True
            else:
                target_month = 12 if MOIS_NAIS == int(12 // freq_eval) else (MOIS_NAIS - int(12 // freq_eval))
                if age == AGE_FIN_CONTRAT and mois_eval == target_month:
                    maturity_occurs = True

            if maturity_occurs:
                diff_ech = MT_GAR_ECH_PROJ - MT_VM_PROJ
                PREST_ECH = -diff_ech * TX_SURVIE if diff_ech > 0 else 0.0
                if diff_ech > 0:
                    MT_VM_PROJ = MT_VM_PROJ + diff_ech
                    MT_GAR_ECH_PROJ = MT_VM_PROJ * PC_GAR_ECH

            # Portfolio rebalance (SAS lines 678-682)
            if MT_VM_ORIG > 0 and MT_VM_PROJ > 0:
                orig_total = (acc[ACCOUNT_IDX_MT_SP500] + acc[ACCOUNT_IDX_MT_TSX] +
                             acc[ACCOUNT_IDX_MT_EAFE] + acc[ACCOUNT_IDX_MT_DEX] +
                             acc[ACCOUNT_IDX_MT_MM])
                if orig_total > 0:
                    MT_SP500_PROJ = MT_VM_PROJ * acc[ACCOUNT_IDX_MT_SP500] / orig_total
                    MT_TSX_PROJ = MT_VM_PROJ * acc[ACCOUNT_IDX_MT_TSX] / orig_total
                    MT_EAFE_PROJ = MT_VM_PROJ * acc[ACCOUNT_IDX_MT_EAFE] / orig_total
                    MT_DEX_PROJ = MT_VM_PROJ * acc[ACCOUNT_IDX_MT_DEX] / orig_total
                    MT_MM_PROJ = MT_VM_PROJ * acc[ACCOUNT_IDX_MT_MM] / orig_total

            comm_vente = 0.0
            frais_acquis = 0.0
            pc_commission_maintien = DEFAULT_COMMISSION_MAINTIEN
            if MT_VM_AV_RETRAIT_FRAIS != 0.0:
                if (duree_max10 < acq_vente_rf.shape[0] and ID_ACQUI < acq_vente_rf.shape[1]):
                    pc_vente_rf_v = acq_vente_rf[duree_max10, ID_ACQUI]
                    pc_vente_ac_v = acq_vente_ac[duree_max10, ID_ACQUI]
                    pc_maintien_rf_v = acq_maintien_rf[duree_max10, ID_ACQUI]
                    pc_maintien_ac_v = acq_maintien_ac[duree_max10, ID_ACQUI]
                    pc_frais_ac_v = acq_frais_ac[duree_max10, ID_ACQUI]
                    pc_frais_rf_v = acq_frais_rf[duree_max10, ID_ACQUI]
                else:
                    pc_vente_rf_v = pc_vente_ac_v = 0.0
                    pc_maintien_rf_v = pc_maintien_ac_v = 0.0
                    pc_frais_ac_v = pc_frais_rf_v = 0.0

                mt_vm_base = acc[ACCOUNT_IDX_MT_VM]
                mt_vm_for_rates = mt_vm_base if mt_vm_base != 0.0 else MT_VM_PROJ
                if mt_vm_for_rates > 0.0:
                    weight_rf = MT_RF / mt_vm_for_rates
                    if weight_rf < 0.0:
                        weight_rf = 0.0
                    if weight_rf > 1.0:
                        weight_rf = 1.0
                    pc_commission_vente = ((pc_vente_ac_v * (1.0 - weight_rf)) + (pc_vente_rf_v * weight_rf)) * AJUSTEMENT_COMMISSION
                    pc_commission_maintien = ((pc_maintien_ac_v * (1.0 - weight_rf)) + (pc_maintien_rf_v * weight_rf)) * AJUSTEMENT_COMMISSION
                    pc_frais_an = (pc_frais_ac_v * (1.0 - weight_rf)) + (pc_frais_rf_v * weight_rf)
                else:
                    pc_commission_vente = 0.0
                    pc_commission_maintien = 0.0
                    pc_frais_an = 0.0

                comm_vente = -pc_commission_vente * depot_futur * TX_SURVIE
                frais_acquis = pc_frais_an * MT_VM_AP_RETRAIT * lapse * TX_SURVIE_DEB * (1.0 - qx)

            fixed_fee_annual = 0.0
            if ID_PRODUIT < fees_lookup.shape[0] and annee_reelle < fees_lookup.shape[1]:
                fixed_fee_annual = fees_lookup[ID_PRODUIT, annee_reelle]
            FRAIS_FIXES = -fixed_fee_annual / freq_eval * AJUST_NOUV_AFFAIRES * TX_SURVIE_DEB if MT_VM_AV_RETRAIT > 0.0 else 0.0

            HON_GEST = -MT_VM_AV_RETRAIT_FRAIS * (math.exp(PC_HONORAIRES_GEST / freq_eval * AJUST_NOUV_AFFAIRES) - 1.0) * TX_SURVIE_DEB

            COMM_MAINTIEN = -MT_VM_AV_RETRAIT_FRAIS * (math.exp(pc_commission_maintien / freq_eval * AJUST_NOUV_AFFAIRES) - 1.0) * TX_SURVIE_DEB

            PRIMES_VARIABLES = MT_VM_AV_RETRAIT_FRAIS * math.exp(-(PC_RFG - PC_REVENU_FDS) / freq_eval * AJUST_NOUV_AFFAIRES) * (-(math.exp(-PC_REVENU_FDS / freq_eval * AJUST_NOUV_AFFAIRES) - 1.0)) * TX_SURVIE_DEB

            VALEUR_MARCHANDE = MT_VM_PROJ * TX_SURVIE

            max_guarantee = MT_GAR_ECH_PROJ
            if MT_GAR_DECES_PROJ + MT_BONI_DECES_PROJ > max_guarantee:
                max_guarantee = MT_GAR_DECES_PROJ + MT_BONI_DECES_PROJ
            if MT_SRG_PROJ > max_guarantee:
                max_guarantee = MT_SRG_PROJ

            if ID_PRODUIT == 22:
                code_cat_produit = 0
            elif ID_PRODUIT == 12 or ID_PRODUIT == 13 or ID_PRODUIT == 14 or ID_PRODUIT == 15 or ID_PRODUIT == 16:
                code_cat_produit = 1
            elif ID_PRODUIT == 17 or ID_PRODUIT == 18 or ID_PRODUIT == 19 or ID_PRODUIT == 20 or ID_PRODUIT == 21:
                code_cat_produit = 2
            elif ID_PRODUIT == 6:
                code_cat_produit = 3
            elif ID_PRODUIT == 4 or ID_PRODUIT == 7:
                code_cat_produit = 4
            elif ID_PRODUIT == 5 or ID_PRODUIT == 8:
                code_cat_produit = 5
            elif ID_PRODUIT == 2 or ID_PRODUIT == 3:
                code_cat_produit = 6
            else:
                code_cat_produit = 7

            pct_rf = 0.0
            if MT_VM_PROJ > 0.0:
                pct_rf = (MT_DEX_PROJ + MT_MM_PROJ) / MT_VM_PROJ

            if code_cat_produit == 0 or code_cat_produit == 6:
                cat_coussin_1 = 0
            elif code_cat_produit == 7 and pct_rf < 0.5:
                cat_coussin_1 = 4
            elif code_cat_produit == 7:
                cat_coussin_1 = 5
            elif pct_rf < (1.0 / 3.0):
                cat_coussin_1 = 1
            elif pct_rf < (2.0 / 3.0):
                cat_coussin_1 = 2
            else:
                cat_coussin_1 = 3

            if code_cat_produit == 7 and VM_VG_RATIO < 0.7:
                cat_coussin_2 = 4
            elif code_cat_produit == 7 and VM_VG_RATIO < 0.9:
                cat_coussin_2 = 5
            elif code_cat_produit == 7:
                cat_coussin_2 = 6
            elif duree_max10 <= 3:
                cat_coussin_2 = 1
            elif duree_max10 <= 6:
                cat_coussin_2 = 2
            else:
                cat_coussin_2 = 3

            if age < 80:
                age_factor = 1.0
            elif age < 90:
                age_factor = facteur_age_80[code_cat_produit, cat_coussin_1, cat_coussin_2]
            else:
                age_factor = facteur_age_90[code_cat_produit, cat_coussin_1, cat_coussin_2]

            tx_passif_v = tx_passif[code_cat_produit, cat_coussin_1, cat_coussin_2]
            tx_credit_v = tx_credit[code_cat_produit, cat_coussin_1, cat_coussin_2]
            tx_marche_v = tx_marche[code_cat_produit, cat_coussin_1, cat_coussin_2]
            tx_depense_v = tx_depense[code_cat_produit, cat_coussin_1, cat_coussin_2]
            tx_decheance_v = tx_decheance[code_cat_produit, cat_coussin_1, cat_coussin_2]
            tx_mortalite_v = tx_mortalite[code_cat_produit, cat_coussin_1, cat_coussin_2]
            tx_depot_v = tx_depot[code_cat_produit, cat_coussin_1, cat_coussin_2]

            if code_cat_produit == 7 and MT_VM_PROJ == 0.0:
                tx_credit_v = 0.0
                tx_marche_v = 0.0
                tx_decheance_v = 0.0
                tx_depot_v = 0.0

            base_passif_v = base_passif[code_cat_produit, cat_coussin_1, cat_coussin_2]
            base_credit_v = base_credit[code_cat_produit, cat_coussin_1, cat_coussin_2]
            base_marche_v = base_marche[code_cat_produit, cat_coussin_1, cat_coussin_2]
            base_depense_v = base_depense[code_cat_produit, cat_coussin_1, cat_coussin_2]
            base_decheance_v = base_decheance[code_cat_produit, cat_coussin_1, cat_coussin_2]
            base_mortalite_v = base_mortalite[code_cat_produit, cat_coussin_1, cat_coussin_2]
            base_depot_v = base_depot[code_cat_produit, cat_coussin_1, cat_coussin_2]

            base_amt_passif = max_guarantee if base_passif_v == 0 else MT_VM_PROJ
            base_amt_credit = max_guarantee if base_credit_v == 0 else MT_VM_PROJ
            base_amt_marche = max_guarantee if base_marche_v == 0 else MT_VM_PROJ
            base_amt_depense = max_guarantee if base_depense_v == 0 else MT_VM_PROJ
            base_amt_decheance = max_guarantee if base_decheance_v == 0 else MT_VM_PROJ
            base_amt_mortalite = max_guarantee if base_mortalite_v == 0 else MT_VM_PROJ
            base_amt_depot = max_guarantee if base_depot_v == 0 else MT_VM_PROJ

            PASSIF_REDRESSE = tx_passif_v * base_amt_passif * age_factor * TX_SURVIE
            COUSSIN_CREDIT = tx_credit_v * base_amt_credit * age_factor * TX_SURVIE
            COUSSIN_MARCHE = tx_marche_v * base_amt_marche * age_factor * TX_SURVIE
            COUSSIN_DEPENSE = tx_depense_v * base_amt_depense * age_factor * TX_SURVIE
            COUSSIN_DECHEANCE = tx_decheance_v * base_amt_decheance * age_factor * TX_SURVIE
            COUSSIN_MORTALITE = tx_mortalite_v * base_amt_mortalite * age_factor * TX_SURVIE
            COUSSIN_DEPOT = tx_depot_v * base_amt_depot * age_factor * TX_SURVIE

            if output_flux_agg is not None and an_eval < output_flux_agg.shape[0] and mois_eval < output_flux_agg.shape[1]:
                cuda.atomic.add(output_flux_agg, (an_eval, mois_eval, FLUX_COMP_IDX_PRIMES_GARANTIES), PRIMES_GARANTIES)
                cuda.atomic.add(output_flux_agg, (an_eval, mois_eval, FLUX_COMP_IDX_PREST_DECES), PREST_DECES)
                cuda.atomic.add(output_flux_agg, (an_eval, mois_eval, FLUX_COMP_IDX_PREST_ECH), PREST_ECH)
                cuda.atomic.add(output_flux_agg, (an_eval, mois_eval, FLUX_COMP_IDX_PREST_MRV), PREST_MRV)
                cuda.atomic.add(output_flux_agg, (an_eval, mois_eval, FLUX_COMP_IDX_FRAIS_ACQUIS), frais_acquis)
                cuda.atomic.add(output_flux_agg, (an_eval, mois_eval, FLUX_COMP_IDX_COMM_VENTE), comm_vente)
                cuda.atomic.add(output_flux_agg, (an_eval, mois_eval, FLUX_COMP_IDX_PRIMES_VARIABLES), PRIMES_VARIABLES)
                cuda.atomic.add(output_flux_agg, (an_eval, mois_eval, FLUX_COMP_IDX_FRAIS_FIXES), FRAIS_FIXES)
                cuda.atomic.add(output_flux_agg, (an_eval, mois_eval, FLUX_COMP_IDX_HON_GEST), HON_GEST)
                cuda.atomic.add(output_flux_agg, (an_eval, mois_eval, FLUX_COMP_IDX_COMM_MAINTIEN), COMM_MAINTIEN)
                cuda.atomic.add(output_flux_agg, (an_eval, mois_eval, FLUX_COMP_IDX_VALEUR_MARCHANDE), VALEUR_MARCHANDE)
                cuda.atomic.add(output_flux_agg, (an_eval, mois_eval, FLUX_COMP_IDX_PASSIF_REDRESSE), PASSIF_REDRESSE)
                cuda.atomic.add(output_flux_agg, (an_eval, mois_eval, FLUX_COMP_IDX_COUSSIN_CREDIT), COUSSIN_CREDIT)
                cuda.atomic.add(output_flux_agg, (an_eval, mois_eval, FLUX_COMP_IDX_COUSSIN_MARCHE), COUSSIN_MARCHE)
                cuda.atomic.add(output_flux_agg, (an_eval, mois_eval, FLUX_COMP_IDX_COUSSIN_DEPENSE), COUSSIN_DEPENSE)
                cuda.atomic.add(output_flux_agg, (an_eval, mois_eval, FLUX_COMP_IDX_COUSSIN_DECHEANCE), COUSSIN_DECHEANCE)
                cuda.atomic.add(output_flux_agg, (an_eval, mois_eval, FLUX_COMP_IDX_COUSSIN_MORTALITE), COUSSIN_MORTALITE)
                cuda.atomic.add(output_flux_agg, (an_eval, mois_eval, FLUX_COMP_IDX_COUSSIN_DEPOT), COUSSIN_DEPOT)

            # === SAVE STATE TO GLOBAL MEMORY ===
            if an_eval < n_years and output_year_idx < output_states.shape[2]:
                output_states[account_idx, scenario_idx, output_year_idx, STATE_IDX_MT_VM] = MT_VM_PROJ
                output_states[account_idx, scenario_idx, output_year_idx, STATE_IDX_MT_GAR_DECES] = MT_GAR_DECES_PROJ
                output_states[account_idx, scenario_idx, output_year_idx, STATE_IDX_MT_GAR_ECH] = MT_GAR_ECH_PROJ
                output_states[account_idx, scenario_idx, output_year_idx, STATE_IDX_MT_SRG] = MT_SRG_PROJ
                output_states[account_idx, scenario_idx, output_year_idx, STATE_IDX_AGE] = float(age)
                output_states[account_idx, scenario_idx, output_year_idx, STATE_IDX_TX_SURVIE] = TX_SURVIE
                output_states[account_idx, scenario_idx, output_year_idx, STATE_IDX_MT_DEX] = MT_DEX_PROJ
                output_states[account_idx, scenario_idx, output_year_idx, STATE_IDX_MT_MM] = MT_MM_PROJ
                output_states[account_idx, scenario_idx, output_year_idx, STATE_IDX_MT_TSX] = MT_TSX_PROJ
                output_states[account_idx, scenario_idx, output_year_idx, STATE_IDX_MT_SP500] = MT_SP500_PROJ
                output_states[account_idx, scenario_idx, output_year_idx, STATE_IDX_MT_EAFE] = MT_EAFE_PROJ
                output_states[account_idx, scenario_idx, output_year_idx, STATE_IDX_MT_BONI_DECES] = MT_BONI_DECES_PROJ

                # Calculate external scenario cashflows for reporting
                # Management fee revenue
                hon_gest_ext = MT_VM_AV_RETRAIT_FRAIS * PC_HONORAIRES_GEST / freq_eval * TX_SURVIE_DEB

                # Guarantee fees
                vp_primes_garanties = PRIMES_GARANTIES * TX_ACTUALISATION

                # Death benefit cost
                vp_prest_deces = PREST_DECES * TX_ACTUALISATION

                # Net cashflow for external scenario
                flux_net = hon_gest_ext + vp_primes_garanties + vp_prest_deces
                output_cashflows[account_idx, scenario_idx, output_year_idx, 0] = flux_net

                # === SAVE DEBUG OUTPUT (only if filter matches) ===
                if debug_output is not None:
                    # Check if this iteration matches the debug filter
                    matches_filter = (
                        (debug_account < 0 or account_idx == debug_account) and
                        (debug_scenario < 0 or scenario_idx == debug_scenario) and
                        (debug_year < 0 or an_eval == debug_year) and
                        (debug_month < 0 or mois_eval == debug_month)
                    )
                    if matches_filter:
                        debug_output[EXT_DEBUG_IDX_VM] = MT_VM_PROJ
                        debug_output[EXT_DEBUG_IDX_AGE] = float(age)
                        debug_output[EXT_DEBUG_IDX_QX] = qx
                        debug_output[EXT_DEBUG_IDX_LAPSE_TOT] = LAPSE_TOT
                        debug_output[EXT_DEBUG_IDX_LAPSE_PART] = LAPSE_PART
                        debug_output[EXT_DEBUG_IDX_TX_SURVIE] = TX_SURVIE
                        debug_output[EXT_DEBUG_IDX_FORWARD_RATE] = FORWARD_RATE
                        debug_output[EXT_DEBUG_IDX_REND_SP500] = r_sp500
                        debug_output[EXT_DEBUG_IDX_REND_TSX] = r_tsx
                        debug_output[EXT_DEBUG_IDX_REND_EAFE] = r_eafe
                        debug_output[EXT_DEBUG_IDX_REND_DEX] = r_dex
                        debug_output[EXT_DEBUG_IDX_RETRAIT] = RETRAIT
                        debug_output[EXT_DEBUG_IDX_PREST_DECES] = PREST_DECES
                        debug_output[EXT_DEBUG_IDX_PRIMES_GARANTIES] = PRIMES_GARANTIES
                        debug_output[EXT_DEBUG_IDX_VM_VG_RATIO] = VM_VG_RATIO

                output_year_idx += 1


@cuda.jit
def nested_valuation_kernel_five_chocs(
        input_states,        # StatesTensor: (batch, ext_scenarios, years, STATE_SIZE)
        account_data,        # AccountData: (n_accounts, n_fields)
        n_internal_scenarios,  # int
        n_internal_years,    # int
        rn_returns_lookups,  # RNReturnsLookups: 6 arrays
        mortality_lookup,    # MortalityLookup: (sex, age, year, product)
        output_metrics,      # MetricsTensor: (batch, ext_scenarios, years, NUM_CHOCS, 2)
        debug_output=None,   # Optional: (NUM_CHOCS, INT_DEBUG_SIZE) - one row per choc for filtered debug
        debug_ts_output=None,  # Optional: (NUM_CHOCS, n_internal_years, INT_TS_DEBUG_IDX_SIZE) - time series for filtered debug
        debug_int_scenario=-1,  # Internal scenario to debug (-1 = disabled, logs all)
        debug_int_year=-1,      # Internal year to debug (-1 = disabled, logs all)
        debug_account=-1,       # Account index to debug (-1 = disabled)
        debug_ext_scenario=-1,  # External scenario to debug (-1 = disabled)
        debug_ext_year=-1,      # External year to debug (-1 = disabled)
):
    """
    KERNEL B: NESTED VALUATOR WITH 5 CHOCS

    Implements the 5 chocs pattern from SAS code:
    - Choc 0: Base scenario (no shock)
    - Choc 1: SP500 shock (-10%)
    - Choc 2: TSX shock (-10%)
    - Choc 3: EAFE shock (-10%)
    - Choc 4: DEX shock (-10%)

    For each external node, runs all 5 chocs with internal scenarios.

    Args:
        input_states: State tensor from Kernel A
        account_data: Account attributes for fee calculations
        n_internal_scenarios: Number of risk-neutral scenarios per node
        n_internal_years: Internal projection horizon
        rn_returns_lookups: (rn_forward_rate, rn_ajust_forward, rn_rend_dex, rn_rend_mm, rn_rend_tsx, rn_rend_sp500, rn_rend_eafe)
        mortality_lookup: Mortality rates (reused from Kernel A)
        output_metrics: Output tensor with reserve/capital per choc
    """
    # Unpack risk-neutral returns tuple (7 arrays)
    rn_forward_rate, rn_ajust_forward, rn_rend_dex, rn_rend_mm, rn_rend_tsx, rn_rend_sp500, rn_rend_eafe = rn_returns_lookups

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
    start_vm = input_states[acc_idx, scn_idx, year_idx, STATE_IDX_MT_VM]
    start_gar_deces = input_states[acc_idx, scn_idx, year_idx, STATE_IDX_MT_GAR_DECES]
    start_gar_ech = input_states[acc_idx, scn_idx, year_idx, STATE_IDX_MT_GAR_ECH]
    start_srg = input_states[acc_idx, scn_idx, year_idx, STATE_IDX_MT_SRG]
    start_age = input_states[acc_idx, scn_idx, year_idx, STATE_IDX_AGE]
    start_tx_survie = input_states[acc_idx, scn_idx, year_idx, STATE_IDX_TX_SURVIE]

    # Load individual asset values from state
    start_mt_dex = input_states[acc_idx, scn_idx, year_idx, STATE_IDX_MT_DEX]
    start_mt_mm = input_states[acc_idx, scn_idx, year_idx, STATE_IDX_MT_MM]
    start_mt_tsx = input_states[acc_idx, scn_idx, year_idx, STATE_IDX_MT_TSX]
    start_mt_sp500 = input_states[acc_idx, scn_idx, year_idx, STATE_IDX_MT_SP500]
    start_mt_eafe = input_states[acc_idx, scn_idx, year_idx, STATE_IDX_MT_EAFE]
    start_boni_deces = input_states[acc_idx, scn_idx, year_idx, STATE_IDX_MT_BONI_DECES]

    # Check if policy is active
    if start_vm <= 0 or start_tx_survie <= 0:
        # Zero out all chocs
        for choc in range(NUM_CHOCS):
            output_metrics[acc_idx, scn_idx, year_idx, choc, METRICS_RESERVE_IDX] = 0.0
            output_metrics[acc_idx, scn_idx, year_idx, choc, METRICS_CAPITAL_IDX] = 0.0
        return

    # Load account parameters using module-level constants
    acc = account_data[acc_idx]
    PC_RFG = acc[ACCOUNT_IDX_PC_RFG]
    FREQ_EVAL = DEFAULT_FREQ_EVAL

    # ===========================================
    # LOOP THROUGH ALL 5 CHOCS
    # ===========================================

    # Pre-compute debug filter match ONCE outside all loops
    is_debug_node = False
    if debug_output is not None or debug_ts_output is not None:
        is_debug_node = (
            (debug_account < 0 or acc_idx == debug_account) and
            (debug_ext_scenario < 0 or scn_idx == debug_ext_scenario) and
            (debug_ext_year < 0 or year_idx == debug_ext_year)
        )

    # Pre-fetch array dimensions ONCE (avoid repeated shape lookups)
    n_scn_avail = rn_rend_dex.shape[0]
    n_yr_avail = rn_rend_dex.shape[1]

    for choc_idx in range(NUM_CHOCS):
        # Apply shock according to SAS logic (lines 219-234)
        if choc_idx == 0:
            # Base scenario - no shock
            mt_sp500_choc = start_mt_sp500
            mt_tsx_choc = start_mt_tsx
            mt_eafe_choc = start_mt_eafe
            mt_dex_choc = start_mt_dex
            mt_mm_choc = start_mt_mm
        elif choc_idx == 1:
            # Choc SP500 (-10%)
            mt_sp500_choc = start_mt_sp500 * SHOCK_FACTOR
            mt_tsx_choc = start_mt_tsx
            mt_eafe_choc = start_mt_eafe
            mt_dex_choc = start_mt_dex
            mt_mm_choc = start_mt_mm
        elif choc_idx == 2:
            # Choc TSX (-10%)
            mt_sp500_choc = start_mt_sp500
            mt_tsx_choc = start_mt_tsx * SHOCK_FACTOR
            mt_eafe_choc = start_mt_eafe
            mt_dex_choc = start_mt_dex
            mt_mm_choc = start_mt_mm
        elif choc_idx == 3:
            # Choc EAFE (-10%)
            mt_sp500_choc = start_mt_sp500
            mt_tsx_choc = start_mt_tsx
            mt_eafe_choc = start_mt_eafe * SHOCK_FACTOR
            mt_dex_choc = start_mt_dex
            mt_mm_choc = start_mt_mm
        elif choc_idx == 4:
            # Choc DEX (-10%)
            mt_sp500_choc = start_mt_sp500
            mt_tsx_choc = start_mt_tsx
            mt_eafe_choc = start_mt_eafe
            mt_dex_choc = start_mt_dex * SHOCK_FACTOR
            mt_mm_choc = start_mt_mm

        # Rebalance total VM after shock (SAS line 233)
        vm_choc_start = mt_sp500_choc + mt_tsx_choc + mt_eafe_choc + mt_dex_choc + mt_mm_choc

        # =====================================
        # INTERNAL SCENARIOS FOR THIS CHOC
        # =====================================

        sum_pv_flux_choc = 0.0
        
        # Variables to capture debug values at specific internal iteration
        debug_curr_vm = 0.0
        debug_fees = 0.0
        debug_pv_path = 0.0
        debug_r_portfolio = 0.0
        debug_fwd_rate = 0.0

        # Pre-compute fee factor (constant across iterations)
        fee_factor = 1.0 - (PC_RFG / FREQ_EVAL)
        fee_rate = PC_RFG / FREQ_EVAL

        for i_int in range(n_internal_scenarios):
            # Track individual asset values through internal projection
            curr_mt_dex = mt_dex_choc
            curr_mt_mm = mt_mm_choc
            curr_mt_tsx = mt_tsx_choc
            curr_mt_sp500 = mt_sp500_choc
            curr_mt_eafe = mt_eafe_choc
            curr_vm = vm_choc_start
            pv_path = 0.0

            # Map internal scenario to available scenarios (wrap if needed)
            scn_idx_int = i_int % n_scn_avail if n_scn_avail > 0 else 0

            for t_int in range(n_internal_years):
                if curr_vm <= 0:
                    break

                # Map internal year to available years (wrap if needed)
                t_idx_int = t_int % n_yr_avail if n_yr_avail > 0 else 0

                # Get risk-neutral returns for each asset class from RENDEMENTS_INT
                if n_scn_avail > 0 and n_yr_avail > 0:
                    r_dex = rn_rend_dex[scn_idx_int, t_idx_int]
                    r_mm = rn_rend_mm[scn_idx_int, t_idx_int]
                    r_tsx = rn_rend_tsx[scn_idx_int, t_idx_int]
                    r_sp500 = rn_rend_sp500[scn_idx_int, t_idx_int]
                    r_eafe = rn_rend_eafe[scn_idx_int, t_idx_int]
                    fwd = rn_forward_rate[scn_idx_int, t_idx_int]
                else:
                    r_dex = DEFAULT_RETURN_RATE
                    r_mm = DEFAULT_RETURN_RATE
                    r_tsx = DEFAULT_RETURN_RATE
                    r_sp500 = DEFAULT_RETURN_RATE
                    r_eafe = DEFAULT_RETURN_RATE
                    fwd = DEFAULT_FORWARD_RATE

                # Apply returns to each asset class separately
                curr_mt_dex *= math.exp(r_dex)
                curr_mt_mm *= math.exp(r_mm)
                curr_mt_tsx *= math.exp(r_tsx)
                curr_mt_sp500 *= math.exp(r_sp500)
                curr_mt_eafe *= math.exp(r_eafe)

                # Recalculate total VM from individual assets
                curr_vm = curr_mt_dex + curr_mt_mm + curr_mt_tsx + curr_mt_sp500 + curr_mt_eafe

                # Apply fees
                fees = curr_vm * fee_rate
                curr_vm -= fees
                # Proportionally reduce each asset by fees
                if curr_vm > 0:
                    curr_mt_dex *= fee_factor
                    curr_mt_mm *= fee_factor
                    curr_mt_tsx *= fee_factor
                    curr_mt_sp500 *= fee_factor
                    curr_mt_eafe *= fee_factor

                # Discount cashflow
                df = math.exp(-fwd * (t_int + 1))
                pv_path += fees * df

                # Debug capture - only for the specific debug node (pre-filtered)
                if is_debug_node and debug_int_scenario >= 0 and i_int == debug_int_scenario:
                    # Calculate weighted portfolio return only when needed for debug
                    if vm_choc_start > 0:
                        r_portfolio = (r_dex * mt_dex_choc + r_mm * mt_mm_choc + r_tsx * mt_tsx_choc + 
                                       r_sp500 * mt_sp500_choc + r_eafe * mt_eafe_choc) / vm_choc_start
                    else:
                        r_portfolio = 0.0
                    
                    if debug_int_year < 0 or t_int == debug_int_year:
                        debug_curr_vm = curr_vm
                        debug_fees = fees
                        debug_pv_path = pv_path
                        debug_r_portfolio = r_portfolio
                        debug_fwd_rate = fwd
                    
                    # Time series debug output
                    if debug_ts_output is not None and (debug_int_year < 0 or t_int == debug_int_year):
                        debug_ts_output[choc_idx, t_int, INT_TS_DEBUG_IDX_CURR_VM] = curr_vm
                        debug_ts_output[choc_idx, t_int, INT_TS_DEBUG_IDX_FEES] = fees
                        debug_ts_output[choc_idx, t_int, INT_TS_DEBUG_IDX_PV_PATH] = pv_path
                        debug_ts_output[choc_idx, t_int, INT_TS_DEBUG_IDX_R_PORTFOLIO] = r_portfolio
                        debug_ts_output[choc_idx, t_int, INT_TS_DEBUG_IDX_FWD_RATE] = fwd
                        debug_ts_output[choc_idx, t_int, INT_TS_DEBUG_IDX_DF] = df

            sum_pv_flux_choc += pv_path

        # Average over internal scenarios for this choc
        avg_pv_flux = sum_pv_flux_choc / n_internal_scenarios if n_internal_scenarios > 0 else 0.0

        # Store results for this choc
        # For simplicity, storing PV flux as both reserve and capital
        # In practice, you might want different calculations
        output_metrics[acc_idx, scn_idx, year_idx, choc_idx, METRICS_RESERVE_IDX] = avg_pv_flux  # Reserve
        output_metrics[acc_idx, scn_idx, year_idx, choc_idx, METRICS_CAPITAL_IDX] = avg_pv_flux  # Capital (same for now)

        # === SAVE DEBUG OUTPUT (one row per choc, only if filter matches) ===
        if debug_output is not None:
            # Check if this node matches the debug filter
            matches_filter = (
                (debug_account < 0 or acc_idx == debug_account) and
                (debug_ext_scenario < 0 or scn_idx == debug_ext_scenario) and
                (debug_ext_year < 0 or year_idx == debug_ext_year)
            )
            if matches_filter:
                debug_output[choc_idx, INT_DEBUG_IDX_START_VM] = start_vm
                debug_output[choc_idx, INT_DEBUG_IDX_VM_CHOC] = vm_choc_start
                debug_output[choc_idx, INT_DEBUG_IDX_AVG_PV_FLUX] = avg_pv_flux
                debug_output[choc_idx, INT_DEBUG_IDX_RESERVE] = avg_pv_flux
                debug_output[choc_idx, INT_DEBUG_IDX_CAPITAL] = avg_pv_flux
                debug_output[choc_idx, INT_DEBUG_IDX_START_TX_SURVIE] = start_tx_survie
                debug_output[choc_idx, INT_DEBUG_IDX_START_AGE] = start_age
                # Values from specific internal iteration
                debug_output[choc_idx, INT_DEBUG_IDX_CURR_VM] = debug_curr_vm
                debug_output[choc_idx, INT_DEBUG_IDX_FEES] = debug_fees
                debug_output[choc_idx, INT_DEBUG_IDX_PV_PATH] = debug_pv_path
                debug_output[choc_idx, INT_DEBUG_IDX_R_PORTFOLIO] = debug_r_portfolio
                debug_output[choc_idx, INT_DEBUG_IDX_FWD_RATE] = debug_fwd_rate


# Backward compatibility aliases
STATE_SIZE = STATE_IDX_SIZE
EXT_DEBUG_SIZE = EXT_DEBUG_IDX_SIZE
INT_DEBUG_SIZE = INT_DEBUG_IDX_SIZE

STATE_MT_VM = STATE_IDX_MT_VM
STATE_MT_GAR_DECES = STATE_IDX_MT_GAR_DECES
STATE_MT_GAR_ECH = STATE_IDX_MT_GAR_ECH
STATE_MT_SRG = STATE_IDX_MT_SRG
STATE_AGE = STATE_IDX_AGE
STATE_TX_SURVIE = STATE_IDX_TX_SURVIE

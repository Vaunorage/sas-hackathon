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
    ACCOUNT_IDX_MAX_BONI_DECES,
    ACCOUNT_IDX_AJUSTEMENT_MENSUEL_GAR,
    ACCOUNT_IDX_PC_GAR_DECES_2, ACCOUNT_IDX_AGE_CHANG_DECES,
    ACCOUNT_IDX_FREQ_RESET_DECES, ACCOUNT_IDX_MAX_RESET_DECES,
    ACCOUNT_IDX_I_RESET_DECES_ECH,
    ACCOUNT_IDX_NB_AN_ECH, ACCOUNT_IDX_PC_RENOUV_ECH, ACCOUNT_IDX_AGE_MAX_RENOUV_ECH,
    ACCOUNT_IDX_MAX_RESET_FACUL_ECH, ACCOUNT_IDX_RATIO_VM_VG_RESET_ECH,
    ACCOUNT_IDX_AGE_MRV_PERMIS, ACCOUNT_IDX_PC_BONI_SRG,
    ACCOUNT_IDX_FREQ_RESET_SRG, ACCOUNT_IDX_MAX_RESET_SRG,
    ACCOUNT_IDX_TABLE_TAUX_MRV_MRG_MRA, ACCOUNT_IDX_MT_TPA_RETRAIT,
    ACCOUNT_IDX_M_MT_MRV_EXCEDENT, ACCOUNT_IDX_MT_TPA_DEPOT,
    ACCOUNT_IDX_VAR_RETRAIT_FCT, ACCOUNT_IDX_PC_RETRAIT_AGE,
    ACCOUNT_IDX_MT_RETRAIT_MAX, ACCOUNT_IDX_I_RESET_FACUL_ECH,
    # State tensor indices (module-level constants for Numba)
    STATE_IDX_MT_VM, STATE_IDX_MT_GAR_DECES, STATE_IDX_MT_GAR_ECH,
    STATE_IDX_MT_SRG, STATE_IDX_AGE, STATE_IDX_TX_SURVIE,
    STATE_IDX_MT_DEX, STATE_IDX_MT_MM, STATE_IDX_MT_TSX,
    STATE_IDX_MT_SP500, STATE_IDX_MT_EAFE, STATE_IDX_MT_BONI_DECES,
    STATE_IDX_MT_BCB, STATE_IDX_MT_MRV_MRG_MRA, STATE_IDX_TAUX_MRV_MRG_MRA,
    STATE_IDX_ANNEE_ECH, STATE_IDX_MOIS_ECH,
    STATE_IDX_TX_ACTUALISATION, STATE_IDX_ANNEE_REELLE, STATE_IDX_MOIS_EVAL,
    STATE_IDX_PC_GAR_DECES_1,
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
    # Cashflow output tensor indices (for standard output matching SAS)
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
        output_cashflows,    # CashflowsTensor: (batch, scenarios, years, CF_OUT_IDX_SIZE)
        debug_output=None,   # Optional: (EXT_DEBUG_SIZE,) - single row for filtered debug
        debug_flux_output=None,  # Optional: (n_years+1, freq_eval, FLUX_COMP_IDX_SIZE) - flux for debug account/scenario
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

    MAX_BONI_DECES = int(acc[ACCOUNT_IDX_MAX_BONI_DECES])

    AJUSTEMENT_MENSUEL_GAR = acc[ACCOUNT_IDX_AJUSTEMENT_MENSUEL_GAR]
    PC_GAR_DECES_2 = acc[ACCOUNT_IDX_PC_GAR_DECES_2]
    AGE_CHANG_DECES = int(acc[ACCOUNT_IDX_AGE_CHANG_DECES])
    FREQ_RESET_DECES = int(acc[ACCOUNT_IDX_FREQ_RESET_DECES])
    MAX_RESET_DECES = int(acc[ACCOUNT_IDX_MAX_RESET_DECES])
    I_RESET_DECES_ECH = int(acc[ACCOUNT_IDX_I_RESET_DECES_ECH])
    NB_AN_ECH = int(acc[ACCOUNT_IDX_NB_AN_ECH])
    PC_RENOUV_ECH = acc[ACCOUNT_IDX_PC_RENOUV_ECH]
    AGE_MAX_RENOUV_ECH = int(acc[ACCOUNT_IDX_AGE_MAX_RENOUV_ECH])
    MAX_RESET_FACUL_ECH = int(acc[ACCOUNT_IDX_MAX_RESET_FACUL_ECH])
    RATIO_VM_VG_RESET_ECH = acc[ACCOUNT_IDX_RATIO_VM_VG_RESET_ECH]
    AGE_MRV_PERMIS = int(acc[ACCOUNT_IDX_AGE_MRV_PERMIS])
    PC_BONI_SRG = acc[ACCOUNT_IDX_PC_BONI_SRG]
    FREQ_RESET_SRG = int(acc[ACCOUNT_IDX_FREQ_RESET_SRG])
    MAX_RESET_SRG = int(acc[ACCOUNT_IDX_MAX_RESET_SRG])
    TABLE_TAUX_MRV_MRG_MRA = int(acc[ACCOUNT_IDX_TABLE_TAUX_MRV_MRG_MRA])
    MT_TPA_RETRAIT = acc[ACCOUNT_IDX_MT_TPA_RETRAIT]
    M_MT_MRV_EXCEDENT = acc[ACCOUNT_IDX_M_MT_MRV_EXCEDENT]
    MT_TPA_DEPOT = acc[ACCOUNT_IDX_MT_TPA_DEPOT]
    VAR_RETRAIT_FCT = int(acc[ACCOUNT_IDX_VAR_RETRAIT_FCT])
    PC_RETRAIT_AGE = acc[ACCOUNT_IDX_PC_RETRAIT_AGE]
    MT_RETRAIT_MAX = acc[ACCOUNT_IDX_MT_RETRAIT_MAX]
    I_RESET_FACUL_ECH = int(acc[ACCOUNT_IDX_I_RESET_FACUL_ECH])

    ANNEE_COTIS = int(acc[ACCOUNT_IDX_ANNEE_COTIS]) if acc[ACCOUNT_IDX_ANNEE_COTIS] > 0 else ANNEE_EVALUATION_INI
    MOIS_COTIS = int(acc[ACCOUNT_IDX_MOIS_COTIS]) if acc[ACCOUNT_IDX_MOIS_COTIS] > 0 else 1

    ANNEE_ECH_PROJ = int(acc[ACCOUNT_IDX_ANNEE_ECH])
    MOIS_ECH_PROJ = int(acc[ACCOUNT_IDX_MOIS_ECH])

    # Scenario-specific processing
    scn_eval = scenario_idx + 1

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

            # Check if policy is still active (SAS line 294: deletes row but continues loop)
            # When TX_SURVIE=0, we continue the loop but skip calculations (outputs zeros)
            if TX_SURVIE == 0 or (MT_VM_PROJ == 0 and I_PRODUIT_REGR == 0):
                continue

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

            if PC_BONI_DECES > 0.0 and age < MAX_BONI_DECES:
                MT_BONI_DECES_PROJ = MT_BONI_DECES_PROJ + (MT_GAR_DECES_PROJ * PC_BONI_DECES / freq_eval * AJUST_NOUV_AFFAIRES)
            else:
                MT_BONI_DECES_PROJ = 0.0

            # Calculate retirement age (SAS line 442)
            AGE_RETRAIT = age + 1

            if I_PRODUIT_REGR == 1:
                base_amount = MT_SRG_PROJ if TABLE_TAUX_MRV_MRG_MRA == 1 else MT_VM_PROJ
                if AGE_RETRAIT < AGE_MRV_PERMIS and base_amount == 0.0:
                    MT_MRV_MRG_MRA_PROJ = 0.0

                if mois_eval == 12 // freq_eval:
                    if TABLE_TAUX_MRV_MRG_MRA == 2:
                        max_age_start = AGE_MRV_PERMIS if AGE_MRV_PERMIS > AGE_DECAISSEMENT else AGE_DECAISSEMENT
                        should_reinit = (AGE_RETRAIT == max_age_start or (MT_SRG_PROJ == MT_VM_PROJ and MT_VM_PROJ != 0.0))
                        if should_reinit:
                            if AGE_RETRAIT < 60:
                                TAUX_MRV_MRG_MRA_PROJ = 0.03
                            elif AGE_RETRAIT < 65:
                                TAUX_MRV_MRG_MRA_PROJ = 0.035
                            elif AGE_RETRAIT < 70:
                                TAUX_MRV_MRG_MRA_PROJ = 0.04
                            elif AGE_RETRAIT < 75:
                                TAUX_MRV_MRG_MRA_PROJ = 0.0425
                            else:
                                TAUX_MRV_MRG_MRA_PROJ = 0.05
                            MT_MRV_MRG_MRA_PROJ = TAUX_MRV_MRG_MRA_PROJ * MT_SRG_PROJ
                        else:
                            tmp_mrv = TAUX_MRV_MRG_MRA_PROJ * MT_SRG_PROJ
                            if tmp_mrv > MT_MRV_MRG_MRA_PROJ:
                                MT_MRV_MRG_MRA_PROJ = tmp_mrv
                    else:
                        if AGE_RETRAIT == AGE_MRV_PERMIS:
                            MT_MRV_MRG_MRA_PROJ = TAUX_MRV_MRG_MRA_PROJ * MT_SRG_PROJ
                        else:
                            tmp_mrv = TAUX_MRV_MRG_MRA_PROJ * MT_SRG_PROJ
                            if tmp_mrv > MT_MRV_MRG_MRA_PROJ:
                                MT_MRV_MRG_MRA_PROJ = tmp_mrv

                    if (M_MT_MRV_EXCEDENT > 1.0 and
                            MOIS_EVALUATION_INI != 12 // freq_eval and
                            an_eval == 2 and
                            mois_eval == 12 // freq_eval):
                        base_excedent = MT_SRG_PROJ if MT_SRG_PROJ > MT_VM_PROJ else MT_VM_PROJ
                        tmp_mrv = TAUX_MRV_MRG_MRA_PROJ * base_excedent
                        if tmp_mrv < MT_MRV_MRG_MRA_PROJ:
                            MT_MRV_MRG_MRA_PROJ = tmp_mrv

            # Withdrawals calculation (SAS lines 498-507)
            RETRAIT = 0.0
            if not (
                AGE_RETRAIT < AGE_DECAISSEMENT or
                (AGE_RETRAIT == AGE_DECAISSEMENT and mois_eval >= MOIS_NAIS) or
                (MT_VM_PROJ <= 0 and I_PRODUIT_REGR == 0)
            ):
                if VAR_RETRAIT_FCT == 1:
                    RETRAIT = MT_TPA_RETRAIT if MT_TPA_RETRAIT > 0.0 else MT_VM_PROJ * PC_RETRAIT_AGE
                elif VAR_RETRAIT_FCT == 2:
                    if MT_TPA_RETRAIT > MT_MIN_FERR_PROJ:
                        RETRAIT = MT_TPA_RETRAIT
                    else:
                        pc_ra = PC_RETRAIT_AGE
                        if pc_ra < 1.0:
                            pc_ra = 1.0
                        RETRAIT = MT_MIN_FERR_PROJ * pc_ra
                elif VAR_RETRAIT_FCT == 3:
                    base_mrv = MT_MRV_MRG_MRA_PROJ if MT_MRV_MRG_MRA_PROJ > MT_MIN_FERR_PROJ else MT_MIN_FERR_PROJ
                    RETRAIT = base_mrv * PC_RETRAIT_AGE
                else:
                    RETRAIT = 0.0

                if RETRAIT > MT_RETRAIT_MAX and MT_RETRAIT_MAX > 0.0:
                    RETRAIT = MT_RETRAIT_MAX
                RETRAIT = RETRAIT / freq_eval

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
                if MT_TPA_DEPOT > 0.0:
                    depot_futur = MT_TPA_DEPOT
                else:
                    base_depot_calc = MT_VM_PROJ if var_depot_fct == 1 else (acc[ACCOUNT_IDX_MT_GAR_DECES] / max(PC_GAR_DECES_1, MIN_GUARANTEE_VALUE))
                    depot_futur = base_depot_calc * pc_depot_annuel
                depot_futur = depot_futur / freq_eval

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
            if annee_reelle == ANNEE_ECH_PROJ and mois_eval == MOIS_ECH_PROJ:
                maturity_occurs = True
            else:
                target_month = 12 if MOIS_NAIS == int(12 // freq_eval) else (MOIS_NAIS - int(12 // freq_eval))
                if age == AGE_FIN_CONTRAT and mois_eval == target_month:
                    maturity_occurs = True

            if maturity_occurs:
                diff_ech = MT_GAR_ECH_PROJ - MT_VM_AP_RETRAIT
                PREST_ECH = -diff_ech * TX_SURVIE if diff_ech > 0 else 0.0
                ANNEE_ECH_PROJ = ANNEE_ECH_PROJ + NB_AN_ECH
                MOIS_ECH_PROJ = mois_eval
                if diff_ech > 0:
                    MT_VM_PROJ = MT_VM_AP_RETRAIT + diff_ech
                else:
                    MT_VM_PROJ = MT_VM_AP_RETRAIT
                MT_GAR_ECH_PROJ = MT_VM_PROJ * PC_GAR_ECH
                if I_RESET_DECES_ECH == 1:
                    MT_GAR_DECES_PROJ = MT_VM_PROJ * PC_GAR_DECES_1
                pc_renouv = PC_RENOUV_ECH
                if age > AGE_MAX_RENOUV_ECH:
                    pc_renouv = 0.0
                TX_SURVIE = TX_SURVIE * pc_renouv

            # Portfolio rebalance (SAS lines 678-682)
            MT_GAR_DECES_PROJ = MT_GAR_DECES_PROJ - AJUSTEMENT_MENSUEL_GAR * 12.0 / freq_eval

            if I_PRODUIT_REGR == 1:
                if (age < MAX_RESET_SRG and
                        MT_SRG_PROJ < MT_VM_PROJ and
                        annee_reelle > ANNEE_COTIS and
                        FREQ_RESET_SRG > 0):
                    years_since_issue = annee_reelle - ANNEE_COTIS
                    if int(years_since_issue / FREQ_RESET_SRG) == years_since_issue / FREQ_RESET_SRG and mois_eval == MOIS_COTIS:
                        MT_SRG_PROJ = MT_VM_PROJ
                        if MT_VM_PROJ > MT_BCB_PROJ:
                            MT_BCB_PROJ = MT_VM_PROJ

                if age < AGE_DECAISSEMENT and mois_eval == 12:
                    MT_SRG_PROJ = MT_SRG_PROJ + PC_BONI_SRG * MT_BCB_PROJ

            if (age < MAX_RESET_DECES and
                    (MT_GAR_DECES_PROJ + MT_BONI_DECES_PROJ) < (MT_VM_PROJ * PC_GAR_DECES_1) and
                    annee_reelle > ANNEE_COTIS):
                should_reset = False
                if FREQ_RESET_DECES > 0:
                    years_since_issue = annee_reelle - ANNEE_COTIS
                    if int(years_since_issue / FREQ_RESET_DECES) == years_since_issue / FREQ_RESET_DECES and mois_eval == MOIS_COTIS:
                        should_reset = True

                target_month = 12 if MOIS_NAIS == int(12 // freq_eval) else (MOIS_NAIS - int(12 // freq_eval))
                if age == MAX_RESET_DECES - 1 and mois_eval == target_month:
                    should_reset = True

                if should_reset:
                    MT_GAR_DECES_PROJ = MT_VM_PROJ * PC_GAR_DECES_1
                    MT_BONI_DECES_PROJ = 0.0

            if mois_eval == 6 or mois_eval == 12:
                if (I_RESET_FACUL_ECH == 1 and
                        age <= MAX_RESET_FACUL_ECH and
                        MT_GAR_ECH_PROJ > 0.0 and
                        (MT_VM_PROJ * PC_GAR_ECH) >= RATIO_VM_VG_RESET_ECH * MT_GAR_ECH_PROJ):
                    tmp_gar = MT_VM_PROJ * PC_GAR_ECH
                    if tmp_gar > MT_GAR_ECH_PROJ:
                        MT_GAR_ECH_PROJ = tmp_gar

                    tmp_annee = annee_reelle + NB_AN_ECH
                    min_annee = ANNEE_NAIS + AGE_ECH_MIN
                    ANNEE_ECH_PROJ = tmp_annee if tmp_annee > min_annee else min_annee
                    if ANNEE_ECH_PROJ == min_annee:
                        MOIS_ECH_PROJ = MOIS_NAIS
                    else:
                        MOIS_ECH_PROJ = mois_eval

            target_month = 12 if MOIS_NAIS == int(12 // freq_eval) else (MOIS_NAIS - int(12 // freq_eval))
            if age == AGE_CHANG_DECES - 1 and mois_eval == target_month:
                if PC_GAR_DECES_1 != 0.0:
                    MT_GAR_DECES_PROJ = MT_GAR_DECES_PROJ * PC_GAR_DECES_2 / PC_GAR_DECES_1
                PC_GAR_DECES_1 = PC_GAR_DECES_2

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

            # === SAVE DEBUG FLUX OUTPUT (for single account/scenario) ===
            if debug_flux_output is not None:
                # Only capture flux for the debug account and scenario
                is_debug_acct_scn = (
                    (debug_account < 0 or account_idx == debug_account) and
                    (debug_scenario < 0 or scenario_idx == debug_scenario)
                )
                if is_debug_acct_scn and an_eval < debug_flux_output.shape[0] and mois_eval < debug_flux_output.shape[1]:
                    # Cashflow components
                    debug_flux_output[an_eval, mois_eval, FLUX_COMP_IDX_PRIMES_GARANTIES] = PRIMES_GARANTIES
                    debug_flux_output[an_eval, mois_eval, FLUX_COMP_IDX_PREST_DECES] = PREST_DECES
                    debug_flux_output[an_eval, mois_eval, FLUX_COMP_IDX_PREST_ECH] = PREST_ECH
                    debug_flux_output[an_eval, mois_eval, FLUX_COMP_IDX_PREST_MRV] = PREST_MRV
                    debug_flux_output[an_eval, mois_eval, FLUX_COMP_IDX_FRAIS_ACQUIS] = frais_acquis
                    debug_flux_output[an_eval, mois_eval, FLUX_COMP_IDX_COMM_VENTE] = comm_vente
                    debug_flux_output[an_eval, mois_eval, FLUX_COMP_IDX_PRIMES_VARIABLES] = PRIMES_VARIABLES
                    debug_flux_output[an_eval, mois_eval, FLUX_COMP_IDX_FRAIS_FIXES] = FRAIS_FIXES
                    debug_flux_output[an_eval, mois_eval, FLUX_COMP_IDX_HON_GEST] = HON_GEST
                    debug_flux_output[an_eval, mois_eval, FLUX_COMP_IDX_COMM_MAINTIEN] = COMM_MAINTIEN
                    debug_flux_output[an_eval, mois_eval, FLUX_COMP_IDX_VALEUR_MARCHANDE] = VALEUR_MARCHANDE
                    debug_flux_output[an_eval, mois_eval, FLUX_COMP_IDX_PASSIF_REDRESSE] = PASSIF_REDRESSE
                    debug_flux_output[an_eval, mois_eval, FLUX_COMP_IDX_COUSSIN_CREDIT] = COUSSIN_CREDIT
                    debug_flux_output[an_eval, mois_eval, FLUX_COMP_IDX_COUSSIN_MARCHE] = COUSSIN_MARCHE
                    debug_flux_output[an_eval, mois_eval, FLUX_COMP_IDX_COUSSIN_DEPENSE] = COUSSIN_DEPENSE
                    debug_flux_output[an_eval, mois_eval, FLUX_COMP_IDX_COUSSIN_DECHEANCE] = COUSSIN_DECHEANCE
                    debug_flux_output[an_eval, mois_eval, FLUX_COMP_IDX_COUSSIN_MORTALITE] = COUSSIN_MORTALITE
                    debug_flux_output[an_eval, mois_eval, FLUX_COMP_IDX_COUSSIN_DEPOT] = COUSSIN_DEPOT
                    # Detailed calculation fields
                    debug_flux_output[an_eval, mois_eval, FLUX_COMP_IDX_MT_VM] = MT_VM_PROJ
                    debug_flux_output[an_eval, mois_eval, FLUX_COMP_IDX_MT_VM_AV_RETRAIT] = MT_VM_AV_RETRAIT
                    debug_flux_output[an_eval, mois_eval, FLUX_COMP_IDX_MT_VM_AP_RETRAIT] = MT_VM_AP_RETRAIT
                    debug_flux_output[an_eval, mois_eval, FLUX_COMP_IDX_AGE] = float(age)
                    debug_flux_output[an_eval, mois_eval, FLUX_COMP_IDX_QX] = qx
                    debug_flux_output[an_eval, mois_eval, FLUX_COMP_IDX_LAPSE_TOT] = LAPSE_TOT
                    debug_flux_output[an_eval, mois_eval, FLUX_COMP_IDX_LAPSE_PART] = LAPSE_PART
                    debug_flux_output[an_eval, mois_eval, FLUX_COMP_IDX_TX_SURVIE] = TX_SURVIE
                    debug_flux_output[an_eval, mois_eval, FLUX_COMP_IDX_RETRAIT] = RETRAIT
                    debug_flux_output[an_eval, mois_eval, FLUX_COMP_IDX_DEPOT_FUTUR] = depot_futur
                    debug_flux_output[an_eval, mois_eval, FLUX_COMP_IDX_MT_GAR_DECES] = MT_GAR_DECES_PROJ
                    debug_flux_output[an_eval, mois_eval, FLUX_COMP_IDX_MT_GAR_ECH] = MT_GAR_ECH_PROJ
                    debug_flux_output[an_eval, mois_eval, FLUX_COMP_IDX_MT_SRG] = MT_SRG_PROJ
                    debug_flux_output[an_eval, mois_eval, FLUX_COMP_IDX_REND_SP500] = r_sp500
                    debug_flux_output[an_eval, mois_eval, FLUX_COMP_IDX_REND_TSX] = r_tsx
                    debug_flux_output[an_eval, mois_eval, FLUX_COMP_IDX_REND_EAFE] = r_eafe
                    debug_flux_output[an_eval, mois_eval, FLUX_COMP_IDX_REND_DEX] = r_dex
                    debug_flux_output[an_eval, mois_eval, FLUX_COMP_IDX_REND_MM] = r_mm
                    debug_flux_output[an_eval, mois_eval, FLUX_COMP_IDX_MT_SP500] = MT_SP500_PROJ
                    debug_flux_output[an_eval, mois_eval, FLUX_COMP_IDX_MT_TSX] = MT_TSX_PROJ
                    debug_flux_output[an_eval, mois_eval, FLUX_COMP_IDX_MT_EAFE] = MT_EAFE_PROJ
                    debug_flux_output[an_eval, mois_eval, FLUX_COMP_IDX_MT_DEX] = MT_DEX_PROJ
                    debug_flux_output[an_eval, mois_eval, FLUX_COMP_IDX_MT_MM] = MT_MM_PROJ
                    debug_flux_output[an_eval, mois_eval, FLUX_COMP_IDX_CAT_COUSSIN_1] = float(cat_coussin_1)
                    debug_flux_output[an_eval, mois_eval, FLUX_COMP_IDX_CAT_COUSSIN_2] = float(cat_coussin_2)

            # === CALCULATE ALL SAS CASHFLOW VARIABLES (every month) ===
            # Present value calculations (matching SAS lines 542, 560, 599, 618, 724, 727, 736, 741, 745, 754, 785, 868-874)
            VP_PRIMES_GARANTIES = PRIMES_GARANTIES * TX_ACTUALISATION
            VP_PREST_DECES = PREST_DECES * TX_ACTUALISATION
            VP_PREST_ECH = PREST_ECH * TX_ACTUALISATION
            VP_PREST_MRV = PREST_MRV * TX_ACTUALISATION
            VP_FRAIS_ACQUIS = frais_acquis * TX_ACTUALISATION
            VP_COMM_VENTE = comm_vente * TX_ACTUALISATION
            VP_FRAIS_FIXES = FRAIS_FIXES * TX_ACTUALISATION
            VP_HON_GEST = HON_GEST * TX_ACTUALISATION
            VP_COMM_MAINTIEN = COMM_MAINTIEN * TX_ACTUALISATION
            VP_PRIMES_VARIABLES = PRIMES_VARIABLES * TX_ACTUALISATION
            VP_VALEUR_MARCHANDE = VALEUR_MARCHANDE * TX_ACTUALISATION / freq_eval
            
            # Cushion present values (SAS lines 868-874)
            VP_PASSIF_REDRESSE = PASSIF_REDRESSE * TX_ACTUALISATION / freq_eval
            VP_COUSSIN_CREDIT = COUSSIN_CREDIT * TX_ACTUALISATION / freq_eval
            VP_COUSSIN_MARCHE = COUSSIN_MARCHE * TX_ACTUALISATION / freq_eval
            VP_COUSSIN_DEPENSE = COUSSIN_DEPENSE * TX_ACTUALISATION / freq_eval
            VP_COUSSIN_DECHEANCE = COUSSIN_DECHEANCE * TX_ACTUALISATION / freq_eval
            VP_COUSSIN_MORTALITE = COUSSIN_MORTALITE * TX_ACTUALISATION / freq_eval
            VP_COUSSIN_DEPOT = COUSSIN_DEPOT * TX_ACTUALISATION / freq_eval
            
            # Additional SAS variables (lines 768, 774, 780, 781, 791)
            # UNITE_COUVERTURE = MAX(MT_VM_PROJ, MT_GAR_DECES_PROJ+MT_BONI_DECES_PROJ, RETRAIT) * TX_SURVIE
            uc_max = MT_VM_PROJ
            if MT_GAR_DECES_PROJ + MT_BONI_DECES_PROJ > uc_max:
                uc_max = MT_GAR_DECES_PROJ + MT_BONI_DECES_PROJ
            if RETRAIT > uc_max:
                uc_max = RETRAIT
            UNITE_COUVERTURE = uc_max * TX_SURVIE
            
            VALEUR_GARANTIE = MT_GAR_DECES_PROJ * TX_SURVIE
            DEPOT_FUTUR_SURVIE = depot_futur * TX_SURVIE
            
            # REM_COMP_INV = ((RETRAIT - PREST_MRV) + MT_VM_AP_RETRAIT_DEPOT * (1 - TX_SURVIE/TX_SURVIE_DEB)) * TX_SURVIE_DEB
            tx_ratio = TX_SURVIE / TX_SURVIE_DEB if TX_SURVIE_DEB != 0.0 else 0.0
            REM_COMP_INV = ((RETRAIT + PREST_MRV) + MT_VM_PROJ * (1.0 - tx_ratio)) * TX_SURVIE_DEB
            
            # === WRITE MONTHLY CASHFLOWS (one row per month) ===
            # Output monthly data to match SAS FLUX_PROJETES format
            if an_eval >= 1 and an_eval <= n_years:
                # Calculate monthly index: (year-1)*12 + (month-1)
                out_idx = (an_eval - 1) * 12 + (mois_eval - 1)
                if out_idx < output_cashflows.shape[2]:
                    # Non-discounted cashflows (write monthly values)
                    output_cashflows[account_idx, scenario_idx, out_idx, CF_OUT_IDX_FRAIS_ACQUIS] = frais_acquis
                    output_cashflows[account_idx, scenario_idx, out_idx, CF_OUT_IDX_COMM_VENTE] = comm_vente
                    output_cashflows[account_idx, scenario_idx, out_idx, CF_OUT_IDX_PRIMES_GARANTIES] = PRIMES_GARANTIES
                    output_cashflows[account_idx, scenario_idx, out_idx, CF_OUT_IDX_PRIMES_VARIABLES] = PRIMES_VARIABLES
                    output_cashflows[account_idx, scenario_idx, out_idx, CF_OUT_IDX_FRAIS_FIXES] = FRAIS_FIXES
                    output_cashflows[account_idx, scenario_idx, out_idx, CF_OUT_IDX_HON_GEST] = HON_GEST
                    output_cashflows[account_idx, scenario_idx, out_idx, CF_OUT_IDX_COMM_MAINTIEN] = COMM_MAINTIEN
                    output_cashflows[account_idx, scenario_idx, out_idx, CF_OUT_IDX_PREST_ECH] = PREST_ECH
                    output_cashflows[account_idx, scenario_idx, out_idx, CF_OUT_IDX_PREST_MRV] = PREST_MRV
                    output_cashflows[account_idx, scenario_idx, out_idx, CF_OUT_IDX_PREST_DECES] = PREST_DECES
                    
                    # Present value cashflows (write monthly values)
                    output_cashflows[account_idx, scenario_idx, out_idx, CF_OUT_IDX_VP_FRAIS_ACQUIS] = VP_FRAIS_ACQUIS
                    output_cashflows[account_idx, scenario_idx, out_idx, CF_OUT_IDX_VP_COMM_VENTE] = VP_COMM_VENTE
                    output_cashflows[account_idx, scenario_idx, out_idx, CF_OUT_IDX_VP_PRIMES_GARANTIES] = VP_PRIMES_GARANTIES
                    output_cashflows[account_idx, scenario_idx, out_idx, CF_OUT_IDX_VP_PRIMES_VARIABLES] = VP_PRIMES_VARIABLES
                    output_cashflows[account_idx, scenario_idx, out_idx, CF_OUT_IDX_VP_FRAIS_FIXES] = VP_FRAIS_FIXES
                    output_cashflows[account_idx, scenario_idx, out_idx, CF_OUT_IDX_VP_HON_GEST] = VP_HON_GEST
                    output_cashflows[account_idx, scenario_idx, out_idx, CF_OUT_IDX_VP_COMM_MAINTIEN] = VP_COMM_MAINTIEN
                    output_cashflows[account_idx, scenario_idx, out_idx, CF_OUT_IDX_VP_PREST_ECH] = VP_PREST_ECH
                    output_cashflows[account_idx, scenario_idx, out_idx, CF_OUT_IDX_VP_PREST_MRV] = VP_PREST_MRV
                    output_cashflows[account_idx, scenario_idx, out_idx, CF_OUT_IDX_VP_PREST_DECES] = VP_PREST_DECES
                    output_cashflows[account_idx, scenario_idx, out_idx, CF_OUT_IDX_VP_VALEUR_MARCHANDE] = VP_VALEUR_MARCHANDE
                    
                    # Coverage and values (write monthly values)
                    output_cashflows[account_idx, scenario_idx, out_idx, CF_OUT_IDX_UNITE_COUVERTURE] = UNITE_COUVERTURE
                    output_cashflows[account_idx, scenario_idx, out_idx, CF_OUT_IDX_DEPOT_FUTUR] = depot_futur
                    output_cashflows[account_idx, scenario_idx, out_idx, CF_OUT_IDX_REM_COMP_INV] = REM_COMP_INV
                    output_cashflows[account_idx, scenario_idx, out_idx, CF_OUT_IDX_VALEUR_MARCHANDE] = VALEUR_MARCHANDE
                    output_cashflows[account_idx, scenario_idx, out_idx, CF_OUT_IDX_VALEUR_GARANTIE] = VALEUR_GARANTIE
                    output_cashflows[account_idx, scenario_idx, out_idx, CF_OUT_IDX_DEPOT_FUTUR_SURVIE] = DEPOT_FUTUR_SURVIE
                    
                    # Cushions non-discounted (write monthly values)
                    output_cashflows[account_idx, scenario_idx, out_idx, CF_OUT_IDX_PASSIF_REDRESSE] = PASSIF_REDRESSE
                    output_cashflows[account_idx, scenario_idx, out_idx, CF_OUT_IDX_COUSSIN_CREDIT] = COUSSIN_CREDIT
                    output_cashflows[account_idx, scenario_idx, out_idx, CF_OUT_IDX_COUSSIN_MARCHE] = COUSSIN_MARCHE
                    output_cashflows[account_idx, scenario_idx, out_idx, CF_OUT_IDX_COUSSIN_DEPENSE] = COUSSIN_DEPENSE
                    output_cashflows[account_idx, scenario_idx, out_idx, CF_OUT_IDX_COUSSIN_DECHEANCE] = COUSSIN_DECHEANCE
                    output_cashflows[account_idx, scenario_idx, out_idx, CF_OUT_IDX_COUSSIN_MORTALITE] = COUSSIN_MORTALITE
                    output_cashflows[account_idx, scenario_idx, out_idx, CF_OUT_IDX_COUSSIN_DEPOT] = COUSSIN_DEPOT
                    
                    # Cushions present value (write monthly values)
                    output_cashflows[account_idx, scenario_idx, out_idx, CF_OUT_IDX_VP_PASSIF_REDRESSE] = VP_PASSIF_REDRESSE
                    output_cashflows[account_idx, scenario_idx, out_idx, CF_OUT_IDX_VP_COUSSIN_CREDIT] = VP_COUSSIN_CREDIT
                    output_cashflows[account_idx, scenario_idx, out_idx, CF_OUT_IDX_VP_COUSSIN_MARCHE] = VP_COUSSIN_MARCHE
                    output_cashflows[account_idx, scenario_idx, out_idx, CF_OUT_IDX_VP_COUSSIN_DEPENSE] = VP_COUSSIN_DEPENSE
                    output_cashflows[account_idx, scenario_idx, out_idx, CF_OUT_IDX_VP_COUSSIN_DECHEANCE] = VP_COUSSIN_DECHEANCE
                    output_cashflows[account_idx, scenario_idx, out_idx, CF_OUT_IDX_VP_COUSSIN_MORTALITE] = VP_COUSSIN_MORTALITE
                    output_cashflows[account_idx, scenario_idx, out_idx, CF_OUT_IDX_VP_COUSSIN_DEPOT] = VP_COUSSIN_DEPOT

            # === SAVE STATE TO GLOBAL MEMORY (only at year-end) ===
            if mois_eval == 12 and an_eval >= 1 and an_eval <= n_years:
                out_idx = an_eval - 1
                if out_idx < output_states.shape[2]:
                    output_states[account_idx, scenario_idx, out_idx, STATE_IDX_MT_VM] = MT_VM_PROJ
                    output_states[account_idx, scenario_idx, out_idx, STATE_IDX_MT_GAR_DECES] = MT_GAR_DECES_PROJ
                    output_states[account_idx, scenario_idx, out_idx, STATE_IDX_MT_GAR_ECH] = MT_GAR_ECH_PROJ
                    output_states[account_idx, scenario_idx, out_idx, STATE_IDX_MT_SRG] = MT_SRG_PROJ
                    output_states[account_idx, scenario_idx, out_idx, STATE_IDX_AGE] = float(age)
                    output_states[account_idx, scenario_idx, out_idx, STATE_IDX_TX_SURVIE] = TX_SURVIE
                    output_states[account_idx, scenario_idx, out_idx, STATE_IDX_MT_DEX] = MT_DEX_PROJ
                    output_states[account_idx, scenario_idx, out_idx, STATE_IDX_MT_MM] = MT_MM_PROJ
                    output_states[account_idx, scenario_idx, out_idx, STATE_IDX_MT_TSX] = MT_TSX_PROJ
                    output_states[account_idx, scenario_idx, out_idx, STATE_IDX_MT_SP500] = MT_SP500_PROJ
                    output_states[account_idx, scenario_idx, out_idx, STATE_IDX_MT_EAFE] = MT_EAFE_PROJ
                    output_states[account_idx, scenario_idx, out_idx, STATE_IDX_MT_BONI_DECES] = MT_BONI_DECES_PROJ
                    output_states[account_idx, scenario_idx, out_idx, STATE_IDX_MT_BCB] = MT_BCB_PROJ
                    output_states[account_idx, scenario_idx, out_idx, STATE_IDX_MT_MRV_MRG_MRA] = MT_MRV_MRG_MRA_PROJ
                    output_states[account_idx, scenario_idx, out_idx, STATE_IDX_TAUX_MRV_MRG_MRA] = TAUX_MRV_MRG_MRA_PROJ
                    output_states[account_idx, scenario_idx, out_idx, STATE_IDX_ANNEE_ECH] = float(ANNEE_ECH_PROJ)
                    output_states[account_idx, scenario_idx, out_idx, STATE_IDX_MOIS_ECH] = float(MOIS_ECH_PROJ)
                    output_states[account_idx, scenario_idx, out_idx, STATE_IDX_TX_ACTUALISATION] = TX_ACTUALISATION
                    output_states[account_idx, scenario_idx, out_idx, STATE_IDX_ANNEE_REELLE] = float(annee_reelle)
                    output_states[account_idx, scenario_idx, out_idx, STATE_IDX_MOIS_EVAL] = float(mois_eval)
                    output_states[account_idx, scenario_idx, out_idx, STATE_IDX_PC_GAR_DECES_1] = PC_GAR_DECES_1

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



@cuda.jit
def nested_valuation_kernel_five_chocs(
        input_states,        # StatesTensor: (batch, ext_scenarios, years, STATE_SIZE)
        account_data,        # AccountData: (n_accounts, n_fields)
        n_internal_scenarios,  # int
        n_internal_years,    # int
        rn_returns_lookups,  # RNReturnsLookups: 6 arrays
        mortality_lookup,    # MortalityLookup: (sex, age, year, product)
        lapse_lookups,       # LapseLookups: 6 arrays
        policy_lookups,      # PolicyLookups: 5 arrays
        commission_lookups,  # CommissionLookups: 6 arrays
        output_metrics,      # MetricsTensor: (batch, ext_scenarios, years, NUM_CHOCS, 2)
        debug_output=None,   # Optional: (NUM_CHOCS, INT_DEBUG_SIZE) - one row per choc for filtered debug
        debug_ts_output=None,  # Optional: (NUM_CHOCS, n_internal_years, INT_TS_DEBUG_IDX_SIZE) - time series for filtered debug
        debug_int_scenario=-1,  # Internal scenario to debug (-1 = disabled, logs all)
        debug_int_year=-1,      # Internal year to debug (-1 = disabled, logs all)
        debug_account=-1,       # Account index to debug (-1 = disabled)
        debug_ext_scenario=-1,  # External scenario to debug (-1 = disabled)
        debug_ext_year=-1,      # External year to debug (-1 = disabled)
        shock_capital_pct=0.35,
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

    # Unpack lookups
    min_ferr_lookup, lapse_part_min, lapse_part_max, lapse_tot_min, lapse_tot_max, lapse_tot_fact = lapse_lookups
    deposits_pc, deposits_var, deposits_age_max, deposits_i_even, fees_lookup = policy_lookups
    acq_vente_rf, acq_vente_ac, acq_maintien_rf, acq_maintien_ac, acq_frais_ac, acq_frais_rf = commission_lookups

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
    start_bcb = input_states[acc_idx, scn_idx, year_idx, STATE_IDX_MT_BCB]
    start_mrv = input_states[acc_idx, scn_idx, year_idx, STATE_IDX_MT_MRV_MRG_MRA]
    start_taux_mrv = input_states[acc_idx, scn_idx, year_idx, STATE_IDX_TAUX_MRV_MRG_MRA]
    start_annee_ech = int(input_states[acc_idx, scn_idx, year_idx, STATE_IDX_ANNEE_ECH])
    start_mois_ech = int(input_states[acc_idx, scn_idx, year_idx, STATE_IDX_MOIS_ECH])
    start_tx_actualisation = input_states[acc_idx, scn_idx, year_idx, STATE_IDX_TX_ACTUALISATION]
    start_annee_reelle = int(input_states[acc_idx, scn_idx, year_idx, STATE_IDX_ANNEE_REELLE])
    start_mois_eval = int(input_states[acc_idx, scn_idx, year_idx, STATE_IDX_MOIS_EVAL])
    start_pc_gar_deces_1 = input_states[acc_idx, scn_idx, year_idx, STATE_IDX_PC_GAR_DECES_1]

    # Check if policy is active
    if start_vm <= 0 or start_tx_survie <= 0:
        # Zero out all chocs
        for choc in range(NUM_CHOCS):
            output_metrics[acc_idx, scn_idx, year_idx, choc, METRICS_RESERVE_IDX] = 0.0
            output_metrics[acc_idx, scn_idx, year_idx, choc, METRICS_CAPITAL_IDX] = 0.0
        return

    # Load account parameters using module-level constants
    acc = account_data[acc_idx]
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
    PC_HONORAIRES_GEST = acc[ACCOUNT_IDX_PC_HONORAIRES_GEST]
    PC_FRAIS_GARANTIE = acc[ACCOUNT_IDX_PC_FRAIS_GARANTIE]
    PC_RFG = acc[ACCOUNT_IDX_PC_RFG]
    PC_REVENU_FDS = acc[ACCOUNT_IDX_PC_REVENU_FDS]
    PC_GAR_ECH = acc[ACCOUNT_IDX_PC_GAR_ECH]
    PC_GAR_ECH_DEP_FUT = acc[ACCOUNT_IDX_PC_GAR_ECH_DEP_FUT]
    AJUSTEMENT_COMMISSION = acc[ACCOUNT_IDX_AJUSTEMENT_COMMISSION]
    MT_RF = acc[ACCOUNT_IDX_MT_RF]
    I_FRAIS_SUR_SRG = int(acc[ACCOUNT_IDX_I_FRAIS_SUR_SRG])
    ANNEE_COTIS = int(acc[ACCOUNT_IDX_ANNEE_COTIS]) if acc[ACCOUNT_IDX_ANNEE_COTIS] > 0 else start_annee_reelle
    MOIS_COTIS = int(acc[ACCOUNT_IDX_MOIS_COTIS]) if acc[ACCOUNT_IDX_MOIS_COTIS] > 0 else 1
    PC_BONI_DECES = acc[ACCOUNT_IDX_PC_BONI_DECES]
    MAX_BONI_DECES = int(acc[ACCOUNT_IDX_MAX_BONI_DECES])
    AJUSTEMENT_MENSUEL_GAR = acc[ACCOUNT_IDX_AJUSTEMENT_MENSUEL_GAR]
    PC_GAR_DECES_2 = acc[ACCOUNT_IDX_PC_GAR_DECES_2]
    AGE_CHANG_DECES = int(acc[ACCOUNT_IDX_AGE_CHANG_DECES])
    FREQ_RESET_DECES = int(acc[ACCOUNT_IDX_FREQ_RESET_DECES])
    MAX_RESET_DECES = int(acc[ACCOUNT_IDX_MAX_RESET_DECES])
    I_RESET_DECES_ECH = int(acc[ACCOUNT_IDX_I_RESET_DECES_ECH])
    NB_AN_ECH = int(acc[ACCOUNT_IDX_NB_AN_ECH])
    PC_RENOUV_ECH = acc[ACCOUNT_IDX_PC_RENOUV_ECH]
    AGE_MAX_RENOUV_ECH = int(acc[ACCOUNT_IDX_AGE_MAX_RENOUV_ECH])
    MAX_RESET_FACUL_ECH = int(acc[ACCOUNT_IDX_MAX_RESET_FACUL_ECH])
    RATIO_VM_VG_RESET_ECH = acc[ACCOUNT_IDX_RATIO_VM_VG_RESET_ECH]
    AGE_MRV_PERMIS = int(acc[ACCOUNT_IDX_AGE_MRV_PERMIS])
    PC_BONI_SRG = acc[ACCOUNT_IDX_PC_BONI_SRG]
    FREQ_RESET_SRG = int(acc[ACCOUNT_IDX_FREQ_RESET_SRG])
    MAX_RESET_SRG = int(acc[ACCOUNT_IDX_MAX_RESET_SRG])
    TABLE_TAUX_MRV_MRG_MRA = int(acc[ACCOUNT_IDX_TABLE_TAUX_MRV_MRG_MRA])
    MT_TPA_RETRAIT = acc[ACCOUNT_IDX_MT_TPA_RETRAIT]
    M_MT_MRV_EXCEDENT = acc[ACCOUNT_IDX_M_MT_MRV_EXCEDENT]
    MT_TPA_DEPOT = acc[ACCOUNT_IDX_MT_TPA_DEPOT]
    VAR_RETRAIT_FCT = int(acc[ACCOUNT_IDX_VAR_RETRAIT_FCT])
    PC_RETRAIT_AGE = acc[ACCOUNT_IDX_PC_RETRAIT_AGE]
    MT_RETRAIT_MAX = acc[ACCOUNT_IDX_MT_RETRAIT_MAX]
    I_RESET_FACUL_ECH = int(acc[ACCOUNT_IDX_I_RESET_FACUL_ECH])

    FREQ_EVAL_INT = 1.0
    AJUST_NOUV_AFFAIRES_INT = 1.0  # SAS line 312: always 1 for mid-period adjustment

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

        sum_pv_reserve_choc = 0.0
        sum_pv_capital_choc = 0.0
        
        # Variables to capture debug values at specific internal iteration
        debug_curr_vm = 0.0
        debug_fees = 0.0
        debug_pv_path = 0.0
        debug_r_portfolio = 0.0
        debug_fwd_rate = 0.0

        # Pre-compute fee factor (constant across iterations)
        fee_factor = 1.0 - (PC_RFG / FREQ_EVAL_INT)
        fee_rate = PC_RFG / FREQ_EVAL_INT

        for i_int in range(n_internal_scenarios):
            scn_idx_int = i_int % n_scn_avail if n_scn_avail > 0 else 0

            pv_reserve = 0.0
            pv_capital = 0.0

            for do_capital in range(2):
                curr_mt_dex = mt_dex_choc
                curr_mt_mm = mt_mm_choc
                curr_mt_tsx = mt_tsx_choc
                curr_mt_sp500 = mt_sp500_choc
                curr_mt_eafe = mt_eafe_choc

                if do_capital == 1 and shock_capital_pct > 0.0:
                    shock_factor = 1.0 - shock_capital_pct
                    curr_mt_dex = curr_mt_dex * shock_factor
                    curr_mt_mm = curr_mt_mm * shock_factor
                    curr_mt_tsx = curr_mt_tsx * shock_factor
                    curr_mt_sp500 = curr_mt_sp500 * shock_factor
                    curr_mt_eafe = curr_mt_eafe * shock_factor

                curr_vm = curr_mt_dex + curr_mt_mm + curr_mt_tsx + curr_mt_sp500 + curr_mt_eafe
                pv_path = 0.0
                tx_actu = start_tx_actualisation
                TX_SURVIE = start_tx_survie

                MT_GAR_DECES_PROJ = start_gar_deces
                MT_GAR_ECH_PROJ = start_gar_ech
                MT_BONI_DECES_PROJ = start_boni_deces
                MT_SRG_PROJ = start_srg
                MT_BCB_PROJ = start_bcb
                MT_MRV_MRG_MRA_PROJ = start_mrv
                TAUX_MRV_MRG_MRA_PROJ = start_taux_mrv
                ANNEE_ECH_PROJ = start_annee_ech
                MOIS_ECH_PROJ = start_mois_ech
                pc_gar_deces_1 = start_pc_gar_deces_1

                if curr_vm > 0.0:
                    w_sp500 = curr_mt_sp500 / curr_vm
                    w_tsx = curr_mt_tsx / curr_vm
                    w_eafe = curr_mt_eafe / curr_vm
                    w_dex = curr_mt_dex / curr_vm
                    w_mm = curr_mt_mm / curr_vm
                else:
                    w_sp500 = 0.0
                    w_tsx = 0.0
                    w_eafe = 0.0
                    w_dex = 0.0
                    w_mm = 0.0

                for t_int in range(n_internal_years):
                    if curr_vm <= 0.0 or TX_SURVIE <= 0.0:
                        break

                    annee_reelle = start_annee_reelle + t_int + 1
                    mois_eval = 12
                    age = int(start_age) + t_int + 1
                    if age < 1:
                        age = 1
                    if age > AGE_FIN_CONTRAT:
                        break

                    MT_VM_PROJ = curr_vm

                    t_idx_int = t_int % n_yr_avail if n_yr_avail > 0 else 0

                    if n_scn_avail > 0 and n_yr_avail > 0:
                        r_dex = rn_rend_dex[scn_idx_int, t_idx_int]
                        r_mm = rn_rend_mm[scn_idx_int, t_idx_int]
                        r_tsx = rn_rend_tsx[scn_idx_int, t_idx_int]
                        r_sp500 = rn_rend_sp500[scn_idx_int, t_idx_int]
                        r_eafe = rn_rend_eafe[scn_idx_int, t_idx_int]
                        fwd = rn_forward_rate[scn_idx_int, t_idx_int]
                        ajust_fwd = rn_ajust_forward[scn_idx_int, t_idx_int]
                    else:
                        r_dex = DEFAULT_RETURN_RATE
                        r_mm = DEFAULT_RETURN_RATE
                        r_tsx = DEFAULT_RETURN_RATE
                        r_sp500 = DEFAULT_RETURN_RATE
                        r_eafe = DEFAULT_RETURN_RATE
                        fwd = DEFAULT_FORWARD_RATE
                        ajust_fwd = 0.0

                    if MT_VM_PROJ == 0.0:
                        fwd = fwd + ajust_fwd

                    curr_mt_sp500 = curr_mt_sp500 * math.exp(r_sp500)
                    curr_mt_tsx = curr_mt_tsx * math.exp(r_tsx)
                    curr_mt_eafe = curr_mt_eafe * math.exp(r_eafe)
                    curr_mt_dex = curr_mt_dex * math.exp(r_dex)
                    curr_mt_mm = curr_mt_mm * math.exp(r_mm)

                    MT_VM_AV_RETRAIT_FRAIS = curr_mt_sp500 + curr_mt_tsx + curr_mt_eafe + curr_mt_dex + curr_mt_mm

                    tx_actu = tx_actu * math.exp(-fwd)

                    TX_SURVIE_DEB = TX_SURVIE

                    current_date = annee_reelle + mois_eval / 12.0
                    issue_date = ANNEE_COTIS + MOIS_COTIS / 12.0
                    duree = int(current_date - issue_date) + 1
                    if duree < 1:
                        duree = 1
                    duree_max10 = duree
                    if duree_max10 > 10:
                        duree_max10 = 10

                    month_diff = MOIS_NAIS - mois_eval
                    if month_diff <= 0:
                        month_diff = month_diff + 12
                    age_mort = age + 1 if month_diff <= MORTALITY_AGE_ADJUSTMENT_THRESHOLD else age
                    if age_mort >= mortality_lookup.shape[1]:
                        age_mort = mortality_lookup.shape[1] - 1

                    if (I_SEXE < mortality_lookup.shape[0] and
                            annee_reelle < mortality_lookup.shape[2] and
                            I_PRODUIT_REGR < mortality_lookup.shape[3]):
                        qx = mortality_lookup[I_SEXE, age_mort, annee_reelle, I_PRODUIT_REGR]
                    else:
                        qx = DEFAULT_MORTALITY_RATE
                    qx = 1.0 - math.pow(1.0 - qx, 1.0 / FREQ_EVAL_INT)

                    LAPSE_TOT = 0.0
                    LAPSE_PART = 0.0
                    VM_VG_RATIO = 0.0

                    if MT_VM_PROJ == 0.0:
                        LAPSE_TOT = 0.0
                        LAPSE_PART = 0.0
                    else:
                        vm_mid_period = (MT_VM_PROJ + MT_VM_AV_RETRAIT_FRAIS) / 2.0
                        pc_gar_ech_ratio = PC_GAR_ECH / max(MT_GAR_ECH_PROJ, MIN_GUARANTEE_VALUE) if MT_GAR_ECH_PROJ > 0 else 1.0
                        pc_gar_deces_ratio = pc_gar_deces_1 / max(MT_BONI_DECES_PROJ + MT_GAR_DECES_PROJ, MIN_GUARANTEE_VALUE) if (MT_BONI_DECES_PROJ + MT_GAR_DECES_PROJ) > 0 else 1.0
                        srg_ratio = 1.0 / max(MT_SRG_PROJ, MIN_GUARANTEE_VALUE) if MT_SRG_PROJ > 0 else 1.0
                        min_ratio = min(pc_gar_ech_ratio, pc_gar_deces_ratio, srg_ratio)
                        VM_VG_RATIO = min(VM_VG_RATIO_MAX, vm_mid_period * min_ratio)

                        if VM_VG_RATIO <= VM_VG_RATIO_LEVEL1_THRESHOLD:
                            LAPSE_NIV_TOT = LAPSE_LEVEL_1
                            LAPSE_NIV_PART = LAPSE_LEVEL_1
                        elif VM_VG_RATIO <= VM_VG_RATIO_LEVEL2_THRESHOLD:
                            LAPSE_NIV_TOT = LAPSE_LEVEL_2
                            LAPSE_NIV_PART = LAPSE_LEVEL_2
                        else:
                            LAPSE_NIV_TOT = LAPSE_LEVEL_3
                            LAPSE_NIV_PART = LAPSE_LEVEL_3

                        if (duree_max10 < lapse_tot_min.shape[0] and ID_LAPSE < lapse_tot_min.shape[1] and
                                LAPSE_NIV_TOT < lapse_tot_min.shape[2]):
                            tx_lapse_tot_min = lapse_tot_min[duree_max10, ID_LAPSE, LAPSE_NIV_TOT]
                            tx_lapse_tot_max = lapse_tot_max[duree_max10, ID_LAPSE, LAPSE_NIV_TOT]
                            fact_dim = lapse_tot_fact[duree_max10, ID_LAPSE, LAPSE_NIV_TOT]
                        else:
                            tx_lapse_tot_min = tx_lapse_tot_max = DEFAULT_LAPSE_RATE_TOT
                            fact_dim = DEFAULT_LAPSE_FACT_DIM

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

                        if age >= AGE_DECAISSEMENT:
                            LAPSE_TOT = LAPSE_TOT * fact_dim

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
                    lapse = 1.0 - math.pow(1.0 - LAPSE_TOT - LAPSE_PART, 1.0 / FREQ_EVAL_INT * AJUST_NOUV_AFFAIRES_INT)

                    TX_SURVIE = TX_SURVIE * (1.0 - qx) * (1.0 - lapse)

                    if PC_BONI_DECES > 0.0 and age < MAX_BONI_DECES:
                        MT_BONI_DECES_PROJ = MT_BONI_DECES_PROJ + (MT_GAR_DECES_PROJ * PC_BONI_DECES / FREQ_EVAL_INT)
                    else:
                        MT_BONI_DECES_PROJ = 0.0

                    MT_VM_AV_RETRAIT = MT_VM_AV_RETRAIT_FRAIS * math.exp(-PC_RFG / FREQ_EVAL_INT)

                    guarantee_fee_amount = 0.0
                    if PC_FRAIS_GARANTIE > 0.0:
                        base_fee_calc = MT_VM_AV_RETRAIT if I_FRAIS_SUR_SRG == 0 else MT_SRG_PROJ
                        guarantee_fee_amount = base_fee_calc * PC_FRAIS_GARANTIE / FREQ_EVAL_INT
                        if guarantee_fee_amount > MT_VM_AV_RETRAIT:
                            guarantee_fee_amount = MT_VM_AV_RETRAIT
                        MT_VM_AV_RETRAIT = MT_VM_AV_RETRAIT - guarantee_fee_amount
                        if MT_VM_AV_RETRAIT < 0.0:
                            MT_VM_AV_RETRAIT = 0.0

                    PRIMES_GARANTIES = guarantee_fee_amount * TX_SURVIE_DEB
                    VP_PRIMES_GARANTIES = PRIMES_GARANTIES * tx_actu

                    if age < min_ferr_lookup.shape[0]:
                        min_ferr_rate = min_ferr_lookup[age]
                    else:
                        min_ferr_rate = DEFAULT_FERR_MIN_RATE
                    MT_MIN_FERR_PROJ = MT_VM_PROJ * min_ferr_rate

                    AGE_RETRAIT = age + 1

                    if I_PRODUIT_REGR == 1:
                        base_amount = MT_SRG_PROJ if TABLE_TAUX_MRV_MRG_MRA == 1 else MT_VM_PROJ
                        if AGE_RETRAIT < AGE_MRV_PERMIS and base_amount == 0.0:
                            MT_MRV_MRG_MRA_PROJ = 0.0

                        if TABLE_TAUX_MRV_MRG_MRA == 2:
                            max_age_start = AGE_MRV_PERMIS if AGE_MRV_PERMIS > AGE_DECAISSEMENT else AGE_DECAISSEMENT
                            should_reinit = (AGE_RETRAIT == max_age_start or (MT_SRG_PROJ == MT_VM_PROJ and MT_VM_PROJ != 0.0))
                            if should_reinit:
                                if AGE_RETRAIT < 60:
                                    TAUX_MRV_MRG_MRA_PROJ = 0.03
                                elif AGE_RETRAIT < 65:
                                    TAUX_MRV_MRG_MRA_PROJ = 0.035
                                elif AGE_RETRAIT < 70:
                                    TAUX_MRV_MRG_MRA_PROJ = 0.04
                                elif AGE_RETRAIT < 75:
                                    TAUX_MRV_MRG_MRA_PROJ = 0.0425
                                else:
                                    TAUX_MRV_MRG_MRA_PROJ = 0.05
                                MT_MRV_MRG_MRA_PROJ = TAUX_MRV_MRG_MRA_PROJ * MT_SRG_PROJ
                            else:
                                tmp_mrv = TAUX_MRV_MRG_MRA_PROJ * MT_SRG_PROJ
                                if tmp_mrv > MT_MRV_MRG_MRA_PROJ:
                                    MT_MRV_MRG_MRA_PROJ = tmp_mrv
                        else:
                            if AGE_RETRAIT == AGE_MRV_PERMIS:
                                MT_MRV_MRG_MRA_PROJ = TAUX_MRV_MRG_MRA_PROJ * MT_SRG_PROJ
                            else:
                                tmp_mrv = TAUX_MRV_MRG_MRA_PROJ * MT_SRG_PROJ
                                if tmp_mrv > MT_MRV_MRG_MRA_PROJ:
                                    MT_MRV_MRG_MRA_PROJ = tmp_mrv

                        if (M_MT_MRV_EXCEDENT > 1.0 and year_idx == 0):
                            base_excedent = MT_SRG_PROJ if MT_SRG_PROJ > MT_VM_PROJ else MT_VM_PROJ
                            tmp_mrv = TAUX_MRV_MRG_MRA_PROJ * base_excedent
                            if tmp_mrv < MT_MRV_MRG_MRA_PROJ:
                                MT_MRV_MRG_MRA_PROJ = tmp_mrv

                    RETRAIT = 0.0
                    if not (
                            AGE_RETRAIT < AGE_DECAISSEMENT or
                            (AGE_RETRAIT == AGE_DECAISSEMENT and mois_eval >= MOIS_NAIS) or
                            (MT_VM_PROJ <= 0 and I_PRODUIT_REGR == 0)
                    ):
                        if VAR_RETRAIT_FCT == 1:
                            RETRAIT = MT_TPA_RETRAIT if MT_TPA_RETRAIT > 0.0 else MT_VM_PROJ * PC_RETRAIT_AGE
                        elif VAR_RETRAIT_FCT == 2:
                            if MT_TPA_RETRAIT > MT_MIN_FERR_PROJ:
                                RETRAIT = MT_TPA_RETRAIT
                            else:
                                pc_ra = PC_RETRAIT_AGE
                                if pc_ra < 1.0:
                                    pc_ra = 1.0
                                RETRAIT = MT_MIN_FERR_PROJ * pc_ra
                        elif VAR_RETRAIT_FCT == 3:
                            base_mrv = MT_MRV_MRG_MRA_PROJ if MT_MRV_MRG_MRA_PROJ > MT_MIN_FERR_PROJ else MT_MIN_FERR_PROJ
                            RETRAIT = base_mrv * PC_RETRAIT_AGE
                        else:
                            RETRAIT = 0.0

                        if RETRAIT > MT_RETRAIT_MAX and MT_RETRAIT_MAX > 0.0:
                            RETRAIT = MT_RETRAIT_MAX
                        RETRAIT = RETRAIT / FREQ_EVAL_INT

                    PREST_MRV = 0.0
                    if I_PRODUIT_REGR == 1:
                        diff_mrv = RETRAIT - MT_VM_AV_RETRAIT
                        PREST_MRV = -diff_mrv * TX_SURVIE_DEB if diff_mrv > 0.0 else 0.0
                    VP_PREST_MRV = PREST_MRV * tx_actu

                    if MT_VM_AV_RETRAIT <= RETRAIT:
                        MT_GAR_ECH_PROJ = 0.0
                        MT_GAR_DECES_PROJ = 0.0
                        MT_BONI_DECES_PROJ = 0.0
                        MT_SRG_PROJ = 0.0
                    else:
                        withdrawal_factor = 1.0 - RETRAIT / MT_VM_AV_RETRAIT
                        MT_GAR_ECH_PROJ = MT_GAR_ECH_PROJ * withdrawal_factor
                        MT_GAR_DECES_PROJ = MT_GAR_DECES_PROJ * withdrawal_factor
                        MT_BONI_DECES_PROJ = MT_BONI_DECES_PROJ * withdrawal_factor
                        MT_SRG_PROJ = MT_SRG_PROJ - RETRAIT
                        if MT_SRG_PROJ < 0.0:
                            MT_SRG_PROJ = 0.0

                    MT_VM_AP_RETRAIT = MT_VM_AV_RETRAIT - RETRAIT
                    if MT_VM_AP_RETRAIT < 0.0:
                        MT_VM_AP_RETRAIT = 0.0

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

                    if (pc_depot_annuel == 0.0 or
                            (i_even_cesse_depot == 1 and AGE_RETRAIT >= AGE_DECAISSEMENT) or
                            (age_max_depot < age) or
                            (MT_VM_AP_RETRAIT <= 0 and I_PRODUIT_REGR == 0)):
                        depot_futur = 0.0
                    else:
                        if MT_TPA_DEPOT > 0.0:
                            depot_futur = MT_TPA_DEPOT
                        else:
                            base_depot_calc = MT_VM_AP_RETRAIT if var_depot_fct == 1 else (MT_GAR_DECES_PROJ / max(pc_gar_deces_1, MIN_GUARANTEE_VALUE))
                            depot_futur = base_depot_calc * pc_depot_annuel
                        depot_futur = depot_futur / FREQ_EVAL_INT

                    if MT_VM_AP_RETRAIT > 0.0:
                        MT_VM_PROJ = MT_VM_AP_RETRAIT + depot_futur
                        MT_GAR_ECH_PROJ = MT_GAR_ECH_PROJ + depot_futur * PC_GAR_ECH_DEP_FUT
                        MT_GAR_DECES_PROJ = MT_GAR_DECES_PROJ + depot_futur
                        if MT_SRG_PROJ > 0.0:
                            MT_SRG_PROJ = MT_SRG_PROJ + depot_futur
                    else:
                        MT_VM_PROJ = MT_VM_AP_RETRAIT

                    claim = MT_GAR_DECES_PROJ + MT_BONI_DECES_PROJ - MT_VM_PROJ
                    PREST_DECES = -qx * claim * TX_SURVIE_DEB if claim > 0.0 else 0.0
                    VP_PREST_DECES = PREST_DECES * tx_actu

                    PREST_ECH = 0.0
                    target_month_ech = 12 if MOIS_NAIS == int(12 // FREQ_EVAL_INT) else (MOIS_NAIS - int(12 // FREQ_EVAL_INT))
                    if ((annee_reelle == ANNEE_ECH_PROJ and mois_eval == MOIS_ECH_PROJ) or
                            (age == AGE_FIN_CONTRAT and mois_eval == target_month_ech)):
                        diff_ech = MT_GAR_ECH_PROJ - MT_VM_AP_RETRAIT
                        PREST_ECH = -diff_ech * TX_SURVIE if diff_ech > 0.0 else 0.0
                        ANNEE_ECH_PROJ = ANNEE_ECH_PROJ + NB_AN_ECH
                        MOIS_ECH_PROJ = mois_eval
                        if diff_ech > 0.0:
                            MT_VM_PROJ = MT_VM_AP_RETRAIT + diff_ech
                        else:
                            MT_VM_PROJ = MT_VM_AP_RETRAIT
                        MT_GAR_ECH_PROJ = MT_VM_PROJ * PC_GAR_ECH
                        if I_RESET_DECES_ECH == 1:
                            MT_GAR_DECES_PROJ = MT_VM_PROJ * pc_gar_deces_1
                        pc_renouv = PC_RENOUV_ECH
                        if age > AGE_MAX_RENOUV_ECH:
                            pc_renouv = 0.0
                        TX_SURVIE = TX_SURVIE * pc_renouv
                    VP_PREST_ECH = PREST_ECH * tx_actu

                    MT_GAR_DECES_PROJ = MT_GAR_DECES_PROJ - AJUSTEMENT_MENSUEL_GAR * 12.0 / FREQ_EVAL_INT

                    if I_PRODUIT_REGR == 1:
                        if (age < MAX_RESET_SRG and
                                MT_SRG_PROJ < MT_VM_PROJ and
                                annee_reelle > ANNEE_COTIS and
                                FREQ_RESET_SRG > 0):
                            years_since_issue = annee_reelle - ANNEE_COTIS
                            if int(years_since_issue / FREQ_RESET_SRG) == years_since_issue / FREQ_RESET_SRG and mois_eval == MOIS_COTIS:
                                MT_SRG_PROJ = MT_VM_PROJ
                                if MT_VM_PROJ > MT_BCB_PROJ:
                                    MT_BCB_PROJ = MT_VM_PROJ

                        if age < AGE_DECAISSEMENT and mois_eval == 12:
                            MT_SRG_PROJ = MT_SRG_PROJ + PC_BONI_SRG * MT_BCB_PROJ

                    if (age < MAX_RESET_DECES and
                            (MT_GAR_DECES_PROJ + MT_BONI_DECES_PROJ) < (MT_VM_PROJ * pc_gar_deces_1) and
                            annee_reelle > ANNEE_COTIS):
                        should_reset = False
                        if FREQ_RESET_DECES > 0:
                            years_since_issue = annee_reelle - ANNEE_COTIS
                            if int(years_since_issue / FREQ_RESET_DECES) == years_since_issue / FREQ_RESET_DECES and mois_eval == MOIS_COTIS:
                                should_reset = True

                        target_month = 12 if MOIS_NAIS == int(12 // FREQ_EVAL_INT) else (MOIS_NAIS - int(12 // FREQ_EVAL_INT))
                        if age == MAX_RESET_DECES - 1 and mois_eval == target_month:
                            should_reset = True

                        if should_reset:
                            MT_GAR_DECES_PROJ = MT_VM_PROJ * pc_gar_deces_1
                            MT_BONI_DECES_PROJ = 0.0

                    if mois_eval == 6 or mois_eval == 12:
                        if (I_RESET_FACUL_ECH == 1 and
                                age <= MAX_RESET_FACUL_ECH and
                                MT_GAR_ECH_PROJ > 0.0 and
                                (MT_VM_PROJ * PC_GAR_ECH) >= RATIO_VM_VG_RESET_ECH * MT_GAR_ECH_PROJ):
                            tmp_gar = MT_VM_PROJ * PC_GAR_ECH
                            if tmp_gar > MT_GAR_ECH_PROJ:
                                MT_GAR_ECH_PROJ = tmp_gar

                            tmp_annee = annee_reelle + NB_AN_ECH
                            min_annee = ANNEE_NAIS + AGE_ECH_MIN
                            ANNEE_ECH_PROJ = tmp_annee if tmp_annee > min_annee else min_annee
                            if ANNEE_ECH_PROJ == min_annee:
                                MOIS_ECH_PROJ = MOIS_NAIS
                            else:
                                MOIS_ECH_PROJ = mois_eval

                    target_month = 12 if MOIS_NAIS == int(12 // FREQ_EVAL_INT) else (MOIS_NAIS - int(12 // FREQ_EVAL_INT))
                    if age == AGE_CHANG_DECES - 1 and mois_eval == target_month:
                        if pc_gar_deces_1 != 0.0:
                            MT_GAR_DECES_PROJ = MT_GAR_DECES_PROJ * PC_GAR_DECES_2 / pc_gar_deces_1
                        pc_gar_deces_1 = PC_GAR_DECES_2

                    if MT_VM_PROJ > 0.0:
                        curr_mt_sp500 = MT_VM_PROJ * w_sp500
                        curr_mt_tsx = MT_VM_PROJ * w_tsx
                        curr_mt_eafe = MT_VM_PROJ * w_eafe
                        curr_mt_dex = MT_VM_PROJ * w_dex
                        curr_mt_mm = MT_VM_PROJ * w_mm
                        curr_vm = MT_VM_PROJ
                    else:
                        curr_mt_sp500 = 0.0
                        curr_mt_tsx = 0.0
                        curr_mt_eafe = 0.0
                        curr_mt_dex = 0.0
                        curr_mt_mm = 0.0
                        curr_vm = 0.0

                    pc_commission_maintien = DEFAULT_COMMISSION_MAINTIEN
                    pc_frais_an = 0.0
                    pc_commission_vente = 0.0

                    if MT_VM_AV_RETRAIT_FRAIS != 0.0:
                        if (duree_max10 < acq_vente_rf.shape[0] and ID_ACQUI < acq_vente_rf.shape[1]):
                            pc_vente_rf_v = acq_vente_rf[duree_max10, ID_ACQUI]
                            pc_vente_ac_v = acq_vente_ac[duree_max10, ID_ACQUI]
                            pc_maintien_rf_v = acq_maintien_rf[duree_max10, ID_ACQUI]
                            pc_maintien_ac_v = acq_maintien_ac[duree_max10, ID_ACQUI]
                            pc_frais_ac_v = acq_frais_ac[duree_max10, ID_ACQUI]
                            pc_frais_rf_v = acq_frais_rf[duree_max10, ID_ACQUI]
                        else:
                            pc_vente_rf_v = 0.0
                            pc_vente_ac_v = 0.0
                            pc_maintien_rf_v = 0.0
                            pc_maintien_ac_v = 0.0
                            pc_frais_ac_v = 0.0
                            pc_frais_rf_v = 0.0

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

                    COMM_VENTE = -pc_commission_vente * depot_futur * TX_SURVIE
                    VP_COMM_VENTE = COMM_VENTE * tx_actu

                    FRAIS_ACQUIS = pc_frais_an * MT_VM_AP_RETRAIT * lapse * TX_SURVIE_DEB * (1.0 - qx)
                    VP_FRAIS_ACQUIS = FRAIS_ACQUIS * tx_actu

                    fixed_fee_annual = 0.0
                    if ID_PRODUIT < fees_lookup.shape[0] and annee_reelle < fees_lookup.shape[1]:
                        fixed_fee_annual = fees_lookup[ID_PRODUIT, annee_reelle]
                    if MT_VM_AV_RETRAIT <= 0.0 and VP_PREST_MRV == 0.0:
                        FRAIS_FIXES = 0.0
                    else:
                        FRAIS_FIXES = -fixed_fee_annual / FREQ_EVAL_INT * TX_SURVIE_DEB
                    VP_FRAIS_FIXES = FRAIS_FIXES * tx_actu

                    HON_GEST = -MT_VM_AV_RETRAIT_FRAIS * (math.exp(PC_HONORAIRES_GEST / FREQ_EVAL_INT) - 1.0) * TX_SURVIE_DEB
                    VP_HON_GEST = HON_GEST * tx_actu

                    COMM_MAINTIEN = -MT_VM_AV_RETRAIT_FRAIS * (math.exp(pc_commission_maintien / FREQ_EVAL_INT) - 1.0) * TX_SURVIE_DEB
                    VP_COMM_MAINTIEN = COMM_MAINTIEN * tx_actu

                    PRIMES_VARIABLES = MT_VM_AV_RETRAIT_FRAIS * math.exp(-(PC_RFG - PC_REVENU_FDS) / FREQ_EVAL_INT) * (-(math.exp(-PC_REVENU_FDS / FREQ_EVAL_INT) - 1.0)) * TX_SURVIE_DEB
                    VP_PRIMES_VARIABLES = PRIMES_VARIABLES * tx_actu

                    fees = -(HON_GEST + COMM_MAINTIEN + FRAIS_FIXES)

                    vp_flux_tot = (VP_COMM_MAINTIEN + VP_HON_GEST + VP_FRAIS_FIXES + VP_FRAIS_ACQUIS + VP_COMM_VENTE +
                                   VP_PRIMES_VARIABLES + VP_PRIMES_GARANTIES + VP_PREST_ECH + VP_PREST_MRV + VP_PREST_DECES)

                    pv_path = pv_path + vp_flux_tot

                    if do_capital == 0 and is_debug_node and debug_int_scenario >= 0 and i_int == debug_int_scenario:
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

                        if debug_ts_output is not None and (debug_int_year < 0 or t_int == debug_int_year):
                            debug_ts_output[choc_idx, t_int, INT_TS_DEBUG_IDX_CURR_VM] = curr_vm
                            debug_ts_output[choc_idx, t_int, INT_TS_DEBUG_IDX_FEES] = fees
                            debug_ts_output[choc_idx, t_int, INT_TS_DEBUG_IDX_PV_PATH] = pv_path
                            debug_ts_output[choc_idx, t_int, INT_TS_DEBUG_IDX_R_PORTFOLIO] = r_portfolio
                            debug_ts_output[choc_idx, t_int, INT_TS_DEBUG_IDX_FWD_RATE] = fwd
                            debug_ts_output[choc_idx, t_int, INT_TS_DEBUG_IDX_DF] = tx_actu

                if do_capital == 0:
                    pv_reserve = pv_path
                else:
                    pv_capital = pv_path

            sum_pv_reserve_choc = sum_pv_reserve_choc + pv_reserve
            sum_pv_capital_choc = sum_pv_capital_choc + pv_capital

        avg_reserve = sum_pv_reserve_choc / n_internal_scenarios if n_internal_scenarios > 0 else 0.0
        avg_capital = sum_pv_capital_choc / n_internal_scenarios if n_internal_scenarios > 0 else 0.0

        output_metrics[acc_idx, scn_idx, year_idx, choc_idx, METRICS_RESERVE_IDX] = avg_reserve
        output_metrics[acc_idx, scn_idx, year_idx, choc_idx, METRICS_CAPITAL_IDX] = avg_capital

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
                debug_output[choc_idx, INT_DEBUG_IDX_AVG_PV_FLUX] = avg_reserve
                debug_output[choc_idx, INT_DEBUG_IDX_RESERVE] = avg_reserve
                debug_output[choc_idx, INT_DEBUG_IDX_CAPITAL] = avg_capital
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

"""
Constants for GPU actuarial calculations.

This module centralizes all magic numbers and configuration values
used throughout the GPU projection code.
"""

# =============================================================================
# ARRAY DIMENSION LIMITS
# =============================================================================
MAX_SEXE = 2                    # Number of sex categories (0=Male, 1=Female)
MAX_AGE = 121                   # Maximum age in mortality tables
MAX_LAPSE_LEVELS = 4            # Number of lapse rate levels (0-3)
MAX_DURATION = 11               # Maximum policy duration for lookups (0-10)
DEFAULT_AGE_MAX_DEPOSIT = 999   # Default max age for deposits when not specified

# =============================================================================
# DEFAULT RATES AND VALUES
# =============================================================================
DEFAULT_MORTALITY_RATE = 0.001          # Default qx when lookup fails
DEFAULT_RETURN_RATE = 0.02              # Default return rate (2%)
DEFAULT_FORWARD_RATE = 0.02             # Default forward rate (2%)
DEFAULT_LAPSE_RATE_TOT = 0.01           # Default total lapse rate (1%)
DEFAULT_LAPSE_RATE_PART = 0.005         # Default partial lapse rate (0.5%)
DEFAULT_LAPSE_FACT_DIM = 1.0            # Default lapse diminishing factor
DEFAULT_FERR_MIN_RATE = 0.05            # Default FERR minimum rate for older ages (5%)
DEFAULT_COMMISSION_MAINTIEN = 0.001     # Default maintenance commission rate (0.1%)

# =============================================================================
# RISK-NEUTRAL SCENARIO DEFAULT RATES
# =============================================================================
RN_DEFAULT_FORWARD_RATE = 0.03          # Risk-neutral forward rate (3%)
RN_DEFAULT_REND_DEX = 0.025             # Risk-neutral DEX return (2.5%)
RN_DEFAULT_REND_MM = 0.02               # Risk-neutral money market return (2%)
RN_DEFAULT_REND_TSX = 0.035             # Risk-neutral TSX return (3.5%)
RN_DEFAULT_REND_SP500 = 0.035           # Risk-neutral SP500 return (3.5%)
RN_DEFAULT_REND_EAFE = 0.03             # Risk-neutral EAFE return (3%)

# =============================================================================
# SHOCK FACTORS
# =============================================================================
SHOCK_FACTOR = 0.9                      # 10% shock = multiply by 0.9
SHOCK_PERCENTAGE = 0.10                 # 10% shock percentage
NUM_CHOCS = 5                           # Number of shock scenarios (base + 4 shocks)

# Choc indices
CHOC_BASE = 0
CHOC_SP500 = 1
CHOC_TSX = 2
CHOC_EAFE = 3
CHOC_DEX = 4

# Choc names for reporting
CHOC_NAMES = ['BASE', 'SP500_SHOCK', 'TSX_SHOCK', 'EAFE_SHOCK', 'DEX_SHOCK']

# =============================================================================
# VM/VG RATIO THRESHOLDS FOR LAPSE CALCULATION
# =============================================================================
VM_VG_RATIO_MAX = 10.0                  # Maximum VM/VG ratio cap
VM_VG_RATIO_LEVEL1_THRESHOLD = 0.5      # Threshold for lapse level 1
VM_VG_RATIO_LEVEL2_THRESHOLD = 0.75     # Threshold for lapse level 2
VM_VG_RATIO_LEVEL3_DIVISOR = 999.24     # Divisor for level 3 interpolation (0.75 to 999.99)

# Lapse level indices
LAPSE_LEVEL_1 = 1
LAPSE_LEVEL_2 = 2
LAPSE_LEVEL_3 = 3

# =============================================================================
# MORTALITY CALCULATION
# =============================================================================
MORTALITY_AGE_ADJUSTMENT_THRESHOLD = 6  # Month threshold for age adjustment in mortality

# =============================================================================
# MEMORY AND BATCH CALCULATION
# =============================================================================
LOOKUP_TABLE_OVERHEAD_MB = 150          # Estimated lookup table memory overhead in MB
DEFAULT_GPU_MEMORY_GB = 12              # Default GPU memory assumption when query fails
MEMORY_SAFETY_FACTOR = 0.6              # Use only 60% of available memory
MEMORY_BATCH_THRESHOLD = 0.9            # Warn if batch uses >90% of free memory

# =============================================================================
# DEBUG OUTPUT DIMENSIONS
# =============================================================================
EXT_DEBUG_COLUMNS = 20                  # Number of columns in external debug output
INT_DEBUG_COLUMNS = 15                  # Number of columns in internal debug output

# =============================================================================
# CALCULATION LIMITS
# =============================================================================
MAX_WITHDRAWAL = 999999.0               # Maximum withdrawal amount cap
MIN_GUARANTEE_VALUE = 0.01              # Minimum guarantee value for ratio calculations

# =============================================================================
# FREQUENCY
# =============================================================================
DEFAULT_FREQ_EVAL = 12.0                # Default evaluation frequency (monthly)

# =============================================================================
# KERNEL CONFIGURATION
# =============================================================================
DEFAULT_THREADS_PER_BLOCK_2D = (16, 16)  # Default 2D thread block size
DEFAULT_THREADS_PER_BLOCK_1D = 256       # Default 1D thread block size

# =============================================================================
# METRICS OUTPUT DIMENSIONS
# =============================================================================
METRICS_RESERVE_IDX = 0                 # Index for reserve in metrics output
METRICS_CAPITAL_IDX = 1                 # Index for capital in metrics output
METRICS_OUTPUT_SIZE = 2                 # Number of metrics per choc (reserve, capital)

# =============================================================================
# ACCOUNT DATA ARRAY INDICES (for GPU kernel access)
# These are module-level constants so Numba can compile them into CUDA kernels
# =============================================================================
# Account identifiers
ACCOUNT_IDX_ID_COMPTE = 0
ACCOUNT_IDX_ANNEE_EVALUATION_INI = 1
ACCOUNT_IDX_MOIS_EVALUATION_INI = 2
ACCOUNT_IDX_ANNEE_NAIS = 3
ACCOUNT_IDX_MOIS_NAIS = 4

# Product information
ACCOUNT_IDX_I_SEXE = 5
ACCOUNT_IDX_I_PRODUIT_REGR = 6
ACCOUNT_IDX_ID_PRODUIT = 7
ACCOUNT_IDX_ID_LAPSE = 8
ACCOUNT_IDX_I_REGIME_2 = 9
ACCOUNT_IDX_ID_DEPOT = 10
ACCOUNT_IDX_ID_ACQUI = 11

# Age thresholds
ACCOUNT_IDX_AGE_ECH_MIN = 12
ACCOUNT_IDX_AGE_FIN_CONTRAT = 13
ACCOUNT_IDX_AGE_DECAISSEMENT = 14

# Financial amounts
ACCOUNT_IDX_MT_VM = 15
ACCOUNT_IDX_MT_GAR_DECES = 16
ACCOUNT_IDX_MT_GAR_ECH = 17
ACCOUNT_IDX_MT_SRG = 18
ACCOUNT_IDX_MT_BCB = 19

# Asset allocations
ACCOUNT_IDX_MT_DEX = 20
ACCOUNT_IDX_MT_MM = 21
ACCOUNT_IDX_MT_TSX = 22
ACCOUNT_IDX_MT_SP500 = 23
ACCOUNT_IDX_MT_EAFE = 24

# Additional amounts
ACCOUNT_IDX_MT_BONI_DECES = 25
ACCOUNT_IDX_MT_MRV_MRG_MRA = 26
ACCOUNT_IDX_TAUX_MRV_MRG_MRA = 27

# Dates
ACCOUNT_IDX_ANNEE_ECH = 28
ACCOUNT_IDX_MOIS_ECH = 29

# Percentage rates
ACCOUNT_IDX_PC_HONORAIRES_GEST = 30
ACCOUNT_IDX_PC_FRAIS_GARANTIE = 31
ACCOUNT_IDX_PC_GAR_DECES_1 = 32
ACCOUNT_IDX_PC_BONI_DECES = 33
ACCOUNT_IDX_PC_RFG = 34
ACCOUNT_IDX_PC_REVENU_FDS = 35
ACCOUNT_IDX_PC_GAR_ECH = 36
ACCOUNT_IDX_PC_GAR_ECH_DEP_FUT = 37

# Additional fields
ACCOUNT_IDX_AJUSTEMENT_COMMISSION = 38
ACCOUNT_IDX_MT_RF = 39
ACCOUNT_IDX_MT_VM_ORIG = 40
ACCOUNT_IDX_ANNEE_COTIS = 41
ACCOUNT_IDX_MOIS_COTIS = 42
ACCOUNT_IDX_MAX_BONI_DECES = 43
ACCOUNT_IDX_I_FRAIS_SUR_SRG = 44

# Total number of fields
ACCOUNT_IDX_TOTAL_FIELDS = 45

# =============================================================================
# STATE TENSOR INDICES (for GPU kernel access)
# These are module-level constants so Numba can compile them into CUDA kernels
# =============================================================================
STATE_IDX_MT_VM = 0
STATE_IDX_MT_GAR_DECES = 1
STATE_IDX_MT_GAR_ECH = 2
STATE_IDX_MT_SRG = 3
STATE_IDX_AGE = 4
STATE_IDX_TX_SURVIE = 5
STATE_IDX_MT_DEX = 6
STATE_IDX_MT_MM = 7
STATE_IDX_MT_TSX = 8
STATE_IDX_MT_SP500 = 9
STATE_IDX_MT_EAFE = 10
STATE_IDX_MT_BONI_DECES = 11
STATE_IDX_SIZE = 12  # Total number of state variables

# =============================================================================
# EXTERNAL DEBUG OUTPUT INDICES (for GPU kernel access)
# =============================================================================
EXT_DEBUG_IDX_VM = 0
EXT_DEBUG_IDX_AGE = 1
EXT_DEBUG_IDX_QX = 2
EXT_DEBUG_IDX_LAPSE_TOT = 3
EXT_DEBUG_IDX_LAPSE_PART = 4
EXT_DEBUG_IDX_TX_SURVIE = 5
EXT_DEBUG_IDX_FORWARD_RATE = 6
EXT_DEBUG_IDX_REND_SP500 = 7
EXT_DEBUG_IDX_REND_TSX = 8
EXT_DEBUG_IDX_REND_EAFE = 9
EXT_DEBUG_IDX_REND_DEX = 10
EXT_DEBUG_IDX_RETRAIT = 11
EXT_DEBUG_IDX_PREST_DECES = 12
EXT_DEBUG_IDX_PRIMES_GARANTIES = 13
EXT_DEBUG_IDX_VM_VG_RATIO = 14
EXT_DEBUG_IDX_SIZE = 15  # Total number of debug columns

# =============================================================================
# INTERNAL DEBUG OUTPUT INDICES (for GPU kernel access)
# =============================================================================
INT_DEBUG_IDX_START_VM = 0
INT_DEBUG_IDX_VM_CHOC = 1
INT_DEBUG_IDX_AVG_PV_FLUX = 2
INT_DEBUG_IDX_RESERVE = 3
INT_DEBUG_IDX_CAPITAL = 4
INT_DEBUG_IDX_START_TX_SURVIE = 5
INT_DEBUG_IDX_START_AGE = 6
INT_DEBUG_IDX_CURR_VM = 7        # VM at debug internal iteration
INT_DEBUG_IDX_FEES = 8           # Fees at debug internal iteration
INT_DEBUG_IDX_PV_PATH = 9        # Cumulative PV at debug internal iteration
INT_DEBUG_IDX_R_PORTFOLIO = 10   # Return applied at debug internal iteration
INT_DEBUG_IDX_FWD_RATE = 11      # Forward rate at debug internal iteration
INT_DEBUG_IDX_SIZE = 12  # Total number of debug columns

INT_TS_DEBUG_IDX_CURR_VM = 0
INT_TS_DEBUG_IDX_FEES = 1
INT_TS_DEBUG_IDX_PV_PATH = 2
INT_TS_DEBUG_IDX_R_PORTFOLIO = 3
INT_TS_DEBUG_IDX_FWD_RATE = 4
INT_TS_DEBUG_IDX_DF = 5
INT_TS_DEBUG_IDX_SIZE = 6

# =============================================================================
# FLUX COMPONENT INDICES (for GPU kernel access)
# =============================================================================
# Cashflow components
FLUX_COMP_IDX_PRIMES_GARANTIES = 0
FLUX_COMP_IDX_PREST_DECES = 1
FLUX_COMP_IDX_PREST_ECH = 2
FLUX_COMP_IDX_PREST_MRV = 3
FLUX_COMP_IDX_FRAIS_ACQUIS = 4
FLUX_COMP_IDX_COMM_VENTE = 5
FLUX_COMP_IDX_PRIMES_VARIABLES = 6
FLUX_COMP_IDX_FRAIS_FIXES = 7
FLUX_COMP_IDX_HON_GEST = 8
FLUX_COMP_IDX_COMM_MAINTIEN = 9
FLUX_COMP_IDX_VALEUR_MARCHANDE = 10
FLUX_COMP_IDX_PASSIF_REDRESSE = 11
FLUX_COMP_IDX_COUSSIN_CREDIT = 12
FLUX_COMP_IDX_COUSSIN_MARCHE = 13
FLUX_COMP_IDX_COUSSIN_DEPENSE = 14
FLUX_COMP_IDX_COUSSIN_DECHEANCE = 15
FLUX_COMP_IDX_COUSSIN_MORTALITE = 16
FLUX_COMP_IDX_COUSSIN_DEPOT = 17
# Detailed calculation fields
FLUX_COMP_IDX_MT_VM = 18
FLUX_COMP_IDX_MT_VM_AV_RETRAIT = 19
FLUX_COMP_IDX_MT_VM_AP_RETRAIT = 20
FLUX_COMP_IDX_AGE = 21
FLUX_COMP_IDX_QX = 22
FLUX_COMP_IDX_LAPSE_TOT = 23
FLUX_COMP_IDX_LAPSE_PART = 24
FLUX_COMP_IDX_TX_SURVIE = 25
FLUX_COMP_IDX_RETRAIT = 26
FLUX_COMP_IDX_DEPOT_FUTUR = 27
FLUX_COMP_IDX_MT_GAR_DECES = 28
FLUX_COMP_IDX_MT_GAR_ECH = 29
FLUX_COMP_IDX_MT_SRG = 30
FLUX_COMP_IDX_REND_SP500 = 31
FLUX_COMP_IDX_REND_TSX = 32
FLUX_COMP_IDX_REND_EAFE = 33
FLUX_COMP_IDX_REND_DEX = 34
FLUX_COMP_IDX_REND_MM = 35
FLUX_COMP_IDX_MT_SP500 = 36
FLUX_COMP_IDX_MT_TSX = 37
FLUX_COMP_IDX_MT_EAFE = 38
FLUX_COMP_IDX_MT_DEX = 39
FLUX_COMP_IDX_MT_MM = 40
FLUX_COMP_IDX_SIZE = 41

# Debug flux output dimensions: (n_years+1, freq_eval, FLUX_COMP_IDX_SIZE)
# This captures flux for a single account/scenario for debugging

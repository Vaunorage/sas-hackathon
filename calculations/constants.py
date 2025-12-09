"""
Constants for GPU actuarial calculations.

This module centralizes all magic numbers and configuration values
used throughout the GPU projection code.
"""
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from numba.cuda.cudadrv.devicearray import DeviceNDArray
    import numpy as np

# =============================================================================
# TYPE ALIASES FOR KERNEL DOCUMENTATION
# =============================================================================
# These are for documentation/IDE hints only - not enforced at runtime

# GPU Device Arrays
GPUArray = "DeviceNDArray"

# Account data: (n_accounts, n_fields) float32
AccountData = "DeviceNDArray"

# Returns lookups tuple: (forward_rate, ajust_forward, rend_dex, rend_mm, rend_tsx, rend_sp500, rend_eafe)
# Each array shape: (n_scenarios, n_years, n_months)
ReturnsLookups = Tuple[GPUArray, GPUArray, GPUArray, GPUArray, GPUArray, GPUArray, GPUArray]

# Lapse lookups tuple: (min_ferr, lapse_part_min, lapse_part_max, lapse_tot_min, lapse_tot_max, lapse_tot_fact)
LapseLookups = Tuple[GPUArray, GPUArray, GPUArray, GPUArray, GPUArray, GPUArray]

# Policy lookups tuple: (deposits_pc, deposits_var, deposits_age_max, deposits_i_even, fees)
PolicyLookups = Tuple[GPUArray, GPUArray, GPUArray, GPUArray, GPUArray]

# Commission lookups tuple: (acq_vente_rf, acq_vente_ac, acq_maintien_rf, acq_maintien_ac, acq_frais_ac, acq_frais_rf)
CommissionLookups = Tuple[GPUArray, GPUArray, GPUArray, GPUArray, GPUArray, GPUArray]

# Risk-neutral returns tuple: (rn_forward_rate, rn_rend_dex, rn_rend_mm, rn_rend_tsx, rn_rend_sp500, rn_rend_eafe)
RNReturnsLookups = Tuple[GPUArray, GPUArray, GPUArray, GPUArray, GPUArray, GPUArray]

# Mortality lookup: (sex, age, year, product) float32
MortalityLookup = GPUArray

# Output tensors
StatesTensor = GPUArray      # (batch, scenarios, years, STATE_SIZE)
CashflowsTensor = GPUArray   # (batch, scenarios, years, 1)
MetricsTensor = GPUArray     # (batch, scenarios, years, NUM_CHOCS, METRICS_OUTPUT_SIZE)

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

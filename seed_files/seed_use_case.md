# INPUTS

## COMPUTE CAPABITILIES
HARDWARE_ARCHITECTURE = 64; /*32-bit or 64-bit */
HARDWARE_CPU_CORES = 24;
HARDWARE_MEMORY_PER_CPU_GB = 8;
STORAGE_TYPE = TO BE TESTED;        /* NETWORK, SSD, HDD */
SAS_SOFTWARE_ARCHTECTURE = 64; /*32-bit or 64-bit */
SAS_LICENSE_TYPE = TO BE TESTED; /* BASE, ENTERPRISE, GRID */
MAX_USER_CPU_LIMIT = 12;      /* Multi-tenant policy limit */

## ALGO SETTINGS
FORCE_SEQUENTIAL = NO;         /* Override parallel detection */
SURVIVAL_THRESHOLD = AUTO;     /* AUTO, or 0.001-0.01 */
CHUNK_STRATEGY = AUTO;         /* AUTO, EQUAL, BALANCED, MEMORY_OPTIMAL */
BUSINESS_HOURS_CONSERVATIVE = YES; /* Reduce resources during business hours */

## DATASETS
POPULATION = "data_in/population.sas7bdat"
RENDEMENT = "data_in/rendement.sas7bdat"
TX_DECES = "data_in/tx_deces.sas7bdat"
TX_INTERET = "data_in/tx_interet.sas7bdat"
TX_INTERET_INT = "data_in/tx_interet_int.sas7bdat"
TX_RETRAIT = "data_in/tx_retrait.sas7bdat"

## PARAMETERS

### Mandatory Business/Regulatory Parameters
CHOC_CAPITAL = 0.35;              /* Capital shock factor (regulatory requirement: 0.20-0.50) */
HURDLE_RT = 0.10;                 /* Hurdle rate for profitability analysis (business target: 0.08-0.15) */
SURVIVAL_THRESHOLD = 0.001;       /* Minimum survival probability threshold for computational efficiency (0.001-0.01) */

### Auto-Inferred Parameters (from input files)
/* Rule: NBCPT = COUNT(*) FROM POPULATION */
NBCPT = AUTO;                     /* Total number of accounts in POPULATION file */

/* Rule: NB_SC = MAX(scn_proj) FROM RENDEMENT WHERE TYPE='EXTERNE' */
NB_SC = AUTO;                     /* Number of external economic scenarios */

/* Rule: NB_AN_PROJECTION = MAX(an_proj) FROM RENDEMENT WHERE TYPE='EXTERNE' */
NB_AN_PROJECTION = AUTO;          /* External projection horizon in years */

/* Rule: NB_SC_INT = MAX(scn_proj) FROM RENDEMENT WHERE TYPE='INTERNE' */
NB_SC_INT = AUTO;                 /* Number of internal scenarios for reserves/capital */

/* Rule: NB_AN_PROJECTION_INT = MAX(an_proj) FROM RENDEMENT WHERE TYPE='INTERNE' */
NB_AN_PROJECTION_INT = AUTO;      /* Internal projection horizon in years */

### Optional Simulation Control Parameters
NBCPT_TO_USE = AUTO;              /* Number of accounts to include in simulation (1 to NBCPT) */
                                  /* AUTO = use all accounts, or specify subset for testing */

SCENARIO_SUBSET_START = 1;        /* First external scenario to include (1 to NB_SC) */
SCENARIO_SUBSET_END = AUTO;       /* Last external scenario to include (AUTO = NB_SC) */

ACCOUNT_SUBSET_START = 1;         /* First account ID to include in simulation */
ACCOUNT_SUBSET_END = AUTO;        /* Last account ID to include (AUTO = NBCPT) */

PROJECTION_YEARS_LIMIT = AUTO;    /* Limit projection horizon (AUTO = NB_AN_PROJECTION) */
                                  /* Use for shorter test runs or specific analysis periods */

INTERNAL_SCENARIOS_LIMIT = AUTO;  /* Limit internal scenarios for faster processing */
                                  /* (AUTO = NB_SC_INT, or reduce for testing) */

AGE_FILTER_MIN = 0;               /* Minimum policyholder age to include (risk segmentation) */
AGE_FILTER_MAX = 99;              /* Maximum policyholder age to include */

FUND_VALUE_FILTER_MIN = 0;        /* Minimum initial fund value to include (portfolio segmentation) */
FUND_VALUE_FILTER_MAX = AUTO;     /* Maximum initial fund value (AUTO = no limit) */

# OUTPUTS

##DATASETS
CALCULS_SOMMAIRE = "calculs_sommaire.sas7bdat"


# SAS Code Analysis - Actuarial Cash Flow Calculation

## Overall Code Flow Pipeline

```
Initialization (Global Variables & Libraries) 
→ Data Loading (Load input tables to memory)
→ Empty Results Tables Creation
→ Main Calculation Loop
  ├─ External Loop Processing (Population expansion & cash flow calculations)
  ├─ Internal Loop Processing (Reserve & capital calculations with nested scenarios)
  ├─ Data Aggregation & Summary
  └─ Final Calculations (Profit & distributable cash flows)
→ Results Storage
→ Memory Cleanup
```

## Pipeline Step Details

### 1. Initialization (Lines 1-30)

**Global Variables Declaration:**
- `NBCPT = 100` - Number of accounts/policies
- `NB_SC = 100` - Number of external scenarios 
- `NB_AN_PROJECTION = 100` - Number of projection years
- `NB_SC_INT = 100` - Number of internal scenarios
- `NB_AN_PROJECTION_INT = 100` - Number of internal projection years
- `CHOC_CAPITAL = 0.35` - Capital shock rate (35%)
- `HURDLE_RT = 0.10` - Hurdle rate (10%)

**Library Setup:**
- `INPUT` - Input data directory
- `SERVEUR` - Output directory  
- `MEMOIRE` - In-memory library for temporary calculations

### 2. Macro Definition - DATA_STEP_CALCUL (Lines 32-220)

**Hash Tables Creation (Lines 45-83):**
- Mortality rates (`TX_DECES`)
- Lapse rates (`TX_RETRAIT`) 
- Investment returns (`RENDEMENT`)
- Discount rates external (`TX_INTERET`)
- Discount rates internal (`TX_INTERET_INT`)

**Variable Initialization (Lines 88-148):**
- External type (year 0): Initialize account values, survival rates, commissions, and expenses
- Internal type (year 0): Set account values with optional capital shock, reset other variables

**Cash Flow Calculations (Lines 156-217):**
- Project fund values with investment returns and fees
- Calculate survival probabilities using mortality and lapse rates
- Compute cash flows: revenues, management fees, commissions, administrative expenses, death benefit payments
- Calculate present values using appropriate discount rates
- Adjust present values for internal loop timing

### 3. Data Loading (Lines 227-250)

**Required Input Files:**
- **POPULATION.sas7bdat** - Policy population data
- **TX_DECES.sas7bdat** - Mortality rates
- **TX_RETRAIT.sas7bdat** - Lapse rates  
- **TX_INTERET.sas7bdat** - External discount rates
- **TX_INTERET_INT.sas7bdat** - Internal discount rates
- **RENDEMENT.sas7bdat** - Investment return scenarios

**File Location:** `INPUT` library (`"&PATH.\Intrants Guy"`)

**Data Structure Requirements:**
- `TX_DECES`: `AGE`, `Qx` (mortality rate by age)
- `TX_RETRAIT`: `an_proj` (projection year), `WX` (lapse rate)
- `TX_INTERET`: `an_proj` (projection year), `TX_ACTU` (discount rate)
- `TX_INTERET_INT`: `an_eval` (evaluation year), `TX_ACTU_INT` (internal discount rate)
- `RENDEMENT`: `scn_proj` (scenario), `an_proj` (projection year), `TYPE`, `RENDEMENT` (return rate)
- `POPULATION`: `ID_COMPTE`, `age_deb`, `MT_VM`, `MT_GAR_DECES`, and other policy characteristics

All files are loaded into the `MEMOIRE` library for efficient hash table lookups during calculations.

### 4. Results Tables Initialization (Lines 252-259)

Create empty output tables:
- `CALCULS_SOMMAIRE` - Summary calculations results
- `SOMMAIRE_RESERVE` - Reserve summary (temporary)

### 5. Main Calculation Loop (Lines 264-417)

**Outer Loop - By Account (j = 1 to NBCPT):**

**External Calculations (Lines 268-296):**
- Expand population by scenarios and projection years
- Apply cash flow calculation macro
- Store external cash flows and population data for internal calculations
- Clean up temporary datasets

**Internal Calculations - Nested Loops (Lines 299-380):**
- Loop by calculation type (m = 1,2): Reserve vs Capital
- Loop by external scenario (k = 1 to NB_SC)
- Expand data by internal scenarios and projection years
- Apply capital shock for capital calculations (35% reduction)
- Run cash flow calculations
- Aggregate results by account/year/external scenario
- Calculate mean across internal scenarios

**Data Integration (Lines 352-379):**
- Merge external cash flows with internal reserve/capital calculations
- Calculate reserve values (type 1) and capital values (type 2, net of reserve)

**Final Calculations (Lines 382-408):**
- Calculate profit = net cash flow + change in reserve
- Calculate distributable cash flows = profit + change in capital  
- Apply present value discount using hurdle rate
- Aggregate by account and external scenario
- Append results to final output table

### 6. Memory Cleanup (Lines 410-424)

- Delete temporary calculation datasets
- Clear memory library
- Restore log output options

## Key Technical Features

- **Hash tables** for efficient lookup of rates and assumptions
- **Nested loop structure** for complex scenario modeling
- **Memory management** with dataset segmentation for performance
- **Present value calculations** with multiple discount rate structures
- **Aggregation** at multiple levels (internal scenarios, external scenarios, accounts)

## Performance Optimization Analysis

### Optimization Techniques Implemented

#### 1. **Hash Tables for Efficient Lookups** (Lines 45-82)
```sas
declare hash h(dataset: "memoire.TX_DECES");
declare hash g(dataset: "memoire.TX_RETRAIT");
```
- Uses hash tables instead of repeated MERGE/JOIN operations
- Provides O(1) lookup time for mortality rates, lapse rates, returns, and discount rates
- Significantly faster than traditional SAS merging for large datasets

#### 2. **Memory-Based Processing** (Line 27)
```sas
libname MEMOIRE "&PATH.\Memlib" MEMLIB;
```
- Utilizes `MEMLIB` option to store intermediate calculations entirely in RAM
- Eliminates disk I/O bottlenecks for temporary datasets
- Faster data access and manipulation

#### 3. **Strategic Dataset Segmentation** (Lines 265-266, 298-301)
```sas
%do j = 1 %to &NBCPT.;        /* By account */
%do m = 1 %to 2;              /* By calculation type */
%do k = 1 %to &NB_SC.;        /* By external scenario */
```
- Processes one account at a time instead of handling all simultaneously
- Code comment states: *"on separe les calculs en plusieurs tables pour que ca soit plus performant"* (we separate calculations into multiple tables for better performance)
- Manages memory constraints by avoiding monolithic datasets

#### 4. **Systematic Memory Cleanup** (Lines 294-296, 348-350, 410-412)
```sas
proc datasets library=MEMOIRE;
    delete calculs_2 calculs_3;
run;
```
- Deletes temporary datasets immediately after each processing step
- Prevents memory accumulation during long-running calculations
- Maintains optimal memory usage throughout execution

#### 5. **Precompiled Data Step** (Line 38)
```sas
*Datastep precompile;
```
- Uses DATA step view compilation for the main calculation engine
- Reduces parsing overhead for repeated macro executions
- Improves performance in iterative processing

#### 6. **Selective Variable Retention** (Lines 288-289)
```sas
data MEMOIRE.FLUX_EXTERNE(keep = ID_COMPTE an_eval scn_eval REVENUS...)
```
- Uses `KEEP=` and `DROP=` options to minimize dataset sizes
- Only retains necessary variables for downstream processing
- Reduces memory footprint and I/O overhead

#### 7. **Log Output Suppression** (Line 262)
```sas
*option nonotes nosource;
```
- Disables verbose logging during intensive calculations (commented out)
- Would reduce I/O overhead from excessive log writing
- Re-enabled after processing completion (Line 420)

#### 8. **Memory Constraint Acknowledgment** (Line 298)
```sas
*manque de memoire alors on doit segmenter le calcul par type et simulation;
```
Translation: *"lack of memory so we must segment the calculation by type and simulation"*
- Explicit recognition of memory limitations
- Implements nested segmentation strategy as a direct response to constraints

### Performance Bottlenecks Still Present

#### Computational Complexity
- **Nested loops** with potentially 100×100×100×100 iterations (100+ million calculations)
- **Sequential processing** of accounts rather than batch operations
- **No parallelization** despite `OPTIONS PRESENV SPOOL` setup for threading

#### Scalability Limitations
- Current design limited by available RAM for in-memory processing
- Account-by-account processing prevents economies of scale
- Complex nested scenario structure creates exponential computation growth

### SAS Compute Server Limitations (Inferred from Code)

#### Memory Constraints
**Line 298:** `*manque de memoire alors on doit segmenter le calcul par type et simulation;`
- Explicit acknowledgment of insufficient RAM
- Forces account-by-account processing instead of batch processing
- Requires nested segmentation strategy to fit within memory limits

**Lines 265-266:** Account-level segmentation (`%do j = 1 %to &NBCPT.`)
- Processing 100 accounts sequentially suggests memory cannot handle full dataset
- Each account processed independently to manage memory usage

#### Processing Power Limitations  
**Line 27:** `libname MEMOIRE "&PATH.\Memlib" MEMLIB;`
- Heavy reliance on in-memory processing suggests limited I/O bandwidth
- Server likely has limited disk I/O capacity relative to computation needs

**Lines 299-301:** Triple nested loops with segmentation
```sas
%do m = 1 %to 2;          /* Reserve vs Capital */
%do k = 1 %to &NB_SC.;    /* External scenarios */  
%do scn_eval_int = 1 to &NB_SC_INT.; /* Internal scenarios */
```
- Further segmentation by calculation type and scenario
- Indicates server cannot handle full cross-product of scenarios simultaneously

#### Threading/Parallelization Constraints
**Line 16:** `OPTIONS PRESENV SPOOL;` 
- Threading options are set but never utilized
- Suggests either:
  - Limited CPU cores available
  - SAS licensing restrictions on parallel processing
  - Network/shared storage bottlenecks preventing effective parallelization

#### Storage/I/O Limitations
**Systematic dataset cleanup** (Lines 294, 348, 410):
```sas
proc datasets library=MEMOIRE;
    delete calculs_2 calculs_3;
run;
```
- Immediate deletion of intermediate datasets after each step
- Suggests limited temporary storage space
- Indicates server cannot maintain large working datasets

**Line 262:** `*option nonotes nosource;` (commented out)
- Log suppression to reduce I/O overhead
- Indicates logging can become a performance bottleneck

#### Network/Shared Storage Constraints
**Path structure:** `\\ssq.local\Groupes\Actuariat_corporatif\...`
- Using network UNC paths for data storage
- Suggests compute server separate from data storage
- Network latency likely impacts performance

### Server Configuration Estimate

#### RAM Limitations
**Inferred: Possibly 8-16 GB RAM**

Evidence suggesting memory constraints:
- Account-by-account processing (100 accounts split sequentially)
- Code comment explicitly mentions memory shortage
- Each account appears to generate ~1M scenario combinations internally
- Segmentation by calculation type (Reserve/Capital) suggests memory pressure

#### CPU Configuration  
**Inferred: Possibly 4-8 cores, potentially underutilized**

Evidence:
- `OPTIONS PRESENV SPOOL` configured but unused indicates threading capability may exist
- Sequential processing could indicate:
  - Limited core count 
  - Network I/O bottlenecks
  - SAS licensing restrictions

#### Storage Architecture
**Inferred: Network-attached storage**

Evidence:
- UNC path: `\\ssq.local\Groupes\Actuariat_corporatif\...` indicates separate compute/storage servers
- Heavy reliance on `MEMLIB` suggests preference for memory over disk I/O
- Aggressive intermediate dataset cleanup may indicate storage space concerns

#### Disk/Temporary Storage
**Inferred: Possibly limited local temp space**

Evidence:
- Immediate deletion of intermediate datasets after each processing step
- Heavy use of memory library instead of temporary disk storage
- May indicate limited local storage capacity

#### Calculation Complexity Analysis
**Computational scale:**
- 100 accounts × 100 external scenarios × 100 projection years = 1M base records
- Each internal calculation: 100 internal scenarios × 100 internal years = 10K sub-records
- Total computational matrix: approximately 10 billion individual calculations

#### SAS Licensing Constraints
**Inferred: Possibly Base SAS with limited parallel processing**

Evidence:
- No apparent utilization of `PROC PARALLEL` or `MP CONNECT` 
- Sequential macro processing instead of threaded execution
- Threading options configured but not implemented

### Possible Server Configuration

**Configuration that could explain the observed patterns:**
- **RAM:** Possibly 8-16 GB 
- **CPU:** May be 4-8 cores
- **Local storage:** Could be 200-500 GB for temp files
- **Network:** Likely corporate network connection to file server
- **SAS License:** May be Base SAS without advanced parallel processing features
- **OS:** Possibly Windows Server (based on UNC paths)

**Potential performance bottleneck hierarchy:**
1. **RAM limitation** (appears to be primary constraint based on code comments)
2. **Network I/O bandwidth** (suggested by storage architecture)
3. **SAS licensing/threading limitations** (inferred from unused threading options)
4. **CPU capacity** (may be adequate but underutilized)

### Overall Assessment

The code demonstrates sophisticated awareness of SAS performance optimization, particularly around:
- **Memory management strategies**
- **Efficient data access patterns** 
- **Resource cleanup practices**

However, the fundamental algorithmic complexity remains a constraint due to the actuarial modeling requirements for comprehensive scenario analysis. The apparent server limitations may have necessitated a highly segmented, sequential processing approach that could be trading execution time for memory efficiency.

## Program Output Structure

### Final Output Table: `SERVEUR.CALCULS_SOMMAIRE`

The program produces a single output table containing aggregated results for each account-scenario combination.

#### Output Fields:

| Field Name | Type | Description | Range/Values |
|------------|------|-------------|--------------|
| `ID_COMPTE` | Numeric | Account identifier | 1 to 100 |
| `scn_eval` | Numeric | External economic scenario number | 1 to 100 |
| `VP_FLUX_DISTRIBUABLES` | Numeric | Present value of distributable cash flows | Real values (can be positive/negative) |

#### Output Dimensions:
- **Number of Rows**: 10,000 (100 accounts × 100 scenarios)
- **Number of Columns**: 3

#### Sample Output Structure:
```
ID_COMPTE    scn_eval    VP_FLUX_DISTRIBUABLES
    1           1              -1,234.56
    1           2               2,345.67
    1           3                 987.43
    ...        ...                ...
    1          100              1,456.78
    2           1              -2,345.67
    2           2               3,456.78
    ...        ...                ...
   100         100              4,567.89
```

### What VP_FLUX_DISTRIBUABLES Represents

The present value of distributable cash flows represents the net cash available for distribution to shareholders after accounting for all policyholder obligations, reserves, and capital requirements.

#### Components (calculated annually and then aggregated):

1. **PROFIT** = FLUX_NET + RESERVE - previous_RESERVE
   - Where FLUX_NET includes:
     - **REVENUS**: Investment fee revenues (negative of investment management fees paid)
     - **FRAIS_GEST**: Management fees charged to policyholders
     - **COMMISSIONS**: Sales and maintenance commissions paid
     - **FRAIS_GEN**: General administrative expenses
     - **PMT_GARANTIE**: Death benefit guarantee payments made

2. **DISTRIBUTABLE AMOUNT** = PROFIT + CAPITAL - previous_CAPITAL
   - Where **CAPITAL** represents required capital under stress scenarios (35% capital shock)

3. **PRESENT VALUE CALCULATION**:
   ```
   VP_FLUX_DISTRIBUABLES = FLUX_DISTRIBUABLES / (1 + HURDLE_RATE)^year
   ```
   - **HURDLE_RATE** = 10% (macro variable HURDLE_RT = 0.10)

### Business Interpretation

- **Positive Values**: Account/scenario combinations generating positive distributable cash flows (profitable for shareholders)
- **Negative Values**: Account/scenario combinations requiring additional capital injection or reserves
- **Aggregate Analysis**: Sum or average across scenarios provides expected value and risk metrics for portfolio management

### Output Generation Process

#### External Loop Processing:
1. Each account projected through 100 economic scenarios over 101 years
2. Cash flows calculated considering mortality, lapses, and investment returns
3. Present values computed using external discount rates

#### Internal Loop Processing:
1. For each external projection point, 100 additional internal scenarios generated
2. Reserve calculations perform standard projections
3. Capital calculations apply 35% capital shock to starting fund values
4. Results aggregated across internal scenarios (mean) to get reserve and capital requirements

#### Final Integration:
1. External cash flows combined with reserve and capital calculations
2. Profit and distributable flows computed with proper lag adjustments
3. Present values calculated using 10% hurdle rate
4. Final aggregation by account and external scenario produces the 10,000-row output table
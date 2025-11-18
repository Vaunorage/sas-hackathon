# ACFC Algorithm CSV Input Data Documentation

## 1. POPULATION.CSV - Policy Data
**Purpose**: Contains the core policy characteristics and parameters for each insurance account in the portfolio. Dataset includes 200 accounts organized into two distinct product groups with different guarantee and fee structures.

### Dataset Overview
- **Total Accounts**: 200 (ID_COMPTE: 1-200)
- **Age Range**: 1-99 years (87 unique ages)
- **Fund Values**: $252.38 - $99,886.85 (unique per account)
- **Product Groups**: Two distinct configurations (Accounts 1-100 vs 101-200)

| Column | Data Type | Description | Business Purpose | Group 1 (1-100) | Group 2 (101-200) |
|--------|-----------|-------------|------------------|------------------|------------------|
| `ID_COMPTE` | Float | **Account Identifier** - Unique identifier for each insurance policy/account | Primary key for linking across all calculations and outputs | 1.0 - 100.0 | 101.0 - 200.0 |
| `MT_VM` | Float | **Market Value (Initial Fund Value)** - Starting fund value in the variable investment account | Base amount for fee calculations and fund projections. Used in formula: `MT_VM(t+1) = MT_VM(t) × [1 + RENDEMENT(s,t) - PC_REVENU_FDS - FRAIS_ADJ(t)]` | $252.38 - $99,886.85 | $252.38 - $99,886.85 |
| `PC_GAR_ECH` | Float | **Maturity Guarantee Percentage** - Percentage of initial premium guaranteed at policy maturity | Determines minimum maturity benefit relative to initial investment | 1.0 (100%) | 0.75 (75%) |
| `MT_GAR_ECH` | Float | **Maturity Guarantee Amount** - Absolute dollar amount guaranteed at policy maturity | Calculated as `MT_VM × PC_GAR_ECH`. Provides downside protection for policyholder | = MT_VM | = MT_VM × 0.75 |
| `PC_GAR_DECES` | Float | **Death Benefit Guarantee Percentage** - Percentage of fund value guaranteed as minimum death benefit | Sets the floor for death benefit calculations relative to current fund value | 1.0 (100%) | 1.0 (100%) |
| `FREQ_RESET_DECES` | Float | **Death Benefit Reset Frequency** - Control parameter for guarantee reset mechanism | Controls when guarantee can increase: `MT_GAR_DECES(t) = MAX(MT_GAR_DECES(t-1), MT_VM(t))`. Value of 1.0 = annual resets, 99.0 = virtually no resets | 1.0 (Annual) | 99.0 (Rare) |
| `MAX_RESET_DECES` | Float | **Maximum Reset Age/Limit** - Maximum threshold for death benefit guarantee resets | Age/parameter limit for guarantee increases. After this threshold, death benefit is locked | 80.0 years | 999.0 (No limit) |
| `MT_GAR_DECES` | Float | **Death Benefit Guarantee Amount** - Current guaranteed death benefit amount | Used in death claim calculation: `Death_Claim(t) = MAX(0, MT_GAR_DECES(t) - MT_VM(t)) × Qx(age+t) × TX_SURVIE(t-1)` | = MT_VM | = MT_VM |
| `PC_REVENU_FDS` | Float | **Fund Revenue Percentage** - Annual management fee rate charged on fund assets | Primary revenue source. Applied as: `FRAIS(t) = -(MT_VM_DEB + RENDEMENT/2) × PC_REVENU_FDS` | 0.022 (2.2%) | 0.020 (2.0%) |
| `PC_HONORAIRES_GEST` | Float | **Management Fee Percentage** - Additional management expense rate | Operating expense: `FRAIS_GEST(t) = -(MT_VM_DEB + RENDEMENT/2) × PC_HONORAIRES_GEST × TX_SURVIE(t-1)` | 0.003 (0.3%) | 0.003 (0.3%) |
| `TX_COMM_VENTE` | Float | **Sales Commission Rate** - Upfront commission paid on initial sale | One-time acquisition cost at policy inception | 0.03 (3%) | 0.03 (3%) |
| `TX_COMM_MAINTIEN` | Float | **Maintenance Commission Rate** - Ongoing commission rate paid annually | Recurring expense: `COMMISSIONS(t) = -(MT_VM_DEB + RENDEMENT/2) × TX_COMM_MAINTIEN × TX_SURVIE(t-1)` | 0.008 (0.8%) | 0.008 (0.8%) |
| `FRAIS_ACQUI` | Float | **Acquisition Expenses** - Fixed dollar amount of initial acquisition costs | One-time setup costs (underwriting, policy issue, etc.) | 900.0 | 900.0 |
| `FRAIS_ADMIN` | Float | **Administrative Expenses** - Fixed annual administrative costs per policy | Recurring fixed cost: `FRAIS_GEN(t) = -FRAIS_ADMIN × TX_SURVIE(t-1)` | 100.0 | 100.0 |
| `age_deb` | Float | **Initial Age** - Policyholder's age at policy inception (1-99 years) | Used for mortality rate lookups: `Qx(age_deb + t)` and determining reset eligibility | 1-99 years | 1-99 years |

### Two Product Configurations

**Product Group 1 (Accounts 1-100): High Guarantee/High Fee**
- **Maturity Guarantee**: 100% of initial fund value
- **Death Benefit Resets**: Annual (FREQ_RESET_DECES = 1.0)
- **Reset Age Limit**: 80 years
- **Management Fee**: 2.2% annually
- **Strategy**: Higher guarantees with higher fees and active guarantee management

**Product Group 2 (Accounts 101-200): Moderate Guarantee/Lower Fee**
- **Maturity Guarantee**: 75% of initial fund value  
- **Death Benefit Resets**: Virtually none (FREQ_RESET_DECES = 99.0)
- **Reset Age Limit**: No practical limit (999 years)
- **Management Fee**: 2.0% annually
- **Strategy**: Lower guarantees with competitive fees and minimal guarantee adjustments

---

## 2. RENDEMENT.CSV - Investment Return Scenarios
**Purpose**: Contains stochastic investment return scenarios for fund value projections under different economic conditions.

| Column | Data Type | Description | Business Purpose | Example Value |
|--------|-----------|-------------|------------------|---------------|
| `an_proj` | Float | **Projection Year** - Year number in the projection timeline (1-100) | Time dimension for investment return lookup in fund projection formula | 1.0 (Year 1) |
| `scn_proj` | Float | **Projection Scenario** - Economic scenario identifier (1-100) | Scenario dimension for stochastic modeling. Links to main algorithm scenario loop | 1.0 (Scenario 1) |
| `RENDEMENT` | Float | **Investment Return** - Annual return rate for the fund in this scenario and year | Core input for fund growth: `MT_VM(t+1) = MT_VM(t) × [1 + RENDEMENT(s,t) - fees]` | 0.1611 (16.11%) |
| `TYPE` | String | **Return Type** - Indicates whether return is for internal or external calculations | Distinguishes between external economic scenarios and internal reserve/capital calculations | "INTERNE" |

---

## 3. TX_DECES.CSV - Mortality Tables
**Purpose**: Provides age-specific mortality rates for calculating death probabilities and survival rates.

| Column | Data Type | Description | Business Purpose | Example Value |
|--------|-----------|-------------|------------------|---------------|
| `AGE` | Float | **Attained Age** - Age of the insured person | Lookup key for mortality rates. Used as `age_deb + projection_year` | 1.0 years |
| `QX` | Float | **Mortality Rate** - Annual probability of death at this age | Used in survival calculation: `TX_SURVIE(t+1) = TX_SURVIE(t) × [1 - Qx(age+t)] × [1 - WX(t)]` and death claims | 0.0002 (0.02%) |

---

## 4. TX_INTERET.CSV - External Discount Rates
**Purpose**: Contains discount factors for present value calculations of external cash flows.

| Column | Data Type | Description | Business Purpose | Example Value |
|--------|-----------|-------------|------------------|---------------|
| `an_proj` | Float | **Projection Year** - Year number for discount factor application | Time dimension for discounting cash flows to present value | 1.0 (Year 1) |
| `TX_ACTU` | Float | **Discount Factor** - Present value factor for this projection year | Applied as: `VP_FLUX_NET(t) = FLUX_NET(t) × TX_ACTU(t)` for external cash flows | 0.9744 |

**Note**: Discount factors typically follow the pattern: `TX_ACTU(t) = 1/(1+r)^t` where r is the discount rate.

---

## 5. TX_RETRAIT.CSV - Lapse Rate Table
**Purpose**: Provides duration-dependent lapse (voluntary surrender) rates for calculating policy persistency.

| Column | Data Type | Description | Business Purpose | Example Value |
|--------|-----------|-------------|------------------|---------------|
| `an_proj` | Float | **Policy Duration (Years)** - Number of years since policy inception | Time dimension for lapse rate lookup based on policy duration | 1.0 (Year 1) |
| `WX` | Float | **Lapse Rate** - Annual probability of voluntary policy surrender | Used in survival calculation: `TX_SURVIE(t+1) = TX_SURVIE(t) × [1 - Qx(age+t)] × [1 - WX(t)]` | 0.01 (1%) |

**Pattern**: Lapse rates typically decrease with policy duration as policyholders become more committed over time.

---

## 6. TX_INTERET_INT.CSV - Internal Discount Rates
**Purpose**: Contains discount factors used for internal reserve and capital calculations, may differ from external rates.

| Column | Data Type | Description | Business Purpose | Example Value |
|--------|-----------|-------------|------------------|---------------|
| `an_eval` | Float | **Evaluation Year** - Year for internal discount rate application | Time dimension for internal present value calculations | 1.0 (Year 1) |
| `TX_ACTU_INT` | Float | **Internal Discount Factor** - Present value factor for internal calculations | Used in internal adjustments: `VP_FLUX_NET_ADJ(t) = VP_FLUX_NET(t) / TX_ACTU_INT(eval_year)` | 0.9744 |

**Usage**: These rates are used in the nested internal calculations for reserve and capital requirements, potentially representing different risk-free rates or regulatory discount requirements.

---

## Data Relationships and Algorithm Integration

### **Primary Data Flow**:
1. **POPULATION** provides the base policy parameters and initial values
2. **RENDEMENT** drives the fund value projections across scenarios
3. **TX_DECES** determines mortality-based decrements and death claims
4. **TX_RETRAIT** determines lapse-based decrements
5. **TX_INTERET** discounts external cash flows to present value
6. **TX_INTERET_INT** handles internal calculation discounting

### **Key Lookup Patterns**:
- **Age-based**: `TX_DECES[age_deb + projection_year]`
- **Duration-based**: `TX_RETRAIT[projection_year]`
- **Scenario-Time based**: `RENDEMENT[scenario, year]`
- **Time-based**: `TX_INTERET[projection_year]` and `TX_INTERET_INT[evaluation_year]`

### **Data Quality Requirements**:
- All rates must be between 0 and 1 (except RENDEMENT which can be negative)
- Mortality and lapse rates should follow actuarial patterns
- Discount factors should decrease with time
- Investment returns should reflect realistic economic scenarios
- All ID_COMPTE values must be unique in POPULATION

This documentation provides the foundation for understanding how each CSV input feeds into the complex ACFC algorithm calculations.
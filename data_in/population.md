# Population Dataset Analysis

## Overview
This dataset contains financial information for 200 individual accounts with various numerical parameters including amounts, percentages, and fees.

## Dataset Structure
- **Rows**: 200 accounts
- **Columns**: 15 variables
- **Data Quality**: Complete dataset with no missing values
- **File Format**: Originally SAS7BDAT, converted to CSV

## Variable Descriptions

### Account Information
- **ID_COMPTE**: Account identifier (1-200, unique for each account)
- **age_deb**: Starting age (ranges from 1-99 years, 87 unique values)

### Financial Values
- **MT_VM**: Monetary amount (252.38 - 99,886.85, unique per account)
- **MT_GAR_ECH**: Monetary amount (equals MT_VM × PC_GAR_ECH)
- **MT_GAR_DECES**: Monetary amount (equals MT_VM for all accounts)

### Guarantee Percentages
- **PC_GAR_ECH**: Maturity guarantee percentage 
  - 100% for accounts 1-100
  - 75% for accounts 101-200
- **PC_GAR_DECES**: Death benefit percentage (100% for all accounts)

### Reset Parameters
- **FREQ_RESET_DECES**: Reset frequency for death benefit
  - 1.0 for accounts 1-100
  - 99.0 for accounts 101-200
- **MAX_RESET_DECES**: Maximum resets allowed
  - 80.0 for accounts 1-100  
  - 999.0 for accounts 101-200

### Fee Structure (Fixed across all accounts)
- **PC_REVENU_FDS**: Fund revenue percentage
  - 2.2% for accounts 1-100
  - 2.0% for accounts 101-200
- **PC_HONORAIRES_GEST**: Management fee percentage (0.3% for all)
- **TX_COMM_VENTE**: Sales commission rate (3.0% for all)
- **TX_COMM_MAINTIEN**: Maintenance commission rate (0.8% for all)
- **FRAIS_ACQUI**: Fixed amount (900.0 for all)
- **FRAIS_ADMIN**: Fixed amount (100.0 for all)

## Key Observations

### Two Distinct Product Groups
The dataset clearly shows two different product configurations:

**Group 1 (Accounts 1-100)**
- PC_GAR_ECH = 1.0 (100%)
- FREQ_RESET_DECES = 1.0
- MAX_RESET_DECES = 80.0
- PC_REVENU_FDS = 0.022 (2.2%)

**Group 2 (Accounts 101-200)**
- PC_GAR_ECH = 0.75 (75%)
- FREQ_RESET_DECES = 99.0
- MAX_RESET_DECES = 999.0
- PC_REVENU_FDS = 0.02 (2.0%)

### Age Distribution
- Wide age range from 1 to 99 years
- 87 unique ages represented across 200 accounts
- Some ages appear multiple times

### Monetary Amounts
- Variable MT_VM values: 252.38 to 99,885.85
- Each account has unique MT_VM value

### Fee Consistency
- Uniform fee structure across all accounts
- Fixed acquisition and administrative costs regardless of account size
- Consistent commission rates for sales and maintenance

## Data Structure Observations

The dataset shows two distinct parameter configurations across accounts 1-100 and 101-200, with different values for guarantee percentages, reset frequencies, and revenue rates.

## Data Integrity
- No missing values detected
- All calculations are consistent (MT_GAR_ECH = MT_VM × PC_GAR_ECH)
- Clear parameter segmentation between the two account groups
- Account IDs are sequential and unique
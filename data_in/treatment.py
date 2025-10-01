import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv('/home/vaunorage/PycharmProjects/sas-hackathon/data_in/population.csv')

# Function to clean currency and percentage columns
def clean_currency(value):
    """Remove $ and commas, convert to float"""
    if pd.isna(value):
        return np.nan
    return float(str(value).replace('$', '').replace(',', ''))

def clean_percentage(value):
    """Remove % and convert to decimal (e.g., 100% -> 1.0)"""
    if pd.isna(value):
        return np.nan
    return float(str(value).replace('%', '')) / 100

# List of currency columns
currency_cols = ['MT_VM', 'MT_GAR_ECH', 'MT_GAR_DECES', 'FRAIS_ACQUI', 'FRAIS_ADMIN']

# List of percentage columns
percentage_cols = ['PC_GAR_ECH', 'PC_GAR_DECES', 'PC_REVENU_FDS',
                   'PC_HONORAIRES_GEST', 'TX_COMM_VENTE', 'TX_COMM_MAINTIEN']

# Clean currency columns
for col in currency_cols:
    df[col] = df[col].apply(clean_currency)

# Clean percentage columns
for col in percentage_cols:
    df[col] = df[col].apply(clean_percentage)

# Display the cleaned dataframe
print(df.dtypes)
print("\n", df.head())

df.to_csv("/home/vaunorage/PycharmProjects/sas-hackathon/data_in/population_fixed.csv")

#%%

# Read the CSV file
df_mortality = pd.read_csv('/home/vaunorage/PycharmProjects/sas-hackathon/data_in/tx_deces.csv')

# Function to clean percentage columns
def clean_percentage(value):
    """Remove % and convert to decimal (e.g., 0.020% -> 0.0002)"""
    if pd.isna(value):
        return np.nan
    return float(str(value).replace('%', '')) / 100

# Clean the QX column
df_mortality['QX'] = df_mortality['QX'].apply(clean_percentage)

# Display the cleaned dataframe
print(df_mortality.dtypes)
print("\n", df_mortality.head())

df_mortality.to_csv("/home/vaunorage/PycharmProjects/sas-hackathon/data_in/tx_deces_fixed.csv")

#%%
# Read the CSV file
df_discount = pd.read_csv('/home/vaunorage/PycharmProjects/sas-hackathon/data_in/tx_interet.csv')

# Function to clean percentage columns
def clean_percentage(value):
    """Remove % and convert to decimal (e.g., 97.44% -> 0.9744)"""
    if pd.isna(value):
        return np.nan
    return float(str(value).replace('%', '')) / 100

# Clean the TX_ACTU column
df_discount['TX_ACTU'] = df_discount['TX_ACTU'].apply(clean_percentage)

# Display the cleaned dataframe
print(df_discount.dtypes)
print("\n", df_discount.head())

df_mortality.to_csv("/home/vaunorage/PycharmProjects/sas-hackathon/data_in/tx_interet_fixed.csv")

#%%
# Read the CSV file
df_discount = pd.read_csv('/home/vaunorage/PycharmProjects/sas-hackathon/data_in/tx_interet_int.csv')

# Function to clean percentage columns
def clean_percentage(value):
    """Remove % and convert to decimal (e.g., 97.44% -> 0.9744)"""
    if pd.isna(value):
        return np.nan
    return float(str(value).replace('%', '')) / 100

# Clean the TX_ACTU column
df_discount['TX_ACTU_INT'] = df_discount['TX_ACTU_INT'].apply(clean_percentage)

# Display the cleaned dataframe
print(df_discount.dtypes)
print("\n", df_discount.head())

df_mortality.to_csv("/home/vaunorage/PycharmProjects/sas-hackathon/data_in/tx_interet_int_fixed.csv")

#%%
# Read the CSV file
df_lapse = pd.read_csv('/home/vaunorage/PycharmProjects/sas-hackathon/data_in/tx_retrait.csv')

# Function to clean percentage columns
def clean_percentage(value):
    """Remove % and convert to decimal (e.g., 1.000% -> 0.01)"""
    if pd.isna(value):
        return np.nan
    return float(str(value).replace('%', '')) / 100

# Clean the WX column
df_lapse['WX'] = df_lapse['WX'].apply(clean_percentage)

# Display the cleaned dataframe
print(df_lapse.dtypes)
print("\n", df_lapse.head())

df_lapse.to_csv("/home/vaunorage/PycharmProjects/sas-hackathon/data_in/tx_retrait_fixed.csv")

# Now you can perform operations like:
# df_lapse[df_lapse['an_proj'] == 3]['WX'].values[0]
# df_lapse['WX'].mean()
# df_lapse[df_lapse['an_proj'] <= 5]['WX'].sum()


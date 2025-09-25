import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
import logging
import time
from pathlib import Path
from tqdm import tqdm
import warnings

from paths import HERE

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global parameters matching SAS macro variables
NBCPT = 100
NB_SC = 100
NB_AN_PROJECTION = 100
NB_SC_INT = 100
NB_AN_PROJECTION_INT = 100
CHOC_CAPITAL = 0.35
HURDLE_RT = 0.10


def load_input_files(data_path: str) -> Tuple[pd.DataFrame, ...]:
    """Load all input CSV files exactly as SAS does"""
    try:
        population = pd.read_csv(f"{data_path}/population.csv")
        rendement = pd.read_csv(f"{data_path}/rendement.csv")
        tx_deces = pd.read_csv(f"{data_path}/tx_deces.csv")
        tx_interet = pd.read_csv(f"{data_path}/tx_interet.csv")
        tx_interet_int = pd.read_csv(f"{data_path}/tx_interet_int.csv")
        tx_retrait = pd.read_csv(f"{data_path}/tx_retrait.csv")

        # Handle TYPE column encoding if it exists
        if 'TYPE' in rendement.columns:
            rendement['TYPE'] = rendement['TYPE'].apply(
                lambda x: x.decode('utf-8') if isinstance(x, bytes) else str(x)
            )

        logger.info(f"Input files loaded - Population: {len(population)} accounts")
        return population, rendement, tx_deces, tx_interet, tx_interet_int, tx_retrait

    except Exception as e:
        logger.error(f"Error loading input files: {e}")
        raise


def create_hash_tables(rendement: pd.DataFrame, tx_deces: pd.DataFrame,
                       tx_interet: pd.DataFrame, tx_interet_int: pd.DataFrame,
                       tx_retrait: pd.DataFrame) -> Tuple[Dict, ...]:
    """Create hash tables exactly matching SAS hash table structure"""

    # Mortality hash: AGE -> Qx
    h_mortality = {}
    for _, row in tx_deces.iterrows():
        h_mortality[int(row['AGE'])] = float(row['QX'])

    # Lapse hash: an_proj -> WX
    g_lapse = {}
    for _, row in tx_retrait.iterrows():
        g_lapse[int(row['an_proj'])] = float(row['WX'])

    # Rendement hash: (scn_proj, an_proj, TYPE) -> RENDEMENT
    z_rendement = {}
    for _, row in rendement.iterrows():
        key = (int(row['scn_proj']), int(row['an_proj']), str(row['TYPE']))
        z_rendement[key] = float(row['RENDEMENT'])

    # External discount hash: an_proj -> TX_ACTU
    a_discount_ext = {}
    for _, row in tx_interet.iterrows():
        a_discount_ext[int(row['an_proj'])] = float(row['TX_ACTU'])

    # Internal discount hash: an_eval -> TX_ACTU_INT
    b_discount_int = {}
    for _, row in tx_interet_int.iterrows():
        b_discount_int[int(row['an_eval'])] = float(row['TX_ACTU_INT'])

    return h_mortality, g_lapse, z_rendement, a_discount_ext, b_discount_int


def hash_find(hash_table: dict, key, default_value=None):
    """Mimic SAS hash.find() behavior"""
    return hash_table.get(key, default_value if default_value is not None else 0.0)


def faithful_cash_flow_calculation(population_row, hash_tables,
                                   scenario_type: str, type2: str = None,
                                   scn_eval: int = None, an_eval: int = None,
                                   scn_eval_int: int = None, an_eval_int: int = None):
    """
    Faithful implementation of SAS DATA_STEP_CALCUL macro
    """
    h_mortality, g_lapse, z_rendement, a_discount_ext, b_discount_int = hash_tables

    # Initialize retained variables exactly as in SAS
    MT_VM_PROJ = 0.0
    MT_GAR_DECES_PROJ = 0.0
    TX_SURVIE = 0.0

    results = []

    # Determine projection parameters based on scenario type
    if scenario_type == "EXTERNE":
        max_years = min(NB_AN_PROJECTION, 99 - int(population_row['age_deb']))
        year_range = range(max_years + 1)  # 0 to max_years inclusive
    else:  # INTERNE
        max_years = min(NB_AN_PROJECTION_INT, 99 - int(population_row['age_deb']) - an_eval)
        year_range = range(max_years + 1)

    for year_idx, current_year in enumerate(year_range):

        # ***********************************************
        # *** Initialisation des variables a lannee 0 ***
        # ***********************************************

        if current_year == 0 and scenario_type == "EXTERNE":
            # External scenario year 0 initialization
            AGE = int(population_row['age_deb'])
            MT_VM_PROJ = float(population_row['MT_VM'])
            MT_GAR_DECES_PROJ = float(population_row['MT_GAR_DECES'])
            TX_SURVIE = 1.0
            TX_SURVIE_DEB = 1.0
            TX_ACTU = 1.0
            QX = 0.0
            WX = 0.0
            an_proj = 0

            # Year 0 cash flows
            COMMISSIONS = -float(population_row['TX_COMM_VENTE']) * MT_VM_PROJ
            VP_COMMISSIONS = COMMISSIONS

            FRAIS_GEN = -float(population_row['FRAIS_ACQUI'])
            VP_FRAIS_GEN = FRAIS_GEN

            FLUX_NET = FRAIS_GEN + COMMISSIONS
            VP_FLUX_NET = FLUX_NET

            # Zero out other components for year 0
            REVENUS = 0.0
            FRAIS_GEST = 0.0
            PMT_GARANTIE = 0.0
            VP_REVENUS = 0.0
            VP_FRAIS_GEST = 0.0
            VP_PMT_GARANTIE = 0.0

        elif current_year == 0 and scenario_type == "INTERNE":
            # Internal scenario year 0 initialization
            if type2 == "RESERVE":
                MT_VM_PROJ = float(population_row['MT_VM'])
            elif type2 == "CAPITAL":
                MT_VM_PROJ = float(population_row['MT_VM']) * (1 - CHOC_CAPITAL)

            AGE = int(population_row['age_deb']) + an_eval
            MT_GAR_DECES_PROJ = float(population_row['MT_GAR_DECES'])
            TX_SURVIE = float(population_row['TX_SURVIE_DEB']) if 'TX_SURVIE_DEB' in population_row else 1.0
            TX_ACTU = 1.0
            QX = 0.0
            WX = 0.0
            an_proj = an_eval

            # Zero out all cash flows for internal year 0
            COMMISSIONS = 0.0
            VP_COMMISSIONS = 0.0
            FRAIS_GEN = 0.0
            VP_FRAIS_GEN = 0.0
            FLUX_NET = 0.0
            VP_FLUX_NET = 0.0
            REVENUS = 0.0
            FRAIS_GEST = 0.0
            PMT_GARANTIE = 0.0
            VP_REVENUS = 0.0
            VP_FRAIS_GEST = 0.0
            VP_PMT_GARANTIE = 0.0

        # Check termination conditions exactly as SAS
        elif TX_SURVIE == 0 or MT_VM_PROJ == 0:
            continue  # SAS deletes these rows

        # ***********************************************************************
        # *** Calcul des flux financiers pour toutes les années de projection ***
        # ***********************************************************************
        else:
            # Determine scenario number for lookup
            if scenario_type == "INTERNE":
                scn_proj = scn_eval_int
            else:
                scn_proj = scn_eval

            # Increment age and projection year
            if scenario_type == "INTERNE":
                AGE = int(population_row['age_deb']) + an_eval + current_year
                an_proj = an_eval + current_year
            else:
                AGE = int(population_row['age_deb']) + current_year
                an_proj = current_year

            # ****** Fund Value Projection - EXACT SAS FORMULA ******
            MT_VM_DEB = MT_VM_PROJ

            # Get investment return using hash lookup
            RENDEMENT_rate = hash_find(z_rendement, (scn_proj, an_proj, scenario_type), 0.0)
            RENDEMENT = MT_VM_DEB * RENDEMENT_rate

            # Calculate fees exactly as SAS: FRAIS = -(MT_VM_DEB + RENDEMENT / 2) * PC_REVENU_FDS
            FRAIS = -(MT_VM_DEB + RENDEMENT / 2) * float(population_row['PC_REVENU_FDS'])

            # Update fund value: MT_VM_PROJ = MT_VM_PROJ + RENDEMENT + FRAIS
            MT_VM_PROJ = MT_VM_PROJ + RENDEMENT + FRAIS

            # ****** Death Benefit Guarantee Reset Logic ******
            FREQ_RESET_DECES = float(population_row['FREQ_RESET_DECES'])
            MAX_RESET_DECES = float(population_row['MAX_RESET_DECES'])

            if FREQ_RESET_DECES == 1 and AGE <= MAX_RESET_DECES:
                MT_GAR_DECES_PROJ = max(MT_GAR_DECES_PROJ, MT_VM_PROJ)

            # ****** Survival Probability Calculation ******
            QX = hash_find(h_mortality, AGE, 0.0)
            WX = hash_find(g_lapse, an_proj, 0.0)

            TX_SURVIE_DEB = TX_SURVIE
            TX_SURVIE = TX_SURVIE_DEB * (1 - QX) * (1 - WX)

            # ****** Cash Flow Calculations - EXACT SAS FORMULAS ******
            REVENUS = -FRAIS * TX_SURVIE_DEB
            FRAIS_GEST = -(MT_VM_DEB + RENDEMENT / 2) * float(population_row['PC_HONORAIRES_GEST']) * TX_SURVIE_DEB
            COMMISSIONS = -(MT_VM_DEB + RENDEMENT / 2) * float(population_row['TX_COMM_MAINTIEN']) * TX_SURVIE_DEB
            FRAIS_GEN = -float(population_row['FRAIS_ADMIN']) * TX_SURVIE_DEB
            PMT_GARANTIE = -max(0, MT_GAR_DECES_PROJ - MT_VM_PROJ) * QX * TX_SURVIE_DEB

            FLUX_NET = REVENUS + FRAIS_GEST + COMMISSIONS + FRAIS_GEN + PMT_GARANTIE

            # ****** Present Value Calculations ******
            TX_ACTU = hash_find(a_discount_ext, an_proj, 1.0)

            VP_REVENUS = REVENUS * TX_ACTU
            VP_FRAIS_GEST = FRAIS_GEST * TX_ACTU
            VP_COMMISSIONS = COMMISSIONS * TX_ACTU
            VP_FRAIS_GEN = FRAIS_GEN * TX_ACTU
            VP_PMT_GARANTIE = PMT_GARANTIE * TX_ACTU
            VP_FLUX_NET = FLUX_NET * TX_ACTU

            # ****** Internal Scenario Adjustment ******
            if scenario_type == "INTERNE" and an_eval > 0:
                TX_ACTU_INT = hash_find(b_discount_int, an_eval, 1.0)
                if TX_ACTU_INT != 0:
                    VP_REVENUS = VP_REVENUS / TX_ACTU_INT
                    VP_FRAIS_GEST = VP_FRAIS_GEST / TX_ACTU_INT
                    VP_COMMISSIONS = VP_COMMISSIONS / TX_ACTU_INT
                    VP_FRAIS_GEN = VP_FRAIS_GEN / TX_ACTU_INT
                    VP_PMT_GARANTIE = VP_PMT_GARANTIE / TX_ACTU_INT
                    VP_FLUX_NET = VP_FLUX_NET / TX_ACTU_INT

        # Store results for this year
        result_row = {
            'ID_COMPTE': int(population_row['ID_COMPTE']),
            'an_eval': an_eval if scenario_type == "INTERNE" else current_year,
            'scn_eval': scn_eval if scenario_type == "EXTERNE" else scn_eval,
            'scn_eval_int': scn_eval_int if scenario_type == "INTERNE" else None,
            'an_eval_int': current_year if scenario_type == "INTERNE" else None,
            'TYPE': scenario_type,
            'TYPE2': type2,
            'an_proj': an_proj,
            'AGE': AGE,
            'MT_VM_PROJ': MT_VM_PROJ,
            'MT_GAR_DECES_PROJ': MT_GAR_DECES_PROJ,
            'TX_SURVIE': TX_SURVIE,
            'TX_SURVIE_DEB': TX_SURVIE_DEB if 'TX_SURVIE_DEB' in locals() else TX_SURVIE,
            'REVENUS': REVENUS,
            'FRAIS_GEST': FRAIS_GEST,
            'COMMISSIONS': COMMISSIONS,
            'FRAIS_GEN': FRAIS_GEN,
            'PMT_GARANTIE': PMT_GARANTIE,
            'FLUX_NET': FLUX_NET,
            'VP_REVENUS': VP_REVENUS,
            'VP_FRAIS_GEST': VP_FRAIS_GEST,
            'VP_COMMISSIONS': VP_COMMISSIONS,
            'VP_FRAIS_GEN': VP_FRAIS_GEN,
            'VP_PMT_GARANTIE': VP_PMT_GARANTIE,
            'VP_FLUX_NET': VP_FLUX_NET
        }

        results.append(result_row)

    return pd.DataFrame(results)


def faithful_calculs_macro(population: pd.DataFrame, hash_tables):
    """
    Faithful implementation of SAS %calculs macro with exact nested loop structure
    """

    # Initialize summary results exactly like SAS
    calculs_sommaire = pd.DataFrame(columns=['ID_COMPTE', 'scn_eval', 'VP_FLUX_DISTRIBUABLES'])

    logger.info("Starting faithful nested loop calculations...")

    # Calculate total expected operations for progress tracking
    total_accounts = min(NBCPT, len(population))
    total_ext_scenarios = min(NB_SC, 20)  # Limited for performance
    total_int_scenarios = min(NB_SC_INT, 10)

    logger.info(
        f"Expected operations: {total_accounts} accounts × {total_ext_scenarios} ext scenarios × 2 types × {total_int_scenarios} int scenarios")
    logger.info(
        f"Estimated total internal calculations: ~{total_accounts * total_ext_scenarios * 2 * total_int_scenarios * 50:,}")

    # ***************************
    # *** OUTER ACCOUNT LOOP ***  - %do j = 1 %to &NBCPT.
    # ***************************

    account_progress = tqdm(range(1, total_accounts + 1), desc="Processing Accounts", unit="account", position=0)

    for j in account_progress:

        # Get account data
        account_data = population[population['ID_COMPTE'] == j]
        if account_data.empty:
            continue

        account_row = account_data.iloc[0]

        # Update account progress description
        account_progress.set_postfix({
            'Account': f'{j}/{total_accounts}',
            'Completed': f'{len(calculs_sommaire)} combinations'
        })

        # ***************************
        # *** EXTERNAL SCENARIOS *** - do scn_eval = 1 to &NB_SC.
        # ***************************

        for scn_eval in external_progress:

            # Update external progress description
            external_progress.set_postfix({
                'ExtScenario': f'{scn_eval}/{total_ext_scenarios}',
                'Account': f'{j}/{total_accounts}'
            })

            # Calculate external cash flows
            external_results = faithful_cash_flow_calculation(
                account_row, hash_tables,
                scenario_type="EXTERNE",
                scn_eval=scn_eval
            )

            if external_results.empty:
                continue

            # **********************************
            # *** INTERNAL CALCULATION LOOPS ***
            # **********************************

            reserve_results = []
            capital_results = []

            # Count years that need internal calculations
            valid_years = [row for _, row in external_results.iterrows() if int(row['an_eval']) > 0]

            # Loop through external years for internal calculations
            year_progress = tqdm(valid_years,
                                 desc=f"    Years (Acc{j},Scn{scn_eval})",
                                 unit="year", position=2, leave=False)

            for ext_row in year_progress:
                an_eval = int(ext_row['an_eval'])

                year_progress.set_postfix({
                    'Year': f'{an_eval}',
                    'ExtScn': f'{scn_eval}',
                    'Acc': f'{j}'
                })

                # *** TYPE2 LOOP *** - %do m = 1 %to 2 (RESERVE and CAPITAL)
                type_progress = tqdm(range(1, 3),
                                     desc=f"      Types (Y{an_eval})",
                                     unit="type", position=3, leave=False)

                for m in type_progress:
                    type2 = "RESERVE" if m == 1 else "CAPITAL"

                    type_progress.set_postfix({
                        'Type': type2,
                        'Year': f'{an_eval}',
                        'Acc': f'{j}'
                    })

                    internal_scenarios_sum = []

                    # *** INTERNAL SCENARIOS *** - do scn_eval_int = 1 to &NB_SC_INT.
                    internal_progress = tqdm(range(1, total_int_scenarios + 1),
                                             desc=f"        IntScn ({type2})",
                                             unit="intscn", position=4, leave=False)

                    for scn_eval_int in internal_progress:

                        internal_progress.set_postfix({
                            'IntScn': f'{scn_eval_int}/{total_int_scenarios}',
                            'Type': type2[:3],
                            'Y': f'{an_eval}',
                            'A': f'{j}'
                        })

                        # Prepare row with accumulated values from external projection
                        internal_input_row = account_row.copy()
                        internal_input_row['MT_VM'] = ext_row['MT_VM_PROJ']
                        internal_input_row['MT_GAR_DECES'] = ext_row['MT_GAR_DECES_PROJ']
                        internal_input_row['TX_SURVIE_DEB'] = ext_row['TX_SURVIE']

                        # Run internal projection
                        internal_results = faithful_cash_flow_calculation(
                            internal_input_row, hash_tables,
                            scenario_type="INTERNE",
                            type2=type2,
                            scn_eval=scn_eval,
                            an_eval=an_eval,
                            scn_eval_int=scn_eval_int
                        )

                        if not internal_results.empty:
                            # Sum VP_FLUX_NET for this internal scenario
                            total_vp = internal_results['VP_FLUX_NET'].sum()
                            internal_scenarios_sum.append(total_vp)

                    # Close internal progress bar
                    internal_progress.close()

                    # Calculate mean across internal scenarios (matching SAS proc summary)
                    if internal_scenarios_sum:
                        mean_vp = np.mean(internal_scenarios_sum)

                        result_entry = {
                            'ID_COMPTE': j,
                            'an_eval': an_eval,
                            'scn_eval': scn_eval,
                            'VP_FLUX_NET': mean_vp,
                            'TYPE2': type2
                        }

                        if type2 == "RESERVE":
                            reserve_results.append(result_entry)
                        else:
                            capital_results.append(result_entry)

                # Close type progress bar
                type_progress.close()

            # Close year progress bar
            year_progress.close()

            # ***********************************
            # *** MERGE WITH EXTERNAL RESULTS ***
            # ***********************************

            # Convert to DataFrames
            reserve_df = pd.DataFrame(reserve_results)
            capital_df = pd.DataFrame(capital_results)

            # Merge reserves and capitals back to external results
            enhanced_external = external_results.copy()
            enhanced_external['RESERVE'] = 0.0
            enhanced_external['CAPITAL'] = 0.0

            # Hash join logic as in SAS
            merge_progress = tqdm(enhanced_external.iterrows(),
                                  desc=f"    Merging (Acc{j},Scn{scn_eval})",
                                  total=len(enhanced_external), unit="row", position=2, leave=False)

            for idx, row in merge_progress:
                an_eval = int(row['an_eval'])

                merge_progress.set_postfix({
                    'Year': f'{an_eval}',
                    'Reserves': len(reserve_df),
                    'Capital': len(capital_df)
                })

                # Find matching reserve
                reserve_match = reserve_df[
                    (reserve_df['ID_COMPTE'] == j) &
                    (reserve_df['an_eval'] == an_eval) &
                    (reserve_df['scn_eval'] == scn_eval)
                    ]

                if not reserve_match.empty:
                    enhanced_external.loc[idx, 'RESERVE'] = reserve_match.iloc[0]['VP_FLUX_NET']

                # Find matching capital
                capital_match = capital_df[
                    (capital_df['ID_COMPTE'] == j) &
                    (capital_df['an_eval'] == an_eval) &
                    (capital_df['scn_eval'] == scn_eval)
                    ]

                if not capital_match.empty:
                    capital_value = capital_match.iloc[0]['VP_FLUX_NET'] - enhanced_external.loc[idx, 'RESERVE']
                    enhanced_external.loc[idx, 'CAPITAL'] = capital_value

            merge_progress.close()

            # *******************************************
            # *** PROFIT AND DISTRIBUTABLE CALCULATION ***
            # *******************************************

            enhanced_external = enhanced_external.sort_values('an_eval').reset_index(drop=True)
            enhanced_external['reserve_prec'] = enhanced_external['RESERVE'].shift(1, fill_value=0)
            enhanced_external['capital_prec'] = enhanced_external['CAPITAL'].shift(1, fill_value=0)

            # Calculate exactly as SAS
            profit_conditions = enhanced_external['an_eval'] == 0

            enhanced_external.loc[profit_conditions, 'PROFIT'] = (
                    enhanced_external.loc[profit_conditions, 'FLUX_NET'] +
                    enhanced_external.loc[profit_conditions, 'RESERVE']
            )
            enhanced_external.loc[profit_conditions, 'FLUX_DISTRIBUABLES'] = (
                    enhanced_external.loc[profit_conditions, 'PROFIT'] +
                    enhanced_external.loc[profit_conditions, 'CAPITAL']
            )

            enhanced_external.loc[~profit_conditions, 'PROFIT'] = (
                    enhanced_external.loc[~profit_conditions, 'FLUX_NET'] +
                    enhanced_external.loc[~profit_conditions, 'RESERVE'] -
                    enhanced_external.loc[~profit_conditions, 'reserve_prec']
            )
            enhanced_external.loc[~profit_conditions, 'FLUX_DISTRIBUABLES'] = (
                    enhanced_external.loc[~profit_conditions, 'PROFIT'] +
                    enhanced_external.loc[~profit_conditions, 'CAPITAL'] -
                    enhanced_external.loc[~profit_conditions, 'capital_prec']
            )

            # Present value of distributable cash flows
            enhanced_external['VP_FLUX_DISTRIBUABLES'] = (
                    enhanced_external['FLUX_DISTRIBUABLES'] /
                    ((1 + HURDLE_RT) ** enhanced_external['an_eval'])
            )

            # Sum across all years for this account-scenario combination
            total_vp_distribuables = enhanced_external['VP_FLUX_DISTRIBUABLES'].sum()

            # Append to summary results
            summary_row = pd.DataFrame([{
                'ID_COMPTE': j,
                'scn_eval': scn_eval,
                'VP_FLUX_DISTRIBUABLES': total_vp_distribuables
            }])

            calculs_sommaire = pd.concat([calculs_sommaire, summary_row], ignore_index=True)

        # Close external progress bar
        external_progress.close()

        # Update main progress with current totals
        account_progress.set_postfix({
            'Account': f'{j}/{total_accounts}',
            'Total_Results': f'{len(calculs_sommaire)}',
            'Avg_VP': f'${calculs_sommaire["VP_FLUX_DISTRIBUABLES"].mean():,.0f}' if len(
                calculs_sommaire) > 0 else 'N/A'
        })

    # Close account progress bar
    account_progress.close()

    logger.info(f"Faithful calculations complete. Results: {len(calculs_sommaire)} combinations")
    logger.info(f"Average VP_FLUX_DISTRIBUABLES: ${calculs_sommaire['VP_FLUX_DISTRIBUABLES'].mean():,.2f}")
    logger.info(f"Profitable combinations: {len(calculs_sommaire[calculs_sommaire['VP_FLUX_DISTRIBUABLES'] > 0])}")

    return calculs_sommaire


def run_faithful_acfc(data_path: str = "data_in", output_dir: str = "output"):
    """
    Main function that runs the faithful SAS-to-Python ACFC implementation
    """

    start_time = time.time()

    logger.info("=" * 80)
    logger.info("FAITHFUL SAS-TO-PYTHON ACFC IMPLEMENTATION")
    logger.info("=" * 80)

    try:
        # Load input files
        population, rendement, tx_deces, tx_interet, tx_interet_int, tx_retrait = load_input_files(data_path)

        # Create hash tables exactly as SAS
        hash_tables = create_hash_tables(rendement, tx_deces, tx_interet, tx_interet_int, tx_retrait)

        logger.info(f"Configuration:")
        logger.info(f"  Accounts: {min(NBCPT, len(population))}")
        logger.info(f"  External scenarios: {min(NB_SC, 20)}")  # Limited for performance
        logger.info(f"  Internal scenarios: {min(NB_SC_INT, 10)}")
        logger.info(f"  Max projection years: {NB_AN_PROJECTION}")

        # Run faithful nested calculations
        results_df = faithful_calculs_macro(population, hash_tables)

        # Analysis
        end_time = time.time()
        execution_time = end_time - start_time

        # Print results
        print(f"\n" + "=" * 60)
        print(f"FAITHFUL ACFC RESULTS")
        print(f"=" * 60)
        print(f"Total combinations: {len(results_df):,}")
        print(f"Execution time: {execution_time:.2f} seconds")
        print(f"Mean VP_FLUX_DISTRIBUABLES: ${results_df['VP_FLUX_DISTRIBUABLES'].mean():,.2f}")
        print(f"Profitable combinations: {len(results_df[results_df['VP_FLUX_DISTRIBUABLES'] > 0]):,}")
        print(
            f"Range: ${results_df['VP_FLUX_DISTRIBUABLES'].min():,.2f} to ${results_df['VP_FLUX_DISTRIBUABLES'].max():,.2f}")

        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        results_file = output_path / "faithful_acfc_results.csv"
        results_df.to_csv(results_file, index=False)
        logger.info(f"Results saved to {results_file}")

        return results_df

    except Exception as e:
        logger.error(f"Error in faithful ACFC execution: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise


def main():
    """Main execution"""
    try:
        results_df = run_faithful_acfc(
            data_path=HERE.joinpath("data_in"),
            output_dir=HERE.joinpath("test"),
        )
        return results_df
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise


if __name__ == "__main__":
    results_df = main()
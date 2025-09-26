"""
Refactored Actuarial Cash Flow Projection Model.

This script performs financial projections for insurance accounts based on
various economic scenarios. It is designed to replicate the logic of an
original SAS program while improving code structure, clarity, and performance
by leveraging Python's object-oriented and data manipulation capabilities.

The main processing logic is encapsulated within the ActuarialModel class.
The core projection loop remains account -> scenario -> year, as this is the
most efficient structure for path-dependent simulations. The refactoring focuses
on breaking down the main calculation macro into modular, manageable methods.
"""
import logging
import time
import warnings
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from paths import HERE

warnings.filterwarnings('ignore')

# --- Configuration ---
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global parameters matching SAS macro variables
NBCPT = 100
NB_SC = 100
NB_AN_PROJECTION = 100
NB_SC_INT = 100
NB_AN_PROJECTION_INT = 100
CHOC_CAPITAL = 0.35
HURDLE_RT = 0.10


class ActuarialModel:
    """
    Encapsulates the entire actuarial projection process.

    This class loads input data, creates lookup tables (hash maps), and runs
    the nested scenario projections to calculate the present value of
    distributable cash flows for each account.
    """

    def __init__(self, data_path: Path):
        """Initializes the model by loading data and creating hash tables."""
        logger.info("Initializing ActuarialModel...")
        self.population, self.hash_tables = self._load_and_prepare_data(data_path)
        (
            self.h_mortality,
            self.g_lapse,
            self.z_rendement,
            self.a_discount_ext,
            self.b_discount_int,
        ) = self.hash_tables
        logger.info("Model initialized successfully.")

    def _load_and_prepare_data(self, data_path: Path) -> Tuple[pd.DataFrame, Tuple[Dict, ...]]:
        """Loads all input files and creates hash tables."""
        try:
            # Load data
            population = pd.read_csv(data_path / "population.csv").head(NBCPT)
            rendement = pd.read_csv(data_path / "rendement.csv")
            tx_deces = pd.read_csv(data_path / "tx_deces.csv")
            tx_interet = pd.read_csv(data_path / "tx_interet.csv")
            tx_interet_int = pd.read_csv(data_path / "tx_interet_int.csv")
            tx_retrait = pd.read_csv(data_path / "tx_retrait.csv")

            if 'TYPE' in rendement.columns:
                rendement['TYPE'] = rendement['TYPE'].str.decode('utf-8').fillna('')

            logger.info(f"Input files loaded - Population: {len(population)} accounts")

            # Create hash tables
            hash_tables = self._create_hash_tables(rendement, tx_deces, tx_interet, tx_interet_int, tx_retrait)
            return population, hash_tables
        except Exception as e:
            logger.error(f"Error loading or preparing data: {e}")
            raise

    @staticmethod
    def _create_hash_tables(rendement, tx_deces, tx_interet, tx_interet_int, tx_retrait) -> Tuple[Dict, ...]:
        """Creates hash tables from input dataframes for fast lookups."""
        h_mortality = dict(zip(tx_deces['AGE'], tx_deces['QX']))
        g_lapse = dict(zip(tx_retrait['an_proj'], tx_retrait['WX']))
        z_rendement = {
            (int(r['scn_proj']), int(r['an_proj']), str(r['TYPE'])): float(r['RENDEMENT'])
            for _, r in rendement.iterrows()
        }
        a_discount_ext = dict(zip(tx_interet['an_proj'], tx_interet['TX_ACTU']))
        b_discount_int = dict(zip(tx_interet_int['an_eval'], tx_interet_int['TX_ACTU_INT']))

        return h_mortality, g_lapse, z_rendement, a_discount_ext, b_discount_int

    def _project_cash_flows(
            self,
            population_row: pd.Series,
            scenario_type: str,
            scn_proj: int,
            type2: Optional[str] = None,
            an_eval: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Performs a single cash flow projection for a given account and scenario.
        This is a faithful implementation of the core SAS calculation logic.
        """
        results = []
        is_external = scenario_type == "EXTERNE"

        # Determine projection horizon
        if is_external:
            max_years = min(NB_AN_PROJECTION, 99 - int(population_row['age_deb']))
        else:  # INTERNE
            max_years = min(NB_AN_PROJECTION_INT, 99 - int(population_row['age_deb']) - an_eval)

        # Year 0 Initialization
        if is_external:
            AGE = int(population_row['age_deb'])
            MT_VM_PROJ = float(population_row['MT_VM'])
            MT_GAR_DECES_PROJ = float(population_row['MT_GAR_DECES'])
            TX_SURVIE = 1.0
            COMMISSIONS = -float(population_row['TX_COMM_VENTE']) * MT_VM_PROJ
            FRAIS_GEN = -float(population_row['FRAIS_ACQUI'])
        else:  # INTERNE
            if type2 == "CAPITAL":
                MT_VM_PROJ = float(population_row['MT_VM']) * (1 - CHOC_CAPITAL)
            else:  # RESERVE
                MT_VM_PROJ = float(population_row['MT_VM'])
            AGE = int(population_row['age_deb']) + an_eval
            MT_GAR_DECES_PROJ = float(population_row['MT_GAR_DECES'])
            TX_SURVIE = float(population_row.get('TX_SURVIE_DEB', 1.0))
            COMMISSIONS, FRAIS_GEN = 0.0, 0.0

        # Common Year 0 initialization
        year_results = {
            'an_proj': 0 if is_external else an_eval, 'AGE': AGE, 'MT_VM_PROJ': MT_VM_PROJ,
            'MT_GAR_DECES_PROJ': MT_GAR_DECES_PROJ, 'TX_SURVIE': TX_SURVIE,
            'TX_SURVIE_DEB': TX_SURVIE, 'REVENUS': 0.0, 'FRAIS_GEST': 0.0,
            'COMMISSIONS': COMMISSIONS, 'FRAIS_GEN': FRAIS_GEN, 'PMT_GARANTIE': 0.0,
        }
        year_results['FLUX_NET'] = year_results['FRAIS_GEN'] + year_results['COMMISSIONS']
        for key in ['REVENUS', 'FRAIS_GEST', 'COMMISSIONS', 'FRAIS_GEN', 'PMT_GARANTIE', 'FLUX_NET']:
            year_results[f'VP_{key}'] = year_results[key]
        results.append(year_results)

        # Projection for subsequent years
        for year_idx in range(1, max_years + 1):
            if TX_SURVIE == 0 or MT_VM_PROJ == 0:
                break

            current_year = year_idx

            # State variables for start of year
            MT_VM_DEB = MT_VM_PROJ
            TX_SURVIE_DEB = TX_SURVIE

            # Determine projection year and age
            if is_external:
                an_proj = current_year
                AGE = int(population_row['age_deb']) + current_year
            else:
                an_proj = an_eval + current_year
                AGE = int(population_row['age_deb']) + an_eval + current_year

            # Project Fund Value
            rendement_rate = self.z_rendement.get((scn_proj, an_proj, scenario_type), 0.0)
            rendement_amt = MT_VM_DEB * rendement_rate
            frais_sur_fds = -(MT_VM_DEB + rendement_amt / 2) * float(population_row['PC_REVENU_FDS'])
            MT_VM_PROJ += rendement_amt + frais_sur_fds

            # Reset Death Benefit Guarantee
            if population_row['FREQ_RESET_DECES'] == 1 and AGE <= population_row['MAX_RESET_DECES']:
                MT_GAR_DECES_PROJ = max(MT_GAR_DECES_PROJ, MT_VM_PROJ)

            # Update Survival Probability
            qx = self.h_mortality.get(AGE, 0.0)
            wx = self.g_lapse.get(an_proj, 0.0)
            TX_SURVIE = TX_SURVIE_DEB * (1 - qx) * (1 - wx)

            # Calculate Cash Flows
            REVENUS = -frais_sur_fds * TX_SURVIE_DEB
            FRAIS_GEST = -(MT_VM_DEB + rendement_amt / 2) * float(population_row['PC_HONORAIRES_GEST']) * TX_SURVIE_DEB
            COMMISSIONS = -(MT_VM_DEB + rendement_amt / 2) * float(population_row['TX_COMM_MAINTIEN']) * TX_SURVIE_DEB
            FRAIS_GEN = -float(population_row['FRAIS_ADMIN']) * TX_SURVIE_DEB
            PMT_GARANTIE = -max(0, MT_GAR_DECES_PROJ - MT_VM_PROJ) * qx * TX_SURVIE_DEB
            FLUX_NET = REVENUS + FRAIS_GEST + COMMISSIONS + FRAIS_GEN + PMT_GARANTIE

            # Calculate Present Values
            tx_actu_ext = self.a_discount_ext.get(an_proj, 1.0)

            year_results = {
                'an_proj': an_proj, 'AGE': AGE, 'MT_VM_PROJ': MT_VM_PROJ,
                'MT_GAR_DECES_PROJ': MT_GAR_DECES_PROJ, 'TX_SURVIE': TX_SURVIE,
                'TX_SURVIE_DEB': TX_SURVIE_DEB, 'REVENUS': REVENUS, 'FRAIS_GEST': FRAIS_GEST,
                'COMMISSIONS': COMMISSIONS, 'FRAIS_GEN': FRAIS_GEN, 'PMT_GARANTIE': PMT_GARANTIE,
                'FLUX_NET': FLUX_NET,
                'VP_REVENUS': REVENUS * tx_actu_ext, 'VP_FRAIS_GEST': FRAIS_GEST * tx_actu_ext,
                'VP_COMMISSIONS': COMMISSIONS * tx_actu_ext, 'VP_FRAIS_GEN': FRAIS_GEN * tx_actu_ext,
                'VP_PMT_GARANTIE': PMT_GARANTIE * tx_actu_ext, 'VP_FLUX_NET': FLUX_NET * tx_actu_ext
            }

            # Adjust for internal scenarios
            if not is_external and an_eval > 0:
                tx_actu_int = self.b_discount_int.get(an_eval, 1.0)
                if tx_actu_int != 0:
                    for k in year_results:
                        if k.startswith('VP_'):
                            year_results[k] /= tx_actu_int

            results.append(year_results)

        df = pd.DataFrame(results)
        # Add identifying columns
        if not df.empty:
            df['ID_COMPTE'] = int(population_row['ID_COMPTE'])
            df['TYPE'] = scenario_type
            df['TYPE2'] = type2
            if is_external:
                df['scn_eval'] = scn_proj
                df['an_eval'] = df['an_proj']
            else:
                df['scn_eval_int'] = scn_proj
                df['an_eval'] = an_eval
                df['an_eval_int'] = df['an_proj'] - an_eval

        return df

    def _calculate_internal_metrics(self, external_results: pd.DataFrame, account_row: pd.Series,
                                    scn_eval: int) -> pd.DataFrame:
        """Runs all internal scenarios for each year of an external projection."""
        internal_summaries = []

        for _, ext_row in external_results.iterrows():
            an_eval = int(ext_row['an_eval'])
            if an_eval == 0:
                continue

            for type2 in ["RESERVE", "CAPITAL"]:
                vp_flux_net_means = []
                for scn_eval_int in range(1, NB_SC_INT + 1):
                    # Prepare input row with state from external projection
                    internal_input_row = account_row.copy()
                    internal_input_row['MT_VM'] = ext_row['MT_VM_PROJ']
                    internal_input_row['MT_GAR_DECES'] = ext_row['MT_GAR_DECES_PROJ']
                    internal_input_row['TX_SURVIE_DEB'] = ext_row['TX_SURVIE']

                    internal_results = self._project_cash_flows(
                        internal_input_row,
                        scenario_type="INTERNE",
                        scn_proj=scn_eval_int,
                        type2=type2,
                        an_eval=an_eval,
                    )
                    if not internal_results.empty:
                        vp_flux_net_means.append(internal_results['VP_FLUX_NET'].sum())

                if vp_flux_net_means:
                    mean_vp = np.mean(vp_flux_net_means)
                    internal_summaries.append({
                        'an_eval': an_eval,
                        'TYPE2': type2,
                        'METRIC': mean_vp,
                    })

        if not internal_summaries:
            return pd.DataFrame(columns=['an_eval', 'RESERVE', 'CAPITAL'])

        # Pivot the results
        summary_df = pd.DataFrame(internal_summaries)
        pivot_df = summary_df.pivot(index='an_eval', columns='TYPE2', values='METRIC').reset_index()
        pivot_df = pivot_df.rename(columns={'RESERVE': 'RESERVE', 'CAPITAL': 'CAPITAL_RAW'})
        pivot_df['CAPITAL'] = pivot_df.get('CAPITAL_RAW', 0) - pivot_df.get('RESERVE', 0)

        return pivot_df[['an_eval', 'RESERVE', 'CAPITAL']].fillna(0)

    @staticmethod
    def _calculate_distributable_flows(df: pd.DataFrame) -> pd.DataFrame:
        """Calculates profit and distributable flows."""
        df = df.sort_values('an_eval').reset_index(drop=True)
        df['reserve_prec'] = df['RESERVE'].shift(1, fill_value=0)
        df['capital_prec'] = df['CAPITAL'].shift(1, fill_value=0)

        # Vectorized calculation instead of row-by-row
        profit = np.where(
            df['an_eval'] == 0,
            df['FLUX_NET'] + df['RESERVE'],
            df['FLUX_NET'] + df['RESERVE'] - df['reserve_prec']
        )

        flux_dist = np.where(
            df['an_eval'] == 0,
            profit + df['CAPITAL'],
            profit + df['CAPITAL'] - df['capital_prec']
        )

        df['PROFIT'] = profit
        df['FLUX_DISTRIBUABLES'] = flux_dist

        df['VP_FLUX_DISTRIBUABLES'] = df['FLUX_DISTRIBUABLES'] / ((1 + HURDLE_RT) ** df['an_eval'])

        return df

    def _process_single_account(self, account_row: pd.Series) -> List[Dict]:
        """Processes all scenarios for a single account."""
        account_results = []
        id_compte = int(account_row['ID_COMPTE'])

        for scn_eval in range(1, NB_SC + 1):
            # 1. Run external projection
            external_results = self._project_cash_flows(
                account_row,
                scenario_type="EXTERNE",
                scn_proj=scn_eval
            )
            if external_results.empty:
                continue

            # 2. Run internal projections to calculate RESERVE and CAPITAL
            internal_metrics = self._calculate_internal_metrics(external_results, account_row, scn_eval)

            # 3. Merge metrics back to external results
            # Using pd.merge is much faster and cleaner than a loop
            enhanced_df = pd.merge(external_results, internal_metrics, on='an_eval', how='left').fillna(0)

            # 4. Calculate final distributable flows
            final_df = self._calculate_distributable_flows(enhanced_df)

            # 5. Summarize results for this account-scenario
            total_vp_distribuables = final_df['VP_FLUX_DISTRIBUABLES'].sum()
            account_results.append({
                'ID_COMPTE': id_compte,
                'scn_eval': scn_eval,
                'VP_FLUX_DISTRIBUABLES': total_vp_distribuables,
            })

        return account_results

    def run_projections(self) -> pd.DataFrame:
        """
        Runs the full projection for all accounts and scenarios.
        This is the main entry point for the calculation.
        """
        logger.info("Starting projection calculations...")
        logger.info(
            f"Configuration: Accounts={len(self.population)}, External Scenarios={NB_SC}, Internal Scenarios={NB_SC_INT}")

        all_results = []

        # --- Main Loop: Account -> Scenario -> Year ---
        # This structure is maintained as it's the most logical and efficient for this simulation.
        # The complexity is managed by breaking the process into smaller methods.
        for _, account_row in tqdm(self.population.iterrows(), total=len(self.population), desc="Processing Accounts"):
            account_summary = self._process_single_account(account_row)
            all_results.extend(account_summary)

        logger.info(f"Calculations complete. Aggregating {len(all_results)} results.")
        return pd.DataFrame(all_results)


def main(data_path: Path, output_dir: Path):
    """Main execution function."""
    start_time = time.time()

    logger.info("=" * 80)
    logger.info("Refactored Actuarial Cash Flow Projection Model")
    logger.info("=" * 80)

    try:
        # Initialize and run the model
        model = ActuarialModel(data_path=data_path)
        results_df = model.run_projections()

        execution_time = time.time() - start_time

        # --- Analysis and Output ---
        if results_df.empty:
            logger.warning("No results were generated.")
            return

        print(f"\n" + "=" * 60)
        print(f"ACFC PROJECTION RESULTS")
        print(f"=" * 60)
        print(f"Total combinations: {len(results_df):,}")
        print(f"Execution time: {execution_time:.2f} seconds")
        print(f"Mean VP_FLUX_DISTRIBUABLES: ${results_df['VP_FLUX_DISTRIBUABLES'].mean():,.2f}")
        print(f"Profitable combinations: {len(results_df[results_df['VP_FLUX_DISTRIBUABLES'] > 0]):,}")
        print(
            f"Range: ${results_df['VP_FLUX_DISTRIBUABLES'].min():,.2f} to ${results_df['VP_FLUX_DISTRIBUABLES'].max():,.2f}")

        # Save results
        output_dir.mkdir(exist_ok=True)
        results_file = output_dir / "acfc_results_refactored.csv"
        results_df.to_csv(results_file, index=False)
        logger.info(f"Results saved to {results_file}")

    except Exception as e:
        logger.error(f"An error occurred during model execution: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    DATA_PATH = HERE.joinpath("data_in")
    OUTPUT_DIR = HERE.joinpath("test")
    main(data_path=DATA_PATH, output_dir=OUTPUT_DIR)
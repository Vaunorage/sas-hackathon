import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional, NamedTuple
import logging
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
import warnings
from enum import Enum
from paths import HERE

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScenarioType(Enum):
    """Enumeration for scenario types"""
    EXTERNE = "EXTERNE"
    INTERNE = "INTERNE"


class InternalType(Enum):
    """Enumeration for internal calculation types"""
    RESERVE = "RESERVE"
    CAPITAL = "CAPITAL"


@dataclass
class Config:
    """Configuration parameters for ACFC calculations"""
    nbcpt: int = 100
    nb_sc: int = 100
    nb_an_projection: int = 100
    nb_sc_int: int = 100
    nb_an_projection_int: int = 100
    choc_capital: float = 0.35
    hurdle_rt: float = 0.10

    # Performance limits for testing
    max_external_scenarios: int = 20
    max_internal_scenarios: int = 10


@dataclass
class HashTables:
    """Container for lookup tables"""
    mortality: Dict[int, float]
    lapse: Dict[int, float]
    rendement: Dict[Tuple[int, int, str], float]
    discount_ext: Dict[int, float]
    discount_int: Dict[int, float]


class CashFlowRow(NamedTuple):
    """Type-safe container for cash flow calculation results"""
    id_compte: int
    an_eval: int
    scn_eval: int
    scn_eval_int: Optional[int]
    an_eval_int: Optional[int]
    scenario_type: str
    type2: Optional[str]
    an_proj: int
    age: int
    mt_vm_proj: float
    mt_gar_deces_proj: float
    tx_survie: float
    tx_survie_deb: float
    revenus: float
    frais_gest: float
    commissions: float
    frais_gen: float
    pmt_garantie: float
    flux_net: float
    vp_revenus: float
    vp_frais_gest: float
    vp_commissions: float
    vp_frais_gen: float
    vp_pmt_garantie: float
    vp_flux_net: float


class DataLoader:
    """Handles loading and validation of input data files"""

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)

    def load_input_files(self) -> Tuple[pd.DataFrame, ...]:
        """Load all input CSV files with validation"""
        try:
            file_configs = {
                'population': {},
                'rendement': {},
                'tx_deces': {},
                'tx_interet': {},
                'tx_interet_int': {},
                'tx_retrait': {}
            }

            loaded_files = {}

            for file_name, config in file_configs.items():
                file_path = self.data_path / f"{file_name}.csv"
                df = pd.read_csv(file_path)

                if 'head' in config:
                    df = df.head(config['head'])

                # Handle encoding for TYPE column
                if 'TYPE' in df.columns:
                    df['TYPE'] = df['TYPE'].apply(
                        lambda x: x.decode('utf-8') if isinstance(x, bytes) else str(x)
                    )

                loaded_files[file_name] = df
                logger.info(f"Loaded {file_name}: {len(df)} rows")

            return tuple(loaded_files.values())

        except Exception as e:
            logger.error(f"Error loading input files: {e}")
            raise


class HashTableBuilder:
    """Builds lookup tables for efficient data access"""

    @staticmethod
    def create_hash_tables(
            rendement: pd.DataFrame,
            tx_deces: pd.DataFrame,
            tx_interet: pd.DataFrame,
            tx_interet_int: pd.DataFrame,
            tx_retrait: pd.DataFrame
    ) -> HashTables:
        """Create all hash tables for lookups"""

        return HashTables(
            mortality=HashTableBuilder._build_mortality_hash(tx_deces),
            lapse=HashTableBuilder._build_lapse_hash(tx_retrait),
            rendement=HashTableBuilder._build_rendement_hash(rendement),
            discount_ext=HashTableBuilder._build_discount_ext_hash(tx_interet),
            discount_int=HashTableBuilder._build_discount_int_hash(tx_interet_int)
        )

    @staticmethod
    def _build_mortality_hash(tx_deces: pd.DataFrame) -> Dict[int, float]:
        """Build mortality lookup: AGE -> Qx"""
        return {int(row['AGE']): float(row['QX']) for _, row in tx_deces.iterrows()}

    @staticmethod
    def _build_lapse_hash(tx_retrait: pd.DataFrame) -> Dict[int, float]:
        """Build lapse lookup: an_proj -> WX"""
        return {int(row['an_proj']): float(row['WX']) for _, row in tx_retrait.iterrows()}

    @staticmethod
    def _build_rendement_hash(rendement: pd.DataFrame) -> Dict[Tuple[int, int, str], float]:
        """Build return lookup: (scn_proj, an_proj, TYPE) -> RENDEMENT"""
        return {
            (int(row['scn_proj']), int(row['an_proj']), str(row['TYPE'])): float(row['RENDEMENT'])
            for _, row in rendement.iterrows()
        }

    @staticmethod
    def _build_discount_ext_hash(tx_interet: pd.DataFrame) -> Dict[int, float]:
        """Build external discount lookup: an_proj -> TX_ACTU"""
        return {int(row['an_proj']): float(row['TX_ACTU']) for _, row in tx_interet.iterrows()}

    @staticmethod
    def _build_discount_int_hash(tx_interet_int: pd.DataFrame) -> Dict[int, float]:
        """Build internal discount lookup: an_eval -> TX_ACTU_INT"""
        return {int(row['an_eval']): float(row['TX_ACTU_INT']) for _, row in tx_interet_int.iterrows()}


class CashFlowCalculator:
    """Handles cash flow calculations for individual scenarios"""

    def __init__(self, config: Config, hash_tables: HashTables):
        self.config = config
        self.hash_tables = hash_tables

    def hash_find(self, hash_table: dict, key, default_value: float = 0.0) -> float:
        """Safe hash table lookup with default value"""
        return hash_table.get(key, default_value)

    def calculate_cash_flows(
            self,
            population_row: pd.Series,
            scenario_type: ScenarioType,
            type2: Optional[InternalType] = None,
            scn_eval: Optional[int] = None,
            an_eval: Optional[int] = None,
            scn_eval_int: Optional[int] = None,
            an_eval_int: Optional[int] = None
    ) -> pd.DataFrame:
        """Calculate cash flows for a given scenario"""

        # Initialize state variables
        state = self._initialize_state(population_row, scenario_type, type2, an_eval)

        # Determine projection range
        year_range = self._get_projection_range(population_row, scenario_type, an_eval)

        results = []

        for current_year in year_range:
            if current_year == 0:
                cash_flows = self._calculate_year_zero_flows(
                    population_row, scenario_type, type2, state, scn_eval, an_eval
                )
            else:
                # Check termination conditions
                if state['TX_SURVIE'] == 0 or state['MT_VM_PROJ'] == 0:
                    continue

                cash_flows = self._calculate_projection_flows(
                    population_row, scenario_type, state, current_year,
                    scn_eval, an_eval, scn_eval_int
                )

            # Create result row
            result_row = self._create_result_row(
                population_row, cash_flows, state, scenario_type, type2,
                current_year, scn_eval, an_eval, scn_eval_int, an_eval_int
            )

            results.append(result_row)

        return pd.DataFrame([row._asdict() for row in results])

    def _initialize_state(
            self,
            population_row: pd.Series,
            scenario_type: ScenarioType,
            type2: Optional[InternalType],
            an_eval: Optional[int]
    ) -> Dict:
        """Initialize calculation state variables"""

        if scenario_type == ScenarioType.INTERNE and type2 == InternalType.CAPITAL:
            mt_vm_initial = float(population_row['MT_VM']) * (1 - self.config.choc_capital)
        else:
            mt_vm_initial = float(population_row['MT_VM'])

        return {
            'MT_VM_PROJ': mt_vm_initial,
            'MT_GAR_DECES_PROJ': float(population_row['MT_GAR_DECES']),
            'TX_SURVIE': float(population_row.get('TX_SURVIE_DEB', 1.0))
        }

    def _get_projection_range(
            self,
            population_row: pd.Series,
            scenario_type: ScenarioType,
            an_eval: Optional[int]
    ) -> range:
        """Determine the range of years for projection"""

        age_start = int(population_row['age_deb'])

        if scenario_type == ScenarioType.EXTERNE:
            max_years = min(self.config.nb_an_projection, 99 - age_start)
        else:  # INTERNE
            max_years = min(
                self.config.nb_an_projection_int,
                99 - age_start - (an_eval or 0)
            )

        return range(max_years + 1)

    def _calculate_year_zero_flows(
            self,
            population_row: pd.Series,
            scenario_type: ScenarioType,
            type2: Optional[InternalType],
            state: Dict,
            scn_eval: Optional[int],
            an_eval: Optional[int]
    ) -> Dict:
        """Calculate cash flows for year 0"""

        if scenario_type == ScenarioType.EXTERNE:
            return {
                'COMMISSIONS': -float(population_row['TX_COMM_VENTE']) * state['MT_VM_PROJ'],
                'FRAIS_GEN': -float(population_row['FRAIS_ACQUI']),
                'REVENUS': 0.0,
                'FRAIS_GEST': 0.0,
                'PMT_GARANTIE': 0.0,
                'QX': 0.0,
                'WX': 0.0,
                'TX_ACTU': 1.0,
                'AGE': int(population_row['age_deb']),
                'an_proj': 0
            }
        else:
            # Internal scenario year 0 - all flows are zero
            return {
                'COMMISSIONS': 0.0,
                'FRAIS_GEN': 0.0,
                'REVENUS': 0.0,
                'FRAIS_GEST': 0.0,
                'PMT_GARANTIE': 0.0,
                'QX': 0.0,
                'WX': 0.0,
                'TX_ACTU': 1.0,
                'AGE': int(population_row['age_deb']) + (an_eval or 0),
                'an_proj': an_eval or 0
            }

    def _calculate_projection_flows(
            self,
            population_row: pd.Series,
            scenario_type: ScenarioType,
            state: Dict,
            current_year: int,
            scn_eval: Optional[int],
            an_eval: Optional[int],
            scn_eval_int: Optional[int]
    ) -> Dict:
        """Calculate cash flows for projection years > 0"""

        # Determine scenario and year parameters
        if scenario_type == ScenarioType.INTERNE:
            scn_proj = scn_eval_int
            age = int(population_row['age_deb']) + (an_eval or 0) + current_year
            an_proj = (an_eval or 0) + current_year
        else:
            scn_proj = scn_eval
            age = int(population_row['age_deb']) + current_year
            an_proj = current_year

        # Fund value projection
        mt_vm_deb = state['MT_VM_PROJ']

        rendement_rate = self.hash_find(
            self.hash_tables.rendement,
            (scn_proj, an_proj, scenario_type.value),
            0.0
        )
        rendement = mt_vm_deb * rendement_rate

        frais = -(mt_vm_deb + rendement / 2) * float(population_row['PC_REVENU_FDS'])
        state['MT_VM_PROJ'] = state['MT_VM_PROJ'] + rendement + frais

        # Death benefit guarantee reset
        self._update_death_guarantee(population_row, state, age)

        # Survival probabilities
        qx = self.hash_find(self.hash_tables.mortality, age, 0.0)
        wx = self.hash_find(self.hash_tables.lapse, an_proj, 0.0)

        tx_survie_deb = state['TX_SURVIE']
        state['TX_SURVIE'] = tx_survie_deb * (1 - qx) * (1 - wx)

        # Cash flow calculations
        revenus = -frais * tx_survie_deb
        frais_gest = -(mt_vm_deb + rendement / 2) * float(population_row['PC_HONORAIRES_GEST']) * tx_survie_deb
        commissions = -(mt_vm_deb + rendement / 2) * float(population_row['TX_COMM_MAINTIEN']) * tx_survie_deb
        frais_gen = -float(population_row['FRAIS_ADMIN']) * tx_survie_deb
        pmt_garantie = -max(0, state['MT_GAR_DECES_PROJ'] - state['MT_VM_PROJ']) * qx * tx_survie_deb

        # Discount factor
        tx_actu = self.hash_find(self.hash_tables.discount_ext, an_proj, 1.0)

        return {
            'REVENUS': revenus,
            'FRAIS_GEST': frais_gest,
            'COMMISSIONS': commissions,
            'FRAIS_GEN': frais_gen,
            'PMT_GARANTIE': pmt_garantie,
            'QX': qx,
            'WX': wx,
            'TX_ACTU': tx_actu,
            'TX_SURVIE_DEB': tx_survie_deb,
            'AGE': age,
            'an_proj': an_proj
        }

    def _update_death_guarantee(self, population_row: pd.Series, state: Dict, age: int):
        """Update death benefit guarantee based on reset frequency"""
        freq_reset = float(population_row['FREQ_RESET_DECES'])
        max_reset = float(population_row['MAX_RESET_DECES'])

        if freq_reset == 1 and age <= max_reset:
            state['MT_GAR_DECES_PROJ'] = max(state['MT_GAR_DECES_PROJ'], state['MT_VM_PROJ'])

    def _create_result_row(
            self,
            population_row: pd.Series,
            cash_flows: Dict,
            state: Dict,
            scenario_type: ScenarioType,
            type2: Optional[InternalType],
            current_year: int,
            scn_eval: Optional[int],
            an_eval: Optional[int],
            scn_eval_int: Optional[int],
            an_eval_int: Optional[int]
    ) -> CashFlowRow:
        """Create a typed result row"""

        # Calculate net flows
        flux_net = sum([
            cash_flows['REVENUS'],
            cash_flows['FRAIS_GEST'],
            cash_flows['COMMISSIONS'],
            cash_flows['FRAIS_GEN'],
            cash_flows['PMT_GARANTIE']
        ])

        # Calculate present values
        tx_actu = cash_flows['TX_ACTU']

        vp_revenus = cash_flows['REVENUS'] * tx_actu
        vp_frais_gest = cash_flows['FRAIS_GEST'] * tx_actu
        vp_commissions = cash_flows['COMMISSIONS'] * tx_actu
        vp_frais_gen = cash_flows['FRAIS_GEN'] * tx_actu
        vp_pmt_garantie = cash_flows['PMT_GARANTIE'] * tx_actu
        vp_flux_net = flux_net * tx_actu

        # Internal scenario adjustment
        if scenario_type == ScenarioType.INTERNE and an_eval and an_eval > 0:
            tx_actu_int = self.hash_find(self.hash_tables.discount_int, an_eval, 1.0)
            if tx_actu_int != 0:
                adjustment_factor = 1 / tx_actu_int
                vp_revenus *= adjustment_factor
                vp_frais_gest *= adjustment_factor
                vp_commissions *= adjustment_factor
                vp_frais_gen *= adjustment_factor
                vp_pmt_garantie *= adjustment_factor
                vp_flux_net *= adjustment_factor

        return CashFlowRow(
            id_compte=int(population_row['ID_COMPTE']),
            an_eval=an_eval if scenario_type == ScenarioType.INTERNE else current_year,
            scn_eval=scn_eval or 0,
            scn_eval_int=scn_eval_int if scenario_type == ScenarioType.INTERNE else None,
            an_eval_int=current_year if scenario_type == ScenarioType.INTERNE else None,
            scenario_type=scenario_type.value,
            type2=type2.value if type2 else None,
            an_proj=cash_flows['an_proj'],
            age=cash_flows['AGE'],
            mt_vm_proj=state['MT_VM_PROJ'],
            mt_gar_deces_proj=state['MT_GAR_DECES_PROJ'],
            tx_survie=state['TX_SURVIE'],
            tx_survie_deb=cash_flows.get('TX_SURVIE_DEB', state['TX_SURVIE']),
            revenus=cash_flows['REVENUS'],
            frais_gest=cash_flows['FRAIS_GEST'],
            commissions=cash_flows['COMMISSIONS'],
            frais_gen=cash_flows['FRAIS_GEN'],
            pmt_garantie=cash_flows['PMT_GARANTIE'],
            flux_net=flux_net,
            vp_revenus=vp_revenus,
            vp_frais_gest=vp_frais_gest,
            vp_commissions=vp_commissions,
            vp_frais_gen=vp_frais_gen,
            vp_pmt_garantie=vp_pmt_garantie,
            vp_flux_net=vp_flux_net
        )


class ACFCEngine:
    """Main engine for running ACFC calculations"""

    def __init__(self, config: Config):
        self.config = config
        self.calculator = None

    def run_calculations(
            self,
            population: pd.DataFrame,
            hash_tables: HashTables
    ) -> pd.DataFrame:
        """Run the complete nested calculation process"""

        self.calculator = CashFlowCalculator(self.config, hash_tables)
        calculs_sommaire = pd.DataFrame(columns=['ID_COMPTE', 'scn_eval', 'VP_FLUX_DISTRIBUABLES'])

        logger.info("Starting ACFC nested loop calculations...")

        # Account loop
        account_limit = min(self.config.nbcpt, len(population))
        scenario_limit = min(self.config.nb_sc, self.config.max_external_scenarios)

        for j in tqdm(range(1, account_limit + 1), desc="Accounts"):
            account_data = population[population['ID_COMPTE'] == j]
            if account_data.empty:
                continue

            account_row = account_data.iloc[0]

            # External scenarios loop
            for scn_eval in range(1, scenario_limit + 1):
                total_vp_distribuables = self._calculate_account_scenario(
                    account_row, j, scn_eval
                )

                # Add to summary
                summary_row = pd.DataFrame([{
                    'ID_COMPTE': j,
                    'scn_eval': scn_eval,
                    'VP_FLUX_DISTRIBUABLES': total_vp_distribuables
                }])

                calculs_sommaire = pd.concat([calculs_sommaire, summary_row], ignore_index=True)

        logger.info(f"Calculations complete. Results: {len(calculs_sommaire)} combinations")
        return calculs_sommaire

    def _calculate_account_scenario(
            self,
            account_row: pd.Series,
            account_id: int,
            scn_eval: int
    ) -> float:
        """Calculate distributable cash flows for one account-scenario combination"""

        # Calculate external cash flows
        external_results = self.calculator.calculate_cash_flows(
            account_row,
            ScenarioType.EXTERNE,
            scn_eval=scn_eval
        )

        if external_results.empty:
            return 0.0

        # Calculate internal scenarios for each external year
        reserve_results = []
        capital_results = []

        for _, ext_row in external_results.iterrows():
            an_eval = int(ext_row['an_eval'])

            if an_eval == 0:
                continue

            # Calculate for both RESERVE and CAPITAL types
            for type2 in [InternalType.RESERVE, InternalType.CAPITAL]:
                internal_scenarios_sum = []

                # Internal scenarios loop
                int_scenario_limit = min(self.config.nb_sc_int, self.config.max_internal_scenarios)

                for scn_eval_int in range(1, int_scenario_limit + 1):
                    internal_input_row = self._prepare_internal_input(account_row, ext_row)

                    internal_results = self.calculator.calculate_cash_flows(
                        internal_input_row,
                        ScenarioType.INTERNE,
                        type2=type2,
                        scn_eval=scn_eval,
                        an_eval=an_eval,
                        scn_eval_int=scn_eval_int
                    )

                    if not internal_results.empty:
                        total_vp = internal_results['vp_flux_net'].sum()
                        internal_scenarios_sum.append(total_vp)

                # Calculate mean across internal scenarios
                if internal_scenarios_sum:
                    mean_vp = np.mean(internal_scenarios_sum)
                    result_entry = {
                        'ID_COMPTE': account_id,
                        'an_eval': an_eval,
                        'scn_eval': scn_eval,
                        'VP_FLUX_NET': mean_vp,
                        'TYPE2': type2.value
                    }

                    if type2 == InternalType.RESERVE:
                        reserve_results.append(result_entry)
                    else:
                        capital_results.append(result_entry)

        # Merge results and calculate distributable flows
        enhanced_external = self._merge_and_calculate_distributable(
            external_results, reserve_results, capital_results, account_id, scn_eval
        )

        return enhanced_external['VP_FLUX_DISTRIBUABLES'].sum()

    def _prepare_internal_input(self, account_row: pd.Series, ext_row: pd.Series) -> pd.Series:
        """Prepare input row for internal calculation"""
        internal_input_row = account_row.copy()
        internal_input_row['MT_VM'] = ext_row['mt_vm_proj']
        internal_input_row['MT_GAR_DECES'] = ext_row['mt_gar_deces_proj']
        internal_input_row['TX_SURVIE_DEB'] = ext_row['tx_survie']
        return internal_input_row

    def _merge_and_calculate_distributable(
            self,
            external_results: pd.DataFrame,
            reserve_results: List[Dict],
            capital_results: List[Dict],
            account_id: int,
            scn_eval: int
    ) -> pd.DataFrame:
        """Merge internal results with external and calculate distributable flows"""

        # Convert to DataFrames
        reserve_df = pd.DataFrame(reserve_results)
        capital_df = pd.DataFrame(capital_results)

        # Initialize enhanced external results
        enhanced_external = external_results.copy()
        enhanced_external['RESERVE'] = 0.0
        enhanced_external['CAPITAL'] = 0.0

        # Merge reserve and capital values
        for idx, row in enhanced_external.iterrows():
            an_eval = int(row['an_eval'])

            # Find matching reserve
            reserve_match = reserve_df[
                (reserve_df['ID_COMPTE'] == account_id) &
                (reserve_df['an_eval'] == an_eval) &
                (reserve_df['scn_eval'] == scn_eval)
                ]

            if not reserve_match.empty:
                enhanced_external.loc[idx, 'RESERVE'] = reserve_match.iloc[0]['VP_FLUX_NET']

            # Find matching capital
            capital_match = capital_df[
                (capital_df['ID_COMPTE'] == account_id) &
                (capital_df['an_eval'] == an_eval) &
                (capital_df['scn_eval'] == scn_eval)
                ]

            if not capital_match.empty:
                capital_value = capital_match.iloc[0]['VP_FLUX_NET'] - enhanced_external.loc[idx, 'RESERVE']
                enhanced_external.loc[idx, 'CAPITAL'] = capital_value

        # Calculate profit and distributable flows
        return self._calculate_profit_and_distributable(enhanced_external)

    def _calculate_profit_and_distributable(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate profit and distributable cash flows"""

        df = df.sort_values('an_eval').reset_index(drop=True)
        df['reserve_prec'] = df['RESERVE'].shift(1, fill_value=0)
        df['capital_prec'] = df['CAPITAL'].shift(1, fill_value=0)

        # Calculate profit
        year_0_mask = df['an_eval'] == 0

        df.loc[year_0_mask, 'PROFIT'] = df.loc[year_0_mask, 'flux_net'] + df.loc[year_0_mask, 'RESERVE']
        df.loc[~year_0_mask, 'PROFIT'] = (
                df.loc[~year_0_mask, 'flux_net'] +
                df.loc[~year_0_mask, 'RESERVE'] -
                df.loc[~year_0_mask, 'reserve_prec']
        )

        # Calculate distributable flows
        df.loc[year_0_mask, 'FLUX_DISTRIBUABLES'] = (
                df.loc[year_0_mask, 'PROFIT'] + df.loc[year_0_mask, 'CAPITAL']
        )
        df.loc[~year_0_mask, 'FLUX_DISTRIBUABLES'] = (
                df.loc[~year_0_mask, 'PROFIT'] +
                df.loc[~year_0_mask, 'CAPITAL'] -
                df.loc[~year_0_mask, 'capital_prec']
        )

        # Present value of distributable flows
        df['VP_FLUX_DISTRIBUABLES'] = (
                df['FLUX_DISTRIBUABLES'] / ((1 + self.config.hurdle_rt) ** df['an_eval'])
        )

        return df


def run_acfc(
        data_path: str = "data_in",
        output_dir: str = "output",
        config: Optional[Config] = None
) -> pd.DataFrame:
    """Main function to run ACFC calculations"""

    if config is None:
        config = Config()

    logger.info("=" * 80)
    logger.info("REFACTORED ACFC IMPLEMENTATION")
    logger.info("=" * 80)

    try:
        # Load data
        loader = DataLoader(data_path)
        population, rendement, tx_deces, tx_interet, tx_interet_int, tx_retrait = loader.load_input_files()

        # Create hash tables
        hash_tables = HashTableBuilder.create_hash_tables(
            rendement, tx_deces, tx_interet, tx_interet_int, tx_retrait
        )

        # Log configuration
        logger.info(f"Configuration:")
        logger.info(f"  Accounts: {min(config.nbcpt, len(population))}")
        logger.info(f"  External scenarios: {min(config.nb_sc, config.max_external_scenarios)}")
        logger.info(f"  Internal scenarios: {min(config.nb_sc_int, config.max_internal_scenarios)}")
        logger.info(f"  Max projection years: {config.nb_an_projection}")

        # Run calculations
        engine = ACFCEngine(config)
        results_df = engine.run_calculations(population, hash_tables)

        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        results_file = output_path / "acfc_results.csv"
        results_df.to_csv(results_file, index=False)
        logger.info(f"Results saved to {results_file}")

        # Print summary
        print(f"\n" + "=" * 60)
        print(f"ACFC RESULTS SUMMARY")
        print(f"=" * 60)
        print(f"Total combinations: {len(results_df):,}")
        print(f"Mean VP_FLUX_DISTRIBUABLES: ${results_df['VP_FLUX_DISTRIBUABLES'].mean():,.2f}")
        print(f"Profitable combinations: {len(results_df[results_df['VP_FLUX_DISTRIBUABLES'] > 0]):,}")
        print(
            f"Range: ${results_df['VP_FLUX_DISTRIBUABLES'].min():,.2f} to ${results_df['VP_FLUX_DISTRIBUABLES'].max():,.2f}")

        return results_df

    except Exception as e:
        logger.error(f"Error in ACFC execution: {str(e)}")
        raise


def main():
    """Main execution function"""
    try:
        # You can customize the configuration here
        custom_config = Config(
            nbcpt=4,  # Use all 4 accounts from population.head(4)
            nb_sc=20,  # External scenarios
            nb_sc_int=10,  # Internal scenarios
            max_external_scenarios=5,  # Limit for testing
            max_internal_scenarios=5  # Limit for testing
        )

        # Assuming paths module exists, otherwise use relative paths
        data_path = HERE.joinpath("data_in")
        output_dir = HERE.joinpath("test")

        results_df = run_acfc(
            data_path=data_path,
            output_dir=output_dir,
            config=custom_config
        )

        return results_df

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise


if __name__ == "__main__":
    results_df = main()
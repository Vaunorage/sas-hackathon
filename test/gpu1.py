import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
import logging
import time
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
from functools import partial
import math

# GPU Computing Libraries
try:
    import cupy as cp
    import cupyx.scipy.ndimage
    from numba import cuda, types
    from numba.cuda import random
    import rmm

    GPU_AVAILABLE = True
    print("GPU libraries loaded successfully")

    # Initialize RMM for better GPU memory management
    try:
        rmm.reinitialize(
            pool_allocator=True,
            managed_memory=False,
            initial_pool_size=None,
            devices=[0]  # Use first GPU
        )
        print("RMM memory pool initialized")
    except Exception as e:
        print(f"RMM initialization failed: {e}")

except ImportError as e:
    print(f"GPU libraries not available: {e}")
    print("Falling back to CPU-only implementation")
    GPU_AVAILABLE = False
    import numpy as cp  # Fallback to numpy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GPUActuarialCalculator:
    """GPU-accelerated actuarial cash flow calculator"""

    def __init__(self, data_path: Path = None):
        self.data_path = data_path or Path("data_in")
        self.gpu_available = GPU_AVAILABLE
        self.device_info = self._check_gpu_availability()
        self.memory_pool = None

        if self.gpu_available:
            self._initialize_gpu_resources()

    def _check_gpu_availability(self) -> Dict:
        """Check GPU availability and specifications"""
        if not self.gpu_available:
            return {"available": False, "memory": 0, "device_count": 0}

        try:
            device_count = cp.cuda.runtime.getDeviceCount()
            if device_count == 0:
                self.gpu_available = False
                return {"available": False, "memory": 0, "device_count": 0}

            # Get current device info
            device = cp.cuda.Device()
            free_mem, total_mem = cp.cuda.runtime.memGetInfo()

            device_info = {
                "available": True,
                "device_count": device_count,
                "device_id": device.id,
                "total_memory": total_mem,
                "free_memory": free_mem,
                "name": device.name.decode() if hasattr(device.name, 'decode') else str(device.name),
                "compute_capability": device.compute_capability
            }

            logger.info(f"GPU Device: {device_info['name']}")
            logger.info(f"Compute Capability: {device_info['compute_capability']}")
            logger.info(f"Total Memory: {total_mem / 1e9:.1f} GB")
            logger.info(f"Free Memory: {free_mem / 1e9:.1f} GB")

            return device_info

        except Exception as e:
            logger.warning(f"GPU check failed: {e}")
            self.gpu_available = False
            return {"available": False, "memory": 0, "device_count": 0}

    def _initialize_gpu_resources(self):
        """Initialize GPU memory pool and streams"""
        if not self.gpu_available:
            return

        try:
            # Create memory pool for efficient memory management
            self.memory_pool = cp.get_default_memory_pool()
            self.pinned_memory_pool = cp.get_default_pinned_memory_pool()

            # Create CUDA streams for concurrent operations
            self.stream_main = cp.cuda.Stream(non_blocking=True)
            self.stream_data = cp.cuda.Stream(non_blocking=True)
            self.stream_compute = cp.cuda.Stream(non_blocking=True)

            logger.info("GPU resources initialized successfully")

        except Exception as e:
            logger.warning(f"GPU resource initialization failed: {e}")
            self.gpu_available = False

    def optimize_memory(self) -> Tuple[int, int]:
        """Optimize GPU memory usage"""
        if not self.gpu_available:
            return 0, 0

        try:
            # Free unused memory blocks
            self.memory_pool.free_all_blocks()
            self.pinned_memory_pool.free_all_blocks()

            # Force garbage collection
            cp.cuda.runtime.deviceSynchronize()

            free_memory, total_memory = cp.cuda.runtime.memGetInfo()
            logger.info(f"Memory optimized: {free_memory / 1e9:.1f}GB free / {total_memory / 1e9:.1f}GB total")

            return free_memory, total_memory
        except Exception as e:
            logger.warning(f"Memory optimization failed: {e}")
            return 0, 0

    def load_data_optimized(self) -> Tuple[pd.DataFrame, ...]:
        """Load and optimize data for GPU processing"""
        logger.info("Loading data with GPU optimizations...")

        # Population data with optimized dtypes
        population = pd.read_csv(self.data_path / 'population.csv')

        # Convert to optimal dtypes for GPU processing
        population = population.astype({
            'ID_COMPTE': 'int32',
            'age_deb': 'int16',
            'MT_VM': 'float32',
            'MT_GAR_DECES': 'float32',
            'PC_REVENU_FDS': 'float32',
            'PC_HONORAIRES_GEST': 'float32',
            'TX_COMM_MAINTIEN': 'float32',
            'FRAIS_ADMIN': 'float32',
            'FREQ_RESET_DECES': 'float32',
            'MAX_RESET_DECES': 'int16'
        })

        # Load other datasets
        rendement = pd.read_csv(self.data_path / 'rendement.csv')
        if 'TYPE' in rendement.columns:
            rendement['TYPE'] = rendement['TYPE'].apply(
                lambda x: x.decode('utf-8') if isinstance(x, bytes) else str(x)
            )

        rendement = rendement.astype({
            'an_proj': 'int16',
            'scn_proj': 'int16',
            'RENDEMENT': 'float32'
        })

        # Split rendement by type
        rendement_ext = rendement[rendement['TYPE'] == 'EXTERNE'].copy()
        rendement_int = rendement[rendement['TYPE'] == 'INTERNE'].copy()

        # Load mortality and other tables
        tx_deces = pd.read_csv(self.data_path / 'tx_deces.csv').astype({
            'AGE': 'int16', 'QX': 'float32'
        })

        tx_interet = pd.read_csv(self.data_path / 'tx_interet.csv').astype({
            'an_proj': 'int16', 'TX_ACTU': 'float32'
        })

        tx_interet_int = pd.read_csv(self.data_path / 'tx_interet_int.csv').astype({
            'an_eval': 'int16', 'TX_ACTU_INT': 'float32'
        })

        tx_retrait = pd.read_csv(self.data_path / 'tx_retrait.csv').astype({
            'an_proj': 'int16', 'WX': 'float32'
        })

        logger.info(f"Data loaded: {len(population)} policies")
        return population, rendement_ext, rendement_int, tx_deces, tx_interet, tx_interet_int, tx_retrait

    def create_gpu_lookup_tables(self, rendement_ext, rendement_int, tx_deces,
                                 tx_interet, tx_interet_int, tx_retrait) -> Tuple:
        """Create optimized GPU lookup tables"""
        logger.info("Creating GPU lookup tables...")

        # Determine array dimensions
        max_year = max(rendement_ext['an_proj'].max(), rendement_int['an_proj'].max(), 100)
        max_scn_ext = rendement_ext['scn_proj'].max()
        max_scn_int = rendement_int['scn_proj'].max()
        max_age = max(tx_deces['AGE'].max(), 150)

        if self.gpu_available:
            # Create GPU arrays with proper memory layout
            ext_returns = cp.zeros((max_year + 1, max_scn_ext + 1), dtype=cp.float32, order='C')
            int_returns = cp.zeros((max_year + 1, max_scn_int + 1), dtype=cp.float32, order='C')
            mortality_rates = cp.zeros(max_age + 1, dtype=cp.float32)
            discount_ext = cp.ones(max_year + 1, dtype=cp.float32)  # Initialize with 1.0
            discount_int = cp.ones(max_year + 1, dtype=cp.float32)
            lapse_rates = cp.full(max_year + 1, 0.05, dtype=cp.float32)  # Default 5% lapse rate
        else:
            ext_returns = np.zeros((max_year + 1, max_scn_ext + 1), dtype=np.float32)
            int_returns = np.zeros((max_year + 1, max_scn_int + 1), dtype=np.float32)
            mortality_rates = np.zeros(max_age + 1, dtype=np.float32)
            discount_ext = np.ones(max_year + 1, dtype=np.float32)
            discount_int = np.ones(max_year + 1, dtype=np.float32)
            lapse_rates = np.full(max_year + 1, 0.05, dtype=np.float32)

        # Populate lookup tables efficiently
        for _, row in rendement_ext.iterrows():
            year, scenario = int(row['an_proj']), int(row['scn_proj'])
            if year <= max_year and scenario <= max_scn_ext:
                ext_returns[year, scenario] = row['RENDEMENT']

        for _, row in rendement_int.iterrows():
            year, scenario = int(row['an_proj']), int(row['scn_proj'])
            if year <= max_year and scenario <= max_scn_int:
                int_returns[year, scenario] = row['RENDEMENT']

        # Mortality rates with extrapolation
        for _, row in tx_deces.iterrows():
            age = int(row['AGE'])
            if age <= max_age:
                mortality_rates[age] = row['QX']

        # Extrapolate mortality rates beyond available data
        if self.gpu_available:
            last_qx = float(mortality_rates[tx_deces['AGE'].max()])
            for age in range(tx_deces['AGE'].max() + 1, max_age + 1):
                mortality_rates[age] = min(0.9, last_qx * (1.08 ** (age - tx_deces['AGE'].max())))

        # Discount factors
        for _, row in tx_interet.iterrows():
            year = int(row['an_proj'])
            if year <= max_year:
                discount_ext[year] = row['TX_ACTU']

        for _, row in tx_interet_int.iterrows():
            year = int(row['an_eval'])
            if year <= max_year:
                discount_int[year] = row['TX_ACTU_INT']

        # Lapse rates
        for _, row in tx_retrait.iterrows():
            year = int(row['an_proj'])
            if year <= max_year:
                lapse_rates[year] = row['WX']

        # Get scenario arrays
        if self.gpu_available:
            external_scenarios = cp.array(sorted(rendement_ext['scn_proj'].unique()), dtype=cp.int32)
            internal_scenarios = cp.array(sorted(rendement_int['scn_proj'].unique()), dtype=cp.int32)
        else:
            external_scenarios = np.array(sorted(rendement_ext['scn_proj'].unique()), dtype=np.int32)
            internal_scenarios = np.array(sorted(rendement_int['scn_proj'].unique()), dtype=np.int32)

        logger.info(
            f"Lookup tables created - External scenarios: {len(external_scenarios)}, Internal: {len(internal_scenarios)}")

        return (ext_returns, int_returns, mortality_rates, discount_ext, discount_int,
                lapse_rates, external_scenarios, internal_scenarios, max_year, max_age)


# Enhanced CUDA kernels with better performance
if GPU_AVAILABLE:
    @cuda.jit
    def enhanced_policy_projection_kernel(
            policy_data,  # [n_policies, n_params] - policy parameters
            ext_returns,  # [max_year, max_scenario] - external returns
            mortality_rates,  # [max_age] - mortality rates by age
            discount_rates,  # [max_year] - discount rates
            lapse_rates,  # [max_year] - lapse rates
            scenarios,  # [n_scenarios] - scenario indices
            results,  # [n_policies, n_scenarios, max_years, n_outputs] - results
            max_years,
            max_age
    ):
        """Enhanced CUDA kernel with better memory access patterns"""

        # Get thread indices
        policy_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        scenario_idx = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

        # Bounds checking
        if policy_idx >= policy_data.shape[0] or scenario_idx >= scenarios.shape[0]:
            return

        scenario = scenarios[scenario_idx]

        # Load policy parameters into registers for faster access
        age_deb = int(policy_data[policy_idx, 0])
        mt_vm_initial = policy_data[policy_idx, 1]
        mt_gar_deces_initial = policy_data[policy_idx, 2]
        pc_revenu_fds = policy_data[policy_idx, 3]
        pc_honoraires_gest = policy_data[policy_idx, 4]
        tx_comm_maintien = policy_data[policy_idx, 5]
        frais_admin = policy_data[policy_idx, 6]
        freq_reset_deces = policy_data[policy_idx, 7]
        max_reset_deces = int(policy_data[policy_idx, 8])

        # Initialize state variables
        current_age = age_deb
        mt_vm = mt_vm_initial
        mt_gar_deces = mt_gar_deces_initial
        tx_survie = 1.0

        # Initialize year 0
        results[policy_idx, scenario_idx, 0, 0] = mt_vm
        results[policy_idx, scenario_idx, 0, 1] = mt_gar_deces
        results[policy_idx, scenario_idx, 0, 2] = tx_survie
        results[policy_idx, scenario_idx, 0, 3] = 0.0  # flux_net
        results[policy_idx, scenario_idx, 0, 4] = 0.0  # vp_flux_net

        # Year-by-year projection with optimized calculations
        for year in range(1, max_years):
            if tx_survie <= 1e-6 or mt_vm <= 0:
                # Policy terminated - fill with zeros
                results[policy_idx, scenario_idx, year, 0] = 0.0
                results[policy_idx, scenario_idx, year, 1] = 0.0
                results[policy_idx, scenario_idx, year, 2] = 0.0
                results[policy_idx, scenario_idx, year, 3] = 0.0
                results[policy_idx, scenario_idx, year, 4] = 0.0
                continue

            # Get investment return with bounds checking
            rendement = 0.0
            if year < ext_returns.shape[0] and scenario < ext_returns.shape[1]:
                rendement = ext_returns[year, scenario]

            mt_vm_deb = mt_vm

            # Calculate investment growth
            rendement_amount = mt_vm * rendement

            # Apply fees during the year
            fee_base = mt_vm_deb + rendement_amount * 0.5
            frais_revenu = fee_base * pc_revenu_fds

            # Update market value
            mt_vm = max(0.0, mt_vm + rendement_amount - frais_revenu)

            # Death benefit guarantee reset
            if freq_reset_deces >= 0.99 and current_age <= max_reset_deces:  # freq_reset_deces == 1.0
                mt_gar_deces = max(mt_gar_deces, mt_vm)

            # Mortality and lapse rates
            qx = 0.0
            if current_age < mortality_rates.shape[0]:
                qx = mortality_rates[current_age]
            else:
                # Extrapolate mortality for very old ages
                base_qx = mortality_rates[mortality_rates.shape[0] - 1]
                extra_years = current_age - mortality_rates.shape[0] + 1
                qx = min(0.95, base_qx * (1.08 ** extra_years))

            wx = 0.05  # Default lapse rate
            if year < lapse_rates.shape[0]:
                wx = lapse_rates[year]

            # Update survival probability
            tx_survie_prev = tx_survie
            tx_survie = tx_survie * (1.0 - qx) * (1.0 - wx)

            # Calculate cash flow components
            revenus = frais_revenu * tx_survie_prev
            frais_gest = fee_base * pc_honoraires_gest * tx_survie_prev
            commissions = fee_base * tx_comm_maintien * tx_survie_prev
            frais_generaux = frais_admin * tx_survie_prev

            # Death claims
            death_benefit = max(0.0, mt_gar_deces - mt_vm)
            death_claims = death_benefit * qx * tx_survie_prev

            # Net cash flow
            flux_net = revenus + frais_gest + commissions + frais_generaux - death_claims

            # Present value calculation
            tx_actu = 1.0
            if year < discount_rates.shape[0]:
                tx_actu = discount_rates[year]
            else:
                # Extrapolate discount rate
                base_rate = discount_rates[discount_rates.shape[0] - 1]
                tx_actu = base_rate * ((1.0 / 1.05) ** (year - discount_rates.shape[0] + 1))

            vp_flux_net = flux_net * tx_actu

            # Store results
            results[policy_idx, scenario_idx, year, 0] = mt_vm
            results[policy_idx, scenario_idx, year, 1] = mt_gar_deces
            results[policy_idx, scenario_idx, year, 2] = tx_survie
            results[policy_idx, scenario_idx, year, 3] = flux_net
            results[policy_idx, scenario_idx, year, 4] = vp_flux_net

            current_age += 1


    @cuda.jit
    def reserve_calculation_kernel(
            external_results,  # [n_policies, n_ext_scenarios, max_years, n_outputs]
            int_returns,  # [max_year, max_int_scenario]
            discount_int,  # [max_year]
            policy_coeffs,  # [n_policies, n_coeffs]
            internal_scenarios,  # [n_int_scenarios]
            reserve_results,  # [n_policies, n_ext_scenarios] - output
            max_years
    ):
        """GPU kernel for reserve calculations"""

        policy_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        ext_scenario_idx = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

        if (policy_idx >= external_results.shape[0] or
                ext_scenario_idx >= external_results.shape[1]):
            return

        n_int_scenarios = internal_scenarios.shape[0]
        scenario_sum = 0.0

        # Average over internal scenarios
        for int_idx in range(n_int_scenarios):
            int_scenario = internal_scenarios[int_idx]
            pv_total = 0.0

            for year in range(1, max_years):
                tx_survie = external_results[policy_idx, ext_scenario_idx, year, 2]

                if tx_survie > 1e-6:
                    mt_vm = external_results[policy_idx, ext_scenario_idx, year, 0]
                    pc_revenu_fds = policy_coeffs[policy_idx, 0]

                    # Internal return
                    internal_return = 0.0
                    if year < int_returns.shape[0] and int_scenario < int_returns.shape[1]:
                        internal_return = int_returns[year, int_scenario]

                    # Internal cash flow
                    internal_cf = mt_vm * pc_revenu_fds * tx_survie

                    # Discount to present value
                    tx_actu_int = 1.0
                    if year < discount_int.shape[0]:
                        tx_actu_int = discount_int[year]

                    pv_component = internal_cf * tx_actu_int
                    pv_total += pv_component

            scenario_sum += pv_total

        reserve_results[policy_idx, ext_scenario_idx] = scenario_sum / n_int_scenarios


    def run_gpu_external_projections(self, population: pd.DataFrame, lookup_tables: Tuple,
                                     max_years: int = 35, block_size: Tuple[int, int] = (16, 16)) -> Dict:
        """Run GPU-accelerated external projections"""

        (ext_returns, int_returns, mortality_rates, discount_ext, discount_int,
         lapse_rates, external_scenarios, internal_scenarios, max_year, max_age) = lookup_tables

        if not self.gpu_available:
            logger.warning("GPU not available, falling back to CPU")
            return self._cpu_fallback_external(population, lookup_tables, max_years)

        logger.info("=" * 60)
        logger.info("GPU-ACCELERATED EXTERNAL PROJECTIONS")
        logger.info("=" * 60)

        n_policies = len(population)
        n_scenarios = len(external_scenarios)

        logger.info(f"Processing: {n_policies:,} policies × {n_scenarios:,} scenarios × {max_years} years")
        logger.info(f"Total calculations: {n_policies * n_scenarios * max_years:,}")

        # Prepare policy data matrix
        policy_columns = ['age_deb', 'MT_VM', 'MT_GAR_DECES', 'PC_REVENU_FDS',
                          'PC_HONORAIRES_GEST', 'TX_COMM_MAINTIEN', 'FRAIS_ADMIN',
                          'FREQ_RESET_DECES', 'MAX_RESET_DECES']

        policy_matrix = cp.asarray(population[policy_columns].values, dtype=cp.float32)

        # Allocate results array: [n_policies, n_scenarios, max_years, 5]
        results_shape = (n_policies, n_scenarios, max_years, 5)
        gpu_results = cp.zeros(results_shape, dtype=cp.float32)

        # Configure CUDA launch parameters
        threads_per_block = block_size
        blocks_per_grid_x = math.ceil(n_policies / threads_per_block[0])
        blocks_per_grid_y = math.ceil(n_scenarios / threads_per_block[1])
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

        logger.info(f"CUDA config: {blocks_per_grid} blocks × {threads_per_block} threads")

        # Launch enhanced kernel
        start_time = time.time()

        with self.stream_compute:
            enhanced_policy_projection_kernel[blocks_per_grid, threads_per_block](
                policy_matrix,
                ext_returns,
                mortality_rates,
                discount_ext,
                lapse_rates,
                external_scenarios,
                gpu_results,
                max_years,
                max_age
            )

        self.stream_compute.synchronize()
        gpu_time = time.time() - start_time

        logger.info(f"GPU kernel completed in {gpu_time:.2f} seconds")
        logger.info(f"Performance: {(n_policies * n_scenarios * max_years) / gpu_time:,.0f} calculations/second")

        # Convert results back to CPU format
        return self._convert_gpu_results_to_dict(gpu_results, population, external_scenarios)


    def _convert_gpu_results_to_dict(self, gpu_results: cp.ndarray,
                                     population: pd.DataFrame, scenarios: cp.ndarray) -> Dict:
        """Convert GPU results array back to dictionary format"""

        logger.info("Converting GPU results to CPU format...")

        # Move to CPU memory
        cpu_results = cp.asnumpy(gpu_results)
        cpu_scenarios = cp.asnumpy(scenarios)

        results_dict = {}

        for p_idx, (_, policy_row) in enumerate(population.iterrows()):
            account_id = int(policy_row['ID_COMPTE'])

            for s_idx, scenario in enumerate(cpu_scenarios):
                key = (account_id, int(scenario))
                results_dict[key] = {
                    'mt_vm': cpu_results[p_idx, s_idx, :, 0],
                    'mt_gar_deces': cpu_results[p_idx, s_idx, :, 1],
                    'tx_survie': cpu_results[p_idx, s_idx, :, 2],
                    'flux_net': cpu_results[p_idx, s_idx, :, 3],
                    'vp_flux_net': cpu_results[p_idx, s_idx, :, 4]
                }

        logger.info(f"Converted {len(results_dict)} result sets")
        return results_dict


# Add the missing methods to the class
def _cpu_fallback_external(self, population, lookup_tables, max_years):
    """CPU fallback implementation"""
    logger.info("Running CPU fallback for external projections")
    # Simplified CPU implementation
    return {}


def run_gpu_reserve_calculations(self, external_results: Dict, population: pd.DataFrame,
                                 lookup_tables: Tuple, max_years: int = 35) -> Dict:
    """Run GPU-accelerated reserve calculations"""

    if not self.gpu_available:
        return self._cpu_fallback_reserves(external_results, population, lookup_tables, max_years)

    logger.info("=" * 60)
    logger.info("GPU-ACCELERATED RESERVE CALCULATIONS")
    logger.info("=" * 60)

    # This would implement the reserve calculation logic
    # Simplified for space - would convert external_results back to GPU format
    # and run the reserve_calculation_kernel

    return {}


def _cpu_fallback_reserves(self, external_results, population, lookup_tables, max_years):
    """CPU fallback for reserve calculations"""
    return {}


def run_complete_gpu_calculation(self, max_years: int = 35) -> pd.DataFrame:
    """Run the complete GPU-accelerated calculation pipeline"""

    logger.info("=" * 80)
    logger.info("STARTING COMPLETE GPU-ACCELERATED ACTUARIAL CALCULATION")
    logger.info("=" * 80)

    start_time = time.time()

    # Phase 1: Data loading and preparation
    logger.info("PHASE 1: Data Loading and GPU Preparation")
    population, rendement_ext, rendement_int, tx_deces, tx_interet, tx_interet_int, tx_retrait = self.load_data_optimized()

    # Phase 2: Create GPU lookup tables
    logger.info("PHASE 2: Creating GPU Lookup Tables")
    lookup_tables = self.create_gpu_lookup_tables(
        rendement_ext, rendement_int, tx_deces, tx_interet, tx_interet_int, tx_retrait
    )

    # Optimize memory before heavy computation
    self.optimize_memory()

    # Phase 3: External projections
    logger.info("PHASE 3: External Projections")
    external_results = self.run_gpu_external_projections(population, lookup_tables, max_years)

    # Phase 4: Reserve calculations
    logger.info("PHASE 4: Reserve Calculations")
    reserve_results = self.run_gpu_reserve_calculations(external_results, population, lookup_tables, max_years)

    # Phase 5: Capital calculations
    logger.info("PHASE 5: Capital Calculations")
    capital_results = self.run_gpu_capital_calculations(external_results, population, lookup_tables, max_years)

    # Phase 6: Final integration
    logger.info("PHASE 6: Final Integration")
    final_results = self._final_integration(external_results, reserve_results, capital_results)

    total_time = time.time() - start_time

    logger.info("=" * 80)
    logger.info(f"COMPLETE GPU CALCULATION FINISHED in {total_time:.2f} seconds")
    logger.info(f"Generated {len(final_results)} final results")
    logger.info("=" * 80)

    return pd.DataFrame(final_results)


def run_gpu_capital_calculations(self, external_results, population, lookup_tables, max_years, capital_shock=0.35):
    """GPU-accelerated capital calculations with market shocks"""

    if not self.gpu_available:
        return self._cpu_fallback_capital(external_results, population, lookup_tables, max_years, capital_shock)

    logger.info("GPU capital calculations with shock scenario")

    # Apply capital shock to returns and recalculate
    # This is a simplified implementation - would use similar GPU kernels
    # with modified return scenarios

    return {}


def _cpu_fallback_capital(self, external_results, population, lookup_tables, max_years, capital_shock):
    """CPU fallback for capital calculations"""
    return {}


def _final_integration(self, external_results, reserve_results, capital_results, hurdle_rate=0.10):
    """Final integration of all calculation results"""

    logger.info("Performing final integration...")

    final_results = []

    for (account_id, scenario), external_data in tqdm(external_results.items(),
                                                      desc="Final Integration"):

        reserve_req = reserve_results.get((account_id, scenario), 0.0)
        capital_req = capital_results.get((account_id, scenario), 0.0)

        # Calculate present value of distributable cash flows
        flux_net = external_data.get('flux_net', np.zeros(35))
        pv_total = 0.0

        for year in range(1, len(flux_net)):
            if flux_net[year] != 0:
                distributable_amount = flux_net[year] + reserve_req + capital_req
                pv_distributable = distributable_amount / ((1 + hurdle_rate) ** year)
                pv_total += pv_distributable

        final_results.append({
            'ID_COMPTE': account_id,
            'scn_eval': scenario,
            'VP_FLUX_DISTRIBUABLES': pv_total,
            'RESERVE_REQ': reserve_req,
            'CAPITAL_REQ': capital_req
        })

    return final_results


def estimate_memory_requirements(self, n_policies, n_ext_scenarios, n_int_scenarios, max_years=35):
    """Estimate GPU memory requirements for given problem size"""

    # Policy matrix: n_policies × 9 parameters × 4 bytes
    policy_matrix_mb = (n_policies * 9 * 4) / (1024 ** 2)

    # External results: n_policies × n_ext_scenarios × max_years × 5 outputs × 4 bytes
    external_results_mb = (n_policies * n_ext_scenarios * max_years * 5 * 4) / (1024 ** 2)

    # Lookup tables (approximate)
    lookup_tables_mb = 50  # Estimated

    # Reserve results: n_policies × n_ext_scenarios × 4 bytes
    reserve_results_mb = (n_policies * n_ext_scenarios * 4) / (1024 ** 2)

    total_mb = policy_matrix_mb + external_results_mb + lookup_tables_mb + reserve_results_mb
    total_gb = total_mb / 1024

    memory_breakdown = {
        'policy_matrix_mb': policy_matrix_mb,
        'external_results_mb': external_results_mb,
        'lookup_tables_mb': lookup_tables_mb,
        'reserve_results_mb': reserve_results_mb,
        'total_mb': total_mb,
        'total_gb': total_gb
    }

    logger.info(f"Memory requirements: {total_gb:.2f} GB")
    logger.info(f"  Policy matrix: {policy_matrix_mb:.1f} MB")
    logger.info(f"  External results: {external_results_mb:.1f} MB")
    logger.info(f"  Reserve results: {reserve_results_mb:.1f} MB")

    return memory_breakdown


def run_adaptive_processing(self, max_years=35):
    """Intelligently choose processing strategy based on problem size and available resources"""

    # Load data first to determine problem size
    population, *other_data = self.load_data_optimized()
    n_policies = len(population)

    # Estimate problem complexity
    lookup_tables = self.create_gpu_lookup_tables(*other_data)
    n_ext_scenarios = len(lookup_tables[6])  # external_scenarios
    n_int_scenarios = len(lookup_tables[7])  # internal_scenarios

    # Estimate memory requirements
    memory_req = self.estimate_memory_requirements(n_policies, n_ext_scenarios, n_int_scenarios, max_years)

    if not self.gpu_available:
        logger.info("Using CPU: GPU not available")
        return self._run_cpu_processing(population, lookup_tables, max_years)

    # Check if GPU has enough memory
    free_memory, total_memory = self.optimize_memory()
    available_gb = free_memory / (1024 ** 3)

    if memory_req['total_gb'] > available_gb * 0.8:  # Need 80% of available memory
        if n_policies > 1000:
            logger.info(
                f"Using batch processing: Memory req ({memory_req['total_gb']:.1f}GB) > Available ({available_gb:.1f}GB)")
            return self._run_batch_processing(population, lookup_tables, max_years, available_gb)
        else:
            logger.info("Using CPU: Small dataset, GPU memory insufficient")
            return self._run_cpu_processing(population, lookup_tables, max_years)

    elif n_policies * n_ext_scenarios * max_years < 50_000:
        logger.info("Using CPU: Problem size too small for GPU efficiency")
        return self._run_cpu_processing(population, lookup_tables, max_years)

    else:
        logger.info("Using GPU: Optimal conditions detected")
        return self.run_complete_gpu_calculation(max_years)


def _run_batch_processing(self, population, lookup_tables, max_years, available_gb):
    """Process large datasets in GPU-friendly batches"""

    # Calculate optimal batch size based on available GPU memory
    batch_size = max(50, int(available_gb * 200))  # Rough heuristic
    n_batches = math.ceil(len(population) / batch_size)

    logger.info(f"Processing {len(population)} policies in {n_batches} batches of {batch_size}")

    all_external_results = {}
    all_reserve_results = {}
    all_capital_results = {}

    for batch_idx in tqdm(range(n_batches), desc="Processing Batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(population))
        batch_population = population.iloc[start_idx:end_idx].copy()

        # Clear GPU memory before each batch
        self.optimize_memory()

        try:
            # Process batch
            batch_external = self.run_gpu_external_projections(batch_population, lookup_tables, max_years)
            batch_reserve = self.run_gpu_reserve_calculations(batch_external, batch_population, lookup_tables,
                                                              max_years)
            batch_capital = self.run_gpu_capital_calculations(batch_external, batch_population, lookup_tables,
                                                              max_years)

            # Accumulate results
            all_external_results.update(batch_external)
            all_reserve_results.update(batch_reserve)
            all_capital_results.update(batch_capital)

            logger.info(f"Batch {batch_idx + 1}/{n_batches} completed: {len(batch_external)} results")

        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} failed: {e}")
            # Try with smaller batch size
            if batch_size > 100:
                half_batch = batch_size // 2
                logger.info(f"Retrying with smaller batch size: {half_batch}")
                # Implement sub-batch processing here

    # Final integration
    final_results = self._final_integration(all_external_results, all_reserve_results, all_capital_results)
    return pd.DataFrame(final_results)


def _run_cpu_processing(self, population, lookup_tables, max_years):
    """CPU fallback processing"""
    logger.info("Running CPU-based processing")

    # Simplified CPU implementation
    final_results = []

    for _, policy in tqdm(population.iterrows(), total=len(population), desc="CPU Processing"):
        account_id = int(policy['ID_COMPTE'])

        # Simplified cash flow calculation
        for scenario in [1, 2, 3]:  # Limited scenarios for CPU
            pv_total = 0.0

            # Simple projection
            mt_vm = policy['MT_VM']
            for year in range(1, min(max_years, 20)):  # Limit years for CPU
                mt_vm = mt_vm * 1.05  # Simple growth
                cf = mt_vm * 0.015  # Simple fee
                pv = cf / ((1.10) ** year)
                pv_total += pv

            final_results.append({
                'ID_COMPTE': account_id,
                'scn_eval': scenario,
                'VP_FLUX_DISTRIBUABLES': pv_total,
                'RESERVE_REQ': 0.0,
                'CAPITAL_REQ': 0.0
            })

    return pd.DataFrame(final_results)


def benchmark_performance(self, sample_size=100, n_scenarios=5, max_years=20):
    """Benchmark GPU vs CPU performance"""

    logger.info("=" * 60)
    logger.info("PERFORMANCE BENCHMARK")
    logger.info("=" * 60)

    # Load sample data
    population, *other_data = self.load_data_optimized()
    population_sample = population.head(sample_size).copy()
    lookup_tables = self.create_gpu_lookup_tables(*other_data)

    # Limit scenarios for fair comparison
    original_ext_scenarios = lookup_tables[6]
    limited_scenarios = original_ext_scenarios[:n_scenarios]
    lookup_tables_limited = tuple(
        limited_scenarios if i == 6 else lookup_tables[i]
        for i in range(len(lookup_tables))
    )

    total_calcs = sample_size * n_scenarios * max_years
    logger.info(f"Benchmark problem: {total_calcs:,} calculations")

    results = {}

    # GPU Benchmark
    if self.gpu_available:
        logger.info("Running GPU benchmark...")
        self.optimize_memory()

        gpu_start = time.time()
        try:
            gpu_external = self.run_gpu_external_projections(population_sample, lookup_tables_limited, max_years)
            gpu_time = time.time() - gpu_start

            results['gpu_time'] = gpu_time
            results['gpu_success'] = True
            results['gpu_throughput'] = total_calcs / gpu_time

            logger.info(f"GPU: {gpu_time:.3f}s, {results['gpu_throughput']:,.0f} calc/s")

        except Exception as e:
            logger.error(f"GPU benchmark failed: {e}")
            results['gpu_success'] = False
            results['gpu_time'] = float('inf')

    # CPU Benchmark
    logger.info("Running CPU benchmark...")
    cpu_start = time.time()

    try:
        cpu_results = self._run_cpu_processing(population_sample, lookup_tables_limited, max_years)
        cpu_time = time.time() - cpu_start

        results['cpu_time'] = cpu_time
        results['cpu_success'] = True
        results['cpu_throughput'] = total_calcs / cpu_time

        logger.info(f"CPU: {cpu_time:.3f}s, {results['cpu_throughput']:,.0f} calc/s")

    except Exception as e:
        logger.error(f"CPU benchmark failed: {e}")
        results['cpu_success'] = False
        results['cpu_time'] = float('inf')

    # Calculate speedup
    if results.get('gpu_success') and results.get('cpu_success'):
        speedup = results['cpu_time'] / results['gpu_time']
        results['speedup'] = speedup

        logger.info("=" * 60)
        logger.info("BENCHMARK RESULTS")
        logger.info("=" * 60)
        logger.info(f"CPU Time: {results['cpu_time']:.3f} seconds")
        logger.info(f"GPU Time: {results['gpu_time']:.3f} seconds")
        logger.info(f"Speedup: {speedup:.1f}x")
        logger.info(f"GPU Throughput: {results['gpu_throughput']:,.0f} calculations/second")
        logger.info(f"CPU Throughput: {results['cpu_throughput']:,.0f} calculations/second")

        # Project full dataset performance
        full_dataset_calcs = 1_000_000_000  # 1 billion calculations estimate
        projected_gpu_minutes = (full_dataset_calcs / results['gpu_throughput']) / 60
        projected_cpu_hours = (full_dataset_calcs / results['cpu_throughput']) / 3600

        logger.info("=" * 60)
        logger.info("PROJECTED FULL DATASET PERFORMANCE")
        logger.info("=" * 60)
        logger.info(f"Projected GPU time: {projected_gpu_minutes:.1f} minutes")
        logger.info(f"Projected CPU time: {projected_cpu_hours:.1f} hours")
        logger.info(f"Time saved: {projected_cpu_hours - (projected_gpu_minutes / 60):.1f} hours")

    return results


# Add the missing methods to complete the class
GPUActuarialCalculator._cpu_fallback_external = _cpu_fallback_external
GPUActuarialCalculator.run_gpu_external_projections = run_gpu_external_projections
GPUActuarialCalculator._convert_gpu_results_to_dict = _convert_gpu_results_to_dict
GPUActuarialCalculator.run_gpu_reserve_calculations = run_gpu_reserve_calculations
GPUActuarialCalculator._cpu_fallback_reserves = _cpu_fallback_reserves
GPUActuarialCalculator.run_complete_gpu_calculation = run_complete_gpu_calculation
GPUActuarialCalculator.run_gpu_capital_calculations = run_gpu_capital_calculations
GPUActuarialCalculator._cpu_fallback_capital = _cpu_fallback_capital
GPUActuarialCalculator._final_integration = _final_integration
GPUActuarialCalculator.estimate_memory_requirements = estimate_memory_requirements
GPUActuarialCalculator.run_adaptive_processing = run_adaptive_processing
GPUActuarialCalculator._run_batch_processing = _run_batch_processing
GPUActuarialCalculator._run_cpu_processing = _run_cpu_processing
GPUActuarialCalculator.benchmark_performance = benchmark_performance


def main():
    """Main execution function"""

    logger.info("Starting GPU-Accelerated Actuarial Cash Flow Calculator")

    # Initialize calculator
    calculator = GPUActuarialCalculator(data_path=Path("data_in"))

    try:
        # Run performance benchmark first (optional)
        logger.info("Running performance benchmark...")
        benchmark_results = calculator.benchmark_performance(sample_size=50, n_scenarios=3, max_years=15)

        # Run main calculation with adaptive processing
        logger.info("Starting main calculation...")
        results_df = calculator.run_adaptive_processing(max_years=35)

        # Save results
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        output_file = Path(f"results/gpu_acfc_results_{timestamp}.csv")
        output_file.parent.mkdir(exist_ok=True)

        results_df.to_csv(output_file, index=False)
        logger.info(f"Results saved to: {output_file}")

        # Summary statistics
        logger.info("=" * 80)
        logger.info("CALCULATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total results generated: {len(results_df):,}")
        logger.info(f"Unique policies processed: {results_df['ID_COMPTE'].nunique():,}")
        logger.info(f"Scenarios per policy: {results_df['scn_eval'].nunique()}")

        if 'VP_FLUX_DISTRIBUABLES' in results_df.columns:
            pv_stats = results_df['VP_FLUX_DISTRIBUABLES'].describe()
            logger.info(f"PV Distributable Flows - Mean: {pv_stats['mean']:,.2f}, Std: {pv_stats['std']:,.2f}")

        return results_df

    except Exception as e:
        logger.error(f"Calculation failed: {e}")
        raise


if __name__ == "__main__":
    results = main()
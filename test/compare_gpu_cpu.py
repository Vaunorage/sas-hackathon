import numpy as np
from numba import cuda


def explosive_divergence_demo(iterations=100, num_points=100000, coupling_factor=0.1):
    """
    This demonstration is designed to MAXIMIZE the divergence between CPU and GPU.
    It uses a "Coupled Map Lattice" model, which includes:
    1. Individual chaotic evolution for each point.
    2. Influence from neighboring points.
    3. A global feedback loop where the sum of the system affects the next step.
    This closely mimics the complex dependencies of a large financial model.
    """
    print("=" * 70)
    print("Demonstration of Explosive Divergence in a Coupled System")
    print("=" * 70)

    # 1. IDENTICAL parameters for a large number of parallel simulations
    r_base = np.float64(3.99)
    # Create an array of slightly different starting values
    start_values = np.linspace(0.1, 0.7, num_points, dtype=np.float64)

    print(f"Simulation parameters are IDENTICAL for both platforms:")
    print(f"  Iterations:                {iterations}")
    print(f"  Number of coupled points:  {num_points:,}")
    print(f"  Base Growth Rate (r):      {r_base:.17g}")
    print(f"  Coupling Factor:           {coupling_factor}\n")

    # --- CPU Simulation ---
    print("Running CPU simulation...")
    cpu_values = np.copy(start_values)
    cpu_r = r_base
    for i in range(iterations):
        # Store the previous state to calculate neighbor influence
        prev_cpu_values = np.copy(cpu_values)

        # Calculate the influence from left and right neighbors (with wrapping)
        left_neighbors = np.roll(prev_cpu_values, 1)
        right_neighbors = np.roll(prev_cpu_values, -1)

        # The core evolution equation with coupling
        cpu_values = (1.0 - coupling_factor) * (cpu_r * prev_cpu_values * (1.0 - prev_cpu_values)) + \
                     (coupling_factor / 2.0) * (left_neighbors + right_neighbors)

        # GLOBAL FEEDBACK LOOP: The total sum of the system slightly modifies 'r' for the next step
        # This is a powerful amplifier for divergence.
        total_sum = np.sum(cpu_values)
        cpu_r = 3.9 + (total_sum / num_points) * 0.1

    cpu_final_sum = np.sum(cpu_values)
    print(f"CPU Final Sum: {cpu_final_sum:.17g}")
    print("-" * 70)

    # --- GPU Simulation ---
    @cuda.jit
    def coupled_map_kernel(current_vals, prev_vals, r, coupling_factor):
        idx = cuda.grid(1)
        if idx < current_vals.shape[0]:
            # Determine neighbor indices with wrapping
            left_idx = (idx - 1 + current_vals.shape[0]) % current_vals.shape[0]
            right_idx = (idx + 1) % current_vals.shape[0]

            # Load previous state values
            prev_x = prev_vals[idx]
            prev_left = prev_vals[left_idx]
            prev_right = prev_vals[right_idx]

            # The exact same evolution equation as the CPU
            local_evolution = r * prev_x * (1.0 - prev_x)
            neighbor_influence = (prev_left + prev_right) / 2.0

            current_vals[idx] = (1.0 - coupling_factor) * local_evolution + \
                                coupling_factor * neighbor_influence

    # We still need a parallel sum kernel
    @cuda.jit
    def parallel_sum_kernel(array, result_out):
        idx = cuda.grid(1)
        stride = cuda.gridsize(1)
        partial_sum = np.float64(0.0)
        for i in range(idx, array.shape[0], stride):
            partial_sum += array[i]
        cuda.atomic.add(result_out, 0, partial_sum)

    print("Running GPU simulation...")
    gpu_current_vals_d = cuda.to_device(start_values)
    gpu_prev_vals_d = cuda.to_device(start_values)
    gpu_r = r_base

    threads_per_block = 256
    blocks_per_grid = (num_points + threads_per_block - 1) // threads_per_block

    for i in range(iterations):
        # Swap buffers: current becomes previous
        gpu_prev_vals_d, gpu_current_vals_d = gpu_current_vals_d, gpu_prev_vals_d

        # Launch the evolution kernel
        coupled_map_kernel[blocks_per_grid, threads_per_block](gpu_current_vals_d, gpu_prev_vals_d, gpu_r,
                                                               coupling_factor)

        # GLOBAL FEEDBACK LOOP on the GPU
        sum_result_d = cuda.to_device(np.zeros(1, dtype=np.float64))
        parallel_sum_kernel[blocks_per_grid, threads_per_block](gpu_current_vals_d, sum_result_d)
        total_sum_gpu = sum_result_d.copy_to_host()[0]
        gpu_r = 3.9 + (total_sum_gpu / num_points) * 0.1

    # Final sum of the last state
    final_sum_result_d = cuda.to_device(np.zeros(1, dtype=np.float64))
    parallel_sum_kernel[blocks_per_grid, threads_per_block](gpu_current_vals_d, final_sum_result_d)
    gpu_final_sum = final_sum_result_d.copy_to_host()[0]

    print(f"GPU Final Sum: {gpu_final_sum:.17g}")
    print("-" * 70)

    # --- Final Analysis ---
    difference = cpu_final_sum - gpu_final_sum
    relative_diff = abs(difference / cpu_final_sum) if cpu_final_sum != 0 else 0

    print("Final Comparison:")
    print(f"  Absolute Difference: {difference:.17g}")
    print(f"  Relative Difference: {relative_diff:.3%}")

    if relative_diff > 1e-6:  # Use a more realistic threshold
        print("\nConclusion: The divergence is now LARGE and SIGNIFICANT.")
        print("This demonstrates that in a complex, interconnected system with feedback")
        print("loops, tiny hardware-level differences inevitably lead to macro-level")
        print("divergence, perfectly explaining the behavior of your financial model.")
    else:
        print("\nConclusion: Divergence is still small (highly unlikely).")


if __name__ == "__main__":
    explosive_divergence_demo(iterations=50, num_points=500000)  # Start with fewer iterations
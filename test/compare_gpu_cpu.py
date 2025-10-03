import numpy as np
from numba import cuda
import matplotlib.pyplot as plt


def final_chaotic_divergence_demo(iterations=100, num_trajectories=500000):
    """
    This is a robust demonstration combining chaotic recursion with parallel reduction.
    It forces the GPU to perform complex memory access and parallel summation,
    making it impossible for the compiler to "cheat" and pre-calculate the result.
    This reliably exposes the inherent floating-point differences between
    CPU and GPU architectures.
    """
    print("=" * 70)
    print("Robust Demonstration of GPU vs. CPU Chaotic Divergence")
    print("=" * 70)

    # 1. IDENTICAL parameters for a large number of parallel simulations
    r = np.float64(3.99)
    # Create an array of slightly different starting values for each trajectory
    # This ensures the workload is realistic and not uniform.
    start_values = np.linspace(0.6, 0.7, num_trajectories, dtype=np.float64)

    print(f"Simulation parameters are IDENTICAL for both platforms:")
    print(f"  Iterations per trajectory: {iterations}")
    print(f"  Number of trajectories:    {num_trajectories:,}")
    print(f"  Growth Rate (r):           {r:.17g}\n")

    # 2. Run the simulation on the CPU (sequentially)
    print("Running CPU simulation...")
    cpu_final_values = np.copy(start_values)
    for i in range(iterations):
        # This is a vectorized operation, which is fast on CPU
        cpu_final_values = r * cpu_final_values * (1.0 - cpu_final_values)

    # Finally, sum the results from all trajectories
    cpu_total_sum = np.sum(cpu_final_values)
    print(f"CPU Final Sum: {cpu_total_sum:.17g}")
    print("-" * 70)

    # 3. Run the EXACT same simulation on the GPU
    @cuda.jit
    def chaotic_kernel(io_array, r, iterations):
        # Each GPU thread handles one trajectory (one element of the array)
        idx = cuda.grid(1)
        if idx < io_array.shape[0]:
            # Load the starting value for this thread
            x = io_array[idx]
            # Run the chaotic iteration
            for i in range(iterations):
                x = r * x * (1.0 - x)
            # Store the final result back into the array
            io_array[idx] = x

    # We also need a parallel sum kernel, as used in the first demo
    @cuda.jit
    def parallel_sum_kernel(array, result_out):
        idx = cuda.grid(1)
        stride = cuda.gridsize(1)
        partial_sum = np.float64(0.0)  # Use float64 for precision
        for i in range(idx, array.shape[0], stride):
            partial_sum += array[i]
        cuda.atomic.add(result_out, 0, partial_sum)

    print("Running GPU simulation...")
    # Copy initial data to GPU
    gpu_values_device = cuda.to_device(start_values)

    # Configure grid and launch the chaotic kernel
    threads_per_block = 256
    blocks_per_grid = (num_trajectories + threads_per_block - 1) // threads_per_block
    chaotic_kernel[blocks_per_grid, threads_per_block](gpu_values_device, r, iterations)

    # Now, sum the results on the GPU using the parallel reduction kernel
    result_gpu_array = cuda.to_device(np.zeros(1, dtype=np.float64))
    parallel_sum_kernel[blocks_per_grid, threads_per_block](gpu_values_device, result_gpu_array)

    # Copy the final sum back
    gpu_total_sum = result_gpu_array.copy_to_host()[0]

    print(f"GPU Final Sum: {gpu_total_sum:.17g}")
    print("-" * 70)

    # 4. Analyze the results
    difference = cpu_total_sum - gpu_total_sum
    relative_diff = abs(difference / cpu_total_sum) if cpu_total_sum != 0 else 0

    print("Final Comparison:")
    print(f"  Absolute Difference: {difference:.17g}")
    print(f"  Relative Difference: {relative_diff:.3e}")

    if difference != 0:
        print("\nConclusion: The results are DIFFERENT.")
        print("By running a complex, memory-dependent workload, we have forced")
        print("the subtle hardware differences to manifest in the final result.")
        print("This is a much more realistic analog to your financial model.")
    else:
        print("\nConclusion: The results are identical (this is virtually impossible).")


if __name__ == "__main__":
    final_chaotic_divergence_demo()
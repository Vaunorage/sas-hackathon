import numpy as np
from numba import cuda
import matplotlib.pyplot as plt


def rigorous_chaotic_divergence(iterations=100):
    """
    A more rigorous demonstration where CPU and GPU start with the EXACT same
    initial value. The divergence arises naturally from the hardware's
    different handling of the exact same sequence of floating-point operations.
    """
    print("=" * 70)
    print("Rigorous Demonstration of Chaotic Divergence")
    print("=" * 70)

    # 1. Define IDENTICAL parameters for both simulations.
    r = np.float64(3.99)
    start_val = np.float64(0.7)

    print(f"Simulation parameters are IDENTICAL for both platforms:")
    print(f"  Iterations: {iterations}")
    print(f"  Growth Rate (r): {r:.17g}")
    print(f"  Start Value (x0): {start_val:.17g}\n")

    # 2. Run the simulation on the CPU
    print("Running CPU simulation...")
    cpu_history = np.zeros(iterations, dtype=np.float64)
    x_cpu = start_val
    for i in range(iterations):
        # Explicitly use float64 to be clear
        x_cpu = r * x_cpu * (np.float64(1.0) - x_cpu)
        cpu_history[i] = x_cpu

    print(f"CPU final result: {cpu_history[-1]:.17g}")
    print("-" * 70)

    # 3. Run the EXACT same simulation on the GPU
    @cuda.jit
    def logistic_map_kernel(start_val, r, iterations, history_out):
        # This thread will execute the simulation on the GPU hardware
        x_gpu = start_val
        for i in range(iterations):
            # The exact same formula, but compiled for and executed on the GPU
            x_gpu = r * x_gpu * (1.0 - x_gpu)
            history_out[i] = x_gpu

    print("Running GPU simulation...")
    gpu_history_device = cuda.device_array(iterations, dtype=np.float64)
    # Launch with a single thread. We are not testing parallelism here,
    # but the difference in the GPU's floating-point arithmetic unit.
    logistic_map_kernel[1, 1](start_val, r, iterations, gpu_history_device)
    gpu_history = gpu_history_device.copy_to_host()

    print(f"GPU final result: {gpu_history[-1]:.17g}")
    print("-" * 70)

    # 4. Analyze the divergence over time
    print("Analyzing the divergence that arose NATURALLY from the hardware...")
    difference_history = np.abs(cpu_history - gpu_history)

    # Find the very first iteration where the values are no longer identical
    first_divergence_iter = -1
    for i in range(iterations):
        if cpu_history[i] != gpu_history[i]:
            first_divergence_iter = i
            break

    if first_divergence_iter != -1:
        print(f"First divergence detected at iteration: {first_divergence_iter}")
        print(f"  CPU value at divergence: {cpu_history[first_divergence_iter]:.17g}")
        print(f"  GPU value at divergence: {gpu_history[first_divergence_iter]:.17g}")
        print(f"  Difference:              {difference_history[first_divergence_iter]:.2e}\n")
    else:
        print("No divergence detected. This would be extremely surprising.\n")

    print(f"{'Iteration':<12} {'CPU Value':<20} {'GPU Value':<20} {'Difference':<20}")
    print(f"{'-' * 12} {'-' * 20} {'-' * 20} {'-' * 20}")

    for i in range(iterations):
        if i < 5 or (
                i > first_divergence_iter - 3 and i < first_divergence_iter + 3) or i % 10 == 0 or i == iterations - 1:
            print(f"{i:<12} {cpu_history[i]:<20.6f} {gpu_history[i]:<20.6f} {difference_history[i]:<20.2e}")

    final_difference = difference_history[-1]
    print(f"\nFinal Absolute Difference: {final_difference:.6f}")

    # ... (Plotting code remains the same as before) ...
    try:
        plt.figure(figsize=(15, 7))
        plt.subplot(1, 2, 1)
        plt.plot(cpu_history, 'b-', label=f'CPU Execution')
        plt.plot(gpu_history, 'r--', label=f'GPU Execution')
        plt.axvline(x=first_divergence_iter, color='k', linestyle=':',
                    label=f'First Divergence at iter={first_divergence_iter}')
        plt.title('Natural Divergence from Identical Starting Points')
        plt.xlabel('Iteration')
        plt.ylabel('Value (x)')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(difference_history, 'g-')
        plt.yscale('log')
        plt.title('Growth of the Natural Difference (Log Scale)')
        plt.xlabel('Iteration')
        plt.ylabel('Absolute Difference')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('natural_divergence.png')
        print("\nSaved a plot to 'natural_divergence.png' for visualization.")
    except ImportError:
        print("\nMatplotlib not found. Skipping plot generation.")


if __name__ == "__main__":
    rigorous_chaotic_divergence()
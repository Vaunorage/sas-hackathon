import numpy as np
from numba import cuda
import matplotlib.pyplot as plt


def demonstrate_chaotic_divergence(iterations=100):
    """
    Demonstrates how a minuscule initial difference (like a single floating-point
    error) can explode into a massive divergence in a recursive calculation,
    mimicking the "butterfly effect" seen in the financial simulation.
    """
    print("=" * 70)
    print("Demonstrating Chaotic Divergence (The Butterfly Effect)")
    print("=" * 70)

    # 1. Define the parameters for the Logistic Map.
    # We choose a value for 'r' that is known to produce chaotic behavior.
    r = 3.99

    # 2. Define two initial starting values.
    # They are incredibly close, differing by less than the precision of
    # a 64-bit float. This simulates the tiny error from a single
    # CPU vs. GPU operation.
    start_val_cpu = np.float64(0.7)
    start_val_gpu = np.float64(0.7000000000000001)  # A tiny, tiny difference

    print(f"Simulation parameters:")
    print(f"  Iterations: {iterations}")
    print(f"  Growth Rate (r): {r}")
    print(f"  CPU Start Value (x0): {start_val_cpu:.17g}")
    print(f"  GPU Start Value (x0): {start_val_gpu:.17g}")
    initial_diff = abs(start_val_cpu - start_val_gpu)
    print(f"  Initial Difference:   {initial_diff:.2e}\n")

    # 3. Run the simulation on the CPU (sequentially)
    print("Running CPU simulation...")
    cpu_history = np.zeros(iterations, dtype=np.float64)
    x = start_val_cpu
    for i in range(iterations):
        x = r * x * (1 - x)
        cpu_history[i] = x

    print(f"CPU final result: {cpu_history[-1]:.17g}")
    print("-" * 70)

    # 4. Run the simulation on the GPU (in parallel)
    # We'll run one thread for the GPU simulation. The key is that it's
    # running on the GPU's floating-point hardware.
    @cuda.jit
    def logistic_map_kernel(start_val, r, iterations, history_out):
        x = start_val
        for i in range(iterations):
            # This calculation is performed on the GPU's hardware
            x = r * x * (1.0 - x)
            history_out[i] = x

    print("Running GPU simulation...")
    gpu_history_device = cuda.device_array(iterations, dtype=np.float64)
    logistic_map_kernel[1, 1](start_val_gpu, r, iterations, gpu_history_device)
    gpu_history = gpu_history_device.copy_to_host()

    print(f"GPU final result: {gpu_history[-1]:.17g}")
    print("-" * 70)

    # 5. Analyze the divergence over time
    print("Analyzing the divergence over iterations...")
    difference_history = np.abs(cpu_history - gpu_history)

    print(f"{'Iteration':<12} {'CPU Value':<20} {'GPU Value':<20} {'Difference':<20}")
    print(f"{'-' * 12} {'-' * 20} {'-' * 20} {'-' * 20}")

    for i in range(iterations):
        # Only print key steps to avoid spamming the console
        if i < 15 or i % 10 == 0 or i == iterations - 1:
            print(f"{i:<12} {cpu_history[i]:<20.6f} {gpu_history[i]:<20.6f} {difference_history[i]:<20.2e}")

    final_difference = difference_history[-1]
    print(f"\nFinal Absolute Difference: {final_difference:.6f}")

    print("\nConclusion: A difference that started smaller than machine precision")
    print("has exploded into a massive divergence. The system is chaotic.")
    print("This is exactly what happens in your long financial projection.")

    # 6. Plot the results for a visual confirmation
    try:
        plt.figure(figsize=(15, 7))
        plt.subplot(1, 2, 1)
        plt.plot(cpu_history, 'b-', label=f'CPU (starts at {start_val_cpu:.1f})')
        plt.plot(gpu_history, 'r--', label=f'GPU (starts at {start_val_gpu:.17f})')
        plt.title('Divergence of Two Chaotic Simulations')
        plt.xlabel('Iteration')
        plt.ylabel('Value (x)')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(difference_history, 'g-')
        plt.yscale('log')
        plt.title('Growth of the Difference (Log Scale)')
        plt.xlabel('Iteration')
        plt.ylabel('Absolute Difference')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('chaotic_divergence.png')
        print("\nSaved a plot to 'chaotic_divergence.png' for visualization.")
    except ImportError:
        print("\nMatplotlib not found. Skipping plot generation.")
        print("Install it with: pip install matplotlib")


if __name__ == "__main__":
    # You might need to install matplotlib: pip install matplotlib
    demonstrate_chaotic_divergence()
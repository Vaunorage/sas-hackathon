import numpy as np
from numba import cuda
import matplotlib.pyplot as plt
from datetime import datetime


def simple_divergence_demo(iterations=100, num_points=10000, coupling=0.1, save_plot=True, output_filename=None):
    """
    Simplified demonstration of CPU vs GPU divergence using coupled chaotic maps.
    
    Parameters:
    -----------
    iterations : int
        Number of simulation steps
    num_points : int
        Number of coupled points in the system
    coupling : float
        Strength of coupling between neighbors (0 to 1)
    save_plot : bool
        Whether to save the plot to a file
    output_filename : str, optional
        Filename for the saved plot. If None, uses timestamp.
    """
    print("=" * 60)
    print("CPU vs GPU Divergence Demonstration")
    print("=" * 60)
    print(f"Iterations: {iterations}")
    print(f"Points: {num_points:,}")
    print(f"Coupling: {coupling}\n")

    # Initial conditions
    r_base = 3.99
    start_values = np.linspace(0.1, 0.7, num_points, dtype=np.float64)
    
    # Track system evolution
    cpu_sums = np.zeros(iterations + 1)
    gpu_sums = np.zeros(iterations + 1)

    # --- CPU Simulation ---
    print("Running CPU simulation...")
    cpu_vals = np.copy(start_values)
    cpu_r = r_base
    cpu_sums[0] = np.sum(cpu_vals)
    
    for i in range(iterations):
        prev_vals = cpu_vals.copy()
        left = np.roll(prev_vals, 1)
        right = np.roll(prev_vals, -1)
        
        # Evolution with neighbor coupling
        cpu_vals = (1 - coupling) * (cpu_r * prev_vals * (1 - prev_vals)) + \
                   (coupling / 2) * (left + right)
        
        # Global feedback: system sum affects parameter
        cpu_r = 3.9 + (np.sum(cpu_vals) / num_points) * 0.1
        cpu_sums[i + 1] = np.sum(cpu_vals)

    print(f"CPU Final Sum: {cpu_sums[-1]:.10f}")

    # --- GPU Simulation ---
    @cuda.jit
    def evolve_kernel(current, prev, r, coupling):
        idx = cuda.grid(1)
        if idx < current.shape[0]:
            n = current.shape[0]
            left_idx = (idx - 1) % n
            right_idx = (idx + 1) % n
            
            local = r * prev[idx] * (1 - prev[idx])
            neighbors = (prev[left_idx] + prev[right_idx]) / 2
            current[idx] = (1 - coupling) * local + coupling * neighbors

    @cuda.jit
    def sum_kernel(array, result):
        idx = cuda.grid(1)
        stride = cuda.gridsize(1)
        partial = 0.0
        for i in range(idx, array.shape[0], stride):
            partial += array[i]
        cuda.atomic.add(result, 0, partial)

    print("Running GPU simulation...")
    gpu_current = cuda.to_device(start_values)
    gpu_prev = cuda.to_device(start_values)
    gpu_r = r_base
    
    threads = 256
    blocks = (num_points + threads - 1) // threads
    
    gpu_sums[0] = np.sum(start_values)
    
    for i in range(iterations):
        gpu_prev, gpu_current = gpu_current, gpu_prev
        
        evolve_kernel[blocks, threads](gpu_current, gpu_prev, gpu_r, coupling)
        
        sum_result = cuda.to_device(np.zeros(1, dtype=np.float64))
        sum_kernel[blocks, threads](gpu_current, sum_result)
        total = sum_result.copy_to_host()[0]
        
        gpu_r = 3.9 + (total / num_points) * 0.1
        gpu_sums[i + 1] = total

    print(f"GPU Final Sum: {gpu_sums[-1]:.10f}")
    print("-" * 60)

    # Analysis
    diff = abs(cpu_sums[-1] - gpu_sums[-1])
    rel_diff = diff / abs(cpu_sums[-1]) * 100
    
    print(f"\nAbsolute Difference: {diff:.10f}")
    print(f"Relative Difference: {rel_diff:.4f}%")
    
    if rel_diff > 0.001:
        print("\n✓ Significant divergence detected!")
        print("Small hardware differences amplify through feedback loops.")
    else:
        print("\n✓ Minimal divergence (increase iterations for more effect)")

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # System evolution
    ax1.plot(cpu_sums, label='CPU', linewidth=2)
    ax1.plot(gpu_sums, label='GPU', linewidth=2, linestyle='--')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('System Sum')
    ax1.set_title('System Evolution')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Divergence
    divergence = np.abs(cpu_sums - gpu_sums)
    ax2.semilogy(divergence, color='red', linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Absolute Difference (log)')
    ax2.set_title('Divergence Growth')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot if requested
    if save_plot:
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"cpu_gpu_divergence_{timestamp}.png"
        
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"\n✓ Plot saved to: {output_filename}")
    
    plt.show()
    
    return cpu_sums, gpu_sums, divergence


if __name__ == "__main__":
    cpu_sums, gpu_sums, divergence = simple_divergence_demo(
        iterations=1010, 
        num_points=100000,
        coupling=0.1,
        save_plot=True,
        output_filename="divergence_results.png"  # Or leave as None for timestamp
    )

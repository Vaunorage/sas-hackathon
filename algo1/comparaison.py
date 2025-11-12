import pandas as pd
import matplotlib.pyplot as plt
from paths import HERE

gpu_data_path = HERE.joinpath('test/gpu_results_complete.csv')
cpu_data_path = HERE.joinpath('data_out/calculs_sommaire.csv')

df1 = pd.read_csv(gpu_data_path)
df2 = pd.read_csv(cpu_data_path, sep=';')

gpu_col = df1['VP_FLUX_DISTRIBUABLES']
cpu_col = df2['VP_FLUX_DISTRIBUABLES']

# Compare distributions
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Histograms
axes[0, 0].hist(gpu_col.dropna(), bins=50, alpha=0.7, label='GPU', edgecolor='black')
axes[0, 0].hist(cpu_col.dropna(), bins=50, alpha=0.7, label='CPU', edgecolor='black')
axes[0, 0].set_xlabel('VP_FLUX_DISTRIBUABLES')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Overlaid Histograms')
axes[0, 0].legend()

# Box plots
axes[0, 1].boxplot([gpu_col.dropna(), cpu_col.dropna()], labels=['GPU', 'CPU'])
axes[0, 1].set_ylabel('VP_FLUX_DISTRIBUABLES')
axes[0, 1].set_title('Box Plots')

# Density plots
gpu_col.dropna().plot(kind='density', ax=axes[1, 0], label='GPU')
cpu_col.dropna().plot(kind='density', ax=axes[1, 0], label='CPU')
axes[1, 0].set_xlabel('VP_FLUX_DISTRIBUABLES')
axes[1, 0].set_title('Density Plots')
axes[1, 0].legend()

# Q-Q plot (quantile comparison)
gpu_sorted = gpu_col.dropna().sort_values().reset_index(drop=True)
cpu_sorted = cpu_col.dropna().sort_values().reset_index(drop=True)
min_len = min(len(gpu_sorted), len(cpu_sorted))
axes[1, 1].scatter(gpu_sorted[:min_len], cpu_sorted[:min_len], alpha=0.5)
axes[1, 1].plot([gpu_sorted.min(), gpu_sorted.max()],
                [gpu_sorted.min(), gpu_sorted.max()], 'r--', label='y=x')
axes[1, 1].set_xlabel('GPU Quantiles')
axes[1, 1].set_ylabel('CPU Quantiles')
axes[1, 1].set_title('Q-Q Plot')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig(HERE.joinpath('test/distribution_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"Plot saved to {HERE.joinpath('test/distribution_comparison.png')}")
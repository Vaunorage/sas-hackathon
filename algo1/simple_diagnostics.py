import pandas as pd
import numpy as np

from paths import HERE

print("=" * 80)
print("SIMPLE DIAGNOSTIC - FINDING THE DIVERGENCE POINT")
print("=" * 80)

# Load results
gpu_df = pd.read_csv(HERE.joinpath('test/gpu_results_complete.csv'))
cpu_df = pd.read_csv(HERE.joinpath('test/acfc_results_fixed.csv'))

# Merge
merged = pd.merge(gpu_df, cpu_df, on=['ID_COMPTE', 'scn_eval'], suffixes=('_gpu', '_cpu'))
merged['diff'] = merged['VP_FLUX_DISTRIBUABLES_gpu'] - merged['VP_FLUX_DISTRIBUABLES_cpu']
merged['abs_diff'] = abs(merged['diff'])

# Categorize accounts
threshold = 0.01
perfect = []
drift = []

for account_id in sorted(merged['ID_COMPTE'].unique()):
    account_data = merged[merged['ID_COMPTE'] == account_id]
    if account_data['abs_diff'].mean() < threshold:
        perfect.append(int(account_id))
    else:
        drift.append(int(account_id))

print(f"\n✓ Perfect matches (< {threshold}): {perfect}")
print(f"✗ Has drift (>= {threshold}): {drift}")

# Load population data
population = pd.read_csv('data_in/population_fixed.csv').head(10)

# Check for FREQ_RESET_DECES pattern
print("\n" + "=" * 80)
print("CHECKING DEATH BENEFIT RESET PATTERN")
print("=" * 80)

print(f"\n{'Account':<10} {'Status':<10} {'FREQ_RESET':<12} {'MAX_RESET':<11} {'Age':<6}")
print("-" * 55)

freq_pattern = {}
for _, row in population.iterrows():
    account_id = int(row['ID_COMPTE'])
    if account_id in perfect:
        status = "PERFECT"
    elif account_id in drift:
        status = "DRIFT"
    else:
        continue

    freq = int(row['FREQ_RESET_DECES'])
    max_reset = int(row['MAX_RESET_DECES'])
    age = int(row['age_deb'])

    print(f"{account_id:<10} {status:<10} {freq:<12} {max_reset:<11} {age:<6}")

    if status not in freq_pattern:
        freq_pattern[status] = []
    freq_pattern[status].append(freq)

# Statistical test
if 'PERFECT' in freq_pattern and 'DRIFT' in freq_pattern:
    perfect_avg = np.mean(freq_pattern['PERFECT'])
    drift_avg = np.mean(freq_pattern['DRIFT'])

    print(f"\nAverage FREQ_RESET_DECES:")
    print(f"  Perfect accounts: {perfect_avg:.2f}")
    print(f"  Drift accounts: {drift_avg:.2f}")

    if abs(perfect_avg - drift_avg) > 0.1:
        print("\n" + "🎯" * 27)
        print("SMOKING GUN FOUND!")
        print("Death benefit reset (FREQ_RESET_DECES) differs between groups!")
        print("The bug is in the death benefit reset logic in gpu_calculate_internal_scenarios")
        print("🎯" * 27)

        print("\nFIX NEEDED:")
        print("In gpu_calculate_internal_scenarios, change:")
        print("  if FREQ_RESET_DECES == 1 and current_age <= MAX_RESET_DECES:")
        print("To:")
        print("  if FREQ_RESET_DECES >= 0.999 and current_age <= MAX_RESET_DECES:")
    else:
        print("\nNo clear pattern with FREQ_RESET_DECES")
        print("Need to investigate other factors...")

# Check specific differences
print("\n" + "=" * 80)
print("DETAILED DIFFERENCES FOR FIRST DRIFT ACCOUNT")
print("=" * 80)

if len(drift) > 0:
    test_account = drift[0]
    test_data = merged[merged['ID_COMPTE'] == test_account].sort_values('scn_eval')

    print(f"\nAccount {test_account} - All Scenarios:")
    print(f"\n{'Scenario':<10} {'GPU':<15} {'CPU':<15} {'Difference':<15}")
    print("-" * 55)

    for _, row in test_data.iterrows():
        print(f"{int(row['scn_eval']):<10} "
              f"{row['VP_FLUX_DISTRIBUABLES_gpu']:<15.2f} "
              f"{row['VP_FLUX_DISTRIBUABLES_cpu']:<15.2f} "
              f"{row['diff']:<15.2f}")

    # Check if all scenarios drift in same direction
    all_positive = all(test_data['diff'] > 0)
    all_negative = all(test_data['diff'] < 0)

    if all_positive:
        print("\n⚠️  ALL scenarios are biased positive (GPU > CPU)")
        print("   This suggests a systematic bias in calculations for this account")
    elif all_negative:
        print("\n⚠️  ALL scenarios are biased negative (GPU < CPU)")
        print("   This suggests a systematic bias in calculations for this account")
    else:
        print("\n⚠️  Scenarios have mixed signs")
        print("   This suggests scenario-dependent variation")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"""
Total accounts: {len(perfect) + len(drift)}
Perfect matches: {len(perfect)} accounts
Has drift: {len(drift)} accounts
Success rate: {len(perfect) / (len(perfect) + len(drift)) * 100:.1f}%

Next steps:
1. If FREQ_RESET_DECES pattern found → Apply the fix mentioned above
2. If no clear pattern → Run the step-by-step tracer (next script)
3. Consider that {len(perfect)} accounts work perfectly, so core logic is correct
""")


def test_young_vs_old_accounts():
    """
    After applying the fix, this function tests whether young and old accounts
    now have similar error rates.
    """
    import pandas as pd

    gpu_df = pd.read_csv(HERE.joinpath('test/gpu_results_complete.csv'))
    cpu_df = pd.read_csv(HERE.joinpath('test/acfc_results_fixed.csv'))
    population = pd.read_csv('data_in/population_fixed.csv').head(10)

    merged = pd.merge(gpu_df, cpu_df, on=['ID_COMPTE', 'scn_eval'], suffixes=('_gpu', '_cpu'))
    merged['abs_diff'] = abs(merged['VP_FLUX_DISTRIBUABLES_gpu'] - merged['VP_FLUX_DISTRIBUABLES_cpu'])

    # Add age information
    age_map = dict(zip(population['ID_COMPTE'], population['age_deb']))
    merged['age'] = merged['ID_COMPTE'].map(age_map)

    print("\nError Analysis by Age Group:")
    print("=" * 60)

    young = merged[merged['age'] < 30]
    middle = merged[(merged['age'] >= 30) & (merged['age'] < 50)]
    old = merged[merged['age'] >= 50]

    if len(young) > 0:
        print(f"Young accounts (age < 30): {len(young) // 10} accounts")
        print(f"  Mean error: {young['abs_diff'].mean():.4f}")
        print(f"  Max error: {young['abs_diff'].max():.4f}")

    if len(middle) > 0:
        print(f"\nMiddle accounts (30 ≤ age < 50): {len(middle) // 10} accounts")
        print(f"  Mean error: {middle['abs_diff'].mean():.4f}")
        print(f"  Max error: {middle['abs_diff'].max():.4f}")

    if len(old) > 0:
        print(f"\nOld accounts (age ≥ 50): {len(old) // 10} accounts")
        print(f"  Mean error: {old['abs_diff'].mean():.4f}")
        print(f"  Max error: {old['abs_diff'].max():.4f}")

    print("\n" + "=" * 60)

    # Check if fix worked
    if len(young) > 0 and len(old) > 0:
        ratio = young['abs_diff'].mean() / old['abs_diff'].mean()
        if ratio < 2.0:
            print("✅ FIX SUCCESSFUL! Young and old accounts have similar error rates.")
        else:
            print(f"⚠️  Young accounts still have {ratio:.1f}x more error than old accounts")
            print("   May need additional investigation")

test_young_vs_old_accounts()
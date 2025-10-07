import pandas as pd
import numpy as np
import sys


def compare_detailed_yearly_results(cpu_file: str, gpu_file: str = None, tolerance: float = 1e-6):
    """
    Compare detailed year-by-year results between CPU and GPU (or analyze CPU alone)
    """
    print("\n" + "=" * 80)
    print("DETAILED YEAR-BY-YEAR ANALYSIS")
    print("=" * 80)

    try:
        cpu_df = pd.read_csv(cpu_file)
        print(f"\n📊 CPU Data loaded: {len(cpu_df)} rows")
        print(f"Columns: {list(cpu_df.columns)}")

        # Show summary statistics
        print(f"\n📈 SUMMARY STATISTICS")
        print("-" * 80)
        for col in ['FLUX_NET_EXT', 'RESERVE', 'CAPITAL_REQUIREMENT', 'PROFIT',
                    'FLUX_DISTRIBUABLE', 'VP_FLUX_DISTRIBUABLE_YEARLY']:
            if col in cpu_df.columns:
                print(f"\n{col}:")
                print(f"  Mean: {cpu_df[col].mean():,.2f}")
                print(f"  Std:  {cpu_df[col].std():,.2f}")
                print(f"  Min:  {cpu_df[col].min():,.2f}")
                print(f"  Max:  {cpu_df[col].max():,.2f}")

        # Show by account and scenario
        print(f"\n📊 BY ACCOUNT AND SCENARIO")
        print("-" * 80)
        for account_id in sorted(cpu_df['ID_COMPTE'].unique()):
            for scenario in sorted(cpu_df[cpu_df['ID_COMPTE'] == account_id]['scn_eval'].unique()):
                subset = cpu_df[(cpu_df['ID_COMPTE'] == account_id) &
                                (cpu_df['scn_eval'] == scenario)]
                total_pv = subset['VP_FLUX_DISTRIBUABLE_YEARLY'].sum()
                print(f"\nAccount {account_id}, Scenario {scenario}:")
                print(f"  Years: {len(subset)}")
                print(f"  Total VP_FLUX_DISTRIBUABLE: {total_pv:,.2f}")

                # Show first few years
                print(f"  First 3 years:")
                for _, row in subset.head(3).iterrows():
                    print(f"    Year {int(row['an_proj'])}: "
                          f"ExtCF={row['FLUX_NET_EXT']:>10,.2f}, "
                          f"Reserve={row['RESERVE']:>10,.2f}, "
                          f"Capital={row['CAPITAL_REQUIREMENT']:>10,.2f}, "
                          f"PV={row['VP_FLUX_DISTRIBUABLE_YEARLY']:>10,.2f}")

        # If GPU file provided, compare
        if gpu_file:
            print(f"\n{'=' * 80}")
            print("COMPARING WITH GPU RESULTS")
            print(f"{'=' * 80}")

            gpu_df = pd.read_csv(gpu_file)
            print(f"\n📊 GPU Data loaded: {len(gpu_df)} rows")

            # Merge on key columns
            merge_cols = ['ID_COMPTE', 'scn_eval', 'an_proj']
            merged = pd.merge(
                cpu_df, gpu_df,
                on=merge_cols,
                suffixes=('_cpu', '_gpu'),
                how='outer',
                indicator=True
            )

            print(f"\n🔗 MERGE RESULTS")
            print("-" * 80)
            print(f"Both: {len(merged[merged['_merge'] == 'both'])}")
            print(f"CPU only: {len(merged[merged['_merge'] == 'left_only'])}")
            print(f"GPU only: {len(merged[merged['_merge'] == 'right_only'])}")

            # Compare each field
            matched = merged[merged['_merge'] == 'both'].copy()

            if len(matched) > 0:
                print(f"\n📊 FIELD-BY-FIELD COMPARISON")
                print("-" * 80)

                fields = ['FLUX_NET_EXT', 'RESERVE', 'CAPITAL_REQUIREMENT', 'PROFIT',
                          'FLUX_DISTRIBUABLE', 'VP_FLUX_DISTRIBUABLE_YEARLY']

                for field in fields:
                    cpu_col = f"{field}_cpu"
                    gpu_col = f"{field}_gpu"

                    if cpu_col in matched.columns and gpu_col in matched.columns:
                        matched[f'{field}_diff'] = np.abs(matched[cpu_col] - matched[gpu_col])
                        matched[f'{field}_rel_diff'] = np.where(
                            matched[cpu_col] != 0,
                            np.abs((matched[cpu_col] - matched[gpu_col]) / matched[cpu_col] * 100),
                            0
                        )

                        max_diff = matched[f'{field}_diff'].max()
                        mean_diff = matched[f'{field}_diff'].mean()
                        max_rel_diff = matched[f'{field}_rel_diff'].max()

                        within_tol = (matched[f'{field}_diff'] < tolerance).sum()
                        pct_within = within_tol / len(matched) * 100

                        status = "✅" if max_diff < tolerance else "⚠️"

                        print(f"\n{status} {field}:")
                        print(f"  Max absolute diff: {max_diff:.6f}")
                        print(f"  Mean absolute diff: {mean_diff:.6f}")
                        print(f"  Max relative diff: {max_rel_diff:.4f}%")
                        print(f"  Within tolerance: {within_tol}/{len(matched)} ({pct_within:.1f}%)")

                        if max_diff >= tolerance:
                            # Show worst cases
                            worst = matched.nlargest(3, f'{field}_diff')
                            print(f"  Worst cases:")
                            for idx, row in worst.iterrows():
                                print(f"    Account {int(row['ID_COMPTE'])}, "
                                      f"Scenario {int(row['scn_eval'])}, "
                                      f"Year {int(row['an_proj'])}: "
                                      f"CPU={row[cpu_col]:.2f}, "
                                      f"GPU={row[gpu_col]:.2f}, "
                                      f"Diff={row[f'{field}_diff']:.2f}")

                # Save comparison
                comparison_file = 'test/detailed_comparison.csv'
                matched.to_csv(comparison_file, index=False)
                print(f"\n✓ Detailed comparison saved to: {comparison_file}")

    except FileNotFoundError as e:
        print(f"\n❌ Error: File not found - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def compare_csv_results(gpu_file: str, cpu_file: str, tolerance: float = 1e-6):
    """
    Compare GPU and CPU CSV results with detailed diagnostics
    """
    print("\n" + "=" * 80)
    print("GPU vs CPU RESULTS COMPARISON")
    print("=" * 80)

    try:
        gpu_df = pd.read_csv(gpu_file)
        cpu_df = pd.read_csv(cpu_file)

        print(f"\n📊 BASIC STATISTICS")
        print("-" * 80)
        print(f"GPU Results: {len(gpu_df)} rows")
        print(f"CPU Results: {len(cpu_df)} rows")

        if len(gpu_df) != len(cpu_df):
            print(f"\n⚠️  WARNING: Different number of rows!")
            print(f"   GPU: {len(gpu_df)}, CPU: {len(cpu_df)}")

        # Check columns
        print(f"\n📋 COLUMNS")
        print("-" * 80)
        gpu_cols = set(gpu_df.columns)
        cpu_cols = set(cpu_df.columns)

        common_cols = gpu_cols & cpu_cols
        gpu_only = gpu_cols - cpu_cols
        cpu_only = cpu_cols - gpu_cols

        print(f"Common columns: {sorted(common_cols)}")
        if gpu_only:
            print(f"GPU only: {sorted(gpu_only)}")
        if cpu_only:
            print(f"CPU only: {sorted(cpu_only)}")

        # Merge on key columns
        merge_cols = ['ID_COMPTE', 'scn_eval']
        if all(col in common_cols for col in merge_cols):
            merged = pd.merge(
                gpu_df, cpu_df,
                on=merge_cols,
                suffixes=('_gpu', '_cpu'),
                how='outer',
                indicator=True
            )

            print(f"\n🔗 MERGE RESULTS")
            print("-" * 80)
            print(f"Both: {len(merged[merged['_merge'] == 'both'])}")
            print(f"GPU only: {len(merged[merged['_merge'] == 'left_only'])}")
            print(f"CPU only: {len(merged[merged['_merge'] == 'right_only'])}")

            # Compare values
            if 'VP_FLUX_DISTRIBUABLES_gpu' in merged.columns and 'VP_FLUX_DISTRIBUABLES_cpu' in merged.columns:
                matched = merged[merged['_merge'] == 'both'].copy()

                if len(matched) > 0:
                    matched['diff'] = matched['VP_FLUX_DISTRIBUABLES_gpu'] - matched['VP_FLUX_DISTRIBUABLES_cpu']
                    matched['abs_diff'] = np.abs(matched['diff'])
                    matched['rel_diff'] = np.abs(matched['diff'] / matched['VP_FLUX_DISTRIBUABLES_gpu']) * 100

                    print(f"\n📈 VP_FLUX_DISTRIBUABLES COMPARISON")
                    print("-" * 80)
                    print(f"GPU Mean: {matched['VP_FLUX_DISTRIBUABLES_gpu'].mean():,.2f}")
                    print(f"CPU Mean: {matched['VP_FLUX_DISTRIBUABLES_cpu'].mean():,.2f}")
                    print(f"\nDifference Statistics:")
                    print(f"  Mean absolute difference: {matched['abs_diff'].mean():,.6f}")
                    print(f"  Max absolute difference: {matched['abs_diff'].max():,.6f}")
                    print(f"  Mean relative difference: {matched['rel_diff'].mean():.4f}%")
                    print(f"  Max relative difference: {matched['rel_diff'].max():.4f}%")

                    # Check tolerance
                    within_tolerance = (matched['abs_diff'] < tolerance).sum()
                    print(f"\nWithin tolerance ({tolerance}): {within_tolerance}/{len(matched)} "
                          f"({within_tolerance / len(matched) * 100:.2f}%)")

                    # Show largest differences
                    if matched['abs_diff'].max() > tolerance:
                        print(f"\n⚠️  LARGEST DIFFERENCES (Top 5):")
                        print("-" * 80)
                        top_diffs = matched.nlargest(5, 'abs_diff')
                        for idx, row in top_diffs.iterrows():
                            print(f"\nAccount {int(row['ID_COMPTE'])}, Scenario {int(row['scn_eval'])}:")
                            print(f"  GPU: {row['VP_FLUX_DISTRIBUABLES_gpu']:,.2f}")
                            print(f"  CPU: {row['VP_FLUX_DISTRIBUABLES_cpu']:,.2f}")
                            print(f"  Difference: {row['diff']:,.2f} ({row['rel_diff']:.4f}%)")
                    else:
                        print(f"\n✅ All differences are within tolerance!")

                    # Account-by-account comparison
                    print(f"\n📊 BY ACCOUNT COMPARISON")
                    print("-" * 80)
                    for account_id in sorted(matched['ID_COMPTE'].unique()):
                        account_data = matched[matched['ID_COMPTE'] == account_id]
                        gpu_mean = account_data['VP_FLUX_DISTRIBUABLES_gpu'].mean()
                        cpu_mean = account_data['VP_FLUX_DISTRIBUABLES_cpu'].mean()
                        diff = gpu_mean - cpu_mean
                        rel_diff = abs(diff / gpu_mean * 100) if gpu_mean != 0 else 0

                        status = "✅" if abs(diff) < tolerance else "⚠️"
                        print(f"\n{status} Account {int(account_id)}:")
                        print(f"  GPU Mean: {gpu_mean:,.2f}")
                        print(f"  CPU Mean: {cpu_mean:,.2f}")
                        print(f"  Difference: {diff:,.2f} ({rel_diff:.4f}%)")
                        print(f"  Scenarios: {len(account_data)}")

        # Statistical tests
        if 'VP_FLUX_DISTRIBUABLES' in gpu_df.columns and 'VP_FLUX_DISTRIBUABLES' in cpu_df.columns:
            print(f"\n📉 DISTRIBUTION COMPARISON")
            print("-" * 80)

            gpu_values = gpu_df['VP_FLUX_DISTRIBUABLES'].values
            cpu_values = cpu_df['VP_FLUX_DISTRIBUABLES'].values

            print(f"GPU: Min={gpu_values.min():,.2f}, Max={gpu_values.max():,.2f}, "
                  f"Median={np.median(gpu_values):,.2f}, Std={gpu_values.std():,.2f}")
            print(f"CPU: Min={cpu_values.min():,.2f}, Max={cpu_values.max():,.2f}, "
                  f"Median={np.median(cpu_values):,.2f}, Std={cpu_values.std():,.2f}")

        print("\n" + "=" * 80)
        print("COMPARISON COMPLETE")
        print("=" * 80)

    except FileNotFoundError as e:
        print(f"\n❌ Error: File not found - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during comparison: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def create_detailed_comparison_report(gpu_file: str, cpu_file: str, output_file: str = 'comparison_report.csv'):
    """
    Create a detailed CSV report comparing GPU and CPU results
    """
    try:
        gpu_df = pd.read_csv(gpu_file)
        cpu_df = pd.read_csv(cpu_file)

        merged = pd.merge(
            gpu_df, cpu_df,
            on=['ID_COMPTE', 'scn_eval'],
            suffixes=('_gpu', '_cpu'),
            how='outer',
            indicator=True
        )

        if 'VP_FLUX_DISTRIBUABLES_gpu' in merged.columns and 'VP_FLUX_DISTRIBUABLES_cpu' in merged.columns:
            merged['abs_diff'] = np.abs(
                merged['VP_FLUX_DISTRIBUABLES_gpu'] - merged['VP_FLUX_DISTRIBUABLES_cpu']
            )
            merged['rel_diff_pct'] = np.abs(
                (merged['VP_FLUX_DISTRIBUABLES_gpu'] - merged['VP_FLUX_DISTRIBUABLES_cpu']) /
                merged['VP_FLUX_DISTRIBUABLES_gpu']
            ) * 100

        merged.to_csv(output_file, index=False)
        print(f"\n📄 Detailed comparison report saved to: {output_file}")

        return merged

    except Exception as e:
        print(f"\n❌ Error creating report: {e}")
        return None


def extract_gpu_detailed_results(gpu_results_array, account_ids, nb_scenarios, nb_years,
                                 reserve_results, capital_results, hurdle_rt=0.10,
                                 output_file='test/gpu_detailed_yearly.csv'):
    """
    Extract detailed year-by-year results from GPU arrays and save to CSV
    This function should be called from the GPU code after calculations

    Parameters:
    -----------
    gpu_results_array : numpy array with shape (n_combinations * (nb_years+1), 9)
        Contains: [account_id, scenario, year, age, fund_value, death_benefit, survival, flux_net, vp_flux_net]
    account_ids : array of account IDs
    nb_scenarios : number of scenarios
    nb_years : number of projection years
    reserve_results : array of reserve values
    capital_results : array of capital values
    hurdle_rt : hurdle rate for PV calculation
    output_file : path to save CSV
    """
    detailed_rows = []

    # Group results by account and scenario
    from collections import defaultdict
    grouped = defaultdict(lambda: defaultdict(list))

    for row in gpu_results_array:
        if row[0] == 0:  # Skip invalid rows
            continue
        account_id = int(row[0])
        scenario = int(row[1])
        year = int(row[2])
        flux_net = row[7]

        grouped[account_id][scenario].append({
            'year': year,
            'flux_net': flux_net
        })

    # Process each account/scenario combination
    idx = 0
    for account_id in sorted(grouped.keys()):
        for scenario in sorted(grouped[account_id].keys()):
            years_data = sorted(grouped[account_id][scenario], key=lambda x: x['year'])

            prev_reserve = 0.0
            prev_capital = 0.0

            for year_data in years_data:
                year = year_data['year']
                flux_net_ext = year_data['flux_net']

                # Get reserve and capital for this combination and year
                # Index calculation: combination_idx * (nb_years+1) + year
                result_idx = idx * (nb_years + 1) + year

                if result_idx < len(reserve_results):
                    current_reserve = reserve_results[result_idx]
                    current_capital = capital_results[result_idx] - reserve_results[result_idx]
                else:
                    current_reserve = 0.0
                    current_capital = 0.0

                if year == 0:
                    delta_reserve = current_reserve
                    delta_capital = current_capital
                    profit = flux_net_ext + current_reserve
                    distributable = profit + current_capital
                else:
                    delta_reserve = current_reserve - prev_reserve
                    delta_capital = current_capital - prev_capital
                    profit = flux_net_ext + delta_reserve
                    distributable = profit + delta_capital

                if year > 0:
                    pv_distributable = distributable / ((1 + hurdle_rt) ** year)
                else:
                    pv_distributable = distributable

                detailed_rows.append({
                    'ID_COMPTE': account_id,
                    'scn_eval': scenario,
                    'an_proj': year,
                    'FLUX_NET_EXT': flux_net_ext,
                    'RESERVE': current_reserve,
                    'CAPITAL_REQUIREMENT': current_capital,
                    'DELTA_RESERVE': delta_reserve,
                    'DELTA_CAPITAL': delta_capital,
                    'PROFIT': profit,
                    'FLUX_DISTRIBUABLE': distributable,
                    'VP_FLUX_DISTRIBUABLE_YEARLY': pv_distributable
                })

                prev_reserve = current_reserve
                prev_capital = current_capital

            idx += 1

    # Save to CSV
    df = pd.DataFrame(detailed_rows)
    df.to_csv(output_file, index=False)
    print(f"✓ GPU detailed results saved to: {output_file}")

    return df


def create_detailed_comparison_report(gpu_file: str, cpu_file: str, output_file: str = 'comparison_report.csv'):
    """
    Create a detailed CSV report comparing GPU and CPU results
    """
    try:
        gpu_df = pd.read_csv(gpu_file)
        cpu_df = pd.read_csv(cpu_file)

        merged = pd.merge(
            gpu_df, cpu_df,
            on=['ID_COMPTE', 'scn_eval'],
            suffixes=('_gpu', '_cpu'),
            how='outer',
            indicator=True
        )

        if 'VP_FLUX_DISTRIBUABLES_gpu' in merged.columns and 'VP_FLUX_DISTRIBUABLES_cpu' in merged.columns:
            merged['abs_diff'] = np.abs(
                merged['VP_FLUX_DISTRIBUABLES_gpu'] - merged['VP_FLUX_DISTRIBUABLES_cpu']
            )
            merged['rel_diff_pct'] = np.abs(
                (merged['VP_FLUX_DISTRIBUABLES_gpu'] - merged['VP_FLUX_DISTRIBUABLES_cpu']) /
                merged['VP_FLUX_DISTRIBUABLES_gpu']
            ) * 100

        merged.to_csv(output_file, index=False)
        print(f"\n📄 Detailed comparison report saved to: {output_file}")

        return merged

    except Exception as e:
        print(f"\n❌ Error creating report: {e}")
        return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Compare GPU and CPU results')
    parser.add_argument('--mode', choices=['summary', 'detailed', 'both'], default='both',
                        help='Comparison mode: summary, detailed, or both')
    parser.add_argument('--tolerance', type=float, default=1e-2,
                        help='Tolerance for numerical comparisons')

    args = parser.parse_args()

    if args.mode in ['summary', 'both']:
        # Compare summary results
        gpu_file = 'test/gpu_results_complete.csv'
        cpu_file = 'test/cpu_results_complete.csv'

        print("=" * 80)
        print("COMPARING SUMMARY RESULTS")
        print("=" * 80)
        compare_csv_results(gpu_file, cpu_file, tolerance=args.tolerance)

        print("\nCreating detailed summary report...")
        report = create_detailed_comparison_report(gpu_file, cpu_file, 'test/comparison_report.csv')

    if args.mode in ['detailed', 'both']:
        # Compare detailed yearly results
        cpu_detailed = 'test/cpu_detailed_yearly.csv'
        gpu_detailed = 'test/gpu_detailed_yearly.csv'  # If you create one from GPU

        print("\n" + "=" * 80)
        print("COMPARING DETAILED YEAR-BY-YEAR RESULTS")
        print("=" * 80)

        # Check if GPU detailed file exists
        import os

        if os.path.exists(gpu_detailed):
            compare_detailed_yearly_results(cpu_detailed, gpu_detailed, tolerance=args.tolerance)
        else:
            print(f"\n⚠️  GPU detailed file not found: {gpu_detailed}")
            print("Analyzing CPU detailed results only...")
            compare_detailed_yearly_results(cpu_detailed, gpu_file=None, tolerance=args.tolerance)

    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)
    print("\nGenerated files:")
    print("  1. test/comparison_report.csv - Summary comparison")
    print("  2. test/detailed_comparison.csv - Year-by-year comparison (if GPU data available)")
    print("\nTo view specific differences, sort CSVs by diff columns")
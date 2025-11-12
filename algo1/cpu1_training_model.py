import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
import json
import time
from datetime import datetime

from paths import HERE

warnings.filterwarnings('ignore')

# Initialize library availability flags
HAS_XGB = False
HAS_LGB = False
HAS_CATBOOST = False
HAS_TENSORFLOW = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    print("XGBoost not installed, skipping...")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("LightGBM not installed, skipping...")

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("CatBoost not installed, skipping...")

print("=" * 80)
print("ACFC Reserve Prediction - Tree-Based & Neural Network Models")
print("=" * 80)

# Create output directory for results
output_dir = HERE.joinpath('model_results')
output_dir.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Load the data
print("[1/8] Loading data...")
df = pd.read_csv(HERE.joinpath('test/training_data/acfc_training_samples.csv'))
print(f"✓ Dataset loaded successfully")
print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(
    f"  ✓ No missing values detected" if df.isnull().sum().sum() == 0 else f"  ⚠ Missing values: {df.isnull().sum().sum()}")

# Feature engineering
print("[2/8] Engineering features...")
df['guarantee_gap'] = df['death_benefit'] - df['fund_value']
df['value_ratio'] = df['fund_value'] / df['initial_fund']
df['death_benefit_ratio'] = df['death_benefit'] / df['initial_death_benefit']
df['age_squared'] = df['age'] ** 2
df['survival_prob_squared'] = df['survival_prob'] ** 2
df['total_fees'] = df['fees_acquisition'] + df['fees_admin']
df['revenue_impact'] = df['fund_value'] * df['pct_revenue_fund']
df['mgmt_impact'] = df['fund_value'] * df['pct_mgmt_fees']

# Interaction features
df['age_survival_interaction'] = df['age'] * df['survival_prob']
df['fund_age_interaction'] = df['fund_value'] * df['age_from_start']
df['guarantee_ratio_age'] = df['fund_to_guarantee_ratio'] * df['age']

print(f"✓ Created 11 engineered features (including 3 interaction terms)")

# Select features for modeling
feature_cols = [
    'year', 'fund_value', 'death_benefit', 'survival_prob', 'age',
    'initial_age', 'initial_fund', 'initial_death_benefit',
    'fees_acquisition', 'fees_admin', 'pct_revenue_fund', 'pct_mgmt_fees',
    'commission_sale', 'commission_maintenance', 'reset_frequency',
    'max_reset_age', 'fund_to_guarantee_ratio', 'age_from_start',
    'fund_growth', 'guarantee_gap', 'value_ratio', 'death_benefit_ratio',
    'age_squared', 'survival_prob_squared', 'total_fees',
    'revenue_impact', 'mgmt_impact', 'age_survival_interaction',
    'fund_age_interaction', 'guarantee_ratio_age'
]

target_cols = ['reserve_value', 'capital_value']

# Prepare data
print("[3/8] Preparing data...")
X = df[feature_cols].copy()
y = df[target_cols].copy()

print(f"  Features: {len(feature_cols)}")
print(f"  Targets: {len(target_cols)}")

# Split data with reset_index to avoid index issues
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# CRITICAL: Reset indices to avoid the error
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

print(f"✓ Data split complete")
print(f"  Training set: {X_train.shape[0]:,} samples")
print(f"  Test set: {X_test.shape[0]:,} samples")

# Scale features
print("[4/8] Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame to preserve column names
X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols)

print(f"✓ Features scaled using StandardScaler")

# Define models (Tree-based only)
print("[5/8] Defining tree-based models to test...")
models_config = {
    # Tree-based ensemble models
    'Random Forest': RandomForestRegressor(
        n_estimators=200, max_depth=15, min_samples_split=5,
        min_samples_leaf=2, random_state=42, n_jobs=-1
    ),
    'Extra Trees': ExtraTreesRegressor(
        n_estimators=200, max_depth=15, min_samples_split=5,
        min_samples_leaf=2, random_state=42, n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
    ),
    'AdaBoost': AdaBoostRegressor(n_estimators=100, random_state=42),
    'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42)
}

if HAS_XGB:
    models_config['XGBoost'] = xgb.XGBRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        random_state=42, n_jobs=-1
    )

if HAS_LGB:
    models_config['LightGBM'] = lgb.LGBMRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        random_state=42, n_jobs=-1, verbose=1
    )

if HAS_CATBOOST:
    models_config['CatBoost'] = CatBoostRegressor(
        iterations=200, depth=5, learning_rate=0.1,
        random_state=42, verbose=50  # Print every 50 iterations
    )

print(f"✓ Configured {len(models_config)} tree-based algorithms")
for name in models_config.keys():
    print(f"  • {name}")

# Define Neural Network architectures
def create_simple_nn(input_dim):
    """Simple 2-layer neural network"""
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss='mse', metrics=['mae'])
    return model

def create_deep_nn(input_dim):
    """Deep 4-layer neural network"""
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss='mse', metrics=['mae'])
    return model

def create_dropout_nn(input_dim):
    """Neural network with dropout for regularization"""
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(1)
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss='mse', metrics=['mae'])
    return model

def create_wide_deep_nn(input_dim):
    """Wide and deep architecture"""
    model = keras.Sequential([
        layers.Dense(256, activation='relu', input_shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss='mse', metrics=['mae'])
    return model

# Store NN configs separately (they need special handling)
nn_configs = {}
if HAS_TENSORFLOW:
    nn_configs = {
        'NN_Simple': create_simple_nn,
        'NN_Deep': create_deep_nn,
        'NN_Dropout': create_dropout_nn,
        'NN_Wide_Deep': create_wide_deep_nn
    }
    print(f"\n✓ Configured {len(nn_configs)} neural network architectures")
    for name in nn_configs.keys():
        print(f"  • {name}")

# Train and evaluate models
all_results = {}
all_metrics = []  # For CSV export

for target_idx, target in enumerate(target_cols):
    print("\n" + "=" * 80)
    print(f"Target {target_idx + 1}/{len(target_cols)}: {target}")
    print("=" * 80)
    print(f"[6/8] Training and evaluating models for {target}...")

    results = {}
    predictions_df = pd.DataFrame()
    predictions_df['actual'] = y_test[target].values

    for name, model in models_config.items():
        try:
            # Print training start
            print(f"\n  Training {name}...")
            start_time = time.time()
            
            # Train
            model.fit(X_train_scaled, y_train[target])
            
            train_time = time.time() - start_time

            # Predict
            y_pred = model.predict(X_test_scaled)

            # Metrics
            mse = mean_squared_error(y_test[target], y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test[target], y_pred)
            r2 = r2_score(y_test[target], y_pred)
            
            # MAPE (Mean Absolute Percentage Error)
            mape = np.mean(np.abs((y_test[target] - y_pred) / y_test[target])) * 100
            
            # Max error
            max_error = np.max(np.abs(y_test[target] - y_pred))

            results[name] = {
                'model': model,
                'predictions': y_pred,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'MAPE': mape,
                'MSE': mse,
                'Max_Error': max_error
            }
            
            # Store predictions
            predictions_df[f'{name}_pred'] = y_pred
            predictions_df[f'{name}_error'] = y_test[target].values - y_pred

            # Store metrics for CSV
            all_metrics.append({
                'Target': target,
                'Model': name,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'MSE': mse,
                'MAPE': mape,
                'Max_Error': max_error,
                'Train_Samples': len(X_train),
                'Test_Samples': len(X_test),
                'Epochs': None,  # Not applicable for non-NN models
                'Training_Time_Seconds': train_time
            })

            print(f"  ✓ {name:20s} - RMSE: {rmse:10.4f} | MAE: {mae:10.4f} | R²: {r2:7.4f} | Time: {train_time:6.2f}s")

        except Exception as e:
            print(f"  ⚠ Error with {name}: {str(e)[:100]}")

    # Train Neural Networks (if available)
    if HAS_TENSORFLOW and nn_configs:
        print(f"\n  Training Neural Networks...")
        
        # Early stopping callback
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1  # Print when early stopping triggers
        )
        
        for nn_name, nn_builder in nn_configs.items():
            try:
                # Print training start
                print(f"\n  Training {nn_name}...")
                
                # Create fresh model for this target
                nn_model = nn_builder(X_train_scaled.shape[1])
                
                # Print model architecture
                print(f"  Architecture: {nn_model.count_params():,} parameters")
                
                start_time = time.time()
                
                # Train with validation split
                history = nn_model.fit(
                    X_train_scaled, y_train[target],
                    epochs=200,
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=[early_stop],
                    verbose=1  # Show progress bar
                )
                
                train_time = time.time() - start_time
                
                # Predict
                y_pred = nn_model.predict(X_test_scaled, verbose=0).flatten()
                
                # Metrics
                mse = mean_squared_error(y_test[target], y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test[target], y_pred)
                r2 = r2_score(y_test[target], y_pred)
                
                # MAPE (Mean Absolute Percentage Error)
                mape = np.mean(np.abs((y_test[target] - y_pred) / y_test[target])) * 100
                
                # Max error
                max_error = np.max(np.abs(y_test[target] - y_pred))
                
                results[nn_name] = {
                    'model': nn_model,
                    'predictions': y_pred,
                    'RMSE': rmse,
                    'MAE': mae,
                    'R2': r2,
                    'MAPE': mape,
                    'MSE': mse,
                    'Max_Error': max_error,
                    'epochs_trained': len(history.history['loss'])
                }
                
                # Store predictions
                predictions_df[f'{nn_name}_pred'] = y_pred
                predictions_df[f'{nn_name}_error'] = y_test[target].values - y_pred
                
                # Store metrics for CSV
                all_metrics.append({
                    'Target': target,
                    'Model': nn_name,
                    'RMSE': rmse,
                    'MAE': mae,
                    'R2': r2,
                    'MSE': mse,
                    'MAPE': mape,
                    'Max_Error': max_error,
                    'Train_Samples': len(X_train),
                    'Test_Samples': len(X_test),
                    'Epochs': len(history.history['loss']),
                    'Training_Time_Seconds': train_time
                })
                
                print(f"  ✓ {nn_name:20s} - RMSE: {rmse:10.4f} | MAE: {mae:10.4f} | R²: {r2:7.4f} | Epochs: {len(history.history['loss'])} | Time: {train_time:6.2f}s")
                
            except Exception as e:
                print(f"  ⚠ Error with {nn_name}: {str(e)[:100]}")

    all_results[target] = results
    
    # Save predictions for this target
    predictions_file = output_dir.joinpath(f'predictions_{target}_{timestamp}.csv')
    predictions_df.to_csv(predictions_file, index=False)
    print(f"\n✓ Predictions saved to: {predictions_file}")

# Save all metrics to CSV
print("\n" + "=" * 80)
print("[7/8] Saving evaluation results...")
print("=" * 80)

metrics_df = pd.DataFrame(all_metrics)
metrics_file = output_dir.joinpath(f'model_metrics_{timestamp}.csv')
metrics_df.to_csv(metrics_file, index=False)
print(f"✓ Model metrics saved to: {metrics_file}")

# Save detailed summary statistics
summary_stats = []
for target in target_cols:
    if target in all_results and all_results[target]:
        for model_name, metrics in all_results[target].items():
            summary_stats.append({
                'Target': target,
                'Model': model_name,
                'RMSE': metrics['RMSE'],
                'MAE': metrics['MAE'],
                'R2': metrics['R2'],
                'MAPE': metrics['MAPE'],
                'MSE': metrics['MSE'],
                'Max_Error': metrics['Max_Error']
            })

summary_df = pd.DataFrame(summary_stats)
summary_file = output_dir.joinpath(f'model_summary_{timestamp}.csv')
summary_df.to_csv(summary_file, index=False)
print(f"✓ Summary statistics saved to: {summary_file}")

# Summary
print("\n" + "=" * 80)
print("MODEL PERFORMANCE SUMMARY")
print("=" * 80)

for target in target_cols:
    print(f"\n{target}:")
    print("-" * 80)

    if target in all_results and all_results[target]:
        # Sort by R2 score
        sorted_models = sorted(
            all_results[target].items(),
            key=lambda x: x[1]['R2'],
            reverse=True
        )

        print(f"{'Rank':<6} {'Model':<20} {'RMSE':<12} {'MAE':<12} {'R²':<10} {'MAPE':<10}")
        print("-" * 80)

        for rank, (name, metrics) in enumerate(sorted_models, 1):
            print(f"{rank:<6} {name:<20} {metrics['RMSE']:<12.4f} {metrics['MAE']:<12.4f} {metrics['R2']:<10.4f} {metrics['MAPE']:<10.2f}%")

        # Best model
        best_model_name = sorted_models[0][0]
        best_metrics = sorted_models[0][1]
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"   RMSE: {best_metrics['RMSE']:.4f}")
        print(f"   MAE: {best_metrics['MAE']:.4f}")
        print(f"   R²: {best_metrics['R2']:.4f}")
        print(f"   MAPE: {best_metrics['MAPE']:.2f}%")

# Feature importance from best tree-based model
print("\n" + "=" * 80)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 80)

all_feature_importance = []

for target in target_cols:
    if target in all_results and all_results[target]:
        # Try to get feature importance from tree-based models
        for model_name in ['Random Forest', 'XGBoost', 'LightGBM', 'Extra Trees', 'Gradient Boosting']:
            if model_name in all_results[target]:
                model = all_results[target][model_name]['model']
                if hasattr(model, 'feature_importances_'):
                    importance_df = pd.DataFrame({
                        'target': target,
                        'model': model_name,
                        'feature': feature_cols,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    all_feature_importance.append(importance_df)

                    print(f"\n{target} - Top 15 Features ({model_name}):")
                    print("-" * 60)
                    for idx, row in importance_df.head(15).iterrows():
                        print(f"  {row['feature']:<35} {row['importance']:>8.4f}")
                    break

# Save feature importance
if all_feature_importance:
    feature_importance_df = pd.concat(all_feature_importance, ignore_index=True)
    importance_file = output_dir.joinpath(f'feature_importance_{timestamp}.csv')
    feature_importance_df.to_csv(importance_file, index=False)
    print(f"\n✓ Feature importance saved to: {importance_file}")

# Create ensemble predictions using top 3 models
print("\n" + "=" * 80)
print("CREATING ENSEMBLE PREDICTIONS")
print("=" * 80)

ensemble_results = []
ensemble_predictions = {}

for target in target_cols:
    if target in all_results and all_results[target]:
        # Get top 3 models by R2
        sorted_models = sorted(
            all_results[target].items(),
            key=lambda x: x[1]['R2'],
            reverse=True
        )[:3]

        # Average their predictions
        predictions_list = [m[1]['predictions'] for m in sorted_models]
        ensemble_pred = np.mean(predictions_list, axis=0)

        # Calculate ensemble metrics
        rmse = np.sqrt(mean_squared_error(y_test[target], ensemble_pred))
        mae = mean_absolute_error(y_test[target], ensemble_pred)
        r2 = r2_score(y_test[target], ensemble_pred)
        mape = np.mean(np.abs((y_test[target] - ensemble_pred) / y_test[target])) * 100

        ensemble_predictions[target] = ensemble_pred
        
        ensemble_results.append({
            'Target': target,
            'Models': ', '.join([m[0] for m in sorted_models]),
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape
        })

        print(f"\n{target} Ensemble (Top 3 Models):")
        print(f"  Models: {', '.join([m[0] for m in sorted_models])}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.4f}")
        print(f"  MAPE: {mape:.2f}%")

# Save ensemble results
ensemble_df = pd.DataFrame(ensemble_results)
ensemble_file = output_dir.joinpath(f'ensemble_results_{timestamp}.csv')
ensemble_df.to_csv(ensemble_file, index=False)
print(f"\n✓ Ensemble results saved to: {ensemble_file}")

# Save ensemble predictions
for target in target_cols:
    if target in ensemble_predictions:
        ensemble_pred_df = pd.DataFrame({
            'actual': y_test[target].values,
            'ensemble_prediction': ensemble_predictions[target],
            'error': y_test[target].values - ensemble_predictions[target]
        })
        ensemble_pred_file = output_dir.joinpath(f'ensemble_predictions_{target}_{timestamp}.csv')
        ensemble_pred_df.to_csv(ensemble_pred_file, index=False)

print(f"✓ Ensemble predictions saved")

# Visualization
print("\n" + "=" * 80)
print("[8/8] Creating visualizations...")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

for idx, target in enumerate(target_cols):
    if target in ensemble_predictions:
        # Actual vs Predicted
        ax = axes[idx, 0]
        ax.scatter(y_test[target], ensemble_predictions[target], alpha=0.3, s=10)
        min_val = min(y_test[target].min(), ensemble_predictions[target].min())
        max_val = max(y_test[target].max(), ensemble_predictions[target].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        ax.set_xlabel('Actual', fontsize=12)
        ax.set_ylabel('Predicted', fontsize=12)
        ax.set_title(f'{target} - Actual vs Predicted (Ensemble)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Residuals
        ax = axes[idx, 1]
        residuals = y_test[target] - ensemble_predictions[target]
        ax.scatter(ensemble_predictions[target], residuals, alpha=0.3, s=10)
        ax.axhline(y=0, color='r', linestyle='--', lw=2)
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Residuals', fontsize=12)
        ax.set_title(f'{target} - Residual Plot (Ensemble)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_file = output_dir.joinpath(f'model_performance_{timestamp}.png')
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"✓ Performance plots saved to: {plot_file}")

# Create a master results file with all information
print("\n" + "=" * 80)
print("CREATING MASTER RESULTS FILE")
print("=" * 80)

master_results = {
    'timestamp': timestamp,
    'dataset_info': {
        'total_samples': len(df),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'num_features': len(feature_cols),
        'num_targets': len(target_cols)
    },
    'models_tested': list(models_config.keys()) + list(nn_configs.keys()) if HAS_TENSORFLOW else list(models_config.keys()),
    'best_models': {},
    'neural_networks_used': HAS_TENSORFLOW
}

for target in target_cols:
    if target in all_results and all_results[target]:
        sorted_models = sorted(
            all_results[target].items(),
            key=lambda x: x[1]['R2'],
            reverse=True
        )
        best = sorted_models[0]
        master_results['best_models'][target] = {
            'model_name': best[0],
            'RMSE': float(best[1]['RMSE']),
            'MAE': float(best[1]['MAE']),
            'R2': float(best[1]['R2']),
            'MAPE': float(best[1]['MAPE'])
        }

master_file = output_dir.joinpath(f'master_results_{timestamp}.json')
with open(master_file, 'w') as f:
    json.dump(master_results, f, indent=2)
print(f"✓ Master results saved to: {master_file}")

print("\n" + "=" * 80)
print("✓ Training complete!")
print("=" * 80)
total_models = len(models_config) + (len(nn_configs) if HAS_TENSORFLOW else 0)
print(f"\nTrained {total_models} models:")
print(f"  • {len(models_config)} tree-based algorithms")
if HAS_TENSORFLOW:
    print(f"  • {len(nn_configs)} neural network architectures")
else:
    print(f"  • 0 neural networks (TensorFlow not installed)")
print(f"\nAll results saved in: {output_dir}")
print(f"\nFiles created:")
print(f"  • model_metrics_{timestamp}.csv - All model metrics")
print(f"  • model_summary_{timestamp}.csv - Summary statistics")
print(f"  • ensemble_results_{timestamp}.csv - Ensemble model results")
print(f"  • feature_importance_{timestamp}.csv - Feature importance analysis")
print(f"  • predictions_*_{timestamp}.csv - Predictions for each target")
print(f"  • ensemble_predictions_*_{timestamp}.csv - Ensemble predictions")
print(f"  • model_performance_{timestamp}.png - Visualization plots")
print(f"  • master_results_{timestamp}.json - Master results summary")
print("=" * 80)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import warnings

from paths import HERE

warnings.filterwarnings('ignore')

try:
    import xgboost as xgb

    HAS_XGB = True
except ImportError:
    HAS_XGB = False
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
print("ACFC Reserve Prediction - Advanced Model Comparison")
print("=" * 80)

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

# Define models
print("[5/8] Defining models to test...")
models_config = {
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
    'Ridge': Ridge(alpha=1.0, random_state=42),
    'Lasso': Lasso(alpha=0.1, random_state=42, max_iter=5000),
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=5000),
    'AdaBoost': AdaBoostRegressor(n_estimators=100, random_state=42),
    'KNN': KNeighborsRegressor(n_neighbors=10, n_jobs=-1),
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
        random_state=42, n_jobs=-1, verbose=-1
    )

if HAS_CATBOOST:
    models_config['CatBoost'] = CatBoostRegressor(
        iterations=200, depth=5, learning_rate=0.1,
        random_state=42, verbose=0
    )

print(f"✓ Configured {len(models_config)} different algorithms")
for name in models_config.keys():
    print(f"  • {name}")

# Train and evaluate models
all_results = {}

for target_idx, target in enumerate(target_cols):
    print("\n" + "=" * 80)
    print(f"Target {target_idx + 1}/{len(target_cols)}: {target}")
    print("=" * 80)
    print(f"[6/6] Training and evaluating models for {target}...")

    results = {}

    for name, model in models_config.items():
        try:
            # Train
            model.fit(X_train_scaled, y_train[target])

            # Predict
            y_pred = model.predict(X_test_scaled)

            # Metrics
            mse = mean_squared_error(y_test[target], y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test[target], y_pred)
            r2 = r2_score(y_test[target], y_pred)

            results[name] = {
                'model': model,
                'predictions': y_pred,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2
            }

            print(f"  ✓ {name:20s} - RMSE: {rmse:10.4f} | MAE: {mae:10.4f} | R²: {r2:7.4f}")

        except Exception as e:
            print(f"  ⚠ Error with {name}: {str(e)[:100]}")

    all_results[target] = results

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

        print(f"{'Rank':<6} {'Model':<20} {'RMSE':<12} {'MAE':<12} {'R²':<10}")
        print("-" * 80)

        for rank, (name, metrics) in enumerate(sorted_models, 1):
            print(f"{rank:<6} {name:<20} {metrics['RMSE']:<12.4f} {metrics['MAE']:<12.4f} {metrics['R2']:<10.4f}")

        # Best model
        best_model_name = sorted_models[0][0]
        best_metrics = sorted_models[0][1]
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"   RMSE: {best_metrics['RMSE']:.4f}")
        print(f"   MAE: {best_metrics['MAE']:.4f}")
        print(f"   R²: {best_metrics['R2']:.4f}")

# Feature importance from best tree-based model
print("\n" + "=" * 80)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 80)

for target in target_cols:
    if target in all_results and all_results[target]:
        # Try to get feature importance from Random Forest or best model
        for model_name in ['Random Forest', 'XGBoost', 'LightGBM', 'Extra Trees']:
            if model_name in all_results[target]:
                model = all_results[target][model_name]['model']
                if hasattr(model, 'feature_importances_'):
                    importance_df = pd.DataFrame({
                        'feature': feature_cols,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)

                    print(f"\n{target} - Top 15 Features ({model_name}):")
                    print("-" * 60)
                    for idx, row in importance_df.head(15).iterrows():
                        print(f"  {row['feature']:<35} {row['importance']:>8.4f}")
                    break

# Create ensemble predictions using top 3 models
print("\n" + "=" * 80)
print("CREATING ENSEMBLE PREDICTIONS")
print("=" * 80)

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

        ensemble_predictions[target] = ensemble_pred

        print(f"\n{target} Ensemble (Top 3 Models):")
        print(f"  Models: {', '.join([m[0] for m in sorted_models])}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.4f}")

# Visualization
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
plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
print("\n✓ Performance plots saved as 'model_performance.png'")

print("\n" + "=" * 80)
print("✓ Training complete!")
print("=" * 80)
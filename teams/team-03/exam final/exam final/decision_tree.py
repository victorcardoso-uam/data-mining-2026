"""
Session Final - Decision Tree Model
Course: Data Mining | Universidad Anáhuac Mayab

Problem: Predict monthly Earth Skin Temperature (°C) in Chihuahua
         using solar irradiance, precipitation, and wind speed.

Target variable (y): temperature_c
Input variables (X): solar_irradiance_wm2, precipitation_mm, wind_speed_ms, MONTH_NUM
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

os.makedirs("data/processed", exist_ok=True)
os.makedirs("outputs/plots", exist_ok=True)

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

print("=" * 65)
print("DECISION TREE MODEL — TEMPERATURE PREDICTION (CHIHUAHUA)")
print("=" * 65)

df = pd.read_csv("data/processed/chihuahua_dataset.csv")
print(f"Dataset shape: {df.shape}")

# --------------------------------------------------
# 2. DEFINE FEATURES AND TARGET
# --------------------------------------------------

FEATURES = ['solar_irradiance_wm2', 'precipitation_mm', 'wind_speed_ms', 'MONTH_NUM']
TARGET = 'temperature_c'

X = df[FEATURES]
y = df[TARGET]

# --------------------------------------------------
# 3. TRAIN / TEST SPLIT (80/20) — same split for all models
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")

# --------------------------------------------------
# 4. HELPER FUNCTION FOR METRICS
# --------------------------------------------------

def evaluate(name, y_test, y_pred):
    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)
    print(f"\n  {name}")
    print(f"    R²   = {r2:.4f}")
    print(f"    MAE  = {mae:.4f}")
    print(f"    MSE  = {mse:.4f}")
    print(f"    RMSE = {rmse:.4f}")
    return {'model': name, 'R2': round(r2,4), 'MAE': round(mae,4),
            'MSE': round(mse,4), 'RMSE': round(rmse,4)}

# --------------------------------------------------
# 5. MODEL A — NO CONSTRAINTS (default parameters)
# --------------------------------------------------

print("\n--- VERSION A: Default Decision Tree (no constraints) ---")
dt_default = DecisionTreeRegressor(random_state=42)
dt_default.fit(X_train, y_train)
pred_default = dt_default.predict(X_test)
metrics_default = evaluate("Decision Tree — Default", y_test, pred_default)

print(f"  Tree depth: {dt_default.get_depth()}")
print(f"  Number of leaves: {dt_default.get_n_leaves()}")

# --------------------------------------------------
# 6. MODEL B — CONTROLLED COMPLEXITY (max_depth + min_samples_split)
# --------------------------------------------------

print("\n--- VERSION B: Pruned Decision Tree (max_depth=5, min_samples_split=10) ---")
dt_pruned = DecisionTreeRegressor(
    max_depth=5,            # limits how deep the tree grows
    min_samples_split=10,   # requires at least 10 samples to split a node
    random_state=42
)
dt_pruned.fit(X_train, y_train)
pred_pruned = dt_pruned.predict(X_test)
metrics_pruned = evaluate("Decision Tree — Pruned (max_depth=5)", y_test, pred_pruned)

print(f"  Tree depth: {dt_pruned.get_depth()}")
print(f"  Number of leaves: {dt_pruned.get_n_leaves()}")

# --------------------------------------------------
# 7. FEATURE IMPORTANCE
# --------------------------------------------------

print("\n--- Feature Importance (Pruned model) ---")
importance_df = pd.DataFrame({
    'Feature': FEATURES,
    'Importance': dt_pruned.feature_importances_
}).sort_values('Importance', ascending=False)
print(importance_df.to_string(index=False))

# --------------------------------------------------
# 8. PLOTS
# --------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Decision Tree — Model Evaluation (Chihuahua Temperature)", fontsize=14, fontweight='bold')

# Predicted vs Actual — Default
axes[0, 0].scatter(y_test, pred_default, alpha=0.6, color='steelblue', edgecolors='k', linewidths=0.3)
lims = [min(y_test.min(), pred_default.min()), max(y_test.max(), pred_default.max())]
axes[0, 0].plot(lims, lims, 'r--', lw=1.5)
axes[0, 0].set_title(f"Predicted vs Actual — Default\nR²={metrics_default['R2']:.4f}")
axes[0, 0].set_xlabel("Actual Temperature (°C)")
axes[0, 0].set_ylabel("Predicted Temperature (°C)")

# Predicted vs Actual — Pruned
axes[0, 1].scatter(y_test, pred_pruned, alpha=0.6, color='darkorange', edgecolors='k', linewidths=0.3)
axes[0, 1].plot(lims, lims, 'r--', lw=1.5)
axes[0, 1].set_title(f"Predicted vs Actual — Pruned\nR²={metrics_pruned['R2']:.4f}")
axes[0, 1].set_xlabel("Actual Temperature (°C)")
axes[0, 1].set_ylabel("Predicted Temperature (°C)")

# Residuals — Default
residuals_default = y_test - pred_default
axes[1, 0].scatter(pred_default, residuals_default, alpha=0.6, color='steelblue', edgecolors='k', linewidths=0.3)
axes[1, 0].axhline(0, color='red', linestyle='--', lw=1.5)
axes[1, 0].set_title("Residual Plot — Default")
axes[1, 0].set_xlabel("Predicted Temperature (°C)")
axes[1, 0].set_ylabel("Residuals")

# Residuals — Pruned
residuals_pruned = y_test - pred_pruned
axes[1, 1].scatter(pred_pruned, residuals_pruned, alpha=0.6, color='darkorange', edgecolors='k', linewidths=0.3)
axes[1, 1].axhline(0, color='red', linestyle='--', lw=1.5)
axes[1, 1].set_title("Residual Plot — Pruned (max_depth=5)")
axes[1, 1].set_xlabel("Predicted Temperature (°C)")
axes[1, 1].set_ylabel("Residuals")

plt.tight_layout()
plt.savefig("outputs/plots/decision_tree_evaluation.png", dpi=150, bbox_inches='tight')
plt.close()
print("\nPlot saved: outputs/plots/decision_tree_evaluation.png")

# --------------------------------------------------
# 9. COMPARISON SUMMARY
# --------------------------------------------------

print("\n" + "=" * 65)
print("COMPARISON: DEFAULT vs PRUNED DECISION TREE")
print("=" * 65)
summary = pd.DataFrame([metrics_default, metrics_pruned])
print(summary.to_string(index=False))

print("""
INTERPRETATION:
  - The DEFAULT tree overfits: it memorizes training data perfectly
    (deep tree, many leaves) but may generalize poorly on unseen data.
  - The PRUNED tree (max_depth=5, min_samples_split=10) restricts
    complexity, which typically reduces variance and improves
    generalization on the test set.
  - If the pruned R² is close to or higher than the default R² on
    the test set, pruning has successfully reduced overfitting.
""")

# Save results
summary.to_csv("data/processed/decision_tree_results.csv", index=False)
print("Results saved to: data/processed/decision_tree_results.csv")

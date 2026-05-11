"""
Session Final - Final Model Comparison
Course: Data Mining | Universidad Anáhuac Mayab

This script loads the results from all models and produces
the final comparison table and charts required by the rubric.
Run this AFTER running all individual model scripts.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

os.makedirs("outputs/plots", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# --------------------------------------------------
# 1. LOAD DATA & COMMON SPLIT
# --------------------------------------------------

print("=" * 70)
print("FINAL MODEL COMPARISON — CHIHUAHUA TEMPERATURE PREDICTION")
print("=" * 70)

df = pd.read_csv("data/processed/chihuahua_dataset.csv")

FEATURES = ['solar_irradiance_wm2', 'precipitation_mm', 'wind_speed_ms', 'MONTH_NUM']
TARGET   = 'temperature_c'

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"Dataset: {df.shape[0]} rows | Train: {len(X_train)} | Test: {len(X_test)}")
print(f"Features: {FEATURES}")
print(f"Target  : {TARGET}\n")

# --------------------------------------------------
# 2. TRAIN ALL MODELS
# --------------------------------------------------

models = {
    "Decision Tree (default)":         DecisionTreeRegressor(random_state=42),
    "Decision Tree (pruned d=5)":       DecisionTreeRegressor(max_depth=5, min_samples_split=10, random_state=42),
    "Linear Regression":               LinearRegression(),
    "Polynomial Regression (degree=2)": Pipeline([('poly', PolynomialFeatures(degree=2, include_bias=False)), ('lr', LinearRegression())]),
    "Polynomial Regression (degree=3)": Pipeline([('poly', PolynomialFeatures(degree=3, include_bias=False)), ('lr', LinearRegression())]),
    "ANN Config A (100,) relu":         MLPRegressor(hidden_layer_sizes=(100,),     activation='relu', max_iter=500,  random_state=42),
    "ANN Config B (100,50) tanh":       MLPRegressor(hidden_layer_sizes=(100, 50),  activation='tanh', max_iter=1000, random_state=42),
    "ANN Config C (128,64,32) relu":    MLPRegressor(hidden_layer_sizes=(128,64,32),activation='relu', max_iter=2000, random_state=42),
}

ANN_MODELS = {"ANN Config A (100,) relu", "ANN Config B (100,50) tanh", "ANN Config C (128,64,32) relu"}

results = []
predictions = {}

for name, model in models.items():
    if name in ANN_MODELS:
        model.fit(X_train_sc, y_train)
        pred = model.predict(X_test_sc)
    else:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, pred)
    mse  = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, pred)

    results.append({'Model': name, 'R2': round(r2,4), 'MAE': round(mae,4),
                    'MSE': round(mse,4), 'RMSE': round(rmse,4)})
    predictions[name] = pred

results_df = pd.DataFrame(results).sort_values('R2', ascending=False).reset_index(drop=True)

# --------------------------------------------------
# 3. PRINT COMPARISON TABLE
# --------------------------------------------------

print("=" * 70)
print("PERFORMANCE METRICS — ALL MODELS (sorted by R²)")
print("=" * 70)
print(results_df.to_string(index=False))

best_model = results_df.iloc[0]
print(f"\n★  BEST MODEL: {best_model['Model']}")
print(f"   R²={best_model['R2']} | MAE={best_model['MAE']} | RMSE={best_model['RMSE']}")

# --------------------------------------------------
# 4. METRICS BAR CHART
# --------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Final Model Comparison — Chihuahua Temperature Prediction", fontsize=13, fontweight='bold')

model_labels = [m.replace(" ", "\n") for m in results_df['Model']]
colors = plt.cm.tab10(np.linspace(0, 1, len(results_df)))

# R² (higher is better)
bars1 = axes[0].bar(range(len(results_df)), results_df['R2'], color=colors, edgecolor='k', linewidth=0.5)
axes[0].set_title("R² Score (higher = better)")
axes[0].set_xticks(range(len(results_df)))
axes[0].set_xticklabels(model_labels, fontsize=7, rotation=15, ha='right')
axes[0].set_ylabel("R²")
axes[0].set_ylim(0, 1.1)
axes[0].axhline(1.0, color='gray', linestyle='--', linewidth=0.8)
for bar, val in zip(bars1, results_df['R2']):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha='center', va='bottom', fontsize=7)

# RMSE (lower is better)
bars2 = axes[1].bar(range(len(results_df)), results_df['RMSE'], color=colors, edgecolor='k', linewidth=0.5)
axes[1].set_title("RMSE (lower = better)")
axes[1].set_xticks(range(len(results_df)))
axes[1].set_xticklabels(model_labels, fontsize=7, rotation=15, ha='right')
axes[1].set_ylabel("RMSE (°C)")
for bar, val in zip(bars2, results_df['RMSE']):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                 f"{val:.3f}", ha='center', va='bottom', fontsize=7)

plt.tight_layout()
plt.savefig("outputs/plots/final_comparison_metrics.png", dpi=150, bbox_inches='tight')
plt.close()
print("\nPlot saved: outputs/plots/final_comparison_metrics.png")

# --------------------------------------------------
# 5. PREDICTED vs ACTUAL — ALL MODELS
# --------------------------------------------------

n = len(models)
ncols = 4
nrows = (n + ncols - 1) // ncols
fig2, axes2 = plt.subplots(nrows, ncols, figsize=(20, 5 * nrows))
axes2 = axes2.flatten()

for i, (name, pred) in enumerate(predictions.items()):
    r2 = r2_score(y_test, pred)
    ax = axes2[i]
    ax.scatter(y_test, pred, alpha=0.6, s=30, edgecolors='k', linewidths=0.2)
    lims = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
    ax.plot(lims, lims, 'r--', lw=1.5)
    ax.set_title(f"{name}\nR²={r2:.4f}", fontsize=8)
    ax.set_xlabel("Actual (°C)", fontsize=7)
    ax.set_ylabel("Predicted (°C)", fontsize=7)

for j in range(i + 1, len(axes2)):
    axes2[j].set_visible(False)

fig2.suptitle("Predicted vs Actual — All Models", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/plots/all_models_predicted_vs_actual.png", dpi=150, bbox_inches='tight')
plt.close()
print("Plot saved: outputs/plots/all_models_predicted_vs_actual.png")

# --------------------------------------------------
# 6. SAVE FINAL TABLE
# --------------------------------------------------

results_df.to_csv("data/processed/final_comparison.csv", index=False)
print("Results saved to: data/processed/final_comparison.csv")

# --------------------------------------------------
# 7. CONCLUSION
# --------------------------------------------------

print("""
================================================================================
FINAL CONCLUSION
================================================================================

  The dataset contains monthly climate data for Chihuahua (2020–2025):
  solar irradiance, precipitation, wind speed → predict Earth Skin Temperature.

  MONTH_NUM is the strongest predictor because temperature follows a clear
  seasonal pattern in Chihuahua's semi-arid climate.

  EXPECTED RANKING (based on problem characteristics):
  ┌─────────────────────────────────────────┬──────────────────────────────────┐
  │ Model                                   │ Why it performs this way         │
  ├─────────────────────────────────────────┼──────────────────────────────────┤
  │ ANN (larger configs)                    │ Captures nonlinear interactions  │
  │ Polynomial Regression (degree=2)        │ Handles seasonal curvature well  │
  │ Decision Tree (pruned)                  │ Good with seasonal splits        │
  │ Linear Regression                       │ Assumes linearity — limited fit  │
  │ Decision Tree (default)                 │ Overfits on small dataset        │
  └─────────────────────────────────────────┴──────────────────────────────────┘

  LIMITATIONS:
  - Small dataset (142 rows after preprocessing) limits model generalization.
  - Geographic coverage is limited to 3 lon × 3 lat grid points.
  - Adding humidity or elevation could improve all models.

  POSSIBLE IMPROVEMENTS:
  - Expand to more years / grid points for more training data.
  - Try Random Forest or Gradient Boosting for ensemble methods.
  - Use cross-validation instead of a single train/test split.
================================================================================
""")

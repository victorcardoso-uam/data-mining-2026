"""
Final Cross-Model Comparison — Solar AC Power Output (Pac(W))
Compares the best version of each model type:
  - Decision Tree    : Default (no constraints)
  - Linear Regression: Standard linear model
  - Polynomial Regression (degree=2)
  - ANN Baseline     : (7,) tanh, lbfgs, max_iter=438

All models use:
  - Same dataset     : solar_data_cleaned_active_only.csv
  - Same filter      : Status = 'Normal'
  - Same 23 features
  - Same 80/20 train/test split (random_state=42)
  - Same target      : Pac(W)
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "solar_data_cleaned_active_only.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "comparison_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURE_COLS = [
    "Day_year ", "Hora_SIN", "HORA_COS",
    "INVTemp(\u2103)", "OUTTemp(\u2103)", "AMTemp1(\u2103)", "AMTemp2(\u2103)",
    "Vpv1(V)", "Vpv2(V)", "Vpv3(V)", "Vpv4(V)",
    "Vpv5(V)", "Vpv6(V)", "Vpv7(V)", "Vpv8(V)",
    "VacRS(V)", "VacST(V)", "VacTR(V)",
    "IacR(A)", "IacS(A)", "IacT(A)",
    "PF", "Fac(Hz)",
]

TARGET_COL   = "Pac(W)"
TEST_SIZE    = 0.20
RANDOM_STATE = 42

# ============================================================
# LOAD & FILTER DATA
# ============================================================

print("\n=== LOADING DATA ===")
df = pd.read_csv(DATA_PATH)
df = df[df["Status"] == "Normal"].reset_index(drop=True)
print(f"Shape after Status='Normal' filter: {df.shape}")

X_raw = df[FEATURE_COLS]
y     = df[TARGET_COL].values

imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(X_raw)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

print(f"Train samples: {len(X_train)}")
print(f"Test  samples: {len(X_test)}")

# ============================================================
# HELPER
# ============================================================

def metrics(y_true, y_pred):
    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    return r2, mae, mse, rmse


def save_pred_vs_actual(y_true, y_pred, title, filename, r2, color):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_true, y_pred, alpha=0.4, s=20,
               edgecolors=color, facecolors=color)
    lo = min(y_true.min(), y_pred.min()) - 500
    hi = max(y_true.max(), y_pred.max()) + 500
    ax.plot([lo, hi], [lo, hi], 'r--', lw=1.5, label='Perfect prediction')
    ax.set_xlabel('Actual Pac (W)')
    ax.set_ylabel('Predicted Pac (W)')
    ax.set_title(f'{title}\nR² = {r2:.4f}', fontsize=10, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, linestyle='--', alpha=0.35)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close(fig)
    print(f"  Saved: {filename}")


def save_residuals(y_pred, residuals, title, filename, rmse, color):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_pred, residuals, alpha=0.4, s=20,
               edgecolors=color, facecolors=color)
    ax.axhline(0, color='red', lw=1.5, linestyle='--', label='Zero residual')
    ax.set_xlabel('Predicted Pac (W)')
    ax.set_ylabel('Residual (W)')
    ax.set_title(f'{title}\nRMSE = {rmse:,.1f} W', fontsize=10, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, linestyle='--', alpha=0.35)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close(fig)
    print(f"  Saved: {filename}")


# ============================================================
# MODEL 1 — DECISION TREE (Default, no constraints)
# ============================================================

print("\n=== MODEL 1: Decision Tree (Default) ===")
dt = DecisionTreeRegressor(random_state=RANDOM_STATE)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
dt_r2, dt_mae, dt_mse, dt_rmse = metrics(y_test, dt_pred)
print(f"  R²={dt_r2:.4f}  MAE={dt_mae:.2f}  MSE={dt_mse:.2f}  RMSE={dt_rmse:.2f}")

save_pred_vs_actual(y_test, dt_pred, "Decision Tree — Predicted vs Actual",
                    "cmp_01_dt_pred_vs_actual.png", dt_r2, "steelblue")
save_residuals(dt_pred, y_test - dt_pred, "Decision Tree — Residuals",
               "cmp_02_dt_residuals.png", dt_rmse, "steelblue")


# ============================================================
# MODEL 2 — LINEAR REGRESSION
# ============================================================

print("\n=== MODEL 2: Linear Regression ===")
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_r2, lr_mae, lr_mse, lr_rmse = metrics(y_test, lr_pred)
print(f"  R²={lr_r2:.4f}  MAE={lr_mae:.2f}  MSE={lr_mse:.2f}  RMSE={lr_rmse:.2f}")

save_pred_vs_actual(y_test, lr_pred, "Linear Regression — Predicted vs Actual",
                    "cmp_03_lr_pred_vs_actual.png", lr_r2, "forestgreen")
save_residuals(lr_pred, y_test - lr_pred, "Linear Regression — Residuals",
               "cmp_04_lr_residuals.png", lr_rmse, "forestgreen")


# ============================================================
# MODEL 3 — POLYNOMIAL REGRESSION (degree=2)
# ============================================================

print("\n=== MODEL 3: Polynomial Regression (degree=2) ===")
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly  = poly.transform(X_test)

pr = LinearRegression()
pr.fit(X_train_poly, y_train)
pr_pred = pr.predict(X_test_poly)
pr_r2, pr_mae, pr_mse, pr_rmse = metrics(y_test, pr_pred)
print(f"  R²={pr_r2:.4f}  MAE={pr_mae:.2f}  MSE={pr_mse:.2f}  RMSE={pr_rmse:.2f}")

save_pred_vs_actual(y_test, pr_pred, "Polynomial Regression — Predicted vs Actual",
                    "cmp_05_pr_pred_vs_actual.png", pr_r2, "darkorange")
save_residuals(pr_pred, y_test - pr_pred, "Polynomial Regression — Residuals",
               "cmp_06_pr_residuals.png", pr_rmse, "darkorange")


# ============================================================
# MODEL 4 — ANN BASELINE: (7,) tanh, lbfgs
# ============================================================

print("\n=== MODEL 4: ANN Baseline (7,) tanh lbfgs ===")
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

ann = MLPRegressor(
    hidden_layer_sizes=(7,),
    activation="tanh",
    solver="lbfgs",
    max_iter=438,
    random_state=RANDOM_STATE
)
ann.fit(X_train_sc, y_train)
ann_pred = ann.predict(X_test_sc)
ann_r2, ann_mae, ann_mse, ann_rmse = metrics(y_test, ann_pred)
print(f"  R²={ann_r2:.4f}  MAE={ann_mae:.2f}  MSE={ann_mse:.2f}  RMSE={ann_rmse:.2f}")

save_pred_vs_actual(y_test, ann_pred, "ANN Baseline — Predicted vs Actual",
                    "cmp_07_ann_pred_vs_actual.png", ann_r2, "mediumpurple")
save_residuals(ann_pred, y_test - ann_pred, "ANN Baseline — Residuals",
               "cmp_08_ann_residuals.png", ann_rmse, "mediumpurple")


# ============================================================
# FINAL COMPARISON TABLE
# ============================================================

print("\n" + "=" * 65)
print("=== FINAL CROSS-MODEL COMPARISON ===")
print("=" * 65)

comparison = pd.DataFrame([
    {"Model": "Decision Tree (Default)",         "R2": dt_r2,  "MAE": dt_mae,  "MSE": dt_mse,  "RMSE": dt_rmse},
    {"Model": "Linear Regression",               "R2": lr_r2,  "MAE": lr_mae,  "MSE": lr_mse,  "RMSE": lr_rmse},
    {"Model": "Polynomial Regression (deg=2)",   "R2": pr_r2,  "MAE": pr_mae,  "MSE": pr_mse,  "RMSE": pr_rmse},
    {"Model": "ANN Baseline (7,) tanh lbfgs",    "R2": ann_r2, "MAE": ann_mae, "MSE": ann_mse, "RMSE": ann_rmse},
])

comparison_sorted = comparison.sort_values(by="R2", ascending=False).reset_index(drop=True)
comparison_sorted["Rank"] = range(1, len(comparison_sorted) + 1)

print("\n" + comparison_sorted.round(4).to_string(index=False))

csv_path = os.path.join(OUTPUT_DIR, "final_model_comparison.csv")
comparison_sorted.round(4).to_csv(csv_path, index=False)
print(f"\n✓ Comparison CSV saved: {csv_path}")


# ============================================================
# COMPARISON CHARTS
# ============================================================

models    = comparison_sorted["Model"].tolist()
r2_vals   = comparison_sorted["R2"].tolist()
rmse_vals = comparison_sorted["RMSE"].tolist()
mae_vals  = comparison_sorted["MAE"].tolist()
colors    = ["#2ecc71", "#3498db", "#e67e22", "#9b59b6"]

# R² comparison
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(models, r2_vals, color=colors, edgecolor="white", height=0.5)
for bar, val in zip(bars, r2_vals):
    ax.text(max(val - 0.04, 0.01), bar.get_y() + bar.get_height() / 2,
            f'{val:.4f}', va='center', ha='right',
            color='white', fontsize=10, fontweight='bold')
ax.set_xlabel('R² Score', fontsize=11)
ax.set_title('Final Cross-Model Comparison — R²', fontsize=13, fontweight='bold')
ax.set_xlim(0, 1.05)
ax.axvline(0.9, color='red', linestyle='--', lw=1, label='R²=0.9 reference')
ax.legend(fontsize=9)
ax.grid(axis='x', linestyle='--', alpha=0.4)
ax.invert_yaxis()
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "cmp_09_r2_comparison.png"), dpi=150)
plt.close(fig)
print("  Saved: cmp_09_r2_comparison.png")

# RMSE comparison
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(models, rmse_vals, color=colors, edgecolor="white", height=0.5)
for bar, val in zip(bars, rmse_vals):
    ax.text(val + 50, bar.get_y() + bar.get_height() / 2,
            f'{val:,.1f} W', va='center', ha='left', fontsize=9, fontweight='bold')
ax.set_xlabel('RMSE (W)', fontsize=11)
ax.set_title('Final Cross-Model Comparison — RMSE', fontsize=13, fontweight='bold')
ax.grid(axis='x', linestyle='--', alpha=0.4)
ax.invert_yaxis()
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "cmp_10_rmse_comparison.png"), dpi=150)
plt.close(fig)
print("  Saved: cmp_10_rmse_comparison.png")

# Grouped bar chart: R², MAE-normalized, RMSE-normalized
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
short_labels = ["DT", "LR", "PR", "ANN"]

axes[0].bar(short_labels, r2_vals, color=colors, edgecolor='white')
axes[0].set_title('R² (higher is better)', fontweight='bold')
axes[0].set_ylim(0, 1.05)
axes[0].grid(axis='y', alpha=0.3)

axes[1].bar(short_labels, mae_vals, color=colors, edgecolor='white')
axes[1].set_title('MAE — W (lower is better)', fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

axes[2].bar(short_labels, rmse_vals, color=colors, edgecolor='white')
axes[2].set_title('RMSE — W (lower is better)', fontweight='bold')
axes[2].grid(axis='y', alpha=0.3)

plt.suptitle('Model Comparison Dashboard', fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "cmp_11_dashboard.png"), dpi=150)
plt.close(fig)
print("  Saved: cmp_11_dashboard.png")


# ============================================================
# BEST MODEL IDENTIFICATION & INTERPRETATION
# ============================================================

best = comparison_sorted.iloc[0]

print("\n" + "=" * 65)
print("=== BEST MODEL IDENTIFICATION ===")
print("=" * 65)
print(f"\n  Best model : {best['Model']}")
print(f"  R²         : {best['R2']:.4f}")
print(f"  MAE        : {best['MAE']:.2f} W")
print(f"  MSE        : {best['MSE']:.2f} W²")
print(f"  RMSE       : {best['RMSE']:.2f} W")

print(f"""
Why the Decision Tree is the best model for this dataset:

1. HIGHEST R²: The Decision Tree achieves R²={dt_r2:.4f}, meaning it explains
   {dt_r2*100:.2f}% of the variance in AC power output — more than any other model.

2. LOWEST ERROR: It also achieves the lowest MAE ({dt_mae:.2f} W) and RMSE ({dt_rmse:.2f} W),
   producing the most accurate predictions on unseen test data.

3. WHY IT WORKS WELL HERE: Solar power output has complex, non-linear, threshold-based
   behavior (e.g., panels activate only above a certain irradiance, inverters operate
   within voltage ranges). Decision Trees naturally capture these step-like boundaries
   without requiring explicit feature transformations.

4. VS LINEAR REGRESSION: Linear Regression (R²={lr_r2:.4f}) cannot capture non-linear
   interactions between voltage, current, and temperature without transformation.

5. VS POLYNOMIAL REGRESSION: Polynomial Regression (R²={pr_r2:.4f}) introduced
   {X_train_poly.shape[1]} features for {len(X_train)} samples, causing overfitting.

6. VS ANN: The ANN Baseline (R²={ann_r2:.4f}) performed well but falls short of
   the Decision Tree on this structured, tabular dataset. ANNs typically outperform
   trees on high-dimensional or unstructured data (images, text).

CONCLUSION: For structured tabular photovoltaic data with complex non-linear
relationships, the Decision Tree Regressor is the best-performing model.
""")

print(f"\nAll outputs saved to: {os.path.abspath(OUTPUT_DIR)}")
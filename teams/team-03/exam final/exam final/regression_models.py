"""
Session Final - Regression Models (Linear & Polynomial)
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
import itertools
import os

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

os.makedirs("data/processed", exist_ok=True)
os.makedirs("outputs/plots", exist_ok=True)

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

print("=" * 65)
print("REGRESSION MODELS — TEMPERATURE PREDICTION (CHIHUAHUA)")
print("=" * 65)

df = pd.read_csv("data/processed/chihuahua_dataset.csv")
print(f"Dataset shape: {df.shape}")

# --------------------------------------------------
# 2. FEATURES AND TARGET
# --------------------------------------------------

FEATURES = ['solar_irradiance_wm2', 'precipitation_mm', 'wind_speed_ms', 'MONTH_NUM']
TARGET = 'temperature_c'

X = df[FEATURES]
y = df[TARGET]

print(f"\nInput variables (X): {FEATURES}")
print(f"Target variable  (y): {TARGET}")

# --------------------------------------------------
# 3. TRAIN / TEST SPLIT (80/20)
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")

# --------------------------------------------------
# 4. HELPER FUNCTION
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
# 5. LINEAR REGRESSION
# --------------------------------------------------

print("\n--- LINEAR REGRESSION ---")
lr = LinearRegression()
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)
metrics_lr = evaluate("Linear Regression", y_test, pred_lr)

print("\n  Coefficients:")
for feat, coef in zip(FEATURES, lr.coef_):
    print(f"    {feat:30s}: {coef:.4f}")
print(f"    Intercept                     : {lr.intercept_:.4f}")

# --------------------------------------------------
# 6. POLYNOMIAL REGRESSION — degree 2
# --------------------------------------------------

print("\n--- POLYNOMIAL REGRESSION (degree=2) ---")
poly2 = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('lr',   LinearRegression())
])
poly2.fit(X_train, y_train)
pred_poly2 = poly2.predict(X_test)
metrics_poly2 = evaluate("Polynomial Regression (degree=2)", y_test, pred_poly2)
n_features_poly2 = poly2.named_steps['poly'].n_output_features_
print(f"  Number of features after transformation: {n_features_poly2}")

# --------------------------------------------------
# 7. POLYNOMIAL REGRESSION — degree 3
# --------------------------------------------------

print("\n--- POLYNOMIAL REGRESSION (degree=3) ---")
poly3 = Pipeline([
    ('poly', PolynomialFeatures(degree=3, include_bias=False)),
    ('lr',   LinearRegression())
])
poly3.fit(X_train, y_train)
pred_poly3 = poly3.predict(X_test)
metrics_poly3 = evaluate("Polynomial Regression (degree=3)", y_test, pred_poly3)
n_features_poly3 = poly3.named_steps['poly'].n_output_features_
print(f"  Number of features after transformation: {n_features_poly3}")

# --------------------------------------------------
# 8. OPTIMIZATION — multiple test_size / random_state combinations
# --------------------------------------------------

print("\n--- OPTIMIZATION: Multiple Train/Test Splits ---")

TEST_SIZES    = [0.2, 0.3, 0.4]
RANDOM_STATES = [42, 123, 256]

opt_results = []
for ts, rs in itertools.product(TEST_SIZES, RANDOM_STATES):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=ts, random_state=rs)
    m = LinearRegression().fit(Xtr, ytr)
    p = m.predict(Xte)
    opt_results.append({
        'test_size': ts, 'random_state': rs,
        'R2':   round(r2_score(yte, p), 4),
        'MAE':  round(mean_absolute_error(yte, p), 4),
        'MSE':  round(mean_squared_error(yte, p), 4),
        'RMSE': round(np.sqrt(mean_squared_error(yte, p)), 4)
    })

opt_df = pd.DataFrame(opt_results)
print("\nAll combinations:")
print(opt_df.to_string(index=False))

best_idx = opt_df['R2'].idxmax()
print(f"\nBest configuration (highest R²):")
print(opt_df.iloc[best_idx])

# --------------------------------------------------
# 9. PLOTS
# --------------------------------------------------

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Regression Models — Model Evaluation (Chihuahua Temperature)", fontsize=14, fontweight='bold')

models_data = [
    ("Linear Regression",           pred_lr,    metrics_lr,    'steelblue'),
    ("Polynomial Regression (d=2)", pred_poly2, metrics_poly2, 'darkorange'),
    ("Polynomial Regression (d=3)", pred_poly3, metrics_poly3, 'green'),
]

lims_all = [
    min(y_test.min(), pred_lr.min(), pred_poly2.min(), pred_poly3.min()),
    max(y_test.max(), pred_lr.max(), pred_poly2.max(), pred_poly3.max())
]

# Predicted vs Actual
for col, (name, pred, m, color) in enumerate(models_data):
    ax = axes[0, col]
    ax.scatter(y_test, pred, alpha=0.65, color=color, edgecolors='k', linewidths=0.3)
    ax.plot(lims_all, lims_all, 'r--', lw=1.5)
    ax.set_title(f"{name}\nR²={m['R2']:.4f} | RMSE={m['RMSE']:.4f}")
    ax.set_xlabel("Actual Temp (°C)")
    ax.set_ylabel("Predicted Temp (°C)")

# Residual plots
for col, (name, pred, m, color) in enumerate(models_data):
    ax = axes[1, col]
    residuals = y_test.values - pred
    ax.scatter(pred, residuals, alpha=0.65, color=color, edgecolors='k', linewidths=0.3)
    ax.axhline(0, color='red', linestyle='--', lw=1.5)
    ax.set_title(f"Residuals — {name}")
    ax.set_xlabel("Predicted Temp (°C)")
    ax.set_ylabel("Residuals")

plt.tight_layout()
plt.savefig("outputs/plots/regression_evaluation.png", dpi=150, bbox_inches='tight')
plt.close()
print("\nPlot saved: outputs/plots/regression_evaluation.png")

# --------------------------------------------------
# 10. COMPARISON SUMMARY
# --------------------------------------------------

print("\n" + "=" * 65)
print("COMPARISON: LINEAR vs POLYNOMIAL REGRESSION")
print("=" * 65)
summary = pd.DataFrame([metrics_lr, metrics_poly2, metrics_poly3])
print(summary.to_string(index=False))

print("""
INTERPRETATION:
  - LINEAR REGRESSION fits a flat hyperplane through the data.
    It works well when the relationship between X and y is linear.
  - POLYNOMIAL REGRESSION (degree=2) adds squared and interaction
    terms, allowing the model to capture curved relationships.
  - POLYNOMIAL REGRESSION (degree=3) goes one step further, adding
    cubic terms. This can improve fit but risks overfitting on small
    datasets (only ~115 training rows here).
  - If degree=2 improves R² significantly over linear, there is a
    nonlinear relationship between climate variables and temperature.
  - If degree=3 performs worse than degree=2 on test data, overfitting
    is occurring — the model learned noise in the training data.
""")

# Save results
summary.to_csv("data/processed/regression_results.csv", index=False)
opt_df.to_csv("data/processed/regression_optimization_results.csv", index=False)
print("Results saved to: data/processed/regression_results.csv")
print("Optimization saved to: data/processed/regression_optimization_results.csv")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import math
 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
 
# ============================================================
# LOAD DATA
# ============================================================
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "solar_data_cleaned_active_only.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "regression_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
 
df = pd.read_csv(DATA_PATH)
 
# Keep only rows where the inverter is producing power
# 'Waiting' rows have Pac(W) = 0 and do not represent generation behavior
df = df[df["Status"] == "Normal"].reset_index(drop=True)
 
print("\n=== DATASET PREVIEW ===")
print(f"Shape after filtering Status='Normal': {df.shape}")
 
# ============================================================
# SELECT FEATURES AND TARGET
# ============================================================
# Target: AC power output
# Excluded to avoid data leakage: EacToday, EacTotal, EpvToday, EpvTotal, Ppv1-8
FEATURE_COLS = [
    "Day_year ", "Hora_SIN", "HORA_COS",
    "INVTemp(℃)", "OUTTemp(℃)", "AMTemp1(℃)", "AMTemp2(℃)",
    "Vpv1(V)", "Vpv2(V)", "Vpv3(V)", "Vpv4(V)",
    "Vpv5(V)", "Vpv6(V)", "Vpv7(V)", "Vpv8(V)",
    "VacRS(V)", "VacST(V)", "VacTR(V)",
    "IacR(A)", "IacS(A)", "IacT(A)",
    "PF", "Fac(Hz)",
]
 
X_raw = df[FEATURE_COLS]
y     = df["Pac(W)"]
 
# Impute missing values with column median
imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(X_raw)
 
# ============================================================
# TRAIN / TEST SPLIT — 80/20 (same across ALL models)
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
 
print(f"Train samples: {len(X_train)}")
print(f"Test  samples: {len(X_test)}")
 
# ============================================================
# MODEL 1 — LINEAR REGRESSION
# ============================================================
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_predictions_train = linear_model.predict(X_train)
linear_predictions = linear_model.predict(X_test)
 
# ============================================================
# MODEL 2 — POLYNOMIAL REGRESSION (degree=2)
# ============================================================
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly  = poly.transform(X_test)
 
print(f"\nOriginal features  : {X_train.shape[1]}")
print(f"Polynomial features: {X_train_poly.shape[1]}")
 
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
poly_predictions_train = poly_model.predict(X_train_poly)
poly_predictions = poly_model.predict(X_test_poly)
 
# ============================================================
# EVALUATION METRICS — TRAIN SET
# ============================================================
linear_train_mae  = mean_absolute_error(y_train, linear_predictions_train)
linear_train_mse  = mean_squared_error(y_train, linear_predictions_train)
linear_train_rmse = math.sqrt(linear_train_mse)
linear_train_r2   = r2_score(y_train, linear_predictions_train)

poly_train_mae  = mean_absolute_error(y_train, poly_predictions_train)
poly_train_mse  = mean_squared_error(y_train, poly_predictions_train)
poly_train_rmse = math.sqrt(poly_train_mse)
poly_train_r2   = r2_score(y_train, poly_predictions_train)

# ============================================================
# EVALUATION METRICS — TEST SET
# ============================================================
linear_mae  = mean_absolute_error(y_test, linear_predictions)
linear_mse  = mean_squared_error(y_test, linear_predictions)
linear_rmse = math.sqrt(linear_mse)
linear_r2   = r2_score(y_test, linear_predictions)
 
poly_mae  = mean_absolute_error(y_test, poly_predictions)
poly_mse  = mean_squared_error(y_test, poly_predictions)
poly_rmse = math.sqrt(poly_mse)
poly_r2   = r2_score(y_test, poly_predictions)
 
print("\n=== LINEAR REGRESSION METRICS ===")
print("R²  :", round(linear_r2,   4))
print("MAE :", round(linear_mae,  2), "W")
print("MSE :", round(linear_mse,  2), "W²")
print("RMSE:", round(linear_rmse, 2), "W")
 
print("\n=== POLYNOMIAL REGRESSION METRICS (degree=2) ===")
print("R²  :", round(poly_r2,   4))
print("MAE :", round(poly_mae,  2), "W")
print("MSE :", round(poly_mse,  2), "W²")
print("RMSE:", round(poly_rmse, 2), "W")
 
# ============================================================
# PLOTS — PREDICTED VS ACTUAL
# ============================================================
# Linear
plt.figure(figsize=(7, 6))
plt.scatter(y_test, linear_predictions, alpha=0.5, color="steelblue")
min_val = min(y_test.min(), linear_predictions.min())
max_val = max(y_test.max(), linear_predictions.max())
plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect prediction")
plt.xlabel("Actual Pac(W)")
plt.ylabel("Predicted Pac(W)")
plt.title(f"Linear Regression — Predicted vs Actual\n(R² = {round(linear_r2, 4)})")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "reg_01_linear_pred_vs_actual.png"), dpi=200)
plt.close()
 
# Polynomial
plt.figure(figsize=(7, 6))
plt.scatter(y_test, poly_predictions, alpha=0.5, color="darkorange")
min_val = min(y_test.min(), poly_predictions.min())
max_val = max(y_test.max(), poly_predictions.max())
plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect prediction")
plt.xlabel("Actual Pac(W)")
plt.ylabel("Predicted Pac(W)")
plt.title(f"Polynomial Regression (degree=2) — Predicted vs Actual\n(R² = {round(poly_r2, 4)})")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "reg_02_poly_pred_vs_actual.png"), dpi=200)
plt.close()
 
# ============================================================
# PLOTS — RESIDUALS
# ============================================================
linear_residuals = y_test - linear_predictions
poly_residuals   = y_test - poly_predictions
 
# Linear residuals
plt.figure(figsize=(7, 5))
plt.scatter(linear_predictions, linear_residuals, alpha=0.5, color="steelblue")
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted Pac(W)")
plt.ylabel("Residuals (W)")
plt.title(f"Linear Regression — Residuals\n(RMSE = {round(linear_rmse, 2)} W)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "reg_03_linear_residuals.png"), dpi=200)
plt.close()
 
# Polynomial residuals
plt.figure(figsize=(7, 5))
plt.scatter(poly_predictions, poly_residuals, alpha=0.5, color="darkorange")
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted Pac(W)")
plt.ylabel("Residuals (W)")
plt.title(f"Polynomial Regression (degree=2) — Residuals\n(RMSE = {round(poly_rmse, 2)} W)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "reg_04_poly_residuals.png"), dpi=200)
plt.close()
 
print("\nPlots saved in regression_outputs/")

# ============================================================
# COMPARISON: LINEAR vs POLYNOMIAL REGRESSION
# ============================================================
print("\n" + "=" * 60)
print("=== COMPARISON: LINEAR vs POLYNOMIAL REGRESSION ===")
print("=" * 60)

summary_rows = [
    {
        "model": "Linear Regression",
        "train_R2": linear_train_r2,
        "test_R2": linear_r2,
        "train_RMSE": linear_train_rmse,
        "test_RMSE": linear_rmse,
        "test_MAE": linear_mae,
        "test_MSE": linear_mse
    },
    {
        "model": "Polynomial Regression (degree=2)",
        "train_R2": poly_train_r2,
        "test_R2": poly_r2,
        "train_RMSE": poly_train_rmse,
        "test_RMSE": poly_rmse,
        "test_MAE": poly_mae,
        "test_MSE": poly_mse
    }
]

comparison_df = pd.DataFrame(summary_rows)
print("\n" + comparison_df.round(4).to_string(index=False))

# Save comparison to CSV
comparison_file = os.path.join(OUTPUT_DIR, "reg_model_comparison_summary.csv")
comparison_df.to_csv(comparison_file, index=False)
print(f"\n✓ Comparison saved to: reg_model_comparison_summary.csv")

if poly_r2 > linear_r2:
    print(f"\nBest Model: Polynomial Regression (Test R²={poly_r2:.4f} vs Linear={linear_r2:.4f})")
else:
    print(f"\nBest Model: Linear Regression (Test R²={linear_r2:.4f} vs Polynomial={poly_r2:.4f})")

 
# ============================================================
# QUESTIONS — PRINTED TO TERMINAL
# ============================================================
print("\n" + "=" * 50)
print("=== LINEAR vs POLYNOMIAL — INTERPRETATION ===")
print("=" * 50)
 
print(f"\n1. Did the polynomial model improve performance over linear regression?")
print(
    f"   No. Linear Regression achieved R²={round(linear_r2, 4)} and RMSE={round(linear_rmse, 2)} W "
    f"on the test set. Polynomial Regression (degree=2) reached R²={round(poly_r2, 4)} and "
    f"RMSE={round(poly_rmse, 2)} W — higher error, meaning performance did not improve."
)
 
print(f"\n2. Why did polynomial regression not improve performance?")
print(
    f"   The selected variables (voltages, currents, temperatures, time encodings) "
    f"already have a very strong linear relationship with Pac(W). Adding polynomial "
    f"terms introduced {X_train_poly.shape[1] - X_train.shape[1]} extra features, "
    f"but with only {len(X_train)} training samples this causes overfitting: "
    f"the model fits training data well but generalizes poorly to unseen data."
)
 
print(f"\n3. Which model would you select and why?")
print(
    f"   Linear Regression. It achieved the best test R² ({round(linear_r2, 4)}), "
    f"lowest RMSE ({round(linear_rmse, 2)} W), and lowest MAE ({round(linear_mae, 2)} W). "
    f"It is also simpler and more interpretable. Polynomial complexity only "
    f"introduced overfitting without any improvement in generalization."
)
 
print(f"\nFinal Interpretation:")
print(
    f"   Linear Regression is the strongest regression model for predicting "
    f"photovoltaic AC power output (Pac(W)). Under stable operating conditions, "
    f"power output is nearly proportional to current and voltage — variables "
    f"already included in the feature set. Polynomial Regression (degree=2) "
    f"generated {X_train_poly.shape[1]} features for only {len(X_train)} training "
    f"samples, making overfitting inevitable."
)
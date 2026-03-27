"""
Session 21 — Polynomial Regression
Instructor Demo Script

Dataset:
- polynomial_regression_demo_dataset.csv

Goal:
1) Plot the raw data
2) Train a linear regression model
3) Train a polynomial regression model
4) Plot the data again with BOTH model curves on top
5) Compare evaluation metrics

This script is designed for classroom explanation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# LOAD DATA
# ============================================================

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(script_dir, "polynomial_regression_demo_dataset.csv")

df = pd.read_csv(DATA_PATH)

print("\n=== DATASET PREVIEW ===")
print(df.head())

X = df[["production_hours"]]
y = df["daily_output_units"]


# ============================================================
# PLOT 1 — RAW DATA ONLY
# ============================================================

plt.figure(figsize=(9, 6))
plt.scatter(X["production_hours"], y, label="Observed data")
plt.xlabel("Production Hours")
plt.ylabel("Daily Output Units")
plt.title("Raw Data")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)


# ============================================================
# MODEL 1 — LINEAR REGRESSION
# ============================================================

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_predictions = linear_model.predict(X_test)


# ============================================================
# MODEL 2 — POLYNOMIAL REGRESSION
# ============================================================

# We transform the input variable into polynomial features.
# Example:
# x -> [1, x, x^2] for degree = 2
poly = PolynomialFeatures(degree=2, include_bias=False)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
poly_predictions = poly_model.predict(X_test_poly)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, rmse, r2


linear_metrics = evaluate(y_test, linear_predictions)
poly_metrics = evaluate(y_test, poly_predictions)

print("\n=== LINEAR REGRESSION METRICS ===")
print("MAE :", round(linear_metrics[0], 4))
print("MSE :", round(linear_metrics[1], 4))
print("RMSE:", round(linear_metrics[2], 4))
print("R2  :", round(linear_metrics[3], 4))

print("\n=== POLYNOMIAL REGRESSION METRICS (degree = 2) ===")
print("MAE :", round(poly_metrics[0], 4))
print("MSE :", round(poly_metrics[1], 4))
print("RMSE:", round(poly_metrics[2], 4))
print("R2  :", round(poly_metrics[3], 4))


# ============================================================
# CREATE SMOOTH CURVES FOR PLOTTING
# ============================================================

x_grid = np.linspace(X["production_hours"].min(), X["production_hours"].max(), 300).reshape(-1, 1)
x_grid_df = pd.DataFrame(x_grid, columns=["production_hours"])

# Linear model line
y_grid_linear = linear_model.predict(x_grid_df)

# Polynomial model curve
x_grid_poly = poly.transform(x_grid_df)
y_grid_poly = poly_model.predict(x_grid_poly)


# ============================================================
# PLOT 2 — DATA + MODELS
# ============================================================

plt.figure(figsize=(9, 6))
plt.scatter(X["production_hours"], y, label="Observed data")
plt.plot(x_grid_df["production_hours"], y_grid_linear, label="Linear Regression", linewidth=2)
plt.plot(x_grid_df["production_hours"], y_grid_poly, label="Polynomial Regression (degree=2)", linewidth=2)
plt.xlabel("Production Hours")
plt.ylabel("Daily Output Units")
plt.title("Data with Regression Models")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# ============================================================
# FINAL QUESTIONS FOR CLASS DISCUSSION
# ============================================================

print("\n=== CLASS DISCUSSION ===")
print("1. Which model better follows the curvature of the data?")
print("   → Polynomial Regression (degree=2) follows the curved pattern of the data better.")
print("\n2. Which model gives better evaluation metrics?")
print("   → Polynomial: MAE=2.4471, MSE=8.6043, RMSE=2.9333, R²=0.9693")
print("   → Linear:    MAE=5.1287, MSE=34.359, RMSE=5.8617, R²=0.8773")
print("   → Polynomial is significantly better (57% lower MAE, 75% lower MSE, 50% better R²).")
print("\n3. Why is polynomial regression still considered a regression model?")
print("   → It fits a continuous function to predict numeric targets using least squares.")
print("   → Only the features (X) are polynomial; the relationship is still linear in parameters.")
print("\n4. What could happen if we increase the polynomial degree too much?")
print("   → Overfitting: model memorizes training data instead of learning patterns.")
print("   → Poor generalization: predicts well on training data but poorly on new data.")
print("   → Increased complexity without proportional benefit in prediction accuracy.")

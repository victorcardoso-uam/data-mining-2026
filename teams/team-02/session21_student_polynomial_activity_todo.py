"""
Session 21 — Polynomial Regression
Student Activity Script (with TODOs)

Dataset:
- polynomial_regression_student_activity_dataset.csv

Goal:
1) Plot the raw data
2) Build a baseline linear regression model
3) Build a polynomial regression model
4) Compare both models using 4 evaluation metrics
5) Plot the data again with the model curve shown on top

IMPORTANT:
This activity uses ONLY ONE predictor for the polynomial comparison:
- outside_temperature_c

Why?
Because plotting polynomial regression is much clearer in 2D
when we use one input variable and one target variable.

YOUR TASK:
Complete all TODO sections.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use file-based backend (saves without display)
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# LOAD DATA
# ============================================================

DATA_PATH = r"C:\Users\david\Downloads\Data Mining Course\Dataset Parcial 2\Data\Raw\polynomial_regression_student_activity_dataset.csv"

df = pd.read_csv(DATA_PATH)

print("\n=== DATASET PREVIEW ===")
print(df.head())


# ============================================================
# TASK 1
# Select the predictor and the target
# Use:
# predictor -> outside_temperature_c
# target    -> cooling_load_kw
# ============================================================

X = df[["outside_temperature_c"]]
y = df["cooling_load_kw"]


# ============================================================
# PLOT 1 — RAW DATA ONLY
# ============================================================

plt.figure(figsize=(9, 6))

# TODO 2:
# Create the scatter plot of the raw data
# x-axis: outside_temperature_c
# y-axis: cooling_load_kw
plt.scatter(X["outside_temperature_c"], y, alpha=0.5, label="Observed Data")

plt.xlabel("Outside Temperature (°C)")
plt.ylabel("Cooling Load (kW)")
plt.title("Raw Data")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('plot_1_raw_data.png', dpi=100, bbox_inches='tight')
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

# TODO 3:
# Fit the linear regression model
linear_model.fit(X_train, y_train)

# TODO 4:
# Generate predictions for X_test
linear_predictions = linear_model.predict(X_test)


# ============================================================
# MODEL 2 — POLYNOMIAL REGRESSION
# ============================================================

# TODO 5:
# Create polynomial features of degree 2
poly = PolynomialFeatures(degree=2)

# TODO 6:
# Transform X_train and X_test
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_model = LinearRegression()

# TODO 7:
# Fit the polynomial regression model
poly_model.fit(X_train_poly, y_train)

# TODO 8:
# Generate predictions for X_test_poly
poly_predictions = poly_model.predict(X_test_poly)


# ============================================================
# TASK 9 — EVALUATION METRICS
# Calculate the 4 metrics used in previous sessions:
# MAE, MSE, RMSE, R2
# ============================================================

# Linear metrics
linear_mae = mean_absolute_error(y_test, linear_predictions)
linear_mse = mean_squared_error(y_test, linear_predictions)
linear_rmse = np.sqrt(linear_mse)
linear_r2 = r2_score(y_test, linear_predictions)

# Polynomial metrics
poly_mae = mean_absolute_error(y_test, poly_predictions)
poly_mse = mean_squared_error(y_test, poly_predictions)
poly_rmse = np.sqrt(poly_mse)
poly_r2 = r2_score(y_test, poly_predictions)

print("\n=== LINEAR REGRESSION METRICS ===")
print("MAE :", round(linear_mae, 4))
print("MSE :", round(linear_mse, 4))
print("RMSE:", round(linear_rmse, 4))
print("R2  :", round(linear_r2, 4))

print("\n=== POLYNOMIAL REGRESSION METRICS ===")
print("MAE :", round(poly_mae, 4))
print("MSE :", round(poly_mse, 4))
print("RMSE:", round(poly_rmse, 4))
print("R2  :", round(poly_r2, 4))


# ============================================================
# TASK 10 — CREATE A SMOOTH CURVE FOR PLOTTING
# ============================================================

x_grid = np.linspace(
    X["outside_temperature_c"].min(),
    X["outside_temperature_c"].max(),
    300
).reshape(-1, 1)

x_grid_df = pd.DataFrame(x_grid, columns=["outside_temperature_c"])

# TODO 11:
# Predict using the linear model on the grid
y_grid_linear = linear_model.predict(x_grid_df)

# TODO 12:
# Transform the grid for the polynomial model and predict
x_grid_poly = poly.transform(x_grid_df)
y_grid_poly = poly_model.predict(x_grid_poly)


# ============================================================
# PLOT 2 — DATA + MODELS
# ============================================================

plt.figure(figsize=(9, 6))

# TODO 13:
# Plot the observed data again as a scatter plot
plt.scatter(X["outside_temperature_c"], y, alpha=0.5, label="Observed Data")

# TODO 14:
# Plot the linear regression line
plt.plot(x_grid_df["outside_temperature_c"], y_grid_linear, color="blue", linewidth=2, label="Linear Regression")

# TODO 15:
# Plot the polynomial regression curve
plt.plot(x_grid_df["outside_temperature_c"], y_grid_poly, color="red", linewidth=2, label="Polynomial Regression")

plt.xlabel("Outside Temperature (°C)")
plt.ylabel("Cooling Load (kW)")
plt.title("Data with Regression Models")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('plot_2_data_with_models.png', dpi=100, bbox_inches='tight')
plt.show()


# ============================================================
# FINAL QUESTIONS
# ============================================================

print("\n=== QUESTIONS FOR YOUR TEAM ===")
print("1. Which model better fits the shape of the data?")
print("2. Which model has better MAE, RMSE, and R2?")
print("3. Why does polynomial regression perform better in this case?")
print("4. What might happen if you increase the degree too much?")


# ============================================================
# ANSWERS TO QUESTIONS
# ============================================================

# 1. Which model better fits the shape of the data?
# ANSWER: Polynomial regression (degree 2) better fits the shape of the data.
# The data shows a curved, non-linear relationship between temperature and cooling load.
# Polynomial regression can follow this curved pattern, while linear regression can only
# fit a straight line, which misses the true shape of the relationship.

# 2. Which model has better MAE, RMSE, and R2?
# ANSWER: Polynomial regression has better metrics across all three measures:
# Polynomial: MAE=1.9668, RMSE=2.4573, R2=0.9667
# Linear:     MAE=3.9455, RMSE=4.6923, R2=0.8785
# The polynomial model has lower errors (about 50% lower MAE and RMSE) and
# a higher R2 score (0.9667 vs 0.8785), meaning it explains more variance in the data.

# 3. Why does polynomial regression perform better in this case?
# ANSWER: Polynomial regression performs better because the actual relationship
# between outside temperature and cooling load is non-linear. As temperature increases,
# cooling load doesn't increase at a constant rate - it follows a curved pattern.
# A degree-2 polynomial (quadratic) can model this curved relationship much more
# accurately than a simple linear model.

# 4. What might happen if you increase the degree too much?
# ANSWER: If you increase the polynomial degree too much (e.g., degree 5, 10, etc.),
# the model will likely suffer from overfitting. It will fit the training data
# extremely well (almost perfectly), but it will capture noise and outliers instead
# of the true underlying pattern. This leads to poor performance on new, unseen data.
# The model becomes too complex and loses its ability to generalize. The best
# practice is to find the right balance - a degree that fits the data well without
# overfitting (degree 2 or 3 often works well for most real-world data).

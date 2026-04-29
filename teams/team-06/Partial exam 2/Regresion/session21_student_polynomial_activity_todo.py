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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "polynomial_regression_student_activity_dataset.csv")

df = pd.read_csv(DATA_PATH)

print("\n=== DATASET PREVIEW ===")
print(df.head())

# ============================================================
# TASK 1: Select the predictor and the target
# ============================================================
X = df[['outside_temperature_c']] 
y = df['cooling_load_kw']

# ============================================================
# PLOT 1 — RAW DATA ONLY
# ============================================================
plt.figure(figsize=(9, 6))

# TODO 2: Scatter plot (Estilo del profesor)
plt.scatter(X["outside_temperature_c"], y, label="Observed data")

plt.xlabel("Outside Temperature (°C)")
plt.ylabel("Cooling Load (kW)")
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

# TODO 3 & 4: Fit and Predict
linear_model.fit(X_train, y_train)
linear_predictions = linear_model.predict(X_test)

# ============================================================
# MODEL 2 — POLYNOMIAL REGRESSION
# ============================================================
# TODO 5, 6, 7 & 8: Degree 2, Transform, Fit and Predict
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
poly_predictions = poly_model.predict(X_test_poly)

# ============================================================
# TASK 9 — EVALUATION METRICS
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

# TODO 11 & 12: Predict on grid
y_grid_linear = linear_model.predict(x_grid_df)
x_grid_poly = poly.transform(x_grid_df)
y_grid_poly = poly_model.predict(x_grid_poly)

# ============================================================
# PLOT 2 — DATA + MODELS
# ============================================================
plt.figure(figsize=(9, 6))

# TODO 13, 14 & 15: Plot con estilo del profesor
plt.scatter(X["outside_temperature_c"], y, label="Observed data")
plt.plot(x_grid_df["outside_temperature_c"], y_grid_linear, label="Linear Regression", linewidth=2, color='red')
plt.plot(x_grid_df["outside_temperature_c"], y_grid_poly, label="Polynomial Regression (degree=2)", linewidth=2, color='green')

plt.xlabel("Outside Temperature (°C)")
plt.ylabel("Cooling Load (kW)")
plt.title("Data with Regression Models")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ============================================================
# FINAL QUESTIONS - PRINTED TO TERMINAL
# ============================================================
print("\n" + "="*40)
print("=== QUESTIONS FOR YOUR TEAM ===")
print("="*40)

print("\n1. Which model better fits the shape of the data?")
print("Respuesta: The Polynomial model (degree 2). The raw data shows a curved trend")
print("that the straight line of the Linear model cannot capture.")

print("\n2. Which model has better MAE, RMSE, and R2?")
print(f"Respuesta: The Polynomial model. (R2: {round(poly_r2, 4)} vs Linear R2: {round(linear_r2, 4)})")
print("It significantly reduced the error metrics (MAE/RMSE).")

print("\n3. Why does polynomial regression perform better in this case?")
print("Respuesta: Because the relationship between temperature and cooling load is non-linear.")
print("As temperature increases, the cooling demand grows at an accelerating rate.")

print("\n4. What might happen if you increase the degree too much?")
print("Respuesta: Overfitting. The model would follow the noise of the data instead of the")
print("actual trend, losing its ability to generalize to new data. git checkout main, git pull origin main -- session_24_ANN_project/session_24_ANN_project.py")
"""
Session 21 — Polynomial Regression
Completed by: Valeria García - Team 05
Date: March 2026
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# 1. SMART DATA LOADING (Funciona en cualquier carpeta)
# ============================================================
# Esta línea detecta automáticamente dónde está tu script y busca el CSV ahí
folder_path = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(folder_path, "polynomial_regression_student_activity_dataset.csv")

if not os.path.exists(DATA_PATH):
    print(f"⚠️ ERROR: No se encuentra el archivo en: {DATA_PATH}")
    print("Asegúrate de que el CSV esté en la misma carpeta que este script.")
else:
    df = pd.read_csv(DATA_PATH)
    print("\n✅ DATASET LOADED SUCCESSFULLY - TEAM 05")
    print(df.head())

    # ============================================================
    # 2. TASK 1: SELECT PREDICTOR AND TARGET
    # ============================================================
    X = df[["outside_temperature_c"]]
    y = df["cooling_load_kw"]

    # ============================================================
    # 3. PLOT 1 — RAW DATA
    # ============================================================
    plt.figure(figsize=(9, 6))
    plt.scatter(X, y, color="royalblue", alpha=0.6, label="Observed Data")
    plt.xlabel("Outside Temperature (°C)")
    plt.ylabel("Cooling Load (kW)")
    plt.title("Step 1: Raw Data Visualization - Valeria García")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # ============================================================
    # 4. TRAIN / TEST SPLIT
    # ============================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # ============================================================
    # 5. MODEL 1 — LINEAR REGRESSION
    # ============================================================
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    linear_predictions = linear_model.predict(X_test)

    # ============================================================
    # 6. MODEL 2 — POLYNOMIAL REGRESSION (Degree 2)
    # ============================================================
    poly = PolynomialFeatures(degree=2)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)
    poly_predictions = poly_model.predict(X_test_poly)

    # ============================================================
    # 7. EVALUATION METRICS
    # ============================================================
    def calculate_metrics(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        return mae, mse, rmse, r2

    l_mae, l_mse, l_rmse, l_r2 = calculate_metrics(y_test, linear_predictions)
    p_mae, p_mse, p_rmse, p_r2 = calculate_metrics(y_test, poly_predictions)

    print("\n" + "="*50)
    print("         PERFORMANCE METRICS COMPARISON")
    print("="*50)
    print(f"{'Metric':<10} | {'Linear Model':<15} | {'Polynomial (D2)':<15}")
    print("-" * 50)
    print(f"{'MAE':<10} | {l_mae:<15.4f} | {p_mae:<15.4f}")
    print(f"{'RMSE':<10} | {l_rmse:<15.4f} | {p_rmse:<15.4f}")
    print(f"{'R2':<10} | {l_r2:<15.4f} | {p_r2:<15.4f}")
    print("="*50)

    # ============================================================
    # 8. PLOT 2 — FINAL COMPARISON
    # ============================================================
    x_grid = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
    x_grid_df = pd.DataFrame(x_grid, columns=["outside_temperature_c"])
    
    y_grid_linear = linear_model.predict(x_grid_df)
    y_grid_poly = poly_model.predict(poly.transform(x_grid_df))

    plt.figure(figsize=(10, 7))
    plt.scatter(X, y, color="gray", alpha=0.3, label="Data Points")
    plt.plot(x_grid, y_grid_linear, color="orange", linestyle="--", label="Linear Model (Underfit)")
    plt.plot(x_grid, y_grid_poly, color="green", linewidth=3, label="Polynomial Model (Degree 2)")
    
    plt.xlabel("Outside Temperature (°C)")
    plt.ylabel("Cooling Load (kW)")
    plt.title("Linear vs Polynomial Regression Analysis - Team 05")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # ============================================================
    # 9. FINAL ANSWERS
    # ============================================================
    print("\n=== FINAL QUESTIONS FOR THE TEAM ===")
    print("1. Which model better fits the shape of the data?")
    print("   ANSWER: The Polynomial model fits better as it follows the curve of the data.")
    
    print("\n2. Which model has better MAE, RMSE, and R2?")
    print(f"   ANSWER: Polynomial. It has lower errors and a higher R2 ({p_r2:.4f}).")
    
    print("\n3. Why does polynomial regression perform better in this case?")
    print("   ANSWER: Because the relationship between temperature and cooling load is non-linear.")
    
    print("\n4. What might happen if you increase the degree too much?")
    print("   ANSWER: Overfitting. The model would learn the noise instead of the trend.")
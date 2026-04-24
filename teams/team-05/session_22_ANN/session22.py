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
"""
Session 18 — Regression Optimization
"""

import pandas as pd
import numpy as np
import itertools
import os  # Añadido para que funcione tu configuración de rutas

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -----------------------------
# CONFIGURATION
# -----------------------------

# Configuración de rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "amazon.csv") 

TARGET = "rating"

FEATURES = [
    "discounted_price",
    "actual_price",
    "rating_count",
    "discount_percentage",
]

# Parámetros para evaluar
TEST_SIZES = [0.2, 0.3, 0.4]
RANDOM_STATES = [42, 123, 256]


# -----------------------------
# LOAD DATA
# -----------------------------

df = pd.read_csv(DATA_PATH)

# --- INICIO DE LIMPIEZA ---
# Limpiar precios (quitar ₹ y comas)
for col in ['discounted_price', 'actual_price']:
    df[col] = df[col].str.replace('₹', '').str.replace(',', '').astype(float)

# Limpiar porcentaje (quitar %)
df['discount_percentage'] = df['discount_percentage'].str.replace('%', '').astype(float)

# Limpiar rating_count (quitar comas)
df['rating_count'] = df['rating_count'].str.replace(',', '').astype(float)

# Limpiar rating (convertir a número, ignorando errores como el '|')
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

# Quitar filas con valores nulos para que el modelo no falle
df.dropna(subset=FEATURES + [TARGET], inplace=True)
# --- FIN DE LIMPIEZA ---

X = df[FEATURES]
y = df[TARGET]


# -----------------------------
# TRAIN / TEST SPLIT AND MODEL TRAINING
# -----------------------------

# Almacenar resultados
results = []

# Iterar sobre todas las combinaciones de test_size y random_state
for test_size, random_state in itertools.product(TEST_SIZES, RANDOM_STATES):
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Entrenar modelo
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Hacer predicciones
    predictions = model.predict(X_test)
    
    # Calcular métricas
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    
    # Guardar resultados
    results.append({
        'test_size': test_size,
        'random_state': random_state,
        'MAE': round(mae, 3),
        'MSE': round(mse, 3),
        'RMSE': round(rmse, 3),
        'R2': round(r2, 3)
    })


# -----------------------------
# EVALUATION METRICS
# -----------------------------

# Crear DataFrame con resultados
results_df = pd.DataFrame(results)

print("\n" + "="*70)
print("EVALUATION METRICS - ALL POSSIBLE COMBINATIONS")
print("="*70)
print(results_df.to_string(index=False))

# Mostrar mejores configuraciones
print("\n" + "="*70)
print("BEST CONFIGURATIONS")
print("="*70)

# Mejor R2 (mejor ajuste del modelo)
best_r2_idx = results_df['R2'].idxmax()
print("\nBest R2 (Best Model Fit):")
print(results_df.iloc[best_r2_idx])

# Mejor MAE (menor error absoluto)
best_mae_idx = results_df['MAE'].idxmin()
print("\nBest MAE (Lowest Absolute Error):")
print(results_df.iloc[best_mae_idx])

# Mejor MSE (menor error cuadrático)
best_mse_idx = results_df['MSE'].idxmin()
print("\nBest MSE (Lowest Mean Squared Error):")
print(results_df.iloc[best_mse_idx])

# Mejor RMSE (menor error cuadrático)
best_rmse_idx = results_df['RMSE'].idxmin()
print("\nBest RMSE (Lowest Squared Error):")
print(results_df.iloc[best_rmse_idx])
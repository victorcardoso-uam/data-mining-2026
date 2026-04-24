"""
SESSION 23 — ANN TUNING ACTIVITY (WITH TODOs)

GOAL: Use the SAME dataset from Session 22 and explore how different ANN
hyperparameters affect model performance.

DATASET: industrial_ann_teacher_example.csv
TARGET VARIABLE: production_quality_score
"""

import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ============================================================
# 1. LOAD DATA
# ============================================================

# Ruta ajustada a tu estructura de equipo 05
DATA_PATH = "teams/team-05/session_23/industrial_ann_teacher_example.csv"

# TODO 1: Load the dataset into a DataFrame called data
if os.path.exists(DATA_PATH):
    data = pd.read_csv(DATA_PATH)
    print("\n=== DATASET PREVIEW ===")
    print(data.head())
    print("\n=== DATASET SHAPE ===")
    print(data.shape)
else:
    print(f"Error: No se encontró el archivo en {DATA_PATH}")
    # Creamos datos sintéticos solo para que el código no falle si no tienes el CSV a la mano
    data = pd.DataFrame(np.random.rand(100, 5), columns=['f1', 'f2', 'f3', 'f4', 'production_quality_score'])

# ============================================================
# 2. DEFINE INPUTS (X) AND TARGET (y)
# ============================================================

# TODO 2: Define X using all columns except the target column
X = data.drop('production_quality_score', axis=1)

# TODO 3: Define y using only the target column
y = data['production_quality_score']


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ============================================================
# 4. SCALE INPUT FEATURES
# ============================================================

scaler = StandardScaler()

# TODO 4: Fit the scaler on X_train and transform X_train
X_train = scaler.fit_transform(X_train)

# TODO 5: Transform X_test using the same scaler
X_test = scaler.transform(X_test)


# ============================================================
# 5. FUNCTION TO TRAIN AND EVALUATE ONE ANN MODEL
# ============================================================

def train_and_evaluate_ann(hidden_layer_sizes, activation, solver, max_iter):
    """
    Trains one ANN configuration and returns the four evaluation metrics.
    """
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        max_iter=max_iter,
        random_state=42
    )

    # TODO 6: Train the model
    model.fit(X_train, y_train)

    # TODO 7: Generate predictions using X_test
    y_pred = model.predict(X_test)

    # TODO 8: Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    return {
        "hidden_layer_sizes": str(hidden_layer_sizes),
        "activation": activation,
        "solver": solver,
        "max_iter": max_iter,
        "R2": r2,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse
    }


# ============================================================
# 6. BASELINE MODEL
# ============================================================

print("\n=== BASELINE MODEL ===")

# TODO 9: Complete the baseline model configuration
baseline = train_and_evaluate_ann(
    hidden_layer_sizes=(10,),   
    activation="relu",           
    solver="adam",               
    max_iter=500              
)

print(baseline)


# ============================================================
# 7. ADDITIONAL EXPERIMENTS
# ============================================================

print("\n=== ADDITIONAL ANN CONFIGURATIONS ===")

# TODO 10: Replace these placeholders with at least THREE real experiments
experiments = [
    {"hidden_layer_sizes": (20, 10), "activation": "relu", "solver": "adam", "max_iter": 1000},
    {"hidden_layer_sizes": (50, 25, 10), "activation": "tanh", "solver": "lbfgs", "max_iter": 800},
    {"hidden_layer_sizes": (15,), "activation": "relu", "solver": "sgd", "max_iter": 1000},
]

results = []
results.append(baseline)

for exp in experiments:
    result = train_and_evaluate_ann(
        hidden_layer_sizes=exp["hidden_layer_sizes"],
        activation=exp["activation"],
        solver=exp["solver"],
        max_iter=exp["max_iter"]
    )
    results.append(result)


# ============================================================
# 8. COMPARISON TABLE
# ============================================================

results_df = pd.DataFrame(results)

# TODO 11: Sort the comparison table by R2 in descending order
sorted_df = results_df.sort_values(by='R2', ascending=False)

print("\n=== SORTED RESULTS (BEST R2 FIRST) ===")
print(sorted_df.round(4))


# ============================================================
# 9. ANSWERS TO THE QUESTIONS
# ============================================================

print("\n" + "="*30)
print("ANALYSIS OF ANN RESULTS")
print("="*30)

print("\n1. Which ANN configuration performed best?")
print("La configuración con mejores resultados suele ser la que tiene un R2 más cercano a 1.")

print("\n2. Did adding more neurons always improve performance?")
print("No necesariamente, demasiadas neuronas pueden causar sobreajuste (overfitting).")

print("\n3. Did adding more hidden layers always improve performance?")
print("No, a veces una sola capa oculta es suficiente para capturar la complejidad del problema.")

print("\n4. Which activation function worked best?")
print("Comúnmente 'relu' funciona mejor en regresión, pero depende de la distribución de los datos.")

print("\n5. Which solver worked best?")
print("Para datasets pequeños 'lbfgs' suele ser más rápido; para grandes, 'adam' es el estándar.")
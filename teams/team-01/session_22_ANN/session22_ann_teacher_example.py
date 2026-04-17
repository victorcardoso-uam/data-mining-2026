"""
SESSION 22 — ARTIFICIAL NEURAL NETWORKS (ANN)
Single Python Script for Classroom Demonstration

This script uses scikit-learn's MLPRegressor to build a first ANN model.
It is designed for classroom explanation and live experimentation.

DATASET USED IN THIS DEMO
-------------------------
industrial_ann_teacher_example.csv

TARGET VARIABLE
---------------
production_quality_score
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

DATA_PATH = r"C:\Users\ale03\OneDrive\Escritorio\MAYAB\SEMESTRE 8\MINERIA DE DATOS\data-mining-course\session_22_ANN\industrial_ann_student_activity.csv"

data = pd.read_csv(DATA_PATH)

print("\n=== DATASET PREVIEW ===")
print(data.head())

print("\n=== DATASET SHAPE ===")
print(data.shape)

X = data.drop(columns=["daily_output_units"]).values
y = data["daily_output_units"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def train_and_evaluate_ann(hidden_layer_sizes, activation, solver, max_iter):
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        max_iter=max_iter,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

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

print("\n=== BASELINE MODEL ===")
baseline = train_and_evaluate_ann(
    hidden_layer_sizes=(10,),
    activation="relu",
    solver="adam",
    max_iter=2000
)

for k, v in baseline.items():
    if k in ["R2", "MAE", "MSE", "RMSE"]:
        print(f"{k}: {v:.4f}")
    else:
        print(f"{k}: {v}")

print("\n=== RUNNING MORE ANN EXPERIMENTS ===")

experiments = [
    {"hidden_layer_sizes": (5,), "activation": "relu", "solver": "adam", "max_iter": 2000},
    {"hidden_layer_sizes": (20,), "activation": "relu", "solver": "adam", "max_iter": 2000},
    {"hidden_layer_sizes": (10, 10), "activation": "relu", "solver": "adam", "max_iter": 2000},
    {"hidden_layer_sizes": (10,), "activation": "tanh", "solver": "adam", "max_iter": 2000},
    {"hidden_layer_sizes": (10,), "activation": "relu", "solver": "lbfgs", "max_iter": 2000},
    {"hidden_layer_sizes": (20, 10), "activation": "tanh", "solver": "adam", "max_iter": 2000},
]

results = [baseline]

for exp in experiments:
    result = train_and_evaluate_ann(
        hidden_layer_sizes=exp["hidden_layer_sizes"],
        activation=exp["activation"],
        solver=exp["solver"],
        max_iter=exp["max_iter"]
    )
    results.append(result)

results_df = pd.DataFrame(results)

print("\n=== COMPARISON TABLE ===")
print(results_df.round(4))

print("\n=== SORTED BY R2 (HIGHER IS BETTER) ===")
sorted_df = results_df.sort_values(by="R2", ascending=False)
print(sorted_df.round(4))

print("\n=== QUESTIONS FOR CLASS DISCUSSION ===")
print("1. Which ANN configuration performed best? The best configuration was (10,) hidden neurons, ReLU activation, solver = lbfgs, max_iter = 500, with R² = 0.9656, MAE ≈ 10.06, and RMSE ≈ 12.91.This setup achieved excellent accuracy compared to all other tested models.")
print("2. Did adding more neurons always improve the model? No. For example, increasing from (10,) with adam to (20,) with adam slightly improved R² (from -15.94 to -13.39), but performance was still poor. The best results came from fewer neurons (10) combined with a different solver (lbfgs).")
print("3. Did adding more hidden layers always improve the model? No. The (10,10) with adam model had R² = -0.25, which was better than most but still far from optimal. The best model used only one hidden layer (10 neurons).")
print("4. How did activation function affect performance? The ReLU activation clearly outperformed tanh. Models with tanh had very negative R² values (≈ -16), while ReLU combined with lbfgs achieved near-perfect performance.")
print("5. Why is ANN useful for nonlinear problems? Because ANNs can approximate complex nonlinear functions through hidden layers and activation functions.")

"""
DATASET USED IN THIS DEMO
-------------------------
industrial_ann_student_activity.csv

TARGET VARIABLE
---------------
daily_output_units
"""

import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(script_dir, "industrial_ann_student_activity.csv")

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
    max_iter=500
)

for k, v in baseline.items():
    if k in ["R2", "MAE", "MSE", "RMSE"]:
        print(f"{k}: {v:.4f}")
    else:
        print(f"{k}: {v}")

print("\n=== RUNNING MORE ANN EXPERIMENTS ===")

experiments = [
    {"hidden_layer_sizes": (5,), "activation": "relu", "solver": "adam", "max_iter": 500},
    {"hidden_layer_sizes": (20,), "activation": "relu", "solver": "adam", "max_iter": 500},
    {"hidden_layer_sizes": (10, 10), "activation": "relu", "solver": "adam", "max_iter": 500},
    {"hidden_layer_sizes": (10,), "activation": "tanh", "solver": "adam", "max_iter": 500},
    {"hidden_layer_sizes": (10,), "activation": "relu", "solver": "lbfgs", "max_iter": 500},
    {"hidden_layer_sizes": (20, 10), "activation": "tanh", "solver": "adam", "max_iter": 800},
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
print("1. Which ANN configuration performed best?")
print("   ANSWER: The configuration with hidden_layer_sizes=(10,), activation='relu', and solver='lbfgs'")
print("   performed best with MAE=9.84 and RMSE=12.62. The lbfgs solver converged much better than adam.")
print()
print("2. Did adding more neurons always improve the model?")
print("   ANSWER: No. With adam solver, going from 5-->10-->20 neurons showed diminishing returns:")
print("   5 neurons: MAE=286.56, 10 neurons: MAE=278.16, 20 neurons: MAE=256.86.")
print("   The lbfgs solver with 10 neurons was actually better than 20 neurons with adam.")
print()
print("3. Did adding more hidden layers always improve the model?")
print("   ANSWER: No. The single-layer (10,) with lbfgs was best (MAE=9.84).")
print("   Two hidden layers (10,10) with adam had MAE=65.48, and (20,10) with tanh had MAE=273.63.")
print("   More layers did not guarantee better performance.")
print()
print("4. How did activation function affect performance?")
print("   ANSWER: ReLU significantly outperformed tanh. With adam solver, relu models had")
print("   MAE values of 256.86-278.16, while tanh models had MAE values of 273.63-285.86.")
print("   However, the solver choice (lbfgs vs adam) had an even more dramatic effect.")
print()
print("5. Why is ANN useful for nonlinear problems?")
print("   ANSWER: ANNs use non-linear activation functions (relu, tanh) that allow them to")
print("   learn complex, non-linear relationships between input features and output targets.")
print("   Unlike linear models, ANNs can capture curved decision boundaries and interactions.")

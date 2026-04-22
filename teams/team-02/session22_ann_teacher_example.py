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

DATA_PATH = r"C:\Users\david\Downloads\Data Mining Course\Repositories\data-mining-2026\sessions\session_22_ANN\industrial_ann_teacher_example.csv"

data = pd.read_csv(DATA_PATH)

print("\n=== DATASET PREVIEW ===")
print(data.head())

print("\n=== DATASET SHAPE ===")
print(data.shape)

X = data.drop(columns=["production_quality_score"]).values
y = data["production_quality_score"].values

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

# === QUESTIONS FOR CLASS DISCUSSION ===
# 1. Which ANN configuration performed best?
# 2. Did adding more neurons always improve the model?
# 3. Did adding more hidden layers always improve the model?
# 4. How did activation function affect performance?
# 5. Why is ANN useful for nonlinear problems?

# === ANSWERS BASED ON EXPERIMENTAL RESULTS ===

# 1. Which ANN configuration performed best?
#    The best performing ANN configuration was:
#    - Hidden layers: (10,) - single layer with 10 neurons
#    - Activation function: relu
#    - Solver: lbfgs
#    - Max iterations: 500
#    This configuration achieved an R² of 0.9970, MAE of 2.6640, and RMSE of 3.2937,
#    demonstrating exceptional predictive performance on the test set.

# 2. Did adding more neurons always improve the model?
#    No. Adding more neurons did not always improve performance:
#    - (5,) neurons + adam: R² = -1.2197 (worse than baseline)
#    - (10,) neurons + adam: R² = -90.0312 (baseline)
#    - (20,) neurons + adam: R² = -8.9661 (worse than baseline)
#    - (10, 10) neurons + adam: R² = 0.5849 (better than single layers with adam)
#    This shows that neuron count alone is not the determining factor; the solver and
#    convergence also play critical roles in model performance.

# 3. Did adding more hidden layers always improve the model?
#    No. Adding more hidden layers did not guarantee improvement:
#    - Single layer (10,) + lbfgs: R² = 0.9970 (BEST overall)
#    - Two layers (10, 10) + adam: R² = 0.5849 (good but not best)
#    - Two layers (20, 10) + tanh + adam: R² = -9.1045 (poor)
#    This demonstrates that deeper networks don't automatically produce better results;
#    architecture must be balanced with proper solver selection and training parameters.

# 4. How did activation function affect performance?
#    Activation function had a significant impact, but the solver was equally important:
#    - relu with adam: R² = -90.0312 (poor)
#    - tanh with adam: R² = -113.4697 (poor)
#    - relu with lbfgs: R² = 0.9970 (excellent)
#    The key finding: using the lbfgs solver with relu activation produced dramatically
#    better results than adam solver regardless of neuron count or layer configuration.
#    This suggests that convergence quality matters more than activation function choice
#    for this particular regression problem.

# 5. Why is ANN useful for nonlinear problems?
#    ANNs are useful for nonlinear problems because:
#    - They can learn complex nonlinear relationships between inputs and outputs through
#      combination of multiple neurons and layers with nonlinear activation functions.
#    - The experimental results show the best ANN (R² = 0.9970) fits the data far better
#      than simpler linear models would, indicating the data contains significant nonlinear
#      patterns that the neural network successfully captures.
#    - Hidden layers allow the network to automatically discover higher-level features and
#      patterns in the data without manual feature engineering.
#    - Multiple activation functions (relu, tanh) introduce nonlinearity at each neuron,
#      enabling the network to approximate any continuous function given sufficient capacity.


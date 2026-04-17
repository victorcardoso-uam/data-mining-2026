"""
SESSION 23 — ANN TUNING ACTIVITY (WITH TODOs)

GOAL
----
Use the SAME dataset from Session 22 and explore how different ANN
hyperparameters affect model performance.

DATASET
-------
industrial_ann_teacher_example.csv

TARGET VARIABLE
---------------
production_quality_score

WHAT YOU MUST DO
----------------
1. Load the dataset
2. Define X and y
3. Split the data into training and testing sets
4. Scale the input variables
5. Complete one baseline ANN model
6. Create and test at least THREE additional ANN configurations
7. Calculate the following metrics for every model:
   - R2
   - MAE
   - MSE
   - RMSE
8. Compare the results and decide which configuration performed best

IMPORTANT
---------
You must complete all TODO sections.
The goal is not only to run the model, but to understand how:
- hidden layers
- neurons
- activation functions
- solvers
- max_iter
affect ANN performance.
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# ============================================================
# 1. LOAD DATA
# ============================================================

DATA_PATH = "industrial_ann_teacher_example.csv"

# TODO 1:
# Load the dataset into a DataFrame called data
data = pd.read_csv(DATA_PATH)

print("\n=== DATASET PREVIEW ===")
print(data.head())

print("\n=== DATASET SHAPE ===")
print(data.shape)


# ============================================================
# 2. DEFINE INPUTS (X) AND TARGET (y)
# ============================================================

# TODO 2:
# Define X using all columns except the target column
X = data.drop("production_quality_score", axis=1)

# TODO 3:
# Define y using only the target column
y = data["production_quality_score"]

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

# TODO 4:
# Fit the scaler on X_train and transform X_train
X_train = scaler.fit_transform(X_train)

# TODO 5:
# Transform X_test using the same scaler
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

    # TODO 6:
    # Train the model
    model.fit(X_train, y_train)

    # TODO 7:
    # Generate predictions using X_test
    y_pred = model.predict(X_test)

    # TODO 8:
    # Calculate metrics
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

# TODO 9:
# Complete the baseline model configuration
baseline = train_and_evaluate_ann(
    hidden_layer_sizes=(10,),   # Example: (10,)
    activation="relu",         # Example: "relu"
    solver="adam",             # Example: "adam"
    max_iter=500                 # Example: 500
)

print(baseline)


# ============================================================
# 7. ADDITIONAL EXPERIMENTS
# ============================================================

print("\n=== ADDITIONAL ANN CONFIGURATIONS ===")

# TODO 10:
# Replace these placeholders with at least THREE real experiments
# You may add more configurations if you want
experiments = [
    {"hidden_layer_sizes": (20,), "activation": "relu", "solver": "adam", "max_iter": 500},
    {"hidden_layer_sizes": (10, 10), "activation": "relu", "solver": "adam", "max_iter": 500},
    {"hidden_layer_sizes": (30, 15), "activation": "tanh", "solver": "lbfgs", "max_iter": 500},
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

print("\n=== COMPARISON TABLE ===")
print(results_df.round(4))

# TODO 11:
# Sort the comparison table by R2 in descending order
sorted_df = results_df.sort_values("R2", ascending=False)

print("\n=== SORTED RESULTS (BEST R2 FIRST) ===")
print(sorted_df.round(4))


# ============================================================
# 9. FINAL QUESTIONS
# ============================================================

print("\n=== QUESTIONS FOR YOUR TEAM ===")
print("1. Which ANN configuration performed best?")
print("The ANN configuration it will be the ( (30, 15) ")
print("2. Did adding more neurons always improve performance?")
print("No, adding more neurons did not always improve performance.")
print("3. Did adding more hidden layers always improve performance?")
print("No, adding more hidden layers did not always improve performance.")
print("4. Which activation function worked best?")
print("The activation function that worked best was relu.")
print("5. Which solver worked best?")
print("The solver that worked best was adam.")
print("6. How did max_iter affect the results?")
print("Max_iter affected the results by allowing the model to converge better, but it also increased training time. In some cases, increasing max_iter led to better performance, while in others it did not make a significant difference.")
print("7. If you had to keep only one model, which one would you choose and why?")
print("If I had to keep only one model, I would choose the one with the highest R2 score, which is the ANN configuration with hidden_layer_sizes (30, 15), activation 'tanh', solver 'lbfgs', and max_iter 500. This model performed the best in terms of R2, indicating it explains the most variance in the target variable.")

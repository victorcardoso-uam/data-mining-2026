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

<<<<<<< HEAD
DATA_PATH = "industrial_ann_teacher_example.csv"
=======
DATA_PATH = r"C:\Users\ale03\OneDrive\Escritorio\MAYAB\SEMESTRE 8\MINERIA DE DATOS\data-mining-course\data-mining-2026\teams\team-01\session_22_ANN\industrial_ann_teacher_example.csv"
>>>>>>> 5b3d665207ca0bf1e39e13c75f9116e33fe91375

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
<<<<<<< HEAD
X = data.drop(columns=["production_quality_score"]).values

# TODO 3:
# Define y using only the target column
y = data["production_quality_score"].values
=======
X = data.drop(columns=["production_quality_score"])

# TODO 3:
# Define y using only the target column
y = data["production_quality_score"]

>>>>>>> 5b3d665207ca0bf1e39e13c75f9116e33fe91375

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
<<<<<<< HEAD
     model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        max_iter=max_iter,
        random_state=42
    )
"""
    Trains one ANN configuration and returns the four evaluation metrics.
    """
def train_and_evaluate_ann(hidden_layer_sizes, activation, solver, max_iter):
=======
    """
    Trains one ANN configuration and returns the four evaluation metrics.
    """

>>>>>>> 5b3d665207ca0bf1e39e13c75f9116e33fe91375
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        max_iter=max_iter,
        random_state=42
    )

    # TODO 6:
    # Train the model
<<<<<<< HEAD
def train_and_evaluate_ann(hidden_layer_sizes, activation, solver, max_iter):
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        max_iter=max_iter,
        random_state=42
    )

    # TODO 7:
    # Generate predictions using X_test
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
=======
    model.fit(X_train, y_train)

    # TODO 7:
    # Generate predictions using X_test
    y_pred = model.predict(X_test)
>>>>>>> 5b3d665207ca0bf1e39e13c75f9116e33fe91375

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
<<<<<<< HEAD
    hidden_layer_sizes=(10,),  # Example: (10,)
    activation="relu",           # Example: "relu"
    solver="adam",               # Example: "adam"
    max_iter=500              # Example: 500
=======
    hidden_layer_sizes=(10,),
    activation="relu",
    solver="adam",
    max_iter=500
>>>>>>> 5b3d665207ca0bf1e39e13c75f9116e33fe91375
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
<<<<<<< HEAD
    {"hidden_layer_sizes": (20,), "activation": "relu", "solver": "adam", "max_iter": 500},
    {"hidden_layer_sizes": (10, 5), "activation": "tanh", "solver": "lbfgs", "max_iter": 500},
    {"hidden_layer_sizes": (15, 10, 5), "activation": "logistic", "solver": "sgd", "max_iter": 500},
=======
    {"hidden_layer_sizes": (20,), "activation": "relu", "solver": "adam", "max_iter": 1000},
    {"hidden_layer_sizes": (10, 10), "activation": "tanh", "solver": "adam", "max_iter": 1000},
    {"hidden_layer_sizes": (5, 5), "activation": "logistic", "solver": "lbfgs", "max_iter": 500},
>>>>>>> 5b3d665207ca0bf1e39e13c75f9116e33fe91375
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
sorted_df = results_df.sort_values(by="R2", ascending=False)
<<<<<<< HEAD
print(sorted_df.round(4))

=======
>>>>>>> 5b3d665207ca0bf1e39e13c75f9116e33fe91375

print("\n=== SORTED RESULTS (BEST R2 FIRST) ===")
print(sorted_df.round(4))


# ============================================================
# 9. FINAL QUESTIONS
# ============================================================

print("\n=== QUESTIONS FOR YOUR TEAM ===")
<<<<<<< HEAD
print("1. Which ANN configuration performed best?")
print("2. Did adding more neurons always improve performance?")
print("3. Did adding more hidden layers always improve performance?")
print("4. Which activation function worked best?")
print("5. Which solver worked best?")
print("6. How did max_iter affect the results?")
print("7. If you had to keep only one model, which one would you choose and why?")
=======
print("1. Which ANN configuration performed best? The configuration with the highest R² in the sorted results table performed best. Check the first row of the sorted table for the best ")
print("2. Did adding more neurons always improve performance? Not always. Sometimes more neurons can help, but too many can lead to overfitting or diminishing returns. Compare the R², MAE, MSE, and RMSE for (10,) vs (20,).")
print("3. Did adding more hidden layers always improve performance? No. More layers can help capture complex patterns, but can also make training harder and may not always improve results. Compare (10,) vs (10, 10) and (5, 5).")
print("4. Which activation function worked best? Check the best R² row for the activation used. Typically, relu or tanh perform well, but your results may vary.")
print("5. Which solver worked best? Again, check the best R² row for the solver used. adam is often robust, but lbfgs can work well for small datasets.")
print("6. How did max_iter affect the results? Higher max_iter allows more training and can improve results if the model hasn't converged, but after a point, it may not help much. Compare models with 500 vs 1000 iterations.")
print("7. If you had to keep only one model, which one would you choose and why? Choose the model with the highest R² and lowest errors (MAE, MSE, RMSE) from the sorted table, as it generalizes best to unseen data.")
>>>>>>> 5b3d665207ca0bf1e39e13c75f9116e33fe91375

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

DATA_PATH = r"C:\Users\david\Downloads\Data Mining Course\Repositories\data-mining-2026\sessions\session_22_ANN\industrial_ann_teacher_example.csv"

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
X = data.drop(columns=["production_quality_score"])

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
    activation="relu",           # Example: "relu"
    solver="adam",               # Example: "adam"
    max_iter=500              # Example: 500
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
    {"hidden_layer_sizes": (10,), "activation": "relu", "solver": "adam", "max_iter": 500},
    {"hidden_layer_sizes": (20, 10), "activation": "tanh", "solver": "lbfgs", "max_iter": 500},
    {"hidden_layer_sizes": (30, 20, 10), "activation": "relu", "solver": "adam", "max_iter": 500},
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

print("\n=== SORTED RESULTS (BEST R2 FIRST) ===")
print(sorted_df.round(4))


# ============================================================
# 9. FINAL QUESTIONS
# ============================================================

"""
=== ANSWERS TO YOUR TEAM QUESTIONS ===

1. Which ANN configuration performed best?
ANSWER: Model 2 with hidden_layer_sizes=(20,10), activation=tanh, and solver=lbfgs performed best with an R2 of 0.7878, significantly outperforming all other configurations.

2. Did adding more neurons always improve performance?
ANSWER: No. Model 1 had 10 neurons and performed poorly (R2=-90.03). Model 2 with 30 total neurons (20+10) performed best (R2=0.7878). Model 3 with 60 total neurons (30+20+10) performed worse than Model 2 (R2=-0.0138). This shows that simply adding more neurons does not guarantee better performance.

3. Did adding more hidden layers always improve performance?
ANSWER: No. Model 1 with 1 hidden layer had R2=-90.03. Model 2 with 2 hidden layers had the best performance (R2=0.7878). Model 3 with 3 hidden layers performed worse (R2=-0.0138). Adding more layers actually degraded performance in this case.

4. Which activation function worked best?
ANSWER: Tanh activation function worked best, achieving R2=0.7878 in Model 2. The ReLU activation used in Models 1 and 3 resulted in much worse performance (R2=-90.03 and R2=-0.0138 respectively).

5. Which solver worked best?
ANSWER: The lbfgs solver worked best, achieving R2=0.7878 in Model 2. The adam solver used in Models 1 and 3 resulted in poor performance (R2=-90.03 and R2=-0.0138). However, lbfgs showed convergence warnings, suggesting it needed more iterations.

6. How did max_iter affect the results?
ANSWER: All models used max_iter=500. Convergence warnings appeared for adam solver (Models 1 and 3) and lbfgs solver (Model 2), indicating that 500 iterations was insufficient for full convergence. Despite not fully converging, Model 2 still achieved good results, suggesting that increasing max_iter might further improve performance for other models as well.

7. If you had to keep only one model, which one would you choose and why?
ANSWER: Model 2 with hidden_layer_sizes=(20,10), activation=tanh, and solver=lbfgs. This model achieved the highest R2 score of 0.7878, the lowest MAE of 3.72, and the lowest RMSE of 5.04. It clearly dominates all other configurations in all metrics and is the only model that achieves good predictive performance.
"""

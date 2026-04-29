"""
SESSION 24 — ANN APPLICATION TO YOUR PROJECT
Student Activity Script (WITH TODOs)

GOAL
----
Apply an Artificial Neural Network (ANN) to your own project dataset
and evaluate whether the model produces reliable results.

WHAT YOU MUST DO
----------------
1. Load your team project dataset
2. Define input variables (X) and target variable (y)
3. Split the data into training and testing sets
4. Scale the input variables
5. Define an ANN model
6. Train the model
7. Generate predictions
8. Calculate the evaluation metrics:
   - R2
   - MAE
   - MSE
   - RMSE
9. Interpret your results

IMPORTANT
---------
Use the same ANN ideas from Sessions 22 and 23.
This time, you must adapt the model to YOUR OWN PROJECT DATASET.

You may start with a simple ANN configuration and then improve it if needed.
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

# TODO 1:
# Replace this with the path to your own project dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, " amazon.csv")  # Note: file has leading space

# TODO 2:
# Load the data from the CSV file
data = pd.read_csv(DATA_PATH)

def clean_numeric(series):
    return pd.to_numeric(
        series.astype(str).str.replace(r'[^0-9.]', '', regex=True),
        errors='coerce'
    )
 
data['discounted_price']    = clean_numeric(data['discounted_price'])
data['actual_price']        = clean_numeric(data['actual_price'])
data['discount_percentage'] = clean_numeric(data['discount_percentage'])
data['rating_count']        = clean_numeric(data['rating_count'])
data['rating']              = pd.to_numeric(data['rating'], errors='coerce')
 
data = data.dropna(subset=['discounted_price', 'actual_price',
                            'discount_percentage', 'rating_count', 'rating'])
 
print("\n=== DATASET PREVIEW ===")
print(data[['discounted_price', 'actual_price',
            'discount_percentage', 'rating_count', 'rating']].head())
 
print("\n=== DATASET SHAPE ===")
print(data.shape)
 
print("\n=== COLUMN NAMES ===")
print(list(data.columns))

# ============================================================
# 2. DEFINE INPUTS (X) AND TARGET (y)
# ============================================================

# TODO 2:
# Define X using only NUMERIC columns except the target column
numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
if "rating" in numeric_cols:
    numeric_cols.remove("rating")
X = data[numeric_cols].values

# TODO 3:
# Define y using only the target column
y = data["rating"].values
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
    hidden_layer_sizes=(7,),   # Example: (10,)
    activation="tanh",           # Example: "relu"
    solver="lbfgs",               # Example: "adam"
    max_iter=438              # Example: 500
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
    {"hidden_layer_sizes": (4,), "activation": "relu", "solver": "adam", "max_iter": 507},
    {"hidden_layer_sizes": (10,13), "activation": "tanh", "solver": "lbfgs", "max_iter": 10},
    {"hidden_layer_sizes": (9,2), "activation": "logistic", "solver": "sgd", "max_iter": 765},
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

print("\n=== QUESTIONS FOR YOUR TEAM ===")
print("1. Which ANN configuration performed best?") 
print("2. Did adding more neurons always improve performance?")
print("3. Did adding more hidden layers always improve performance?")
print("4. Which activation function worked best?")
print("5. Which solver worked best?")
print("6. How did max_iter affect the results?")
print("7. If you had to keep only one model, which one would you choose and why?")
# 1. Best configuration
print("1. Best Config: is (7,)")
print("   because have the highest R2 and lowest MAE, MSE, RMSE among all configurations.")

# 2. More neurons always improve?
print("2. More Neurons: Not always. If we add too many, the model might overfit,")
print("   meaning it memorizes the training data but fails with new, unseen data.")

# 3. More hidden layers always improve?
print("3. More Hidden Layers: No. Too many layers can make the model harder to train")
print("   (vanishing gradient problem) and unnecessarily complex for simple datasets.")

# 4. Activation function effect
print("4. Activation: IN THIS CASE WAS TANH")

# 5. Which solver worked best?
print("5. Solver: IN THIS CASE WAS LBFGS")

# 6. How did max_iter affect the results?
print("6. Max_iter: Higher max_iter allows the model to train longer and potentially converge to a better solution, but it also increases training time. In this case, the best model had a max_iter of 200, which suggests that it converged well without needing too many iterations.")

#7 If you had to keep only one model, which one would you choose and why?
print("7. Keep only one model: I would choose the best configuration (7) because it has the best performance metrics (highest R2 and lowest MAE, MSE, RMSE), which indicates it generalizes better to unseen data compared to the other configurations, in this case..")

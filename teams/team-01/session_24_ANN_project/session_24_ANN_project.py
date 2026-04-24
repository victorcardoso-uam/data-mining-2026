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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# ============================================================
# 1. LOAD DATA
# ============================================================


# Use the mat.csv dataset in the root folder
DATA_PATH = r"mat.csv"


# Load the dataset
data = pd.read_csv(DATA_PATH)

print("\n=== DATASET PREVIEW ===")
print(data.head())

print("\n=== DATASET SHAPE ===")
print(data.shape)

print("\n=== COLUMN NAMES ===")
print(list(data.columns))


# ============================================================
# 2. DEFINE INPUTS (X) AND TARGET (y)
# ============================================================


# Use G3 (final grade) as the target
TARGET_COLUMN = "G3"


# For simplicity, use only numeric columns as predictors (drop categorical columns)
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols.remove(TARGET_COLUMN)
X = data[numeric_cols].values


y = data[TARGET_COLUMN].values


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================


# Use 20% test size and random_state=42
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ============================================================
# 4. SCALE INPUT FEATURES
# ============================================================

scaler = StandardScaler()


X_train = scaler.fit_transform(X_train)


X_test = scaler.transform(X_test)


# ============================================================
# 5. DEFINE ANN MODEL
# ============================================================


# Simple ANN configuration (can be tuned later)
model = MLPRegressor(
    hidden_layer_sizes=(10,),
    activation="relu",
    solver="adam",
    max_iter=500,
    random_state=42
)

experiments = [
    {"hidden_layer_sizes": (10,), "activation": "relu", "solver": "adam", "max_iter": 500},
    {"hidden_layer_sizes": (20, 10), "activation": "relu", "solver": "adam", "max_iter": 500},
    {"hidden_layer_sizes": (30, 20, 10), "activation": "relu", "solver": "adam", "max_iter": 500},
]

results = []
results.append(model)

for exp in experiments:
    result = MLPRegressor(
        hidden_layer_sizes=exp["hidden_layer_sizes"],
        activation=exp["activation"],
        solver=exp["solver"],
        max_iter=exp["max_iter"]
    )
    results.append(result)
# ============================================================
# 6. TRAIN MODEL
# ============================================================


model.fit(X_train, y_train)


# ============================================================
# 7. GENERATE PREDICTIONS
# ============================================================


y_pred = model.predict(X_test)


# ============================================================
# 8. CALCULATE EVALUATION METRICS
# ============================================================


r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\n=== MODEL RESULTS ===")
print("R2  :", round(r2, 4))
print("MAE :", round(mae, 4))
print("MSE :", round(mse, 4))
print("RMSE:", round(rmse, 4))


# ============================================================
# 9. TEAM INTERPRETATION
# ============================================================

print("\n=== QUESTIONS FOR YOUR TEAM ===")
print("1. What dataset did you use? The dataset used is mat.csv, which contains student information and grades (395 rows × 33 columns).")
print("2. What is your target variable? The target variable is G3 (final grade).")
print("3. Which ANN configuration did you choose? A simple MLPRegressor with one hidden layer of 10 neurons, ReLU activation, Adam solver, max_iter=500, and random_state=42.")
print("4. What do the evaluation metrics tell you about the model? R2: 0.5404 (the model explains about 54% of the variance in the target). MAE: 2.3789 (on average, predictions are off by about 2.38 grade points). MSE: 9.4242, RMSE: 3.0699 (average squared error and its root, indicating moderate prediction error). ")
print("5. Do you think the model is reliable for your project? The reliability depends on the specific requirements and acceptable error margins. The model has moderate predictive power (R2 ~0.54), but the error (MAE/RMSE) suggests predictions can be off by several grade points. It is somewhat reliable but not highly accurate.")
print("6. If you had more time, what would you improve? Potential improvements include hyperparameter tuning, feature engineering, and trying different architectures. Tune the ANN (more layers/neurons, different hyperparameters).Try feature engineering or include categorical variables. Test other models and compare results. Address convergence warnings by increasing max_iter or adjusting learning rate.")

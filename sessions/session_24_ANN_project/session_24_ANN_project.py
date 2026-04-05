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

# TODO 1:
# Replace this with the path to your own project dataset
DATA_PATH = "your_project_dataset.csv"

# TODO 2:
# Load the dataset into a DataFrame called data
data = ?

print("\n=== DATASET PREVIEW ===")
print(data.head())

print("\n=== DATASET SHAPE ===")
print(data.shape)

print("\n=== COLUMN NAMES ===")
print(list(data.columns))


# ============================================================
# 2. DEFINE INPUTS (X) AND TARGET (y)
# ============================================================

# TODO 3:
# Replace with the exact name of your target column
TARGET_COLUMN = "your_target_column"

# TODO 4:
# Define X using all predictor columns
# You may use:
# data.drop(columns=[TARGET_COLUMN]).values
X = ?

# TODO 5:
# Define y using only the target column
y = ?


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

# TODO 6:
# You may keep test_size=0.2 and random_state=42
# or modify them if justified
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=?, random_state=?
)


# ============================================================
# 4. SCALE INPUT FEATURES
# ============================================================

scaler = StandardScaler()

# TODO 7:
# Fit the scaler on X_train and transform X_train
X_train = ?

# TODO 8:
# Transform X_test using the same scaler
X_test = ?


# ============================================================
# 5. DEFINE ANN MODEL
# ============================================================

# TODO 9:
# Complete the ANN configuration
# Suggested parameters to modify:
# - hidden_layer_sizes
# - activation
# - solver
# - max_iter
model = MLPRegressor(
    hidden_layer_sizes=?,
    activation=?,
    solver=?,
    max_iter=?,
    random_state=42
)


# ============================================================
# 6. TRAIN MODEL
# ============================================================

# TODO 10:
# Train the model
?


# ============================================================
# 7. GENERATE PREDICTIONS
# ============================================================

# TODO 11:
# Generate predictions using X_test
y_pred = ?


# ============================================================
# 8. CALCULATE EVALUATION METRICS
# ============================================================

# TODO 12:
# Calculate all four evaluation metrics
r2 = ?
mae = ?
mse = ?
rmse = ?

print("\n=== MODEL RESULTS ===")
print("R2  :", round(r2, 4))
print("MAE :", round(mae, 4))
print("MSE :", round(mse, 4))
print("RMSE:", round(rmse, 4))


# ============================================================
# 9. TEAM INTERPRETATION
# ============================================================

print("\n=== QUESTIONS FOR YOUR TEAM ===")
print("1. What dataset did you use?")
print("2. What is your target variable?")
print("3. Which ANN configuration did you choose?")
print("4. What do the evaluation metrics tell you about the model?")
print("5. Do you think the model is reliable for your project?")
print("6. If you had more time, what would you improve?")

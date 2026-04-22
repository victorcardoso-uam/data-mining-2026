"""
Session 20 — Case 3
Full dataset WITH the categorical variable, USING one-hot encoding

GOAL
----
Train a linear regression model using:
- the numeric predictors
- the categorical predictor after one-hot encoding

What you should learn
---------------------
- One-hot encoding transforms categories into binary numeric columns.
- This allows regression models to use categorical variables correctly.
- The evaluation metrics can now be compared against Case 1.

YOUR TASK
---------
Complete the TODO sections below.
Do not change the general structure of the script.
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "categorical_regression_production.csv"

# TODO 1:
# Write the target column name
TARGET = ?

# TODO 2:
# Write the categorical column name
CATEGORICAL_COLUMN = ?

TEST_SIZE = 0.20
RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print("\n=== DATASET PREVIEW ===")
print(df.head())

print("\n=== DATA TYPES ===")
print(df.dtypes)


# ============================================================
# CASE 3
# Full dataset WITH the categorical variable, WITH one-hot encoding
# ============================================================

# TODO 3:
# Create X_case3 by removing only the target column
X_case3 = ?

# TODO 4:
# Create y using the target column
y = ?

# TODO 5:
# Apply one-hot encoding using:
# pd.get_dummies(..., columns=[...], drop_first=True)
X_case3_encoded = ?

print("\n=== DATASET AFTER ONE-HOT ENCODING ===")
print(X_case3_encoded.head())

print("\n=== PREDICTOR COLUMNS USED IN CASE 3 ===")
print(X_case3_encoded.columns.tolist())


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_case3_encoded, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)


# ============================================================
# TRAIN LINEAR REGRESSION
# ============================================================

model = LinearRegression()

# TODO 6:
# Fit the model
?

# TODO 7:
# Generate predictions on X_test
predictions = ?


# ============================================================
# EVALUATION METRICS
# ============================================================

# TODO 8:
# Calculate the four metrics used in Session 18:
# - MAE
# - MSE
# - RMSE
# - R2
mae = ?
mse = ?
rmse = ?
r2 = ?

print("\n=== CASE 3 RESULTS ===")
print("MAE :", round(mae, 4))
print("MSE :", round(mse, 4))
print("RMSE:", round(rmse, 4))
print("R2  :", round(r2, 4))


# ============================================================
# TEAM REFLECTION
# ============================================================

print("\n=== QUESTIONS FOR YOUR TEAM ===")
print("1. Why does this case work correctly?")
print("2. What new columns were created by one-hot encoding?")
print("3. Compare these metrics with Case 1. Did performance improve?")
print("4. What does that tell you about the predictive value of the categorical variable?")

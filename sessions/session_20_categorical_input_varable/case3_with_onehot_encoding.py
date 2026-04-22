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
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# CONFIGURATION
# ============================================================

# Get the directory of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "categorical_regression_production.csv")

# TODO 1:
# Write the target column name
TARGET = "daily_output_units"

# TODO 2:
# Write the categorical column name
CATEGORICAL_COLUMN = "shift"

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
X_case3 = df.drop([TARGET], axis=1)

# TODO 4:
# Create y using the target column
y = df[TARGET]

# TODO 5:
# Apply one-hot encoding using:
# pd.get_dummies(..., columns=[...], drop_first=True)
X_case3_encoded = pd.get_dummies(X_case3, columns=[CATEGORICAL_COLUMN], drop_first=True)

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
model.fit(X_train, y_train)

# TODO 7:
# Generate predictions on X_test
predictions = model.predict(X_test)


# ============================================================
# EVALUATION METRICS
# ============================================================

# TODO 8:
# Calculate the four metrics used in Session 18:
# - MAE
# - MSE
# - RMSE
# - R2
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

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
"""
TEAM REFLECTION ANSWERS
----------------------
1. Why does this case work correctly?
    This case works because one-hot encoding transforms the categorical variable ('shift') into numeric columns, allowing the regression model to use it as input.

2. What new columns were created by one-hot encoding?
    The columns 'shift_Morning' and 'shift_Night' were created (with 'shift_Evening' as the reference due to drop_first=True).

3. Compare these metrics with Case 1. Did performance improve?
    Yes, performance improved:
    - Case 1 R2: 0.8853, MAE: 10.9304, RMSE: 13.1515
    - Case 3 R2: 0.9597, MAE: 5.8435, RMSE: 7.792

4. What does that tell you about the predictive value of the categorical variable?
    Including the categorical variable (with proper encoding) adds predictive value and improves model performance, indicating that 'shift' is an important predictor for daily output units.

Script results after running:
    - MAE: 5.8435
    - MSE: 60.7157
    - RMSE: 7.792
    - R2: 0.9597
"""

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
script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(script_dir, "categorical_regression_production.csv")

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
X_case3 = df.drop(columns=[TARGET])

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
print("   → One-hot encoding converts categorical text into binary numeric columns that LinearRegression can process.")
print("\n2. What new columns were created by one-hot encoding?")
print("   → shift_Morning and shift_Night (Evening is omitted to avoid multicollinearity).")
print("\n3. Compare these metrics with Case 1. Did performance improve?")
print("   → YES: MAE 47% better, MSE 65% better, RMSE 41% better, R² 8.4% better.")
print("\n4. What does that tell you about the predictive value of the categorical variable?")
print("   → The 'shift' variable has HIGH predictive value and significantly improves model accuracy.")

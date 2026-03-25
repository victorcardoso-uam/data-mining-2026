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

DATA_PATH = r"C:\Users\david\Downloads\Data Mining Course\Dataset Parcial 2\Data\Raw\categorical_regression_production.csv"

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
print("   ANSWER: One-hot encoding converts the categorical variable 'shift' into binary")
print("   numeric columns (shift_Morning, shift_Night), which allows linear regression to")
print("   properly use categorical data. With drop_first=True, we avoid multicollinearity")
print("   by using one category (Evening) as the baseline reference.")
print()
print("2. What new columns were created by one-hot encoding?")
print("   ANSWER: The 'shift' column with values (Evening, Morning, Night) was transformed")
print("   into two binary columns: 'shift_Morning' and 'shift_Night'. Evening is the")
print("   baseline (dropped) category represented when both columns are False.")
print()
print("3. Compare these metrics with Case 1. Did performance improve?")
print("   ANSWER: Case 3 has excellent metrics (R² = 0.9597, RMSE = 7.792), indicating")
print("   that including the categorical variable 'shift' improves the model. Case 1")
print("   (without shift) likely had lower R² and higher RMSE, proving that the shift")
print("   variable adds important predictive information.")
print()
print("4. What does that tell you about the predictive value of the categorical variable?")
print("   ANSWER: The high R² (0.9597) and low RMSE demonstrate that the 'shift' variable")
print("   has strong predictive value. Different work shifts significantly influence daily")
("   output units. This categorical feature should definitely be included in the model.")


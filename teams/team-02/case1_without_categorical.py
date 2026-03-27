"""
Session 20 — Case 1
Full dataset WITHOUT the categorical variable

GOAL
----
Train a linear regression model using only the numeric predictors.
In this case, you must EXCLUDE the categorical variable completely.

What you should learn
---------------------
- A regression model can run correctly when all inputs are numeric.
- Ignoring a categorical variable may reduce model performance.
- The evaluation metrics will help you compare this case with the others.

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
# CASE 1
# Full dataset WITHOUT the categorical variable
# ============================================================

# TODO 3:
# Create X_case1 by removing:
# - the target column
# - the categorical column
X_case1 = df.drop(columns=[TARGET, CATEGORICAL_COLUMN])

# TODO 4:
# Create y using the target column
y = df[TARGET]

print("\n=== PREDICTOR COLUMNS USED IN CASE 1 ===")
print(X_case1.columns.tolist())


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_case1, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)


# ============================================================
# TRAIN LINEAR REGRESSION
# ============================================================

model = LinearRegression()

# TODO 5:
# Fit the model
model.fit(X_train, y_train)

# TODO 6:
# Generate predictions on X_test
predictions = model.predict(X_test)


# ============================================================
# EVALUATION METRICS
# ============================================================

# TODO 7:
# Calculate the four metrics used in Session 18:
# - MAE
# - MSE
# - RMSE
# - R2
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print("\n=== CASE 1 RESULTS ===")
print("MAE :", round(mae, 4))
print("MSE :", round(mse, 4))
print("RMSE:", round(rmse, 4))
print("R2  :", round(r2, 4))


# ============================================================
# TEAM REFLECTION
# ============================================================

print("\n=== QUESTIONS FOR YOUR TEAM ===")
print("1. Which predictors were used in this case?")
print("2. Why was the categorical variable excluded?")
print("3. Do you think excluding that variable may reduce performance?")
print("4. Keep these metrics to compare them later with Case 3.")


# ============================================================
# ANSWERS TO REFLECTION QUESTIONS
# ============================================================

# 1. Which predictors were used in this case?
#    ANSWER: Four numeric predictors were used:
#    - production_hours
#    - machine_temperature_c
#    - operator_experience_years
#    - material_hardness_index
#    The categorical variable 'shift' was intentionally excluded.

# 2. Why was the categorical variable excluded?
#    ANSWER: To demonstrate Case 1 as a baseline model using ONLY numeric features.
#    By excluding the 'shift' variable, we can later compare performance with
#    Case 3 (which includes 'shift' after one-hot encoding) to see if the
#    categorical variable improves the model.

# 3. Do you think excluding that variable may reduce performance?
#    ANSWER: YES. The R² of 0.8853 in Case 1 is lower than Case 3's R² of 0.9597.
#    This 7.4% improvement in R² shows that the categorical variable 'shift'
#    contains important predictive information. Excluding it reduces model fit.

# 4. Keep these metrics to compare them later with Case 3.
#    ANSWER: Case 1 Metrics (WITHOUT categorical variable):
#    - MAE:  10.9304 (higher error than Case 3)
#    - MSE:  172.9613 (higher error variance than Case 3)
#    - RMSE: 13.1515 (larger typical prediction error than Case 3)
#    - R²:   0.8853 (explains 88.53% of variance)
#
#    Case 3 Metrics (WITH categorical variable via one-hot encoding):
#    - MAE:  5.8435 (47% lower error!)
#    - MSE:  60.7157 (65% lower error variance!)
#    - RMSE: 7.792 (41% lower typical error!)
#    - R²:   0.9597 (explains 95.97% of variance - 8% improvement!)
#
#    CONCLUSION: Including the properly-encoded categorical variable significantly
#    improves model performance across all metrics.

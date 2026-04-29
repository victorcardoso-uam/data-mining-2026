"""
Session 20 — Case 2
Full dataset WITH the categorical variable, but WITHOUT one-hot encoding

GOAL
----
Try to use the categorical variable directly in a linear regression model.

What you should learn
---------------------
- Standard linear regression in scikit-learn requires numeric inputs.
- A categorical variable stored as text (object/string) cannot be used directly.
- This case is expected to FAIL, and that failure is part of the learning objective.

YOUR TASK
---------
Complete the TODO sections below.
Run the script and observe the error carefully.
Then answer the reflection questions at the end.
"""

import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# ============================================================
# CONFIGURATION
# ============================================================

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(script_dir, "categorical_regression_production.csv")

# TODO 1:
# Write the target column name
TARGET = "daily_output_units"

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
# CASE 2
# Full dataset WITH the categorical variable, WITHOUT encoding
# ============================================================

# TODO 2:
# Create X_case2 by removing ONLY the target column.
# Important:
# Do NOT remove the categorical variable in this case.
X_case2 = df.drop(columns=[TARGET])

# TODO 3:
# Create y using the target column
y = df[TARGET]

print("\n=== PREDICTOR COLUMNS USED IN CASE 2 ===")
print(X_case2.columns.tolist())


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_case2, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)


# ============================================================
# TRAIN LINEAR REGRESSION
# ============================================================

model = LinearRegression()

# TODO 4:
# Try fitting the model.
# The script is expected to fail here.
model.fit(X_train, y_train)

# TODO 5:
# Try generating predictions.
# This line may also fail depending on the previous error.
predictions = model.predict(X_test)


# ============================================================
# TEAM REFLECTION
# ============================================================

print("\n=== QUESTIONS FOR YOUR TEAM ===")
print("1. What error message appeared?")
print("   → ValueError: could not convert string to float: 'Night'")
print("\n2. Why does the model fail in this case?")
print("   → LinearRegression requires numeric inputs; it cannot process text/string values directly.")
print("\n3. What type of data is stored in the categorical variable?")
print("   → String/text values: 'Morning', 'Evening', 'Night' (object dtype in pandas).")
print("\n4. What must be done before using that variable in regression?")
print("   → Apply encoding (e.g., one-hot encoding) to convert categories into numeric binary columns.")

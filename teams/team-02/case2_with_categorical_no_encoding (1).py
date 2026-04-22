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

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = r"C:\Users\david\Downloads\Data Mining Course\Dataset Parcial 2\Data\Raw\categorical_regression_production.csv"

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

# ============================================================
# TEAM REFLECTION
# ============================================================

# 1. What error message appeared?
#    ANSWER: ValueError: could not convert string to float: 'Night'
#    The LinearRegression model attempted to convert the 'shift' column values
#    (Evening, Morning, Night) to numeric format and failed because they are text.

# 2. Why does the model fail in this case?
#    ANSWER: scikit-learn's LinearRegression requires all input features to be
#    numeric. It cannot process string/categorical data directly. The model tried
#    to fit numeric weights to text values, which is mathematically impossible.

# 3. What type of data is stored in the categorical variable?
#    ANSWER: The 'shift' variable is stored as object/string type containing
#    three categories: 'Evening', 'Morning', and 'Night'. This is qualitative
#    (non-numeric) data that needs encoding before regression.

# 4. What must be done before using that variable in regression?
#    ANSWER: The categorical variable must be encoded into numeric format.
#    Two common approaches are:
#    - Label Encoding: Convert categories to integers (0, 1, 2)
#    - One-Hot Encoding: Create binary (0/1) columns for each category
#    Case 3 uses one-hot encoding to properly include this variable.


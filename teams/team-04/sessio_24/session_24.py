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
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# ============================================================
# 1. LOAD DATA
# ============================================================

# TODO 1:
# Replace this with the path to your own project dataset
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_PATH = PROJECT_ROOT / "teams" / "team-04" / "project_dataset" / "clean_school_offenses_dataset.csv"

# TODO 2:
# Load the dataset into a DataFrame called data
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

# TODO 3:
# Replace with the exact name of your target column
TARGET_COLUMN = "Number of Schools"

# TODO 4:
# Define X using all predictor columns
# You may use:
# data.drop(columns=[TARGET_COLUMN]).values
X = data.drop(columns=[TARGET_COLUMN]).values

# TODO 5:
# Define y using only the target column
y = data[TARGET_COLUMN].values


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

# TODO 6:
# You may keep test_size=0.2 and random_state=42
# or modify them if justified
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ============================================================
# 4. SCALE INPUT FEATURES
# ============================================================

scaler = StandardScaler()

# TODO 7:
# Fit the scaler on X_train and transform X_train
X_train = scaler.fit_transform(X_train)

# TODO 8:
# Transform X_test using the same scaler
X_test = scaler.transform(X_test)


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
    hidden_layer_sizes=(50, 25),
    activation="relu",
    solver="adam",
    max_iter=500,
    random_state=42
)


# ============================================================
# 6. TRAIN MODEL
# ============================================================

# TODO 10:
# Train the model
model.fit(X_train, y_train)


# ============================================================
# 7. GENERATE PREDICTIONS
# ============================================================

# TODO 11:
# Generate predictions using X_test
y_pred = model.predict(X_test)


# ============================================================
# 8. CALCULATE EVALUATION METRICS
# ============================================================

# TODO 12:
# Calculate all four evaluation metrics
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
print("1. What dataset did you use?")
print("   Answer: clean_school_offenses_dataset.csv")
print("2. What is your target variable?")
print("   Answer: Number of Schools")
print("3. Which ANN configuration did you choose?")
print("   Answer: MLPRegressor(hidden_layer_sizes=(50, 25), activation='relu', solver='adam', max_iter=500, random_state=42)")
print("4. What do the evaluation metrics tell you about the model?")
print("   Answer: The model produced R2 = -2.7126, MAE = 2563.4636, MSE = 22169392.2287, RMSE = 4708.4384. This indicates poor predictive performance on the test data.")
print("5. Do you think the model is reliable for your project?")
print("   Answer: No. The current ANN is not reliable because negative R2 and large errors show it does not generalize well.")
print("6. If you had more time, what would you improve?")
print("   Answer: I would tune hyperparameters, try different feature/target choices, use cross-validation, and explore larger or more informative datasets.")

"""
SESSION 24 — ANN APPLICATION TO YOUR PROJECT
"""

import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# ============================================================
# 1. LOAD DATA
# ============================================================

DATA_PATH = os.path.join(os.path.dirname(__file__), "industrial_ann_student_activity.csv")

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

TARGET_COLUMN = "line_temperature_c"

X = data.drop(columns=[TARGET_COLUMN]).values
y = data[TARGET_COLUMN].values


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

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

model = MLPRegressor(
    hidden_layer_sizes=(10, 10),
    activation="relu",
    solver="adam",
    max_iter=500,
    random_state=42
)


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
print("1. What dataset did you use?")
print("2. What is your target variable?")
print("3. Which ANN configuration did you choose?")
print("4. What do the evaluation metrics tell you about the model?")
print("5. Do you think the model is reliable for your project?")
print("6. If you had more time, what would you improve?")
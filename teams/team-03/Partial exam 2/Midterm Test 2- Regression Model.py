# Midterm Test 2 — Complete Solution

import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "industrial_regression_ann_exam.csv")

TARGET = "production_efficiency_score"


# -----------------------------
# LOAD DATA (robust)
# -----------------------------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found at: {DATA_PATH}\n"
        "Make sure the CSV file is in the same folder as this script."
    )

df = pd.read_csv(DATA_PATH)

# Convert categorical variable
df = pd.get_dummies(df, columns=["shift"], drop_first=True)

X = df.drop(columns=[TARGET])
y = df[TARGET]


# -----------------------------
# TRAIN / TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# -----------------------------
# FUNCTION FOR METRICS
# -----------------------------
def evaluate_model(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return {
        "Model": name,
        "MAE": round(mae, 3),
        "MSE": round(mse, 3),
        "RMSE": round(rmse, 3),
        "R2": round(r2, 3)
    }


results = []


# =========================================================
# TASK 1 — LINEAR REGRESSION
# =========================================================
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

pred_linear = linear_model.predict(X_test)
results.append(evaluate_model("Linear Regression", y_test, pred_linear))


# =========================================================
# TASK 2 — POLYNOMIAL REGRESSION
# =========================================================
poly_model = Pipeline([
    ("poly", PolynomialFeatures(degree=2)),
    ("linear", LinearRegression())
])

poly_model.fit(X_train, y_train)
pred_poly = poly_model.predict(X_test)

results.append(evaluate_model("Polynomial Regression (deg=2)", y_test, pred_poly))


# =========================================================
# TASK 3 — NEURAL NETWORK (ANN)
# =========================================================
ann_model = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        max_iter=2000,   # ↑ para evitar warning de convergencia
        random_state=42
    ))
])

ann_model.fit(X_train, y_train)
pred_ann = ann_model.predict(X_test)

results.append(evaluate_model("Neural Network (ANN)", y_test, pred_ann))


# =========================================================
# TASK 4 — COMPARISON
# =========================================================
results_df = pd.DataFrame(results)

print("\n" + "="*70)
print("MODEL COMPARISON")
print("="*70)
print(results_df.to_string(index=False))


# Best model (by R2)
best_model = results_df.iloc[results_df["R2"].idxmax()]

print("\n" + "="*70)
print("BEST MODEL")
print("="*70)
print(best_model)


# -----------------------------
# SHORT EXPLANATION
# -----------------------------
print("\n" + "="*70)
print("EXPLANATION")
print("="*70)

print("""
Linear Regression:
- Simple and interpretable model.
- Assumes linear relationships between variables.

Polynomial Regression:
- Captures non-linear relationships.
- Can improve performance but may overfit.

Neural Network (ANN):
- Most flexible model.
- Captures complex patterns.
- Requires scaling and hyperparameter tuning.

The best model is selected based on highest R² and lowest error metrics.
""")
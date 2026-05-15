"""
=============================================================
Final Project — Data Mining
Models: Logistic Regression + Polynomial Logistic Regression
Dataset: diabetes_cleaned.csv
Team 05 - Valeria Garcia
Date: May 2026
Universidad Anahuac Mayab

Note: Since our target variable (diabetes) is binary (0 or 1),
we use Logistic Regression as the classification equivalent
of Linear Regression. Polynomial features are added to capture
non-linear relationships between variables.

What you should learn:
- Logistic Regression creates a linear decision boundary.
- Polynomial features allow capturing non-linear patterns.
- Both models are evaluated under the same conditions for fair comparison.
=============================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, mean_absolute_error, mean_squared_error,
    r2_score, confusion_matrix, ConfusionMatrixDisplay
)

# ============================================================
# CONFIGURATION
# ============================================================
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH    = os.path.join(SCRIPT_DIR, "diabetes_cleaned.csv")
TARGET_COL   = "diabetes"
FEATURE_COLS = ["age", "bmi", "HbA1c_level", "blood_glucose_level",
                "hypertension", "heart_disease"]
TEST_SIZE    = 0.20
RANDOM_STATE = 42

# ============================================================
# 1. LOAD DATA
# ============================================================
print("\n" + "="*60)
print("  STEP 1: LOADING DATASET")
print("="*60)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Make sure diabetes_cleaned.csv is in the same folder."
    )

df = pd.read_csv(DATA_PATH)
df = df.dropna()

print("\n=== DATASET PREVIEW ===")
print(df.head(10).to_string())
print("\n=== DATASET SHAPE ===")
print(f"  Rows: {df.shape[0]} | Columns: {df.shape[1]}")
print("\n=== AVAILABLE COLUMNS ===")
print(f"  {list(df.columns)}")
print("\n=== DATA TYPES ===")
print(df.dtypes)
print("\n=== TARGET DISTRIBUTION ===")
print(df[TARGET_COL].value_counts())

# ============================================================
# 2. DEFINE INPUTS (X) AND TARGET (y)
# ============================================================
print("\n" + "="*60)
print("  STEP 2: DEFINING VARIABLES")
print("="*60)
print(f"\n  Target variable (y): {TARGET_COL}")
print(f"  Input variables (X): {FEATURE_COLS}")

X = df[FEATURE_COLS]
y = df[TARGET_COL]

print("\n=== PREDICTOR COLUMNS USED ===")
print(X.columns.tolist())
print("\n=== INPUT VARIABLE STATISTICS ===")
print(X.describe().round(3).to_string())

# ============================================================
# 3. TRAIN / TEST SPLIT (80/20)
# ============================================================
print("\n" + "="*60)
print("  STEP 3: TRAIN / TEST SPLIT (80/20)")
print("="*60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print(f"\n  Training samples : {len(X_train)}")
print(f"  Testing samples  : {len(X_test)}")

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def evaluate(y_true, y_pred, model_name):
    """Calculate and print all required evaluation metrics."""
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)
    print(f"\n  >>> {model_name}")
    print(f"  {'Metric':<12} {'Value':>10}")
    print(f"  {'-'*24}")
    print(f"  {'Accuracy':<12} {acc:>10.4f}")
    print(f"  {'Precision':<12} {prec:>10.4f}")
    print(f"  {'Recall':<12} {rec:>10.4f}")
    print(f"  {'F1-Score':<12} {f1:>10.4f}")
    print(f"  {'MAE':<12} {mae:>10.4f}")
    print(f"  {'MSE':<12} {mse:>10.4f}")
    print(f"  {'RMSE':<12} {rmse:>10.4f}")
    print(f"  {'R2':<12} {r2:>10.4f}")
    return {"model": model_name, "accuracy": acc, "precision": prec,
            "recall": rec, "f1": f1, "mae": mae, "mse": mse, "rmse": rmse, "r2": r2}

def plot_results(y_true, y_pred, model_name, filename):
    """Generate Predicted vs Actual and Residual plots."""
    residuals = y_true.values - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].scatter(y_true, y_pred, alpha=0.3, color="steelblue", edgecolors="none")
    axes[0].plot([0, 1], [0, 1], "r--", lw=2, label="Perfect prediction")
    axes[0].set_xlabel("Actual Values", fontsize=12)
    axes[0].set_ylabel("Predicted Values", fontsize=12)
    axes[0].set_title(f"Predicted vs Actual\n{model_name}", fontsize=13, fontweight="bold")
    axes[0].set_xticks([0, 1]); axes[0].set_yticks([0, 1])
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].scatter(y_pred, residuals, alpha=0.3, color="darkorange", edgecolors="none")
    axes[1].axhline(0, color="r", linestyle="--", lw=2, label="Zero residual")
    axes[1].set_xlabel("Predicted Values", fontsize=12)
    axes[1].set_ylabel("Residuals (Actual - Predicted)", fontsize=12)
    axes[1].set_title(f"Residual Plot\n{model_name}", fontsize=13, fontweight="bold")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.suptitle(f"Model Evaluation — {model_name}", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Plot saved: {filename}")

def plot_confusion_matrix(y_true, y_pred, model_name, filename):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Diabetes", "Diabetes"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title(f"Confusion Matrix\n{model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Confusion matrix saved: {filename}")

# ============================================================
# 4. MODEL 1 — LOGISTIC REGRESSION (linear boundary)
# ============================================================
print("\n" + "="*60)
print("  STEP 4: MODEL 1 — LOGISTIC REGRESSION (Linear)")
print("="*60)
print("  Uses a linear decision boundary in the feature space.")
print("  Input variables are scaled with StandardScaler before training.")

linear_model = Pipeline([
    ("scaler",     StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
])
linear_model.fit(X_train, y_train)
pred_linear = linear_model.predict(X_test)

results_linear = evaluate(y_test, pred_linear, "Logistic Regression (Linear)")
plot_results(y_test, pred_linear, "Logistic Regression (Linear)", "regression_linear_plots.png")
plot_confusion_matrix(y_test, pred_linear, "Logistic Regression", "regression_linear_cm.png")

# ============================================================
# 5. MODEL 2 — POLYNOMIAL LOGISTIC REGRESSION (degree=2)
# ============================================================
print("\n" + "="*60)
print("  STEP 5: MODEL 2 — POLYNOMIAL LOGISTIC REGRESSION (Degree 2)")
print("="*60)
print("  Transforms input variables using PolynomialFeatures(degree=2).")
print("  This adds squared terms and interaction terms between variables.")
print("  Allows the model to learn non-linear decision boundaries.")

poly_model = Pipeline([
    ("poly",       PolynomialFeatures(degree=2, include_bias=False)),
    ("scaler",     StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
])
poly_model.fit(X_train, y_train)
pred_poly = poly_model.predict(X_test)

# Show how many features were created
poly_step = poly_model.named_steps["poly"]
print(f"\n  Original features  : {len(FEATURE_COLS)}")
print(f"  Features after poly: {poly_step.transform(X_train[:1]).shape[1]}")

results_poly = evaluate(y_test, pred_poly, "Polynomial Logistic Regression (Degree 2)")
plot_results(y_test, pred_poly, "Polynomial Regression (Degree 2)", "regression_polynomial_plots.png")
plot_confusion_matrix(y_test, pred_poly, "Polynomial Regression", "regression_polynomial_cm.png")

# ============================================================
# 6. FINAL COMPARISON TABLE
# ============================================================
print("\n" + "="*60)
print("  STEP 6: FINAL COMPARISON TABLE")
print("="*60)
results_df = pd.DataFrame([results_linear, results_poly]).set_index("model")
print("\n" + results_df.round(4).to_string())

# ============================================================
# 7. COMPARISON ANALYSIS — Required answers
# ============================================================
print("\n" + "="*60)
print("  STEP 7: COMPARISON ANALYSIS")
print("="*60)
print("""
  Q1: Does the polynomial model improve performance?
  A: Compare F1-Score between both models above.
     If Polynomial F1 > Linear F1 => YES, the polynomial transformation improved it.
     If similar => the relationship is mostly linear in this dataset.

  Q2: Why does this improvement happen (or not)?
  A: Polynomial features add squared terms (e.g., bmi^2) and interaction terms
     (e.g., age * HbA1c_level). This allows the model to capture patterns like:
     - A patient with HIGH bmi AND HIGH HbA1c has a disproportionately higher risk.
     - The relationship between age and diabetes risk is not purely linear.
     If no improvement: the linear model already captures the main patterns,
     and polynomial complexity does not add meaningful information.

  Q3: Which model would you recommend for deployment?
  A: The model with higher F1-Score AND lower MAE/RMSE is preferable.
     If both perform similarly, prefer the simpler linear model (faster, more interpretable).
""")

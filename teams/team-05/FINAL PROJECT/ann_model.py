"""
=============================================================
Final Project — Data Mining
Model: Artificial Neural Network (ANN) — MLPClassifier
Dataset: diabetes_cleaned.csv
Team 05 - Valeria Garcia
Date: May 2026
Universidad Anahuac Mayab

Based on Session 24 ANN Project structure.
Adapted for binary classification of diabetes.

What you must do:
1. Load your team project dataset
2. Define input variables (X) and target variable (y)
3. Split the data into training and testing sets
4. Scale the input variables (ESSENTIAL for ANN)
5. Define two ANN configurations with different parameters
6. Train both models
7. Generate predictions
8. Calculate evaluation metrics: Accuracy, Precision, Recall, F1, MAE, MSE, RMSE, R2
9. Generate Predicted vs Actual and Residual plots
10. Interpret and compare results
=============================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, mean_absolute_error, mean_squared_error,
    r2_score, confusion_matrix, ConfusionMatrixDisplay
)

# ============================================================
# CONFIGURATION
# ============================================================
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH     = os.path.join(SCRIPT_DIR, "diabetes_cleaned.csv")
TARGET_COLUMN = "diabetes"
# Features numéricas para ANN (las variables categóricas se excluyen)
FEATURE_COLS  = ["age", "bmi", "HbA1c_level", "blood_glucose_level",
                 "hypertension", "heart_disease"]

# ============================================================
# 1. LOAD DATA
# ============================================================
print("\n" + "="*60)
print("  STEP 1: LOADING DATASET")
print("="*60)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Make sure diabetes_cleaned.csv is in the same folder as this script."
    )

data = pd.read_csv(DATA_PATH)

print("\n=== DATASET PREVIEW ===")
print(data.head(10).to_string())
print("\n=== DATASET SHAPE ===")
print(f"  Shape: {data.shape}")
print("\n=== COLUMN NAMES ===")
print(f"  Columns: {list(data.columns)}")
print("\n=== DATA TYPES ===")
print(data.dtypes)

# Remover filas con valores faltantes
data = data.dropna()
print("\n=== SHAPE AFTER REMOVING NaN ===")
print(f"  Shape: {data.shape}")

print("\n=== TARGET DISTRIBUTION ===")
print(data[TARGET_COLUMN].value_counts())
print(f"  Diabetes rate: {data[TARGET_COLUMN].mean()*100:.2f}%")

# ============================================================
# 2. DEFINE INPUTS (X) AND TARGET (y)
# ============================================================
print("\n" + "="*60)
print("  STEP 2: DEFINING VARIABLES")
print("="*60)

X = data[FEATURE_COLS]
y = data[TARGET_COLUMN]

print(f"\n  Target variable (y) : {TARGET_COLUMN}")
print(f"  Input features  (X) : {FEATURE_COLS}")
print("\n=== INPUT VARIABLE STATISTICS ===")
print(X.describe().round(3).to_string())

# ============================================================
# 3. TRAIN / TEST SPLIT (80/20)
# ============================================================
print("\n" + "="*60)
print("  STEP 3: TRAIN / TEST SPLIT (80/20)")
print("="*60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n  Training samples : {len(X_train)}")
print(f"  Testing samples  : {len(X_test)}")

# ============================================================
# 4. SCALE INPUT FEATURES (required for ANN)
# ============================================================
print("\n" + "="*60)
print("  STEP 4: SCALING INPUT FEATURES")
print("="*60)
print("  Scaling is ESSENTIAL for ANN.")
print("  Without it, variables with large ranges (like blood_glucose_level)")
print("  would dominate the training and produce poor results.")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("\n  ✅ Features scaled with StandardScaler (fit on train, transform on test).")

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
    disp.plot(ax=ax, colorbar=True, cmap="Oranges")
    ax.set_title(f"Confusion Matrix\n{model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Confusion matrix saved: {filename}")

# ============================================================
# 5. ANN CONFIGURATION 1
# ============================================================
print("\n" + "="*60)
print("  STEP 5: ANN CONFIGURATION 1")
print("  hidden_layer_sizes=(100, 50) | activation=relu | max_iter=1000")
print("="*60)
print("  - 2 hidden layers: 100 neurons and 50 neurons")
print("  - ReLU activation: fast and effective for classification")
print("  - Adam optimizer: adaptive learning rate")

model_1 = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    activation="relu",
    solver="adam",
    max_iter=1000,
    random_state=42
)
model_1.fit(X_train_scaled, y_train)
y_pred_1 = model_1.predict(X_test_scaled)

print(f"  Iterations run   : {model_1.n_iter_}")
print(f"  Loss (final)     : {model_1.loss_:.6f}")

results_1 = evaluate(y_test, y_pred_1, "ANN Config 1 — (100,50) relu max_iter=1000")
plot_results(y_test, y_pred_1, "ANN Config 1", "ann_config1_plots.png")
plot_confusion_matrix(y_test, y_pred_1, "ANN Config 1", "ann_config1_cm.png")

# ============================================================
# 6. ANN CONFIGURATION 2
# ============================================================
print("\n" + "="*60)
print("  STEP 6: ANN CONFIGURATION 2")
print("  hidden_layer_sizes=(128,64,32) | activation=tanh | max_iter=2000")
print("="*60)
print("  - 3 hidden layers: 128, 64, and 32 neurons")
print("  - tanh activation: centers outputs near 0, useful for complex patterns")
print("  - More iterations allow the optimizer to find a better minimum")

model_2 = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation="tanh",
    solver="adam",
    max_iter=2000,
    random_state=42
)
model_2.fit(X_train_scaled, y_train)
y_pred_2 = model_2.predict(X_test_scaled)

print(f"  Iterations run   : {model_2.n_iter_}")
print(f"  Loss (final)     : {model_2.loss_:.6f}")

results_2 = evaluate(y_test, y_pred_2, "ANN Config 2 — (128,64,32) tanh max_iter=2000")
plot_results(y_test, y_pred_2, "ANN Config 2", "ann_config2_plots.png")
plot_confusion_matrix(y_test, y_pred_2, "ANN Config 2", "ann_config2_cm.png")

# ============================================================
# 7. FINAL COMPARISON TABLE
# ============================================================
print("\n" + "="*60)
print("  STEP 7: FINAL COMPARISON TABLE")
print("="*60)
results_df = pd.DataFrame([results_1, results_2]).set_index("model")
print("\n" + results_df.round(4).to_string())

# ============================================================
# 8. TEAM INTERPRETATION
# ============================================================
print("\n" + "="*60)
print("  STEP 8: TEAM INTERPRETATION")
print("="*60)
print("""
  Q1: What dataset did you use?
  A: diabetes_cleaned.csv — a dataset of patient health records
     used to predict whether a patient has diabetes (0 = no, 1 = yes).

  Q2: What is your target variable?
  A: diabetes (binary: 0 or 1).
     Input features: age, bmi, HbA1c_level, blood_glucose_level,
     hypertension, heart_disease.

  Q3: Which ANN configuration did you choose?
  A: Config 1 — 2 layers (100,50), relu, 1000 iterations.
     Config 2 — 3 layers (128,64,32), tanh, 2000 iterations.

  Q4: How does changing parameters affect results?
  A: More hidden layers allow the network to learn more complex patterns.
     tanh activation can help when features have both positive and negative
     interactions. More iterations give the optimizer more time to converge.
     However, too many layers or iterations can cause overfitting.

  Q5: Which configuration performs better and why?
  A: See the comparison table above. The config with higher F1-Score
     and lower MAE/RMSE is preferred.
     For medical datasets, Recall is especially important because
     we want to minimize False Negatives (missing a diabetes diagnosis).

  Q6: If you had more time, what would you improve?
  A: - Perform hyperparameter tuning (learning rate, regularization)
     - Try cross-validation for more robust evaluation
     - Handle class imbalance (diabetes cases are less frequent)
     - Add feature engineering (e.g., BMI * glucose interaction)
""")

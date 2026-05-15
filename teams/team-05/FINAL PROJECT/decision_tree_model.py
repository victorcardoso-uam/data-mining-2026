"""
=============================================================
Final Project — Data Mining
Model: Decision Tree Classifier
Dataset: diabetes_cleaned.csv
Team 05 - Valeria Garcia
Date: May 2026
Universidad Anahuac Mayab

Goal:
- Train a baseline decision tree (no constraints)
- Apply pre-pruning parameters (max_depth, min_samples_split)
- Compare both versions using metrics and plots
- Interpret results and identify best configuration
=============================================================
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, mean_absolute_error, mean_squared_error,
    r2_score, confusion_matrix, ConfusionMatrixDisplay
)

# ============================================================
# CONFIGURATION — Edit these values if needed
# ============================================================
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH    = os.path.join(SCRIPT_DIR, "diabetes_cleaned.csv")
TARGET_COL   = "diabetes"
FEATURE_COLS = ["age", "bmi", "HbA1c_level", "blood_glucose_level",
                "hypertension", "heart_disease"]
TEST_SIZE    = 0.20
RANDOM_STATE = 42

# Two pre-pruning configurations to compare
PREPRUNE_CANDIDATES = [
    {"max_depth": 3, "min_samples_split": 10,  "min_samples_leaf": 5},
    {"max_depth": 5, "min_samples_split": 20,  "min_samples_leaf": 10},
]

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
print(f"  Diabetes rate: {df[TARGET_COL].mean()*100:.2f}%")

# ============================================================
# 2. DEFINE INPUTS (X) AND TARGET (y)
# ============================================================
print("\n" + "="*60)
print("  STEP 2: DEFINING VARIABLES")
print("="*60)

X = df[FEATURE_COLS]
y = df[TARGET_COL]

print(f"\n  Target variable (y): {TARGET_COL}")
print(f"  Input variables (X): {FEATURE_COLS}")
print(f"  Total samples: {len(y)}")

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

    # Predicted vs Actual
    axes[0].scatter(y_true, y_pred, alpha=0.3, color="steelblue", edgecolors="none")
    axes[0].plot([0, 1], [0, 1], "r--", lw=2, label="Perfect prediction")
    axes[0].set_xlabel("Actual Values", fontsize=12)
    axes[0].set_ylabel("Predicted Values", fontsize=12)
    axes[0].set_title(f"Predicted vs Actual\n{model_name}", fontsize=13, fontweight="bold")
    axes[0].set_xticks([0, 1]); axes[0].set_yticks([0, 1])
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # Residual Plot
    axes[1].scatter(y_pred, residuals, alpha=0.3, color="darkorange", edgecolors="none")
    axes[1].axhline(0, color="r", linestyle="--", lw=2, label="Zero residual")
    axes[1].set_xlabel("Predicted Values", fontsize=12)
    axes[1].set_ylabel("Residuals (Actual - Predicted)", fontsize=12)
    axes[1].set_title(f"Residual Plot\n{model_name}", fontsize=13, fontweight="bold")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.suptitle(f"Model Evaluation — {model_name}", fontsize=14, y=1.02)
    plt.tight_layout()
    save_path = os.path.join(SCRIPT_DIR, filename)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Plot saved: {filename}")

def plot_confusion_matrix(y_true, y_pred, model_name, filename):
    """Generate confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Diabetes", "Diabetes"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title(f"Confusion Matrix\n{model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_path = os.path.join(SCRIPT_DIR, filename)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Confusion matrix saved: {filename}")

# ============================================================
# 4. MODEL A — BASELINE TREE (no constraints)
# ============================================================
print("\n" + "="*60)
print("  STEP 4A: BASELINE DECISION TREE (No constraints)")
print("="*60)
print("  Parameters: default (no max_depth, no min_samples_split)")

base_model = DecisionTreeClassifier(random_state=RANDOM_STATE)
base_model.fit(X_train, y_train)
pred_base = base_model.predict(X_test)

print(f"  Tree depth (baseline): {base_model.get_depth()}")
print(f"  Number of leaves     : {base_model.get_n_leaves()}")

results_base = evaluate(y_test, pred_base, "Decision Tree — Baseline (No Constraints)")
plot_results(y_test, pred_base, "DT Baseline", "dt_baseline_plots.png")
plot_confusion_matrix(y_test, pred_base, "DT Baseline", "dt_baseline_cm.png")

# ============================================================
# 5. MODEL B — PRE-PRUNED TREES
# ============================================================
print("\n" + "="*60)
print("  STEP 4B: DECISION TREE WITH COMPLEXITY CONTROL")
print("="*60)

all_results = [results_base]

for i, params in enumerate(PREPRUNE_CANDIDATES, start=1):
    print(f"\n  --- Candidate {i}: {params} ---")
    model = DecisionTreeClassifier(random_state=RANDOM_STATE, **params)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print(f"  Tree depth : {model.get_depth()}")
    print(f"  Num leaves : {model.get_n_leaves()}")

    results = evaluate(y_test, pred, f"Decision Tree — Pruned Config {i} {params}")
    plot_results(y_test, pred, f"DT Pruned Config {i}", f"dt_pruned_config{i}_plots.png")
    plot_confusion_matrix(y_test, pred, f"DT Pruned Config {i}", f"dt_pruned_config{i}_cm.png")

    # Visualize the tree (only for pruned — baseline is too large)
    fig, ax = plt.subplots(figsize=(20, 8))
    plot_tree(model, feature_names=FEATURE_COLS,
              class_names=["No Diabetes", "Diabetes"],
              filled=True, ax=ax, fontsize=9)
    ax.set_title(f"Decision Tree Structure — Config {i} (max_depth={params['max_depth']})",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, f"dt_tree_structure_config{i}.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Tree structure saved: dt_tree_structure_config{i}.png")

    all_results.append(results)

# ============================================================
# 6. FINAL COMPARISON TABLE
# ============================================================
print("\n" + "="*60)
print("  STEP 5: FINAL COMPARISON TABLE")
print("="*60)

results_df = pd.DataFrame(all_results)
results_df = results_df.set_index("model")
print("\n" + results_df.round(4).to_string())

# ============================================================
# 7. COMPARISON ANALYSIS — Required answers
# ============================================================
print("\n" + "="*60)
print("  STEP 6: COMPARISON ANALYSIS")
print("="*60)
print("""
  Q1: How did the complexity control parameter affect the model?
  A: max_depth limits how many levels the tree can grow.
     Without it (baseline), the tree grows fully and memorizes the training data.
     With max_depth=3, the tree is simpler and more interpretable.
     With max_depth=5, we allow more complexity while still controlling overfitting.
     min_samples_split prevents splits on very small groups, reducing noise sensitivity.

  Q2: Did performance improve with controlled complexity?
  A: Compare the F1-Score and Accuracy values above.
     If the pruned model has equal or higher test F1 than the baseline,
     then pruning improved generalization.
     A large gap between train and test accuracy in the baseline
     is a sign of overfitting — pruning reduces this gap.

  Q3: Which configuration performed best?
  A: See the comparison table above for the model with the highest F1-Score.
     For imbalanced datasets like diabetes, F1-Score is more informative than Accuracy.
""")

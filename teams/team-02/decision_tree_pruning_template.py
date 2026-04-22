"""
Session 15 — Decision Tree Pruning Template
Course: Data Mining (Spring 2026)

Goal:
- Train a baseline decision tree (classification OR regression)
- Apply pre-pruning parameters
- Apply cost-complexity pruning (ccp_alpha) and select alpha using validation/CV
- Compare train vs test performance and interpret results
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score


# -----------------------------
# ✅ 1) CONFIG — EDIT THESE
# -----------------------------
DATA_PATH = r"D:\Data mining course\Midterm exam1\DATA\PROCESSED\renewable_energy_cleaned_exam.csv"      # TODO
TARGET_COL = "power_kw"           # TODO
FEATURE_COLS: List[str] = ["irradiance_wm2", "humidity_pct"]    # TODO: set list or keep None (numeric auto-select)
TASK_TYPE = "regression"                          # "auto" | "classification" | "regression"

TEST_SIZE = 0.30 #this can change based on the dataset size, but 20-30% is common for test splits
RANDOM_STATE = 42

# Try at least two pre-pruning configurations
PREPRUNE_CANDIDATES = [
    {"max_depth": 3, "min_samples_leaf": 5, "min_samples_split": 10},
    {"max_depth": 5, "min_samples_leaf": 10, "min_samples_split": 20},
]

# Alpha selection (cost-complexity pruning)
N_FOLDS = 5


# -----------------------------
# Helpers
# -----------------------------
def infer_task_type(y: pd.Series) -> str:
    if y.dtype == "object" or str(y.dtype).startswith("category"):
        return "classification"
    unique = y.dropna().nunique()
    if unique <= 15 and unique / max(len(y.dropna()), 1) < 0.10:
        return "classification"
    return "regression"


def select_features(df: pd.DataFrame, target: str, feature_cols: Optional[List[str]]) -> List[str]:
    if feature_cols is not None:
        return feature_cols
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    return [c for c in numeric_cols if c != target]


@dataclass
class EvalResult:
    train_score: float
    test_score: float
    extra: str


def train_eval_classifier(X_train, X_test, y_train, y_test, **params) -> Tuple[DecisionTreeClassifier, EvalResult]:
    model = DecisionTreeClassifier(random_state=RANDOM_STATE, **params)
    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    train_acc = accuracy_score(y_train, pred_train)
    test_acc = accuracy_score(y_test, pred_test)
    cm = confusion_matrix(y_test, pred_test)

    extra = f"Confusion matrix (test):\n{cm}\n"
    return model, EvalResult(train_acc, test_acc, extra)


def train_eval_regressor(X_train, X_test, y_train, y_test, **params) -> Tuple[DecisionTreeRegressor, EvalResult]:
    model = DecisionTreeRegressor(random_state=RANDOM_STATE, **params)
    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    train_r2 = r2_score(y_train, pred_train)
    test_r2 = r2_score(y_test, pred_test)
    test_mse = mean_squared_error(y_test, pred_test)

    extra = f"Test MSE: {test_mse:.4f}\n"
    return model, EvalResult(train_r2, test_r2, extra)


def crossval_mean(model, X, y, task: str) -> float:
    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = "accuracy" if task == "classification" else "r2"
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    return float(np.mean(scores))


def main() -> None:
    print("\n=== Session 15: Decision Tree Pruning ===")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Update DATA_PATH to point to your project dataset CSV."
        )

    df = pd.read_csv(DATA_PATH)

    if TARGET_COL not in df.columns:
        raise ValueError(f"TARGET_COL '{TARGET_COL}' not found. Columns: {list(df.columns)}")

    df = df.dropna(subset=[TARGET_COL]).copy()

    features = select_features(df, TARGET_COL, FEATURE_COLS)
    if not features:
        raise ValueError("No features selected. Set FEATURE_COLS explicitly.")

    X = df[features]
    y = df[TARGET_COL]

    task = TASK_TYPE if TASK_TYPE != "auto" else infer_task_type(y)
    print(f"Task: {task}")
    print(f"Target: {TARGET_COL}")
    print(f"Features ({len(features)}): {features}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # -----------------------------
    # A) Baseline
    # -----------------------------
    print("\n--- A) Baseline Tree ---")
    if task == "classification":
        base_model, base_res = train_eval_classifier(X_train, X_test, y_train, y_test)
        score_name = "Accuracy"
    else:
        base_model, base_res = train_eval_regressor(X_train, X_test, y_train, y_test)
        score_name = "R2"

    print(f"Train {score_name}: {base_res.train_score:.4f}")
    print(f"Test  {score_name}: {base_res.test_score:.4f}")
    print(base_res.extra)

    # -----------------------------
    # B) Pre-pruning
    # -----------------------------
    print("\n--- B) Pre-pruning Trials ---")
    for i, params in enumerate(PREPRUNE_CANDIDATES, start=1):
        print(f"\nCandidate {i}: {params}")
        if task == "classification":
            _, res = train_eval_classifier(X_train, X_test, y_train, y_test, **params)
        else:
            _, res = train_eval_regressor(X_train, X_test, y_train, y_test, **params)

        print(f"Train {score_name}: {res.train_score:.4f}")
        print(f"Test  {score_name}: {res.test_score:.4f}")

    # -----------------------------
    # C) Cost-complexity pruning (ccp_alpha)
    # -----------------------------
    print("\n--- C) Cost-Complexity Pruning (ccp_alpha) ---")
    if task == "classification":
        base_for_path = DecisionTreeClassifier(random_state=RANDOM_STATE)
    else:
        base_for_path = DecisionTreeRegressor(random_state=RANDOM_STATE)

    base_for_path.fit(X_train, y_train)
    path = base_for_path.cost_complexity_pruning_path(X_train, y_train)
    alphas = path.ccp_alphas

    if len(alphas) > 1:
        alphas = alphas[:-1]

    # Keep it lightweight
    if len(alphas) > 20:
        idx = np.linspace(0, len(alphas) - 1, 20).astype(int)
        alphas = alphas[idx]

    best_alpha = None
    best_cv = -np.inf

    for a in alphas:
        if task == "classification":
            m = DecisionTreeClassifier(random_state=RANDOM_STATE, ccp_alpha=float(a))
        else:
            m = DecisionTreeRegressor(random_state=RANDOM_STATE, ccp_alpha=float(a))

        mean_cv = crossval_mean(m, X_train, y_train, task)
        if mean_cv > best_cv:
            best_cv = mean_cv
            best_alpha = float(a)

    print(f"Best alpha (CV): {best_alpha:.6f} | mean CV: {best_cv:.4f}")

    # Train final pruned model
    if task == "classification":
        _, pruned_res = train_eval_classifier(X_train, X_test, y_train, y_test, ccp_alpha=best_alpha)
    else:
        _, pruned_res = train_eval_regressor(X_train, X_test, y_train, y_test, ccp_alpha=best_alpha)

    print("\nFinal pruned results:")
    print(f"Train {score_name}: {pruned_res.train_score:.4f}")
    print(f"Test  {score_name}: {pruned_res.test_score:.4f}")
    print(pruned_res.extra)

    print("\n=== REQUIRED (write as comments in your submission) ===")
    print("1) Which model generalizes best and why?")
    print("2) Did pruning reduce overfitting? Use train vs test evidence.")
    print("3) Which settings (pre-pruning or ccp_alpha) worked best for your dataset?")


if __name__ == "__main__":
    main()

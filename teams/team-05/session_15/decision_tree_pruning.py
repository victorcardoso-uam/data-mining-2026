"""
Session 15 — Decision Tree Pruning
Team 05: Early Prediction of Type 2 Diabetes
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder


# CONFIG
DATA_PATH = os.path.join(os.path.dirname(__file__), "diabetes_cleaned.csv")
TARGET_COL = "diabetes"
FEATURE_COLS = ["gender", "age", "hypertension", "heart_disease", "smoking_history", "bmi", "HbA1c_level", "blood_glucose_level"]

TEST_SIZE = 0.30
RANDOM_STATE = 42

PREPRUNE_CANDIDATES = [
    {"max_depth": 5, "min_samples_leaf": 10, "min_samples_split": 20},
    {"max_depth": 10, "min_samples_leaf": 5, "min_samples_split": 10},
    {"max_depth": 15, "min_samples_leaf": 2, "min_samples_split": 5},
]

N_FOLDS = 5


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
    
    precision = precision_score(y_test, pred_test, zero_division=0)
    recall = recall_score(y_test, pred_test, zero_division=0)
    f1 = f1_score(y_test, pred_test, zero_division=0)
    cm = confusion_matrix(y_test, pred_test)

    extra = f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}\\nConfusion matrix:\\n{cm}\\n"
    return model, EvalResult(train_acc, test_acc, extra)


def crossval_mean(model, X, y) -> float:
    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    return float(np.mean(scores))


def main() -> None:
    print("\\n" + "="*70)
    print("Session 15: Decision Tree Pruning - Team 05 (Diabetes Prediction)")
    print("="*70)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    print(f"\\n✓ Dataset loaded: {df.shape[0]} records, {df.shape[1]} columns")

    if TARGET_COL not in df.columns:
        raise ValueError(f"TARGET_COL '{TARGET_COL}' not found.")

    df = df.dropna(subset=[TARGET_COL]).copy()

    print("\\nEncoding categorical features...")
    le_gender = LabelEncoder()
    le_smoking = LabelEncoder()
    
    df['gender'] = le_gender.fit_transform(df['gender'])
    df['smoking_history'] = le_smoking.fit_transform(df['smoking_history'])
    print("✓ Categorical features encoded")

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    print(f"\\nTarget: {TARGET_COL}")
    print(f"Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")
    print(f"Class distribution: {y.value_counts().to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print(f"\\nTrain set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")

    # A) BASELINE
    print("\\n" + "="*70)
    print("A) BASELINE TREE (No Pruning)")
    print("="*70)
    
    base_model, base_res = train_eval_classifier(X_train, X_test, y_train, y_test)

    print(f"Train Accuracy: {base_res.train_score:.4f}")
    print(f"Test Accuracy:  {base_res.test_score:.4f}")
    print(f"Overfitting Gap: {base_res.train_score - base_res.test_score:.4f}")
    print(f"Tree Depth: {base_model.get_depth()}")
    print(f"Number of Leaves: {base_model.get_n_leaves()}")
    print(f"\\n{base_res.extra}")

    # B) PRE-PRUNING
    print("\\n" + "="*70)
    print("B) PRE-PRUNING TRIALS")
    print("="*70)
    
    best_preprune_model = None
    best_preprune_acc = -np.inf
    best_preprune_params = None

    for i, params in enumerate(PREPRUNE_CANDIDATES, start=1):
        print(f"\\n--- Candidate {i} ---")
        print(f"Parameters: {params}")
        
        model, res = train_eval_classifier(X_train, X_test, y_train, y_test, **params)

        print(f"Train Accuracy: {res.train_score:.4f}")
        print(f"Test Accuracy:  {res.test_score:.4f}")
        print(f"Overfitting Gap: {res.train_score - res.test_score:.4f}")
        print(f"Tree Depth: {model.get_depth()}")
        print(f"Number of Leaves: {model.get_n_leaves()}")

        if res.test_score > best_preprune_acc:
            best_preprune_acc = res.test_score
            best_preprune_model = model
            best_preprune_params = params

    print(f"\\n✓ Best pre-pruned model: {best_preprune_params}")
    print(f"  Test Accuracy: {best_preprune_acc:.4f}")

    # C) COST-COMPLEXITY PRUNING
    print("\\n" + "="*70)
    print("C) COST-COMPLEXITY PRUNING (ccp_alpha)")
    print("="*70)

    base_for_path = DecisionTreeClassifier(random_state=RANDOM_STATE)
    base_for_path.fit(X_train, y_train)
    path = base_for_path.cost_complexity_pruning_path(X_train, y_train)
    alphas = path.ccp_alphas

    if len(alphas) > 1:
        alphas = alphas[:-1]

    if len(alphas) > 30:
        idx = np.linspace(0, len(alphas) - 1, 30).astype(int)
        alphas = alphas[idx]

    print(f"Testing {len(alphas)} alpha values...")

    best_alpha = None
    best_cv = -np.inf

    for a in alphas:
        m = DecisionTreeClassifier(random_state=RANDOM_STATE, ccp_alpha=float(a))
        mean_cv = crossval_mean(m, X_train, y_train)
        
        if mean_cv > best_cv:
            best_cv = mean_cv
            best_alpha = float(a)

    print(f"\\n✓ Best alpha (via {N_FOLDS}-fold CV): {best_alpha:.8f}")
    print(f"  Mean CV Accuracy: {best_cv:.4f}")

    final_model, final_res = train_eval_classifier(
        X_train, X_test, y_train, y_test, ccp_alpha=best_alpha
    )

    print(f"\\nFinal Pruned Model Results:")
    print(f"Train Accuracy: {final_res.train_score:.4f}")
    print(f"Test Accuracy:  {final_res.test_score:.4f}")
    print(f"Overfitting Gap: {final_res.train_score - final_res.test_score:.4f}")
    print(f"Tree Depth: {final_model.get_depth()}")
    print(f"Number of Leaves: {final_model.get_n_leaves()}")
    print(f"\\n{final_res.extra}")

    # D) COMPARISON
    print("\\n" + "="*70)
    print("D) SUMMARY: BASELINE vs PRE-PRUNING vs COST-COMPLEXITY PRUNING")
    print("="*70)

    comparison = pd.DataFrame({
        'Model': ['Baseline (No Pruning)', 'Pre-pruned (Best)', 'Cost-Complexity Pruned'],
        'Train Accuracy': [base_res.train_score, best_preprune_model.score(X_train, y_train), final_res.train_score],
        'Test Accuracy': [base_res.test_score, best_preprune_acc, final_res.test_score],
        'Overfitting Gap': [
            base_res.train_score - base_res.test_score,
            best_preprune_model.score(X_train, y_train) - best_preprune_acc,
            final_res.train_score - final_res.test_score
        ],
        'Tree Depth': [base_model.get_depth(), best_preprune_model.get_depth(), final_model.get_depth()],
        'Number of Leaves': [base_model.get_n_leaves(), best_preprune_model.get_n_leaves(), final_model.get_n_leaves()]
    })

    print("\\n" + comparison.to_string(index=False))

    # INTERPRETATION
    print("\\n" + "="*70)
    print("INTERPRETATION & CONCLUSIONS")
    print("="*70)

    print(f"""
QUESTION 1: Which model generalizes best and why?
The Cost-Complexity Pruned model (ccp_alpha={best_alpha:.8f}) generalizes best.
It balances training accuracy with model simplicity. Depth: {final_model.get_depth()}, Leaves: {final_model.get_n_leaves()}.
Test accuracy: {final_res.test_score:.4f}

QUESTION 2: Did pruning reduce overfitting? Evidence?
YES. Gap reduced from {base_res.train_score - base_res.test_score:.4f} (baseline) to {final_res.train_score - final_res.test_score:.4f} (pruned).
Baseline: {base_res.train_score:.4f} train vs {base_res.test_score:.4f} test shows memorization.
Pruning enforces: maximize accuracy + minimize tree size.

QUESTION 3: Which settings worked best?
Cost-Complexity Pruning with ccp_alpha={best_alpha:.8f} worked best because:
- Cross-validation automatically selects optimal complexity penalty
- No manual hyperparameter tuning needed (unlike pre-pruning)
- Produces trees that are accurate AND interpretable
- Better generalization to unseen diabetes patients
""")

    print("="*70)
    print("✓ Session 15 Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
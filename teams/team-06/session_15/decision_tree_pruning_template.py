"""
Session 15 — Decision Tree Pruning (Amazon Dataset)
Course: Data Mining (Spring 2026)
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
# ✅ 1) CONFIG — RUTA AUTOMÁTICA
# -----------------------------
# Esto busca el archivo en la misma carpeta que este script .py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "amazon.csv")

TARGET_COL = "rating"           
FEATURE_COLS = ["actual_price", "discount_percentage", "rating_count"] 
TASK_TYPE = "regression" 

TEST_SIZE = 0.30
RANDOM_STATE = 42

PREPRUNE_CANDIDATES = [
    {"max_depth": 3, "min_samples_leaf": 5, "min_samples_split": 10},
    {"max_depth": 5, "min_samples_leaf": 10, "min_samples_split": 20},
]

N_FOLDS = 5

# -----------------------------
# Helpers & Eval
# -----------------------------
@dataclass
class EvalResult:
    train_score: float
    test_score: float
    extra: str

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
    scoring = "r2"
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    return float(np.mean(scores))

# -----------------------------
# Main Execution
# -----------------------------
def main() -> None:
    print("\n=== Session 15: Decision Tree Pruning (Amazon) ===")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset NO encontrado en la carpeta del script.\n"
            f"Asegúrate de haber movido 'amazon.csv' a:\n{BASE_DIR}"
        )

    # CARGA DE DATOS
    df = pd.read_csv(DATA_PATH)

    print("Limpiando datos de Amazon...")
    # Limpieza de caracteres especiales
    df['rating'] = pd.to_numeric(df['rating'].str.replace('|', '', regex=False), errors='coerce')
    df['actual_price'] = df['actual_price'].str.replace('₹', '').str.replace(',', '').astype(float)
    df['discount_percentage'] = df['discount_percentage'].str.replace('%', '').astype(float)
    df['rating_count'] = df['rating_count'].str.replace(',', '').astype(float)
    
    # Eliminar nulos
    df = df.dropna(subset=[TARGET_COL] + FEATURE_COLS).copy()

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    print(f"Dataset cargado con éxito. Filas: {X.shape[0]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # A) Baseline
    print("\n--- A) Baseline Tree ---")
    base_model, base_res = train_eval_regressor(X_train, X_test, y_train, y_test)
    print(f"Train R2: {base_res.train_score:.4f} | Test R2: {base_res.test_score:.4f}")

    # B) Pre-pruning
    print("\n--- B) Pre-pruning Trials ---")
    for i, params in enumerate(PREPRUNE_CANDIDATES, start=1):
        _, res = train_eval_regressor(X_train, X_test, y_train, y_test, **params)
        print(f"Candidate {i} {params}: Train R2: {res.train_score:.4f} | Test R2: {res.test_score:.4f}")

    # C) Cost-complexity pruning (ccp_alpha)
    print("\n--- C) Cost-Complexity Pruning (ccp_alpha) ---")
    base_for_path = DecisionTreeRegressor(random_state=RANDOM_STATE)
    base_for_path.fit(X_train, y_train)
    path = base_for_path.cost_complexity_pruning_path(X_train, y_train)
    alphas = path.ccp_alphas

    if len(alphas) > 20:
        idx = np.linspace(0, len(alphas) - 1, 20).astype(int)
        alphas = alphas[idx]

    best_alpha = 0.0
    best_cv = -np.inf

    for a in alphas:
        m = DecisionTreeRegressor(random_state=RANDOM_STATE, ccp_alpha=float(a))
        mean_cv = crossval_mean(m, X_train, y_train, "regression")
        if mean_cv > best_cv:
            best_cv = mean_cv
            best_alpha = float(a)

    print(f"Best alpha (CV): {best_alpha:.6f} | mean CV R2: {best_cv:.4f}")

    # Resultado Final
    _, pruned_res = train_eval_regressor(X_train, X_test, y_train, y_test, ccp_alpha=best_alpha)
    print("\nFinal pruned results:")
    print(f"Train R2: {pruned_res.train_score:.4f} | Test R2: {pruned_res.test_score:.4f}")
    print(pruned_res.extra)

if __name__ == "__main__":
    main()
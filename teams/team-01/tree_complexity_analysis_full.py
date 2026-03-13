"""
Session 16 — Autonomous Tree Growth
Course: Data Mining (Jan–May 2026)

PURPOSE OF THIS SCRIPT
----------------------
This script helps each team analyze the trade-off between:

1) Model complexity
2) Model performance

using Decision Trees on YOUR OWN PROJECT DATASET.

The main idea of this session is:
- Let the tree grow with different values of max_depth
- Observe how training error changes
- Observe how validation error changes
- Detect the point where the tree starts to overfit
- Compare a shallow tree, the best-depth tree, and a deeper tree

WHAT THIS SCRIPT PRODUCES
-------------------------
1) A plot of Training Error vs Validation Error as max_depth changes
2) A plot showing how tree complexity grows (number of leaves)
3) A model comparison summary printed in the terminal
4) Optional 1D regression plot (if you use exactly ONE numeric feature in a regression task)
   similar to the classic step-like plots used to explain decision trees

IMPORTANT
---------
This file is intentionally long and heavily commented because:
- You are still learning how the algorithm behaves
- The comments explain the concepts that matter
- Your team should understand the logic, not just run the code

HOW TO USE IT
-------------
1) Put this file in:
   teams/team-XX/session_16/tree_complexity_analysis.py

2) Update the CONFIG section below:
   - DATA_PATH
   - TARGET_COL
   - FEATURE_COLS
   - TASK_TYPE

3) Run:
   python tree_complexity_analysis.py

4) Read the terminal output and inspect the generated plots.

5) In your final submission, keep the comments and add your own interpretation
   in a separate markdown file.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    r2_score,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


# =============================================================================
# 1) CONFIGURATION — TEAMS MUST EDIT THIS SECTION
# =============================================================================

# TODO 1:
# Replace this with the path to YOUR project dataset.
# Example:
# DATA_PATH = "datasets/session_16/my_project_data.csv"
DATA_PATH = r"C:\Users\Fernando\Desktop\Data-mining-course\student-mat.csv"

# TODO 2:
# Replace this with the exact name of the column you want to predict.
TARGET_COL = "G3"

# TODO 3:
# Replace this with the features you want to use.
# If you leave FEATURE_COLS = None, the script will use ALL columns except the target.
FEATURE_COLS: List[str] = ["failures", "absences", "studytime"]

# TODO 4:
# Choose one:
# - "auto"            -> script tries to infer classification vs regression
# - "classification"  -> target is categorical / class label
# - "regression"      -> target is numeric / continuous value
TASK_TYPE = "regression"

# Split configuration
TEST_SIZE = 0.30
RANDOM_STATE = 42

# Depth search range
MIN_DEPTH = 1
MAX_DEPTH = 15

# Three trees that will be compared in the final summary
SHALLOW_DEPTH = 2
DEEP_DEPTH = 10

# Plot output folder (optional)
SAVE_PLOTS = True
OUTPUT_DIR = "session_16_outputs"


# =============================================================================
# 2) HELPER FUNCTIONS
# =============================================================================

def infer_task_type(y: pd.Series) -> str:
    """
    Very simple heuristic to infer the problem type.

    If the target is text/category -> classification
    If the target is numeric with very few unique values -> classification
    Otherwise -> regression
    """
    if y.dtype == "object" or str(y.dtype).startswith("category") or str(y.dtype) == "bool":
        return "classification"

    unique_values = y.nunique(dropna=True)
    if unique_values <= 12 and unique_values / max(len(y), 1) < 0.10:
        return "classification"

    return "regression"


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build preprocessing steps that can handle both numeric and categorical features.

    Why do we need this?
    --------------------
    Many project datasets contain:
    - numeric columns (temperature, irradiance, cost, etc.)
    - categorical columns (city, material, customer_type, etc.)

    Decision Trees in scikit-learn need numeric inputs.
    Therefore:
    - numeric features: fill missing values with median
    - categorical features: fill missing values + one-hot encode
    """
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


def build_model(task: str, max_depth: Optional[int]) -> object:
    """
    Return the correct Decision Tree model depending on the task type.
    """
    if task == "classification":
        return DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=RANDOM_STATE
        )
    else:
        return DecisionTreeRegressor(
            max_depth=max_depth,
            random_state=RANDOM_STATE
        )


def evaluate_model(task: str, y_true, y_pred) -> Dict[str, float]:
    """
    Evaluate predictions.

    For classification:
    - accuracy
    - error = 1 - accuracy

    For regression:
    - R²
    - RMSE
    - error = RMSE

    Why use error curves?
    ---------------------
    In this session we want to visualize how error behaves as tree depth increases.
    """
    if task == "classification":
        acc = accuracy_score(y_true, y_pred)
        return {
            "score": acc,
            "error": 1 - acc
        }
    else:
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        return {
            "score": r2,
            "error": rmse
        }


def fit_tree_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    task: str,
    max_depth: Optional[int]
) -> Tuple[Pipeline, np.ndarray]:
    """
    Train a tree inside a pipeline:
    preprocessing -> decision tree

    Why pipeline?
    -------------
    It keeps preprocessing and model training in one object.
    That means:
    - the exact same transformations are applied to train and test data
    - the code is cleaner and less error-prone
    """
    preprocessor = build_preprocessor(X_train)
    model = build_model(task, max_depth=max_depth)

    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_eval)
    return pipe, preds


def get_tree_complexity(pipe: Pipeline) -> Tuple[int, int]:
    """
    Return:
    - actual tree depth
    - number of leaves

    This is important because:
    - model complexity is not only about performance
    - we also want to see how big the tree becomes
    """
    tree_model = pipe.named_steps["model"]
    return tree_model.get_depth(), tree_model.get_n_leaves()


def ensure_output_dir(path_str: str) -> Path:
    out = Path(path_str)
    out.mkdir(parents=True, exist_ok=True)
    return out


# =============================================================================
# 3) MAIN SCRIPT
# =============================================================================

def main() -> None:
    print("\n=== Session 16 — Autonomous Tree Growth Analysis ===")

    # -------------------------------------------------------------------------
    # Step A: Load dataset
    # -------------------------------------------------------------------------
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"\nDataset not found: {DATA_PATH}\n"
            "Please update DATA_PATH in the CONFIG section."
        )

    df = pd.read_csv(DATA_PATH)
    print(f"\nDataset loaded successfully: {DATA_PATH}")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))

    if TARGET_COL not in df.columns:
        raise ValueError(
            f"\nTARGET_COL '{TARGET_COL}' was not found in the dataset.\n"
            f"Available columns: {list(df.columns)}"
        )

    # -------------------------------------------------------------------------
    # Step B: Define features and target
    # -------------------------------------------------------------------------
    if FEATURE_COLS is None:
        feature_cols = [c for c in df.columns if c != TARGET_COL]
    else:
        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            raise ValueError(
                f"\nThese FEATURE_COLS are missing from the dataset: {missing}"
            )
        feature_cols = FEATURE_COLS

    X = df[feature_cols].copy()
    y = df[TARGET_COL].copy()

    # Drop rows where target is missing
    valid_mask = y.notna()
    X = X.loc[valid_mask].reset_index(drop=True)
    y = y.loc[valid_mask].reset_index(drop=True)

    # -------------------------------------------------------------------------
    # Step C: Infer or use task type
    # -------------------------------------------------------------------------
    task = TASK_TYPE if TASK_TYPE != "auto" else infer_task_type(y)

    print("\nTask type:", task)
    print("Target column:", TARGET_COL)
    print("Number of features:", len(feature_cols))
    print("Feature columns:", feature_cols)

    # -------------------------------------------------------------------------
    # Step D: Train-test split
    # -------------------------------------------------------------------------
    # This split is the foundation of generalization analysis.
    # We train on one subset and evaluate on unseen data.
    if task == "classification":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )

    print("\nTrain size:", len(X_train))
    print("Test size:", len(X_test))

    # -------------------------------------------------------------------------
    # Step E: Evaluate trees across multiple depths
    # -------------------------------------------------------------------------
    depths = list(range(MIN_DEPTH, MAX_DEPTH + 1))
    train_errors = []
    valid_errors = []
    train_scores = []
    valid_scores = []
    actual_depths = []
    leaves_counts = []

    print("\n=== Growing trees with different max_depth values ===")

    for depth in depths:
        pipe_train, train_preds = fit_tree_pipeline(X_train, y_train, X_train, task, depth)
        _, test_preds = fit_tree_pipeline(X_train, y_train, X_test, task, depth)

        train_eval = evaluate_model(task, y_train, train_preds)
        test_eval = evaluate_model(task, y_test, test_preds)

        tree_depth, n_leaves = get_tree_complexity(pipe_train)

        train_errors.append(train_eval["error"])
        valid_errors.append(test_eval["error"])
        train_scores.append(train_eval["score"])
        valid_scores.append(test_eval["score"])
        actual_depths.append(tree_depth)
        leaves_counts.append(n_leaves)

        print(
            f"max_depth={depth:2d} | "
            f"train_error={train_eval['error']:.4f} | "
            f"validation_error={test_eval['error']:.4f} | "
            f"actual_depth={tree_depth:2d} | leaves={n_leaves:3d}"
        )

    # -------------------------------------------------------------------------
    # Step F: Select best depth based on validation error
    # -------------------------------------------------------------------------
    # Important idea:
    # The best tree is NOT the one with the lowest training error.
    # It is the one that performs best on unseen data.
    best_index = int(np.argmin(valid_errors))
    best_depth = depths[best_index]

    print("\n=== Best depth according to validation error ===")
    print("Best max_depth:", best_depth)
    print("Best validation error:", valid_errors[best_index])

    # -------------------------------------------------------------------------
    # Step G: Build three final comparison models
    # -------------------------------------------------------------------------
    comparison_depths = {
        "Shallow Tree": SHALLOW_DEPTH,
        "Best Tree": best_depth,
        "Deep Tree": DEEP_DEPTH,
    }

    summary_rows = []

    for label, depth in comparison_depths.items():
        pipe_train, train_preds = fit_tree_pipeline(X_train, y_train, X_train, task, depth)
        _, test_preds = fit_tree_pipeline(X_train, y_train, X_test, task, depth)

        train_eval = evaluate_model(task, y_train, train_preds)
        test_eval = evaluate_model(task, y_test, test_preds)
        tree_depth, n_leaves = get_tree_complexity(pipe_train)

        summary_rows.append({
            "model": label,
            "requested_max_depth": depth,
            "actual_tree_depth": tree_depth,
            "n_leaves": n_leaves,
            "train_score": train_eval["score"],
            "test_score": test_eval["score"],
            "train_error": train_eval["error"],
            "test_error": test_eval["error"],
        })

    summary_df = pd.DataFrame(summary_rows)

    print("\n=== Comparison Summary ===")
    print(summary_df)

    # -------------------------------------------------------------------------
    # Step H: Plot 1 — Training vs Validation Error
    # -------------------------------------------------------------------------
    # This is the most important plot of the session.
    # We want students to observe:
    # - training error usually decreases as complexity increases
    # - validation error decreases at first, then may increase
    plt.figure(figsize=(10, 6))
    plt.plot(depths, train_errors, marker="o", label="Training Error")
    plt.plot(depths, valid_errors, marker="o", label="Validation Error")
    plt.axvline(best_depth, color="gray", linestyle="--", label=f"Best depth = {best_depth}")
    plt.xlabel("Max. Depth")
    if task == "classification":
        plt.ylabel("Prediction Error (1 - Accuracy)")
        plt.title("Training vs. Validation Error (Decision Tree Classification)")
    else:
        plt.ylabel("Prediction Error (RMSE)")
        plt.title("Training vs. Validation Error (Decision Tree Regression)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_dir = ensure_output_dir(OUTPUT_DIR)
    if SAVE_PLOTS:
        plt.savefig(out_dir / "plot_01_training_vs_validation_error.png", dpi=200)
    plt.show()

    # -------------------------------------------------------------------------
    # Step I: Plot 2 — Complexity Growth (Leaves vs max_depth)
    # -------------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(depths, leaves_counts, marker="s", color="darkgreen")
    plt.xlabel("Max. Depth")
    plt.ylabel("Number of Leaves")
    plt.title("Tree Complexity Growth")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if SAVE_PLOTS:
        plt.savefig(out_dir / "plot_02_complexity_growth.png", dpi=200)
    plt.show()

    # -------------------------------------------------------------------------
    # Step J: Optional Plot 3 — 1D Regression Visualization
    # -------------------------------------------------------------------------
    # If the team uses exactly ONE numeric feature in a regression task,
    # we can draw a classic "step-function" tree plot.
    #
    # This plot is similar to the classic examples used to explain how
    # Decision Tree Regressors create piecewise-constant predictions.
    #
    # This only works in 1D regression because the x-axis must represent
    # a single feature.
    if task == "regression" and len(feature_cols) == 1:
        single_feature = feature_cols[0]

        if pd.api.types.is_numeric_dtype(X[single_feature]):
            x_full = X[[single_feature]].copy()

            x_grid = np.linspace(
                x_full[single_feature].min(),
                x_full[single_feature].max(),
                500
            ).reshape(-1, 1)

            pipes_for_plot = {}
            for label, depth in comparison_depths.items():
                pipe, _ = fit_tree_pipeline(X_train, y_train, X_train, task, depth)
                pipes_for_plot[label] = pipe

            pred_shallow = pipes_for_plot["Shallow Tree"].predict(pd.DataFrame(x_grid, columns=[single_feature]))
            pred_best = pipes_for_plot["Best Tree"].predict(pd.DataFrame(x_grid, columns=[single_feature]))
            pred_deep = pipes_for_plot["Deep Tree"].predict(pd.DataFrame(x_grid, columns=[single_feature]))

            plt.figure(figsize=(10, 6))
            plt.scatter(
                X[single_feature],
                y,
                s=25,
                edgecolor="black",
                label="data"
            )
            plt.plot(x_grid, pred_shallow, label=f"max_depth={SHALLOW_DEPTH}", linewidth=2)
            plt.plot(x_grid, pred_best, label=f"best_depth={best_depth}", linewidth=2)
            plt.plot(x_grid, pred_deep, label=f"max_depth={DEEP_DEPTH}", linewidth=2)

            plt.xlabel(single_feature)
            plt.ylabel(TARGET_COL)
            plt.title("Decision Tree Regression — Complexity Comparison")
            plt.legend()
            plt.tight_layout()
            if SAVE_PLOTS:
                plt.savefig(out_dir / "plot_03_regression_tree_comparison.png", dpi=200)
            plt.show()

    # -------------------------------------------------------------------------
    # Step K: Save summary table
    # -------------------------------------------------------------------------
    if SAVE_PLOTS:
        summary_df.to_csv(out_dir / "model_comparison_summary.csv", index=False)

    # -------------------------------------------------------------------------
    # Step L: Interpretation prompts for teams
    # -------------------------------------------------------------------------
    print("\n=== REQUIRED INTERPRETATION FOR YOUR SUBMISSION ===")
    print("Add answers as comments at the bottom of your Python file.")
    print()
    print("1. Which tree has the lowest training error?")
    print("2. Which tree has the lowest validation/test error?")
    print("3. At what depth does validation performance stop improving?")
    print("4. Does your dataset show overfitting as tree depth increases?")
    print("5. Which tree would you choose for deployment, and why?")
    print("6. Explain the trade-off between model complexity and performance in your own dataset.")
    print()

    print("If SAVE_PLOTS = True, your plots and summary table were saved in:")
    print(out_dir.resolve())
#Answer of the questions:
# 1. Deep tree has the lowest training error whit (3.404090)
# 2. shallo tree and best tree have the lowest test error with (4.242425)
# 3. validation performance stops improving at depht 2
# 4. not really, the validation error is quite similar across all depths.
# 5. I would choose the shallow tree, because has the lower test error


if __name__ == "__main__":
    main()

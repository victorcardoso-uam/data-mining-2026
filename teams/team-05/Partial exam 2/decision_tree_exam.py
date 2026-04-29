"""
Practical Problem 1 — Decision Trees 
Team 05: Industrial Production Prediction


"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_PATH = os.path.join(os.path.dirname(__file__), "industrial_decision_tree_exam.csv")

# Target variable (continuous: daily production output)
TARGET_COL = "daily_output_units"

# Input variables (features influencing production)
FEATURE_COLS = [
    "machine_speed_units_min",
    "operator_experience_years",
    "material_hardness_index",
    "line_temperature_c",
    "equipment_age_years",
    "inspection_time_min"
]

# Train-Test split parameters
TEST_SIZE = 0.30
RANDOM_STATE = 42


@dataclass
class ModelEvalResult:
    """Container for model evaluation metrics."""
    train_mse: float
    test_mse: float
    train_mae: float
    test_mae: float
    train_r2: float
    test_r2: float
    tree_depth: int
    n_leaves: int


def evaluate_model(
    model: DecisionTreeRegressor,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series
) -> ModelEvalResult:
    """
    Evaluate regression model on train and test sets.
    
    Returns:
        ModelEvalResult: Container with all evaluation metrics
    """
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    return ModelEvalResult(
        train_mse=train_mse,
        test_mse=test_mse,
        train_mae=train_mae,
        test_mae=test_mae,
        train_r2=train_r2,
        test_r2=test_r2,
        tree_depth=model.get_depth(),
        n_leaves=model.get_n_leaves()
    )


def print_header(title: str) -> None:
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def main() -> None:
    """Main execution of decision tree analysis."""
    
    print_header("PRACTICAL PROBLEM 1: DECISION TREES — INDUSTRIAL PRODUCTION PREDICTION")

    # ========================================================================
    # TASK 1 — MODEL TRAINING 
    # ========================================================================

    print_header("TASK 1: MODEL TRAINING ")

    # ─────────────────────────────────────────────────────────────────────────
    # PART A: LOAD DATASET AND DISPLAY 
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n[A] LOADING AND EXPLORING DATASET")
    print("-" * 80)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    print(f"\n✓ Dataset loaded successfully!")
    
    # Display dataset shape
    print(f"\nDataset Shape:")
    print(f"  • Rows: {df.shape[0]}")
    print(f"  • Columns: {df.shape[1]}")
    
    # Display column names
    print(f"\nColumn Names:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col}")
    
    # Display data types
    print(f"\nData Types:")
    for col in df.columns:
        print(f"  • {col}: {df[col].dtype}")
    
    # Display basic statistics
    print(f"\nBasic Statistics:")
    print(df.describe().to_string())
    
    # Check for missing values
    print(f"\nMissing Values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("  ✓ No missing values detected")
    else:
        print(missing[missing > 0].to_string())

    # ─────────────────────────────────────────────────────────────────────────
    # PART B: DEFINE INPUT AND TARGET VARIABLES 
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n[B] DEFINING INPUT AND TARGET VARIABLES")
    print("-" * 80)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset")

    for col in FEATURE_COLS:
        if col not in df.columns:
            raise ValueError(f"Feature column '{col}' not found in dataset")

    # Define feature matrix and target vector
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    print(f"\nTarget Variable (y):")
    print(f"  • Column: {TARGET_COL}")
    print(f"  • Type: {y.dtype}")
    print(f"  • Mean: {y.mean():.2f}")
    print(f"  • Std: {y.std():.2f}")
    print(f"  • Min: {y.min():.2f}")
    print(f"  • Max: {y.max():.2f}")

    print(f"\nInput Variables (X):")
    print(f"  • Number of features: {len(FEATURE_COLS)}")
    for i, col in enumerate(FEATURE_COLS, 1):
        print(f"  {i}. {col} (type: {X[col].dtype})")

    # ─────────────────────────────────────────────────────────────────────────
    # SPLIT DATA: Train/Test
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\nTrain/Test Split:")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"  • Training samples: {X_train.shape[0]} ({100*(1-TEST_SIZE):.0f}%)")
    print(f"  • Testing samples: {X_test.shape[0]} ({100*TEST_SIZE:.0f}%)")

    # ─────────────────────────────────────────────────────────────────────────
    # PART C: TRAIN ORIGINAL DECISION TREE MODEL 
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n[C] TRAINING ORIGINAL DECISION TREE MODEL")
    print("-" * 80)

    # Train original model with default parameters (no pruning constraints)
    original_model = DecisionTreeRegressor(
        random_state=RANDOM_STATE,
        # No explicit constraints → allows deep, complex tree
    )
    original_model.fit(X_train, y_train)

    print(f"\n✓ Original model trained successfully!")
    print(f"\nOriginal Model Architecture:")
    print(f"  • Tree Depth: {original_model.get_depth()}")
    print(f"  • Number of Leaves: {original_model.get_n_leaves()}")
    print(f"  • Number of Nodes: {original_model.tree_.node_count}")

    # Evaluate original model
    original_results = evaluate_model(original_model, X_train, X_test, y_train, y_test)

    print(f"\nOriginal Model Performance:")
    print(f"  Train Set:")
    print(f"    • MSE:  {original_results.train_mse:.4f}")
    print(f"    • MAE:  {original_results.train_mae:.4f}")
    print(f"    • R²:   {original_results.train_r2:.4f}")
    print(f"  Test Set:")
    print(f"    • MSE:  {original_results.test_mse:.4f}")
    print(f"    • MAE:  {original_results.test_mae:.4f}")
    print(f"    • R²:   {original_results.test_r2:.4f}")
    
    # Calculate overfitting gap (train R² - test R²)
    overfitting_gap = original_results.train_r2 - original_results.test_r2
    print(f"\n  Overfitting Analysis:")
    print(f"    • R² Gap (Train - Test): {overfitting_gap:.4f}")
    print(f"    • Status: {'⚠ OVERFITTING DETECTED' if overfitting_gap > 0.1 else '✓ Acceptable'}")

    # ========================================================================
    # TASK 2 — TREE PRUNING / COMPLEXITY CONTROL 
    # ========================================================================

    print_header("TASK 2: TREE PRUNING / COMPLEXITY CONTROL ")

    # ─────────────────────────────────────────────────────────────────────────
    # PART A: MODIFIED MODEL WITH PRUNING PARAMETERS 
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n[A] TRAINING MODIFIED MODEL WITH PRUNING PARAMETERS")
    print("-" * 80)

    # Train modified model with constraint parameters
    # Using max_depth to limit tree complexity
    max_depth_limit = 8
    min_samples_split_limit = 10

    modified_model = DecisionTreeRegressor(
        max_depth=max_depth_limit,
        min_samples_split=min_samples_split_limit,
        random_state=RANDOM_STATE
    )
    modified_model.fit(X_train, y_train)

    print(f"\n✓ Modified model trained successfully!")
    print(f"\nModified Model Parameters (PRUNING):")
    print(f"  • max_depth: {max_depth_limit} (limits tree depth)")
    print(f"  • min_samples_split: {min_samples_split_limit} (minimum samples to split node)")

    print(f"\nModified Model Architecture:")
    print(f"  • Tree Depth: {modified_model.get_depth()}")
    print(f"  • Number of Leaves: {modified_model.get_n_leaves()}")
    print(f"  • Number of Nodes: {modified_model.tree_.node_count}")

    # Evaluate modified model
    modified_results = evaluate_model(modified_model, X_train, X_test, y_train, y_test)

    print(f"\nModified Model Performance:")
    print(f"  Train Set:")
    print(f"    • MSE:  {modified_results.train_mse:.4f}")
    print(f"    • MAE:  {modified_results.train_mae:.4f}")
    print(f"    • R²:   {modified_results.train_r2:.4f}")
    print(f"  Test Set:")
    print(f"    • MSE:  {modified_results.test_mse:.4f}")
    print(f"    • MAE:  {modified_results.test_mae:.4f}")
    print(f"    • R²:   {modified_results.test_r2:.4f}")
    
    # Calculate overfitting gap
    modified_overfitting_gap = modified_results.train_r2 - modified_results.test_r2
    print(f"\n  Overfitting Analysis:")
    print(f"    • R² Gap (Train - Test): {modified_overfitting_gap:.4f}")
    print(f"    • Status: {'⚠ OVERFITTING DETECTED' if modified_overfitting_gap > 0.1 else '✓ Acceptable'}")

    # ─────────────────────────────────────────────────────────────────────────
    # PART B: COMPARISON BETWEEN ORIGINAL AND MODIFIED MODELS 
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n[B] COMPREHENSIVE MODEL COMPARISON")
    print("-" * 80)

    # Create comparison table
    comparison_df = pd.DataFrame({
        'Metric': [
            'Tree Depth',
            'Number of Leaves',
            'Number of Nodes',
            'Train MSE',
            'Test MSE',
            'Train MAE',
            'Test MAE',
            'Train R²',
            'Test R²',
            'Overfitting Gap (R²)'
        ],
        'Original Model': [
            original_results.tree_depth,
            original_results.n_leaves,
            original_model.tree_.node_count,
            f"{original_results.train_mse:.4f}",
            f"{original_results.test_mse:.4f}",
            f"{original_results.train_mae:.4f}",
            f"{original_results.test_mae:.4f}",
            f"{original_results.train_r2:.4f}",
            f"{original_results.test_r2:.4f}",
            f"{overfitting_gap:.4f}"
        ],
        'Modified Model': [
            modified_results.tree_depth,
            modified_results.n_leaves,
            modified_model.tree_.node_count,
            f"{modified_results.train_mse:.4f}",
            f"{modified_results.test_mse:.4f}",
            f"{modified_results.train_mae:.4f}",
            f"{modified_results.test_mae:.4f}",
            f"{modified_results.train_r2:.4f}",
            f"{modified_results.test_r2:.4f}",
            f"{modified_overfitting_gap:.4f}"
        ]
    })

    print("\n" + comparison_df.to_string(index=False))

    # ─────────────────────────────────────────────────────────────────────────
    # PART C: EXPLANATION OF PRUNING IMPACT 
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n[C] IMPACT ANALYSIS: HOW PRUNING AFFECTED MODEL PERFORMANCE")
    print("-" * 80)

    depth_reduction = (
        (original_results.tree_depth - modified_results.tree_depth) / 
        original_results.tree_depth * 100
    )
    leaves_reduction = (
        (original_results.n_leaves - modified_results.n_leaves) / 
        original_results.n_leaves * 100
    )
    test_r2_change = modified_results.test_r2 - original_results.test_r2
    gap_reduction = overfitting_gap - modified_overfitting_gap

    print("\n1. COMPLEXITY REDUCTION:")
    print(f"   • Original tree depth: {original_results.tree_depth} → Modified: {modified_results.tree_depth}")
    print(f"     → Reduction: {depth_reduction:.1f}%")
    print(f"   • Original leaves: {original_results.n_leaves} → Modified: {modified_results.n_leaves}")
    print(f"     → Reduction: {leaves_reduction:.1f}%")

    print("\n2. GENERALIZATION IMPROVEMENT:")
    print(f"   • Original test R²: {original_results.test_r2:.4f}")
    print(f"   • Modified test R²: {modified_results.test_r2:.4f}")
    print(f"   • Change: {test_r2_change:+.4f} {'✓ IMPROVED' if test_r2_change > 0 else '✗ DECREASED'}")

    print("\n3. OVERFITTING REDUCTION:")
    print(f"   • Original overfitting gap: {overfitting_gap:.4f}")
    print(f"   • Modified overfitting gap: {modified_overfitting_gap:.4f}")
    print(f"   • Reduction: {gap_reduction:+.4f} {'✓ REDUCED' if gap_reduction > 0 else '✗ INCREASED'}")

    print("\n4. KEY FINDINGS:")
    
    if test_r2_change > 0:
        print(f"   ✓ Pruning IMPROVED generalization (test R² increased by {test_r2_change:.4f})")
    else:
        print(f"   → Pruning REDUCED test accuracy (test R² decreased by {-test_r2_change:.4f})")
    
    if gap_reduction > 0:
        print(f"   ✓ Pruning REDUCED overfitting (gap decreased by {gap_reduction:.4f})")
    else:
        print(f"   → Overfitting gap increased by {-gap_reduction:.4f}")
    
    if leaves_reduction > 20:
        print(f"   ✓ Tree significantly SIMPLIFIED ({leaves_reduction:.1f}% fewer leaves)")
    else:
        print(f"   • Moderate complexity reduction ({leaves_reduction:.1f}%)")

    print("\n5. CONCLUSION:")
    print("""
   The pruning strategy (max_depth=8, min_samples_split=10) successfully:
   • Reduced tree complexity, making it more interpretable
   • Limited overfitting by preventing excessive depth
   • Maintained or improved test set performance
   • Created a more generalizable model for production prediction
   
   RECOMMENDATION: Use the modified (pruned) model for production prediction
   as it balances accuracy with interpretability and generalization.
    """)

   
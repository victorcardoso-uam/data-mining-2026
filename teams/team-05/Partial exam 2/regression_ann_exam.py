"""
Midterm Test 2 — Regression and Neural Networks 
Team 05: Industrial Production Efficiency Prediction

This script demonstrates:
- TASK 1: Linear Regression Model 
  a) Dataset loading
  b) Input/target variable definition
  c) Linear Regression model training
  d) Model evaluation (R², MAE, MSE, RMSE)

- TASK 2: Polynomial Regression 
  a) Polynomial feature transformation
  b) Polynomial Regression model training
  c) Comparison with Linear Regression
  d) Explanation of appropriateness

- TASK 3: Neural Network Model 
  a) Input feature scaling
  b) ANN model training
  c) Hyperparameter modification
  d) Model evaluation with same metrics

- TASK 4: Model Comparison 
  a) Compare all three models
  b) Identify best performing model
  c) Provide detailed explanation

TARGET VARIABLE: production_efficiency_score (continuous regression task)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_PATH = os.path.join(os.path.dirname(__file__), "industrial_regression_ann_exam.csv")

# Target variable (continuous: production efficiency score)
TARGET_COL = "production_efficiency_score"

# Input variables (features for prediction)
FEATURE_COLS = [
    "assembly_time_min",
    "workers_count",
    "humidity_pct",
    "operating_hours",
    "line_temperature_c"
]

# Categorical feature to encode
CATEGORICAL_COL = "shift"

# Train-Test split parameters
TEST_SIZE = 0.20
RANDOM_STATE = 42


@dataclass
class RegressionResults:
    """Container for regression model evaluation metrics."""
    r2: float
    mae: float
    mse: float
    rmse: float


def evaluate_regression_model(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> RegressionResults:
    """
    Evaluate regression model using standard metrics.
    
    Args:
        y_true: Actual target values
        y_pred: Predicted target values
        
    Returns:
        RegressionResults: Container with all evaluation metrics
    """
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    return RegressionResults(r2=r2, mae=mae, mse=mse, rmse=rmse)


def print_header(title: str) -> None:
    """Print formatted section header."""
    print("\n" + "=" * 85)
    print(f"  {title}")
    print("=" * 85)


def print_subheader(title: str) -> None:
    """Print formatted sub-section header."""
    print("\n" + "-" * 85)
    print(f"  {title}")
    print("-" * 85)


def main() -> None:
    """Main execution of regression and ANN analysis."""
    
    print_header("MIDTERM TEST 2: REGRESSION & NEURAL NETWORKS — INDUSTRIAL EFFICIENCY")

    # ========================================================================
    # DATA LOADING AND PREPARATION
    # ========================================================================

    print_subheader("DATA LOADING AND PREPARATION")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    print(f"\n✓ Dataset loaded successfully!")
    print(f"  • Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  • Columns: {list(df.columns)}")
    print(f"\nDataset Preview:")
    print(df.head(10).to_string())
    
    # Check for missing values
    missing = df.isnull().sum().sum()
    print(f"\n✓ No missing values" if missing == 0 else f"⚠ Found {missing} missing values")

    # ─────────────────────────────────────────────────────────────────────────
    # TASK 1 — LINEAR REGRESSION MODEL 
    # ─────────────────────────────────────────────────────────────────────────

    print_header("TASK 1: LINEAR REGRESSION MODEL ")

    print_subheader("[A] LOADING DATASET ")
    print(f"\n✓ Dataset loaded with {df.shape[0]} samples and {df.shape[1]} features")
    print(f"  • Target variable (y): {TARGET_COL}")
    print(f"  • Number of input features: {len(FEATURE_COLS)}")

    print_subheader("[B] DEFINING INPUT AND TARGET VARIABLES ")
    
    # Define feature matrix and target vector
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()

    print(f"\nInput Variables (X):")
    print(f"  • Features: {FEATURE_COLS}")
    for col in FEATURE_COLS:
        print(f"    - {col}: mean={X[col].mean():.2f}, std={X[col].std():.2f}")

    print(f"\nTarget Variable (y):")
    print(f"  • Column: {TARGET_COL}")
    print(f"  • Mean: {y.mean():.2f}")
    print(f"  • Std Dev: {y.std():.2f}")
    print(f"  • Range: [{y.min():.2f}, {y.max():.2f}]")

    # Train-Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"\nTrain/Test Split:")
    print(f"  • Training samples: {X_train.shape[0]} ({100*(1-TEST_SIZE):.0f}%)")
    print(f"  • Testing samples: {X_test.shape[0]} ({100*TEST_SIZE:.0f}%)")

    print_subheader("[C] TRAINING LINEAR REGRESSION MODEL")
    
    # Train Linear Regression model
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)

    print(f"\n✓ Linear Regression model trained successfully!")
    print(f"\nModel Coefficients:")
    for feature, coef in zip(FEATURE_COLS, linear_model.coef_):
        print(f"  • {feature}: {coef:.4f}")
    print(f"  • Intercept: {linear_model.intercept_:.4f}")

    # Make predictions
    y_train_pred_linear = linear_model.predict(X_train)
    y_test_pred_linear = linear_model.predict(X_test)

    print_subheader("[D] EVALUATING LINEAR REGRESSION MODEL ")
    
    # Evaluate Linear Regression
    train_results_linear = evaluate_regression_model(y_train, y_train_pred_linear)
    test_results_linear = evaluate_regression_model(y_test, y_test_pred_linear)

    print(f"\nLinear Regression — Training Set:")
    print(f"  • R²:   {train_results_linear.r2:.4f}")
    print(f"  • MAE:  {train_results_linear.mae:.4f}")
    print(f"  • MSE:  {train_results_linear.mse:.4f}")
    print(f"  • RMSE: {train_results_linear.rmse:.4f}")

    print(f"\nLinear Regression — Test Set:")
    print(f"  • R²:   {test_results_linear.r2:.4f}")
    print(f"  • MAE:  {test_results_linear.mae:.4f}")
    print(f"  • MSE:  {test_results_linear.mse:.4f}")
    print(f"  • RMSE: {test_results_linear.rmse:.4f}")

    # ========================================================================
    # TASK 2 — POLYNOMIAL REGRESSION 
    # ========================================================================

    print_header("TASK 2: POLYNOMIAL REGRESSION ")

    print_subheader("[A] TRANSFORMING FEATURES USING POLYNOMIAL FEATURES ")
    
    # Create polynomial features (degree 2)
    poly_degree = 2
    poly_transformer = PolynomialFeatures(degree=poly_degree, include_bias=False)
    
    X_train_poly = poly_transformer.fit_transform(X_train)
    X_test_poly = poly_transformer.transform(X_test)

    print(f"\n✓ Polynomial features transformation completed!")
    print(f"  • Original number of features: {X_train.shape[1]}")
    print(f"  • Polynomial degree: {poly_degree}")
    print(f"  • Transformed number of features: {X_train_poly.shape[1]}")
    print(f"\nNew features include:")
    print(f"  • Original features")
    print(f"  • Squared terms (x²)")
    print(f"  • Interaction terms (x₁·x₂)")

    print_subheader("[B] TRAINING POLYNOMIAL REGRESSION MODEL ")
    
    # Train Polynomial Regression model
    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)

    print(f"\n✓ Polynomial Regression model trained successfully!")
    print(f"  • Model type: LinearRegression with polynomial features (degree={poly_degree})")
    print(f"  • Number of coefficients: {len(poly_model.coef_)}")
    print(f"  • Intercept: {poly_model.intercept_:.4f}")

    # Make predictions
    y_train_pred_poly = poly_model.predict(X_train_poly)
    y_test_pred_poly = poly_model.predict(X_test_poly)

    print_subheader("[C] COMPARING POLYNOMIAL AND LINEAR REGRESSION ")
    
    # Evaluate Polynomial Regression
    train_results_poly = evaluate_regression_model(y_train, y_train_pred_poly)
    test_results_poly = evaluate_regression_model(y_test, y_test_pred_poly)

    print(f"\nPolynomial Regression — Training Set:")
    print(f"  • R²:   {train_results_poly.r2:.4f}")
    print(f"  • MAE:  {train_results_poly.mae:.4f}")
    print(f"  • MSE:  {train_results_poly.mse:.4f}")
    print(f"  • RMSE: {train_results_poly.rmse:.4f}")

    print(f"\nPolynomial Regression — Test Set:")
    print(f"  • R²:   {test_results_poly.r2:.4f}")
    print(f"  • MAE:  {test_results_poly.mae:.4f}")
    print(f"  • MSE:  {test_results_poly.mse:.4f}")
    print(f"  • RMSE: {test_results_poly.rmse:.4f}")

    # Create comparison table
    comparison_linear_poly = pd.DataFrame({
        'Metric': ['R² (Test)', 'MAE (Test)', 'MSE (Test)', 'RMSE (Test)'],
        'Linear Regression': [
            f"{test_results_linear.r2:.4f}",
            f"{test_results_linear.mae:.4f}",
            f"{test_results_linear.mse:.4f}",
            f"{test_results_linear.rmse:.4f}"
        ],
        'Polynomial Regression': [
            f"{test_results_poly.r2:.4f}",
            f"{test_results_poly.mae:.4f}",
            f"{test_results_poly.mse:.4f}",
            f"{test_results_poly.rmse:.4f}"
        ]
    })

    print(f"\n" + comparison_linear_poly.to_string(index=False))
    
    # Calculate improvements
    r2_improvement = (test_results_poly.r2 - test_results_linear.r2) / abs(test_results_linear.r2) * 100
    mae_improvement = (test_results_linear.mae - test_results_poly.mae) / test_results_linear.mae * 100
    mse_improvement = (test_results_linear.mse - test_results_poly.mse) / test_results_linear.mse * 100

    print(f"\nPerformance Changes (Polynomial vs Linear):")
    print(f"  • R² change: {r2_improvement:+.2f}%")
    print(f"  • MAE improvement: {mae_improvement:+.2f}%")
    print(f"  • MSE improvement: {mse_improvement:+.2f}%")

    print_subheader("[D] WHEN POLYNOMIAL REGRESSION IS MORE APPROPRIATE ")
    
    print(f"""
Polynomial Regression is more appropriate when:

1. NON-LINEAR RELATIONSHIPS:
   - Data exhibits curved patterns rather than straight lines
   - Target variable changes at different rates across input ranges
   - Example: Efficiency initially increases with temperature, then decreases

2. HIGHER-ORDER INTERACTIONS:
   - Features interact in complex ways (e.g., assembly_time × workers_count)
   - Multiplicative or quadratic effects between variables
   - Two variables together produce non-additive effects

3. DIMINISHING RETURNS:
   - Output increases with input but at a decreasing rate
   - Common in production: more workers help, but marginal benefit decreases

4. BETTER FIT QUALITY:
   - Test R² is significantly higher than linear model (threshold: >5% improvement)
   - Residuals are more uniformly distributed
   - Predictions are more accurate across input ranges

CURRENT ANALYSIS:
- Linear model test R²: {test_results_linear.r2:.4f}
- Polynomial model test R²: {test_results_poly.r2:.4f}
- Improvement: {r2_improvement:+.2f}%

RECOMMENDATION:
""")
    
    if r2_improvement > 5:
        print(f"  ✓ POLYNOMIAL REGRESSION is RECOMMENDED (>5% R² improvement)")
    elif r2_improvement > 0:
        print(f"  → Slight improvement, but LINEAR may be preferred for interpretability")
    else:
        print(f"  → LINEAR REGRESSION is preferred (no improvement, simpler model)")

    # ========================================================================
    # TASK 3 — NEURAL NETWORK MODEL 
    # ========================================================================

    print_header("TASK 3: NEURAL NETWORK MODEL ")

    print_subheader("[A] SCALING INPUT VARIABLES ")
    
    # Scale input features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\n✓ Input scaling completed using StandardScaler!")
    print(f"  • Scaling method: Standardization (mean=0, std=1)")
    print(f"\nScaling Statistics (Training Set):")
    for i, col in enumerate(FEATURE_COLS):
        print(f"  • {col}: mean={X_train_scaled[:, i].mean():.4f}, std={X_train_scaled[:, i].std():.4f}")

    print_subheader("[B] TRAINING ANN MODEL ")
    
    # Train baseline ANN model
    baseline_ann = MLPRegressor(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=RANDOM_STATE,
        early_stopping=True,
        validation_fraction=0.1
    )
    baseline_ann.fit(X_train_scaled, y_train)

    print(f"\n✓ Baseline ANN model trained successfully!")
    print(f"\nBaseline ANN Architecture:")
    print(f"  • Hidden layer sizes: (100, 50)")
    print(f"  • Activation function: relu")
    print(f"  • Solver: adam")
    print(f"  • Max iterations: 500")
    print(f"  • Training loss: {baseline_ann.loss_:.4f}")

    # Make predictions with baseline
    y_train_pred_ann_base = baseline_ann.predict(X_train_scaled)
    y_test_pred_ann_base = baseline_ann.predict(X_test_scaled)

    # Evaluate baseline ANN
    train_results_ann_base = evaluate_regression_model(y_train, y_train_pred_ann_base)
    test_results_ann_base = evaluate_regression_model(y_test, y_test_pred_ann_base)

    print(f"\nBaseline ANN — Training Set:")
    print(f"  • R²:   {train_results_ann_base.r2:.4f}")
    print(f"  • MAE:  {train_results_ann_base.mae:.4f}")
    print(f"  • MSE:  {train_results_ann_base.mse:.4f}")
    print(f"  • RMSE: {train_results_ann_base.rmse:.4f}")

    print(f"\nBaseline ANN — Test Set:")
    print(f"  • R²:   {test_results_ann_base.r2:.4f}")
    print(f"  • MAE:  {test_results_ann_base.mae:.4f}")
    print(f"  • MSE:  {test_results_ann_base.mse:.4f}")
    print(f"  • RMSE: {test_results_ann_base.rmse:.4f}")

    print_subheader("[C] MODIFYING HYPERPARAMETERS ")
    
    # Train modified ANN with different hyperparameters
    print(f"\n✓ Training MODIFIED ANN model with adjusted hyperparameters...")
    
    modified_ann = MLPRegressor(
        hidden_layer_sizes=(150, 75, 25),  # More complex architecture
        activation='tanh',                   # Different activation
        solver='lbfgs',                      # Different solver
        max_iter=1000,                       # More iterations
        random_state=RANDOM_STATE,
        early_stopping=True,
        validation_fraction=0.1
    )
    modified_ann.fit(X_train_scaled, y_train)

    print(f"\n✓ Modified ANN model trained successfully!")
    print(f"\nModified ANN Architecture:")
    print(f"  • Hidden layer sizes: (150, 75, 25)  [Changed from (100, 50)]")
    print(f"  • Activation function: tanh  [Changed from relu]")
    print(f"  • Solver: lbfgs  [Changed from adam]")
    print(f"  • Max iterations: 1000  [Changed from 500]")
    print(f"  • Training loss: {modified_ann.loss_:.4f}")

    # Make predictions with modified
    y_train_pred_ann_mod = modified_ann.predict(X_train_scaled)
    y_test_pred_ann_mod = modified_ann.predict(X_test_scaled)

    # Evaluate modified ANN
    train_results_ann_mod = evaluate_regression_model(y_train, y_train_pred_ann_mod)
    test_results_ann_mod = evaluate_regression_model(y_test, y_test_pred_ann_mod)

    print(f"\nModified ANN — Training Set:")
    print(f"  • R²:   {train_results_ann_mod.r2:.4f}")
    print(f"  • MAE:  {train_results_ann_mod.mae:.4f}")
    print(f"  • MSE:  {train_results_ann_mod.mse:.4f}")
    print(f"  • RMSE: {train_results_ann_mod.rmse:.4f}")

    print(f"\nModified ANN — Test Set:")
    print(f"  • R²:   {test_results_ann_mod.r2:.4f}")
    print(f"  • MAE:  {test_results_ann_mod.mae:.4f}")
    print(f"  • MSE:  {test_results_ann_mod.mse:.4f}")
    print(f"  • RMSE: {test_results_ann_mod.rmse:.4f}")

    # Compare baseline vs modified ANN
    comparison_ann = pd.DataFrame({
        'Metric': ['R² (Test)', 'MAE (Test)', 'MSE (Test)', 'RMSE (Test)'],
        'Baseline ANN': [
            f"{test_results_ann_base.r2:.4f}",
            f"{test_results_ann_base.mae:.4f}",
            f"{test_results_ann_base.mse:.4f}",
            f"{test_results_ann_base.rmse:.4f}"
        ],
        'Modified ANN': [
            f"{test_results_ann_mod.r2:.4f}",
            f"{test_results_ann_mod.mae:.4f}",
            f"{test_results_ann_mod.mse:.4f}",
            f"{test_results_ann_mod.rmse:.4f}"
        ]
    })

    print(f"\n" + comparison_ann.to_string(index=False))

    print_subheader("[D] EVALUATING ANN MODEL USING SAME METRICS ")
    
    print(f"\n✓ ANN models evaluated using R², MAE, MSE, RMSE")
    print(f"  • See results above for both Baseline and Modified ANN configurations")

    # ========================================================================
    # TASK 4 — MODEL COMPARISON 
    # ========================================================================

    print_header("TASK 4: COMPREHENSIVE MODEL COMPARISON ")

    print_subheader("[A] COMPARING ALL THREE MODELS ")
    
    # Create comprehensive comparison table
    all_models_comparison = pd.DataFrame({
        'Model': [
            'Linear Regression',
            'Polynomial Regression',
            'ANN (Baseline)',
            'ANN (Modified)'
        ],
        'Train R²': [
            f"{train_results_linear.r2:.4f}",
            f"{train_results_poly.r2:.4f}",
            f"{train_results_ann_base.r2:.4f}",
            f"{train_results_ann_mod.r2:.4f}"
        ],
        'Test R²': [
            f"{test_results_linear.r2:.4f}",
            f"{test_results_poly.r2:.4f}",
            f"{test_results_ann_base.r2:.4f}",
            f"{test_results_ann_mod.r2:.4f}"
        ],
        'Test MAE': [
            f"{test_results_linear.mae:.4f}",
            f"{test_results_poly.mae:.4f}",
            f"{test_results_ann_base.mae:.4f}",
            f"{test_results_ann_mod.mae:.4f}"
        ],
        'Test MSE': [
            f"{test_results_linear.mse:.4f}",
            f"{test_results_poly.mse:.4f}",
            f"{test_results_ann_base.mse:.4f}",
            f"{test_results_ann_mod.mse:.4f}"
        ],
        'Test RMSE': [
            f"{test_results_linear.rmse:.4f}",
            f"{test_results_poly.rmse:.4f}",
            f"{test_results_ann_base.rmse:.4f}",
            f"{test_results_ann_mod.rmse:.4f}"
        ]
    })

    print(f"\n" + all_models_comparison.to_string(index=False))

    print_subheader("[B] IDENTIFYING BEST PERFORMING MODEL ")
    
    # Find best model by Test R²
    test_r2_values = [
        test_results_linear.r2,
        test_results_poly.r2,
        test_results_ann_base.r2,
        test_results_ann_mod.r2
    ]
    model_names = [
        'Linear Regression',
        'Polynomial Regression',
        'ANN (Baseline)',
        'ANN (Modified)'
    ]
    best_idx = np.argmax(test_r2_values)
    best_model_name = model_names[best_idx]
    best_r2 = test_r2_values[best_idx]

    print(f"\n✓ BEST PERFORMING MODEL: {best_model_name}")
    print(f"  • Test R² Score: {best_r2:.4f}")
    
    # Additional insights
    print(f"\nModel Rankings (by Test R²):")
    sorted_indices = np.argsort(test_r2_values)[::-1]
    for rank, idx in enumerate(sorted_indices, 1):
        print(f"  {rank}. {model_names[idx]}: R² = {test_r2_values[idx]:.4f}")

    print_subheader("[C] EXPLANATION: WHY THIS MODEL PERFORMS BEST ")
    
    print(f"""
DETAILED MODEL ANALYSIS:

1. LINEAR REGRESSION:
   - Test R²: {test_results_linear.r2:.4f}
   - Pros: Simple, interpretable, fast
   - Cons: Assumes linear relationships
   - Status: Baseline for comparison

2. POLYNOMIAL REGRESSION:
   - Test R²: {test_results_poly.r2:.4f}
   - Improvement over linear: {(test_results_poly.r2 - test_results_linear.r2):.4f}
   - Pros: Captures non-linear patterns
   - Cons: Risk of overfitting, higher complexity
   - Status: {"Recommended" if test_results_poly.r2 > test_results_linear.r2 else "Not recommended"}

3. ANN (Baseline):
   - Test R²: {test_results_ann_base.r2:.4f}
   - Architecture: 2 hidden layers (100, 50)
   - Pros: Can learn complex patterns
   - Cons: Requires scaling, more hyperparameters
   - Status: Baseline neural network

4. ANN (Modified):
   - Test R²: {test_results_ann_mod.r2:.4f}
   - Architecture: 3 hidden layers (150, 75, 25)
   - Improvements: Different solver (lbfgs), activation (tanh)
   - Pros: More complex architecture
   - Cons: Longer training time
   - Status: {"Best model" if best_idx == 3 else "Secondary choice"}

WHY {best_model_name.upper()} PERFORMS BEST:

""")

    if best_idx == 0:
        print(f"""
• OPTIMAL SIMPLICITY: The relationship between assembly time, workers, humidity,
  operating hours, temperature and efficiency is primarily LINEAR.
  
• NO OVERFITTING: Linear model generalizes well without unnecessary complexity.
  
• INTERPRETABILITY: Easy to understand which factors drive efficiency.
  
• EFFICIENCY: Fastest predictions, lowest computational overhead.

CONCLUSION: Linear relationships are sufficient for this industrial dataset.
Polynomial and neural network models would introduce complexity without
significant performance gains.
""")
    elif best_idx == 1:
        print(f"""
• CAPTURES NON-LINEARITY: The production efficiency has curved relationships with
  variables like temperature and operating hours.
  
• INTERACTION EFFECTS: Variables like (assembly_time × workers_count) are important.
  
• BETTER GENERALIZATION: Polynomial degree={poly_degree} provides right balance
  between complexity and accuracy.
  
• REASONABLE COMPLEXITY: More complex than linear but simpler than neural networks.

CONCLUSION: Polynomial regression successfully models the non-linear patterns
while maintaining interpretability better than neural networks.
""")
    else:
        print(f"""
• COMPLEX PATTERN DETECTION: The ANN captures non-linear relationships and
  multi-way interactions that simpler models miss.
  
• SUPERIOR ACCURACY: Test R² of {best_r2:.4f} significantly outperforms
  alternatives.
  
• FEATURE INTERACTIONS: Neural networks automatically learn which variable
  combinations matter most.
  
• LAYER DEPTH: {"Modified" if best_idx == 3 else "Baseline"} architecture
  ({"3" if best_idx == 3 else "2"} hidden layers) provides optimal capacity.

CONCLUSION: The complex relationships in industrial production efficiency
require the representational power of neural networks for accurate predictions.
""")

    print(f"\nFINAL RECOMMENDATION:")
    print(f"Use {best_model_name} for production efficiency predictions.")
    print(f"This model achieves {best_r2:.4f} R² on unseen test data.")

    print_header(" MIDTERM TEST ")


if __name__ == "__main__":
    main()
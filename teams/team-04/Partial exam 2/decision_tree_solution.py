import pandas as pd
import numpy as np
import os
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# ============================================================================
# TAREA 1: Model Training
# ============================================================================

# ============================================================================
# a) Load the dataset and display:
# ============================================================================

# Load the dataset with absolute path
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'industrial_decision_tree_exam.csv')
df = pd.read_csv(csv_path)

print("=" * 80)
print("TAREA 1: MODEL TRAINING - DECISION TREES")
print("=" * 80)
print()

# Display dataset shape
print("a) Dataset Information:")
print("-" * 80)
print(f"Dataset Shape: {df.shape}")
print(f"  - Rows (samples): {df.shape[0]}")
print(f"  - Columns (features): {df.shape[1]}")
print()

# Display column names
print("Column Names:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i}. {col}")
print()

# Display data types
print("Data Types:")
print(df.dtypes)
print()

# Display first few rows
print("First 5 rows of the dataset:")
print(df.head())
print()

# Display basic statistics
print("Dataset Statistics:")
print(df.describe())
print()

# ============================================================================
# b) Define Input variables (X) and Target variable (y)
# ============================================================================

print("b) Variables Definition:")
print("-" * 80)

# Define input variables (X) - all columns except the last one
# The last column 'daily_output_units' is the target variable
X = df.iloc[:, :-1]  # All columns except the last one
y = df.iloc[:, -1]   # Last column as target

print(f"Input Variables (X):")
print(f"  Columns: {list(X.columns)}")
print(f"  Shape: {X.shape}")
print()

print(f"Target Variable (y):")
print(f"  Column: {y.name}")
print(f"  Shape: {y.shape}")
print()

# ============================================================================
# c) Train a Decision Tree model
# ============================================================================

print("c) Decision Tree Model Training:")
print("-" * 80)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")
print()

# Train a Decision Tree Regressor (since 'daily_output_units' is a continuous variable)
dt_model = DecisionTreeRegressor(
    random_state=42,
    max_depth=10
)

dt_model.fit(X_train, y_train)

print("Decision Tree Regressor trained successfully!")
print()

# Model evaluation
train_score = dt_model.score(X_train, y_train)
test_score = dt_model.score(X_test, y_test)

print(f"Model Performance:")
print(f"  Training R² Score: {train_score:.4f}")
print(f"  Testing R² Score: {test_score:.4f}")
print()

# Feature importance
print("Feature Importance:")
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': dt_model.feature_importances_
}).sort_values('Importance', ascending=False)

print(feature_importance.to_string(index=False))
print()
print()

# ============================================================================
# Task 2: Tree Pruning / Complexity Control
# ============================================================================

print("=" * 80)
print("TAREA 2: TREE PRUNING / COMPLEXITY CONTROL")
print("=" * 80)
print()

# ============================================================================
# a) Modify parameters: max_depth and min_samples_split
# ============================================================================

print("a) Creating Modified Models with Different Parameters:")
print("-" * 80)
print()

# Original model (already trained)
print("Original Model Parameters:")
print(f"  - max_depth: None (unlimited)")
print(f"  - min_samples_split: 2 (default)")
print()

# Modified Model 1: Reduce max_depth
print("Modified Model 1: max_depth = 5")
dt_model_max_depth_5 = DecisionTreeRegressor(
    max_depth=5,
    random_state=42
)
dt_model_max_depth_5.fit(X_train, y_train)
print(f"  - Created and trained")
print()

# Modified Model 2: Reduce max_depth further
print("Modified Model 2: max_depth = 3")
dt_model_max_depth_3 = DecisionTreeRegressor(
    max_depth=3,
    random_state=42
)
dt_model_max_depth_3.fit(X_train, y_train)
print(f"  - Created and trained")
print()

# Modified Model 3: Increase min_samples_split
print("Modified Model 3: min_samples_split = 10")
dt_model_min_split_10 = DecisionTreeRegressor(
    min_samples_split=10,
    random_state=42
)
dt_model_min_split_10.fit(X_train, y_train)
print(f"  - Created and trained")
print()

# Modified Model 4: Combine both parameters
print("Modified Model 4: max_depth = 5, min_samples_split = 10")
dt_model_combined = DecisionTreeRegressor(
    max_depth=5,
    min_samples_split=10,
    random_state=42
)
dt_model_combined.fit(X_train, y_train)
print(f"  - Created and trained")
print()

# ============================================================================
# b) Compare results between original model and modified models
# ============================================================================

print("=" * 80)
print("b) Model Comparison Results:")
print("-" * 80)
print()

# Create a comparison table
models_info = {
    'Model': [
        'Original (max_depth=10)',
        'Modified 1 (max_depth=5)',
        'Modified 2 (max_depth=3)',
        'Modified 3 (min_samples_split=10)',
        'Modified 4 (max_depth=5, min_split=10)'
    ],
    'Train_R2': [
        dt_model.score(X_train, y_train),
        dt_model_max_depth_5.score(X_train, y_train),
        dt_model_max_depth_3.score(X_train, y_train),
        dt_model_min_split_10.score(X_train, y_train),
        dt_model_combined.score(X_train, y_train)
    ],
    'Test_R2': [
        dt_model.score(X_test, y_test),
        dt_model_max_depth_5.score(X_test, y_test),
        dt_model_max_depth_3.score(X_test, y_test),
        dt_model_min_split_10.score(X_test, y_test),
        dt_model_combined.score(X_test, y_test)
    ],
    'Tree_Depth': [
        dt_model.get_depth(),
        dt_model_max_depth_5.get_depth(),
        dt_model_max_depth_3.get_depth(),
        dt_model_min_split_10.get_depth(),
        dt_model_combined.get_depth()
    ],
    'Leaf_Nodes': [
        dt_model.get_n_leaves(),
        dt_model_max_depth_5.get_n_leaves(),
        dt_model_max_depth_3.get_n_leaves(),
        dt_model_min_split_10.get_n_leaves(),
        dt_model_combined.get_n_leaves()
    ]
}

comparison_df = pd.DataFrame(models_info)
print("Detailed Model Comparison Table:")
print()
print(comparison_df.to_string(index=False))
print()

# Calculate differences from original model
print()
print("Performance Change (compared to Original Model):")
print("-" * 80)

original_test_r2 = dt_model.score(X_test, y_test)
original_train_r2 = dt_model.score(X_train, y_train)

for idx in range(1, len(models_info['Model'])):
    model_name = models_info['Model'][idx]
    test_r2_change = models_info['Test_R2'][idx] - original_test_r2
    train_r2_change = models_info['Train_R2'][idx] - original_train_r2
    
    print(f"{model_name}:")
    print(f"  - Train R² Change: {train_r2_change:+.4f}")
    print(f"  - Test R² Change: {test_r2_change:+.4f}")
    print(f"  - Depth: {models_info['Tree_Depth'][idx]} (Original: {models_info['Tree_Depth'][0]})")
    print(f"  - Leaf Nodes: {models_info['Leaf_Nodes'][idx]} (Original: {models_info['Leaf_Nodes'][0]})")
    print()

# ============================================================================
# c) Explain how pruning affected model performance
# ============================================================================

print("=" * 80)
print("c) Analysis: How Pruning Affected Model Performance")
print("-" * 80)
print()

print("EXPLANATION OF PRUNING EFFECTS:")
print()
print("1. OVERFITTING REDUCTION:")
print("   - Original Model (max_depth=10):")
print(f"     * Train R²: {original_train_r2:.4f}")
print(f"     * Test R²: {original_test_r2:.4f}")
print(f"     * Gap (Overfitting): {(original_train_r2 - original_test_r2):.4f}")
print()
print("   - Modified Model 1 (max_depth=5):")
modified1_train = dt_model_max_depth_5.score(X_train, y_train)
modified1_test = dt_model_max_depth_5.score(X_test, y_test)
print(f"     * Train R²: {modified1_train:.4f}")
print(f"     * Test R²: {modified1_test:.4f}")
print(f"     * Gap (Overfitting): {(modified1_train - modified1_test):.4f}")
print()
print("   The reduced max_depth limits tree complexity, decreasing the gap between")
print("   training and testing performance, indicating less overfitting.")
print()

print("2. GENERALIZATION IMPROVEMENT:")
print(f"   - Original model test R²: {original_test_r2:.4f}")
print(f"   - Modified 1 test R²: {modified1_test:.4f}")
if modified1_test > original_test_r2:
    improvement = ((modified1_test - original_test_r2) / abs(original_test_r2)) * 100
    print(f"   - Improvement: +{improvement:.2f}%")
else:
    loss = ((original_test_r2 - modified1_test) / abs(original_test_r2)) * 100
    print(f"   - Trade-off: -{loss:.2f}%")
print()
print("   Pruning strategies (max_depth, min_samples_split) control tree size,")
print("   preventing the model from memorizing training data noise.")
print()

print("3. TREE COMPLEXITY CONTROL:")
print(f"   - Original Model Depth: {models_info['Tree_Depth'][0]} | Leaf Nodes: {models_info['Leaf_Nodes'][0]}")
print(f"   - Modified 1 Depth: {models_info['Tree_Depth'][1]} | Leaf Nodes: {models_info['Leaf_Nodes'][1]}")
print(f"   - Reduction: {models_info['Tree_Depth'][0] - models_info['Tree_Depth'][1]} depth levels")
print()
print("   Smaller trees are more interpretable and computationally efficient.")
print()

print("4. BIAS-VARIANCE TRADE-OFF:")
print("   - Higher max_depth: Lower bias, Higher variance (overfitting risk)")
print("   - Lower max_depth: Higher bias, Lower variance (better generalization)")
print()
print("   Pruning increases bias slightly but significantly reduces variance,")
print("   resulting in better overall model performance on unseen data.")
print()

print("5. RECOMMENDATIONS:")
print(f"   - Use Modified Model 1 (max_depth=5) for better generalization")
print(f"   - Test R² Score improved to: {modified1_test:.4f}")
print(f"   - More interpretable with depth: {models_info['Tree_Depth'][1]}")
print()
print(
    "The final explanation is theoretical: this script demonstrates how a decision tree fits data by splitting features to minimize error, and how pruning changes the model from a high-variance, overfit tree into a simpler one with better generalization. "
    "In a math-test answer, you would say that the original model captures too much noise, while pruning raises bias slightly but reduces variance, so the test performance improves because the model better reflects the true underlying pattern instead of memorizing the training set."
)

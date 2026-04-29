
# Decision Tree Regression Demo
# Session 14 – Decision Tree Analysis

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, r2_score
import os

# Load dataset - find it relative to this script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "industrial_decision_tree_exam.csv")
df = pd.read_csv(csv_path)

print("\n=== Dataset Preview ===")
print(df.head())

# Define features and target
X = df[[
    "machine_speed_units_min",
    "operator_experience_years",
    "line_temperature_c",
    "equipment_age_years",
    "inspection_time_min"
]]
y = df["daily_output_units"]

## Train-test split
# Why split?
# Train model on one subset
# Evaluate on unseen data:
#    30% test data
#   70% training data
# Without this, we cannot measure generalization.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

## Train Decision Tree Regressor - ORIGINAL MODEL
# What happens inside .fit()?
# The algorithm:
#    * Tries every feature
#    * Tries multiple thresholds
#    * Calculates impurity reduction (MSE reduction in regression)
#    * Selects best split
#    * Repeats recursively
#    * Stops at max_depth = 3
model = DecisionTreeRegressor(max_depth=3, random_state=42)
model.fit(X_train, y_train)

## Train Decision Tree Regressor - MODIFIED MODEL
# Modified parameters:
# - max_depth = 6 (increased from 3, allows deeper tree)
# - min_samples_split = 5 (requires at least 5 samples to split a node)
model_modified = DecisionTreeRegressor(max_depth=6, min_samples_split=5, random_state=42)
model_modified.fit(X_train, y_train)

## Why max_depth Matters?

# max_depth = 3 means:

#   * The tree can only grow 3 levels deep
#   * Prevents overfitting
#   * Produces simpler rules

# If we increase max_depth:

#   * Tree becomes more complex
#   * Training error decreases
#   * Risk of overfitting increases

# ----------------------------------
## Predictions - ORIGINAL MODEL
# The model now:
#   * Takes unseen inputs
#   * Applies learned decision rules
#   * Outputs predicted energy values
y_pred = model.predict(X_test)

## Predictions - MODIFIED MODEL
y_pred_modified = model_modified.predict(X_test)

#-------------------------------
## Evaluation - ORIGINAL MODEL
# Mean Squared Error (MSE): 
# Average squared difference between true and predicted values
# Lower is better

# R² Score:
# Proportion of variance explained
# Closer to 1 = better model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n=== MODEL EVALUATION - ORIGINAL (max_depth=3) ===")
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

## Evaluation - MODIFIED MODEL
mse_modified = mean_squared_error(y_test, y_pred_modified)
r2_modified = r2_score(y_test, y_pred_modified)

print("\n=== MODEL EVALUATION - MODIFIED (max_depth=6, min_samples_split=5) ===")
print("Mean Squared Error:", mse_modified)
print("R2 Score:", r2_modified)

## Comparison
print("\n" + "="*60)
print("COMPARISON: Original vs Modified Model")
print("="*60)
print(f"MSE Improvement: {mse - mse_modified:.4f} (Lower is better)")
print(f"R² Improvement: {r2_modified - r2:.4f} (Higher is better)")

if r2_modified > r2:
    print(f"✓ Modified model is BETTER (R² improved by {(r2_modified - r2)*100:.2f}%)")
elif r2_modified < r2:
    print(f"✗ Modified model is WORSE (R² decreased by {(r2 - r2_modified)*100:.2f}%)")
else:
    print("Both models perform equally")

print("="*60)

## Feature importance - ORIGINAL MODEL
# This tells us:
# Which variables contributed most to impurity reduction.

# Interpretation:
# Higher value → greater influence on predictions.
print("\n=== Feature Importances - ORIGINAL Model ===")
for feature, importance in zip(X.columns, model.feature_importances_):
    print(f"{feature}: {importance:.4f}")

## Feature importance - MODIFIED MODEL
print("\n=== Feature Importances - MODIFIED Model ===")
for feature, importance in zip(X.columns, model_modified.feature_importances_):
    print(f"{feature}: {importance:.4f}")


# Plot trees (side by side comparison)
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Original Model Tree
plot_tree(model, feature_names=X.columns, filled=True, ax=axes[0])
axes[0].set_title("Original Model: Decision Tree Regression (max_depth=3)", fontsize=12, fontweight='bold')

# Modified Model Tree
plot_tree(model_modified, feature_names=X.columns, filled=True, ax=axes[1])
axes[1].set_title("Modified Model: Decision Tree Regression (max_depth=6, min_samples_split=5)", fontsize=12, fontweight='bold')


plt.tight_layout()
plt.show()

# c) Explain how pruning affected model performance  
print("\n=== EXPLANATION: Effect of Pruning / Complexity Control ===")
print("""
In this experiment, pruning was applied to control the complexity of the decision tree by 
limiting its maximum depth and setting a minimum number of samples required to split a
node. The simpler model with max_depth=3 achieved lower MSE and higher R² compared to the deeper 
model (max_depth=6, min_samples_split=5), showing that the pruned tree generalized better to unseen 
data. By contrast, the deeper tree overfit the training set, capturing noise rather than meaningful patterns, }
which led to weaker performance on the test data. Overall, pruning proved essential in preventing overfitting and improving the model’s ability to generalize effectively.
""")

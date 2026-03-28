
# Decision Tree Regression Demo
# Session 14 – Decision Tree Analysis

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("solar_farm_production.csv")

print("\n=== Dataset Preview ===")
print(df.head())

# Define features and target
X = df[[
    "solar_irradiance_wm2",
    "temperature_c",
    "humidity_pct",
    "wind_speed_ms"
]]
y = df["energy_output_kw"]

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

## Train Decision Tree Regressor
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
## Predictions
# The model now:

#   * Takes unseen inputs
#   * Applies learned decision rules
#   * Outputs predicted energy values
y_pred = model.predict(X_test)

#-------------------------------
## Evaluation
# Mean Squared Error (MSE): 
# Average squared difference between true and predicted values
# Lower is better

# R² Score:
# Proportion of variance explained
# Closer to 1 = better model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n=== Model Evaluation ===")
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

## Feature importance
# This tells us:
# Which variables contributed most to impurity reduction.

# Interpretation:
# Higher value → greater influence on predictions.
print("\n=== Feature Importances ===")
for feature, importance in zip(X.columns, model.feature_importances_):
    print(f"{feature}: {importance:.4f}")


# Plot tree (single plot)
plt.figure(figsize=(12, 6))
plot_tree(model, feature_names=X.columns, filled=True)
plt.title("Decision Tree Regression (max_depth=3)")
plt.savefig("decision_tree_visualization.png", dpi=100, bbox_inches='tight')
print("\nDecision tree visualization saved as 'decision_tree_visualization.png'")

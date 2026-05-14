# Final Project - Data Mining
# Universidad Anáhuac Mayab
# Profesor: Víctor Cardoso Fernández
# Alumno: Ale

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder

# ============================
# 1. Load Dataset
# ============================
DATA_PATH = r"C:\Users\ale03\OneDrive\Escritorio\MAYAB\SEMESTRE 8\MINERIA DE DATOS\data-mining-course\data-mining-2026\teams\team-01\FINAL PROJECT\mat.csv"
df = pd.read_csv(DATA_PATH)

print("\n=== DATASET PREVIEW ===")
print(df.head())

# Encode categorical variables (compatible with pandas 2 and 3)
for col in df.select_dtypes(include=["object", "string"]).columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# Define input (X) and target (y)
X = df.drop("G3", axis=1)   # Predict final grade
y = df["G3"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Helper function for metrics
def evaluate_model(name, y_true, y_pred):
    print(f"\n{name} Performance:")
    print("R2:", r2_score(y_true, y_pred))
    print("MAE:", mean_absolute_error(y_true, y_pred))
    print("MSE:", mean_squared_error(y_true, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))

    # Plot Predicted vs Actual
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{name} - Predicted vs Actual")
    plt.show()

    # Residual Plot
    residuals = y_true - y_pred
    plt.scatter(y_pred, residuals, alpha=0.7)
    plt.axhline(y=0, color="red", linestyle="--")
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title(f"{name} - Residuals")
    plt.show()


# ============================
# 2. Decision Tree
# ============================
# Default model
tree_default = DecisionTreeRegressor(random_state=42)
tree_default.fit(X_train, y_train)
y_pred_tree_default = tree_default.predict(X_test)
evaluate_model("Decision Tree (Default)", y_test, y_pred_tree_default)
print("\n[Interpretation] The default Decision Tree is fully grown and may overfit the training data. Observe the metrics and plots to check for overfitting (high train, low test performance).\n")

# Controlled complexity
tree_pruned = DecisionTreeRegressor(max_depth=5, random_state=42)
tree_pruned.fit(X_train, y_train)
y_pred_tree_pruned = tree_pruned.predict(X_test)
evaluate_model("Decision Tree (Pruned)", y_test, y_pred_tree_pruned)

print(f"""
Compare both versions and briefly explain:
How the parameter affected the model:
  - Setting max_depth=5 pruned the tree, reducing its complexity and risk of overfitting.
  - The pruned tree (R² = {r2_score(y_test, y_pred_tree_pruned):.3f}) slightly outperformed the default (R² = {r2_score(y_test, y_pred_tree_default):.3f}), and had lower MAE and RMSE.
Whether performance improved or not:
  - Yes, pruning improved generalization, as seen by the slightly better metrics on the test set.
""")
# ============================
# 3. Regression Models
# ============================
# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lin = lin_reg.predict(X_test)
evaluate_model("Linear Regression", y_test, y_pred_lin)

# Polynomial Regression (degree=2)
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_train)
y_pred_poly = poly_reg.predict(X_test_poly)
evaluate_model("Polynomial Regression (deg=2)", y_test, y_pred_poly)

print(f"""
Polynomial Regression vs Linear Regression:
- Does the polynomial model improve performance?
  - No. The polynomial model (R² = {r2_score(y_test, y_pred_poly):.3f}) performed much worse than the linear model (R² = {r2_score(y_test, y_pred_lin):.3f}).
- Why this improvement happens (or not):
  - The relationship between features and the target is mostly linear. Adding polynomial terms increased model complexity without capturing meaningful patterns, leading to overfitting and poor generalization.
""")

# ============================
# 4. Artificial Neural Network
# ============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================
# ADDITIONAL EXPERIMENTS: ANN CONFIGURATIONS
# ============================
print("\n============================")
print("ADDITIONAL EXPERIMENTS: ANN CONFIGURATIONS")
print("============================\n")

# Define a list of at least three different ANN configurations
ann_experiments = [
    {"name": "ANN Config 1", "hidden_layer_sizes": (50,), "activation": "relu", "solver": "adam", "max_iter": 500},
    {"name": "ANN Config 2", "hidden_layer_sizes": (100, 50), "activation": "tanh", "solver": "adam", "max_iter": 1000},
    {"name": "ANN Config 3", "hidden_layer_sizes": (30, 30, 10), "activation": "relu", "solver": "lbfgs", "max_iter": 500},
    {"name": "ANN Config 4", "hidden_layer_sizes": (20,), "activation": "logistic", "solver": "adam", "max_iter": 700},
]

ann_results = []

for config in ann_experiments:
    ann = MLPRegressor(
        hidden_layer_sizes=config["hidden_layer_sizes"],
        activation=config["activation"],
        solver=config["solver"],
        max_iter=config["max_iter"],
        random_state=42
    )
    ann.fit(X_train_scaled, y_train)
    y_pred = ann.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    ann_results.append({
        "Name": config["name"],
        "Hidden Layers": str(config["hidden_layer_sizes"]),
        "Activation": config["activation"],
        "Solver": config["solver"],
        "Max Iter": config["max_iter"],
        "R2": r2,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse
    })

# Print results as a table
ann_results_df = pd.DataFrame(ann_results)
print("\n=== ANN CONFIGURATIONS COMPARISON TABLE ===")
print(ann_results_df.round(4).sort_values(by="R2", ascending=False).reset_index(drop=True))

best_ann = ann_results_df.sort_values(by="R2", ascending=False).iloc[0]
print(f"""
How changing parameters affects the results:
- Increasing the number of layers/neurons or changing activation/solver can impact performance and convergence.
- In this case, {best_ann['Name']} performed best (R² = {best_ann['R2']:.3f}), but still did not outperform Linear Regression or Decision Trees.
Which configuration performs better and why:
- {best_ann['Name']} likely balanced complexity and convergence best for this dataset, but overall, simpler models worked better.
""")
# ============================
# Final Comparison
# ============================

# ============================
# Final Comparison & Summary
# ============================
print("\n=== FINAL MODEL COMPARISON TABLE ===")
final_results = [
    {"Model": "Decision Tree (Default)", "R2": r2_score(y_test, y_pred_tree_default), "MAE": mean_absolute_error(y_test, y_pred_tree_default), "MSE": mean_squared_error(y_test, y_pred_tree_default), "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_tree_default))},
    {"Model": "Decision Tree (Pruned)", "R2": r2_score(y_test, y_pred_tree_pruned), "MAE": mean_absolute_error(y_test, y_pred_tree_pruned), "MSE": mean_squared_error(y_test, y_pred_tree_pruned), "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_tree_pruned))},
    {"Model": "Linear Regression", "R2": r2_score(y_test, y_pred_lin), "MAE": mean_absolute_error(y_test, y_pred_lin), "MSE": mean_squared_error(y_test, y_pred_lin), "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_lin))},
    {"Model": "Polynomial Regression (deg=2)", "R2": r2_score(y_test, y_pred_poly), "MAE": mean_absolute_error(y_test, y_pred_poly), "MSE": mean_squared_error(y_test, y_pred_poly), "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_poly))},
    {"Model": f"Best ANN ({best_ann['Name']})", "R2": best_ann['R2'], "MAE": best_ann['MAE'], "MSE": best_ann['MSE'], "RMSE": best_ann['RMSE']},
]
final_results_df = pd.DataFrame(final_results)
print(final_results_df.round(4).sort_values(by="R2", ascending=False).reset_index(drop=True))

print("""
---
FINAL COMPARISON & SUMMARY (Analysis)

Model Performance Overview:
- Best Model: The Linear Regression model achieved the highest R² (0.75) and the lowest RMSE (2.24) among all models tested, indicating it explains about 75% of the variance in the target (final grade) and has the lowest average prediction error.
- Decision Trees: Both the default and pruned Decision Tree models performed well (R² ≈ 0.72–0.73), with the pruned tree slightly outperforming the default, suggesting that controlling complexity helps generalization.
- Polynomial Regression: This model performed poorly (R² ≈ -0.01), indicating that a quadratic relationship does not fit the data and may lead to overfitting or instability.
- ANNs: The best ANN configuration (Config 4: (20,) neurons, logistic activation) achieved R² ≈ 0.69, which is lower than Linear Regression and Decision Trees. Other ANN configurations performed slightly worse, and some did not converge fully (as indicated by warnings).

Interpretation:
- Why Linear Regression is Best: The linear model captures the main trend in the data without overfitting, as shown by its high R² and low error metrics. The residuals are likely randomly scattered, indicating a good fit.
- Complexity vs. Performance: Increasing model complexity (more layers/neurons in ANN, higher polynomial degree) did not improve results and sometimes made them worse. This suggests the relationship between features and the target is mostly linear, and simpler models generalize better for this dataset.
- Overfitting/Underfitting: The pruned Decision Tree and the best ANN both avoid overfitting, but do not outperform the linear model. Polynomial regression overfits or fails to generalize.

Conclusion:
For this dataset, Linear Regression is the most reliable and interpretable model, balancing accuracy and simplicity. More complex models (deep ANNs, polynomial regression) do not provide additional benefit and may introduce unnecessary complexity or instability.
---
""")

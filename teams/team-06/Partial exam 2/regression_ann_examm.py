import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import matplotlib.pyplot as plt
# =============================
# Task 1 — Linear Regression Model
# =============================
# a) Load the dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "industrial_regression_ann_exam.csv")
data = pd.read_csv(DATA_PATH)
print("\n=== DATASET PREVIEW ===")
print(data.head())

# b) Define input and target variables
# Use all columns except the target and categorical for X, and 'production_efficiency_score' as y
X = data.drop(columns=["production_efficiency_score", "shift"])
y = data["production_efficiency_score"]

# c) Encode categorical variables (shift)
X = pd.concat([X, pd.get_dummies(data["shift"], drop_first=True)], axis=1)

# d) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# e) Train Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_predictions = linear_model.predict(X_test)

# f) Evaluate Linear Regression
linear_mae = mean_absolute_error(y_test, linear_predictions)
linear_mse = mean_squared_error(y_test, linear_predictions)
linear_rmse = np.sqrt(linear_mse)
linear_r2 = r2_score(y_test, linear_predictions)

print("\n=== LINEAR REGRESSION METRICS ===")
print("MAE :", round(linear_mae, 4))
print("MSE :", round(linear_mse, 4))
print("RMSE:", round(linear_rmse, 4))
print("R2  :", round(linear_r2, 4))

# =============================
# Task 2 — Polynomial Regression Model
# =============================
# a) Transform input variables using polynomial features (degree=2)
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# b) Train Polynomial Regression model
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
poly_predictions = poly_model.predict(X_test_poly)

# c) Compare results with linear regression
print("\n=== TASK 2c: COMPARISON BETWEEN LINEAR AND POLYNOMIAL REGRESSION ===")
comparison_2 = pd.DataFrame({
    "Model": ["Linear", "Polynomial (deg=2)"],
    "R2": [linear_r2, poly_r2],
    "MAE": [linear_mae, poly_mae],
    "MSE": [linear_mse, poly_mse],
    "RMSE": [linear_rmse, poly_rmse]
})
print(comparison_2.round(4))
if poly_r2 > linear_r2:
    print(f"\nPolynomial Regression performs better with R2 = {poly_r2:.4f} vs Linear R2 = {linear_r2:.4f}")
else:
    print(f"\nLinear Regression performs better with R2 = {linear_r2:.4f} vs Polynomial R2 = {poly_r2:.4f}")

# d) Evaluate Polynomial Regression
poly_mae = mean_absolute_error(y_test, poly_predictions)
poly_mse = mean_squared_error(y_test, poly_predictions)
poly_rmse = np.sqrt(poly_mse)
poly_r2 = r2_score(y_test, poly_predictions)

print("\n=== POLYNOMIAL REGRESSION METRICS ===")
print("MAE :", round(poly_mae, 4))
print("MSE :", round(poly_mse, 4))
print("RMSE:", round(poly_rmse, 4))
print("R2  :", round(poly_r2, 4))

# e) Plot data, linear and polynomial regression (using the first numeric feature)
main_feature = X.columns[0]
x_grid = np.linspace(X[main_feature].min(), X[main_feature].max(), 300).reshape(-1, 1)
if X.shape[1] > 1:
    X_mean = X.mean()
    X_grid_full = np.tile(X_mean.values, (x_grid.shape[0], 1))
    X_grid_full[:, 0] = x_grid.flatten()
else:
    X_grid_full = x_grid

y_grid_linear = linear_model.predict(X_grid_full)
x_grid_poly = poly.transform(X_grid_full)
y_grid_poly = poly_model.predict(x_grid_poly)

plt.figure(figsize=(9, 6))
plt.scatter(X[main_feature], y, label="Observed data")
plt.plot(x_grid, y_grid_linear, label="Linear Regression", linewidth=2, color='red')
plt.plot(x_grid, y_grid_poly, label="Polynomial Regression (degree=2)", linewidth=2, color='green')
plt.xlabel(main_feature)
plt.ylabel("production_efficiency_score")
plt.title("Data with Regression Models")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# =============================
# Task 3 — Neural Network Model (ANN)
# =============================
# a) Scale the input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def train_and_evaluate_ann(hidden_layer_sizes, activation, solver, max_iter):
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        max_iter=max_iter,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    return {
        "hidden_layer_sizes": str(hidden_layer_sizes),
        "activation": activation,
        "solver": solver,
        "max_iter": max_iter,
        "R2": r2,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse
    }

print("\n=== BASELINE ANN MODEL ===")
baseline = train_and_evaluate_ann(
    hidden_layer_sizes=(8,),
    activation="relu",
    solver="adam",
    max_iter=277
)
print(baseline)

print("\n=== ADDITIONAL ANN CONFIGURATIONS ===")
experiments = [
    {"hidden_layer_sizes": (11,), "activation": "relu", "solver": "adam", "max_iter": 347},
    {"hidden_layer_sizes": (4,), "activation": "tanh", "solver": "lbfgs", "max_iter": 200},
    {"hidden_layer_sizes": (3,6), "activation": "logistic", "solver": "sgd", "max_iter": 365},
]
results = [baseline]
for exp in experiments:
    result = train_and_evaluate_ann(
        hidden_layer_sizes=exp["hidden_layer_sizes"],
        activation=exp["activation"],
        solver=exp["solver"],
        max_iter=exp["max_iter"]
    )
    results.append(result)

results_df = pd.DataFrame(results)
print("\n=== ANN COMPARISON TABLE ===")
print(results_df.round(4))
sorted_df = results_df.sort_values(by="R2", ascending=False)
print("\n=== SORTED ANN RESULTS (BEST R2 FIRST) ===")
print(sorted_df.round(4))

# =============================
# Task 4 — Model Comparison
# =============================

# a) Compare all three model types
print("\n=== TASK 4a: COMPARISON OF ALL THREE MODELS ===")

# Get the best ANN model from the sorted results
best_ann = sorted_df.iloc[0]

# Create comparison dataframe
comparison_all = pd.DataFrame({
    "Model": ["Linear Regression", "Polynomial Regression (deg=2)", "Best ANN Configuration"],
    "R2": [linear_r2, poly_r2, best_ann["R2"]],
    "MAE": [linear_mae, poly_mae, best_ann["MAE"]],
    "MSE": [linear_mse, poly_mse, best_ann["MSE"]],
    "RMSE": [linear_rmse, poly_rmse, best_ann["RMSE"]]
})

print(comparison_all.round(4))

# b) Identify which model performs best
print("\n=== TASK 4b: BEST PERFORMING MODEL ===")

models = {
    "Linear Regression": linear_r2,
    "Polynomial Regression (deg=2)": poly_r2,
    "Best ANN Configuration": best_ann["R2"]
}

best_model = max(models, key=models.get)
best_r2 = models[best_model]

print(f"Best Model: {best_model}")
print(f"R2 Score: {best_r2:.4f}")

# c) Explanation of why this model performs best
print("\n=== TASK 4c: EXPLANATION ===")
print(f"\nThe {best_model} model performs best because:")
print(f"1. It achieves the highest R2 score of {best_r2:.4f}")
print(f"2. This means it explains {best_r2*100:.2f}% of the variance in production_efficiency_score")

if best_model == "Polynomial Regression (deg=2)":
    print(f"3. Polynomial features capture non-linear relationships between predictors and target")
    print(f"4. The degree-2 polynomial is complex enough to fit the data patterns without overfitting")
elif best_model == "Best ANN Configuration":
    best_config = best_ann["hidden_layer_sizes"]
    print(f"3. The ANN with {best_config} hidden layers and {best_ann['activation']} activation")
    print(f"4. Neural networks can learn complex non-linear patterns through their hidden layers")
    print(f"5. The scaled input features and optimization algorithm ({best_ann['solver']}) enable better convergence")
else:
    print(f"3. Linear relationships between the predictors and target are sufficient")
    print(f"4. Simple models often generalize better and avoid overfitting")

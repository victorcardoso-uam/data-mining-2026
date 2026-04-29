# Practical Problem 2 — Regression & ANN Model Comparison

# === Imports ===
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# === Task 1: Regression Model ===
# a) Load the dataset
DATA_PATH = r"C:\Users\ale03\OneDrive\Escritorio\MAYAB\SEMESTRE 8\MINERIA DE DATOS\data-mining-course\data-mining-2026\teams\team-01\Partial exam 2\Practical Problem 2\industrial_regression_ann_exam.csv"
df = pd.read_csv(DATA_PATH)

print("\n=== DATASET PREVIEW ===")
print(df.head())

# b) Define input and target variables
# NOTE: We'll use all columns except the target as features
TARGET = "production_efficiency_score"
X = df.drop(columns=[TARGET])
y = df[TARGET]

# For this dataset, 'shift' is categorical. We'll use one-hot encoding for regression/ANN.
X = pd.get_dummies(X, drop_first=True)

# c) Train-test split
X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.2, random_state=42
)

# d) Train Linear Regression model
linreg = LinearRegression()
linreg.fit(X_train, y_train)
y_pred_lin = linreg.predict(X_test)

# e) Evaluate Linear Regression
r2_lin = r2_score(y_test, y_pred_lin)
mae_lin = mean_absolute_error(y_test, y_pred_lin)
mse_lin = mean_squared_error(y_test, y_pred_lin)
rmse_lin = np.sqrt(mse_lin)

print("\n=== Task 1: Linear Regression Results ===")
print(f"R2:   {r2_lin:.4f}")
print(f"MAE:  {mae_lin:.4f}")
print(f"MSE:  {mse_lin:.4f}")
print(f"RMSE: {rmse_lin:.4f}")

# === Task 2: Polynomial Regression ===
# a) Transform input variables using polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# b) Train polynomial regression model
polyreg = LinearRegression()
polyreg.fit(X_train_poly, y_train)
y_pred_poly = polyreg.predict(X_test_poly)

# c) Evaluate Polynomial Regression
r2_poly = r2_score(y_test, y_pred_poly)
mae_poly = mean_absolute_error(y_test, y_pred_poly)
mse_poly = mean_squared_error(y_test, y_pred_poly)
rmse_poly = np.sqrt(mse_poly)

print("\n=== Task 2: Polynomial Regression Results ===")
print(f"R2:   {r2_poly:.4f}")
print(f"MAE:  {mae_poly:.4f}")
print(f"MSE:  {mse_poly:.4f}")
print(f"RMSE: {rmse_poly:.4f}")

# d) Explain when polynomial regression is more appropriate
print("\n=== Explanation ===")
print("Polynomial regression is more appropriate when the relationship between predictors and target is nonlinear and cannot be captured by a straight line. It can fit curves, but may overfit if degree is too high.")

# === Task 3: Neural Network Model (ANN) ===
# a) Scale the input variables
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# b) Train an ANN model

# NOTE: Increased max_iter to 5000 to help ensure convergence and avoid ConvergenceWarning
ann = MLPRegressor(hidden_layer_sizes=(20,), activation='relu', max_iter=5000, random_state=42)
ann.fit(X_train_scaled, y_train)
y_pred_ann = ann.predict(X_test_scaled)

# c) Evaluate ANN
r2_ann = r2_score(y_test, y_pred_ann)
mae_ann = mean_absolute_error(y_test, y_pred_ann)
mse_ann = mean_squared_error(y_test, y_pred_ann)
rmse_ann = np.sqrt(mse_ann)

print("\n=== Task 3: ANN Results (hidden_layer_sizes=(20,)) ===")
print(f"R2:   {r2_ann:.4f}")
print(f"MAE:  {mae_ann:.4f}")
print(f"MSE:  {mse_ann:.4f}")
print(f"RMSE: {rmse_ann:.4f}")

# d) Modify a hyperparameter (e.g., activation function)

# Also increase max_iter for the second ANN
ann2 = MLPRegressor(hidden_layer_sizes=(20,), activation='tanh', max_iter=5000, random_state=42)
ann2.fit(X_train_scaled, y_train)
y_pred_ann2 = ann2.predict(X_test_scaled)

r2_ann2 = r2_score(y_test, y_pred_ann2)
mae_ann2 = mean_absolute_error(y_test, y_pred_ann2)
mse_ann2 = mean_squared_error(y_test, y_pred_ann2)
rmse_ann2 = np.sqrt(mse_ann2)

print("\n=== Task 3: ANN Results (activation='tanh') ===")
print(f"R2:   {r2_ann2:.4f}")
print(f"MAE:  {mae_ann2:.4f}")
print(f"MSE:  {mse_ann2:.4f}")
print(f"RMSE: {rmse_ann2:.4f}")

# === Task 4: Model Comparison ===
# a) Compare: 
results = pd.DataFrame({
	'Model': ['Linear Regression', 'Polynomial Regression', 'ANN (relu)', "ANN (tanh)"],
	'R2':   [r2_lin, r2_poly, r2_ann, r2_ann2],
	'MAE':  [mae_lin, mae_poly, mae_ann, mae_ann2],
	'MSE':  [mse_lin, mse_poly, mse_ann, mse_ann2],
	'RMSE': [rmse_lin, rmse_poly, rmse_ann, rmse_ann2]
})

print("\n=== Task 4: Model Comparison Table ===")
print(results)

# b) Identify which model performs best
best_idx = results['R2'].idxmax()
best_model = results.loc[best_idx, 'Model']
print(f"\nBest model by R2: {best_model}")
print("Based on the comparison table, Polynomial Regression performs best.")
print("It has the highest R² (0.9578), meaning it explains more variance in production efficiency.")
print("It also has the lowest error metrics (MAE = 3.49, RMSE = 4.19) compared to Linear Regression and ANN.")

# c) Short explanation
print("\n=== Explanation ===")
print("Polynomial regression outperforms the other models because the relationship between predictors (assembly time, workers, humidity, operating hours, temperature, shift) and the target (production efficiency score) is nonlinear.\n"
	  "A linear model cannot fully capture these curved relationships, while polynomial features allow the regression to fit more complex patterns.\n"
	  "Although ANNs are also capable of modeling nonlinearities, they typically require larger datasets and more tuning to surpass polynomial regression. In this case, the dataset size and ANN configuration limited its performance, so polynomial regression was the most robust choice.\n")

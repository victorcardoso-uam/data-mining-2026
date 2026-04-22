"""
SESSION 23 — ANN TUNING ACTIVITY (WITH TODOs)

GOAL
----
Use the SAME dataset from Session 22 and explore how different ANN
hyperparameters affect model performance.

DATASET
-------
industrial_ann_teacher_example.csv

TARGET VARIABLE
---------------
production_quality_score

WHAT YOU MUST DO
----------------
1. Load the dataset
2. Define X and y
3. Split the data into training and testing sets
4. Scale the input variables
5. Complete one baseline ANN model
6. Create and test at least THREE additional ANN configurations
7. Calculate the following metrics for every model:
   - R2
   - MAE
   - MSE
   - RMSE
8. Compare the results and decide which configuration performed best

IMPORTANT
---------
You must complete all TODO sections.
The goal is not only to run the model, but to understand how:
- hidden layers
- neurons
- activation functions
- solvers
- max_iter
affect ANN performance.
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# ============================================================
# 1. LOAD DATA
# ============================================================

DATA_PATH = r"C:\Users\mario\OneDrive\Escritorio\School\8to Semestre\Data Miing\data-mining-course\Repositories\data-mining-2026\teams\team-04\session_23\industrial_ann_teacher_example.csv"

# TODO 1:
# Load the dataset into a DataFrame called data
data = pd.read_csv(DATA_PATH)

print("\n=== DATASET PREVIEW ===")
print(data.head())

print("\n=== DATASET SHAPE ===")
print(data.shape)


# ============================================================
# 2. DEFINE INPUTS (X) AND TARGET (y)
# ============================================================

# TODO 2:
# Define X using all columns except the target column
X = data.drop('production_quality_score', axis=1)

# TODO 3:
# Define y using only the target column
y = data['production_quality_score']


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ============================================================
# 4. SCALE INPUT FEATURES
# ============================================================

scaler = StandardScaler()

# TODO 4:
# Fit the scaler on X_train and transform X_train
X_train = scaler.fit_transform(X_train)

# TODO 5:
# Transform X_test using the same scaler
X_test = scaler.transform(X_test)


# ============================================================
# 5. FUNCTION TO TRAIN AND EVALUATE ONE ANN MODEL
# ============================================================

def train_and_evaluate_ann(hidden_layer_sizes, activation, solver, max_iter):
    """
    Trains one ANN configuration and returns the four evaluation metrics.
    """

    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        max_iter=max_iter,
        random_state=42
    )

    # TODO 6:
    # Train the model
    model.fit(X_train, y_train)

    # TODO 7:
    # Generate predictions using X_test
    y_pred = model.predict(X_test)

    # TODO 8:
    # Calculate metrics
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


# ============================================================
# 6. BASELINE MODEL
# ============================================================

print("\n=== BASELINE MODEL ===")

# TODO 9:
# Complete the baseline model configuration
baseline = train_and_evaluate_ann(
    hidden_layer_sizes=(10,),   # Example: (10,)
    activation="relu",           # Example: "relu"
    solver="adam",               # Example: "adam"
    max_iter=500              # Example: 500
)

print(baseline)


# ============================================================
# 7. ADDITIONAL EXPERIMENTS
# ============================================================

print("\n=== ADDITIONAL ANN CONFIGURATIONS ===")

# TODO 10:
# Replace these placeholders with at least THREE real experiments
# You may add more configurations if you want
experiments = [
    {"hidden_layer_sizes": (20, 10), "activation": "relu", "solver": "adam", "max_iter": 1000},
    {"hidden_layer_sizes": (50, 25, 10), "activation": "tanh", "solver": "lbfgs", "max_iter": 800},
    {"hidden_layer_sizes": (15,), "activation": "relu", "solver": "sgd", "max_iter": 1000},
]

results = []
results.append(baseline)

for exp in experiments:
    result = train_and_evaluate_ann(
        hidden_layer_sizes=exp["hidden_layer_sizes"],
        activation=exp["activation"],
        solver=exp["solver"],
        max_iter=exp["max_iter"]
    )
    results.append(result)


# ============================================================
# 8. COMPARISON TABLE
# ============================================================

results_df = pd.DataFrame(results)

print("\n=== COMPARISON TABLE ===")
print(results_df.round(4))

# TODO 11:
# Sort the comparison table by R2 in descending order
sorted_df = results_df.sort_values(by='R2', ascending=False)

print("\n=== SORTED RESULTS (BEST R2 FIRST) ===")
print(sorted_df.round(4))


# ============================================================
# 9. FINAL QUESTIONS
# ============================================================

print("\n=== QUESTIONS FOR YOUR TEAM ===")
print("1. Which ANN configuration performed best?")
print("2. Did adding more neurons always improve performance?")
print("3. Did adding more hidden layers always improve performance?")
print("4. Which activation function worked best?")
print("5. Which solver worked best?")
print("6. How did max_iter affect the results?")
print("7. If you had to keep only one model, which one would you choose and why?")


# ============================================================
# 10. ANSWERS TO THE QUESTIONS
# ============================================================

print("\n" + "="*80)
print("ANSWERS TO SESSION 23 - ANN TUNING QUESTIONS")
print("="*80)

# Answer 1
print("\n1. Which ANN configuration performed best?")
print("-" * 80)
best_model = sorted_df.iloc[0]
print(f"✓ Best Model: Configuration with {best_model['hidden_layer_sizes']}")
print(f"  - Activation: {best_model['activation']}")
print(f"  - Solver: {best_model['solver']}")
print(f"  - Max iterations: {int(best_model['max_iter'])}")
print(f"  - R²: {best_model['R2']:.4f}")
print(f"  - MAE: {best_model['MAE']:.4f}")
print(f"  - RMSE: {best_model['RMSE']:.4f}")

# Answer 2
print("\n2. Did adding more neurons always improve performance?")
print("-" * 80)
print("✗ NO. More neurons did NOT guarantee better performance:")
print(f"  - Baseline (10 neurons): R² = {sorted_df[sorted_df['hidden_layer_sizes'] == '(10,)']['R2'].values[0]:.2f}")
print(f"  - Model 1 (20, 10 neurons): R² = {sorted_df[sorted_df['hidden_layer_sizes'] == '(20, 10)']['R2'].values[0]:.2f}")
print(f"  - Model 3 (15 neurons): R² = {sorted_df[sorted_df['hidden_layer_sizes'] == '(15,)']['R2'].values[0]:.4f} ← BEST")
print(f"\n→ The simpler model with FEWER neurons (15) outperformed others")
print(f"→ This shows that simplicity generalizes better for small datasets")

# Answer 3
print("\n3. Did adding more hidden layers always improve performance?")
print("-" * 80)
print("✗ NO. More layers did NOT guarantee better performance:")
print(f"  - Baseline (1 layer): R² = {sorted_df[sorted_df['hidden_layer_sizes'] == '(10,)']['R2'].values[0]:.2f}")
print(f"  - Model 2 (3 layers: 50-25-10): R² = {sorted_df[sorted_df['hidden_layer_sizes'] == '(50, 25, 10)']['R2'].values[0]:.4f}")
print(f"  - Model 3 (1 layer: 15): R² = {sorted_df[sorted_df['hidden_layer_sizes'] == '(15,)']['R2'].values[0]:.4f} ← BEST")
print(f"\n→ A simple ONE-layer architecture was most effective")
print(f"→ For small datasets (240 samples), simplicity is better")

# Answer 4
print("\n4. Which activation function worked best?")
print("-" * 80)
relu_r2 = sorted_df[sorted_df['activation'] == 'relu']['R2'].max()
tanh_r2 = sorted_df[sorted_df['activation'] == 'tanh']['R2'].max()
print(f"✓ relu was more effective in this case:")
print(f"  - relu best: R² = {relu_r2:.4f}")
print(f"  - tanh best: R² = {tanh_r2:.4f}")
print(f"\n→ relu outperformed tanh")
print(f"→ relu is computationally more efficient")

# Answer 5
print("\n5. Which solver worked best?")
print("-" * 80)
sgd_r2 = sorted_df[sorted_df['solver'] == 'sgd']['R2'].max()
adam_r2 = sorted_df[sorted_df['solver'] == 'adam']['R2'].max()
lbfgs_r2 = sorted_df[sorted_df['solver'] == 'lbfgs']['R2'].max()
print(f"✓ sgd was the best solver:")
print(f"  - sgd best: R² = {sgd_r2:.4f} ← WINNER")
print(f"  - adam best: R² = {adam_r2:.4f}")
print(f"  - lbfgs best: R² = {lbfgs_r2:.4f}")
print(f"\n→ sgd outperformed adam and lbfgs for this problem")
print(f"→ sgd proved robust with proper iteration tuning")

# Answer 6
print("\n6. How did max_iter affect the results?")
print("-" * 80)
results_500 = results_df[results_df['max_iter'] == 500]['R2'].values[0]
results_800 = results_df[results_df['max_iter'] == 800]['R2'].values[0]
results_1000_sgd = sorted_df[sorted_df['solver'] == 'sgd']['R2'].values[0]
results_1000_adam = results_df[(results_df['max_iter'] == 1000) & (results_df['solver'] == 'adam')]['R2'].values[0]
print(f"✓ Increasing max_iter helped in most cases:")
print(f"  - max_iter=500 (adam): R² = {results_500:.2f}")
print(f"  - max_iter=800 (lbfgs): R² = {results_800:.4f}")
print(f"  - max_iter=1000 (adam): R² = {results_1000_adam:.4f}")
print(f"  - max_iter=1000 (sgd): R² = {results_1000_sgd:.4f}")
print(f"\n→ max_iter=1000 was generally better than 500")
print(f"→ But solver choice and architecture were MORE critical factors")

# Answer 7
print("\n7. If you had to keep only one model, which one would you choose and why?")
print("-" * 80)
print(f"✓ CHOOSE: Model with {best_model['hidden_layer_sizes']} neurons, {best_model['activation']}, {best_model['solver']}, max_iter={int(best_model['max_iter'])}")
print(f"\nReasons:")
print(f"  1. BEST PERFORMANCE: R² = {best_model['R2']:.4f} (explains {best_model['R2']*100:.2f}% of variance)")
print(f"  2. SIMPLEST: Only 1 hidden layer (easier to interpret and maintain)")
print(f"  3. LOWEST ERROR: MAE = {best_model['MAE']:.4f}, RMSE = {best_model['RMSE']:.4f}")
print(f"  4. GOOD GENERALIZATION: Simple architecture prevents overfitting on small dataset")
print(f"  5. FAST TRAINING: Fewer neurons = quicker training time")
print(f"\n→ This model demonstrates that 'less is more' in machine learning")
print(f"→ Implicit regularization from simplicity was more effective than complexity")

print("\n" + "="*80)
print("SUMMARY TABLE (SORTED BY R² - BEST FIRST)")
print("="*80)
print(sorted_df[['hidden_layer_sizes', 'activation', 'solver', 'max_iter', 'R2', 'MAE', 'MSE', 'RMSE']].round(4).to_string())
print("="*80)

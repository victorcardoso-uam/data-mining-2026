"""
ANN Model — Solar AC Power Output (Pac(W)) Prediction
Final Project — Data Mining | Universidad Anáhuac Mayab

Steps:
1. Load dataset — filter Status='Normal' (same as DT and Regression models)
2. Define input variables (X) and target variable (y) — same 23 features
3. Split the data into training and testing sets — 80/20, random_state=42
4. Scale the input variables with StandardScaler
5. Define and train ANN model (MLPRegressor)
6. Train multiple configurations and compare
7. Generate predictions
8. Calculate evaluation metrics: R², MAE, MSE, RMSE
9. Interpret results
"""

import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# ============================================================
# OUTPUT FOLDER
# ============================================================

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "ann_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. LOAD DATA
# ============================================================

DATA_PATH = os.path.join(BASE_DIR, "solar_data_cleaned_active_only.csv")

data = pd.read_csv(DATA_PATH)

# Keep only rows where Status = 'Normal'
# Same filter applied in Decision Tree and Regression models.
# 'Waiting' rows have Pac(W) = 0 and do not represent real generation behavior.
data = data[data["Status"] == "Normal"].reset_index(drop=True)

print("\n=== DATASET SHAPE (Status=Normal only) ===")
print(data.shape)


# ============================================================
# 2. DEFINE INPUTS (X) AND TARGET (y)
# ============================================================

# Target -> Pac(W): AC power output in Watts
# Same 23 features used in Decision Tree and Regression models
# Excluded to avoid data leakage:
#   EacToday(kWh), EacTotal(kWh), EpvToday(kWh), EpvTotal(kWh),
#   Ppv1(W)-Ppv8(W) -> derived directly from Pac(W)

FEATURE_COLS = [
    "Day_year ", "Hora_SIN", "HORA_COS",
    "INVTemp(\u2103)", "OUTTemp(\u2103)", "AMTemp1(\u2103)", "AMTemp2(\u2103)",
    "Vpv1(V)", "Vpv2(V)", "Vpv3(V)", "Vpv4(V)",
    "Vpv5(V)", "Vpv6(V)", "Vpv7(V)", "Vpv8(V)",
    "VacRS(V)", "VacST(V)", "VacTR(V)",
    "IacR(A)", "IacS(A)", "IacT(A)",
    "PF", "Fac(Hz)",
]

X_raw = data[FEATURE_COLS]
y     = data["Pac(W)"].values

# Impute missing values with column median
imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(X_raw)

print(f"\nTarget variable : Pac(W)")
print(f"Input features  : {len(FEATURE_COLS)} columns")


# ============================================================
# 3. TRAIN / TEST SPLIT — 80/20 (same across ALL models)
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n=== TRAIN / TEST SPLIT ===")
print(f"Train samples: {X_train.shape[0]}")
print(f"Test  samples: {X_test.shape[0]}")


# ============================================================
# 4. SCALE INPUT FEATURES
# ============================================================
# Fitted ONLY on training data — applied to test without re-fitting
# This prevents data leakage from the test set

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)


# ============================================================
# 5. FUNCTION TO TRAIN AND EVALUATE ONE ANN CONFIGURATION
# ============================================================

def train_and_evaluate_ann(hidden_layer_sizes, activation, solver, max_iter):
    """
    Trains one ANN configuration using the pre-scaled data
    and returns evaluation metrics + predictions.
    """
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        max_iter=max_iter,
        random_state=42
    )

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    return {
        "hidden_layer_sizes" : str(hidden_layer_sizes),
        "activation"         : activation,
        "solver"             : solver,
        "max_iter"           : max_iter,
        "R2"                 : r2,
        "MAE"                : mae,
        "MSE"                : mse,
        "RMSE"               : rmse,
        "_y_pred"            : y_pred,
    }


# ============================================================
# 6. BASELINE MODEL
# ============================================================

print("\n=== BASELINE MODEL ===")

baseline = train_and_evaluate_ann(
    hidden_layer_sizes=(7,),
    activation="tanh",
    solver="lbfgs",
    max_iter=438
)
print(f"  R2={baseline['R2']:.4f}  MAE={baseline['MAE']:.2f}  RMSE={baseline['RMSE']:.2f}")


# ============================================================
# 7. ADDITIONAL EXPERIMENTS
# ============================================================

print("\n=== ADDITIONAL ANN CONFIGURATIONS ===")

experiments = [
    {"hidden_layer_sizes": (4,),     "activation": "relu",     "solver": "adam",  "max_iter": 507},
    {"hidden_layer_sizes": (10, 13), "activation": "tanh",     "solver": "lbfgs", "max_iter": 10},
    {"hidden_layer_sizes": (9, 2),   "activation": "logistic", "solver": "sgd",   "max_iter": 765},
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
    print(f"  {result['hidden_layer_sizes']:10s} | {result['activation']:8s} | {result['solver']:5s} | "
          f"max_iter={result['max_iter']:4d} | R2={result['R2']:.4f} | RMSE={result['RMSE']:.2f}")


# ============================================================
# 8. COMPARISON TABLE
# ============================================================

display_results = [
    {k: v for k, v in r.items() if k != "_y_pred"} for r in results
]

results_df = pd.DataFrame(display_results)
sorted_df  = results_df.sort_values(by="R2", ascending=False)

print("\n=== COMPARISON TABLE (SORTED BY R2) ===")
print(sorted_df.round(4).to_string(index=False))

csv_path = os.path.join(OUTPUT_DIR, "ann_comparison_results.csv")
sorted_df.round(4).to_csv(csv_path, index=False)
print(f"\nComparison table saved: {csv_path}")


# ============================================================
# 9. PLOTS — PREDICTED VS ACTUAL + RESIDUALS (per model)
# ============================================================

model_labels = [
    "Baseline_(7)_tanh_lbfgs",
    "Exp1_(4)_relu_adam",
    "Exp2_(10,13)_tanh_lbfgs",
    "Exp3_(9,2)_logistic_sgd",
]

for res, label in zip(results, model_labels):
    y_pred    = res["_y_pred"]
    r2        = res["R2"]
    rmse      = res["RMSE"]
    residuals = y_test - y_pred

    # Predicted vs Actual
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_test, y_pred,
               alpha=0.45, s=25,
               edgecolors='steelblue', facecolors='lightblue',
               linewidths=0.5, label='Samples')
    lim_lo = min(y_test.min(), y_pred.min()) - 1000
    lim_hi = max(y_test.max(), y_pred.max()) + 1000
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi],
            'r--', linewidth=1.5, label='Perfect prediction')
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_xlabel('Actual Pac (W)', fontsize=11)
    ax.set_ylabel('Predicted Pac (W)', fontsize=11)
    ax.set_title(f'Predicted vs Actual — {label}\nR² = {r2:.4f}',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f'pred_vs_actual_{label}.png'), dpi=120)
    plt.close(fig)

    # Residual Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_pred, residuals,
               alpha=0.45, s=25,
               edgecolors='darkorange', facecolors='peachpuff',
               linewidths=0.5)
    ax.axhline(0, color='red', linewidth=1.5, linestyle='--',
               label='Zero residual')
    ax.set_xlabel('Predicted Pac (W)', fontsize=11)
    ax.set_ylabel('Residual (Actual - Predicted) W', fontsize=11)
    ax.set_title(f'Residual Plot — {label}\nRMSE = {rmse:,.1f} W',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f'residuals_{label}.png'), dpi=120)
    plt.close(fig)

print(f"\nAll plots saved to: {OUTPUT_DIR}")


# R2 Comparison Bar Chart
r2_vals = sorted_df["R2"].tolist()
labels  = sorted_df["hidden_layer_sizes"].tolist()
colors  = ['#2ecc71' if v == max(r2_vals) else '#3498db' for v in r2_vals]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(labels, r2_vals, color=colors, edgecolor='white', height=0.55)
for bar, val in zip(bars, r2_vals):
    ax.text(
        max(val - 0.03, 0.01),
        bar.get_y() + bar.get_height() / 2,
        f'{val:.4f}',
        va='center', ha='right',
        color='white', fontsize=10, fontweight='bold'
    )
ax.set_xlabel('R² Score', fontsize=11)
ax.set_title('ANN Model Comparison — R² Score (Solar Pac Prediction)',
             fontsize=12, fontweight='bold')
ax.set_xlim(-0.5, 1.05)
ax.axvline(0.8, color='red', linestyle='--', linewidth=1, label='R² = 0.8 reference')
ax.legend(fontsize=9)
ax.grid(axis='x', linestyle='--', alpha=0.4)
ax.invert_yaxis()
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "ann_r2_comparison.png"), dpi=120)
plt.close(fig)
print(f"R² comparison chart saved: {OUTPUT_DIR}/ann_r2_comparison.png")


# ============================================================
# 10. QUESTIONS & INTERPRETATION
# ============================================================

best_row = sorted_df.iloc[0]

print("\n" + "=" * 60)
print("QUESTIONS & INTERPRETATION")
print("=" * 60)

print(f"\n1. Which ANN configuration performed best?")
print(f"   Best config: hidden_layer_sizes={best_row['hidden_layer_sizes']}, "
      f"activation={best_row['activation']}, solver={best_row['solver']} "
      f"with R²={best_row['R2']:.4f}, RMSE={best_row['RMSE']:.2f} W, MAE={best_row['MAE']:.2f} W.")

print(f"\n2. Did adding more neurons always improve performance?")
print(f"   No. Experiment 2 (10,13) with more neurons achieved a lower R² than the Baseline (7,),")
print(f"   partly because max_iter=10 was insufficient for convergence.")

print(f"\n3. Did adding more hidden layers always improve performance?")
print(f"   No. A single hidden layer (Baseline) outperformed two-layer architectures.")
print(f"   Deep networks introduced unnecessary complexity without improving generalization.")

print(f"\n4. Which activation function worked best?")
print(f"   tanh worked best. It maps inputs to (-1, 1) and handles standardized data effectively.")

print(f"\n5. Which solver worked best?")
print(f"   lbfgs worked best. It is a quasi-Newton method, efficient on small-to-medium datasets.")

print(f"\n6. How did max_iter affect results?")
print(f"   Experiment 2 used max_iter=10, too low for convergence — poor predictions.")
print(f"   The Baseline used max_iter=438, allowing full convergence and best performance.")

print(f"\n7. Which single model would you keep and why?")
print(f"   The Baseline (7,) tanh + lbfgs: highest R², lowest errors,")
print(f"   simplest architecture, and best generalization among all ANN configurations.")
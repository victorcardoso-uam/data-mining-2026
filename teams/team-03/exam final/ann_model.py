"""
Session Final - Artificial Neural Network (ANN)
Course: Data Mining | Universidad Anáhuac Mayab

Problem: Predict monthly Earth Skin Temperature (°C) in Chihuahua
         using solar irradiance, precipitation, and wind speed.

Target variable (y): temperature_c
Input variables (X): solar_irradiance_wm2, precipitation_mm, wind_speed_ms, MONTH_NUM

Note: Input variables MUST be scaled before training an ANN.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

os.makedirs("data/processed", exist_ok=True)
os.makedirs("outputs/plots", exist_ok=True)

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

print("=" * 65)
print("ARTIFICIAL NEURAL NETWORK — TEMPERATURE PREDICTION (CHIHUAHUA)")
print("=" * 65)

df = pd.read_csv("data/processed/chihuahua_dataset.csv")
print(f"Dataset shape: {df.shape}")


# --------------------------------------------------
# 2. FEATURES AND TARGET
# --------------------------------------------------

FEATURES = ['solar_irradiance_wm2', 'precipitation_mm', 'wind_speed_ms', 'MONTH_NUM']
TARGET   = 'temperature_c'


X = df[FEATURES]
y = df[TARGET]

# --------------------------------------------------
# 3. TRAIN / TEST SPLIT (80/20)
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")

# --------------------------------------------------
# 4. SCALE FEATURES (REQUIRED for ANN)
# --------------------------------------------------

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit only on training data
X_test_scaled  = scaler.transform(X_test)         # apply same scale to test

print("\nFeature scaling applied (StandardScaler):")
print("  Mean (train):", scaler.mean_.round(3))
print("  Std  (train):", scaler.scale_.round(3))

# --------------------------------------------------
# 5. HELPER FUNCTION
# --------------------------------------------------

def evaluate(name, y_test, y_pred):
    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)
    print(f"\n  {name}")
    print(f"    R²   = {r2:.4f}")
    print(f"    MAE  = {mae:.4f}")
    print(f"    MSE  = {mse:.4f}")
    print(f"    RMSE = {rmse:.4f}")
    return {'model': name, 'R2': round(r2,4), 'MAE': round(mae,4),
            'MSE': round(mse,4), 'RMSE': round(rmse,4)}

# --------------------------------------------------
# 6. ANN CONFIGURATION A
#    Small network — 1 hidden layer, relu, 500 iterations
# --------------------------------------------------

print("\n--- ANN Configuration A: (100,) | relu | max_iter=500 ---")
ann_a = MLPRegressor(
    hidden_layer_sizes=(100,),    # 1 hidden layer with 100 neurons
    activation='relu',            # Rectified Linear Unit activation
    max_iter=500,
    random_state=42
)
ann_a.fit(X_train_scaled, y_train)
pred_a = ann_a.predict(X_test_scaled)
metrics_a = evaluate("ANN — Config A: (100,) relu 500 iter", y_test, pred_a)
print(f"  Converged: {ann_a.n_iter_ < ann_a.max_iter} | Iterations run: {ann_a.n_iter_}")

# --------------------------------------------------
# 7. ANN CONFIGURATION B
#    Larger network — 2 hidden layers, tanh, 1000 iterations
# --------------------------------------------------

print("\n--- ANN Configuration B: (100, 50) | tanh | max_iter=1000 ---")
ann_b = MLPRegressor(
    hidden_layer_sizes=(100, 50),  # 2 hidden layers: 100 → 50 neurons
    activation='tanh',             # Hyperbolic tangent activation
    max_iter=1000,
    random_state=42
)
ann_b.fit(X_train_scaled, y_train)
pred_b = ann_b.predict(X_test_scaled)
metrics_b = evaluate("ANN — Config B: (100,50) tanh 1000 iter", y_test, pred_b)
print(f"  Converged: {ann_b.n_iter_ < ann_b.max_iter} | Iterations run: {ann_b.n_iter_}")

# --------------------------------------------------
# 8. ANN CONFIGURATION C (BONUS)
#    Deeper network — 3 layers, relu, 2000 iterations
# --------------------------------------------------

print("\n--- ANN Configuration C: (128, 64, 32) | relu | max_iter=2000 ---")
ann_c = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32),  # 3 hidden layers
    activation='relu',
    max_iter=2000,
    random_state=42
)
ann_c.fit(X_train_scaled, y_train)
pred_c = ann_c.predict(X_test_scaled)
metrics_c = evaluate("ANN — Config C: (128,64,32) relu 2000 iter", y_test, pred_c)
print(f"  Converged: {ann_c.n_iter_ < ann_c.max_iter} | Iterations run: {ann_c.n_iter_}")

# --------------------------------------------------
# 9. PLOTS
# --------------------------------------------------

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Artificial Neural Network — Model Evaluation (Chihuahua Temperature)", fontsize=14, fontweight='bold')

ann_data = [
    ("Config A: (100,) relu", pred_a, metrics_a, 'steelblue'),
    ("Config B: (100,50) tanh", pred_b, metrics_b, 'darkorange'),
    ("Config C: (128,64,32) relu", pred_c, metrics_c, 'green'),
]

lims_all = [
    min(y_test.min(), pred_a.min(), pred_b.min(), pred_c.min()),
    max(y_test.max(), pred_a.max(), pred_b.max(), pred_c.max())
]

for col, (name, pred, m, color) in enumerate(ann_data):
    # Predicted vs Actual
    ax = axes[0, col]
    ax.scatter(y_test, pred, alpha=0.65, color=color, edgecolors='k', linewidths=0.3)
    ax.plot(lims_all, lims_all, 'r--', lw=1.5)
    ax.set_title(f"{name}\nR²={m['R2']:.4f} | RMSE={m['RMSE']:.4f}")
    ax.set_xlabel("Actual Temp (°C)")
    ax.set_ylabel("Predicted Temp (°C)")

    # Residuals
    ax2 = axes[1, col]
    residuals = y_test.values - pred
    ax2.scatter(pred, residuals, alpha=0.65, color=color, edgecolors='k', linewidths=0.3)
    ax2.axhline(0, color='red', linestyle='--', lw=1.5)
    ax2.set_title(f"Residuals — {name}")
    ax2.set_xlabel("Predicted Temp (°C)")
    ax2.set_ylabel("Residuals")

plt.tight_layout()
plt.savefig("outputs/plots/ann_evaluation.png", dpi=150, bbox_inches='tight')
plt.close()
print("\nPlot saved: outputs/plots/ann_evaluation.png")

# --------------------------------------------------
# 10. LOSS CURVE PLOT
# --------------------------------------------------

fig2, ax = plt.subplots(figsize=(10, 5))
ax.plot(ann_a.loss_curve_, label='Config A: (100,) relu', color='steelblue')
ax.plot(ann_b.loss_curve_, label='Config B: (100,50) tanh', color='darkorange')
ax.plot(ann_c.loss_curve_, label='Config C: (128,64,32) relu', color='green')
ax.set_title("ANN Training Loss Curves")
ax.set_xlabel("Iteration")
ax.set_ylabel("Training Loss (MSE)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/plots/ann_loss_curves.png", dpi=150, bbox_inches='tight')
plt.close()
print("Plot saved: outputs/plots/ann_loss_curves.png")

# --------------------------------------------------
# 11. COMPARISON SUMMARY
# --------------------------------------------------

print("\n" + "=" * 65)
print("COMPARISON: ANN CONFIGURATIONS")
print("=" * 65)
summary = pd.DataFrame([metrics_a, metrics_b, metrics_c])
print(summary.to_string(index=False))

print("""
INTERPRETATION:
  - Config A uses a single hidden layer with relu. Simple and fast,
    good baseline. relu avoids vanishing gradients but can produce
    dead neurons if learning rate is too high.
  - Config B uses two hidden layers with tanh. tanh centers outputs
    around zero, which can improve convergence. Two layers allow
    the network to learn more complex feature interactions.
  - Config C uses three hidden layers (128→64→32). The decreasing
    layer size acts as a funnel, forcing the network to compress
    information into increasingly abstract representations.
  - More iterations (max_iter) allow the optimizer to converge
    further. If a model hasn't converged, increasing max_iter helps.
  - Scaling is ESSENTIAL: without it, features on different scales
    cause the gradient descent to oscillate and converge slowly.
""")

# Save results
summary.to_csv("data/processed/ann_results.csv", index=False)
print("Results saved to: data/processed/ann_results.csv")

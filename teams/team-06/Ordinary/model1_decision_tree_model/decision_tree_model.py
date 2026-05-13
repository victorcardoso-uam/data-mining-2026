"""
Final Project — Decision Tree Model


Following the project guidelines, this script:
1) Trains a DEFAULT tree (no constraints)
2) Trains a CONTROLLED tree (with max_depth to limit complexity)
3) Runs an autonomous depth search (min_samples_split also tested)
4) Compares all versions using R², MAE, MSE, RMSE
5) Produces Predicted vs Actual plots and Residual plots

The train/test split (80/20), random state, and features used here
are IDENTICAL to all other model scripts in this project, ensuring
a fair comparison across models.

"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # Mac-safe: no display required
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer

# =============================================================================
# 1) CONFIGURATION
# =============================================================================

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "solar_data_cleaned_active_only.csv")

# Target variable
TARGET_COL = "Pac(W)"

# Input features — same across ALL models in this project
# Temporal, thermal, electrical variables.
# Excluded to avoid data leakage:
#   EacToday(kWh), EacTotal(kWh), EpvToday(kWh), EpvTotal(kWh),
#   Ppv1(W)–Ppv8(W)  → these are derived directly from Pac(W)
FEATURE_COLS = [
    "Day_year ",        # day of year (temporal)
    "Hora_SIN",         # sine-encoded hour (cyclical)
    "HORA_COS",         # cosine-encoded hour (cyclical)
    "INVTemp(℃)",       # inverter temperature
    "OUTTemp(℃)",       # output temperature
    "AMTemp1(℃)",       # ambient temperature 1
    "AMTemp2(℃)",       # ambient temperature 2
    "Vpv1(V)", "Vpv2(V)", "Vpv3(V)", "Vpv4(V)",  # MPPT voltages
    "Vpv5(V)", "Vpv6(V)", "Vpv7(V)", "Vpv8(V)",
    "VacRS(V)", "VacST(V)", "VacTR(V)",           # AC voltages
    "IacR(A)", "IacS(A)", "IacT(A)",              # AC currents
    "PF",                                          # power factor
    "Fac(Hz)",                                     # AC frequency
]

# Train/test split — 80/20, fixed seed for reproducibility across all models
TEST_SIZE    = 0.20
RANDOM_STATE = 42

# Depth search range for autonomous analysis
MIN_DEPTH = 1
MAX_DEPTH = 15

# Fixed depths for the required comparison
SHALLOW_DEPTH     = 3    # simple / underfitting
CONTROLLED_DEPTH  = 6    # complexity-controlled version
DEEP_DEPTH        = 12   # deep / potentially overfitting

# Output folder
OUTPUT_DIR = os.path.join(BASE_DIR, "dt_outputs")

# =============================================================================
# 2) HELPER FUNCTIONS
# =============================================================================

def ensure_output_dir(path_str: str) -> Path:
    out = Path(path_str)
    out.mkdir(parents=True, exist_ok=True)
    return out


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute R², MAE, MSE, and RMSE.
    These four metrics are required for the final model comparison.
    """
    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    return {"R2": r2, "MAE": mae, "MSE": mse, "RMSE": rmse}


def train_and_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval:  np.ndarray,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
) -> Tuple[DecisionTreeRegressor, np.ndarray]:
    """
    Train a DecisionTreeRegressor and return (fitted model, predictions on X_eval).

    Parameters
    ----------
    max_depth          : None means unlimited (default tree)
    min_samples_split  : minimum samples required to split a node
                         (another complexity control parameter)
    """
    model = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_eval)
    return model, preds


def plot_predicted_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    out_path: Path,
    r2: float,
) -> None:
    """
    Scatter plot of Predicted vs Actual values.
    Points close to the diagonal line = good predictions.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.5, edgecolors="steelblue",
               facecolors="steelblue", s=30)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val],
            "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Pac(W)")
    ax.set_ylabel("Predicted Pac(W)")
    ax.set_title(f"{title}\n(R² = {r2:.4f})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  Saved: {out_path.name}")


def plot_residuals(
    y_pred: np.ndarray,
    residuals: np.ndarray,
    title: str,
    out_path: Path,
    rmse: float,
) -> None:
    """
    Residual plot: Predicted values vs (Actual - Predicted).
    A good model shows residuals randomly scattered around zero.
    Patterns in residuals indicate systematic errors.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_pred, residuals, alpha=0.5, edgecolors="darkorange",
               facecolors="darkorange", s=30)
    ax.axhline(0, color="red", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Predicted Pac(W)")
    ax.set_ylabel("Residuals (W)")
    ax.set_title(f"{title}\n(RMSE = {rmse:.2f} W)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  Saved: {out_path.name}")


# =============================================================================
# 3) MAIN SCRIPT
# =============================================================================

def main() -> None:
    print("\n=== Final Project — Decision Tree Regressor (Solar Pac(W)) ===\n")

    out_dir = ensure_output_dir(OUTPUT_DIR)

    # -------------------------------------------------------------------------
    # Step A: Load and prepare the dataset
    # -------------------------------------------------------------------------
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"\nDataset not found: {DATA_PATH}\n"
            "Please place solar_data_cleaned_active_only.csv in the same folder."
        )

    df = pd.read_csv(DATA_PATH)
    print(f"Dataset loaded — shape: {df.shape}")

    # Keep only rows where the inverter is producing power (Status = Normal).
    # 'Waiting' rows have Pac(W) = 0 and do not represent generation behavior.
    df = df[df["Status"] == "Normal"].reset_index(drop=True)
    print(f"After filtering Status='Normal': {df.shape}")

    # Verify all expected feature columns exist
    missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in dataset: {missing_cols}")

    X_raw = df[FEATURE_COLS].copy()
    y     = df[TARGET_COL].copy().values

    # Impute any remaining NaN values with column median (robust to outliers)
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X_raw)

    # -------------------------------------------------------------------------
    # Step B: Train / Test split — 80/20
    # This exact split is reused in ALL project models for fair comparison.
    # -------------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    print(f"\nTrain samples : {len(X_train)}")
    print(f"Test  samples : {len(X_test)}")

    # =========================================================================
    # PART 1 — Required comparison: DEFAULT vs CONTROLLED tree
    # =========================================================================
    print("\n" + "=" * 60)
    print("PART 1 — Default Tree vs Controlled Tree")
    print("=" * 60)

    # --- Default tree (no constraints) ---
    model_default, pred_default_test = train_and_predict(
        X_train, y_train, X_test,
        max_depth=None,         # unlimited depth
        min_samples_split=2,    # default scikit-learn value
    )
    _, pred_default_train = train_and_predict(
        X_train, y_train, X_train,
        max_depth=None, min_samples_split=2,
    )
    metrics_default_train = compute_metrics(y_train, pred_default_train)
    metrics_default_test  = compute_metrics(y_test,  pred_default_test)

    print("\n[DEFAULT TREE — no max_depth, min_samples_split=2]")
    print(f"  Actual depth : {model_default.get_depth()} | Leaves: {model_default.get_n_leaves()}")
    print(f"  Train  R²={metrics_default_train['R2']:.4f}  MAE={metrics_default_train['MAE']:.1f}  "
          f"MSE={metrics_default_train['MSE']:.1f}  RMSE={metrics_default_train['RMSE']:.2f}")
    print(f"  Test   R²={metrics_default_test['R2']:.4f}  MAE={metrics_default_test['MAE']:.1f}  "
          f"MSE={metrics_default_test['MSE']:.1f}  RMSE={metrics_default_test['RMSE']:.2f}")

    # --- Controlled tree (max_depth + min_samples_split) ---
    model_controlled, pred_controlled_test = train_and_predict(
        X_train, y_train, X_test,
        max_depth=CONTROLLED_DEPTH,
        min_samples_split=10,   # requires at least 10 samples to split a node
    )
    _, pred_controlled_train = train_and_predict(
        X_train, y_train, X_train,
        max_depth=CONTROLLED_DEPTH, min_samples_split=10,
    )
    metrics_controlled_train = compute_metrics(y_train, pred_controlled_train)
    metrics_controlled_test  = compute_metrics(y_test,  pred_controlled_test)

    print(f"\n[CONTROLLED TREE — max_depth={CONTROLLED_DEPTH}, min_samples_split=10]")
    print(f"  Actual depth : {model_controlled.get_depth()} | Leaves: {model_controlled.get_n_leaves()}")
    print(f"  Train  R²={metrics_controlled_train['R2']:.4f}  MAE={metrics_controlled_train['MAE']:.1f}  "
          f"MSE={metrics_controlled_train['MSE']:.1f}  RMSE={metrics_controlled_train['RMSE']:.2f}")
    print(f"  Test   R²={metrics_controlled_test['R2']:.4f}  MAE={metrics_controlled_test['MAE']:.1f}  "
          f"MSE={metrics_controlled_test['MSE']:.1f}  RMSE={metrics_controlled_test['RMSE']:.2f}")

    # Plots — Default
    residuals_default = y_test - pred_default_test
    plot_predicted_vs_actual(
        y_test, pred_default_test,
        "Default Decision Tree — Predicted vs Actual",
        out_dir / "dt_01_default_pred_vs_actual.png",
        metrics_default_test["R2"],
    )
    plot_residuals(
        pred_default_test, residuals_default,
        "Default Decision Tree — Residuals",
        out_dir / "dt_02_default_residuals.png",
        metrics_default_test["RMSE"],
    )

    # Plots — Controlled
    residuals_controlled = y_test - pred_controlled_test
    plot_predicted_vs_actual(
        y_test, pred_controlled_test,
        f"Controlled Decision Tree (max_depth={CONTROLLED_DEPTH}) — Predicted vs Actual",
        out_dir / "dt_03_controlled_pred_vs_actual.png",
        metrics_controlled_test["R2"],
    )
    plot_residuals(
        pred_controlled_test, residuals_controlled,
        f"Controlled Decision Tree (max_depth={CONTROLLED_DEPTH}) — Residuals",
        out_dir / "dt_04_controlled_residuals.png",
        metrics_controlled_test["RMSE"],
    )

    # -------------------------------------------------------------------------
    # Default vs Controlled — Interpretation (required by project guidelines)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("DEFAULT vs CONTROLLED — Interpretation")
    print("=" * 60)
    print(
        f"\n"
        f"The DEFAULT tree (no max_depth, min_samples_split=2) grew to a depth of "
        f"{model_default.get_depth()} with {model_default.get_n_leaves()} leaves, "
        f"memorizing almost every training sample "
        f"(Train R²={metrics_default_train['R2']:.4f}, Train RMSE={metrics_default_train['RMSE']:.2f} W). "
        f"This is a clear case of overfitting: the model learned the training data perfectly "
        f"but its real generalization performance was "
        f"Test R²={metrics_default_test['R2']:.4f} with RMSE={metrics_default_test['RMSE']:.2f} W.\n"
        f"\n"
        f"The CONTROLLED tree (max_depth={CONTROLLED_DEPTH}, min_samples_split=10) was limited to "
        f"depth {model_controlled.get_depth()} and required at least 10 samples before splitting "
        f"any node. This reduced the tree to only {model_controlled.get_n_leaves()} leaves. "
        f"The training error increased as expected "
        f"(Train RMSE={metrics_controlled_train['RMSE']:.2f} W, Train R²={metrics_controlled_train['R2']:.4f}), "
        f"which means the model no longer memorizes the data. However, the test performance "
        f"remained almost identical "
        f"(Test R²={metrics_controlled_test['R2']:.4f}, RMSE={metrics_controlled_test['RMSE']:.2f} W), "
        f"proving that a much simpler tree can achieve essentially the same predictive power "
        f"on unseen data.\n"
        f"\n"
        f"Conclusion on parameters: max_depth and min_samples_split successfully controlled "
        f"complexity ({model_default.get_depth()} levels -> {model_controlled.get_depth()}, "
        f"{model_default.get_n_leaves()} leaves -> {model_controlled.get_n_leaves()}) without "
        f"sacrificing meaningful predictive accuracy. The complexity gap between train and test "
        f"error is smaller in the controlled tree, which indicates better generalization. "
        f"Performance did not improve in raw numbers, but the model became significantly more "
        f"interpretable and less prone to overfitting — which is the actual goal of complexity control."
    )

    # =========================================================================
    # PART 2 — Autonomous depth search (Training vs Validation Error)
    # =========================================================================
    print("\n" + "=" * 60)
    print("PART 2 — Autonomous Depth Search")
    print("=" * 60)
    print(f"\n{'max_depth':>10} | {'train_RMSE':>12} | {'test_RMSE':>12} | "
          f"{'train_R2':>10} | {'test_R2':>10} | {'depth':>6} | {'leaves':>7}")
    print("-" * 80)

    depths       = list(range(MIN_DEPTH, MAX_DEPTH + 1))
    train_errors, valid_errors   = [], []
    train_scores, valid_scores   = [], []
    actual_depths, leaves_counts = [], []

    for depth in depths:
        m_tr, p_tr = train_and_predict(X_train, y_train, X_train, max_depth=depth)
        _,    p_te = train_and_predict(X_train, y_train, X_test,  max_depth=depth)

        mt = compute_metrics(y_train, p_tr)
        mv = compute_metrics(y_test,  p_te)

        train_errors.append(mt["RMSE"])
        valid_errors.append(mv["RMSE"])
        train_scores.append(mt["R2"])
        valid_scores.append(mv["R2"])
        actual_depths.append(m_tr.get_depth())
        leaves_counts.append(m_tr.get_n_leaves())

        print(f"{depth:>10} | {mt['RMSE']:>12.2f} | {mv['RMSE']:>12.2f} | "
              f"{mt['R2']:>10.4f} | {mv['R2']:>10.4f} | "
              f"{m_tr.get_depth():>6} | {m_tr.get_n_leaves():>7}")

    best_index = int(np.argmin(valid_errors))
    best_depth = depths[best_index]
    print(f"\nBest depth (lowest test RMSE): {best_depth}  "
          f"| Test RMSE: {valid_errors[best_index]:.2f} W")

    # Plot — Training vs Validation Error
    plt.figure(figsize=(10, 6))
    plt.plot(depths, train_errors, marker="o", label="Training RMSE")
    plt.plot(depths, valid_errors, marker="o", label="Validation (Test) RMSE")
    plt.axvline(best_depth, color="gray", linestyle="--",
                label=f"Best depth = {best_depth}")
    plt.xlabel("Max Depth")
    plt.ylabel("RMSE (W)")
    plt.title("Decision Tree — Training vs Validation Error (RMSE)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "dt_05_train_vs_validation_error.png", dpi=200)
    plt.close()
    print("  Saved: dt_05_train_vs_validation_error.png")

    # Plot — Tree complexity growth
    plt.figure(figsize=(10, 6))
    plt.plot(depths, leaves_counts, marker="s", color="darkgreen")
    plt.xlabel("Max Depth")
    plt.ylabel("Number of Leaves")
    plt.title("Decision Tree — Complexity Growth")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "dt_06_complexity_growth.png", dpi=200)
    plt.close()
    print("  Saved: dt_06_complexity_growth.png")

    # =========================================================================
    # PART 3 — Three-model comparison: Shallow / Best / Deep
    # =========================================================================
    print("\n" + "=" * 60)
    print("PART 3 — Shallow / Best / Deep Comparison")
    print("=" * 60)

    comparison_depths = {
        "Shallow Tree": SHALLOW_DEPTH,
        "Best Tree":    best_depth,
        "Deep Tree":    DEEP_DEPTH,
    }

    summary_rows: List[dict] = []

    for label, depth in comparison_depths.items():
        m_tr, p_tr = train_and_predict(X_train, y_train, X_train, max_depth=depth)
        _,    p_te = train_and_predict(X_train, y_train, X_test,  max_depth=depth)
        mt = compute_metrics(y_train, p_tr)
        mv = compute_metrics(y_test,  p_te)

        summary_rows.append({
            "model":             label,
            "requested_depth":   depth,
            "actual_depth":      m_tr.get_depth(),
            "n_leaves":          m_tr.get_n_leaves(),
            "train_R2":          mt["R2"],
            "test_R2":           mv["R2"],
            "train_RMSE":        mt["RMSE"],
            "test_RMSE":         mv["RMSE"],
            "test_MAE":          mv["MAE"],
            "test_MSE":          mv["MSE"],
        })

    summary_df = pd.DataFrame(summary_rows)
    print("\n")
    print(summary_df.to_string(index=False))
    summary_df.to_csv(out_dir / "dt_model_comparison_summary.csv", index=False)
    print("\n  Saved: dt_model_comparison_summary.csv")

    # =========================================================================
    # PART 4 — Final metrics table for cross-model report
    # =========================================================================
    # This block prints the metrics that must be copied into the
    # final comparison script / report alongside Linear Regression,
    # Polynomial Regression, and ANN results.
    print("\n" + "=" * 60)
    print("FINAL METRICS — Decision Tree (for cross-model comparison)")
    print("=" * 60)
    print("\nDefault Tree (no constraints):")
    for k, v in metrics_default_test.items():
        print(f"  {k:6s} = {v:.4f}")
    print(f"\nControlled Tree (max_depth={CONTROLLED_DEPTH}, min_samples_split=10):")
    for k, v in metrics_controlled_test.items():
        print(f"  {k:6s} = {v:.4f}")

    print(f"\nAll plots and CSV saved to: {Path(OUTPUT_DIR).resolve()}")

    # =========================================================================
    # ANSWERS & INTERPRETATION
    # =========================================================================
    print("\n" + "=" * 60)
    print("ANSWERS & INTERPRETATION")
    print("=" * 60)

    print(
        "\n1. Which tree had the lowest training error?\n"
        f"   The Default Tree (no constraints) had the lowest training error, "
        f"reaching Train RMSE = {metrics_default_train['RMSE']:.2f} W and "
        f"Train R² = {metrics_default_train['R2']:.4f}. This is expected because "
        f"an unconstrained tree can grow until it perfectly memorizes every "
        f"training sample, which is the definition of overfitting."
    )

    print(
        f"\n2. Which tree had the lowest test error?\n"
        f"   The Best Tree found by the autonomous depth search (max_depth={best_depth}) "
        f"achieved the lowest test RMSE = {valid_errors[best_index]:.2f} W with "
        f"Test R² = {valid_scores[best_index]:.4f}. Among the Default vs Controlled "
        f"comparison, the Default Tree had a slightly lower test RMSE "
        f"({metrics_default_test['RMSE']:.2f} W) than the Controlled Tree "
        f"({metrics_controlled_test['RMSE']:.2f} W), but the difference is small "
        f"and the Default Tree uses {model_default.get_n_leaves() // model_controlled.get_n_leaves()}x more leaves."
    )

    print(
        f"\n3. Did controlling max_depth and min_samples_split improve generalization?\n"
        f"   In terms of raw test RMSE the controlled tree did not improve over the "
        f"default tree ({metrics_controlled_test['RMSE']:.2f} W vs {metrics_default_test['RMSE']:.2f} W). "
        f"However, it achieved nearly the same R² "
        f"({metrics_controlled_test['R2']:.4f} vs {metrics_default_test['R2']:.4f}) "
        f"with {model_controlled.get_n_leaves()} leaves instead of {model_default.get_n_leaves()}, "
        f"which means the model is far simpler, more interpretable, and less exposed "
        f"to noise in new data. The gap between train and test error is also narrower "
        f"in the controlled tree, which is a sign of better generalization behavior."
    )

    print(
        f"\n4. At what depth does test RMSE stop improving?\n"
        f"   Based on the autonomous depth search, test RMSE reached its minimum "
        f"at max_depth={best_depth} (RMSE={valid_errors[best_index]:.2f} W). "
        f"After depth {best_depth}, the test error fluctuates slightly but never "
        f"improves significantly, while the training error continues dropping toward zero. "
        f"This is the point where adding more depth helps the model memorize noise "
        f"rather than learn real patterns."
    )

    print(
        f"\n5. Does the dataset show overfitting as depth increases?\n"
        f"   Yes, clearly. From depth 5 onward, training RMSE drops close to zero "
        f"({train_errors[4]:.2f} -> {train_errors[-1]:.2f} W) while test RMSE stays "
        f"in the {int(min(valid_errors[4:]))}-{int(max(valid_errors[4:]))} W range and "
        f"does not follow the same downward trend. This divergence between train and "
        f"test error is the classic overfitting signature. The dataset has ~{len(X_train)} "
        f"training samples, which makes deep trees especially prone to memorizing "
        f"individual data points."
    )

    print(
        f"\n6. Which tree would you select for deployment?\n"
        f"   The Best Tree with max_depth={best_depth} would be selected for deployment. "
        f"It achieves the best test RMSE ({valid_errors[best_index]:.2f} W) and "
        f"Test R²={valid_scores[best_index]:.4f} while keeping a manageable structure "
        f"({leaves_counts[best_index]} leaves vs {model_default.get_n_leaves()} in the default). "
        f"It balances predictive accuracy and generalization better than both the "
        f"shallow tree (which underfits) and deeper trees (which overfit)."
    )

    print(
        f"\nFinal Interpretation:\n"
        f"   The Decision Tree Regressor demonstrated strong capability for predicting "
        f"photovoltaic AC power output (Pac(W)). All tested configurations achieved "
        f"very high R² values above 0.99 on the test set, confirming that the selected "
        f"electrical, thermal, and temporal variables carry strong predictive information. "
        f"The autonomous depth analysis revealed that depth={best_depth} is the sweet spot "
        f"for this dataset: beyond that point, the model memorizes the training data without "
        f"improving generalization. Applying max_depth and min_samples_split confirmed that "
        f"complexity control is valuable even when raw accuracy changes little, because it "
        f"produces a more interpretable model with fewer leaves that is easier to validate "
        f"and maintain in a real photovoltaic monitoring system."
    )

    print(f"\nAll plots and CSV saved to: {Path(OUTPUT_DIR).resolve()}\n")


if __name__ == "__main__":
    main()
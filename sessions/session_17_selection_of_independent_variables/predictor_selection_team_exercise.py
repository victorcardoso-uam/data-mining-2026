"""
Session 17 — Regression Analysis (5.1 Selection of Independent Variables)
Part 3 — Team Exercise Dataset

Goal
----
Use this script with the dataset assigned to your team.
Your task is NOT to fit a regression model yet.
Your task is to identify and justify the independent variables.

What you must understand
------------------------
1) The dependent variable is the one you want to predict.
2) Independent variables are the variables you choose as possible predictors.
3) That choice depends on the target.
4) If the target changes, the independent variables may also change.
5) Some columns should be excluded:
   - IDs
   - names / codes
   - variables that are clearly not useful
   - variables that leak information from the future
"""

from __future__ import annotations
import os
import pandas as pd

DATA_PATH = r"C:\Users\mario\OneDrive\Escritorio\School\8to Semestre\Data Miing\data-mining-course\Repositories\data-mining-2026\sessions\session_17_selection_of_independent_variables\team04_warehouse_picking_time.csv"
TARGET_COL = "picking_time_min"

def main() -> None:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Update DATA_PATH first."
        )

    df = pd.read_csv(DATA_PATH)

    if TARGET_COL not in df.columns:
        raise ValueError(
            f"TARGET_COL '{TARGET_COL}' not found.\nAvailable columns: {list(df.columns)}"
        )

    print("\n=== DATASET PREVIEW ===")
    print(df.head())

    print("\n=== SHAPE ===")
    print(df.shape)

    print("\n=== DATA TYPES ===")
    print(df.dtypes)

    print("\n=== MISSING VALUES ===")
    print(df.isna().sum().sort_values(ascending=False))

    print("\n=== TARGET VARIABLE ===")
    print(TARGET_COL)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    candidate_cols = [c for c in df.columns if c != TARGET_COL]

    if TARGET_COL in numeric_cols:
        print("\n=== NUMERIC CORRELATIONS WITH TARGET ===")
        corr_series = df[numeric_cols].corr(numeric_only=True)[TARGET_COL].drop(labels=[TARGET_COL], errors="ignore")
        corr_sorted = corr_series.sort_values(key=abs, ascending=False)
        print(corr_sorted)
        
        # Variables with strong correlation (abs > 0.5)
        strong_vars = corr_sorted[corr_sorted.abs() > 0.5].index.tolist()
        print("\n=== VARIABLES WITH STRONG CORRELATION (abs > 0.5) ===")
        if strong_vars:
            for var in strong_vars:
                print(f"- {var}: {corr_sorted[var]:.4f}")
        else:
            print("No variables with abs correlation > 0.5")

    print("\n=== ALL POSSIBLE CANDIDATE COLUMNS ===")
    for c in candidate_cols:
        print("-", c)

    print("\n=== TEAM TASK ===")
    print("Discuss and answer these questions:")
    print("1. Which variables could be used as independent variables?")
    # The variables that could be used as independent variables are: items_to_pick, travel_distance_m, picker_experience_years, aisle_congestion_index, cart_weight_kg.
    # These are numeric variables that could potentially influence the picking time.
    print("2. Which variables should be excluded?")
    # "warehouse_order_code and shift columns should be excluded because they are identifiers that do not provide predictive information about the target variable. They are unique to each order and shift, and thus do not have a meaningful relationship with the picking time."
    print("3. Why do your selected variables make sense for predicting the target?")
    # The selected variables make sense because they directly relate to the factors affecting picking time: 
    # - items_to_pick: More items likely require more time.
    # - travel_distance_m: Longer distances mean more time spent traveling.
    # - picker_experience_years: More experienced pickers might be faster.
    # - aisle_congestion_index: Higher congestion could slow down picking.
    # - cart_weight_kg: Heavier carts might be harder to maneuver, increasing time.
    print("4. Would your answer change if the target variable were different?")
    # Yes, if the target were something like "picker fatigue" or "order accuracy", different variables might be more relevant, such as shift timing or picker experience, and some current variables like travel distance might be less important.

if __name__ == "__main__":
    main()

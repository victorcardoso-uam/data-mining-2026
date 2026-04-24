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

DATA_PATH = "team04_warehouse_picking_time.csv"
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
    print("2. Which variables should be excluded?")
    print("3. Why do your selected variables make sense for predicting the target?")
    print("4. Would your answer change if the target variable were different?")

    print("\n=== TEAM 04 ANSWERS ===")
    print("\n1. Which variables could be used as independent variables?")
    print("   Answer: items_to_pick, travel_distance_m, picker_experience_years, shift, cart_weight_kg")
    print("   These are all predictive factors that could influence warehouse picking time.")
    
    print("\n2. Which variables should be excluded?")
    print("   Answer: warehouse_order_code (identifier), aisle_congestion_index (weak correlation)")
    print("   - warehouse_order_code: This is a unique identifier, not a predictor.")
    print("   - aisle_congestion_index: Has very weak correlation (0.198) and minimal predictive value.")
    
    print("\n3. Why do your selected variables make sense for predicting the target?")
    print("   Answer:")
    print("   - items_to_pick (0.783 corr): More items = more time needed. Strong logical connection.")
    print("   - travel_distance_m (0.344 corr): Longer distances require more walking/travel time.")
    print("   - picker_experience_years (-0.245 corr): Experienced pickers are more efficient.")
    print("   - shift (categorical): Different shifts may have different productivity patterns.")
    print("   - cart_weight_kg (0.017 corr): Weak but included; heavy carts might slow picking slightly.")
    
    print("\n4. Would your answer change if the target variable were different?")
    print("   Answer: YES, definitely.")
    print("   - If predicting 'picker_accuracy': We'd focus on shift, experience, and congestion.")
    print("   - If predicting 'picker_safety': We'd include cart_weight, travel_distance, experience.")
    print("   - If predicting 'items_damaged': Different variables would be important.")
    print("   - The target defines which variables are relevant.")
    
    print("\n=== FINAL INDEPENDENT VARIABLES RECOMMENDATION ===")
    print("Use these variables for regression modeling:")
    print("  PRIMARY: items_to_pick (strongest predictor)")
    print("  SECONDARY: travel_distance_m (moderate predictor)")
    print("  OPTIONAL: picker_experience_years, shift (weak but contextually useful)")
    print("  NOT RECOMMENDED: cart_weight_kg, warehouse_order_code, aisle_congestion_index")

if __name__ == "__main__":
    main()

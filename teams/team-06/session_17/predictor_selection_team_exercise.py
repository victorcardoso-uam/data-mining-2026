"""
Session 17 — Regression Analysis (5.1 Selection of Independent Variables)
Part 3 — Team Exercise Dataset

Goal
----
Use this script with the dataset assigned to your team.
Your task is NOT to fit a regression model yet.
Your task is to identify and justify the independent variables.
"""

from __future__ import annotations
import os
import pandas as pd

# Path configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "team06_wind_turbine_output.csv")
TARGET_COL = "power_output_kw"

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

    # Step 2 & 5: Defining Predictors and Exclusions
    exclude_cols = ["turbine_code", "region", TARGET_COL]
    candidate_cols = [
        "wind_speed_ms", 
        "air_density_kgm3", 
        "rotor_diameter_m", 
        "blade_pitch_deg", 
        "turbine_age_years"
    ]

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

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

    # === TEAM 06 FINAL ANALYSIS (ENGLISH) ===
    print("\n" + "="*45)
    print("         TEAM 06 - FINAL ANALYSIS")
    print("="*45)
    
    print(f"\n1. Independent Variables (Predictors):")
    print(f"   {candidate_cols}")
    
    print(f"\n2. Excluded Variables (Per Rule 5):")
    print(f"   {exclude_cols}")
    print("   Note: IDs and names were removed as they lack mathematical significance.")
    
    print("\n3. Why these variables make sense:")
    print("   The selection is based on physical laws of energy. Wind speed,")
    print("   air density, and rotor diameter are the primary drivers of")
    print("   kinetic energy conversion in a wind turbine.")
    
    print("\n4. Impact of changing the Target:")
    print("   If the target were 'turbine_age_years', the logic would reverse.")
    print("   'power_output_kw' would become an independent variable because")
    print("   a decrease in efficiency serves as a predictor for machine age.")
    print("="*45)

if __name__ == "__main__":
    main()
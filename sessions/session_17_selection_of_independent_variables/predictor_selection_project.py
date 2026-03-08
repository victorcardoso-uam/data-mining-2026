"""
Session 17 — Regression Analysis (5.1 Selection of Independent Variables)
Part 2 — Apply variable selection thinking to YOUR FINAL PROJECT DATASET

This script does NOT automatically decide the "correct" predictors for you.
It helps your team think about variable selection in a structured way.

Main ideas
----------
1) The dependent variable (target) is the variable you want to predict.
2) Independent variables (predictors/features) are the variables you measure and use to explain or predict the target.
3) The "best" independent variables depend on the prediction goal.
4) Not every available column should be used.
5) A good predictor should make sense:
   - statistically
   - logically
   - from the domain / engineering context

Instructions
------------
1) Update DATA_PATH and TARGET_COL below
2) Run:
   python predictor_selection_project.py
3) Read the output carefully
4) Decide which variables should be used as independent variables
5) Add your explanation in comments or in the deliverable requested by your professor
"""

from __future__ import annotations
import os
from typing import List
import pandas as pd

DATA_PATH = "PATH/TO/YOUR/PROJECT_DATASET.csv"
TARGET_COL = "YOUR_TARGET_COLUMN"
MANUAL_EXCLUDE = []

def classify_columns(df: pd.DataFrame, target_col: str):
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]
    candidate_cols = [c for c in df.columns if c != target_col]
    return numeric_cols, categorical_cols, candidate_cols

def likely_identifier(col_name: str) -> bool:
    col = col_name.lower()
    keywords = ["id", "name", "code", "folio", "record", "index", "comment", "description"]
    return any(k in col for k in keywords)

def main() -> None:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Update DATA_PATH before running the script."
        )

    df = pd.read_csv(DATA_PATH)

    if TARGET_COL not in df.columns:
        raise ValueError(
            f"TARGET_COL '{TARGET_COL}' not found.\nAvailable columns: {list(df.columns)}"
        )

    print("\n=== DATASET OVERVIEW ===")
    print("Shape:", df.shape)

    print("\n=== COLUMNS ===")
    for c in df.columns:
        print("-", c)

    print("\n=== DATA TYPES ===")
    print(df.dtypes)

    print("\n=== MISSING VALUES ===")
    print(df.isna().sum().sort_values(ascending=False))

    numeric_cols, categorical_cols, candidate_cols = classify_columns(df, TARGET_COL)

    print("\n=== TARGET VARIABLE ===")
    print("Target:", TARGET_COL)
    print("Target dtype:", df[TARGET_COL].dtype)

    print("\n=== NUMERIC COLUMNS ===")
    print(numeric_cols)

    print("\n=== CATEGORICAL / OTHER COLUMNS ===")
    print(categorical_cols)

    if pd.api.types.is_numeric_dtype(df[TARGET_COL]):
        print("\n=== NUMERIC CORRELATION WITH TARGET ===")
        corr_series = df[numeric_cols].corr(numeric_only=True)[TARGET_COL].drop(labels=[TARGET_COL], errors="ignore")
        corr_df = corr_series.abs().sort_values(ascending=False).to_frame(name="abs_correlation")
        corr_df["signed_correlation"] = corr_series.loc[corr_df.index]
        print(corr_df)

        print("\nInterpretation note:")
        print("- Higher absolute correlation means stronger linear relationship.")
        print("- But correlation alone does NOT guarantee a variable should be used.")
        print("- Domain meaning still matters.")
    else:
        print("\nTarget is not numeric. Correlation with target is not shown.")

    suggested_exclude: List[str] = []
    for col in candidate_cols:
        if col in MANUAL_EXCLUDE:
            suggested_exclude.append(col)
            continue
        if likely_identifier(col):
            suggested_exclude.append(col)
            continue
        if df[col].nunique(dropna=True) == len(df):
            suggested_exclude.append(col)

    suggested_exclude = sorted(list(set(suggested_exclude)))

    print("\n=== SUGGESTED EXCLUSIONS ===")
    if suggested_exclude:
        for col in suggested_exclude:
            print("-", col)
    else:
        print("No obvious exclusions detected automatically.")

    candidate_predictors = [c for c in candidate_cols if c not in suggested_exclude]
    print("\n=== CANDIDATE INDEPENDENT VARIABLES ===")
    for col in candidate_predictors:
        print("-", col)

    print("\n=== TEAM DECISION QUESTIONS ===")
    questions = [
        "1. Which variable are you trying to predict?",
        "2. Which columns are true candidate predictors?",
        "3. Which columns should be excluded, and why?",
        "4. Which predictors make engineering or business sense?",
        "5. Are any variables redundant or too similar to each other?",
        "6. Are there variables that should NOT be used because they leak target information?",
        "7. If you changed the target variable, would your independent variables also change?"
    ]
    for q in questions:
        print(q)

    print("\n=== IMPORTANT CONCEPTUAL MESSAGE ===")
    print("Independent variables are NOT universal.")
    print("They depend on the prediction goal.")
    print("If you change the target, the best independent variables may also change.")

if __name__ == "__main__":
    main()

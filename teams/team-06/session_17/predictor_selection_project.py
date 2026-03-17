<<<<<<< HEAD
import os
import pandas as pd

# Configuración de rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Usamos el nombre exacto de tu archivo según la captura
DATA_PATH = os.path.join(BASE_DIR, "amazon.csv") 
TARGET_COL = "rating"

def main() -> None:
    if not os.path.exists(DATA_PATH):
        print(f"Error: No se encontró el archivo en {DATA_PATH}")
        print("Asegúrate de que el archivo se llame amazon.csv y esté en la misma carpeta.")
        return

    # 1. Cargar datos
    df = pd.read_csv(DATA_PATH)

    # 2. LIMPIEZA DE DATOS (Interna, no crea archivos nuevos)
    # Quitamos el símbolo ₹ y las comas para que Python pueda hacer cálculos
    cols_to_fix = ['discounted_price', 'actual_price', 'rating_count']
    for col in cols_to_fix:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('₹', '').str.replace(',', '')
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Limpieza de la columna rating (el target)
    if TARGET_COL in df.columns:
        df[TARGET_COL] = pd.to_numeric(df[TARGET_COL].astype(str).replace('|', '0'), errors='coerce')
    
    # Borramos filas que hayan quedado vacías tras la limpieza
    df = df.dropna(subset=[TARGET_COL] + [c for c in cols_to_fix if c in df.columns])

    print("\n=== DATASET OVERVIEW (AMAZON) ===")
    print(f"Total rows after cleaning: {df.shape[0]}")
    print(f"Target: {TARGET_COL}")

    # 3. Selección de Variables
    exclude_cols = ["product_id", "product_name", "category", "user_id", "review_id"]
    candidate_cols = ["discounted_price", "actual_price", "rating_count"]

    # 4. Mostrar Correlación
    print("\n=== NUMERIC CORRELATION WITH RATING ===")
    correlations = df[candidate_cols + [TARGET_COL]].corr()[TARGET_COL].drop(TARGET_COL)
    print(correlations.sort_values(ascending=False))

    # --- TEAM 06 FINAL ANALYSIS (ENGLISH) ---
    print("\n" + "="*50)
    print("         TEAM 06 - AMAZON ANALYSIS")
    print("="*50)
    
    print(f"\n1. Target Variable: {TARGET_COL}")
    print(f"2. Independent Variables: {candidate_cols}")
    print(f"3. Excluded Variables: IDs, Names, and URLs (Rule 5).")
    
    print("\n4. Logic:")
    print("   We selected prices and rating count because they represent")
    print("   the financial value and the popularity of the product,")
    print("   which are logical predictors for the overall rating.")
    
    print("\n5. Redundancy:")
    print("   'discounted_price' and 'actual_price' show high collinearity.")
    print("   In the final model, we should choose only one to avoid noise.")
    
    print("\n6. Direction of Causality:")
    print("   If the target changed to 'discounted_price', the 'rating'")
    print("   would then become an independent variable to help predict price.")
    print("="*50)

if __name__ == "__main__":
    main()
=======
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "amazon.csv")
# The target column must exist in the dataset. Updated from 'product_rating' to
# an actual column name observed in the CSV file. Adjust as needed.
TARGET_COL = "rating"
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

    # read dataset and coerce fully-numeric columns to proper number dtype.
    # pandas 3.x no longer accepts errors='ignore', so we try coercion and only
    # assign the result when no values are lost.
    df = pd.read_csv(DATA_PATH)
    # try to coerce columns with numeric-like content (e.g. prices with ₹, commas,
    # percentages) into numbers. We strip out non-numeric characters and
    # convert; only assign the numeric series if most values parse successfully.
    for col in df.columns:
        # pandas 3.x often uses StringDtype (shown as 'str'), so detect any
        # string-like column rather than just object dtype.
        if pd.api.types.is_string_dtype(df[col]):
            # remove anything except digits, decimal point, or minus sign
            cleaned = df[col].astype(str).str.replace(r"[^0-9.\-]+", "", regex=True)
            converted = pd.to_numeric(cleaned, errors="coerce")
            nonnull_ratio = converted.notna().mean()
            if nonnull_ratio > 0.8:
                df[col] = converted

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

    # Store correlation info for later use
    strong_correlation_vars = []
    
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
        
        # Variables with strong correlation (abs > 0.5)
        strong_correlation_vars = corr_df[corr_df["abs_correlation"] > 0.5].index.tolist()
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

    # build a more sensible predictor list:
    #  * always include numeric columns unless explicitly excluded
    #  * include categorical cols only if low-cardinality (<=10)
    #  * never include suggested exclusions unless they have strong correlation
    candidate_predictors: List[str] = []
    for c in candidate_cols:
        if c in suggested_exclude and c not in strong_correlation_vars:
            continue
        if c in numeric_cols:
            candidate_predictors.append(c)
        else:
            # for non-numeric variables, require low cardinality to be a candidate
            nuniq = df[c].nunique(dropna=True)
            if nuniq <= 10:
                candidate_predictors.append(c)
            elif c in strong_correlation_vars:
                candidate_predictors.append(c)

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
>>>>>>> 0acf51b20749251753b048e28d808f654ee3e298

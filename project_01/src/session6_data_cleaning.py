
"""
Session 2.4 - Data Cleaning
Course: Data Mining

This script applies basic data cleaning techniques
to the Seattle Streets dataset.

Steps covered:
- Handling missing values
- Removing duplicates
- Fixing basic integrity issues
- Saving a cleaned dataset
"""

import pandas as pd

# Load raw dataset
df = pd.read_csv("data/raw/Seattle_Streets_1_-5073353257610679043.csv")

print("=== ORIGINAL DATASET ===")
print("Original shape (rows, columns):", df.shape)

# -----------------------------
# 1. HANDLE MISSING VALUES
# -----------------------------

# Numerical columns: fill missing values with median
numeric_cols = df.select_dtypes(include="number").columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Categorical columns: fill missing values with 'Unknown'
categorical_cols = df.select_dtypes(include="object").columns
df[categorical_cols] = df[categorical_cols].fillna("Unknown")

print("\nMissing values handled.")

# -----------------------------
# 2. REMOVE DUPLICATES
# -----------------------------
duplicates_before = df.duplicated().sum()
df = df.drop_duplicates()
duplicates_after = df.duplicated().sum()

print(f"Duplicates before: {duplicates_before}")
print(f"Duplicates after: {duplicates_after}")
print("Shape after removing duplicates:", df.shape)

# -----------------------------
# 3. BASIC INTEGRITY FIXES
# -----------------------------

# Speed limit cannot be negative
df.loc[df["SPEEDLIMIT"] < 0, "SPEEDLIMIT"] = 0

# Segment length cannot be negative
df.loc[df["SEGLENGTH"] < 0, "SEGLENGTH"] = 0

# Surface width cannot be negative
df.loc[df["SURFACEWIDTH"] < 0, "SURFACEWIDTH"] = 0

# Pavement condition index should be between 0 and 100
df["PVMTCONDINDX1"] = df["PVMTCONDINDX1"].clip(lower=0, upper=100)
if "PVMTCONDINDX2" in df.columns:
    df["PVMTCONDINDX2"] = df["PVMTCONDINDX2"].clip(lower=0, upper=100)

print("Basic integrity rules applied.")

# -----------------------------
# 4. SAVE CLEANED DATASET
# -----------------------------
df.to_csv("data/processed/seattle_streets_cleaned.csv", index=False)

print("\nCleaned dataset saved at: data/processed/seattle_streets_cleaned.csv")

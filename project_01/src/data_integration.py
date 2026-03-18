"""
Data Integration Script
Course: Data Mining

This script integrates multiple datasets:
- customers.csv
- transactions.csv
- seattle_streets_cleaned.csv

Steps covered:
- Load all datasets
- Merge customers with transactions on customer_id
- Perform data validation
- Save integrated dataset
"""

import pandas as pd
import numpy as np

print("=" * 60)
print("DATA INTEGRATION - MERGING DATASETS")
print("=" * 60)

# ========================================
# 1. LOAD ALL DATASETS
# ========================================
print("\n[1] Loading datasets...")

customers = pd.read_csv("data/processed/customers.csv")
transactions = pd.read_csv("data/processed/transactions.csv")
streets = pd.read_csv("data/processed/seattle_streets_cleaned.csv")

print(f"✓ Customers loaded: {customers.shape}")
print(f"✓ Transactions loaded: {transactions.shape}")
print(f"✓ Seattle Streets loaded: {streets.shape}")

# ========================================
# 2. MERGE CUSTOMERS WITH TRANSACTIONS
# ========================================
print("\n[2] Merging customers with transactions...")

# Merge on customer_id
integrated = pd.merge(
    transactions,
    customers,
    on="customer_id",
    how="left"
)

print(f"✓ Merged dataset shape: {integrated.shape}")
print(f"✓ Columns in integrated dataset: {integrated.shape[1]}")

# ========================================
# 3. DATA VALIDATION
# ========================================
print("\n[3] Validating integrated data...")

# Check for missing values
missing_values = integrated.isnull().sum()
if missing_values.sum() > 0:
    print("⚠ Missing values detected:")
    print(missing_values[missing_values > 0])
else:
    print("✓ No missing values detected")

# Check data types
print("\nData types:")
print(integrated.dtypes)

# Basic statistics
print(f"\n✓ Transaction count: {len(integrated)}")
print(f"✓ Unique customers: {integrated['customer_id'].nunique()}")
print(f"✓ Date range: {integrated['transaction_date'].min()} to {integrated['transaction_date'].max()}")
print(f"✓ Product categories: {integrated['product_category'].nunique()}")

# ========================================
# 4. SUMMARY STATISTICS
# ========================================
print("\n[4] Integrated Dataset Summary:")
print(f"{'Metric':<30} {'Value':<20}")
print("-" * 50)
print(f"{'Total transactions':<30} {len(integrated):<20}")
print(f"{'Total customers':<30} {integrated['customer_id'].nunique():<20}")
print(f"{'Date range':<30} {len(integrated['transaction_date'].unique())} days")
print(f"{'Regions':<30} {integrated['region'].nunique():<20}")
print(f"{'Product categories':<30} {integrated['product_category'].nunique():<20}")

# Revenue analysis
integrated['total_amount'] = integrated['quantity'] * integrated['unit_price']
print(f"{'Total revenue':<30} ${integrated['total_amount'].sum():,.2f}")

# ========================================
# 5. SAVE INTEGRATED DATASET
# ========================================
print("\n[5] Saving integrated dataset...")

output_path = "data/processed/customers_transactions_integrated.csv"
integrated.to_csv(output_path, index=False)

print(f"✓ Integrated dataset saved: {output_path}")

# ========================================
# 6. DISPLAY SAMPLE
# ========================================
print("\n[6] Sample of integrated data (first 5 rows):")
print(integrated.head())

print("\n" + "=" * 60)
print("DATA INTEGRATION COMPLETED SUCCESSFULLY")
print("=" * 60)

"""
Session 17 — Regression Analysis (5.1 Selection of Independent Variables)
Final Project — Amazon Dataset Selection

Goal: Identify and justify the independent variables to predict 'rating'.
"""

from __future__ import annotations
import os
import pandas as pd

# Path configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Note: I created amazon_cleaned.csv to handle currency symbols and commas
DATA_PATH = os.path.join(BASE_DIR, "amazon_cleaned.csv")
TARGET_COL = "rating"

def main() -> None:
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found. Cleaning the data first...")
        # (Internal cleaning logic already performed)
        return

    df = pd.read_csv(DATA_PATH)

    print("\n=== DATASET OVERVIEW (AMAZON) ===")
    print(f"Shape: {df.shape}")
    print(f"Target Variable: {TARGET_COL}")

    # Defining variables for the project
    # We exclude IDs, names, URLs and long text descriptions
    exclude_cols = [
        "product_id", "product_name", "category", "about_product", 
        "user_id", "user_name", "review_id", "review_title", 
        "review_content", "img_link", "product_link", "discount_percentage"
    ]
    
    candidate_cols = ["discounted_price", "actual_price", "rating_count"]

    # Show correlation with Target
    print("\n=== NUMERIC CORRELATION WITH RATING ===")
    corr_series = df[candidate_cols + [TARGET_COL]].corr()[TARGET_COL].drop(TARGET_COL)
    print(corr_series.sort_values(ascending=False))

    # --- TEAM DECISION & ANALYSIS ---
    print("\n" + "="*50)
    print("         TEAM ANALYSIS - AMAZON PROJECT")
    print("="*50)
    
    print("\n1. Variable to predict (Dependent):")
    print(f"   - {TARGET_COL}")
    
    print("\n2. Candidate Predictors (Independent):")
    print(f"   - {candidate_cols}")
    
    print("\n3. Excluded Columns and Why:")
    print("   - Excluded IDs, names, and URLs (Rule 5): Non-numeric identifiers.")
    print("   - Excluded descriptions and reviews: Unstructured text data.")
    print("   - Excluded discount_percentage: Redundant with prices.")
    
    print("\n4. Business/Engineering Sense:")
    print("   - Prices and popularity (rating_count) logically influence customer")
    print("     satisfaction and perceived value of the product.")
    
    print("\n5. Redundancy Check:")
    print("   - 'discounted_price' and 'actual_price' are highly similar.")
    print("     We should likely pick only one to avoid collinearity.")
    
    print("\n6. Target Information Leakage:")
    print("   - 'review_title' and 'review_content' were excluded because they")
    print("     contain the sentiment that directly forms the rating (leakage).")
    
    print("\n7. Change of Target Analysis:")
    print("   - If we predicted 'actual_price', then 'discounted_price' would")
    print("     be the main predictor. The choice depends on the goal.")
    print("="*50)

if __name__ == "__main__":
    main()
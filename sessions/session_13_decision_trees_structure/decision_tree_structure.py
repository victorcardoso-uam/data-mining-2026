"""
Session 13 — Decision Trees (4.1 Structure)
Starter file: decision_tree_structure.py

Goal:
- Train and visualize a decision tree classifier
- Understand how splits represent decision logic
- Interpret the root split using an engineering perspective

Dataset:
- datasets/session_13/wind_turbine_operations.csv
"""

from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

DATA_PATH = Path(__file__).resolve().parents[2] / "datasets" / "session_13" / "wind_turbine_operations.csv"

def main():
    # 1) Load dataset
    df = pd.read_csv(DATA_PATH)

    # 2) Quick inspection
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print(df.head(3))

    # 3) Create target variable (binary classification)
    # Adjust the threshold if needed after looking at df["power_kw"].describe()
    threshold_kw = 1200
    df["high_output"] = (df["power_kw"] > threshold_kw).astype(int)

    # 4) Select features (keep it simple for beginners)
    feature_cols = ["wind_speed_ms", "ambient_temp_c", "gearbox_temp_c"]

    # 5) Handle missing values (simple strategy for this session)
    X = df[feature_cols].copy()
    y = df["high_output"].copy()

    # Fill missing numeric values with median
    X = X.fillna(X.median(numeric_only=True))

    # 6) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # 7) Train decision tree (keep it shallow so it stays interpretable)
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    # 8) Quick accuracy (not the focus today, but helpful feedback)
    acc = model.score(X_test, y_test)
    print(f"Test accuracy: {acc:.3f}")

    # 9) Visualize tree
    plt.figure(figsize=(14, 7))
    plot_tree(
        model,
        feature_names=feature_cols,
        class_names=["low_output", "high_output"],
        filled=True,
        rounded=True,
        fontsize=9
    )
    plt.title("Decision Tree (max_depth=3)")
    plt.tight_layout()
    plt.show()

    # 10) Interpretation prompts (students answer in comments in their deliverable)
    print("\nINTERPRETATION PROMPTS (answer in comments in your deliverable):")
    print("- What is the first split (root node) based on?")
    print("- What threshold is used in that split?")
    print("- What does that decision mean physically for the turbine?")
    print("- Provide one example of a decision path from root to a leaf.")

if __name__ == "__main__":
    main()

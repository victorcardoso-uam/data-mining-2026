import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error


def main():
    # 1. Load dataset
    df = pd.read_csv("industrial_decision_tree_exam.csv")

    # 2. Display info
    print("Shape:", df.shape)
    print("Columns:", df.columns)
    print("Data types:\n", df.dtypes)

    # 3. Define X and y
    y = df["daily_output_units"]
    X = df.drop("daily_output_units", axis=1)

    # 4. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 5. Train model
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)

    # 6. Evaluate
    y_pred = model.predict(X_test)
    print("MSE:", mean_squared_error(y_test, y_pred))


if __name__ == "__main__":
    # Run original model
    print("\n--- Original Decision Tree ---")
    main()

    # Task 2: Tree Pruning / Complexity Control
    # Load data again for fair comparison
    df = pd.read_csv("industrial_decision_tree_exam.csv")
    y = df["daily_output_units"]
    X = df.drop("daily_output_units", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Pruned model: limit tree depth
    pruned_model = DecisionTreeRegressor(max_depth=3, random_state=42)
    pruned_model.fit(X_train, y_train)
    y_pred_pruned = pruned_model.predict(X_test)
    pruned_mse = mean_squared_error(y_test, y_pred_pruned)

    print("\n--- Pruned Decision Tree (max_depth=3) ---")
    print("MSE:", pruned_mse)

    # Comparison
    # Lower MSE is better. Pruning (max_depth=3) usually increases bias but reduces variance, which can help generalization if the original tree overfits.
    print("\n--- Comparison & Explanation ---")
    print("Original MSE is from a fully grown tree (likely overfits). Pruned tree (max_depth=3) may have higher MSE if underfits, but can generalize better on new data. Compare both MSEs to see the effect.")
    print("If pruned MSE is lower or similar, pruning helped prevent overfitting. If much higher, the tree may be too simple.")

    # Short explanation (for deliverable):
    # Pruning a decision tree by setting max_depth limits its complexity, reducing overfitting risk. This can improve test set performance if the original tree overfits, but too much pruning can cause underfitting. Always compare metrics to choose the best model.
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)


def load_data(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def evaluate_model(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    metrics = {
        "Model": name,
        "R2": r2,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
    }
    print(f"\n{name}")
    print("R2:\t", round(r2, 4))
    print("MAE:\t", round(mae, 4))
    print("MSE:\t", round(mse, 4))
    print("RMSE:\t", round(rmse, 4))
    return metrics


def main() -> None:
    data_path = Path(__file__).parent / "industrial_regression_ann_exam.csv"
    df = load_data(data_path)

    print("Loaded dataset:", data_path)
    print("Shape:", df.shape)
    print(df.head(5).to_string(index=False))

    target_column = "production_efficiency_score"
    X = df.drop(columns=[target_column])
    y = df[target_column].values

    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    print(f"\nTraining samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    print("\nTask 1 - Linear Regression")
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    y_pred_linear = linear_model.predict(X_test)
    linear_metrics = evaluate_model("Linear Regression", y_test, y_pred_linear)

    print("\nTask 2 - Polynomial Regression")
    polynomial_pipeline = Pipeline(
        [
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("linear", LinearRegression()),
        ]
    )
    polynomial_pipeline.fit(X_train, y_train)
    y_pred_poly = polynomial_pipeline.predict(X_test)
    poly_metrics = evaluate_model("Polynomial Regression (degree=2)", y_test, y_pred_poly)

    print("\nPolynomial regression is more appropriate when the relationship between inputs and target is nonlinear and there are interaction or curvature effects that a straight line cannot capture.")

    print("\nTask 3 - Neural Network Model")
    ann_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "ann",
                MLPRegressor(
                    hidden_layer_sizes=(50, 25),
                    activation="tanh",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )
    ann_pipeline.fit(X_train, y_train)
    y_pred_ann = ann_pipeline.predict(X_test)
    ann_metrics = evaluate_model("ANN Regression", y_test, y_pred_ann)

    print("\nTask 4 - Model Comparison")
    results = [linear_metrics, poly_metrics, ann_metrics]
    sorted_results = sorted(results, key=lambda x: x["RMSE"])

    print("\nSummary of results sorted by RMSE:")
    for row in sorted_results:
        print(f"{row['Model']}: R2={row['R2']:.4f}, RMSE={row['RMSE']:.4f}, MAE={row['MAE']:.4f}")

    best_model = sorted_results[0]
    print(f"\nBest performing model: {best_model['Model']}")
    print(
        "The best model is selected by lowest RMSE and highest predictive performance on the test set."
    )
    print(
        "A neural network can outperform linear models when the underlying patterns are nonlinear and there are interactions across features, while polynomial regression is useful when a moderate nonlinear relationship exists but model complexity remains manageable."
    )

    print(
        "\nThis script uses linear regression as a base model that assumes a straight-line relationship between the input variables and the target. "
        "It then uses polynomial regression to capture nonlinear curvature and interaction effects by expanding the feature set. "
        "Finally, it trains an artificial neural network with scaled inputs so the model can learn more complex nonlinear patterns through hidden layers. "
        "The evaluation metrics compare how well each approach balances fit and generalization on unseen data, and the chosen best model is the one that minimizes prediction error while avoiding overfitting."
    )


if __name__ == "__main__":
    main()

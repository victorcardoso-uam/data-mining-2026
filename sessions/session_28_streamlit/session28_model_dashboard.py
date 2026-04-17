"""
Session 28 — Model Implementation Workshop
Streamlit app for comparing predictive models as reusable modules

How to run:
    streamlit run session28_model_dashboard.py

Recommended dataset:
    session28_industrial_model_comparison.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor


st.set_page_config(page_title="Model Implementation Workshop", layout="wide")
st.title("Model Implementation Workshop")
st.write(
    "This dashboard treats predictive models as reusable modules. "
    "Load one dataset, choose the target variable, train several models, "
    "and compare which one adapts better to the data."
)

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

@st.cache_data
def load_data(file) -> pd.DataFrame:
    return pd.read_csv(file)

def split_columns(df: pd.DataFrame, target_col: str):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    return X, y, numeric_cols, categorical_cols

def build_preprocessor(numeric_cols, categorical_cols, scale_numeric=False):
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    numeric_transformer = Pipeline(steps=num_steps)

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
    }

def run_linear_regression(X_train, X_test, y_train, y_test, numeric_cols, categorical_cols):
    preprocessor = build_preprocessor(numeric_cols, categorical_cols, scale_numeric=False)
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression())
        ]
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return preds, evaluate_model(y_test, preds)

def run_polynomial_regression(X_train, X_test, y_train, y_test, numeric_cols, categorical_cols, degree: int):
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression())
        ]
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return preds, evaluate_model(y_test, preds)

def run_decision_tree(X_train, X_test, y_train, y_test, numeric_cols, categorical_cols, max_depth):
    preprocessor = build_preprocessor(numeric_cols, categorical_cols, scale_numeric=False)
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", DecisionTreeRegressor(max_depth=max_depth, random_state=42))
        ]
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return preds, evaluate_model(y_test, preds)

def run_ann(X_train, X_test, y_train, y_test, numeric_cols, categorical_cols, hidden_layers, activation, max_iter):
    preprocessor = build_preprocessor(numeric_cols, categorical_cols, scale_numeric=True)
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", MLPRegressor(
                hidden_layer_sizes=hidden_layers,
                activation=activation,
                solver="adam",
                max_iter=max_iter,
                random_state=42
            ))
        ]
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return preds, evaluate_model(y_test, preds)

if uploaded_file is not None:
    data = load_data(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    target_col = st.selectbox("Select the target variable", data.columns)

    test_size = st.slider("Test size", min_value=0.10, max_value=0.40, value=0.20, step=0.05)
    poly_degree = st.slider("Polynomial degree", min_value=2, max_value=4, value=2, step=1)
    tree_depth = st.slider("Decision Tree max_depth", min_value=2, max_value=12, value=5, step=1)

    ann_options = {
        "(10,)": (10,),
        "(20,)": (20,),
        "(10, 10)": (10, 10),
        "(20, 10)": (20, 10),
    }
    ann_hidden_label = st.selectbox("ANN hidden_layer_sizes", list(ann_options.keys()), index=2)
    ann_hidden = ann_options[ann_hidden_label]
    ann_activation = st.selectbox("ANN activation", ["relu", "tanh", "logistic"], index=0)
    ann_max_iter = st.slider("ANN max_iter", min_value=200, max_value=1000, value=500, step=100)

    if st.button("Run Model Comparison"):
        X, y, numeric_cols, categorical_cols = split_columns(data, target_col)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        results = []
        predictions_dict = {}

        try:
            preds_lr, metrics_lr = run_linear_regression(X_train, X_test, y_train, y_test, numeric_cols, categorical_cols)
            model_name = "Linear Regression"
            results.append({"Model": model_name, **metrics_lr})
            predictions_dict[model_name] = preds_lr
        except Exception as e:
            st.warning(f"Linear Regression failed: {e}")

        try:
            model_name = f"Polynomial Regression (deg={poly_degree})"
            preds_poly, metrics_poly = run_polynomial_regression(
                X_train, X_test, y_train, y_test, numeric_cols, categorical_cols, degree=poly_degree
            )
            results.append({"Model": model_name, **metrics_poly})
            predictions_dict[model_name] = preds_poly
        except Exception as e:
            st.warning(f"Polynomial Regression failed: {e}")

        try:
            model_name = f"Decision Tree (depth={tree_depth})"
            preds_tree, metrics_tree = run_decision_tree(
                X_train, X_test, y_train, y_test, numeric_cols, categorical_cols, max_depth=tree_depth
            )
            results.append({"Model": model_name, **metrics_tree})
            predictions_dict[model_name] = preds_tree
        except Exception as e:
            st.warning(f"Decision Tree failed: {e}")

        try:
            model_name = f"ANN {ann_hidden} / {ann_activation}"
            preds_ann, metrics_ann = run_ann(
                X_train, X_test, y_train, y_test, numeric_cols, categorical_cols,
                hidden_layers=ann_hidden, activation=ann_activation, max_iter=ann_max_iter
            )
            results.append({"Model": model_name, **metrics_ann})
            predictions_dict[model_name] = preds_ann
        except Exception as e:
            st.warning(f"ANN failed: {e}")

        if results:
            results_df = pd.DataFrame(results).sort_values(by="R2", ascending=False).reset_index(drop=True)

            st.subheader("Model Comparison Table")
            st.dataframe(results_df.style.format({"R2": "{:.4f}", "MAE": "{:.4f}", "MSE": "{:.4f}", "RMSE": "{:.4f}"}))

            best_model = results_df.iloc[0]["Model"]
            st.success(f"Best model for this dataset: {best_model}")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("R² by Model")
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar(results_df["Model"], results_df["R2"])
                ax.set_ylabel("R²")
                plt.xticks(rotation=25, ha="right")
                plt.tight_layout()
                st.pyplot(fig)

            with col2:
                st.subheader("RMSE by Model")
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar(results_df["Model"], results_df["RMSE"])
                ax.set_ylabel("RMSE")
                plt.xticks(rotation=25, ha="right")
                plt.tight_layout()
                st.pyplot(fig)

            selected_model = st.selectbox("Select one model for detailed diagnostics", results_df["Model"].tolist())
            y_pred_selected = predictions_dict[selected_model]

            st.subheader("Predicted vs Actual")
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(y_test, y_pred_selected, alpha=0.7)
            min_val = min(np.min(y_test), np.min(y_pred_selected))
            max_val = max(np.max(y_test), np.max(y_pred_selected))
            ax.plot([min_val, max_val], [min_val, max_val], linestyle="--")
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title(selected_model)
            plt.tight_layout()
            st.pyplot(fig)

            st.subheader("Residual Plot")
            residuals = y_test - y_pred_selected
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(y_pred_selected, residuals, alpha=0.7)
            ax.axhline(0, linestyle="--")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Residuals")
            ax.set_title(selected_model)
            plt.tight_layout()
            st.pyplot(fig)

            st.info(
                "Interpretation guide: A better model usually has higher R², lower error metrics, "
                "points closer to the diagonal in Predicted vs Actual, and residuals randomly distributed around zero."
            )
else:
    st.info("Upload a CSV file to begin. You can use the example dataset provided with this session.")

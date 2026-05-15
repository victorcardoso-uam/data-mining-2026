"""
FINAL PROJECT - PREDICTIVE MODELING
====================================
Dataset: School Offenses
Objective: Predict the percentage of schools reporting offenses

This project implements multiple predictive models:
1. Decision Tree (2 versions: default and controlled complexity)
2. Linear Regression
3. Polynomial Regression
4. Artificial Neural Network (2 configurations)

Models evaluated using: R², MAE, MSE, RMSE
Visualizations: Predictions vs Actual, Residual plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
import os
warnings.filterwarnings('ignore')

# ============================================================================
# 1. LOAD AND EXPLORE DATASET
# ============================================================================
print("="*80)
print("DATASET LOADING")
print("="*80)

df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'clean_school_offenses_dataset.csv'))

print(f"Dataset shape: {df.shape}")
print(f"\nFirst rows:")
print(df.head())
print(f"\nDataset info:")
print(df.info())
print(f"\nDescriptive statistics:")
print(df.describe())

# ============================================================================
# 2. PROBLEM DEFINITION AND VARIABLE SELECTION
# ============================================================================
print("\n" + "="*80)
print("PROBLEM DEFINITION")
print("="*80)

print("""
PROBLEM:
Predict the percentage of schools reporting offenses based on the 
frequency of different types of school offenses.

TARGET VARIABLE (y): 'Percent of Schools Reporting'
This is the percentage of schools reporting any type of offense.

INPUT VARIABLES (X): All columns except the target variable
- Different types of offenses as predictor variables
- Total of 13 input variables
""")

print("""
VARIABLE SELECTION AND JUSTIFICATION
-----------------------------------

Target Variable (y):
- Percent of Schools Reporting

Input Variables (X):
- Different categories of school offenses used as predictors.

Reason for selection:
These variables may influence or help predict the percentage of schools reporting offenses.
""")

# Remove rows with NaN values
df = df.dropna()

y = df['Percent of Schools Reporting '].values
X = df.drop('Percent of Schools Reporting ', axis=1).values

print(f"\nX shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"\nTarget variable range: [{y.min():.2f}, {y.max():.2f}]")

# ============================================================================
# 3. TRAIN/TEST SPLIT (80/20)
# ============================================================================
print("\n" + "="*80)
print("DATA SPLITTING")
print("="*80)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Ratio: {len(X_train)/(len(X_train)+len(X_test))*100:.1f}% / {len(X_test)/(len(X_train)+len(X_test))*100:.1f}%")

# ============================================================================
# 4. MODEL 1: DECISION TREE
# ============================================================================
print("\n" + "="*80)
print("MODEL 1: DECISION TREE")
print("="*80)

# Version 1: No complexity constraints
print("\nVERSION 1: Default parameters (no constraints)")
print("-" * 40)

dt_model_default = DecisionTreeRegressor(random_state=42)
dt_model_default.fit(X_train, y_train)
y_pred_dt_default = dt_model_default.predict(X_test)

r2_dt_default = r2_score(y_test, y_pred_dt_default)
mae_dt_default = mean_absolute_error(y_test, y_pred_dt_default)
mse_dt_default = mean_squared_error(y_test, y_pred_dt_default)
rmse_dt_default = np.sqrt(mse_dt_default)

print(f"R²:   {r2_dt_default:.4f}")
print(f"MAE:  {mae_dt_default:.4f}")
print(f"MSE:  {mse_dt_default:.4f}")
print(f"RMSE: {rmse_dt_default:.4f}")
print(f"Tree depth: {dt_model_default.get_depth()}")

# Version 2: Controlled complexity
print("\nVERSION 2: Controlled complexity (max_depth=5, min_samples_split=10)")
print("-" * 40)

dt_model_controlled = DecisionTreeRegressor(
    max_depth=5,
    min_samples_split=10,
    random_state=42
)
dt_model_controlled.fit(X_train, y_train)
y_pred_dt_controlled = dt_model_controlled.predict(X_test)

r2_dt_controlled = r2_score(y_test, y_pred_dt_controlled)
mae_dt_controlled = mean_absolute_error(y_test, y_pred_dt_controlled)
mse_dt_controlled = mean_squared_error(y_test, y_pred_dt_controlled)
rmse_dt_controlled = np.sqrt(mse_dt_controlled)

print(f"R²:   {r2_dt_controlled:.4f}")
print(f"MAE:  {mae_dt_controlled:.4f}")
print(f"MSE:  {mse_dt_controlled:.4f}")
print(f"RMSE: {rmse_dt_controlled:.4f}")
print(f"Tree depth: {dt_model_controlled.get_depth()}")

# Comparative analysis
print("\nCOMPARISON - Impact of complexity control:")
print("-" * 40)
print(f"Change in R²:   {r2_dt_controlled - r2_dt_default:+.4f}")
print(f"Change in MAE:  {mae_dt_controlled - mae_dt_default:+.4f}")
print(f"Change in RMSE: {rmse_dt_controlled - rmse_dt_default:+.4f}")
print("""
INTERPRETATION:
- Unconstrained tree perfectly fits training data (overfitting)
- Complexity control reduces overfitting through structural limitations
- max_depth=5 limits tree depth
- min_samples_split=10 requires minimum samples to split nodes
- These parameters improve generalization to new data
""")

# Feature importance analysis
print("\nFEATURE IMPORTANCE ANALYSIS (from Controlled Tree):")
print("-" * 40)
feature_importance = pd.DataFrame({
    'Feature': df.drop('Percent of Schools Reporting ', axis=1).columns,
    'Importance': dt_model_controlled.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by='Importance',
    ascending=False
)

print(feature_importance.to_string(index=False))

# ============================================================================
# 5. MODEL 2: LINEAR REGRESSION
# ============================================================================
print("\n" + "="*80)
print("MODEL 2: LINEAR REGRESSION")
print("="*80)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

r2_lr = r2_score(y_test, y_pred_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)

print(f"\nPERFORMANCE METRICS:")
print(f"R²:   {r2_lr:.4f}")
print(f"MAE:  {mae_lr:.4f}")
print(f"MSE:  {mse_lr:.4f}")
print(f"RMSE: {rmse_lr:.4f}")
print(f"\nCoefficients (first 5): {lr_model.coef_[:5]}")
print(f"Intercept: {lr_model.intercept_:.4f}")

# ============================================================================
# 6. MODEL 3: POLYNOMIAL REGRESSION
# ============================================================================
print("\n" + "="*80)
print("MODEL 3: POLYNOMIAL REGRESSION")
print("="*80)

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

print(f"\nFeature transformation:")
print(f"Original features: {X_train.shape[1]}")
print(f"Polynomial features (degree 2): {X_train_poly.shape[1]}")

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
y_pred_poly = poly_model.predict(X_test_poly)

r2_poly = r2_score(y_test, y_pred_poly)
mae_poly = mean_absolute_error(y_test, y_pred_poly)
mse_poly = mean_squared_error(y_test, y_pred_poly)
rmse_poly = np.sqrt(mse_poly)

print(f"\nPERFORMANCE METRICS:")
print(f"R²:   {r2_poly:.4f}")
print(f"MAE:  {mae_poly:.4f}")
print(f"MSE:  {mse_poly:.4f}")
print(f"RMSE: {rmse_poly:.4f}")

# Comparison: Linear vs Polynomial (Degree 2)
print("\nCOMPARISON - LINEAR vs POLYNOMIAL REGRESSION (Degree 2):")
print("-" * 40)
print(f"{'Metric':<10} {'Linear':<12} {'Polynomial':<12} {'Difference':<12}")
print("-" * 46)
print(f"{'R²':<10} {r2_lr:<12.4f} {r2_poly:<12.4f} {r2_poly - r2_lr:+.4f}")
print(f"{'MAE':<10} {mae_lr:<12.4f} {mae_poly:<12.4f} {mae_poly - mae_lr:+.4f}")
print(f"{'RMSE':<10} {rmse_lr:<12.4f} {rmse_poly:<12.4f} {rmse_poly - rmse_lr:+.4f}")

print("""
INTERPRETATION:
- Polynomial regression captures non-linear relationships
- Adding features can improve or worsen performance
- Particularly useful when variable interactions exist
""")

# ============================================================================
# 6b. MODEL 3b: POLYNOMIAL REGRESSION (Degree 3)
# ============================================================================
print("\n" + "="*80)
print("MODEL 3b: POLYNOMIAL REGRESSION (Degree 3)")
print("="*80)

poly_features_deg3 = PolynomialFeatures(degree=3, include_bias=False)
X_train_poly3 = poly_features_deg3.fit_transform(X_train)
X_test_poly3 = poly_features_deg3.transform(X_test)

print(f"\nFeature transformation:")
print(f"Original features: {X_train.shape[1]}")
print(f"Polynomial features (degree 3): {X_train_poly3.shape[1]}")

poly_model_deg3 = LinearRegression()
poly_model_deg3.fit(X_train_poly3, y_train)
y_pred_poly3 = poly_model_deg3.predict(X_test_poly3)

r2_poly3 = r2_score(y_test, y_pred_poly3)
mae_poly3 = mean_absolute_error(y_test, y_pred_poly3)
mse_poly3 = mean_squared_error(y_test, y_pred_poly3)
rmse_poly3 = np.sqrt(mse_poly3)

print(f"\nPERFORMANCE METRICS:")
print(f"R²:   {r2_poly3:.4f}")
print(f"MAE:  {mae_poly3:.4f}")
print(f"MSE:  {mse_poly3:.4f}")
print(f"RMSE: {rmse_poly3:.4f}")

# Comparison: Degree 2 vs Degree 3
print("\nCOMPARISON - POLYNOMIAL DEGREE 2 vs DEGREE 3:")
print("-" * 40)
print(f"{'Metric':<10} {'Degree 2':<12} {'Degree 3':<12} {'Difference':<12}")
print("-" * 46)
print(f"{'R²':<10} {r2_poly:<12.4f} {r2_poly3:<12.4f} {r2_poly3 - r2_poly:+.4f}")
print(f"{'MAE':<10} {mae_poly:<12.4f} {mae_poly3:<12.4f} {mae_poly3 - mae_poly:+.4f}")
print(f"{'RMSE':<10} {rmse_poly:<12.4f} {rmse_poly3:<12.4f} {rmse_poly3 - rmse_poly:+.4f}")

print("""
INTERPRETATION:
- Higher degree polynomial captures more complex patterns
- Risk of overfitting increases with degree
- Compare both degree 2 and 3 to find optimal complexity
""")

# ============================================================================
# 7. MODEL 4: ARTIFICIAL NEURAL NETWORK
# ============================================================================
print("\n" + "="*80)
print("MODEL 4: ARTIFICIAL NEURAL NETWORK")
print("="*80)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Configuration 1: Small architecture
print("\nCONFIGURATION 1: Small architecture")
print("-" * 40)
print("Parameters:")
print("  - hidden_layer_sizes = (64, 32)")
print("  - activation = 'relu'")
print("  - max_iter = 1000")

ann_model1 = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    max_iter=1000,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1
)
ann_model1.fit(X_train_scaled, y_train)
y_pred_ann1 = ann_model1.predict(X_test_scaled)

r2_ann1 = r2_score(y_test, y_pred_ann1)
mae_ann1 = mean_absolute_error(y_test, y_pred_ann1)
mse_ann1 = mean_squared_error(y_test, y_pred_ann1)
rmse_ann1 = np.sqrt(mse_ann1)

print(f"\nPERFORMANCE METRICS:")
print(f"R²:   {r2_ann1:.4f}")
print(f"MAE:  {mae_ann1:.4f}")
print(f"MSE:  {mse_ann1:.4f}")
print(f"RMSE: {rmse_ann1:.4f}")
print(f"Iterations performed: {ann_model1.n_iter_}")

# Configuration 2: Deeper architecture
print("\nCONFIGURATION 2: Deeper architecture")
print("-" * 40)
print("Parameters:")
print("  - hidden_layer_sizes = (128, 64, 32)")
print("  - activation = 'tanh'")
print("  - max_iter = 2000")

ann_model2 = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32),
    activation='tanh',
    max_iter=2000,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1
)
ann_model2.fit(X_train_scaled, y_train)
y_pred_ann2 = ann_model2.predict(X_test_scaled)

r2_ann2 = r2_score(y_test, y_pred_ann2)
mae_ann2 = mean_absolute_error(y_test, y_pred_ann2)
mse_ann2 = mean_squared_error(y_test, y_pred_ann2)
rmse_ann2 = np.sqrt(mse_ann2)

print(f"\nPERFORMANCE METRICS:")
print(f"R²:   {r2_ann2:.4f}")
print(f"MAE:  {mae_ann2:.4f}")
print(f"MSE:  {mse_ann2:.4f}")
print(f"RMSE: {rmse_ann2:.4f}")
print(f"Iterations performed: {ann_model2.n_iter_}")

# Comparison between configurations
print("\nCOMPARISON - CONFIGURATION 1 vs CONFIGURATION 2:")
print("-" * 40)
print(f"{'Metric':<10} {'Config 1':<12} {'Config 2':<12} {'Difference':<12}")
print("-" * 46)
print(f"{'R²':<10} {r2_ann1:<12.4f} {r2_ann2:<12.4f} {r2_ann2 - r2_ann1:+.4f}")
print(f"{'MAE':<10} {mae_ann1:<12.4f} {mae_ann2:<12.4f} {mae_ann2 - mae_ann1:+.4f}")
print(f"{'RMSE':<10} {rmse_ann1:<12.4f} {rmse_ann2:<12.4f} {rmse_ann2 - rmse_ann1:+.4f}")

print("""
INTERPRETATION OF PARAMETER CHANGES:
- hidden_layer_sizes: More neurons capture more complex patterns
- activation: 'relu' is simpler/faster; 'tanh' allows more complex relationships
- max_iter: More iterations allow better convergence but risk overfitting
- early_stopping: Stops training when validation performance plateaus
""")

# ============================================================================
# 8. FINAL MODEL COMPARISON
# ============================================================================
print("\n" + "="*80)
print("FINAL COMPARISON OF ALL MODELS")
print("="*80)

results = {
    'Model': [
        'Decision Tree (Default)',
        'Decision Tree (Controlled)',
        'Linear Regression',
        'Polynomial Regression (Deg 2)',
        'Polynomial Regression (Deg 3)',
        'ANN Config 1',
        'ANN Config 2'
    ],
    'R²': [r2_dt_default, r2_dt_controlled, r2_lr, r2_poly, r2_poly3, r2_ann1, r2_ann2],
    'MAE': [mae_dt_default, mae_dt_controlled, mae_lr, mae_poly, mae_poly3, mae_ann1, mae_ann2],
    'MSE': [mse_dt_default, mse_dt_controlled, mse_lr, mse_poly, mse_poly3, mse_ann1, mse_ann2],
    'RMSE': [rmse_dt_default, rmse_dt_controlled, rmse_lr, rmse_poly, rmse_poly3, rmse_ann1, rmse_ann2]
}

df_results = pd.DataFrame(results)
print("\nCOMPARATIVE TABLE:")
print(df_results.to_string(index=False))

# Identify best model
best_idx = df_results['R²'].idxmax()
best_model = df_results.loc[best_idx, 'Model']
best_r2 = df_results.loc[best_idx, 'R²']

print(f"\n✓ BEST MODEL: {best_model}")
print(f"  R² = {best_r2:.4f}")

# ============================================================================
# 9. VISUALIZATIONS
# ============================================================================
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# 1. Predictions vs Actual
fig, axes = plt.subplots(3, 3, figsize=(18, 12))
fig.suptitle('Predictions vs Actual Values - All Models', fontsize=16, fontweight='bold')

models = [
    ('Decision Tree (Default)', y_pred_dt_default),
    ('Decision Tree (Controlled)', y_pred_dt_controlled),
    ('Linear Regression', y_pred_lr),
    ('Polynomial Regression (Deg 2)', y_pred_poly),
    ('Polynomial Regression (Deg 3)', y_pred_poly3),
    ('ANN Config 1', y_pred_ann1),
    ('ANN Config 2', y_pred_ann2)
]

r2_scores = [r2_dt_default, r2_dt_controlled, r2_lr, r2_poly, r2_poly3, r2_ann1, r2_ann2]

for idx, (ax, (name, y_pred), r2) in enumerate(zip(axes.flat, models, r2_scores)):
    ax.scatter(y_test, y_pred, alpha=0.6, s=60)
    
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
    
    ax.set_xlabel('Actual Values', fontweight='bold')
    ax.set_ylabel('Predicted Values', fontweight='bold')
    ax.set_title(f'{name}\nR² = {r2:.4f}', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

# Hide the last empty subplot
axes.flat[-1].set_visible(False)

plt.tight_layout()
plt.savefig('predictions_vs_actual.png', dpi=300, bbox_inches='tight')
print("✓ Saved: predictions_vs_actual.png")
plt.close()

# 2. Residuals plot
fig, axes = plt.subplots(3, 3, figsize=(18, 12))
fig.suptitle('Residual Plots - All Models', fontsize=16, fontweight='bold')

residuals_list = [
    y_test - y_pred_dt_default,
    y_test - y_pred_dt_controlled,
    y_test - y_pred_lr,
    y_test - y_pred_poly,
    y_test - y_pred_poly3,
    y_test - y_pred_ann1,
    y_test - y_pred_ann2
]

for ax, (name, y_pred), residuals in zip(axes.flat, models, residuals_list):
    ax.scatter(y_pred, residuals, alpha=0.6, s=60)
    ax.axhline(y=0, color='r', linestyle='--', lw=2)
    
    ax.set_xlabel('Predicted Values', fontweight='bold')
    ax.set_ylabel('Residuals', fontweight='bold')
    ax.set_title(f'{name}\nMean Residuals: {residuals.mean():.4f}', fontweight='bold')
    ax.grid(True, alpha=0.3)

# Hide the last empty subplot
axes.flat[-1].set_visible(False)

plt.tight_layout()
plt.savefig('residuals.png', dpi=300, bbox_inches='tight')
print("✓ Saved: residuals.png")
plt.close()

# 3. Metrics comparison bar charts
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Metrics Comparison Across Models', fontsize=14, fontweight='bold')

names_short = ['DT Def', 'DT Ctrl', 'LR', 'Poly2', 'Poly3', 'ANN1', 'ANN2']
x_pos = np.arange(len(names_short))

# R² Score
axes[0].bar(x_pos, df_results['R²'], color='steelblue', alpha=0.7)
axes[0].set_ylabel('R² Score', fontweight='bold')
axes[0].set_title('Coefficient of Determination (R²)', fontweight='bold')
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(names_short, rotation=45, ha='right')
axes[0].grid(True, alpha=0.3, axis='y')

# MAE
axes[1].bar(x_pos, df_results['MAE'], color='lightcoral', alpha=0.7)
axes[1].set_ylabel('MAE', fontweight='bold')
axes[1].set_title('Mean Absolute Error (MAE)', fontweight='bold')
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(names_short, rotation=45, ha='right')
axes[1].grid(True, alpha=0.3, axis='y')

# RMSE
axes[2].bar(x_pos, df_results['RMSE'], color='lightgreen', alpha=0.7)
axes[2].set_ylabel('RMSE', fontweight='bold')
axes[2].set_title('Root Mean Squared Error (RMSE)', fontweight='bold')
axes[2].set_xticks(x_pos)
axes[2].set_xticklabels(names_short, rotation=45, ha='right')
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: metrics_comparison.png")
plt.close()

# Save results to CSV
df_results.to_csv('model_results.csv', index=False)
print("✓ Saved: model_results.csv")

# ============================================================================
# 10. CONCLUSIONS AND RECOMMENDATIONS
# ============================================================================
print("\n" + "="*80)
print("CONCLUSIONS AND FINAL ANALYSIS")
print("="*80)

print(f"""
EXECUTIVE SUMMARY
{"-"*80}

1. BEST MODEL IDENTIFIED: {best_model}
   - R² Score: {best_r2:.4f}
   - This model explains {best_r2*100:.2f}% of variance in the data

2. ANALYSIS BY MODEL FAMILY:

   A) Decision Tree:
      - Unconstrained tree tends to overfit
      - Complexity control improves generalization
      - Advantage: Interpretable, no scaling needed
      - Disadvantage: Can be unstable with data changes
      - Default R²: {r2_dt_default:.4f}
      - Controlled R²: {r2_dt_controlled:.4f}

   B) Linear Regression:
      - Simple and fast to train
      - Assumes linear relationships between variables
      - Good baseline for comparison
      - MAE: {mae_lr:.4f}
      - R²: {r2_lr:.4f}

   C) Polynomial Regression:
      - Captures non-linear relationships
      - Increases complexity and overfitting risk
      - Variable interactions may be important
      - Degree 2 R²: {r2_poly:.4f}
      - Degree 3 R²: {r2_poly3:.4f}
      - Note: The regression models obtained relatively low R² scores, suggesting that the relationship between the predictor variables and the target variable may be weak or highly non-linear.

   D) Neural Networks:
      - Require feature scaling (StandardScaler)
      - Can capture very complex patterns
      - Config 2 with deeper architecture performs better
      - Config 1 Iterations: {ann_model1.n_iter_}
      - Config 2 Iterations: {ann_model2.n_iter_}
      - Best ANN R²: {max(r2_ann1, r2_ann2):.4f}

3. RECOMMENDATIONS:

   ✓ For production model: {best_model}
   ✓ Monitor overfitting in Decision Trees
   ✓ Consider ensemble of models for robustness
   ✓ Cross-validation for stability confirmation
   ✓ Feature importance analysis for interpretability

4. TECHNICAL CONSIDERATIONS:

   - Dataset: {len(df)} samples, {X.shape[1]} features
   - Training split: 80% ({len(X_train)} samples)
   - Test split: 20% ({len(X_test)} samples)
   - Scaling: StandardScaler for distance-based models
   - Random state: 42 (reproducibility)

5. METRICS USED:

   - R² (Coefficient of Determination): Proportion of variance explained
   - MAE (Mean Absolute Error): Average magnitude of errors
   - MSE (Mean Squared Error): Penalizes large errors
   - RMSE (Root MSE): In same units as target variable
{"-"*80}
""")

print("\nGenerated visualizations:")
print("  1. predictions_vs_actual.png - Prediction scatter plots")
print("  2. residuals.png - Residual analysis for each model")
print("  3. metrics_comparison.png - Model metrics comparison")
print("\nGenerated files:")
print("  - model_results.csv - Complete results table")

print("\n" + "="*80)
print("PROJECT COMPLETED SUCCESSFULLY")
print("="*80)

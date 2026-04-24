import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ============================================================
# 1. LOAD DATA
# ============================================================

# TODO 1: Path to the dataset
DATA_PATH = "diabetes_cleaned.csv"

# TODO 2: Load the dataset
data = pd.read_csv(DATA_PATH)

# --- PRE-PROCESSING ---
# Convert categorical variables (gender, smoking_history) to numeric using One-Hot Encoding
data = pd.get_dummies(data, columns=['gender', 'smoking_history'], drop_first=True)

print("\n=== DATASET PREVIEW ===")
print(data.head())

print("\n=== DATASET SHAPE ===")
print(data.shape)

print("\n=== COLUMN NAMES ===")
print(list(data.columns))


# ============================================================
# 2. DEFINE INPUTS (X) AND TARGET (y)
# ============================================================

# TODO 3: Target column name
TARGET_COLUMN = "diabetes"

# TODO 4: Define X (all columns except the target)
X = data.drop(columns=[TARGET_COLUMN]).values

# TODO 5: Define y (only the target column)
y = data[TARGET_COLUMN].values


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

# TODO 6: Split the data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ============================================================
# 4. SCALE INPUT FEATURES
# ============================================================

scaler = StandardScaler()

# TODO 7: Fit and transform X_train
X_train = scaler.fit_transform(X_train)

# TODO 8: Transform X_test
X_test = scaler.transform(X_test)


# ============================================================
# 5. DEFINE ANN MODEL
# ============================================================

# TODO 9: ANN configuration
model = MLPRegressor(
    hidden_layer_sizes=(64, 32), 
    activation='relu',           
    solver='adam',               
    max_iter=500,                
    random_state=42
)


# ============================================================
# 6. TRAIN MODEL
# ============================================================

# TODO 10: Train the model
model.fit(X_train, y_train)


# ============================================================
# 7. GENERATE PREDICTIONS
# ============================================================

# TODO 11: Generate predictions using X_test
y_pred = model.predict(X_test)


# ============================================================
# 8. CALCULATE EVALUATION METRICS
# ============================================================

# TODO 12: Calculate evaluation metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\n=== MODEL RESULTS ===")
print("R2  :", round(r2, 4))
print("MAE :", round(mae, 4))
print("MSE :", round(mse, 4))
print("RMSE:", round(rmse, 4))


# ============================================================
# 9. TEAM INTERPRETATION
# ============================================================

print("\n=== QUESTIONS FOR YOUR TEAM ===")

# 1. What dataset did you use?
# Answer: We used the 'diabetes_cleaned.csv' dataset, which includes health 
# indicators such as BMI, HbA1c levels, and blood glucose.

# 2. What is your target variable?
# Answer: The target variable is 'diabetes', a binary feature where 1 indicates 
# the presence of the disease and 0 indicates its absence.

# 3. Which ANN configuration did you choose?
# Answer: We implemented an MLPRegressor with two hidden layers (64, 32), 
# using 'relu' activation and the 'adam' solver for 500 iterations.

# 4. What do the evaluation metrics tell you about the model?
# Answer: The R2 of 0.6857 shows the model captures nearly 69% of the data variance. 
# The MAE of 0.0596 and RMSE of 0.1600 are very low, confirming high precision 
# in our predictions.

# 5. Do you think the model is reliable for your project?
# Answer: Yes, the model is reliable. The consistent low error rates (MAE/RMSE) 
# suggest that the neural network successfully identified the key clinical 
# patterns related to diabetes.

# 6. If you had more time, what would you improve?
# Answer: I would experiment with an MLPClassifier to generate a confusion matrix 
# and focus on improving the 'Recall' metric, which is critical for medical diagnosis.
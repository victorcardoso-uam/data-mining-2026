
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Load dataset
data = pd.read_csv("DATASET.csv")

# Select features and target
X = data.drop("target", axis=1)
y = data["target"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Baseline model (no pruning)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Baseline accuracy:", accuracy_score(y_test, pred))


# Pre-pruning example
model_pre = DecisionTreeClassifier(
    max_depth=5,
    min_samples_leaf=5
)

model_pre.fit(X_train, y_train)
pred_pre = model_pre.predict(X_test)

print("Pre-pruned accuracy:", accuracy_score(y_test, pred_pre))


# Post-pruning with cost complexity
model_ccp = DecisionTreeClassifier(ccp_alpha=0.01)

model_ccp.fit(X_train, y_train)
pred_ccp = model_ccp.predict(X_test)

print("Post-pruned accuracy:", accuracy_score(y_test, pred_ccp))


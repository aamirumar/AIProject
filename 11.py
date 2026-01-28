import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from skopt import BayesSearchCV
import joblib

# -------------------------------------------------
# Step 1: Load the Dataset
# -------------------------------------------------
data = load_iris()
X, y = data.data, data.target

# -------------------------------------------------
# Step 2: Train-Test Split
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------------------------
# Step 3: Define the Model
# -------------------------------------------------
model = RandomForestClassifier(random_state=42)

# -------------------------------------------------
# Step 4: Define Bayesian Search Space
# -------------------------------------------------
param_space = {
    "n_estimators": (10, 200),
    "max_depth": (1, 20),
    "min_samples_split": (2, 10),
    "min_samples_leaf": (1, 10),
    "max_features": ["sqrt", "log2", None]
}

# -------------------------------------------------
# Step 5: Bayesian Optimization
# -------------------------------------------------
optimizer = BayesSearchCV(
    estimator=model,
    search_spaces=param_space,
    n_iter=20,        # reduced for faster execution
    cv=3,
    scoring="accuracy",
    random_state=42,
    n_jobs=-1
)

print("Starting Bayesian Optimization...")
optimizer.fit(X_train, y_train)

# -------------------------------------------------
# Step 6: Evaluate Best Model
# -------------------------------------------------
best_model = optimizer.best_estimator_
y_pred = best_model.predict(X_test)

print("\nBest Hyperparameters:")
print(optimizer.best_params_)

print("\nTest Accuracy:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -------------------------------------------------
# Step 7: Save the Optimized Model
# -------------------------------------------------
joblib.dump(best_model, "optimized_rf_model.pkl")
print("\nModel saved as 'optimized_rf_model.pkl'")

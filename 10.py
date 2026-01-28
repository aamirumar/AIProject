import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------------------------
# Step 1: Load Dataset (Titanic)
# -------------------------------------------------
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
data = pd.read_csv(url)

# -------------------------------------------------
# Step 2: Basic Preprocessing
# -------------------------------------------------
# Drop unnecessary columns
data = data.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

# Handle missing values
data["Age"].fillna(data["Age"].median(), inplace=True)
data["Embarked"].fillna(data["Embarked"].mode()[0], inplace=True)

# Separate features and target
X = data.drop("Survived", axis=1)
y = data["Survived"]

# -------------------------------------------------
# Step 3: Train-Test Split
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------------------------
# Step 4: Define Preprocessing Pipelines
# -------------------------------------------------
numerical_features = ["Age", "Fare"]
categorical_features = ["Sex", "Embarked", "Pclass"]

numerical_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# -------------------------------------------------
# Step 5: Define ML Pipeline
# -------------------------------------------------
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("feature_selection", SelectKBest(score_func=f_classif)),
    ("classifier", RandomForestClassifier(random_state=42))
])

# -------------------------------------------------
# Step 6: Hyperparameter Grid (SAFE & ERROR-FREE)
# -------------------------------------------------
param_grid = [
    {
        "feature_selection__k": [5, 7, 9],
        "classifier": [RandomForestClassifier(random_state=42)],
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [None, 5, 10]
    },
    {
        "feature_selection__k": [5, 7, 9],
        "classifier": [SVC(random_state=42)],
        "classifier__C": [0.1, 1, 10],
        "classifier__kernel": ["linear", "rbf"]
    }
]

# -------------------------------------------------
# Step 7: Grid Search for Model Selection
# -------------------------------------------------
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    scoring="accuracy",
    verbose=2
)

# -------------------------------------------------
# Step 8: Train the Pipeline
# -------------------------------------------------
grid_search.fit(X_train, y_train)

# -------------------------------------------------
# Step 9: Evaluate Best Model
# -------------------------------------------------
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nTest Accuracy:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

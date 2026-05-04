import pandas as pd
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from sklearn.preprocessing import label_binarize

# Load data
df = pd.read_csv("data/train.csv")

# Clean column names
df.columns = [col.strip().lower() for col in df.columns]

# Target column
target_col = "class"

# Features and target
X = df.drop(columns=[target_col])
y = df[target_col].astype(str).str.strip().str.lower()

# Separate categorical and numerical columns
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()


preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numerical_cols)
    ]
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {}

models["logistic"] = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42))
])

models["random_forest"] = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

models["svm"] = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", SVC(probability=True, random_state=42))
])

models["xgboost"] = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(eval_metric="logloss", random_state=42))
])


results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, f"model/saved_model_{name}.pkl")

    # Predict
    y_pred = model.predict(X_test)

    # Convert to 0/1 for binary AUC
    y_true_bin = label_binarize(y_test, classes=sorted(y.unique())).flatten()
    y_pred_bin = label_binarize(y_pred, classes=sorted(y.unique())).flatten()

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    rep = classification_report(y_test, y_pred, output_dict=True)

    results[name] = {
        "accuracy": acc,
        "precision": rep["weighted avg"]["precision"],
        "recall": rep["weighted avg"]["recall"],
        "f1": rep["weighted avg"]["f1-score"],
        "auc": 0.0  
    }

    print(f"{name} - Accuracy: {acc:.4f}")


joblib.dump(results, "model/model_results.pkl")
print("All models saved successfully!")

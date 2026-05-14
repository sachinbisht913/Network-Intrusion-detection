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


df = pd.read_csv("data/train.csv")


df.columns = [col.strip().lower() for col in df.columns]


target_col = "class"


X = df.drop(columns=[target_col])
y = df[target_col].astype(str).str.strip().str.lower()


categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()


preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numerical_cols)
    ]
)

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




results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)

   
    joblib.dump(model, f"model/saved_model_{name}.pkl")

   
    y_pred = model.predict(X_test)

   
    y_true_bin = label_binarize(y_test, classes=sorted(y.unique())).flatten()
    y_pred_bin = label_binarize(y_pred, classes=sorted(y.unique())).flatten()

  
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

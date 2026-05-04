
import pandas as pd
import numpy as np
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import os

# File paths
TRAIN_PATH = "data/KDDTrain+.txt"
TEST_PATH = "data/KDDTest+.txt"


COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 
    'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
    'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
    'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'label', 'difficulty'
]

def load_txt_data(path):

    print(f"Loading {path}...")
    

    df = pd.read_csv(path, header=None, names=COLUMNS)
    
    print("Original labels:", df['label'].value_counts().head())
    
    attack_types = [
        'neptune', 'satan', 'ipsweep', 'portsweep', 'smurf', 'mscan', 'saint',
        'back', 'warezclient', 'teardrop', 'pod', 'land', 'buffer_overflow',
        'loadmodule', 'rootkit', 'imap', 'guess_passwd', 'ftp_write',
        'multihop', 'phf', 'perl', 'spy', 'warezmaster'
    ]
    
    df['class'] = df['label'].apply(lambda x: 0 if str(x).lower() == 'normal' else 1)
    
    # Drop difficulty column 
    df = df.drop('difficulty', axis=1)
    df = df.drop('label', axis=1)
    
    # Clean categorical columns FIRST 
    cat_cols = ['protocol_type', 'service', 'flag']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
    
    # Convert numerical columns to float
    num_cols = [col for col in df.columns if col not in ['class'] + cat_cols]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    print(f"✓ Loaded {len(df)} rows")
    print(f"✓ Binary classes: {dict(df['class'].value_counts())}")
    return df

def main():
    print("🔧 NSL-KDD INTRUSION DETECTION - FIXED")
    print("="*50)
    
    # Check files
    if not os.path.exists(TRAIN_PATH):
        print(f" {TRAIN_PATH} missing!")
        return
    if not os.path.exists(TEST_PATH):
        print(f" {TEST_PATH} missing!")
        return
    
    # Load data
    train_df = load_txt_data(TRAIN_PATH)
    test_df = load_txt_data(TEST_PATH)
    
    # Prepare features
    X_train = train_df.drop('class', axis=1)
    y_train = train_df['class']
    X_test = test_df.drop('class', axis=1)
    y_test = test_df['class']
    
    # Define categorical/numerical columns explicitly
    cat_cols = ['protocol_type', 'service', 'flag']
    num_cols = [col for col in X_train.columns if col not in cat_cols]
    
    print(f"Features: {len(cat_cols)} cat + {len(num_cols)} num = {len(X_train.columns)} total")
    
    # Preprocessor
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
        ("num", StandardScaler(), num_cols)
    ])
    
    # Models (simplified for speed)
    models = {
        "random_forest": Pipeline([
            ("prep", preprocessor),
            ("clf", RandomForestClassifier(n_estimators=50, random_state=42))
        ]),
        "logistic": Pipeline([
            ("prep", preprocessor),
            ("clf", LogisticRegression(max_iter=500, random_state=42))
        ])
    }
    
    # Train
    os.makedirs("model", exist_ok=True)
    results = {}
    
    print("\n🚀 TRAINING...")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        joblib.dump(model, f"model/{name}.pkl")
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = {"accuracy": acc}
        print(f"  ✅ Accuracy: {acc:.3f}")
    
 
    joblib.dump(results, "model/results.pkl")
    
    print("\n🎉 FIXED & COMPLETE!")
    print("Saved:")
    for name in models:
        print(f"  model/{name}.pkl")
    print("Next: Flask prediction app!")

if __name__ == "__main__":
    main()

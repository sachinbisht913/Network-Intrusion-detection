import pandas as pd
import numpy as np
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os


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
    'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate',
    'dst_host_srv_serror_rate',
    'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate',
    'label',
    'difficulty'
]

def load_txt_data(path):

    print(f"\nLoading {path}...")

    df = pd.read_csv(path, header=None, names=COLUMNS)

    print("Original labels:")
    print(df['label'].value_counts().head())

    # =========================
    # Attack Categories
    # =========================

    dos_attacks = [
        'neptune',
        'smurf',
        'back',
        'teardrop',
        'pod',
        'land'
    ]

    probe_attacks = [
        'satan',
        'ipsweep',
        'portsweep',
        'nmap',
        'mscan',
        'saint'
    ]

    r2l_attacks = [
        'guess_passwd',
        'ftp_write',
        'imap',
        'multihop',
        'phf',
        'spy',
        'warezclient',
        'warezmaster'
    ]

    u2r_attacks = [
        'buffer_overflow',
        'loadmodule',
        'rootkit',
        'perl'
    ]

    def categorize_attack(label):

        label = str(label).lower()

        if label == "normal":
            return "Normal"

        elif label in dos_attacks:
            return "DoS"

        elif label in probe_attacks:
            return "Probe"

        elif label in r2l_attacks:
            return "R2L"

        elif label in u2r_attacks:
            return "U2R"

        else:
            return "Other"

  
    df['class'] = df['label'].apply(categorize_attack)

    
    df = df.drop('difficulty', axis=1)
    df = df.drop('label', axis=1)


    cat_cols = ['protocol_type', 'service', 'flag']

    for col in cat_cols:

        if col in df.columns:

            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.lower()
            )

   
    num_cols = [
        col for col in df.columns
        if col not in ['class'] + cat_cols
    ]

    for col in num_cols:

        df[col] = pd.to_numeric(
            df[col],
            errors='coerce'
        ).fillna(0)

    print(f"✓ Loaded {len(df)} rows")
    print(f"✓ Classes: {dict(df['class'].value_counts())}")

    return df


def main():

   
    print(" NSL-KDD Intrusion Detection Training")


    # Check dataset files
    if not os.path.exists(TRAIN_PATH):
        print(f" Missing: {TRAIN_PATH}")
        return

    if not os.path.exists(TEST_PATH):
        print(f" Missing: {TEST_PATH}")
        return

   
    train_df = load_txt_data(TRAIN_PATH)
    test_df = load_txt_data(TEST_PATH)

    
    X_train = train_df.drop('class', axis=1)
    y_train = train_df['class']

    X_test = test_df.drop('class', axis=1)
    y_test = test_df['class']

   
    cat_cols = ['protocol_type', 'service', 'flag']

    num_cols = [
        col for col in X_train.columns
        if col not in cat_cols
    ]

    print(f"\nFeatures:")
    print(f"Categorical: {len(cat_cols)}")
    print(f"Numerical: {len(num_cols)}")
    print(f"Total: {len(X_train.columns)}")

    # Preprocessing
    preprocessor = ColumnTransformer([

        (
            "cat",
            OneHotEncoder(
                handle_unknown='ignore',
                sparse_output=False
            ),
            cat_cols
        ),

        (
            "num",
            StandardScaler(),
            num_cols
        )
    ])

    # Models
    models = {

        "random_forest": Pipeline([

            ("prep", preprocessor),

            (
                "clf",
                RandomForestClassifier(
                    n_estimators=200,

                    max_depth=20,

                    min_samples_split=5,

                    min_samples_leaf=2,

                    class_weight='balanced',

                    random_state=42
                )
            )
        ]),

        "logistic": Pipeline([

            ("prep", preprocessor),

            (
                "clf",
                LogisticRegression(
                    max_iter=1200,
                    class_weight='balanced',
                    random_state=42

                )
            )
        ])
    }

   
    os.makedirs("model", exist_ok=True)

    results = {}

    print("\n")
    print(" TRAINING MODELS")
  

    for name, model in models.items():

        print(f"\nTraining {name}...")

        
        model.fit(X_train, y_train)

       
        joblib.dump(
            model,
            f"model/{name}.pkl"
        )

        
        y_pred = model.predict(X_test)

        
        acc = accuracy_score(y_test, y_pred)

        results[name] = {
            "accuracy": acc
        }

        print(f" Accuracy: {acc:.4f}")

  
    joblib.dump(
        results,
        "model/results.pkl"
    )

    print("\n")
    print(" TRAINING COMPLETED SUCCESSFULLY")
  

    print("\nSaved Models:")

    for name in models:
        print(f" model/{name}.pkl")

    print("\nRun Flask App:")
    print(" python app.py")


if __name__ == "__main__":
    main()
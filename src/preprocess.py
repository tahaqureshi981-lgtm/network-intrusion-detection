import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = "data"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# NSL-KDD has no header — these are the 41 feature names + label + difficulty
COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes",
    "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty"
]

# Map specific attack names → 5 categories
ATTACK_MAP = {
    # Normal
    "normal": "Normal",
    # DoS attacks
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "mailbomb": "DoS",
    "apache2": "DoS", "processtable": "DoS", "udpstorm": "DoS",
    # Probe attacks
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe",
    "satan": "Probe", "mscan": "Probe", "saint": "Probe",
    # R2L attacks
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L",
    "multihop": "R2L", "phf": "R2L", "spy": "R2L", "warezclient": "R2L",
    "warezmaster": "R2L", "sendmail": "R2L", "named": "R2L",
    "snmpgetattack": "R2L", "snmpguess": "R2L", "xlock": "R2L",
    "xsnoop": "R2L", "httptunnel": "R2L",
    # U2R attacks
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R",
    "rootkit": "U2R", "ps": "U2R", "sqlattack": "U2R",
    "xterm": "U2R", "worm": "U2R"
}


def load_data(train_path: str, test_path: str):
    """Load NSL-KDD train and test sets."""
    train = pd.read_csv(train_path, header=None, names=COLUMNS)
    test  = pd.read_csv(test_path,  header=None, names=COLUMNS)

    print(f"Train shape: {train.shape}")
    print(f"Test shape : {test.shape}")
    print(f"\nUnique labels in train:\n{train['label'].value_counts()}")

    return train, test


def map_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Map specific attack names to 5 categories."""
    df = df.copy()
    df["attack_category"] = df["label"].map(ATTACK_MAP)

    # Any unmapped labels → treat as their closest category
    unmapped = df[df["attack_category"].isna()]["label"].unique()
    if len(unmapped) > 0:
        print(f"⚠️  Unmapped labels (treating as DoS): {unmapped}")
        df["attack_category"] = df["attack_category"].fillna("DoS")

    print(f"\nAttack category distribution:")
    print(df["attack_category"].value_counts())

    return df


def encode_features(df: pd.DataFrame,
                    encoders: dict = None,
                    fit: bool = True) -> tuple:
    """
    Label encode categorical features.
    fit=True for training, fit=False for test (use existing encoders).
    """
    df = df.copy()
    categorical_cols = ["protocol_type", "service", "flag"]

    if fit:
        encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    else:
        for col in categorical_cols:
            le = encoders[col]
            # Handle unseen labels by mapping to most frequent
            df[col] = df[col].astype(str).apply(
                lambda x: x if x in le.classes_ else le.classes_[0]
            )
            df[col] = le.transform(df[col])

    return df, encoders


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop non-feature columns."""
    drop_cols = ["label", "difficulty", "attack_category"]
    return df.drop(columns=[c for c in drop_cols if c in df.columns])


def run(train_path: str = None, test_path: str = None):
    """Full preprocessing pipeline."""
    if train_path is None:
        train_path = f"{DATA_DIR}/KDDTrain+.txt"
    if test_path is None:
        test_path = f"{DATA_DIR}/KDDTest+.txt"

    # 1. Load
    train_df, test_df = load_data(train_path, test_path)

    # 2. Map labels
    train_df = map_labels(train_df)
    test_df  = map_labels(test_df)

    # 3. Encode categorical features
    train_df, encoders = encode_features(train_df, fit=True)
    test_df,  _        = encode_features(test_df, encoders=encoders, fit=False)

    # 4. Prepare X and y
    X_train = prepare_features(train_df)
    y_train = train_df["attack_category"]
    X_test  = prepare_features(test_df)
    y_test  = test_df["attack_category"]

    # 5. Scale numeric features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns
    )

    # 6. Save encoders + scaler + feature names
    joblib.dump(encoders,               f"{MODELS_DIR}/encoders.pkl")
    joblib.dump(scaler,                 f"{MODELS_DIR}/scaler.pkl")
    joblib.dump(list(X_train.columns),  f"{MODELS_DIR}/feature_names.pkl")
    joblib.dump(list(y_train.unique()), f"{MODELS_DIR}/classes.pkl")

    print(f"\n✅ Preprocessing complete")
    print(f"   X_train: {X_train_scaled.shape}")
    print(f"   X_test : {X_test_scaled.shape}")
    print(f"   Classes : {sorted(y_train.unique())}")

    return X_train_scaled, X_test_scaled, y_train, y_test


if __name__ == "__main__":
    run()
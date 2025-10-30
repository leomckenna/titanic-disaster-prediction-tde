# src/run/model.py
import argparse
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BANNER = "=" * 72


def log(msg):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def load_csv(path):
    log(f"Loading file: {path}")
    df = pd.read_csv(path)
    log(f"Loaded shape: {df.shape}")
    return df


def add_adjust_features(df, is_train=True):
    log("15) ADD/ADJUST: Creating simple engineered features (FamilySize, IsAlone, Title).")

    # Ensure required base columns exist (create placeholders if missing)
    for col in ["SibSp", "Parch"]:
        if col not in df.columns:
            df[col] = 0
            log(f"Added missing column '{col}' with zeros (not found).")

    # Family-based features
    df["FamilySize"] = df["SibSp"].fillna(0) + df["Parch"].fillna(0) + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Title from name
    if "Name" in df.columns:
        df["Title"] = (
            df["Name"]
            .fillna("")
            .str.extract(r",\s*([^\.]+)\.", expand=False)
            .fillna("Unknown")
            .str.strip()
        )
    else:
        df["Title"] = "Unknown"
        log("Column 'Name' not found; setting Title='Unknown'.")

    # Features to use
    candidate_features = [
        # numeric
        "Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone",
        # categorical
        "Sex", "Embarked", "Title",
    ]

    # Create placeholder columns if missing so the pipeline can impute
    for c in candidate_features:
        if c not in df.columns:
            df[c] = np.nan
            log(f"Added missing column '{c}' as NaN (placeholder).")

    log(f"Using features: {candidate_features}")
    return df, candidate_features


def build_pipeline(numeric_features, categorical_features):
    log("Building preprocessing + LogisticRegression pipeline.")
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),  # works with sparse output
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=True)),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )

    # Use a solver that supports sparse matrices
    clf = LogisticRegression(max_iter=1000, solver="liblinear")

    pipe = Pipeline(steps=[
        ("preprocess", pre),
        ("logreg", clf),
    ])
    return pipe


def fit_and_report(pipe, X_train, y_train, label="TRAIN"):
    log(f"Fitting model on {label} set...")
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_train)
    acc = accuracy_score(y_train, preds)
    cm = confusion_matrix(y_train, preds)
    log(f"16) ACCURACY ({label}): {acc:.4f}")
    log(f"{label} Confusion Matrix:\n{cm}")
    return pipe, acc


def evaluate(pipe, X, y, label="TEST"):
    log(f"Evaluating on {label} set...")
    preds = pipe.predict(X)
    acc = accuracy_score(y, preds)
    cm = confusion_matrix(y, preds)
    log(f"18) ACCURACY ({label}): {acc:.4f}")
    log(f"{label} Confusion Matrix:\n{cm}")
    return acc


def main(args):
    data_dir = args.data_dir
    train_path = os.path.join(data_dir, "train.csv")
    test_path  = os.path.join(data_dir, "test.csv")

    if not os.path.exists(train_path):
        log(f"ERROR: {train_path} not found.")
        sys.exit(1)
    if not os.path.exists(test_path):
        log(f"ERROR: {test_path} not found.")
        sys.exit(1)

    train = load_csv(train_path)
    test  = load_csv(test_path)

    if "Survived" not in train.columns:
        log("ERROR: TRAIN must contain 'Survived' column.")
        sys.exit(1)

    # 15) Add/Adjust features
    train, features = add_adjust_features(train, is_train=True)
    test,  _        = add_adjust_features(test,  is_train=False)

    # Split X/y
    X_train = train[features]
    y_train = train["Survived"].astype(int)

    # Build pipeline
    num_feats = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_feats = [c for c in features if c not in num_feats]
    log(f"Numeric features: {num_feats}")
    log(f"Categorical features: {cat_feats}")
    pipe = build_pipeline(num_feats, cat_feats)

    # 16) Fit & training accuracy
    pipe, train_acc = fit_and_report(pipe, X_train, y_train, label="TRAIN")

    # 17) Predict on TEST, 18) Measure TEST accuracy (if labels exist)
    if "Survived" in test.columns:
        y_test = test["Survived"].astype(int)
        X_test = test[features]
        log("17) PREDICT: Generating predictions for TEST set with labels present.")
        _ = evaluate(pipe, X_test, y_test, label="TEST")
    else:
        X_test = test[features]
        log("17) PREDICT: TEST has no 'Survived' column. Producing predictions only.")
        preds = pipe.predict(X_test)
        log(f"Sample predictions (first 10): {preds[:10].tolist()}")
        log("18) ACCURACY: Skipped because TEST has no labels.")

    log("DONE.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Titanic Logistic Regression runner")
    parser.add_argument("--data_dir", type=str, default="src/data",
                        help="Directory containing train.csv and test.csv")
    args = parser.parse_args()
    main(args)

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


def log(msg: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def load_csv(path: str) -> pd.DataFrame:
    log(f"Loading file: {path}")
    df = pd.read_csv(path)
    log(f"Loaded shape: {df.shape}")
    return df


def add_adjust_features(df: pd.DataFrame, which: str) -> tuple[pd.DataFrame, list[str]]:
    log(f"15) ADD/ADJUST ({which}): Creating features (FamilySize, IsAlone, Title).")

    # Ensure required base columns exist
    for col in ["SibSp", "Parch"]:
        if col not in df.columns:
            df[col] = 0
            log(f"Added missing column '{col}' with zeros.")

    # Family-based features
    df["FamilySize"] = df["SibSp"].fillna(0) + df["Parch"].fillna(0) + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Title from Name
    if "Name" in df.columns:
        df["Title"] = (
            df["Name"].fillna("")
            .str.extract(r",\s*([^\.]+)\.", expand=False)
            .fillna("Unknown")
            .str.strip()
        )
    else:
        df["Title"] = "Unknown"
        log("Column 'Name' not found; setting Title='Unknown'.")

    # Features we will use
    features = [
        # numeric
        "Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone",
        # categorical
        "Sex", "Embarked", "Title",
    ]

    # Create placeholder columns if missing so the pipeline can impute
    for c in features:
        if c not in df.columns:
            df[c] = np.nan
            log(f"Added missing column '{c}' as NaN placeholder.")

    log(f"Using features: {features}")
    return df, features


def build_pipeline(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    log("Building preprocessing + LogisticRegression pipeline.")
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        # with_mean=False keeps sparse compatibility when combined with OneHot
        ("scaler", StandardScaler(with_mean=False)),
    ])
    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )

    clf = LogisticRegression(max_iter=1000, solver="liblinear")

    return Pipeline(steps=[
        ("preprocess", pre),
        ("logreg", clf),
    ])


def fit_and_report(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, label: str) -> Pipeline:
    log(f"Fitting model on {label} set…")
    pipe.fit(X, y)
    preds = pipe.predict(X)
    acc = accuracy_score(y, preds)
    cm = confusion_matrix(y, preds)
    log(f"16) ACCURACY ({label}): {acc:.4f}")
    log(f"{label} Confusion Matrix:\n{cm}")
    return pipe


def main(args):
    train_path = os.path.join(args.data_dir, "train.csv")
    test_path  = os.path.join(args.data_dir, "test.csv")

    if not os.path.exists(train_path):
        log(f"ERROR: {train_path} not found."); sys.exit(1)
    if not os.path.exists(test_path):
        log(f"ERROR: {test_path} not found."); sys.exit(1)

    # Load
    train_df = load_csv(train_path)
    test_df  = load_csv(test_path)

    if "Survived" not in train_df.columns:
        log("ERROR: TRAIN must contain 'Survived' column."); sys.exit(1)

    # 15) Feature engineering
    train_df, features = add_adjust_features(train_df, which="TRAIN")
    test_df,  _        = add_adjust_features(test_df,  which="TEST")

    # Split X/y
    X_train = train_df[features]
    y_train = train_df["Survived"].astype(int)

    # Build pipeline
    num_feats = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_feats = [c for c in features if c not in num_feats]
    log(f"Numeric features: {num_feats}")
    log(f"Categorical features: {cat_feats}")

    pipe = build_pipeline(num_feats, cat_feats)

    # 16) Fit & training accuracy
    pipe = fit_and_report(pipe, X_train, y_train, label="TRAIN")

    # 17–18) Predict on TEST; measure accuracy only if labels exist
    X_test = test_df[features]
    if "Survived" in test_df.columns:
        y_test = test_df["Survived"].astype(int)
        log("17) PREDICT: Generating predictions for TEST with labels.")
        preds = pipe.predict(X_test)
        acc = accuracy_score(y_test, preds)
        cm = confusion_matrix(y_test, preds)
        log(f"18) ACCURACY (TEST): {acc:.4f}")
        log(f"TEST Confusion Matrix:\n{cm}")
    else:
        log("17) PREDICT: TEST has no 'Survived' column. Producing predictions only.")
        preds = pipe.predict(X_test)
        log(f"Sample predictions (first 10): {list(map(int, preds[:10]))}")
    
        # Try loading gender_submission.csv to evaluate test accuracy
        gender_path = os.path.join(args.data_dir, "gender_submission.csv")
        if os.path.exists(gender_path):
            log(f"Found gender_submission.csv: {gender_path}")
            gender_df = pd.read_csv(gender_path)
            if "Survived" in gender_df.columns:
                # Align on PassengerId to ensure order matches
                merged = pd.merge(
                    test_df[["PassengerId"]].copy(),
                    gender_df[["PassengerId", "Survived"]],
                    on="PassengerId",
                    how="inner"
                )
                true_y = merged["Survived"].astype(int)
                pred_y = preds[:len(true_y)]
                acc = accuracy_score(true_y, pred_y)
                cm = confusion_matrix(true_y, pred_y)
                log(f"18) ACCURACY (TEST via gender_submission): {acc:.4f}")
                log(f"TEST Confusion Matrix:\n{cm}")
            else:
                log("gender_submission.csv found, but no 'Survived' column.")
        else:
            log("18) ACCURACY: Skipped because TEST has no labels or gender_submission.csv.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Titanic Logistic Regression runner")
    parser.add_argument("--data_dir", type=str, default="src/data",
                        help="Directory containing train.csv and test.csv")
    main(parser.parse_args())

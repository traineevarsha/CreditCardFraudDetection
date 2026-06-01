# phase1_models.py
# Phase 1: Train 3 fraud detection models
# This version is safer for unseen/random input.
# It avoids exact merchant/city/job memorization.

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, roc_auc_score

from imblearn.over_sampling import SMOTE


SEED = 42
DATA_PATH = "data/fraudTrain.csv"
MODEL_DIR = "models"

np.random.seed(SEED)
os.makedirs(MODEL_DIR, exist_ok=True)


def load_data():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    return df


def prepare_features(df):
    print("Preparing features...")

    df["hour"] = pd.to_datetime(df["trans_date_trans_time"]).dt.hour
    df["age"] = (pd.Timestamp.today() - pd.to_datetime(df["dob"])).dt.days // 365

    # Simple behavior-based features
    df["is_night"] = (df["hour"] < 6).astype(int)
    df["is_high_amount"] = (df["amt"] > 500).astype(int)
    df["is_small_city"] = (df["city_pop"] < 5000).astype(int)

    # Drop columns not useful for unseen/random user input
    drop_cols = [
        "Unnamed: 0",
        "trans_date_trans_time",
        "cc_num",
        "first",
        "last",
        "street",
        "zip",
        "trans_num",
        "unix_time",
        "dob",
        "lat",
        "long",
        "merch_lat",
        "merch_long",

        # Removed for better unseen-data support
        "merchant",
        "city",
        "job"
    ]

    df.drop(columns=drop_cols, inplace=True)

    return df


def encode_columns(df):
    print("Encoding categorical columns...")

    encoders = {}
    text_cols = ["category", "gender", "state"]

    for col in text_cols:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col].astype(str))
        encoders[col] = encoder

    joblib.dump(encoders, f"{MODEL_DIR}/encoders.pkl")

    return df


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    print("Training models...")

    models = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=15,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=SEED
        ),

        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=SEED
        ),

        "Neural Network": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=100,
            early_stopping=True,
            random_state=SEED
        )
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        report = classification_report(y_test, y_pred, output_dict=True)

        recall = report["1"]["recall"]
        precision = report["1"]["precision"]
        f1 = report["1"]["f1-score"]
        auc = roc_auc_score(y_test, y_prob)

        results[name] = {
            "model": model,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc
        }

        print(
            f"{name:<22} "
            f"Precision: {precision:.3f}  "
            f"Recall: {recall:.3f}  "
            f"F1: {f1:.3f}  "
            f"AUC: {auc:.3f}"
        )

    return results


def save_models(results, X_train, X_test, y_train, y_test):
    print("Saving models...")

    joblib.dump(results["Decision Tree"]["model"], f"{MODEL_DIR}/decision_tree.pkl")
    joblib.dump(results["Logistic Regression"]["model"], f"{MODEL_DIR}/logistic_regression.pkl")
    joblib.dump(results["Neural Network"]["model"], f"{MODEL_DIR}/neural_network.pkl")

    joblib.dump((X_train, y_train), f"{MODEL_DIR}/train_data.pkl")
    joblib.dump((X_test, y_test), f"{MODEL_DIR}/test_data.pkl")

    best_name = max(results, key=lambda name: results[name]["recall"])
    best_model = results[best_name]["model"]

    joblib.dump(best_model, f"{MODEL_DIR}/best_model.pkl")

    with open(f"{MODEL_DIR}/best_model_name.txt", "w") as file:
        file.write(best_name)

    print("\nBest Phase 1 model:", best_name)


def main():
    df = load_data()
    df = prepare_features(df)
    df = encode_columns(df)

    X = df.drop("is_fraud", axis=1)
    y = df["is_fraud"]

    feature_names = list(X.columns)
    joblib.dump(feature_names, f"{MODEL_DIR}/feature_names.pkl")

    print("\nFeatures used:")
    for feature in feature_names:
        print("-", feature)

    print(f"\nTotal rows: {len(df):,}")
    print(f"Fraud cases: {y.sum():,}")
    print(f"Fraud rate: {y.mean() * 100:.2f}%")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=SEED,
        stratify=y
    )

    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=feature_names
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=feature_names
    )

    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")

    print("\nApplying SMOTE...")
    smote = SMOTE(sampling_strategy=0.3, random_state=SEED)

    X_train_balanced, y_train_balanced = smote.fit_resample(
        X_train_scaled,
        y_train
    )

    results = train_and_evaluate_models(
        X_train_balanced,
        X_test_scaled,
        y_train_balanced,
        y_test
    )

    save_models(
        results,
        X_train_balanced,
        X_test_scaled,
        y_train_balanced,
        y_test
    )

    print("\nPhase 1 training complete.")


if __name__ == "__main__":
    main()
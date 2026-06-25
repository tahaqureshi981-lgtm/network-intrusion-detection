import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from collections import Counter
from src.preprocess import run as preprocess

MODELS_DIR  = "models"
REPORTS_DIR = "reports"
os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


def get_models():
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"   # handles class imbalance
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            random_state=42,
            eval_metric="mlogloss",
            verbosity=0,
            use_label_encoder=False
        )
    }


def encode_labels(y_train, y_test):
    """Encode string labels to integers for XGBoost."""
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc  = le.transform(y_test)
    joblib.dump(le, f"{MODELS_DIR}/label_encoder.pkl")
    return y_train_enc, y_test_enc, le


def evaluate(name, model, X_test, y_test, y_test_enc, le):
    """Evaluate model and print metrics."""
    y_pred_enc = model.predict(X_test)
    y_pred     = le.inverse_transform(y_pred_enc)
    y_true     = le.inverse_transform(y_test_enc)

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="weighted")

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 (weighted): {f1:.4f}")
    print(f"\n{classification_report(y_true, y_pred)}")

    return {"model": name, "accuracy": acc, "f1_weighted": f1,
            "y_pred": y_pred, "y_true": y_true}


def plot_confusion_matrix(name, y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_title(f"Confusion Matrix — {name}", fontweight="bold", fontsize=13)
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    fname = f"{REPORTS_DIR}/cm_{name.lower()}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"✅ Saved: {fname}")


def plot_model_comparison(results):
    df = pd.DataFrame([{"Model": r["model"],
                         "Accuracy": r["accuracy"],
                         "F1 Weighted": r["f1_weighted"]}
                        for r in results])
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(df))
    width = 0.35
    ax.bar(x - width/2, df["Accuracy"],    width, label="Accuracy",    color="#4C9BE8")
    ax.bar(x + width/2, df["F1 Weighted"], width, label="F1 Weighted", color="#E8614C")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Model"])
    ax.set_ylim(0.7, 1.0)
    ax.set_title("Model Comparison", fontweight="bold", fontsize=13)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/model_comparison.png", dpi=150)
    plt.close()
    print("✅ Saved: model_comparison.png")


def plot_class_distribution(y_train):
    counts = pd.Series(y_train).value_counts()
    fig, ax = plt.subplots(figsize=(8, 4))
    counts.plot(kind="bar", ax=ax, color="#4C9BE8", edgecolor="white")
    ax.set_title("Training Set Class Distribution", fontweight="bold")
    ax.set_xlabel("Attack Category")
    ax.set_ylabel("Count")
    ax.set_xticklabels(counts.index, rotation=15)
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height()):,}",
                    (p.get_x() + p.get_width()/2, p.get_height() + 200),
                    ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/class_distribution.png", dpi=150)
    plt.close()
    print("✅ Saved: class_distribution.png")


def run():
    # 1. Preprocess
    print("⏳ Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess()

    # 2. Encode labels
    y_train_enc, y_test_enc, le = encode_labels(y_train, y_test)
    # Apply SMOTE to balance rare classes
    print("⏳ Applying SMOTE to balance classes...")
    sm = SMOTE(random_state=42, k_neighbors=3)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train_enc)
    print(f"After SMOTE: {Counter(le.inverse_transform(y_train_res))}")
    classes = list(le.classes_)
    print(f"\nClasses: {classes}")

    # 3. Plot class distribution
    plot_class_distribution(y_train)

    # 4. Train + evaluate models
    models  = get_models()
    results = []
    best_f1    = 0
    best_model = None
    best_name  = ""

    for name, model in models.items():
        print(f"\n⏳ Training {name}...")
        model.fit(X_train_res, y_train_res)

        res = evaluate(name, model, X_test, y_test_enc, y_test_enc, le)
        results.append(res)

        plot_confusion_matrix(name, res["y_true"], res["y_pred"], classes)

        joblib.dump(model, f"{MODELS_DIR}/{name.lower()}.pkl")
        print(f"✅ Saved: models/{name.lower()}.pkl")

        if res["f1_weighted"] > best_f1:
            best_f1    = res["f1_weighted"]
            best_model = model
            best_name  = name

    # 5. Save best model
    joblib.dump(best_model, f"{MODELS_DIR}/best_model.pkl")
    joblib.dump(best_name,  f"{MODELS_DIR}/best_model_name.pkl")
    joblib.dump(classes,    f"{MODELS_DIR}/classes.pkl")

    # 6. Comparison chart
    plot_model_comparison(results)

    print(f"\n🏆 Best model: {best_name} (F1: {best_f1:.4f})")
    print(f"✅ All models saved to models/")

    return best_model, best_name, classes


if __name__ == "__main__":
    run()
import argparse
import mlflow
from types import SimpleNamespace
from lib.config import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
)

def build_main_pipeline(max_features):
    text_tfidf = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        analyzer="word",
        ngram_range=(1, 2),
        max_features=max_features,
        min_df=2
    )
    clf = LogisticRegression(
        penalty="l2",
        C=2.0,
        solver="lbfgs",
        max_iter=100,
        class_weight=None,
        n_jobs=-1
    )
    pipe = Pipeline([
        ("tfidf", text_tfidf),
        ("clf", clf)
    ])
    return pipe


def plot_confusion_matrix(y_true, y_pred, labels):
    """Return a Matplotlib Figure for the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    return fig


def plot_decision_boundary(model, X_sparse, y, label_names, title: str):
    """2D projection with SVD internally for visualization only; returns a Figure."""
    svd = TruncatedSVD(n_components=2, random_state=42)
    X_2d = svd.fit_transform(X_sparse)
    X_2d = StandardScaler().fit_transform(X_2d)

    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    clf_2d = clone(model)
    clf_2d.fit(X_2d, y)

    Z = clf_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    for lbl in np.unique(y):
        ax.scatter(X_2d[y == lbl, 0], X_2d[y == lbl, 1],
                   label=label_names[lbl], s=10)
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    return fig


def main():
    args = SimpleNamespace(
        data_path="/opt/airflow/data/mobile-reviews.csv",
        experiment_name="Sentiment LR",
        registered_model_name="sentiment-logreg",
        tracking_uri=MLFLOW_TRACKING_URI,
        test_size=0.2,
        max_features=100,
        random_state=42,
    )
    
    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    # Load data
    df = pd.read_csv(args.data_path)
    X_text = df["review_text"].astype(str).tolist()
    y_raw = df["sentiment"].astype(str).values

    # Encode labels consistently
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = list(le.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    pipe = build_main_pipeline(max_features=args.max_features,)

    with mlflow.start_run() as run:
        run_id = run.info.run_id

        # Train
        pipe.fit(X_train, y_train)

        # Evaluate
        y_pred = pipe.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro")
        micro_f1 = f1_score(y_test, y_pred, average="micro")
        weighted_f1 = f1_score(y_test, y_pred, average="weighted")
        macro_precision = precision_score(y_test, y_pred, average="macro")
        macro_recall = recall_score(y_test, y_pred, average="macro")

        mlflow.log_params({
            "model_type": "LogisticRegression",
            "tfidf_max_features": args.max_features,
            "test_size": args.test_size,
            "random_state": args.random_state,
        })

        mlflow.log_metrics({
            "accuracy": acc,
            "macro_f1": macro_f1,
            "micro_f1": micro_f1,
            "weighted_f1": weighted_f1,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall
        })

        # Binary ROC-AUC if applicable
        try:
            if len(class_names) == 2:
                y_score = pipe.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, y_score)
                mlflow.log_metric("roc_auc", roc_auc)
        except Exception:
            pass

        # Confusion matrix (log as figure)
        fig_cm = plot_confusion_matrix(
            y_true=y_test,
            y_pred=y_pred,
            labels=list(range(len(class_names)))
        )
        mlflow.log_figure(fig_cm, "confusion_matrix.png")
        plt.close(fig_cm)

        # Decision boundary (2D projection, log as figure)
        tfidf = pipe.named_steps["tfidf"]
        X_all_sparse = tfidf.transform(X_train + X_test)
        y_all = np.concatenate([y_train, y_test])
        fig_db = plot_decision_boundary(
            model=LogisticRegression(solver="lbfgs", max_iter=1000),
            X_sparse=X_all_sparse,
            y=y_all,
            label_names=class_names,
            title="Decision Boundary (2D SVD projection)"
        )
        mlflow.log_figure(fig_db, "decision_boundary.png")
        plt.close(fig_db)

        # Classification report (log as JSON dict)
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        mlflow.log_dict(report, "classification_report.json")

        print(f"Run ID: {run_id}")
        print(f"Registered model name: {args.registered_model_name}")
        print("All artifacts logged to MLflow")


if __name__ == "__main__":
    main()

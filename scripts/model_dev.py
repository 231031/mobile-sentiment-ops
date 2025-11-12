import argparse
import json
import tempfile
from pathlib import Path

import mlflow
import mlflow.sklearn as mlflow_sklearn
from mlflow.models import infer_signature

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
)

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False


class Densifier(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X): return X.toarray() if hasattr(X, "toarray") else X


# -----------------------
# CLI
# -----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True, help="CSV path with columns: review_text, sentiment")
    p.add_argument("--experiment_name", default="Sentiment CLS")
    p.add_argument("--registered_model_name", default="sentiment")
    p.add_argument("--tracking_uri", default=None)
    p.add_argument("--test_size", type=float, default=0.5)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--max_features", type=int, default=300)
    return p.parse_args()


# -----------------------
# Figure
# -----------------------
def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    return fig


def plot_decision_boundary(model, X_sparse, y, label_names, title: str):
    """2D viz only (SVD inside for plotting)."""
    from sklearn.decomposition import TruncatedSVD
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
        ax.scatter(X_2d[y == lbl, 0], X_2d[y == lbl, 1], label=label_names[lbl], s=10)
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    return fig


# -----------------------
# Model Pipelines
# -----------------------
def build_pipelines(args):
    tfidf = TfidfVectorizer(
        lowercase=True, strip_accents="unicode", analyzer="word",
        ngram_range=(1, 2), max_features=args.max_features, min_df=2
    )

    lr = Pipeline([
        ("tfidf", tfidf),
        ("clf", LogisticRegression(
            penalty="l2", C=2.0, solver="lbfgs",
            max_iter=100, class_weight=None, n_jobs=-1))
    ])

    rf = Pipeline([
        ("tfidf", tfidf),
        ("to_dense", Densifier()),
        ("clf", RandomForestClassifier(
            n_estimators=100, max_depth=6, random_state=42, n_jobs=-1))
    ])

    models = {"lr": lr, "rf": rf}
    if XGB_AVAILABLE:
        xgb = Pipeline([
            ("tfidf", tfidf),
            ("to_dense", FunctionTransformer(lambda X: X.toarray(), accept_sparse=True)),
            ("clf", XGBClassifier(
                learning_rate=0.01, n_estimators=100,
                subsample=0.5, random_state=42, n_jobs=-1))
        ])
        models["xgb"] = xgb
    return models


# -----------------------
# Logging experiment
# -----------------------
def train_eval_log(model_key, pipe, X_train, y_train, X_val, y_val, class_names, args, label_encoder):
    model_names = {"lr": "LogisticRegression", "rf": "RandomForest", "xgb": "XGBoost"}
    run_name = model_names[model_key]
    registered_name = f"{args.registered_model_name}-{model_key}"

    with mlflow.start_run(run_name=run_name):
        # ---- Train
        pipe.fit(X_train, y_train)

        # ---- Predict & metrics
        y_pred = pipe.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        metrics = {
            "accuracy": float(acc),
            "macro_f1": float(f1_score(y_val, y_pred, average="macro")),
            "micro_f1": float(f1_score(y_val, y_pred, average="micro")),
            "weighted_f1": float(f1_score(y_val, y_pred, average="weighted")),
            "macro_precision": float(precision_score(y_val, y_pred, average="macro")),
            "macro_recall": float(recall_score(y_val, y_pred, average="macro"))
        }
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # Optional ROC-AUC for binary
        if len(class_names) == 2 and hasattr(pipe, "predict_proba"):
            try:
                y_score = pipe.predict_proba(X_val)[:, 1]
                mlflow.log_metric("roc_auc", float(roc_auc_score(y_val, y_score)))
            except Exception:
                pass

        # ---- Flat-file artifacts (no folders)
        # Confusion matrix
        fig_cm = plot_confusion_matrix(y_true=y_val, y_pred=y_pred, labels=list(range(len(class_names))))
        mlflow.log_figure(fig_cm, f"{model_key}__confusion_matrix.png")
        plt.close(fig_cm)

        # Decision boundary (uses TF-IDF transform for projection)
        try:
            tfidf = pipe.named_steps["tfidf"]
            X_all_sparse = tfidf.transform(pd.concat([X_train, X_val]).tolist())
            y_all = np.concatenate([y_train, y_val])

            # clone the pipeline's final estimator so the plot reflects the actual model type (LR/RF/XGB)
            plot_estimator = clone(pipe.named_steps["clf"]) if "clf" in pipe.named_steps else LogisticRegression(solver="lbfgs", max_iter=1000)

            fig_db = plot_decision_boundary(
                model=plot_estimator,
                X_sparse=X_all_sparse,
                y=y_all,
                label_names=class_names,
                title=f"Decision Boundary (2D SVD) – {run_name}"
            )
            mlflow.log_figure(fig_db, f"{model_key}__decision_boundary.png")
            plt.close(fig_db)
        except Exception as e:
            print(f"[warn] decision boundary plot failed: {e}")

        # Classification report
        report = classification_report(y_val, y_pred, target_names=class_names, output_dict=True)
        mlflow.log_dict(report, f"{model_key}__classification_report.json")

        # ---- Metadata as a single JSON (flat)
        meta = {
            "model_key": model_key,
            "model_name": run_name,
            "registered_model_name": registered_name,
            "tfidf": {
                "max_features": pipe.named_steps["tfidf"].max_features,
                "ngram_range": pipe.named_steps["tfidf"].ngram_range,
                "min_df": pipe.named_steps["tfidf"].min_df
            },
            "params_specific": {},
            "data": {
                "test_size": args.test_size,
                "random_state": args.random_state
            },
            "labels": {
                "id_to_label": {int(i): str(lbl) for i, lbl in enumerate(class_names)},
                "label_to_id": {str(lbl): int(i) for i, lbl in enumerate(class_names)}
            },
            "metrics": metrics
        }
        if model_key == "rf":
            clf = pipe.named_steps["clf"]
            meta["params_specific"] = {
                "n_estimators": clf.n_estimators,
                "max_depth": clf.max_depth,
                "max_features": clf.max_features
            }
        elif model_key == "xgb" and XGB_AVAILABLE:
            clf = pipe.named_steps["clf"]
            meta["params_specific"] = {
                "learning_rate": float(clf.learning_rate),
                "n_estimators": int(clf.n_estimators),
                "subsample": float(clf.subsample)
            }

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / f"{model_key}__metadata.json"
            p.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
            mlflow.log_artifact(p)

            # Also persist the label encoder mapping for serving
            le_map = {"classes_": label_encoder.classes_.tolist()}
            p_le = Path(td) / f"{model_key}__label_encoder.json"
            p_le.write_text(json.dumps(le_map, indent=2, ensure_ascii=False))
            mlflow.log_artifact(p_le)

        # ---- Log model (per-run) + optional registration
        input_example = pd.DataFrame({"review_text": ["great phone", "แบตอึดมาก"]})
        signature = infer_signature(model_input=input_example)
        mlflow_sklearn.log_model(
            sk_model=pipe,
            name=f"{model_key}_model",
            registered_model_name=registered_name,
            input_example=input_example,
            signature=signature
        )

        print(f"[{run_name}] run_id={mlflow.active_run().info.run_id} | registered_name={registered_name}")


def main():
    args = parse_args()
    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    # Load & clean data
    df = pd.read_csv(args.data_path).dropna(subset=["review_text", "sentiment"]).reset_index(drop=True)

    # Encode labels
    le = LabelEncoder()
    df["sentiment"] = le.fit_transform(df["sentiment"])
    class_names = list(le.classes_)

    # Split
    train_df, val_df = train_test_split(
        df, test_size=args.test_size, stratify=df["sentiment"], random_state=args.random_state
    )

    # Build pipelines
    pipelines = build_pipelines(args)

    # ---- One run per model (no parent/nested runs)
    order = ["lr", "rf"] + (["xgb"] if XGB_AVAILABLE else [])
    for key in order:
        train_eval_log(
            model_key=key,
            pipe=pipelines[key],
            X_train=train_df['review_text'], y_train=train_df['sentiment'],
            X_val=val_df['review_text'], y_val=val_df['sentiment'],
            class_names=class_names,
            args=args,
            label_encoder=le
        )

    print("Logged artifacts to mlflow!")


if __name__ == "__main__":
    main()

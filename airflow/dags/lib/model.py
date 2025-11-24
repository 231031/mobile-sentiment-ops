import pandas as pd
import json
import tempfile
from pathlib import Path

import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn as mlflow_sklearn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False


def build_pipelines(max_features):
    tfidf = TfidfVectorizer(
        lowercase=True, strip_accents="unicode", analyzer="word",
        ngram_range=(1, 2), max_features=max_features, min_df=2
    )

    lr = Pipeline([
        ("tfidf", tfidf),
        ("clf", LogisticRegression(
            penalty="l2", C=2.0, solver="lbfgs",
            max_iter=100, class_weight=None, n_jobs=-1))
    ])

    rf = Pipeline([
        ("tfidf", tfidf),
        ("clf", RandomForestClassifier(
            n_estimators=10, max_depth=3, random_state=42, n_jobs=-1))
    ])

    models = {"lr": lr, "rf": rf}
    if XGB_AVAILABLE:
        xgb = Pipeline([
            ("tfidf", tfidf),
            ("clf", XGBClassifier(
                learning_rate=0.01, n_estimators=10,
                max_depth=3, subsample=0.7, 
                random_state=42, n_jobs=-1))
        ])
        models["xgb"] = xgb
    return models


def log_model_info(model_name, model_key, pipe, run_name, registered_name,
                   class_names, metrics, args):
    meta = {
        "model_name": model_name,
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
    elif model_key == "xgb":
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
        le_map = {"classes_": class_names}
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

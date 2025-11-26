import pandas as pd
import json
import tempfile
from pathlib import Path
import os
import requests

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
import mlflow.sklearn as mlflow_sklearn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

def build_pipelines(max_features):
    """Builds classification pipelines with TF-IDF and 3 different classifiers."""
    tfidf = TfidfVectorizer(
        lowercase=True, strip_accents="unicode", analyzer="word",
        ngram_range=(1, 2), max_features=max_features, min_df=2
    )

    nb = Pipeline([
        ("tfidf", tfidf),
        ("clf", MultinomialNB(alpha=4.0, fit_prior=True))
    ])

    rf = Pipeline([
        ("tfidf", tfidf),
        ("clf", RandomForestClassifier(
            n_estimators=10, max_depth=3, random_state=42, n_jobs=-1))
    ])

    models = {"nb": nb, "rf": rf}
    if XGB_AVAILABLE:
        xgb = Pipeline([
            ("tfidf", tfidf),
            ("clf", XGBClassifier(
                objective="multi:softprob",
                learning_rate=0.01, n_estimators=10,
                max_depth=6, subsample=0.5, 
                random_state=42, n_jobs=-1))
        ])
        models["xgb"] = xgb
    return models


def log_model_info(model_name, model_key, pipe, run_name, registered_name,
                   class_names, metrics, args):
    """save model metadata and log to mlflow artifacts"""
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
        mlflow.log_artifact(p, artifact_path="model")

        # Also persist the label encoder mapping for serving
        le_map = {"classes_": class_names}
        p_le = Path(td) / f"{model_key}__label_encoder.json"
        p_le.write_text(json.dumps(le_map, indent=2, ensure_ascii=False))
        mlflow.log_artifact(p_le, artifact_path="model")

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


def promote_best_model(experiment_name: str = "Sentiment CLS", alias: str = "Production"):
    """
    Fetch eval metrics from all runs in an experiment, identify the best model by macro_f1,
    and promote it to the given alias (default 'Production').
    """
    client = MlflowClient()
    
    # Get experiment by name
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        return {
            "promoted": False,
            "message": f"Experiment '{experiment_name}' not found."
        }
    
    # Search all runs in the experiment (limit to recent runs or all if necessary)
    runs = client.search_runs(experiment_ids=[experiment.experiment_id], max_results=1000)
    
    if not runs:
        return {
            "promoted": False,
            "message": f"No runs found in experiment '{experiment_name}'."
        }
    
    # Find best run by macro_f1 metric
    best_run = None
    best_macro_f1 = -1
    
    for run in runs:
        macro_f1 = run.data.metrics.get("macro_f1", -1)
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_run = run
    
    if not best_run:
        return {
            "promoted": False,
            "message": "No runs with macro_f1 metric found."
        }
    
    print(f"Best run found: {best_run.info.run_id} with macro_f1={best_macro_f1}")
    
    # Find registered model version(s) for the best run
    versions = client.search_model_versions(f"run_id='{best_run.info.run_id}'")
    
    if not versions:
        return {
            "promoted": False,
            "best_run_id": best_run.info.run_id,
            "best_macro_f1": best_macro_f1,
            "message": f"No registered model version found for best run {best_run.info.run_id}."
        }
    
    best_version = versions[0]
    model_name = best_version.name
    version_number = best_version.version
    
    try:
        # Check if a current production alias exists
        try:
            current_prod = client.get_model_version_by_alias(model_name, alias)
            # If same version, no promotion needed
            if current_prod.version == version_number:
                return {
                    "promoted": False,
                    "best_model_name": model_name,
                    "best_version": version_number,
                    "best_run_id": best_run.info.run_id,
                    "best_macro_f1": best_macro_f1,
                    "message": f"Model {model_name} v{version_number} is already at @{alias}."
                }
            # Demote current production to Staging (best-effort)
            try:
                client.set_registered_model_alias(model_name, "Staging", current_prod.version)
                print(f"Demoted {model_name} v{current_prod.version} to @Staging")
            except Exception as e:
                print(f"[warn] Could not demote current production: {e}")
        except Exception:
            # No current alias for this model, proceed with promotion
            pass
        
        # Promote best model to alias
        client.set_registered_model_alias(model_name, alias, version_number)
        
        return {
            "promoted": True,
            "best_model_name": model_name,
            "best_version": version_number,
            "best_run_id": best_run.info.run_id,
            "best_macro_f1": best_macro_f1,
            "message": f"Successfully promoted {model_name} v{version_number} to @{alias} (macro_f1={best_macro_f1})"
        }
    
    except Exception as e:
        return {
            "promoted": False,
            "best_model_name": model_name,
            "best_version": version_number,
            "best_run_id": best_run.info.run_id,
            "best_macro_f1": best_macro_f1,
            "message": f"Promotion failed: {e}"
        }
    
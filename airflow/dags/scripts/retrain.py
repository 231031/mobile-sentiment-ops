import argparse
import os
import json
import requests

import mlflow
from mlflow.tracking import MlflowClient

from train_model import prepare_dataset
import lib.model as model_module
from lib.artifacts import evaluate_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=False, default="/opt/airflow/data/mobile-reviews.csv")
    p.add_argument("--experiment_name", default="Sentiment CLS")
    p.add_argument("--registered_model_name", default="sentiment")
    p.add_argument("--tracking_uri", default=os.getenv("MLFLOW_TRACKING_URI"))
    p.add_argument("--backend_url", default=os.getenv("BACKEND_URL"))
    p.add_argument("--test_size", type=float, default=0.7)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--max_features", type=int, default=100)
    p.add_argument("--promote", action="store_true", help="Promote best new model to production alias if better")
    p.add_argument("--alias", default="Production", help="Alias to treat as production (case-sensitive)")
    return p.parse_args()


def find_any_production_model(client: MlflowClient, alias: str):
    """Return a tuple (registered_model_name, ModelVersion) or (None, None)"""
    for rm in client.search_registered_models():
        name = rm.name
        try:
            mv = client.get_model_version_by_alias(name, alias)
            return name, mv
        except Exception:
            continue
    return None, None


def main():
    args = parse_args()
    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
    else:
        print("[warn] No tracking URI supplied; defaulting to local ./mlruns store.")

    mlflow.set_experiment(args.experiment_name)
    client = MlflowClient()

    # Prepare dataset using helper from train_model.py
    train_df, val_df, class_names = prepare_dataset(args)

    pipelines = model_module.build_pipelines(args.max_features)
    order = [k for k in pipelines.keys()]

    best_new_score = -1
    best_new_run_id = None
    best_model_key = None

    # Train candidate models and log to mlflow
    for key in order:
        model_names = {"nb": "NaiveBayes", "rf": "RandomForest", "xgb": "XGBoost"}
        run_name = model_names.get(key, key)
        registered_name = f"{args.registered_model_name}-{key}"

        with mlflow.start_run(run_name=run_name):
            metrics = evaluate_model(pipelines[key], model_names.get(key, key),
                                     train_df['review_text'], train_df['sentiment'],
                                     X_val=val_df['review_text'], y_val=val_df['sentiment'])

            # log metadata and register model
            model_module.log_model_info(
                model_name=model_names.get(key, key),
                model_key=key,
                pipe=pipelines[key],
                run_name=run_name,
                registered_name=registered_name,
                class_names=class_names,
                metrics=metrics,
                args=args,
            )

            run_id = mlflow.active_run().info.run_id
            score = metrics.get("macro_f1", 0)
            print(f"Trained {key} (run_id={run_id}) macro_f1={score}")

            if score > best_new_score:
                best_new_score = score
                best_new_run_id = run_id
                best_model_key = key

    summary = {
        "best_model_key": best_model_key,
        "best_new_run_id": best_new_run_id,
        "best_new_score": best_new_score,
        "promoted": False,
        "promotion_context": None,
    }

    # Compare with existing production alias (if any)
    try:
        prod_name, prod_mv = find_any_production_model(client, args.alias)
        if prod_mv:
            prod_run = client.get_run(prod_mv.run_id)
            prod_score = prod_run.data.metrics.get("macro_f1", 0)
            summary["current_production"] = {"name": prod_name, "version": prod_mv.version, "run_id": prod_mv.run_id, "score": prod_score}
            print(f"Found production model {prod_name} v{prod_mv.version} (score={prod_score})")

            # Promote if requested and better
            if args.promote and best_new_score > prod_score:
                # demote current production to Staging (best-effort)
                try:
                    client.set_registered_model_alias(prod_name, "Staging", prod_mv.version)
                except Exception:
                    pass
                
                # find version for our best run
                versions = client.search_model_versions(f"run_id='{best_new_run_id}'")
                if versions:
                    new_v = versions[0]
                    client.set_registered_model_alias(new_v.name, args.alias, new_v.version)
                    summary["promoted"] = True
                    summary["promotion_context"] = f"Promoted {new_v.name} v{new_v.version} over {prod_name} v{prod_mv.version}"
                    print(summary["promotion_context"])


                    try:
                        response = requests.get(args.backend_url, timeout=3)
                        print(f"Retrain triggered. Status Code: {response.status_code}")
                    except requests.exceptions.RequestException as req_err:
                        print(f"Failed to trigger retrain: {req_err}")

                else:
                    summary["promotion_context"] = "Could not find registered version for best run; not promoted"
                

            else:
                summary["promotion_context"] = "Not promoted (either not requested or no improvement)"
        else:
            # No production model exists; optionally promote the new best
            print("No existing production alias found.")
            versions = client.search_model_versions(f"run_id='{best_new_run_id}'")
            if args.promote and versions:
                new_v = versions[0]
                client.set_registered_model_alias(new_v.name, args.alias, new_v.version)
                summary["promoted"] = True
                summary["promotion_context"] = f"First Production Model: {new_v.name} v{new_v.version}"
            else:
                summary["promotion_context"] = "Not promoted (use --promote to enable promotion)"

    except Exception as e:
        summary["promotion_context"] = f"Production check/promotion failed: {e}"

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

import os
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import mlflow
from types import SimpleNamespace

from lib.train_with_dags import build_pipelines, train_eval_log
from app.config import *
from app.data_pipeline import DataHandler

class MLOpsHandler:
    def __init__(self):
        self.dataHandler = DataHandler()

    def train_startup_model(self):
        print("System Startup: No Production model found.")
        print(f"Initializing Bootstrap Training from {INITIAL_DATA_PATH}...")

        if os.path.exists(INITIAL_DATA_PATH):
            df = pd.read_csv(INITIAL_DATA_PATH)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save Bootstrap Reference
            # We call _upload_safe to handle the emulator 404 quirks
            self.dataHandler._upload_safe(f"data_label/labeled_BOOTSTRAP_{timestamp}.csv", df[[REVIEW_COLUMN, TARGET_COULUM]].to_csv(index=False), 'text/csv')

            # EDA artifact (logs to MLflow under artifact path "eda")
            try:
                # Ensure the experiment exists before starting the run (default experiment may be missing in fresh mlflow server)
                mlflow.set_experiment("Sentiment_Analysis_Production")
                with mlflow.start_run(run_name=f"EDA_bootstrap_{timestamp}"):
                    self.dataHandler.run_full_eda(
                        df=df,
                        label_column=TARGET_COULUM,
                        report_prefix=f"bootstrap_{timestamp}"
                    )
            except Exception as exc:
                print(f"⚠️ Skipping EDA logging due to error: {exc}")
            
            # PREPARE AND TRAIN
            df = df.dropna(subset=[REVIEW_COLUMN, TARGET_COULUM]).reset_index(drop=True)
            le = LabelEncoder()
            df[TARGET_COULUM] = le.fit_transform(df[TARGET_COULUM])
            class_names = list(le.classes_)

            args = SimpleNamespace(
                max_features=300,
                registered_model_name=MODEL_NAME,
                test_size=0.5,
                random_state=42,
            )
                

            train_df, val_df = train_test_split(
                df, test_size=args.test_size, stratify=df[TARGET_COULUM], random_state=args.random_state
            )

            mlflow.set_experiment("Sentiment_Analysis_Production")
            pipelines = build_pipelines(args)
            order = ["lr", "rf"] + (["xgb"] if XGB_AVAILABLE else [])
            
            best_score = -1
            best_run_id = None

            for key in order:
                train_eval_log(
                    model_key=key, pipe=pipelines[key],
                    X_train=train_df[REVIEW_COLUMN], y_train=train_df[TARGET_COULUM],
                    X_val=val_df[REVIEW_COLUMN], y_val=val_df[TARGET_COULUM],
                    class_names=class_names, args=args, label_encoder=le
                )
                last_run = mlflow.search_runs(max_results=1, order_by=["start_time DESC"]).iloc[0]
                if last_run['metrics.accuracy'] > best_score:
                    best_score = last_run['metrics.accuracy']
                    best_run_id = last_run.run_id

            if best_run_id:
                print(f"Promoting Bootstrap Model (Run ID: {best_run_id}, Acc: {best_score:.4f})")
                versions = client.search_model_versions(f"run_id='{best_run_id}'")
                if versions:
                    best_version = versions[0]
                    client.set_registered_model_alias(best_version.name, "Production", best_version.version)
                    print(f"Successfully aliased '{best_version.name}' (v{best_version.version}) as @Production")
                else:
                    print(f"Warning: No registered model found for Run ID {best_run_id}. Attempting Fallback...")

        else:
            print("Error: Initial data not found.")


    def train_model(self):
        training_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"--- Retraining Sequence ID: {training_id} ---")
        mlflow.set_experiment("Sentiment_Analysis_Production")
        
        # 1. Get Data (Prefer latest labeled data, fallback to initial)
        df = self.dataHandler.get_lastest_file(prefix="data_label/labeled_")
        if df.empty:
            print("No new labeled data found. Using Initial Data.")
            if os.path.exists(INITIAL_DATA_PATH):
                df = pd.read_csv(INITIAL_DATA_PATH)
            else:
                return {"status": "failed", "reason": "No data available"}
        else:
             print(f"Using latest labeled data.")

        # 2. Prepare Data
        df = df.dropna(subset=[REVIEW_COLUMN, TARGET_COULUM]).reset_index(drop=True)
        le = LabelEncoder()
        df[TARGET_COULUM] = le.fit_transform(df[TARGET_COULUM])
        class_names = list(le.classes_)

        args = SimpleNamespace(
            max_features=300,
            registered_model_name=MODEL_NAME,
            test_size=0.2, 
            random_state=42,
        )

        train_df, val_df = train_test_split(
            df, test_size=args.test_size, stratify=df[TARGET_COULUM], random_state=args.random_state
        )

        # 3. Train Candidates
        pipelines = build_pipelines(args)
        order = ["lr", "rf"] + (["xgb"] if XGB_AVAILABLE else [])
        
        best_new_score = -1
        best_new_run_id = None
        best_model_key = None

        for key in order:
            run_id, metrics = train_eval_log(
                model_key=key, pipe=pipelines[key],
                X_train=train_df[REVIEW_COLUMN], y_train=train_df[TARGET_COULUM],
                X_val=val_df[REVIEW_COLUMN], y_val=val_df[TARGET_COULUM],
                class_names=class_names, args=args, label_encoder=le
            )
            # Use macro_f1 as the primary metric
            score = metrics.get("macro_f1", 0)
            if score > best_new_score:
                best_new_score = score
                best_new_run_id = run_id
                best_model_key = key

        print(f"Best New Model: {best_model_key} (Run: {best_new_run_id}, Score: {best_new_score:.4f})")

        # 4. Compare with Production
        promoted = False
        promotion_context = "No improvement"
        
        try:
            # Check if Production alias exists
            prod_model = client.get_model_version_by_alias(MODEL_NAME, "Production")
            prod_run = client.get_run(prod_model.run_id)
            prod_score = prod_run.data.metrics.get("macro_f1", 0)
            print(f"Current Production Score: {prod_score:.4f}")
            
            if best_new_score > prod_score:
                print("🚀 New model is better! Promoting...")
                # Demote old Production to Staging
                client.set_registered_model_alias(MODEL_NAME, "Staging", prod_model.version)
                # Promote new to Production
                versions = client.search_model_versions(f"run_id='{best_new_run_id}'")
                if versions:
                    client.set_registered_model_alias(MODEL_NAME, "Production", versions[0].version)
                    promoted = True
                    promotion_context = f"Promoted v{versions[0].version} (Score: {best_new_score:.4f}) over v{prod_model.version} (Score: {prod_score:.4f})"
            else:
                print("New model is not better. Keeping current Production.")
                promotion_context = f"Kept v{prod_model.version} (Score: {prod_score:.4f}) >= New (Score: {best_new_score:.4f})"

        except Exception as e:
            # No production model exists, or error fetching it
            print(f"Production model check failed ({e}). Promoting new model as first Production.")
            versions = client.search_model_versions(f"run_id='{best_new_run_id}'")
            if versions:
                client.set_registered_model_alias(MODEL_NAME, "Production", versions[0].version)
                promoted = True
                promotion_context = "First Production Model"

        return {
            "status": "success",
            "training_id": training_id,
            "best_run_id": best_new_run_id,
            "best_score": best_new_score,
            "promoted": promoted,
            "promotion_context": promotion_context
        }

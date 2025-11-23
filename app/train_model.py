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
        
        # data from use for retrain

        # 2. Archive the Fresh Data Snapshot (For Record Keeping)
        # We save this specific export as a CSV to data_label
        
        # archive_name = f"data_label/labeled_{training_id}.csv"
        # self.dataHandler._upload_safe(archive_name, df_fresh[[REVIEW_COLUMN, TARGET_COULUM]].to_csv(index=False), 'text/csv')

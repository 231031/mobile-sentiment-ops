import json

import mlflow
from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset

from app.config import *
from app.data_pipeline import *

def map_model_type(model_type):
    if model_type == "XGBoost":
        return "xgb"
    elif model_type == "RandomForest":
        return "rf"
    else:
        return "lr"

class PredictionHandler:
    def __init__(self):
        self.dataHandler = DataHandler()
        self.production_model = None
        self.id_to_label = None
        self.metrics = {}
        
    def find_any_production_model(self, alias: str = ALIAS) -> str | None:
        """
        Returns a model URI like 'models:/SomeName@Production'
        for the first registered model that has the given alias.
        """
        for rm in client.search_registered_models():   # iterate over all model names
            name = rm.name
            try:
                mv = client.get_model_version_by_alias(name, alias)
                run = client.get_run(mv.run_id)

                # if alias doesn't exist for this model, this raises -> we skip
                print(f"Found model with @{alias}: {name} (version {mv.version})")

                # Fetch run metrics
                try:
                    self.metrics = run.data.metrics
                    print(f"Loaded metrics: {self.metrics}")
                except Exception as e:
                    print(f"Could not fetch metrics for run {mv.run_id}: {e}")
                    self.metrics = {}

                # get label encoder
                try: 
                    model_type = run.data.tags.get("mlflow.runName")
                    model_key = map_model_type(model_type)
                    le_artifact = f"{model_key}__label_encoder.json"
                     
                    # download the label encoder artifact from the run
                    local_path = client.download_artifacts(mv.run_id, le_artifact, "/backend/temp")
                    with open(local_path, "r", encoding="utf-8") as f:
                        le_payload = json.load(f)

                    classes = le_payload.get("classes_", [])
                except Exception as e:
                    print(f"Could not fetch label encoder : {e}")

                if classes:
                    # classes_ is in encoded order, so index == encoded id
                    self.id_to_label = {int(i): str(lbl) for i, lbl in enumerate(classes)}
                    print(f"Loaded label map: {self.id_to_label}")
                else:
                    print("No classes_ in label encoder artifact")
                    self.id_to_label = None

                return f"models:/{name}@{alias}"
            except Exception:
                continue
        return None

    def refresh_production_model(self, alias: str = ALIAS) -> str | None:
        uri = self.find_any_production_model(alias=alias)
        if not uri:
            return None
        self.production_model = mlflow.sklearn.load_model(uri)
        return uri
    
    def check_data_drift(self, ref_df, cur_df, request_id):
        html_path = REPORTS_DIR / f"drift_{request_id}.html"
        json_path = REPORTS_DIR / f"drift_{request_id}.json"


        definition = DataDefinition(text_columns=[REVIEW_COLUMN])
        ref_data = Dataset.from_pandas(ref_df,data_definition=definition)
        cur_data = Dataset.from_pandas(cur_df,data_definition=definition)

        report = Report([
            DataDriftPreset(), 
        ])
        drift_eval = report.run(
            reference_data=ref_data, 
            current_data=cur_data,
        )
        drift_eval.save_html(str(html_path))
        drift_eval.save_json(str(json_path))

        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        self.dataHandler._upload_safe(
            f"reports/drift_{request_id}.html",
            html_content,
            'text/html',
        )

        with open(json_path, "r", encoding="utf-8") as f:
            json_content = f.read()
        self.dataHandler._upload_safe(
            f"reports/drift_{request_id}.json",
            json_content,
            'text/json',
        )

        rep_dict = drift_eval.dict()
        metrics = rep_dict.get("metrics", [])
        if metrics:
            drift_share = metrics[0]["config"]["drift_share"]
        else:
            print(f"can not get metrics from datadrift")
            drift_share = 0.0
        return drift_share


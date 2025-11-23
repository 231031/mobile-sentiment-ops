import pandas as pd
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Response
import mlflow
from contextlib import asynccontextmanager

from app.prediction import PredictionHandler
from app.train_model import MLOpsHandler
from app.data_pipeline import DataHandler
from app.config import *

mlops = MLOpsHandler()
predictHandler = PredictionHandler()
dataHandler = DataHandler()


def _retrain_background_job():
    try:
        summary = mlops.train_model()
        if summary.get("promoted"):
            refreshed = predictHandler.refresh_production_model()
            print(f"Retraining finished and Production updated: {refreshed}")
        else:
            print(f"Retraining finished without promotion: {summary.get('promotion_context')}")
    except Exception as exc:
        print(f"Retraining failed: {exc}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP
    print("🔄 LIFESPAN: Starting up...")
    
    # 1. Ensure MLflow state is clean
    if mlflow.active_run():
        mlflow.end_run()

    prod_uri = predictHandler.find_any_production_model()
    try:
        if not prod_uri:
            raise ValueError("No Production model found")
        predictHandler.production_model = mlflow.sklearn.load_model(prod_uri)
        print("✅ System Startup: Existing Production model loaded.")
    except Exception:
        print("❌ System Not Startup: Cannot find any Production Model")
        mlops.train_startup_model()
        prod_uri = predictHandler.find_any_production_model()
        if prod_uri:
            predictHandler.production_model = mlflow.sklearn.load_model(prod_uri)
            print("the Production model is loaded")
        else:
            print("Critical: Failed to load Production model even after startup training.")
    
    yield # App runs here
    print("🛑 LIFESPAN: Shutting down...")

app = FastAPI(lifespan=lifespan)

# --- Endpoints ---
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    request_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    df_original = pd.read_csv(file.file)
    df_input = df_original.copy()

    try:
        if predictHandler.production_model is None:
            print("the Production model to use in prediction not found")
            df_input[TARGET_COULUM] = "model_not_found"
        else:
            y_pred = predictHandler.production_model.predict(df_input[REVIEW_COLUMN])

            # 🔁 map numeric ids back to original labels if we have the mapping
            if predictHandler.id_to_label:
                df_input[TARGET_COULUM] = [
                    predictHandler.id_to_label.get(int(p), str(p)) for p in y_pred
                ]
            else:
                # fallback: keep numeric predictions
                df_input[TARGET_COULUM] = y_pred
    except Exception as e:
        print(f"Prediction error: {e}")
        df_input[TARGET_COULUM] = "model_error"

    drift_detected = False
    try:
        # Use latest labeled file as reference
        ref_df = dataHandler.get_lastest_file(prefix="data_label/labeled_")

        if not ref_df.empty:
            # specify some retrain conditions
            drift_share = predictHandler.check_data_drift(
                ref_df=ref_df[[REVIEW_COLUMN]],
                cur_df=df_original[[REVIEW_COLUMN]],
                request_id=request_id
            )
            if drift_share > 0.5:
                print(f"data drift is more than threshold - wait for data is labeled : {drift_share}")
                drift_detected = True
                FastAPI.post("/retrain")
            else:
                print(f"data drift is not more than threshold - use the same model : {drift_share}")

    except Exception as e:
        print(f"Drift check error: {e}")

    csv_str = df_input[[REVIEW_COLUMN, TARGET_COULUM]].to_csv(index=False)
    dataHandler._upload_safe(
        f"data_prediction/predicted_{request_id}.csv",
        csv_str, 'text/csv'
    )
    return Response(
        content=csv_str,
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="predicted_{request_id}.csv"',
            "X-Drift-Detected": str(drift_detected).lower(),
            "X-Request-ID": request_id,
        },
    )

@app.post("/retrain")
async def trigger_retrain(background_tasks: BackgroundTasks):
    background_tasks.add_task(_retrain_background_job)
    return {"status": "Retraining scheduled"}
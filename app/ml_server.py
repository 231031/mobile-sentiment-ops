import pandas as pd
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import mlflow
from contextlib import asynccontextmanager

from app.prediction import PredictionHandler
from app.data_pipeline import DataHandler
from app.config import *

predictHandler = PredictionHandler()
dataHandler = DataHandler()
background_tasks: BackgroundTasks

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
        print("System Startup: Existing Production model loaded.")
    except Exception:
        print("System : Cannot find any Production Model")
    
    yield

    print("🛑 LIFESPAN: Shutting down...")

app = FastAPI(lifespan=lifespan)

from fastapi.staticfiles import StaticFiles
import os

# Ensure report directory exists
os.makedirs("report", exist_ok=True)

# Mount the report directory to serve static files
app.mount("/reports", StaticFiles(directory="report"), name="reports")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    text: str
    prediction: str
    confidence: float
    probabilities: Dict[str, float]

@app.post("/predict_json", response_model=PredictionResponse)
async def predict_json(request: TextRequest):
    # Prepare data
    df_input = pd.DataFrame([request.text], columns=[REVIEW_COLUMN])
    
    if predictHandler.production_model is None:
        return {
            "text": request.text,
            "prediction": "model_not_found",
            "confidence": 0.0,
            "probabilities": {}
        }

    # Predict
    y_pred = predictHandler.production_model.predict(df_input[REVIEW_COLUMN])
    
    # Try to get probabilities if supported
    try:
        y_proba = predictHandler.production_model.predict_proba(df_input[REVIEW_COLUMN])
        max_proba = float(y_proba[0].max())
        
        probs = {}
        if predictHandler.id_to_label:
            for i, prob in enumerate(y_proba[0]):
                label = predictHandler.id_to_label.get(i, str(i))
                probs[label] = float(prob)
        else:
             for i, prob in enumerate(y_proba[0]):
                probs[str(i)] = float(prob)
                
    except AttributeError:
        max_proba = 1.0 # Fallback
        probs = {}

    # Map prediction to label
    predicted_id = int(y_pred[0])
    predicted_label = predictHandler.id_to_label.get(predicted_id, str(predicted_id)) if predictHandler.id_to_label else str(predicted_id)

    return {
        "text": request.text,
        "prediction": predicted_label,
        "confidence": max_proba,
        "probabilities": probs
    }

@app.get("/model/metrics")
async def get_metrics():
    if predictHandler.metrics:
        return predictHandler.metrics
    return {"error": "No metrics available"}

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

@app.get("/loadmodel")
async def trigger_retrain():
    prod_uri = predictHandler.find_any_production_model()
    try:
        if not prod_uri:
            raise ValueError("No Production model found")
        predictHandler.production_model = mlflow.sklearn.load_model(prod_uri)
        print("System : Load new model")
    except Exception:
        print("System : Cannot find any Production Model")
    return {"status": 200}

@app.get("/healthcheck")
async def healthcheck():
    return {"status": 200}
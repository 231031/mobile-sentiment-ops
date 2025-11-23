import os
from pathlib import Path
from google.cloud import storage
from google.auth.credentials import AnonymousCredentials
from google.api_core.client_options import ClientOptions
from mlflow.tracking import MlflowClient
import mlflow

GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "mobile-reviews-bucket")
GCS_ENDPOINT = os.getenv("GCS_ENDPOINT", "http://gcs-emulator:4443")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:8000")
INITIAL_DATA_PATH = os.getenv("INITIAL_DATA_PATH", "/backend/data/mobile-reviews.csv")
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", "/backend/report"))
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
Path("/backend/temp").mkdir(parents=True, exist_ok=True)
MODEL_NAME = "MobileSentimentModel"
ALIAS = "Production"
XGB_AVAILABLE=True
FORCE_RETRAIN_ON_PREDICT = os.getenv("FORCE_RETRAIN_ON_PREDICT", "false").lower() == "true"

REVIEW_COLUMN = 'review_text'
TARGET_COULUM = 'sentiment'

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

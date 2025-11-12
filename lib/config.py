from dotenv import load_dotenv
from pathlib import Path
import os
import glob

HERE = Path(__file__).resolve().parent           # /opt/airflow/lib/model_registry
ENV_PATH = HERE / ".env" 
env_files = glob.glob(os.path.join(ENV_PATH))

if env_files:
    # Load the first .env file found
    print(f"Loading environment variables from: {env_files[0]}")
    load_dotenv(dotenv_path=env_files[0])
else:
    print(f"No .env file found. from {__name__}")
# load_dotenv(dotenv_path="example.env")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
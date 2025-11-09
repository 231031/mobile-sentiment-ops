from datetime import datetime, timedelta
import os
import pendulum

from airflow.operators.python import PythonVirtualenvOperator
from airflow import DAG
import dotenv
dotenv_path = dotenv.find_dotenv()
if dotenv_path:
    print(f"Found .env file at {dotenv_path}")
    dotenv.load_dotenv(dotenv_path)

default_args = {
    "owner":           "myself",
    "depends_on_past": False,
    "retries":         2,
    "retry_delay":     timedelta(minutes=10),
}

local_tz = pendulum.timezone("Asia/Bangkok")

with DAG(
    dag_id              = "dags_test",
    start_date          = pendulum.datetime(2025, 11, 1, tz=local_tz),
    schedule            = "0 * * * *",
    catchup             = False,
    default_args        = default_args,
    max_active_runs     = 1,
    tags                = ["sentiment", "nlp"],
): 

    def compare_models():
        import sys
        sys.path.append("/opt/airflow/lib")
        from lib.lr_demo_dags import main
        main()
    
    compare_model_task = PythonVirtualenvOperator(
        task_id="model_comparison",
        python_callable=compare_models,
        requirements=["mlflow","pandas","scikit-learn","numpy","matplotlib","seaborn"],
        python_version="3.10",
    )

    compare_model_task
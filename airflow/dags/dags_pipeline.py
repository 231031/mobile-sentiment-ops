from datetime import datetime
from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.empty import EmptyOperator

# Put these where you like in your Airflow directory:
DATA_DIR   = "/opt/airflow/data"
TMP_DIR    = "/opt/airflow/tmp"
DATASET_ID = "mohankrishnathalla/mobile-reviews-sentiment-and-specification"
ADDED_DATASET = "abhi8923shriv/sentiment-analysis-dataset"

with DAG(
    dag_id="mobile_sentiment",
    start_date=datetime(2025, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["kaggle", "mlops"],
) as dag:
    dependencies = BashOperator(
        task_id="python_dependencies",
        bash_command="""
        pip install kaggle
        """
    )

    fetch_mobile_reviews = BashOperator(
        task_id="fetch_mobile_reviews",
        bash_command=f"""
        set -euo pipefail

        # 1) Guarantee our dirs exist
        mkdir -p "{DATA_DIR}" "{TMP_DIR}"
        cd "{TMP_DIR}"

        # 2) Download via Kaggle CLI (always lands here)
        kaggle datasets download -d {DATASET_ID} --force
        kaggle datasets download -d {ADDED_DATASET} --force

        echo "Listing files in tmp before unzip:"
        ls -la

        unzip -o "{TMP_DIR}/mobile-reviews-sentiment-and-specification.zip" -d {TMP_DIR}
        unzip -o "{TMP_DIR}/sentiment-analysis-dataset.zip" -d {TMP_DIR}

        echo "Listing files in tmp after unzip:"
        ls -la

        # 4) Move to the final location. Try the expected filename first,
        mv "{TMP_DIR}/Mobile Reviews Sentiment.csv" "{DATA_DIR}/mobile-reviews.csv"
        mv "{TMP_DIR}/train.csv" "{DATA_DIR}/train_added.csv"
        mv "{TMP_DIR}/test.csv" "{DATA_DIR}/test_added.csv"

        rm -rf "{TMP_DIR}"
        """,
    )
    
    eda = BashOperator(
        task_id="eda",
        bash_command=f"""
        python /opt/airflow/dags/scripts/eda.py --data_path "{DATA_DIR}" --mlflow_run_name "EDA"
        """
    )

    model_pipeline = BashOperator(
        task_id="model_pipeline",
        bash_command=f"""
        echo "Starting model pipeline with data from {DATA_DIR}"
        python /opt/airflow/dags/scripts/train_model.py --data_path "{DATA_DIR}"
        """,
    )
    
    retrain = BashOperator(
        task_id="retraining",
        bash_command=f"""
        echo "Running retraining pipeline..."
        python /opt/airflow/dags/scripts/retrain.py --data_path "{DATA_DIR}" --promote
        """,
    )

    end = EmptyOperator(task_id="end")

dependencies >> fetch_mobile_reviews >> [eda, model_pipeline] >> retrain >> end

from datetime import datetime
from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.empty import EmptyOperator

# Put these where you like in your Airflow directory:
DATA_DIR   = "/opt/airflow/data"
TMP_DIR    = "/opt/airflow/tmp/"
DATASET_ID = "mohankrishnathalla/mobile-reviews-sentiment-and-specification"
OUTPUT     = f"{DATA_DIR}/mobile-reviews.csv"

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
        pip install --no-cache-dir -r /opt/airflow/lib/requirements.txt
        pip install --no-cache-dir kaggle
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

        echo "Listing files in tmp before unzip:"
        ls -la

        unzip -o "{TMP_DIR}/mobile-reviews-sentiment-and-specification.zip" -d {TMP_DIR}
        rm -rf "{TMP_DIR}"

        echo "Listing files in tmp after unzip:"
        ls -la

        # 4) Move to the final location. Try the expected filename first,
        mv "{TMP_DIR}/Mobile Reviews Sentiment.csv" "{OUTPUT}"
        """,
    )

    model_pipeline = BashOperator(
        task_id="model_pipeline",
        bash_command=f"""
        pip install --no-cache-dir -e /opt/airflow/lib
        
        echo "Starting model pipeline with data from {OUTPUT}"
        python /opt/airflow/lib/main.py --data_path "{OUTPUT}"
        """,
    )

    end = EmptyOperator(task_id="end")

dependencies >> fetch_mobile_reviews >> model_pipeline >> end

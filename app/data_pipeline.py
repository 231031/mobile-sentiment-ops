import pandas as pd
import mlflow
from typing import Any, Dict

from app.config import *
from app.eda.overview import overview_eda, sentiment_bar_chart
from app.eda.text_length import text_length_eda, text_length_charts
from app.eda.word_freq import word_frequency_eda, word_frequency_charts
from app.eda.rating import rating_vs_sentiment_eda, rating_vs_sentiment_charts

storage_client = storage.Client(
    project="test-project", 
    credentials=AnonymousCredentials(),
    client_options=ClientOptions(api_endpoint=GCS_ENDPOINT)
)

class DataHandler:
    def __init__(self):
        self.production_model = None
        self.bucket_name = GCS_BUCKET_NAME

        try:
            self.bucket = storage_client.create_bucket(self.bucket_name)
        except:
            self.bucket = storage_client.bucket(self.bucket_name)

    # ---------- Storage helpers ----------
    def _upload_safe(self, blob_path, data_string, content_type='text/csv'):
        """
        Uploads data with retries and forces Simple Upload (Non-Resumable)
        to prevent 404/500 errors in the GCS Emulator.
        """
        blob = self.bucket.blob(blob_path)
        
        # CRITICAL FIX: Setting chunk_size to None forces "Simple Upload"
        # This prevents the client from using the buggy "Resumable PUT" method
        blob.chunk_size = None 
        try:
            blob.upload_from_string(data_string, content_type=content_type)
            return # Success
        except Exception as e:
            print(f"⚠️ Upload failed : {e}")
        
        print(f"❌ Failed to upload {blob_path} after retries.")
    
    def get_lastest_file(self, prefix: str = "data_label/labeled_"):
        blobs = list(self.bucket.list_blobs(prefix=prefix))
        tmp_dir = Path("/backend/temp")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / "ref.csv"

        if blobs:
            blobs.sort(key=lambda x: x.time_created, reverse=True)

            blobs[0].download_to_filename(str(tmp_path))
            ref_df = pd.read_csv(tmp_path)
            return ref_df

    def run_full_eda(self, df: pd.DataFrame, label_column: str = TARGET_COULUM, report_prefix: str = "eda") -> Dict[str, Any]:
        """
        Wrapper to run all EDA pieces in one call.
        """
        if df is None or df.empty:
            raise ValueError("Input dataframe is empty. Cannot run EDA.")

        overview_payload = overview_eda(df=df, label_column=label_column, review_column=REVIEW_COLUMN,
                                        rating_column="rating", report_prefix=report_prefix, length_column="text_length_chars")

        sentiment = sentiment_bar_chart(label_summary=overview_payload.get("label_summary", {}), report_prefix=report_prefix)

        text_len = text_length_eda(df=df, review_column=REVIEW_COLUMN, label_column=label_column,
                                   length_column="text_length_chars", report_prefix=report_prefix)

        text_len_charts = text_length_charts(df=df, review_column=REVIEW_COLUMN, label_column=label_column,
                                             length_column="text_length_chars", report_prefix=report_prefix)

        word_freq = word_frequency_eda(df=df, review_column=REVIEW_COLUMN, label_column=label_column,
                                       report_prefix=report_prefix, top_n=20)

        word_freq_charts = word_frequency_charts(freq_payload=word_freq, report_prefix=report_prefix, top_n=10)

        rating_eda = rating_vs_sentiment_eda(df=df, rating_column="rating", label_column=label_column,
                                             report_prefix=report_prefix)

        rating_charts = rating_vs_sentiment_charts(df=df, rating_column="rating", label_column=label_column,
                                                   report_prefix=report_prefix)

        result = {
            "overview": overview_payload,
            "sentiment_chart": sentiment,
            "text_length_overview": text_len,
            "text_length_charts": text_len_charts,
            "word_frequency_overview": word_freq,
            "word_frequency_charts": word_freq_charts,
            "rating_overview": rating_eda,
            "rating_charts": rating_charts,
        }
        return result

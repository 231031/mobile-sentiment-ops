import pandas as pd
import mlflow
from typing import Any, Dict
from pathlib import Path
from datetime import datetime

from app.config import *
from app.eda.overview import overview_eda, sentiment_bar_chart
from app.eda.text_length import text_length_eda, text_length_charts
from app.eda.word_freq import word_frequency_eda, word_frequency_charts, word_cloud_charts
from app.eda.duplicates import duplicate_review_eda, duplicate_review_charts
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

    def _upload_file(self, file_path: Path, dest_prefix: str):
        """
        Upload a local file into GCS emulator under the given prefix.
        """
        if not file_path.exists():
            print(f"[warn] report file missing, skip upload: {file_path}")
            return

        suffix = file_path.suffix.lower()
        content_type = {
            ".json": "application/json",
            ".png": "image/png",
            ".csv": "text/csv",
            ".html": "text/html",
        }.get(suffix, "application/octet-stream")

        blob_path = f"{dest_prefix}/{file_path.name}"
        data = file_path.read_bytes() if suffix in {".png"} else file_path.read_text(encoding="utf-8")
        try:
            self._upload_safe(blob_path, data, content_type=content_type)
        except Exception as e:
            print(f"[warn] upload failed for {file_path}: {e}")

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

        # Use a stable prefix for this batch; if caller didn't provide one, make a timestamped tag
        used_prefix = report_prefix if report_prefix else f"eda_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        overview_payload = overview_eda(df=df, label_column=label_column, review_column=REVIEW_COLUMN,
                                        rating_column="rating", report_prefix=used_prefix, length_column="text_length_chars")

        sentiment = sentiment_bar_chart(label_summary=overview_payload.get("label_summary", {}), report_prefix=used_prefix)

        text_len = text_length_eda(df=df, review_column=REVIEW_COLUMN, label_column=label_column,
                                   length_column="text_length_chars", report_prefix=used_prefix)

        text_len_charts = text_length_charts(df=df, review_column=REVIEW_COLUMN, label_column=label_column,
                                             length_column="text_length_chars", report_prefix=used_prefix)

        word_freq = word_frequency_eda(df=df, review_column=REVIEW_COLUMN, label_column=label_column,
                                       report_prefix=used_prefix, top_n=20)

        word_freq_charts = word_frequency_charts(freq_payload=word_freq, report_prefix=used_prefix, top_n=10)

        word_cloud = word_cloud_charts(freq_payload=word_freq, report_prefix=used_prefix)

        duplicates_summary = duplicate_review_eda(df=df, review_column=REVIEW_COLUMN,
                                                  report_prefix=used_prefix)

        duplicates_charts = duplicate_review_charts(summary_payload=duplicates_summary, report_prefix=used_prefix)

        rating_eda = rating_vs_sentiment_eda(df=df, rating_column="rating", label_column=label_column,
                                             report_prefix=used_prefix)

        rating_charts = rating_vs_sentiment_charts(df=df, rating_column="rating", label_column=label_column,
                                                   report_prefix=used_prefix)

        # Collect and upload EDA reports to storage
        def _collect_paths(obj):
            paths = []
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k == "report_path" and isinstance(v, str):
                        paths.append(Path(v))
                    else:
                        paths.extend(_collect_paths(v))
            elif isinstance(obj, list):
                for item in obj:
                    paths.extend(_collect_paths(item))
            return paths

        all_paths = _collect_paths({
            "overview": overview_payload,
            "sentiment_chart": sentiment,
            "text_length_overview": text_len,
            "text_length_charts": text_len_charts,
            "word_frequency_overview": word_freq,
            "word_frequency_charts": word_freq_charts,
            "word_clouds": word_cloud,
            "duplicates_summary": duplicates_summary,
            "duplicates_charts": duplicates_charts,
            "rating_overview": rating_eda,
            "rating_charts": rating_charts,
        })
        upload_prefix = f"reports/eda/{used_prefix}"
        for p in all_paths:
            self._upload_file(p, upload_prefix)

        result = {
            "overview": overview_payload,
            "sentiment_chart": sentiment,
            "text_length_overview": text_len,
            "text_length_charts": text_len_charts,
            "word_frequency_overview": word_freq,
            "word_frequency_charts": word_freq_charts,
            "word_clouds": word_cloud,
            "duplicates_summary": duplicates_summary,
            "duplicates_charts": duplicates_charts,
            "rating_overview": rating_eda,
            "rating_charts": rating_charts,
        }
        return result

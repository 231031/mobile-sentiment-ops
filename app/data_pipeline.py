import pandas as pd
import json
from io import StringIO
import requests

from app.config import *

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
    
    def save_for_labeling(self, df, timestamp_id):
        """
        Saves Clean Data to 'data_wait_label/'.
        """
        prefix = f"wlabel_{timestamp_id}"
        folder = "data_wait_label"
        
        # 1. Save Batch CSV (Good for backup/debugging)
        self._upload_safe(f"{folder}/{prefix}.csv", df.to_csv(index=False), 'text/csv')

        # 2. Save Individual JSON Tasks (For Label Studio)
        for index, row in df.iterrows():
            task_data = row.to_dict()
            # Generate name based on timestamp + row index
            json_filename = f"{folder}/tasks/{prefix}_row{index}.json"
            self._upload_safe(
                json_filename, 
                json.dumps(task_data), 
                content_type='application/json'
            )
        print(f"Saved {len(df)} tasks to {folder}/tasks/")
    
    def cleanup_processed_data(self, labeled_df):
        """
        Content-Based Cleanup:
        Matches the exported text against the 'data_wait_label' folder.
        Since we run this every time, the folder size remains small.
        """
        if labeled_df.empty: return

        print("Starting Content-Based Cleanup...")
        
        # Create a set of processed text for O(1) lookup
        labeled_texts = set(labeled_df[REVIEW_COLUMN].astype(str).tolist())
        
        # Scan the wait folder (Should be small if we clean regularly)
        blobs = list(self.bucket.list_blobs(prefix="data_wait_label/tasks/"))
        deleted_count = 0

        for blob in blobs:
            if not blob.name.endswith(".json"): continue
            
            try:
                content = blob.download_as_text()
                task_data = json.loads(content)
                
                # If the text in the file was just labeled/exported, delete the file
                if str(task_data.get(REVIEW_COLUMN, '')) in labeled_texts:
                    blob.delete()
                    deleted_count += 1
            except Exception as e:
                print(f"Error checking blob {blob.name}: {e}")

        print(f"Cleanup Complete. Deleted {deleted_count} files from queue.")


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

    # visualize and eda
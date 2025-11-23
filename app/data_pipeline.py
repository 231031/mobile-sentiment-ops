import pandas as pd

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
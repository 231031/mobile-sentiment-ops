## Usage:

1. Start mlflow server
```bash
mlflow server --port 5000
```

2. run the script
``` bash
python model_dev.py \
  --data_path ../data/mobile-reviews.csv \
  --experiment_name "Sentiment classification" \
  --registered_model_name "sentiment" \
  --tracking_uri http://localhost:5000 \
  --test_size 0.5 \
  --max_features 300
```

See results in mlflow UI: artifacts, metrics

## Usage:

1. Start mlflow server
```bash
mlflow server --port 5000
```

2. run the script
``` bash
python lr_demo.py \
  --data_path ../data/mobile-reviews.csv \
  --experiment_name "Sentiment LR" \
  --registered_model_name "sentiment-logreg" \
  --tracking_uri http://localhost:5000 \
  --test_size 0.2 \
  --max_features 100
```

see results in mlflow UI: artifacts, metrics

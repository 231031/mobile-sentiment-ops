import argparse
import os

import mlflow
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from model import build_pipelines, log_model_info
from artifacts import evaluate_model


# -----------------------
# CLI
# -----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True, default="data/mobile.reviews.csv", help="CSV path with columns: review_text, sentiment")
    p.add_argument("--experiment_name", default="Sentiment CLS")
    p.add_argument("--registered_model_name", default="sentiment")
    p.add_argument("--tracking_uri", default=os.getenv("MLFLOW_TRACKING_URI"))
    p.add_argument("--test_size", type=float, default=0.7)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--max_features", type=int, default=100)
    return p.parse_args()


def prepare_dataset(args):
    df = pd.read_csv(args.data_path).dropna(subset=["review_text", "sentiment"]).reset_index(drop=True)

    # Encode labels
    le = LabelEncoder()
    df["sentiment"] = le.fit_transform(df["sentiment"])
    class_names = le.classes_.tolist()

    # Split
    train_df, val_df = train_test_split(
        df, test_size=args.test_size, stratify=df["sentiment"], random_state=args.random_state
    )
    return train_df, val_df, class_names


# -----------------------
# Logging experiment
# -----------------------
def train_eval_log(model_key, pipe, 
                   X_train, y_train, X_val, y_val, 
                   class_names, args):
    model_names = {"lr": "LogisticRegression", "rf": "RandomForest", "xgb": "XGBoost"}
    run_name = model_names[model_key]
    registered_name = f"{args.registered_model_name}-{model_key}"

    with mlflow.start_run(run_name=run_name):        
        metrics = evaluate_model(pipe, model_names[model_key], 
                       X_train, y_train, X_val=X_val, y_val=y_val)

        # log Metadata as a single JSON
        log_model_info(
            model_name=model_names[model_key],
            model_key=model_key,
            pipe=pipe,
            run_name=run_name,
            registered_name=registered_name,
            class_names=class_names,
            metrics=metrics,
            args=args
        )


def main():
    args = parse_args()
    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
    else:
        print("[warn] No tracking URI supplied; defaulting to local ./mlruns store. Use --tracking_uri or set MLFLOW_TRACKING_URI to log against the MLflow server.")
    mlflow.set_experiment(args.experiment_name)

    # pipeline setup
    train_df, val_df, class_names = prepare_dataset(args)
    pipelines = build_pipelines(args.max_features)

    # train and eval each model
    order = ["lr", "rf"] + (["xgb"] if "xgb" in pipelines.keys() else [])
    for key in order:
        train_eval_log(
            model_key=key,
            pipe=pipelines[key],
            X_train=train_df['review_text'], y_train=train_df['sentiment'],
            X_val=val_df['review_text'], y_val=val_df['sentiment'],
            class_names=class_names,
            args=args,
        )

    print("Logged artifacts to mlflow!")


if __name__ == "__main__":
    main()
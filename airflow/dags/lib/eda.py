import json
from typing import Dict, Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud

import argparse
import mlflow


def _clean_text(text: str) -> str:
    """Lower and do minimal cleaning for the wordcloud.

    Avoid heavy NLP dependencies so function is safe inside Airflow tasks.
    """
    if not isinstance(text, str):
        return ""
    text =  text.lower()
    # remove common punctuation/URLs and collapse spaces
    text = json.loads(json.dumps(text)) if isinstance(text, str) else str(text)
    text = text.replace("\n", " ").replace("\r", " ")
    return " ".join(p for p in text.split())


def _clean_text(text: str) -> str:
    """Minimal cleaning for wordcloud generation."""
    if not isinstance(text, str):
        return ""
    text = text.replace("\n", " ").replace("\r", " ").lower()
    return " ".join(text.split())


def plot_sentiment_per_brand(df: pd.DataFrame, brand_col: str, sentiment_col: str):
    """Return a matplotlib Figure: stacked bar of sentiment counts per brand."""
    cross = pd.crosstab(df[brand_col], df[sentiment_col], margins=False)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.5 * len(cross))))
    cross.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title("Sentiment counts per brand")
    ax.set_ylabel("Count")
    ax.legend(title=sentiment_col)
    plt.tight_layout()
    return fig, cross


def plot_class_distribution(df: pd.DataFrame, sentiment_col: str):
    """Return a matplotlib Figure: bar plot of sentiment class distribution."""
    class_counts = df[sentiment_col].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=class_counts.index.astype(str), y=class_counts.values, ax=ax, 
                hue=class_counts.index.astype(str), legend=False)
    ax.set_title("Sentiment class distribution")
    ax.set_xlabel(sentiment_col)
    ax.set_ylabel("Count")
    plt.tight_layout()
    return fig, class_counts


def plot_wordcloud(df: pd.DataFrame, text_col: str, max_words: int = 200):
    """Return a matplotlib Figure with a generated wordcloud (or (None, None) if empty)."""
    texts = df[text_col].astype(str).map(_clean_text)
    full_text = " ".join(texts.values.tolist())
    if not full_text.strip():
        return None, None
    wc = WordCloud(width=1200, height=600, max_words=max_words, background_color="white").generate(full_text)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    plt.tight_layout()
    return fig, wc


def _make_serializable(x):
    if isinstance(x, dict):
        return {k: _make_serializable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_make_serializable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.ndarray,)):
        return [_make_serializable(v) for v in x.tolist()]
    try:
        json.dumps(x)
        return x
    except Exception:
        return str(x)


def eda(
    df: pd.DataFrame,
    text_col: str = "review_text",
    sentiment_col: str = "sentiment",
    brand_col: str = "brand",
    max_words: int = 200,
    log_to_mlflow: bool = True,
    mlflow_run_name: Optional[str] = "Metadata",
) -> Dict[str, Any]:
    """Compute EDA summaries, create figures and (optionally) log to MLflow.

    This function DOES NOT write local files. All artifacts are logged directly
    to MLflow (when `log_to_mlflow=True`) using `mlflow.log_figure` and
    `mlflow.log_dict`.
    """
    results: Dict[str, Any] = {}

    # basic validation
    if text_col not in df.columns:
        raise ValueError(f"Text column '{text_col}' not found in dataframe")
    if sentiment_col not in df.columns:
        raise ValueError(f"Sentiment column '{sentiment_col}' not found in dataframe")

    # 1) duplications
    total = len(df)
    dup_mask = df.duplicated(subset=[text_col], keep=False)
    dup_count = int(dup_mask.sum())
    dup_examples = df.loc[dup_mask, text_col].value_counts().head(5).to_dict()
    results["duplicates"] = {"total_rows": total, "duplicate_rows": dup_count, "examples": dup_examples}

    # 2) sentiment per brand
    fig_brand, cross = plot_sentiment_per_brand(df, brand_col, sentiment_col)
    results["sentiment_per_brand"] = {
        "counts": _make_serializable(cross),
        "percent": _make_serializable(cross.div(cross.sum(axis=1), axis=0).fillna(0)),
    }

    # 3) class distribution
    fig_class, class_counts = plot_class_distribution(df, sentiment_col)
    results["class_distribution"] = _make_serializable(class_counts)

    # 4) wordcloud
    fig_wc, wc_obj = plot_wordcloud(df, text_col, max_words=max_words)
    results["wordcloud"] = None if fig_wc is None else "wordcloud_generated"

    # Log to MLflow (no local saving)
    mlflow_info: Optional[Dict[str, str]] = None
    if log_to_mlflow:
        if mlflow is None:
            raise RuntimeError("mlflow is required for logging but is not importable in this environment")

        started_here = False
        active = mlflow.active_run()
        if active is None:
            mlflow.start_run(run_name=mlflow_run_name)
            started_here = True

        try:
            # log figures if created
            if fig_brand is not None:
                mlflow.log_figure(fig_brand, "eda/sentiment_per_brand.png")
                plt.close(fig_brand)
            if fig_class is not None:
                mlflow.log_figure(fig_class, "eda/sentiment_class_distribution.png")
                plt.close(fig_class)
            if fig_wc is not None:
                mlflow.log_figure(fig_wc, "eda/review_text_wordcloud.png")
                plt.close(fig_wc)

            # log summary
            mlflow.log_dict(_make_serializable(results), "eda/summary.json")

            mlflow_info = {"run_id": mlflow.active_run().info.run_id, "artifact_uri": mlflow.get_artifact_uri()}
        finally:
            if started_here:
                mlflow.end_run()

        results["mlflow"] = mlflow_info

    return results


def _cli():
    p = argparse.ArgumentParser(description="Run EDA and log artifacts to MLflow.")
    p.add_argument("--data_path", required=True, help="Path to CSV file with review_text and sentiment columns")
    p.add_argument("--mlflow_run_name", default="Metadata", required=False, help="Optional MLflow run name for EDA")
    p.add_argument("--experiment_name", default="Sentiment CLS", help="MLflow experiment name")
    args = p.parse_args()

    mlflow.set_experiment(args.experiment_name)
    df = pd.read_csv(args.data_path).dropna(subset=["review_text", "sentiment"])
    df = df.drop_duplicates(subset=["review_text"]).reset_index(drop=True)
    res = eda(df, log_to_mlflow=True, mlflow_run_name=args.mlflow_run_name)
    print("EDA finished. MLflow run info:", res.get("mlflow"))


if __name__ == "__main__":
    _cli()

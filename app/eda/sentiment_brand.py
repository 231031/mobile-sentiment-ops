from typing import Any, Dict

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from app.eda.utils import ensure_columns, save_json_report, save_figure


def sentiment_brand_eda(
    df: pd.DataFrame,
    brand_column: str,
    label_column: str,
    report_prefix: str = "sentiment_brand",
) -> Dict[str, Any]:
    """Summarize sentiment counts per brand and save as JSON report."""
    if df is None or df.empty:
        raise ValueError("Input dataframe is empty. Cannot run sentiment/brand EDA.")
    ensure_columns(df, [brand_column, label_column])

    counts_by_sentiment: Dict[str, Dict[str, int]] = {}
    for lbl, sub in df[[label_column, brand_column]].dropna().groupby(label_column):
        brand_counts = sub[brand_column].value_counts()
        counts_by_sentiment[str(lbl)] = {str(b): int(c) for b, c in brand_counts.items()}

    payload = {
        "generated_at": None,  # filled by save_json_report
        "label_column": label_column,
        "brand_column": brand_column,
        "counts_by_sentiment": counts_by_sentiment,
    }

    return save_json_report(payload, "eda_sentiment_brand", report_prefix)


def sentiment_brand_charts(
    df: pd.DataFrame,
    brand_column: str,
    label_column: str,
    report_prefix: str = "sentiment_brand",
) -> Dict[str, Any]:
    """Plot bar charts of sentiment counts per brand for each sentiment class."""
    if df is None or df.empty:
        raise ValueError("Input dataframe is empty. Cannot plot sentiment/brand charts.")
    ensure_columns(df, [brand_column, label_column])

    results: Dict[str, Dict[str, Any]] = {}
    sentiment_labels = ["Positive", "Negative", "Neutral"]
    brands = sorted(df[brand_column].dropna().astype(str).unique().tolist())
    if not brands:
        return {
            "positive_chart": {"report_path": None, "logged_to_mlflow": False},
            "negative_chart": {"report_path": None, "logged_to_mlflow": False},
            "neutral_chart": {"report_path": None, "logged_to_mlflow": False},
        }
    brand_palette = sns.color_palette("husl", len(brands))

    for sentiment in sentiment_labels:
        sub = df[df[label_column] == sentiment]
        if sub.empty:
            results[f"{sentiment.lower()}_chart"] = {"report_path": None, "logged_to_mlflow": False}
            continue
        counts = (
            sub[brand_column]
            .astype(str)
            .value_counts()
            .reindex(brands, fill_value=0)
            .reset_index()
        )
        counts.columns = ["brand", "count"]

        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(data=counts, x="brand", y="count", palette=brand_palette, ax=ax, order=brands)
        ax.set_title(f"{sentiment} sentiment per brand")
        ax.set_xlabel("Brand")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=45)

        saved = save_figure(fig, f"eda_sentiment_brand_{sentiment.lower()}", report_prefix)
        results[f"{sentiment.lower()}_chart"] = {
            "report_path": saved["report_path"],
            "logged_to_mlflow": saved["logged_to_mlflow"],
        }

    return results

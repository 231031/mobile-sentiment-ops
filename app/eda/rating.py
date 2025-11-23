from typing import Any, Dict

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from app.config import TARGET_COULUM
from app.eda.utils import numeric_stats, save_json_report, save_figure, sentiment_palette


def rating_vs_sentiment_eda(
    df: pd.DataFrame,
    rating_column: str = "rating",
    label_column: str = TARGET_COULUM,
    report_prefix: str = "rating_sentiment",
) -> Dict[str, Any]:
    if df is None or df.empty:
        raise ValueError("Input dataframe is empty. Cannot run rating vs sentiment EDA.")

    if rating_column not in df.columns or label_column not in df.columns:
        return {
            "rating_available": False,
            "report_path": None,
            "logged_to_mlflow": False,
        }

    df_use = df[[rating_column, label_column]].copy()
    df_use = df_use[pd.to_numeric(df_use[rating_column], errors="coerce").notna()]
    df_use[rating_column] = df_use[rating_column].astype(float)
    if df_use.empty:
        return {
            "rating_available": False,
            "report_path": None,
            "logged_to_mlflow": False,
        }

    global_summary = numeric_stats(df_use[rating_column])
    summary_by_label = {}
    counts_by_rating_and_label = {}
    for lbl, sub in df_use.groupby(label_column):
        summary_by_label[str(lbl)] = numeric_stats(sub[rating_column])
        counts_by_rating_and_label[str(lbl)] = {
            str(k): int(v) for k, v in sub[rating_column].value_counts().sort_index().to_dict().items()
        }

    payload = {
        "rating_available": True,
        "rating_column": rating_column,
        "label_column": label_column,
        "global_summary": global_summary,
        "summary_by_label": summary_by_label,
        "counts_by_rating_and_label": counts_by_rating_and_label,
    }

    return save_json_report(payload, "eda_rating_sentiment", report_prefix)


def rating_vs_sentiment_charts(
    df: pd.DataFrame,
    rating_column: str = "rating",
    label_column: str = TARGET_COULUM,
    report_prefix: str = "rating_sentiment",
) -> Dict[str, Any]:
    results = {
        "mean_bar_chart": {"report_path": None, "logged_to_mlflow": False},
        "boxplot": {"report_path": None, "logged_to_mlflow": False},
    }

    if df is None or df.empty or rating_column not in df.columns or label_column not in df.columns:
        return results

    df_use = df[[rating_column, label_column]].copy()
    df_use = df_use[pd.to_numeric(df_use[rating_column], errors="coerce").notna()]
    df_use[rating_column] = df_use[rating_column].astype(float)
    if df_use.empty:
        return results

    means = df_use.groupby(label_column)[rating_column].mean().reset_index()
    fig_mean, ax_mean = plt.subplots(figsize=(6, 4))
    sns.barplot(data=means, x=label_column, y=rating_column,
                palette=sentiment_palette(means[label_column].tolist()), ax=ax_mean)
    ax_mean.set_title("Average rating by sentiment label")
    ax_mean.set_xlabel("Label")
    ax_mean.set_ylabel("Mean rating")
    mean_saved = save_figure(fig_mean, "eda_rating_mean_by_label", report_prefix)

    fig_box, ax_box = plt.subplots(figsize=(7, 4))
    labels = df_use[label_column].dropna().unique().tolist()
    sns.boxplot(data=df_use, x=label_column, y=rating_column, ax=ax_box,
                palette=sentiment_palette(labels))
    ax_box.set_title("Rating distribution by sentiment label")
    ax_box.set_xlabel("Label")
    ax_box.set_ylabel("Rating")
    box_saved = save_figure(fig_box, "eda_rating_box_by_label", report_prefix)

    results["mean_bar_chart"] = {"report_path": mean_saved["report_path"], "logged_to_mlflow": mean_saved["logged_to_mlflow"]}
    results["boxplot"] = {"report_path": box_saved["report_path"], "logged_to_mlflow": box_saved["logged_to_mlflow"]}
    return results

from typing import Any, Dict

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from app.config import REVIEW_COLUMN
from app.eda.utils import ensure_columns, save_json_report, save_figure


def duplicate_review_eda(
    df: pd.DataFrame,
    review_column: str = REVIEW_COLUMN,
    report_prefix: str = "duplicates",
) -> Dict[str, Any]:
    """Summarize duplicate review_text counts and list all repeated texts."""
    if df is None or df.empty:
        raise ValueError("Input dataframe is empty. Cannot run duplicate review EDA.")
    ensure_columns(df, [review_column])

    series = df[review_column].astype(str)
    total_rows = int(len(series))
    unique_value_count = int(series.nunique())
    value_counts = series.value_counts()
    duplicate_value_counts = value_counts[value_counts > 1]
    duplicate_value_count = int(len(duplicate_value_counts))
    duplicate_rows = int((series.duplicated(keep=False)).sum())
    unique_rows = total_rows - duplicate_rows

    duplicated_reviews = [
        {"text": text, "count": int(count)}
        for text, count in duplicate_value_counts.items()
    ]

    payload = {
        "review_column": review_column,
        "total_rows": total_rows,
        "unique_value_count": unique_value_count,
        "duplicate_value_count": duplicate_value_count,
        "duplicate_rows": duplicate_rows,
        "unique_rows": unique_rows,
        "duplicated_reviews": duplicated_reviews,
    }

    return save_json_report(payload, "eda_duplicates", report_prefix)


def duplicate_review_charts(
    summary_payload: dict,
    report_prefix: str = "duplicates",
) -> Dict[str, Any]:
    """Plot bar chart of unique vs duplicate review counts."""
    if not summary_payload:
        raise ValueError("summary_payload is required to plot duplicate review chart.")

    unique_rows = summary_payload.get("unique_rows", 0)
    duplicate_rows = summary_payload.get("duplicate_rows", 0)

    plot_df = pd.DataFrame(
        {
            "type": ["unique", "duplicate"],
            "count": [unique_rows, duplicate_rows],
        }
    )

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.barplot(data=plot_df, x="type", y="count", palette=["#4CAF50", "#F44336"], ax=ax)
    ax.set_title("Unique vs Duplicate Reviews")
    ax.set_xlabel("")
    ax.set_ylabel("Count")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    saved = save_figure(fig, "eda_duplicates_bar", report_prefix)

    return {
        "bar_chart": {
            "report_path": saved["report_path"],
            "logged_to_mlflow": saved["logged_to_mlflow"],
        }
    }

from typing import Any, Dict

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from app.config import REVIEW_COLUMN, TARGET_COULUM
from app.eda.utils import ensure_text_length_column, numeric_stats, save_json_report, save_figure, sentiment_palette


def text_length_eda(
    df: pd.DataFrame,
    review_column: str = REVIEW_COLUMN,
    label_column: str = TARGET_COULUM,
    length_column: str = "text_length_chars",
    report_prefix: str = "text_length",
) -> Dict[str, Any]:
    """Summarize text length stats overall and by label, save as JSON."""
    if df is None or df.empty:
        raise ValueError("Input dataframe is empty. Cannot run text length EDA.")

    length_col = ensure_text_length_column(df, review_column, length_column)
    lengths = df[length_col].dropna()

    global_summary = numeric_stats(lengths)

    summary_by_label = {}
    if label_column in df.columns:
        for lbl, sub in df[[label_column, length_col]].dropna(subset=[length_col]).groupby(label_column):
            summary_by_label[str(lbl)] = numeric_stats(sub[length_col])

    payload = {
        "length_column": length_col,
        "global_summary": global_summary,
        "summary_by_label": summary_by_label,
    }

    return save_json_report(payload, "eda_text_length", report_prefix)


def text_length_charts(
    df: pd.DataFrame,
    review_column: str = REVIEW_COLUMN,
    label_column: str = TARGET_COULUM,
    length_column: str = "text_length_chars",
    report_prefix: str = "text_length",
) -> Dict[str, Any]:
    """Generate boxplot figures for text length across labels."""
    length_col = ensure_text_length_column(df, review_column, length_column)
    if length_col not in df.columns:
        raise ValueError(f"Length column '{length_col}' not found.")

    fig_box, ax_box = plt.subplots(figsize=(7, 4))
    labels = df[label_column].dropna().unique().tolist()
    sns.boxplot(data=df[[label_column, length_col]].dropna(), x=label_column, y=length_col, ax=ax_box,
                palette=sentiment_palette(labels))
    ax_box.set_title("Text length by label")
    ax_box.set_xlabel("Label")
    ax_box.set_ylabel("Length (chars)")

    saved = save_figure(fig_box, "eda_text_length_box", report_prefix)

    return {
        "boxplot": {
            "report_path": saved["report_path"],
            "logged_to_mlflow": saved["logged_to_mlflow"],
        },
    }

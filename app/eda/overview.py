from typing import Any, Dict

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from app.config import REVIEW_COLUMN, TARGET_COULUM
from app.eda.utils import ensure_text_length_column, save_json_report, save_figure, sentiment_palette


def overview_eda(
    df: pd.DataFrame,
    label_column: str = TARGET_COULUM,
    review_column: str = REVIEW_COLUMN,
    rating_column: str = "rating",
    report_prefix: str = "overview",
    length_column: str = "text_length_chars",
) -> Dict[str, Any]:
    if df is None or df.empty:
        raise ValueError("Input dataframe is empty. Cannot generate EDA overview.")

    length_col = ensure_text_length_column(df, review_column, length_column)

    rows, cols = df.shape
    dtype_info = [{"column": col, "dtype": str(dtype)} for col, dtype in df.dtypes.items()]

    label_counts = df[label_column].value_counts(dropna=False) if label_column in df.columns else pd.Series(dtype=int)
    label_summary = {
        "label_column": label_column,
        "num_classes": int(len(label_counts)),
        "classes": [str(c) for c in label_counts.index.tolist()],
        "counts": {str(k): int(v) for k, v in label_counts.to_dict().items()},
    }

    pct = (lambda c: round((c / rows) * 100, 2) if rows else 0.0)
    missing_review = int(df[review_column].isna().sum()) if review_column in df.columns else None
    missing_label = int(df[label_column].isna().sum()) if label_column in df.columns else None
    dup_rows = int(df.duplicated(keep="first").sum())
    dup_review = int(df[review_column].duplicated(keep="first").sum()) if review_column in df.columns else None

    if rating_column in df.columns:
        series = df[rating_column]
        outlier_mask = series.notna() & ~series.between(1, 5)
        outlier_count = int(outlier_mask.sum())
        rating_min = float(series.min()) if not series.empty else None
        rating_max = float(series.max()) if not series.empty else None
    else:
        outlier_count = None
        rating_min = None
        rating_max = None

    payload = {
        "shape": {"rows": rows, "columns": cols},
        "dtypes": dtype_info,
        "label_summary": label_summary,
        "missing": {
            "review_text": {"count": missing_review, "pct": pct(missing_review) if missing_review is not None else None},
            "label": {"count": missing_label, "pct": pct(missing_label) if missing_label is not None else None},
        },
        "duplicates": {
            "full_rows": {"count": dup_rows, "pct": pct(dup_rows)},
            "review_text": {"count": dup_review, "pct": pct(dup_review) if dup_review is not None else None},
        },
        "rating_outliers": {
            "column": rating_column,
            "count": outlier_count,
            "pct": pct(outlier_count) if outlier_count is not None else None,
            "min": rating_min,
            "max": rating_max,
            "expected_range": [1, 5],
        },
        "text_length_column": length_col,
    }

    return save_json_report(payload, "eda_overview", report_prefix)


def sentiment_bar_chart(
    label_summary: dict,
    report_prefix: str = "sentiment_bar",
) -> Dict[str, Any]:
    if not label_summary or "counts" not in label_summary:
        raise ValueError("label_summary with counts is required.")

    label_order = label_summary.get("classes") or ["Positive", "Neutral", "Negative"]
    counts_dict = {str(k): int(v) for k, v in label_summary["counts"].items()}

    counts = [counts_dict.get(lbl, 0) for lbl in label_order]
    total = int(sum(counts))
    percents = [round((c / total) * 100, 2) if total else 0.0 for c in counts]

    plot_df = pd.DataFrame({"label": label_order, "count": counts})
    palette = sentiment_palette(label_order)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=plot_df, x="label", y="count", palette=palette, ax=ax)
    ax.set_ylabel("Count")
    ax.set_xlabel("")
    ax.set_title("Sentiment distribution")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for p, count, percent in zip(ax.patches, counts, percents):
        ax.annotate(
            f"{count} ({percent}%)",
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha="center",
            va="bottom",
        )

    saved = save_figure(fig, "eda_sentiment", report_prefix)
    return {
        "counts": dict(zip(label_order, counts)),
        "percents": dict(zip(label_order, percents)),
        "report_path": saved["report_path"],
        "logged_to_mlflow": saved["logged_to_mlflow"],
    }

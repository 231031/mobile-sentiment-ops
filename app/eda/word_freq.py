from typing import Any, Dict

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import re

from app.config import REVIEW_COLUMN, TARGET_COULUM
from app.eda.utils import tokenize, save_json_report, save_figure, ensure_columns, categorical_palette


def word_frequency_eda(
    df: pd.DataFrame,
    review_column: str = REVIEW_COLUMN,
    label_column: str = TARGET_COULUM,
    report_prefix: str = "word_freq",
    top_n: int = 20,
) -> Dict[str, Any]:
    """Compute top token frequencies per label and save as JSON report."""
    if df is None or df.empty:
        raise ValueError("Input dataframe is empty. Cannot run word frequency EDA.")
    ensure_columns(df, [review_column, label_column])

    top_words_by_label = {}
    for lbl, sub in df[[label_column, review_column]].dropna(subset=[review_column]).groupby(label_column):
        counter = Counter()
        for text in sub[review_column]:
            counter.update(tokenize(str(text)))
        top_words = counter.most_common(top_n)
        top_words_by_label[str(lbl)] = [{"word": w, "count": int(c)} for w, c in top_words]

    payload = {
        "text_column": review_column,
        "label_column": label_column,
        "top_n": top_n,
        "top_words_by_label": top_words_by_label,
    }

    return save_json_report(payload, "eda_word_freq", report_prefix)


def word_frequency_charts(
    freq_payload: dict,
    report_prefix: str = "word_freq",
    top_n: int = 10,
) -> Dict[str, Any]:
    """Plot bar charts for top tokens per label from frequency payload."""
    results = {}
    for lbl, items in freq_payload.get("top_words_by_label", {}).items():
        if not items:
            continue
        plot_df = pd.DataFrame(items[:top_n])
        fig, ax = plt.subplots(figsize=(7, 4))
        palette = categorical_palette(len(plot_df))
        sns.barplot(data=plot_df, x="word", y="count", palette=palette, ax=ax)
        ax.set_title(f"Top words for label = {lbl}")
        ax.set_xlabel("word")
        ax.set_ylabel("count")
        ax.tick_params(axis='x', rotation=45)

        safe_lbl = re.sub(r'[^a-zA-Z0-9]+', '_', lbl)
        saved = save_figure(fig, f"eda_wordfreq_bar_{safe_lbl}", report_prefix)
        results[lbl] = {"report_path": saved["report_path"], "logged_to_mlflow": saved["logged_to_mlflow"]}

    return {"bar_charts": results}


def word_cloud_charts(
    freq_payload: dict,
    report_prefix: str = "word_freq",
) -> Dict[str, Any]:
    """Generate word clouds per label from frequency payload."""
    try:
        from wordcloud import WordCloud
    except ImportError:
        return {"wordclouds": {}, "error": "wordcloud package not installed"}

    results = {}
    for lbl, items in freq_payload.get("top_words_by_label", {}).items():
        if not items:
            continue
        freqs = {entry["word"]: entry["count"] for entry in items if "word" in entry and "count" in entry}
        if not freqs:
            continue

        wc = WordCloud(width=800, height=400, background_color="white", colormap="viridis")
        wc.generate_from_frequencies(freqs)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"Word cloud for label = {lbl}")

        safe_lbl = re.sub(r'[^a-zA-Z0-9]+', '_', lbl)
        saved = save_figure(fig, f"eda_wordcloud_{safe_lbl}", report_prefix)
        results[lbl] = {"report_path": saved["report_path"], "logged_to_mlflow": saved["logged_to_mlflow"]}

    return {"wordclouds": results}

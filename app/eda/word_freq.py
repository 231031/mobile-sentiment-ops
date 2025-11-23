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

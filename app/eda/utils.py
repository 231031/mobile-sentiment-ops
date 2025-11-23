import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import nltk
import pandas as pd
from nltk.corpus import stopwords

from app.config import REPORTS_DIR


# ---------- Path / saving helpers ----------
# Build a timestamped path under reports for a given base name/prefix/ext.
def timestamped_path(base_name: str, prefix: str, ext: str) -> Tuple[str, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return timestamp, REPORTS_DIR / f"{base_name}_{timestamp}.{ext}"


# Decide artifact subfolder for mlflow; if a run is active, log at root.
def artifact_path_for_run(run) -> Optional[str]:
    """Return artifact subdir for mlflow logging; if run exists, log to root."""
    return None if run is not None else "eda"


def save_json_report(payload: Dict[str, Any], base_name: str, report_prefix: str) -> Dict[str, Any]:
    timestamp, report_path = timestamped_path(base_name, report_prefix, "json")
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    active_run = mlflow.active_run()
    if active_run is not None:
        artifact_path = artifact_path_for_run(active_run)
        if artifact_path:
            mlflow.log_artifact(str(report_path), artifact_path=artifact_path)
        else:
            mlflow.log_artifact(str(report_path))
    return payload | {"report_path": str(report_path), "logged_to_mlflow": active_run is not None, "generated_at": timestamp}


# Save a matplotlib figure and log it as an artifact if mlflow run is active.
def save_figure(fig, base_name: str, report_prefix: str) -> Dict[str, Any]:
    timestamp, img_path = timestamped_path(base_name, report_prefix, "png")
    fig.tight_layout()
    fig.savefig(img_path)
    mlflow_run = mlflow.active_run()
    if mlflow_run is not None:
        artifact_path = artifact_path_for_run(mlflow_run)
        if artifact_path:
            mlflow.log_artifact(str(img_path), artifact_path=artifact_path)
        else:
            mlflow.log_artifact(str(img_path))
    return {"report_path": str(img_path), "logged_to_mlflow": mlflow_run is not None, "generated_at": timestamp}


# ---------- Data helpers ----------
# Ensure required columns exist in the dataframe.
def ensure_columns(df: pd.DataFrame, required_cols: List[str]):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Required columns missing: {missing}")


# Basic numeric summary stats with NaN handling.
def numeric_stats(series: pd.Series) -> Dict[str, Any]:
    if series is None or series.empty:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None, "std": None}
    series = series.dropna()
    if series.empty:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None, "std": None}
    return {
        "count": int(series.count()),
        "min": float(series.min()),
        "max": float(series.max()),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "std": float(series.std()),
    }


# ---------- Text helpers ----------
try:
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    STOPWORDS = set(stopwords.words("english"))


def tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z0-9']+", text.lower())
    return [t for t in tokens if t and t not in STOPWORDS]


# Add a text length column if absent and return its name.
def ensure_text_length_column(df: pd.DataFrame, review_column: str, length_column: str = "text_length_chars") -> str:
    if review_column not in df.columns:
        raise ValueError(f"Review column '{review_column}' not found.")
    if length_column not in df.columns:
        df[length_column] = df[review_column].fillna("").astype(str).str.len()
    return length_column


# ---------- Visualization helpers ----------
SENTIMENT_COLORS = {
    "Positive": "#4CAF50",
    "Neutral": "#FFC107",
    "Negative": "#F44336",
}


def sentiment_palette(labels: List[str]) -> List[str]:
    """Return a list of colors matching sentiment labels; fall back to a default color."""
    default = "#4C78A8"
    return [SENTIMENT_COLORS.get(lbl, default) for lbl in labels]


# Palette helper for categorical charts.
def categorical_palette(count: int) -> List[str]:
    """Diverse palette for categorical bars."""
    import seaborn as sns  # local import to avoid hard dep when not plotting
    return sns.color_palette("husl", count)

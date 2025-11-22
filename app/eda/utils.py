import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import mlflow
import pandas as pd

from app.config import REPORTS_DIR


# ---------- Path / saving helpers ----------
def timestamped_path(base_name: str, prefix: str, ext: str) -> Tuple[str, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix_tag = f"{prefix}_" if prefix else ""
    return timestamp, REPORTS_DIR / f"{base_name}_{prefix_tag}{timestamp}.{ext}"


def save_json_report(payload: Dict[str, Any], base_name: str, report_prefix: str) -> Dict[str, Any]:
    timestamp, report_path = timestamped_path(base_name, report_prefix, "json")
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    active_run = mlflow.active_run()
    if active_run is not None:
        mlflow.log_artifact(str(report_path), artifact_path="eda")
    return payload | {"report_path": str(report_path), "logged_to_mlflow": active_run is not None, "generated_at": timestamp}


def save_figure(fig, base_name: str, report_prefix: str) -> Dict[str, Any]:
    timestamp, img_path = timestamped_path(base_name, report_prefix, "png")
    fig.tight_layout()
    fig.savefig(img_path)
    mlflow_run = mlflow.active_run()
    if mlflow_run is not None:
        mlflow.log_artifact(str(img_path), artifact_path="eda")
    return {"report_path": str(img_path), "logged_to_mlflow": mlflow_run is not None, "generated_at": timestamp}


# ---------- Data helpers ----------
def ensure_columns(df: pd.DataFrame, required_cols: List[str]):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Required columns missing: {missing}")


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
STOPWORDS = {
    "the", "a", "an", "is", "are", "of", "and", "to", "in", "it", "for", "on",
    "this", "that", "with", "as", "was", "were", "be", "at", "by", "or", "from",
    "but", "not", "have", "has", "had", "you", "i", "we", "they", "them", "us",
    "will", "would", "can", "could", "should", "my", "your", "our", "their"
}


def tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z0-9']+", text.lower())
    return [t for t in tokens if t and t not in STOPWORDS]


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


def categorical_palette(count: int) -> List[str]:
    """Diverse palette for categorical bars."""
    import seaborn as sns  # local import to avoid hard dep when not plotting
    return sns.color_palette("husl", count)

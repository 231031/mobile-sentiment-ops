import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import tempfile
import os

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.pipeline.column_mapping import ColumnMapping

from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    ConfusionMatrixDisplay, confusion_matrix, classification_report
)

# ==============
# Figures
# ==============
def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    return fig


def plot_decision_boundary(model, X_sparse, y, label_names, title: str):
    """2D viz only (SVD inside for plotting)."""
    svd = TruncatedSVD(n_components=2, random_state=42)
    X_2d = svd.fit_transform(X_sparse)
    X_2d = StandardScaler().fit_transform(X_2d)

    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    clf_2d = clone(model)
    clf_2d.fit(X_2d, y)
    Z = clf_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    for lbl in np.unique(y):
        ax.scatter(X_2d[y == lbl, 0], X_2d[y == lbl, 1], label=label_names[lbl], s=10)
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    return fig


def evidently_report(pipe, X_train, y_train, X_val, y_val):
    y_train_pred = pipe.predict(X_train)
    tfidf = pipe.named_steps["tfidf"]
    
    def tfidf_features(X):
        X_tfidf = tfidf.transform(X)
        try:
            X_arr = X_tfidf.toarray()
        except Exception:
            X_arr = X_tfidf
        X_df = pd.DataFrame(X_arr, columns=tfidf.get_feature_names_out())
        return X_df

    # TF-IDF returns a sparse matrix; convert to dense array before creating DataFrame
    X_train_tfidf = tfidf_features(X_train)
    ref_df = pd.DataFrame({
        "review_text": X_train,
        "target": pd.Series(y_train).tolist(),
    })
    ref_df = pd.concat([ref_df.reset_index(drop=True), X_train_tfidf.reset_index(drop=True)], axis=1)
    if y_train_pred is not None:
        ref_df["prediction"] = pd.Series(y_train_pred).tolist()

    y_pred = pipe.predict(X_val)
    X_val_tfidf = tfidf_features(X_val)
    curr_df = pd.DataFrame({
        "review_text": X_val,
        "target": pd.Series(y_val).tolist(),
        "prediction": pd.Series(y_pred).tolist(),
    })
    curr_df = pd.concat([curr_df.reset_index(drop=True), X_val_tfidf.reset_index(drop=True)], axis=1)

    column_mapping = ColumnMapping(target="target", prediction="prediction", 
                                   text_features=["review_text"], 
                                   numerical_features=X_train_tfidf.columns.tolist())
    report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])
    report.run(reference_data=ref_df, current_data=curr_df, column_mapping=column_mapping)
        
    HTML_PATH = "data_drift_report.html"
    JSON_PATH = "data_drift_report.json"
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_path = os.path.join(tmpdirname, HTML_PATH)
        report.save_html(tmp_path)
        mlflow.log_artifact(tmp_path, artifact_path="reports")
        
        tmp_path = os.path.join(tmpdirname, JSON_PATH)
        report.save_json(tmp_path)
        mlflow.log_artifact(tmp_path, artifact_path="reports")

# ==============
# Utility
# ==============
def evaluate_model(pipe, model_name, X_train, y_train, X_val, y_val):    
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_val)
    metrics = {
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "macro_f1": float(f1_score(y_val, y_pred, average="macro")),
        "micro_f1": float(f1_score(y_val, y_pred, average="micro")),
        "weighted_f1": float(f1_score(y_val, y_pred, average="weighted")),
        "macro_precision": float(precision_score(y_val, y_pred, average="macro")),
        "macro_recall": float(recall_score(y_val, y_pred, average="macro"))
    }
    for k, v in metrics.items():
        mlflow.log_metric(k, v)
    print(mlflow.get_registry_uri())
    print(mlflow.active_run())
    
    # Determine class labels (actual values) and display names
    if hasattr(pipe, "named_steps") and "clf" in pipe.named_steps:
        clf = pipe.named_steps["clf"]
    else:
        clf = pipe

    if hasattr(clf, "classes_"):
        class_labels = list(clf.classes_)
    else:
        class_labels = list(np.unique(np.concatenate([y_train, y_val])))

    # Confusion matrix (uses actual label values for `labels`)
    fig_cm = plot_confusion_matrix(y_true=y_val, y_pred=y_pred, labels=class_labels)
    mlflow.log_figure(fig_cm, f"{model_name}__confusion_matrix.png")
    plt.close(fig_cm)
    
    # Decision boundary (uses TF-IDF transform for projection)
    class_display_names = [str(c) for c in class_labels]
    try:
        tfidf = pipe.named_steps["tfidf"]
        X_all_sparse = tfidf.transform(pd.concat([X_train, X_val]).tolist())
        y_all = np.concatenate([y_train, y_val])

        plot_estimator = clone(pipe.named_steps["clf"])
        label_name_map = {val: name for val, name in zip(class_labels, class_display_names)}
        fig_db = plot_decision_boundary(
            model=plot_estimator,
            X_sparse=X_all_sparse,
            y=y_all,
            label_names=label_name_map,
            title=f"Decision Boundary (2D SVD)"
        )
        mlflow.log_figure(fig_db, f"{model_name}__decision_boundary.png")
        plt.close(fig_db)
    except Exception as e:
        print(f"[warn] decision boundary plot failed: {e}")

    # Classification report
    report = classification_report(y_val, y_pred, target_names=class_display_names, output_dict=True)
    mlflow.log_dict(report, f"{model_name}__classification_report.json")
    
    # Evidently report (data drift + classification performance)
    evidently_report(pipe, X_train, y_train, X_val, y_val)
    return metrics

"""Metrics, baseline comparison and report generation.

This module never tunes anything: it scores a frozen pipeline on data it is
handed and writes the evidence files the dashboard reads.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from . import config

LABELS = ("ham", "spam")


def spam_scores(pipeline, texts) -> np.ndarray:
    """Return the model's raw score for the spam class (uncalibrated)."""
    return pipeline.predict_proba(list(texts))[:, 1]


def binary_metrics(y_true, y_pred, y_score) -> dict:
    """Compute the metric set used by the dashboard and the model card."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_score = np.asarray(y_score, dtype=float)
    y_binary = (y_true == config.POSITIVE_LABEL).astype(int)
    predicted_binary = (y_pred == config.POSITIVE_LABEL).astype(int)

    matrix = confusion_matrix(y_binary, predicted_binary, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())

    return {
        "rows": int(len(y_true)),
        "accuracy": float(accuracy_score(y_binary, predicted_binary)),
        "precision": float(precision_score(y_binary, predicted_binary, zero_division=0)),
        "recall": float(recall_score(y_binary, predicted_binary, zero_division=0)),
        "f1": float(f1_score(y_binary, predicted_binary, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_binary, y_score)),
        "pr_auc": float(average_precision_score(y_binary, y_score)),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "support": {"ham": int((y_binary == 0).sum()), "spam": int((y_binary == 1).sum())},
    }


def keyword_rule_predict(texts) -> np.ndarray:
    """Single-keyword baseline: spam if any configured keyword appears."""
    predictions = []
    for text in texts:
        lowered = str(text).lower()
        predictions.append(
            config.POSITIVE_LABEL
            if any(keyword in lowered for keyword in config.BASELINE_KEYWORDS)
            else "ham"
        )
    return np.asarray(predictions)


def build_baselines(train_frame: pd.DataFrame, test_frame: pd.DataFrame) -> dict:
    """Score two trivial baselines on the identical held-out split."""
    majority = DummyClassifier(strategy="most_frequent")
    majority.fit(train_frame["text"], train_frame["label"])
    majority_pred = majority.predict(test_frame["text"])
    majority_score = np.zeros(len(test_frame))

    keyword_pred = keyword_rule_predict(test_frame["text"])
    keyword_score = (keyword_pred == config.POSITIVE_LABEL).astype(float)

    return {
        "majority_class": binary_metrics(test_frame["label"], majority_pred, majority_score),
        "keyword_rule": binary_metrics(test_frame["label"], keyword_pred, keyword_score),
    }


def comparison_rows(model_metrics: dict, baselines: dict) -> list[dict]:
    rows = [
        {
            "model": "logistic_regression_tfidf",
            "accuracy": model_metrics["accuracy"],
            "precision": model_metrics["precision"],
            "recall": model_metrics["recall"],
            "f1": model_metrics["f1"],
            "roc_auc": model_metrics["roc_auc"],
            "pr_auc": model_metrics["pr_auc"],
        }
    ]
    for name, metrics in baselines.items():
        rows.append(
            {
                "model": name,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
            }
        )
    return rows


def write_baseline_csv(rows: list[dict], path: Path | None = None) -> Path:
    path = Path(path or config.BASELINE_CSV)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_confusion_csv(metrics: dict, path: Path | None = None) -> Path:
    path = Path(path or config.CONFUSION_CSV)
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = metrics["confusion_matrix"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["", "predicted_ham", "predicted_spam"])
        writer.writerow(["actual_ham", matrix["tn"], matrix["fp"]])
        writer.writerow(["actual_spam", matrix["fn"], matrix["tp"]])
    return path


def write_metrics(payload: dict, path: Path | None = None) -> Path:
    """Persist the metrics report exactly as produced by the training script."""
    path = Path(path or config.METRICS_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_metrics(path: Path | None = None) -> dict:
    path = Path(path or config.METRICS_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"Metrics report missing at {path}. Run: python scripts\\train_model.py"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def library_versions() -> dict:
    from . import __version__

    return {
        "python": ".".join(str(part) for part in __import__("sys").version_info[:3]),
        "scikit_learn": sklearn.__version__,
        "scamdetector": __version__,
    }

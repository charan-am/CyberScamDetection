"""Train, select a threshold on validation data, and persist one artifact.

The test split is scored once, after all choices are frozen. Nothing in this
module prints a metric as if it were a target: every number it writes comes from
the run that produced it.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from . import config, evaluate
from .data import load_all_splits, sha256_file
from .features import build_vectorizer


def build_pipeline(ngram_range: tuple[int, int] = (1, 2), c_value: float = 1.0) -> Pipeline:
    """TF-IDF followed by class-weighted logistic regression."""
    return Pipeline(
        steps=[
            ("tfidf", build_vectorizer(ngram_range)),
            (
                "clf",
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    max_iter=config.MAX_ITER,
                    solver="liblinear",
                    random_state=config.SEED,
                ),
            ),
        ]
    )


def config_fingerprint() -> str:
    """Stable hash of the settings that determine the model's behaviour."""
    payload = {
        "tfidf": {
            "min_df": config.TFIDF_MIN_DF,
            "max_features": config.TFIDF_MAX_FEATURES,
            "sublinear_tf": config.TFIDF_SUBLINEAR_TF,
            "ngram_choices": [list(pair) for pair in config.NGRAM_CHOICES],
        },
        "model": {
            "type": "LogisticRegression",
            "class_weight": "balanced",
            "solver": "liblinear",
            "max_iter": config.MAX_ITER,
        },
        "c_grid": list(config.C_GRID),
        "seed": config.SEED,
        "threshold": {
            "beta": config.THRESHOLD_BETA,
            "min": config.THRESHOLD_MIN,
            "max": config.THRESHOLD_MAX,
            "steps": config.THRESHOLD_STEPS,
            "selected_on": "validation",
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def search_hyperparameters(
    train_frame: pd.DataFrame, folds: int = config.CV_FOLDS
) -> dict:
    """Small deterministic grid search on the training split only."""
    smallest_class = int(train_frame["label"].value_counts().min())
    effective_folds = max(2, min(folds, smallest_class))
    splitter = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=config.SEED)

    results: list[dict] = []
    for ngram_range in config.NGRAM_CHOICES:
        for c_value in config.C_GRID:
            pipeline = build_pipeline(ngram_range, c_value)
            scores = cross_val_score(
                pipeline,
                train_frame["text"],
                train_frame["label"],
                cv=splitter,
                scoring=make_scorer(f1_score, pos_label=config.POSITIVE_LABEL),
                error_score="raise",
            )
            results.append(
                {
                    "ngram_range": list(ngram_range),
                    "C": c_value,
                    "cv_f1_mean": float(np.mean(scores)),
                    "cv_f1_std": float(np.std(scores)),
                    "folds": effective_folds,
                }
            )

    best = max(results, key=lambda row: (row["cv_f1_mean"], -row["C"]))
    return {"best": best, "results": results}


def select_threshold(y_true, scores) -> float:
    """Pick the validation threshold maximising F-beta (beta > 1 favours recall)."""
    truth = (np.asarray(y_true) == config.POSITIVE_LABEL).astype(int)
    scores = np.asarray(scores, dtype=float)
    beta_squared = config.THRESHOLD_BETA**2

    best_threshold = 0.5
    best_score = -1.0
    for threshold in np.linspace(config.THRESHOLD_MIN, config.THRESHOLD_MAX, config.THRESHOLD_STEPS):
        predicted = (scores >= threshold).astype(int)
        true_positive = int(((predicted == 1) & (truth == 1)).sum())
        false_positive = int(((predicted == 1) & (truth == 0)).sum())
        false_negative = int(((predicted == 0) & (truth == 1)).sum())

        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        denominator = beta_squared * precision + recall
        fbeta = (1 + beta_squared) * precision * recall / denominator if denominator else 0.0

        if fbeta >= best_score:
            best_score = fbeta
            best_threshold = float(threshold)
    return round(best_threshold, 4)


def _predicted_labels(scores, threshold: float) -> np.ndarray:
    return np.where(np.asarray(scores) >= threshold, config.POSITIVE_LABEL, "ham")


def _validation_metrics(labels, scores, threshold: float) -> dict:
    return evaluate.binary_metrics(labels, _predicted_labels(scores, threshold), scores)


def train_and_persist(
    model_path: Path | None = None,
    metrics_path: Path | None = None,
    reports_dir: Path | None = None,
) -> dict:
    """Run the full training job and write the artifact, checksum and reports."""
    model_path = Path(model_path or config.MODEL_PATH)
    reports_dir = Path(reports_dir or config.REPORTS_DIR)
    reports_dir.mkdir(parents=True, exist_ok=True)

    splits = load_all_splits()
    train_frame, val_frame, test_frame = splits["train"], splits["val"], splits["test"]
    dataset_hash = sha256_file(config.RAW_DATASET)

    search = search_hyperparameters(train_frame)
    best = search["best"]
    ngram_range = tuple(best["ngram_range"])
    c_value = float(best["C"])

    validation_pipeline = build_pipeline(ngram_range, c_value).fit(
        train_frame["text"], train_frame["label"]
    )
    val_scores = evaluate.spam_scores(validation_pipeline, val_frame["text"])
    threshold = select_threshold(val_frame["label"], val_scores)
    val_metrics = _validation_metrics(val_frame["label"], val_scores, threshold)

    # Refit on train + validation with the frozen settings, then touch test once.
    combined = pd.concat([train_frame, val_frame], ignore_index=True)
    combined = combined.sort_values("dedupe_key").reset_index(drop=True)
    pipeline = build_pipeline(ngram_range, c_value).fit(combined["text"], combined["label"])

    test_scores = evaluate.spam_scores(pipeline, test_frame["text"])
    test_pred = _predicted_labels(test_scores, threshold)
    test_metrics = evaluate.binary_metrics(test_frame["label"], test_pred, test_scores)

    baselines = evaluate.build_baselines(train_frame, test_frame)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    metrics_payload = {
        "generated_at": now,
        "dataset": {
            "sha256": dataset_hash,
            "raw_rows": config.EXPECTED_RAW_ROWS,
            "splits": {
                "train": int(len(train_frame)),
                "val": int(len(val_frame)),
                "test": int(len(test_frame)),
            },
            "test_evaluated_once": True,
        },
        "model": {
            "type": "LogisticRegression",
            "C": c_value,
            "ngram_range": list(ngram_range),
            "class_weight": "balanced",
            "max_iter": config.MAX_ITER,
            "threshold": threshold,
            "threshold_selected_on": "validation",
            "threshold_objective": f"max F{config.THRESHOLD_BETA:g}",
            "tfidf": {
                "min_df": config.TFIDF_MIN_DF,
                "max_features": config.TFIDF_MAX_FEATURES,
                "sublinear_tf": config.TFIDF_SUBLINEAR_TF,
            },
        },
        "selection": {"cv_results": search["results"], "chosen": best},
        "validation": val_metrics,
        "test": test_metrics,
        "baselines": baselines,
        "versions": evaluate.library_versions(),
    }

    evaluate.write_metrics(metrics_payload, reports_dir / "metrics.json")
    evaluate.write_baseline_csv(
        evaluate.comparison_rows(test_metrics, baselines), reports_dir / "baseline_comparison.csv"
    )
    evaluate.write_confusion_csv(test_metrics, reports_dir / "confusion_matrix.csv")

    artifact = {
        "pipeline": pipeline,
        "threshold": threshold,
        "labels": list(config.LABELS),
        "model_params": {
            "C": c_value,
            "ngram_range": list(ngram_range),
            "class_weight": "balanced",
            "max_iter": config.MAX_ITER,
            "solver": "liblinear",
        },
        "seed": config.SEED,
        "config_hash": config_fingerprint(),
        "dataset_hash": dataset_hash,
        "sklearn_version": sklearn.__version__,
        "scamdetector_version": evaluate.library_versions()["scamdetector"],
        "trained_at": now,
        "metrics": {"validation": val_metrics, "test": test_metrics},
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path, compress=3)
    digest = sha256_file(model_path)
    checksum_path = model_path.with_suffix(model_path.suffix + ".sha256")
    checksum_path.write_text(digest + "\n", encoding="utf-8")

    size_bytes = model_path.stat().st_size
    return {
        "artifact": str(model_path),
        "size_bytes": size_bytes,
        "committable": size_bytes <= config.ARTIFACT_MAX_BYTES,
        "sha256": digest,
        "threshold": threshold,
        "chosen_params": best,
        "validation": val_metrics,
        "test": test_metrics,
        "baselines": baselines,
        "metrics_path": str(reports_dir / "metrics.json"),
    }

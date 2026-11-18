import json

import pytest

from scamdetector import config
from scamdetector.evaluate import load_metrics

# Floor taken from the accepted run's recorded test F1 (0.9394). A regression test
# should fail on a real drop, not on tiny floating-point drift.
RECORDED_TEST_F1_FLOOR = 0.9194


@pytest.fixture(scope="module")
def metrics():
    if not config.METRICS_PATH.exists():
        pytest.skip("metrics.json not generated yet; run scripts\\train_model.py")
    return load_metrics()


@pytest.mark.integration
def test_recorded_test_f1_has_not_regressed(metrics):
    assert metrics["test"]["f1"] >= RECORDED_TEST_F1_FLOOR


@pytest.mark.integration
def test_test_split_size_matches_the_prepared_splits(metrics):
    assert metrics["test"]["rows"] == 515
    assert metrics["dataset"]["splits"]["test"] == 515


@pytest.mark.integration
def test_threshold_was_selected_on_validation(metrics):
    assert metrics["model"]["threshold_selected_on"] == "validation"
    assert 0.0 < metrics["model"]["threshold"] < 1.0


@pytest.mark.integration
def test_model_beats_the_majority_baseline(metrics):
    assert metrics["test"]["f1"] > metrics["baselines"]["majority_class"]["f1"]
    assert metrics["test"]["recall"] > metrics["baselines"]["majority_class"]["recall"]


@pytest.mark.integration
def test_reports_are_json_serialisable_and_complete(metrics):
    assert set(metrics) >= {"dataset", "model", "validation", "test", "baselines", "versions"}
    json.dumps(metrics)
    for key in ("precision", "recall", "f1", "roc_auc", "pr_auc"):
        assert 0.0 <= metrics["test"][key] <= 1.0


@pytest.mark.integration
def test_confusion_matrix_counts_match_support(metrics):
    matrix = metrics["test"]["confusion_matrix"]
    support = metrics["test"]["support"]
    assert matrix["tn"] + matrix["fp"] == support["ham"]
    assert matrix["fn"] + matrix["tp"] == support["spam"]

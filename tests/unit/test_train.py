import numpy as np
import pytest

from scamdetector import config
from scamdetector.train import build_pipeline, config_fingerprint, select_threshold


@pytest.mark.unit
def test_pipeline_has_expected_steps():
    pipeline = build_pipeline((1, 2), 10.0)
    assert list(pipeline.named_steps) == ["tfidf", "clf"]
    assert pipeline.named_steps["clf"].C == 10.0
    assert pipeline.named_steps["tfidf"].ngram_range == (1, 2)


@pytest.mark.unit
def test_threshold_sits_between_the_classes():
    labels = ["ham"] * 20 + ["spam"] * 20
    scores = np.concatenate([np.linspace(0.02, 0.30, 20), np.linspace(0.55, 0.98, 20)])
    threshold = select_threshold(labels, scores)
    assert 0.30 < threshold <= 0.55


@pytest.mark.unit
def test_threshold_is_rounded_to_four_places():
    labels = ["ham", "spam"]
    scores = np.array([0.2, 0.8])
    threshold = select_threshold(labels, scores)
    assert threshold == round(threshold, 4)


@pytest.mark.unit
def test_threshold_favours_recall_over_precision():
    # 10 positives: 9 easy (0.8), 1 borderline (0.45); 1 negative at 0.5.
    labels = ["spam"] * 10 + ["ham"]
    scores = np.array([0.8] * 9 + [0.45] + [0.5])
    threshold = select_threshold(labels, scores)
    assert threshold <= 0.5, "a recall-first threshold must keep the borderline positive"


@pytest.mark.unit
def test_config_fingerprint_is_stable_and_hex():
    first = config_fingerprint()
    assert first == config_fingerprint()
    assert len(first) == 64
    int(first, 16)


@pytest.mark.unit
def test_pipeline_fit_is_deterministic(toy_frame):
    first = build_pipeline((1, 1), 1.0).fit(toy_frame["text"], toy_frame["label"])
    second = build_pipeline((1, 1), 1.0).fit(toy_frame["text"], toy_frame["label"])
    assert np.allclose(
        first.named_steps["clf"].coef_, second.named_steps["clf"].coef_
    )
    assert np.array_equal(first.named_steps["tfidf"].get_feature_names_out(), second.named_steps["tfidf"].get_feature_names_out())


@pytest.mark.unit
def test_random_state_is_the_configured_seed():
    pipeline = build_pipeline((1, 1), 1.0)
    assert pipeline.named_steps["clf"].random_state == config.SEED

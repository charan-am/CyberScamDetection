import joblib
import pytest

from scamdetector import config
from scamdetector.predict import load_artifact, predict_message, sha256_file
from scamdetector.train import build_pipeline


@pytest.mark.integration
def test_train_save_load_predict_roundtrip(tmp_path, toy_frame):
    pipeline = build_pipeline((1, 1), 1.0).fit(toy_frame["text"], toy_frame["label"])
    artifact = {
        "pipeline": pipeline,
        "threshold": 0.5,
        "labels": list(config.LABELS),
        "model_params": {"C": 1.0, "ngram_range": [1, 1]},
        "config_hash": "roundtrip",
        "dataset_hash": "toy",
        "sklearn_version": "1.9.1",
        "trained_at": "2026-11-18T00:00:00+00:00",
    }
    model_path = tmp_path / "roundtrip.joblib"
    joblib.dump(artifact, model_path, compress=3)
    (tmp_path / "roundtrip.joblib.sha256").write_text(sha256_file(model_path) + "\n", encoding="utf-8")

    loaded = load_artifact(model_path, tmp_path / "roundtrip.joblib.sha256")
    result = predict_message("free prize win", artifact=loaded)

    assert result.label == "spam"
    assert loaded["threshold"] == 0.5
    load_artifact.cache_clear()


@pytest.mark.integration
def test_predictions_are_repeatable(real_artifact):
    message = "You have won a cash prize, call now to claim"
    first = predict_message(message, artifact=real_artifact)
    second = predict_message(message, artifact=real_artifact)
    assert first == second


@pytest.fixture
def real_artifact():
    if not config.MODEL_PATH.exists():
        pytest.skip("real artifact not trained yet; run scripts\\train_model.py")
    return load_artifact()

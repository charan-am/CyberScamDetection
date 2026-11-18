import pytest

from scamdetector import config
from scamdetector.predict import ArtifactError, load_artifact, predict_message, verify_artifact


@pytest.fixture
def real_artifact():
    if not config.MODEL_PATH.exists():
        pytest.skip("real artifact not trained yet; run scripts\\train_model.py")
    return load_artifact()


@pytest.mark.integration
def test_recorded_artifact_verifies(real_artifact):
    digest = verify_artifact()
    assert len(digest) == 64
    assert real_artifact["dataset_hash"] == config.DATASET_SHA256


@pytest.mark.integration
def test_real_model_flags_a_prize_message(real_artifact):
    result = predict_message(
        "WINNER!! You have won a free prize. Claim now by calling 09061701461",
        artifact=real_artifact,
    )
    assert result.label == "spam"
    assert result.raw_score >= result.threshold
    assert result.top_positive, "spam prediction should have at least one positive term"


@pytest.mark.integration
def test_real_model_accepts_an_ordinary_message(real_artifact):
    result = predict_message("See you at the station at 6, bring the tickets", artifact=real_artifact)
    assert result.label == "ham"
    assert result.flagged is False


@pytest.mark.integration
def test_tampered_copy_of_real_artifact_is_refused(tmp_path, real_artifact):
    import shutil

    from scamdetector.predict import load_artifact as load

    model_copy = tmp_path / "model.joblib"
    checksum_copy = tmp_path / "model.joblib.sha256"
    shutil.copyfile(config.MODEL_PATH, model_copy)
    checksum_copy.write_text(verify_artifact() + "\n", encoding="utf-8")
    model_copy.write_bytes(model_copy.read_bytes()[:-1] + b"\x00")

    with pytest.raises(ArtifactError, match="checksum"):
        load(model_copy, checksum_copy)
    load.cache_clear()


@pytest.mark.integration
def test_model_artifact_is_small_enough_to_commit(real_artifact):
    assert config.MODEL_PATH.stat().st_size <= config.ARTIFACT_MAX_BYTES

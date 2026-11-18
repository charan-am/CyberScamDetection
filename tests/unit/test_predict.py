import pytest

from scamdetector import config
from scamdetector.explain import Contribution
from scamdetector.predict import (
    ArtifactError,
    MessageValidationError,
    is_probably_non_english,
    load_artifact,
    predict_message,
    validate_message,
    verify_artifact,
)


@pytest.mark.unit
@pytest.mark.parametrize("value", ["", "   ", "\n\t "])
def test_empty_messages_are_rejected(value):
    with pytest.raises(MessageValidationError, match="Enter a message"):
        validate_message(value)


@pytest.mark.unit
def test_oversized_message_is_rejected_with_counts():
    with pytest.raises(MessageValidationError, match="limit is 5,000"):
        validate_message("a" * (config.MAX_MESSAGE_CHARS + 1))


@pytest.mark.unit
def test_non_text_input_is_rejected():
    with pytest.raises(ValueError, match="must be text"):
        validate_message(42)


@pytest.mark.unit
def test_valid_message_is_trimmed():
    assert validate_message("  hello  ") == "hello"


@pytest.mark.unit
def test_non_english_heuristic_flags_kannada_only_sentences():
    assert is_probably_non_english("ನಿಮ್ಮ ಖಾತೆಯಿಂದ ಹಣ ಕಳೆದುಹೋಗಿದೆ ಎಂದು ಹೇಳಲಾಗಿದೆ")
    assert not is_probably_non_english("Your account has been blocked, call now")


@pytest.mark.unit
def test_prediction_contract_for_spam_like_text(tiny_artifact):
    result = predict_message("win free prize claim now", artifact=tiny_artifact["artifact"])
    assert result.label in config.LABELS
    assert result.flagged is (result.raw_score >= result.threshold)
    assert 0.0 <= result.raw_score <= 1.0
    assert result.message_length == len("win free prize claim now")
    assert all(isinstance(item, Contribution) for item in result.top_positive)


@pytest.mark.unit
def test_prediction_flags_ham_like_text(tiny_artifact):
    result = predict_message("meet me at the station tomorrow", artifact=tiny_artifact["artifact"])
    assert result.label == "ham"
    assert result.flagged is False


@pytest.mark.unit
def test_verify_artifact_rejects_missing_model(tmp_path):
    with pytest.raises(ArtifactError, match="not found"):
        verify_artifact(tmp_path / "absent.joblib", tmp_path / "absent.sha256")


@pytest.mark.unit
def test_verify_artifact_rejects_missing_checksum(tmp_path, tiny_artifact):
    with pytest.raises(ArtifactError, match="Checksum file not found"):
        verify_artifact(tiny_artifact["model_path"], tmp_path / "absent.sha256")


@pytest.mark.unit
def test_tampered_artifact_is_refused_before_loading(tiny_artifact, monkeypatch):
    tampered = tiny_artifact["model_path"].read_bytes() + b"tamper"
    tiny_artifact["model_path"].write_bytes(tampered)

    def explode(*args, **kwargs):
        raise AssertionError("joblib.load must not run on a failed checksum")

    monkeypatch.setattr("scamdetector.predict.joblib.load", explode)
    with pytest.raises(ArtifactError, match="checksum"):
        load_artifact(tiny_artifact["model_path"], tiny_artifact["checksum_path"])
    load_artifact.cache_clear()


@pytest.mark.unit
def test_artifact_without_metadata_is_refused(tmp_path, monkeypatch):
    import joblib

    from scamdetector.predict import sha256_file

    model_path = tmp_path / "incomplete.joblib"
    joblib.dump({"pipeline": None}, model_path)
    checksum_path = tmp_path / "incomplete.joblib.sha256"
    checksum_path.write_text(sha256_file(model_path) + "\n", encoding="utf-8")

    with pytest.raises(ArtifactError, match="missing required metadata"):
        load_artifact(model_path, checksum_path)
    load_artifact.cache_clear()

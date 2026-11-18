import joblib
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from scamdetector import config
from scamdetector.predict import sha256_file
from scamdetector.train import build_pipeline


@pytest.fixture
def toy_frame() -> pd.DataFrame:
    """A tiny, clearly separable corpus used for fast model tests."""
    rows = []
    for index in range(40):
        rows.append(("spam", f"win free prize claim now {index}"))
        rows.append(("ham", f"meet me at the station tomorrow {index}"))
    return pd.DataFrame(rows, columns=["label", "text"])


@pytest.fixture
def tiny_pipeline(toy_frame) -> Pipeline:
    pipeline = build_pipeline((1, 1), 1.0)
    pipeline.fit(toy_frame["text"], toy_frame["label"])
    return pipeline


@pytest.fixture
def tiny_artifact(tmp_path, tiny_pipeline) -> dict:
    artifact = {
        "pipeline": tiny_pipeline,
        "threshold": 0.5,
        "labels": list(config.LABELS),
        "model_params": {"C": 1.0, "ngram_range": [1, 1]},
        "config_hash": "test-config",
        "dataset_hash": "test-dataset",
        "sklearn_version": "1.9.1",
        "trained_at": "2026-11-18T00:00:00+00:00",
        "scamdetector_version": "test",
        "metrics": {},
    }
    model_path = tmp_path / "toy.joblib"
    checksum_path = tmp_path / "toy.joblib.sha256"
    joblib.dump(artifact, model_path, compress=3)
    checksum_path.write_text(sha256_file(model_path) + "\n", encoding="utf-8")
    return {"artifact": artifact, "model_path": model_path, "checksum_path": checksum_path}

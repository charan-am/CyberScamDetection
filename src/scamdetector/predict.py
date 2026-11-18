"""Artifact verification, input validation and inference.

This module never trains and never downloads anything. It refuses to load a
``joblib`` file whose checksum does not match, because unpickling executes code.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import joblib
import sklearn

from . import config
from .explain import Contribution, top_contributions
from .features import ensure_text

REQUIRED_ARTIFACT_KEYS = (
    "pipeline",
    "threshold",
    "labels",
    "model_params",
    "config_hash",
    "dataset_hash",
    "sklearn_version",
    "trained_at",
)


class ArtifactError(RuntimeError):
    """Raised when the model artifact is missing, tampered with or incompatible."""


class MessageValidationError(ValueError):
    """Raised when submitted text cannot be analysed."""


@dataclass(frozen=True)
class PredictionResult:
    """Typed outcome of one analysis. Holds no curated guidance and no verdict."""

    label: str
    raw_score: float
    threshold: float
    flagged: bool
    message_length: int
    top_positive: tuple[Contribution, ...]
    top_negative: tuple[Contribution, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_artifact(
    model_path: Path | None = None,
    checksum_path: Path | None = None,
) -> str:
    """Verify the artifact's SHA-256 before any deserialisation happens."""
    model_path = Path(model_path or config.MODEL_PATH)
    checksum_path = Path(checksum_path or config.MODEL_CHECKSUM_PATH)

    if not model_path.exists():
        raise ArtifactError(
            f"Model artifact not found at {model_path}. "
            "Train it with: python scripts\\train_model.py"
        )
    if not checksum_path.exists():
        raise ArtifactError(
            f"Checksum file not found at {checksum_path}. "
            "Re-run: python scripts\\train_model.py"
        )

    expected = checksum_path.read_text(encoding="utf-8").strip().lower()
    actual = sha256_file(model_path)
    if expected != actual:
        raise ArtifactError(
            "Model artifact checksum does not match; refusing to load it. "
            "Re-run: python scripts\\train_model.py"
        )
    return actual


def _compatible_versions(artifact_version: str) -> bool:
    def major_minor(version: str) -> tuple[str, str]:
        parts = version.split(".")
        return parts[0], parts[1] if len(parts) > 1 else "0"

    return major_minor(artifact_version) == major_minor(sklearn.__version__)


@lru_cache(maxsize=4)
def load_artifact(
    model_path: Path = config.MODEL_PATH,
    checksum_path: Path = config.MODEL_CHECKSUM_PATH,
) -> dict:
    """Verify, then load the project-local artifact once per process."""
    verify_artifact(model_path, checksum_path)
    artifact = joblib.load(model_path)
    missing = [key for key in REQUIRED_ARTIFACT_KEYS if key not in artifact]
    if missing:
        raise ArtifactError(f"Artifact is missing required metadata: {missing}")
    if not _compatible_versions(str(artifact["sklearn_version"])):
        raise ArtifactError(
            "Artifact was trained with scikit-learn "
            f"{artifact['sklearn_version']} but {sklearn.__version__} is installed. "
            "Re-train with: python scripts\\train_model.py"
        )
    return artifact


def load_model() -> dict:
    """Public accessor used by the Streamlit layer."""
    return load_artifact()


def validate_message(text: object) -> str:
    """Return cleaned text or raise a readable validation error."""
    raw = ensure_text(text).replace("\x00", "")
    cleaned = raw.strip()
    if not cleaned:
        raise MessageValidationError("Enter a message to analyse.")
    if len(cleaned) > config.MAX_MESSAGE_CHARS:
        raise MessageValidationError(
            f"Message is {len(cleaned):,} characters; the limit is "
            f"{config.MAX_MESSAGE_CHARS:,}. Shorten it and try again."
        )
    return cleaned


def is_probably_non_english(text: str) -> bool:
    """Heuristic advisory only: many letters, few ASCII letters."""
    letters = [character for character in text if character.isalpha()]
    if len(letters) < 20:
        return False
    ascii_letters = sum(1 for character in letters if character.isascii())
    return ascii_letters / len(letters) < 0.3


def predict_message(text: object, artifact: dict | None = None) -> PredictionResult:
    """Analyse one message. The message is never logged or persisted."""
    cleaned = validate_message(text)
    artifact = artifact or load_model()

    pipeline = artifact["pipeline"]
    threshold = float(artifact["threshold"])
    raw_score = float(pipeline.predict_proba([cleaned])[0][1])
    flagged = raw_score >= threshold
    positive, negative = top_contributions(pipeline, cleaned)

    return PredictionResult(
        label=config.POSITIVE_LABEL if flagged else "ham",
        raw_score=raw_score,
        threshold=threshold,
        flagged=flagged,
        message_length=len(cleaned),
        top_positive=tuple(positive),
        top_negative=tuple(negative),
    )

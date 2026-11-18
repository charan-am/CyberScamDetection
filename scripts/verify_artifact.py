"""Verify the committed artifact before a demo or a fresh clone.

Checks the SHA-256, the recorded scikit-learn version and whether the dataset on
disk still matches the one the model was trained on.

Usage::

    python scripts\\verify_artifact.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import sklearn  # noqa: E402

from scamdetector import config  # noqa: E402
from scamdetector.data import sha256_file  # noqa: E402
from scamdetector.predict import ArtifactError, load_artifact, verify_artifact  # noqa: E402


def main() -> int:
    try:
        digest = verify_artifact()
        artifact = load_artifact()
    except ArtifactError as error:
        print(f"FAILED: {error}")
        return 1

    current_dataset = sha256_file(config.RAW_DATASET)
    dataset_matches = current_dataset == artifact["dataset_hash"]

    print("Artifact OK")
    print(f"  sha256:          {digest}")
    print(f"  trained_at:      {artifact['trained_at']}")
    print(f"  threshold:       {artifact['threshold']:.4f}")
    print(f"  sklearn (file):  {artifact['sklearn_version']}")
    print(f"  sklearn (local): {sklearn.__version__}")
    print(f"  dataset sha256:  {artifact['dataset_hash']}")
    print(f"  dataset match:   {'yes' if dataset_matches else 'NO - retrain before claiming metrics'}")
    return 0 if dataset_matches else 1


if __name__ == "__main__":
    raise SystemExit(main())

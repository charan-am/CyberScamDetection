# Model Artifact

| File | Purpose |
|---|---|
| `scam_clf.joblib` | Fitted pipeline plus metadata (committed, under 10 MB) |
| `scam_clf.joblib.sha256` | SHA-256 of the artifact, verified before loading |

## Current artifact

| Field | Value |
|---|---|
| Size | 194,174 bytes (0.19 MB) |
| SHA-256 | `e0b3ee1bf70c4e5aa6c3bf08c673d4161c8324f3c0f51e076c710c8a18987a3d` |
| Trained | 2026-11-18T08:48:38+00:00 |
| scikit-learn | 1.9.1 |
| Threshold | 0.2700 (chosen on validation, max F2) |
| Dataset SHA-256 | `7d039a24a6083ed9ef0f806ebad56bbb976e3aeb8de05669173bfdc4996c239d` |

The artifact is committed because it is far below the 10 MB limit agreed in the
approved design. If a future retrain exceeds 10 MB, delete the artifact from Git,
commit only this README with the new size, and document that users must train first.

## Regenerating

```cmd
python scripts\prepare_data.py
python scripts\train_model.py
python scripts\verify_artifact.py
```

Training is deterministic: fixed seed 42, `liblinear` solver, fixed split files.
Re-running on the same data reproduces the same metrics.

## What the artifact contains

`pipeline` (TF-IDF + LogisticRegression), `threshold`, `labels`, `model_params`,
`seed`, `config_hash`, `dataset_hash`, `sklearn_version`, `scamdetector_version`,
`trained_at`, and the validation/test metrics from the accepted run.

## Load-safety note

`joblib` uses `pickle`, which can execute arbitrary code on load. This project
therefore loads artifacts **only** from this directory, and only after
`verify_artifact()` has matched the file against the committed SHA-256. Never point
the app at a downloaded or user-supplied model file. If the checksum file is
missing, loading is refused rather than assumed safe.

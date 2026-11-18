# Architecture

## Layers

```
┌──────────────────────────────────────────────────────────────┐
│ Presentation        app/streamlit_app.py                     │
│  four tabs; no training; loads the artifact once per process │
└───────────────────────────┬──────────────────────────────────┘
                            │ PredictionResult, content dataclasses
┌───────────────────────────▼──────────────────────────────────┐
│ Domain              src/scamdetector/                        │
│  config · data · features · train · evaluate · predict       │
│  explain · present · awareness                               │
└───────────────────────────┬──────────────────────────────────┘
                            │ joblib artifact + checksum, reports/
┌───────────────────────────▼──────────────────────────────────┐
│ Artifacts           models/, reports/, data/processed/       │
└──────────────────────────────────────────────────────────────┘
```

Rules enforced by the code layout and tests:

- `train.py` never imports Streamlit; `app/` never trains.
- Prediction requires a checksum match and a compatible scikit-learn version.
- `present.py` contains every user-facing sentence and imports no Streamlit, so wording
  can be unit tested.
- Awareness content is data (`content/*.json`), validated on load, not code.

## Module responsibilities

| Module | Responsibility |
|---|---|
| `config.py` | Paths, seed, split ratios, TF-IDF and model hyper-parameters, threshold search bounds, message limits, curated term lists |
| `data.py` | Parse the raw UCI file, normalise and dedupe, resolve conflicting labels, stratified splitting, leakage verification, CSV persistence |
| `features.py` | Sensitive-token masking, text normalisation, TF-IDF vectoriser construction |
| `train.py` | Pipeline construction, config fingerprint, cross-validated search, validation threshold selection, final refit, artifact and report writing |
| `evaluate.py` | Binary metrics, spam scores, majority and keyword baselines, report writers, metrics loader |
| `predict.py` | Checksum verification, artifact loading and caching, input validation, inference, non-English heuristic |
| `explain.py` | Exact linear-model term contributions and rule-based message signals |
| `present.py` | Plain-language labels, honest score captions, table rows, disclaimer and privacy text |
| `awareness.py` | Schema validation and loading of bilingual content and reporting channels |

## Training flow

1. `scripts/get_data.py` downloads the UCI archive, verifies the SHA-256 and row count,
   and keeps `data/raw/SMSSpamCollection`.
2. `scripts/prepare_data.py` parses the file, removes exact duplicates (reported in
   `reports/conflict_report.json`), splits 80/10/10 with stratification and seed 42,
   asserts zero leakage, writes CSVs and `reports/split_summary.json`.
3. `scripts/train_model.py` runs a 5-fold cross-validated search over `C` and n-gram
   range, picks the threshold that maximises F2 on validation, refits on train +
   validation, scores the test split exactly once, and writes:
   - `models/scam_clf.joblib` and its `.sha256` checksum;
   - `reports/metrics.json`, `reports/baseline_comparison.csv`,
     `reports/confusion_matrix.csv`.
4. `scripts/verify_artifact.py` recomputes the checksum and checks the dataset hash.

## Inference flow

```
text ──► validate_message ──► normalise + mask ──► TF-IDF ──► LogisticRegression
                                                                   │
                          raw spam score ──► compare with threshold 0.27
                                                                   │
                    PredictionResult(label, raw_score, threshold, contributions)
                                                                   │
              app: label card → explanation tables → rule signals → guidance
```

The submitted message is never written to disk, logged, or sent anywhere.

## Failure handling

| Failure | Behaviour |
|---|---|
| Empty or whitespace message | `MessageValidationError`, shown as a warning |
| Message over 5,000 characters | `MessageValidationError` |
| Artifact missing | `ArtifactError` with the training command to run |
| Checksum mismatch | `ArtifactError`; the file is not deserialised |
| scikit-learn major/minor mismatch | `ArtifactError` asking for retraining |
| Missing or invalid content file | `ContentError`, shown in the tab instead of crashing |
| Missing metrics file | Performance tab explains how to generate it |

## Reproducibility

- Fixed seed 42 for splitting and model training.
- Pinned dependencies in `requirements.txt` and `requirements-dev.txt`.
- Artifact metadata stores dataset hash, config hash, scikit-learn version and timestamp.
- Re-running preparation produced byte-identical CSVs during development.
- Tests include a metric-regression floor (F1 ≥ 0.9194) so a silently degraded model
  fails the suite.

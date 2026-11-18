# AI-Based Cyber Scam Detection and Awareness System

An explainable machine-learning application that classifies English SMS messages as
ordinary (ham) or scam-like (spam), explains the model's decision, and teaches Indian
users to recognise and report real scam patterns in **English and Kannada**.

The project was built as a PBL (project-based learning) submission. It is a genuine
end-to-end ML system: a real public dataset, reproducible training, a checksummed model
artifact, a tested Streamlit interface, and documented performance measured on held-out
data. No metric in this repository is invented.

---

## What the application does

Four user-facing areas, available as tabs in one Streamlit app:

| Area | Contents |
|---|---|
| **Message detection** | Paste an SMS, get the predicted class, an animated 0–10 **Model Spam Signal** gauge with the real decision threshold marked, the words that moved the decision, rule-based message characteristics, related awareness topics and safety guidance |
| **Cyber awareness** | Seven Indian scam categories — UPI, OTP, phishing, fake jobs, lottery/prize, investment, banking impersonation/KYC — in English and Kannada, plus official reporting channels with verified dates |
| **Model performance** | Real test-split metrics, confusion matrix, baseline comparison, model configuration, and an explicit list of what the numbers do not cover |
| **About & privacy** | How a prediction is produced, dataset attribution, model artifact details, honest local-versus-hosted privacy explanation, licences |

The three analysis layers are deliberately kept separate and labelled in the interface:

1. **Model prediction and explanation** — logistic-regression coefficients × TF-IDF values.
2. **Rule-based characteristics** — links, OTP requests, prize claims, urgency, amounts.
3. **Verified awareness guidance** — curated advice from `content/*.json`, independent of the model.

The classifier is **English-only**. The interface detects Kannada script and does not
send those messages through the English model; it explains that Kannada detection is
not available yet instead of guessing. Kannada awareness content exists but is marked
`reviewed: false` until a Kannada speaker verifies it. Romanised Kannada is not
supported. The dataset assessment is in `docs/kannada-assessment.md`.

---

## Quick start (Windows, CMD)

Requirements: Windows, [uv](https://docs.astral.sh/uv/), network access for the first
setup. Python 3.12.16 is installed and pinned by uv; Python 3.14 is rejected by the
test suite. Use `uv pip`, never bare `pip` (blocked on the development machine), and
launch Streamlit through `python -m streamlit` (the `.exe` launcher may be blocked by
Application Control).

```cmd
cd /d "%USERPROFILE%\CyberScamDetection"
uv venv --python 3.12.16 .venv
.venv\Scripts\activate
uv pip install -r requirements.txt -r requirements-dev.txt
uv pip install -e .
python -m pytest
python -m streamlit run app\streamlit_app.py
```

The trained artifact is committed, so the app runs immediately after setup. If you want
to reproduce training from the raw dataset:
```cmd
python scripts\get_data.py
python scripts\prepare_data.py
python scripts\train_model.py
python scripts\verify_artifact.py
```

`python -m pytest` runs the full suite (99 tests). The expected result is
`99 passed`.

---

## Results (measured, not estimated)

Trained on 4,115 messages, threshold chosen on a 514-message validation split, evaluated
**once** on a 515-message test split that no earlier step used.

| Metric (test split) | Value |
|---|---|
| Accuracy | 0.9845 |
| Precision (spam) | 0.8986 |
| Recall (spam) | 0.9841 |
| F1 (spam) | 0.9394 |
| ROC-AUC | 0.9981 |
| PR-AUC | 0.9889 |

Confusion matrix: 445 true ham, 7 false spam, 1 missed spam, 62 true spam.

| Model (same test split) | Precision | Recall | F1 |
|---|---|---|---|
| TF-IDF + Logistic Regression | 0.8986 | 0.9841 | **0.9394** |
| Majority class (always ham) | 0.0000 | 0.0000 | 0.0000 |
| Single-keyword rule | 0.5843 | 0.8254 | 0.6842 |

Full details are in `reports/metrics.json`, `reports/baseline_comparison.csv`,
`reports/confusion_matrix.csv` and `docs/model-card.md`.

The dataset is small, from 2011, and SMS-only. These numbers describe performance on
that dataset, not on modern WhatsApp, email or voice scams.

---

## Repository layout

```
CyberScamDetection/
├── app/
│   └── streamlit_app.py          # the only Streamlit entry point
├── content/
│   ├── awareness_en.json         # reviewed English awareness content
│   ├── awareness_kn.json         # Kannada content, review status false
│   └── reporting.json            # official Indian reporting channels + dates
├── data/
│   ├── raw/                      # original UCI files (attribution in data/README.md)
│   └── processed/                # generated splits (git-ignored, reproducible)
├── docs/
│   ├── dataset.md                # provenance, licence, integrity checks
│   ├── model-card.md             # intended use, metrics, limitations, ethics
│   ├── architecture.md           # layers, data flow, module responsibilities
│   ├── demo-script.md            # five-minute demonstration script
│   ├── deployment.md             # Streamlit Community Cloud readiness and privacy
│   ├── kannada-assessment.md     # dataset feasibility findings and decision
│   ├── verification.md           # test and runtime evidence
│   ├── phase0-verification.md    # environment evidence
│   └── superpowers/              # approved design and plans
├── models/
│   ├── scam_clf.joblib           # 194 KB trained pipeline
│   ├── scam_clf.joblib.sha256    # checksum verified before every load
│   └── README.md
├── reports/                      # metrics, baselines, split and conflict reports
├── scripts/                      # get_data, prepare_data, train_model, verify_artifact
├── src/scamdetector/             # domain package (no Streamlit imports)
│   ├── config.py                 # paths, seed, hyper-parameters, term lists
│   ├── data.py                   # loading, dedupe, leakage-free splits
│   ├── features.py               # masking, normalisation, TF-IDF
│   ├── train.py                  # search, threshold selection, persistence
│   ├── evaluate.py               # metrics, baselines, reports
│   ├── predict.py                # checksum verification, validation, inference
│   ├── explain.py                # contributions and rule-based signals
│   ├── language.py               # script detection for English/Kannada routing
│   ├── guidance.py               # message characteristics -> awareness topics
│   ├── present.py                # honest UI strings and 0-10 gauge maths
│   └── awareness.py              # validated bilingual content and channels
├── .streamlit/config.toml        # shared theme for local and hosted runs
└── tests/                        # unit and integration tests
```

Training code never imports Streamlit. The app never trains: it loads
`models/scam_clf.joblib` only after verifying its SHA-256, and refuses artifacts whose
scikit-learn major/minor version differs from the installed one.

---

## Dataset

**UCI SMS Spam Collection** — Almeida, T. A., Gómez Hidalgo, J. M., and Yamakami, A.
(2011). *Contributions to the study of SMS spam filtering: new collection and results.*
Distributed by the UCI Machine Learning Repository under **CC BY 4.0**.

- 5,574 raw messages (4,827 ham, 747 spam), SHA-256 verified on download.
- 430 exact duplicate messages removed before splitting; 0 conflicting labels.
- 5,144 unique messages split 4,115 / 514 / 515 (train / validation / test), stratified.
- Zero overlap between splits, asserted by an automated leakage test.

Attribution and integrity details: `data/README.md`, `docs/dataset.md`,
`THIRD_PARTY_NOTICES.md`.

---

## Testing

```cmd
python -m pytest
python -m pytest -m "not slow"
```

The suite covers data parsing and integrity, dedupe and leakage, threshold selection,
artifact checksum and version checks, prediction validation, script routing, contextual
topic selection, gauge conversion and its disclaimer, explanation direction, presentation
strings, content schema and review flags, and the Streamlit app itself through
`streamlit.testing.v1.AppTest`.

---

## Privacy

The application's own code does not intentionally store submitted messages. A message
is held in memory for the current session, used to produce the result you see, and is
not written to disk by this code. Links, emails, amounts and phone numbers are replaced
with placeholders before the text reaches the model.

What happens next depends on where the app runs:

- **Local execution** — the message is processed by the Python process on your own
  machine. Inference uses the bundled model file; there is no external AI service, no
  paid API and no analytics.
- **Streamlit Community Cloud** — the message is sent to Streamlit's servers so the app
  can render a result. This project's code still does not persist messages or call an
  external inference API, but Streamlit's infrastructure logging, retention and access
  controls are outside this codebase and have not been independently verified here.

Never submit a real OTP, PIN, card number or bank account number to any online tool.
Deployment notes are in `docs/deployment.md`.

---

## Limitations

- English SMS detection only; Kannada messages are recognised but deliberately not
  scored, and romanised Kannada is not supported at all.
- Dataset from 2011; modern scam channels (WhatsApp, RCS, email, voice, deepfakes) are
  out of distribution.
- Binary ham/spam classification; it does not identify scam type and cannot confirm
  that a message is a crime.
- The score is uncalibrated. The 0–10 gauge is a linear display rescaling of raw model
  output — not a probability, not a percentage and not a risk band.
- 1 of 63 spam messages in the test split was missed; 7 ham messages were flagged.
- Kannada content is an unreviewed translation and is labelled as such in the app.
- Screenshots are not bundled because the build machine has no browser automation;
  the interface is verified by AppTest integration tests instead.

---

## Licences

Original code: **MIT** (`LICENSE`). Dataset: **CC BY 4.0** (UCI), attributed separately
in `THIRD_PARTY_NOTICES.md`. The two licences are not interchangeable.

## Acknowledgements

Dataset authors and the UCI Machine Learning Repository; the Government of India
National Cyber Crime Reporting Portal, RBI Sachet and CERT-In for the public reporting
information referenced in `content/reporting.json`.

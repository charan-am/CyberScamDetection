# AI-Based Cyber Scam Detection and Awareness System

An explainable machine-learning application that classifies English SMS messages as
legitimate or potentially suspicious, and teaches Indian users to recognise and report
real scam patterns.

**Status:** Phase 0 complete — environment and project skeleton. The Streamlit
interface and trained model arrive in later phases.

## What this project will do

- Classify an SMS with a TF-IDF + Logistic Regression pipeline trained on the
  UCI SMS Spam Collection.
- Show the predicted class together with the model's raw output, explicitly labelled
  as uncalibrated, plus the words the model weighted.
- Separate model explanations, rule-based message indicators, and verified awareness
  guidance into three distinct layers.
- Provide Indian cyber awareness material (UPI, OTP, phishing, fake jobs, lottery,
  investment, banking impersonation) in English and Kannada, with official reporting
  information and recorded verification dates.
- Report the model's actual measured performance, including its limitations.

The classifier is **English-only**. Kannada appears only in educational content, is
marked `reviewed: false` until a Kannada speaker verifies it, and no Kannada detection
ability is claimed.

## Requirements

- Windows (CMD commands below)
- [uv](https://docs.astral.sh/uv/) — dependencies are installed with `uv pip`
- Python **3.12.16** (uv-managed). Python 3.14 is not supported and the test suite
  fails if it is used. `pip.exe` is blocked on this machine, so do not use bare `pip`.

## Setup

```cmd
cd /d "%USERPROFILE%\CyberScamDetection"
uv venv --python 3.12.16 .venv
.venv\Scripts\activate
uv pip install -r requirements.txt -r requirements-dev.txt
uv pip install -e .
python -m pytest
```

Expected: `3 passed`.

## Run the application

Available from Phase 3:

```cmd
streamlit run app\streamlit_app.py
```

## Project layout

| Path | Purpose |
|---|---|
| `src/scamdetector/` | Domain package: data, features, training, evaluation, prediction, explanation |
| `app/` | Streamlit interface (later phase) |
| `tests/` | Pytest suite |
| `data/` | Dataset and provenance |
| `content/` | Bilingual awareness content and reporting sources |
| `models/` | Trained artifact and checksum |
| `reports/` | Generated metrics and split reports |
| `docs/` | Design document, plans, model card |

## Licensing

Original code is MIT licensed (see `LICENSE`). The UCI SMS Spam Collection is used
under CC BY 4.0 and is attributed separately in `THIRD_PARTY_NOTICES.md`. The two
licenses are not interchangeable.

## Privacy

The application is designed to hold submitted messages in session state only: not
written to disk, not logged, and not sent to any third party.

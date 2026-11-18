# Verification record

Commands were run from the project root on Windows with the project virtual
environment (Python 3.12.16). Outputs below are copied from real runs.

## Automated tests

```
python -m pytest
........................................................................ [ 55%]
.........................................................                [100%]
129 passed in 49.55s
```

Coverage by area:

| Area | Tests |
|---|---|
| Environment and interpreter guard | 3 |
| Data parsing, dedupe, splitting, leakage | 15 + 4 leakage |
| Features and masking | 7 |
| Training, threshold selection, config | 7 |
| Prediction validation and artifact checksum | 13 + 5 checksum |
| Explanation contributions and signals | 9 |
| Script detection and language routing | 7 |
| Awareness-topic selection | 10 |
| Presentation strings and gauge rescaling | 17 |
| Awareness content and reporting channels | 10 |
| Metric regression and pipeline round trip | 6 + 2 |
| Streamlit interface (AppTest) | 14 |

Package metadata matches the version:

```
uv pip install -e .
 - scamdetector==0.1.0
 + scamdetector==0.1.1
```

## Application starts

```
python -m streamlit run app\streamlit_app.py --server.headless=true --server.port=8599
GET http://localhost:8599/  ->  HTTP 200
```

The process was stopped after the check.

## Interface behaviour (AppTest)

- Four tabs render: Message detection, Cyber awareness, Model performance, About & privacy.
- Prize/OTP message → "Likely spam", three labelled layers, gauge, and a caption that
  contains "linear rescaling" and no `%`.
- Gauge renders "Model Spam Signal", both end labels, and the threshold marker at
  `2.7/10` for the stored 0.27.
- Ordinary message → "No spam pattern detected" plus "not a guarantee".
- Empty and whitespace-only input → validation warning, no exception.
- Clear button resets the text box and removes the previous result.
- Kannada awareness selection → unreviewed-translation warning in Kannada.
- Kannada-script message → "Kannada message detection is not available yet", English
  named as supported, no gauge and no spam label.
- Tamil message → script not supported, message not scored.
- Performance tab shows 0.9394 / 0.8986 / 0.9841 from `reports/metrics.json`.
- Footer credit "Built by Charan AM" renders.
- About tab explains Streamlit Community Cloud, says hosting behaviour has not been
  independently verified, and no longer claims the app "runs entirely on the local
  machine".

## Gauge rescaling

Measured raw outputs on the test split: 0.000781 to 0.999516, all inside [0, 1].

| Raw output | Gauge | Decision at 0.27 |
|---|---|---|
| 0.2700 | 2.7/10 (threshold) | boundary |
| 0.93 | 9.3/10 | spam |
| 0.00 | 0.0/10 | ham |

`present.gauge_score` multiplies by ten and rounds to one decimal; values outside the
model's range are clamped. No model file was changed for the gauge.

## Artifact integrity

```
python scripts\verify_artifact.py
Artifact OK
  sha256:          e0b3ee1bf70c4e5aa6c3bf08c673d4161c8324f3c0f51e076c710c8a18987a3d
  trained_at:      2026-11-18T08:48:38+00:00
  threshold:       0.2700
  sklearn (file):  1.9.1
  sklearn (local): 1.9.1
  dataset sha256:  7d039a24a6083ed9ef0f806ebad56bbb976e3aeb8de05669173bfdc4996c239d
  dataset match:   yes
```

The app refuses to load the artifact if the checksum, required metadata, or the
scikit-learn major/minor version does not match.

## Reproducibility

- `scripts/prepare_data.py` run twice produced byte-identical split CSVs.
- Leakage check reported 0 shared messages for every split pair.
- Training is deterministic for the configured seed and pinned library versions.
- The English model, its metrics and its threshold are unchanged from v0.1.0.

## Not verified

- Kannada ML detection is not implemented. No open, expert-labelled Kannada SMS dataset
  was obtained; see `docs/kannada-assessment.md`. Nothing in the app claims otherwise.
- Kannada awareness content has not been reviewed by a native speaker; the app states this.
- Hosted privacy wording describes what the application code does. Streamlit Cloud's own
  logging and retention have not been independently verified here.
- No browser screenshots are included; no browser automation was available on the
  build machine. The interface is covered by the AppTest integration tests above.
- No evaluation on modern, romanised or non-SMS scam messages has been performed.

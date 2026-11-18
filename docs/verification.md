# Verification record

Commands were run from the project root on Windows with the project virtual
environment active (Python 3.12.16). Outputs below are copied from real runs.

## Automated tests

```
python -m pytest
........................................................................ [ 72%]
...........................                                              [100%]
99 passed in 10.43s
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
| Presentation strings | 8 |
| Awareness content and reporting channels | 10 |
| Metric regression and pipeline round trip | 6 + 2 |
| Streamlit interface (AppTest) | 10 |

## Application starts

```
python -m streamlit run app\streamlit_app.py --server.headless=true --server.port=8599
GET http://localhost:8599/  ->  HTTP 200
```

The process was stopped after the check.

## Interface behaviour (AppTest)

- Four tabs render: Message detection, Cyber awareness, Model performance, About & privacy.
- Prize/OTP message → "Likely spam", three labelled layers, no `%` in the score caption.
- Ordinary message → "Likely ham".
- Empty and whitespace-only input → validation warning, no exception.
- Clear button resets the text box and removes the previous result.
- Kannada selection → unreviewed-translation warning in Kannada.
- Non-English input → English-only limitation notice.
- Performance tab shows 0.9394 / 0.8986 / 0.9841 from `reports/metrics.json`.

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

## Not verified

- Kannada content has not been reviewed by a native speaker; the app states this.
- No browser screenshots are included; no browser automation was available on the
  build machine. The interface is covered by the AppTest integration tests above.
- No evaluation on modern or non-SMS scam messages has been performed.

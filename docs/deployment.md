# Deployment readiness — Streamlit Community Cloud

Date: 2026-11-18
Target: Streamlit Community Cloud, Python 3.12, single entry point
`app/streamlit_app.py`. No paid APIs, no external inference service, no secrets.

## Application facts the platform depends on

| Requirement | Status in this repository |
|---|---|
| Entry point | `app/streamlit_app.py` |
| Python | `3.12` (`.python-version` pins 3.12.16 locally; select 3.12 in Cloud) |
| Dependencies | `requirements.txt`, 39 pinned runtime packages including `streamlit==1.67.0` |
| Model artifact | `models/scam_clf.joblib` (194 KB) committed with `models/scam_clf.joblib.sha256` |
| Content | `content/*.json` committed |
| Metrics | `reports/metrics.json` committed; Performance tab reads it |
| Paths | Resolved from `config.PROJECT_ROOT` via `__file__`, not from the shell's directory |
| Case sensitivity | All imports, module names and content paths are lowercase and match their files |
| `src/` imports | `app/streamlit_app.py` inserts `<repo>/src` into `sys.path`, so no `pip install -e .` is needed at runtime |
| Secrets | None used; no `.env`, no API keys |
| Telemetry | `.streamlit/config.toml` sets `gatherUsageStats = false` |
| Theme | `.streamlit/config.toml` applies the dark theme on any host |

## Deploy steps (owner action, not performed here)

The repository has **not** been pushed to GitHub by this project. Once the owner pushes:

1. Sign in to Streamlit Community Cloud and choose **New app**.
2. Repository: `charan-am/CyberScamDetection`; branch: `master`.
3. Main file path: `app/streamlit_app.py`.
4. Advanced settings: Python version **3.12**.
5. Deploy. No secrets need to be added.

Resource note: the model is 194 KB, so the default Community Cloud memory is ample. The
first page load reads and checksums the artifact, which takes well under a second on a
warm process.

## Post-deploy checks

| Check | Expected |
|---|---|
| App loads | Four tabs: Message detection, Cyber awareness, Model performance, About & privacy |
| Example message | "Likely spam", gauge around 9/10, threshold marker near 2.7 |
| Ham message | "No spam pattern detected" plus the safety caveat |
| Kannada message | Warning that Kannada detection is not available; no gauge, no prediction |
| Cyber awareness | English and Kannada content; Kannada warning visible |
| Performance | F1 0.9394, precision 0.8986, recall 0.9841 from the committed metrics file |
| About & privacy | Local and hosted explanations, both displayed |
| Credit | "Built by Charan AM" at the bottom |

## Hosted privacy wording

The About tab states the following, and it should stay in sync with
`present.hosted_privacy_note()`:

> When this app runs on Streamlit Community Cloud, the message you submit is sent to
> Streamlit's servers so that the app can render a result. The application code does not
> write messages to disk and does not send message content to an external AI inference
> API. Streamlit's own infrastructure logging, retention and access controls are outside
> this project's code and have not been independently verified here. If that matters for
> your situation, run the app locally instead.

Do not replace this with "runs entirely on your machine" or a blanket "no data is ever
stored" claim: the first is false on Cloud, and the second would be a guarantee about
infrastructure this project does not control.

## Known platform limitations

- Community Cloud apps sleep after inactivity; the first request after sleep is slower.
- The hosting provider's Python patch version may differ from 3.12.16. The pinned
  scikit-learn 1.9.1 artifact refuses to load if the installed major/minor version
  differs; rebuild the artifact with `python scripts\train_model.py` if Cloud ever
  installs a different scikit-learn minor version.
- The Kannada content is still `reviewed: false`; hosting does not change that.

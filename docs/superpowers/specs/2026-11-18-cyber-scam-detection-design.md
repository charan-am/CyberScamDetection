# AI-Based Cyber Scam Detection & Awareness System — Design & Implementation Roadmap

**Status:** Approved (Revision 3) — 2026-11-18, with amendments A–F recorded in Appendix B
**Date:** 2026-11-18
**Revision:** 3 (supersedes revisions 1 and 2)
**Project type:** College PBL project, engineering-quality demonstration
**Approved constraints:** see §10 (decisions) and Appendix B (amendments)

---

## 1. Intent

Build a genuinely working, explainable and honest machine-learning application that (a) classifies an English SMS as legitimate or potentially suspicious, and (b) teaches Indian users to recognise real scam patterns and report them through official channels.

The project's differentiator is not model complexity. It is **intellectual honesty**: a real dataset, a real held-out evaluation, an explanation that is clearly labelled as a model explanation rather than proof, an uncalibrated model output never presented as certainty, and bilingual awareness content with cited official sources.

### Goals

1. A user pastes an SMS and receives a predicted class, the raw **uncalibrated** model output with an explicit label, and the message features that drove the result.
2. The app clearly separates three things: model explanation, rule-based message characteristics, and verified guidance.
3. Indian cyber awareness content (7 categories) in English and Kannada, with official reporting information and genuine verification dates.
4. A performance dashboard showing the actual measured metrics, confusion matrix, dataset facts and limitations.
5. A modular, tested, reproducible codebase publishable to GitHub with a detailed README.

### Non-goals (this release)

- Multilingual *detection*. The classifier is English-only. Kannada exists only in educational content.
- URL reputation lookups, browsing-history analysis, or any live threat-intelligence feed.
- Deep learning, embeddings, or transformer fine-tuning.
- User accounts, databases, or persistence of submitted messages.
- Claims of real-world accuracy beyond the corpus the model was trained on.
- Displaying risk percentages or risk bands before calibration has been validated (Amendment A).

### Anti-goals

- Hard-coded keyword rules masquerading as ML predictions.
- Synthetic or LLM-generated training data used to inflate metrics.
- Hand-edited metrics files, cherry-picked runs, or a test set reused during development.
- Presenting an uncalibrated score as "92% confident this is a scam".
- Imposed performance targets presented as results, or a run reported only when it flatters the model.
- Claiming a feature works before it is implemented and tested.
- Loading a `joblib` artifact from an untrusted source.

### Constraints

| Constraint | Detail |
|---|---|
| Interpreter | Python **3.12.16** (uv-managed build already present). Python 3.14 is explicitly avoided. |
| Install path | `pip.exe` is blocked by an Application Control policy. Use `uv pip` or `python -m pip`. |
| Dataset | UCI SMS Spam Collection only. Real labelled data, attributed. |
| Stack | pandas, numpy, scikit-learn, joblib, streamlit, pytest. Nothing else without justification. |
| Privacy | Submitted message text is never written to disk, logs, or analytics. |
| Deployment | Docker is a Phase 7 enhancement. The core application must run without it. |
| Demo | Laptop-only, Windows, CMD commands documented. |
| Licensing | MIT for original code; UCI dataset keeps its own CC BY 4.0 terms, documented separately. |

---

## 2. Fixed technical decisions

| Decision | Choice | Reason |
|---|---|---|
| Labels | Binary: `ham` (0) / `spam` (1) | Matches the dataset's ground truth. No invented sub-categories. |
| Deduplication | Normalise → exact-match key → drop duplicates → split by group | Amendment C: exact normalised duplicates removed **before** splitting. |
| Conflicting labels | Detect and record in `reports/conflict_report.json`; majority label wins, with the count disclosed | Otherwise model and metrics silently disagree with the data. |
| Split | 80 / 10 / 10 stratified, `random_state=42`, saved to disk | Reproducible; validation used for tuning and threshold; test touched once. |
| Features | TF-IDF, `(1,2)`-grams, `min_df=2`, `max_features=20_000`, `sublinear_tf=True` | Standard, interpretable, cheap. |
| Model | Logistic Regression, `class_weight='balanced'`, `max_iter=1000` | Linear weights give exact local explanations. |
| Tuning | Small grid over `C ∈ {0.1,1,10}` and uni/bigrams, 5-fold CV on train only | Beginner-tractable, defensible. |
| Baselines | `DummyClassifier(most_frequent)` + single-keyword rule, same test split | Comparison is required; **no target number is imposed** (Amendment B). |
| Threshold | Selected on **validation** for recall priority, frozen in config and artifact | Missing a scam is costlier than a false alarm; avoids test-set tuning. |
| Output display | **Predicted class + raw uncalibrated model output**, labelled "uncalibrated, not a probability" | Amendment A: no 0–100 risk percentage, no risk bands until calibration is validated. |
| Calibration | Optional Phase 5 experiment with reliability diagram and Brier score; risk bands only if it measurably improves | Bands without calibration are arbitrary. |
| Artifact | `joblib` dict: pipeline, threshold, metrics, config hash, dataset hash, sklearn version, `trained_at` | Drift is detectable, not silent. |
| Artifact storage | Committed to Git **only if under 10 MB**, with a committed SHA-256 checksum | Decision 2: reproducible and inspectable; the checksum makes tampering visible. |
| Artifact loading | Local repository artifact only; checksum verified before `joblib.load`; never a user-supplied or downloaded file | `joblib` uses pickle, which executes code on load. |
| Licensing | MIT `LICENSE` for code; dataset terms in `data/raw/` and `docs/dataset.md`; `THIRD_PARTY_NOTICES.md` | Decision 4: the two licenses must never be conflated. |

---

## 3. Final architecture

Four layers, one direction of dependency. The app never trains; the trainer never imports Streamlit.

```
┌──────────────────────────────────────────────────────────────┐
│ PRESENTATION   app/streamlit_app.py + app/pages/* + ui/       │
│   multipage UI, theme, components, session state              │
└───────────────────────────┬──────────────────────────────────┘
                            │ imports public functions only
┌───────────────────────────▼──────────────────────────────────┐
│ DOMAIN   src/scamdetector/                                    │
│   data → features → train → evaluate → predict                │
│   explain (model weights) · presentation (verdict wording)     │
│   awareness (bilingual content + reporting loader)            │
└───────────────────────────┬──────────────────────────────────┘
                            │ reads/writes
┌───────────────────────────▼──────────────────────────────────┐
│ ARTIFACTS  data/raw · data/processed · models/ · reports/      │
│            content/*.json                                      │
└──────────────────────────────────────────────────────────────┘
```

### Module responsibilities

| Module | Responsibility | Must not |
|---|---|---|
| `config.py` | Paths, seed, split ratios, TF-IDF and model params, threshold, version constants | Contain logic |
| `data.py` | Locate dataset, schema validation, normalisation, dedupe, conflict detection, group-aware stratified split, save/load splits | Fit any model |
| `features.py` | Text normalisation (URL/email/phone/amount masking) + TF-IDF vectoriser factory | Know about Streamlit |
| `train.py` | Build pipeline, grid search, threshold selection, fit final model, persist artifact + checksum | Print unreviewed numbers as truth |
| `evaluate.py` | Metrics, confusion matrix, baseline comparison, calibration check, write `reports/metrics.json` | Tune anything |
| `predict.py` | Verify artifact checksum, load it once, validate input, return a typed result dataclass | Retrain or download models |
| `explain.py` | Top-k n-grams by coefficient × TF-IDF value, both directions | Call them evidence |
| `present.py` | Verdict wording, uncalibrated-output labelling, disclaimer text; band logic only after validated calibration | Invent percentages or bands |
| `awareness.py` | Load/validate bilingual content JSON, filter by category and language, expose `last_verified` and `reviewed` | Translate at runtime |
| `app/*` | Render state; call domain functions | Contain ML logic or file IO |

### Prediction data flow

```
message → validate (type, length, encoding)
        → normalise → TF-IDF transform → LogisticRegression
        → raw model output (uncalibrated, labelled as such)
        → threshold → predicted class
        → explain: top +k / −k n-grams
        → signals: URL / phone / amount / caps ratio / urgency terms (rule-based, separate panel)
        → guidance: category-matched awareness card + official reporting link
        → result object → UI (session state only, never persisted)
```

### Explainability is three separate layers

| Layer | Source | UI label | Honest caveat shown |
|---|---|---|---|
| Model explanation | TF-IDF × LR coefficients | "Words the model weighted" | "Statistical association learned from the training corpus, not proof of fraud." |
| Message characteristics | Deterministic regex/lexicon | "Patterns detected in the message" | "Rule-based checks, independent of the model." |
| Verified guidance | Curated content + official sources | "What to do" / "Report" | Source name, URL, `last_verified` date, review status. |

No explainability library is required for v1: for a linear model over TF-IDF, `coefficient × feature_value` is the exact contribution, which is both correct and teachable. LIME/SHAP are listed as Phase 7 comparisons.

---

## 4. Folder structure

```
CyberScamDetection/
├─ .python-version                     # 3.12.16, guards against 3.14
├─ .gitignore
├─ LICENSE                             # MIT — original code only
├─ THIRD_PARTY_NOTICES.md              # UCI dataset CC BY 4.0 attribution
├─ README.md
├─ pyproject.toml                      # package metadata, pytest config, editable install
├─ requirements.txt                    # pinned runtime deps (resolved, not hand-written)
├─ requirements-dev.txt                # pytest only
│
├─ app/
│  ├─ streamlit_app.py                 # entry point: theme, sidebar, landing
│  ├─ pages/
│  │  ├─ 1_Check_a_Message.py
│  │  ├─ 2_Learn.py
│  │  ├─ 3_Model_Performance.py
│  │  └─ 4_About_and_Privacy.py
│  └─ ui/
│     ├─ theme.py                      # CSS, palette, typography
│     ├─ components.py                 # verdict card, output display, signal chips, metrics table
│     └─ assets/                       # logo, icons (SVG only)
│
├─ content/
│  ├─ awareness_en.json                # 7 categories, English
│  ├─ awareness_kn.json                # 7 categories, Kannada, reviewed: false
│  ├─ reporting.json                   # official channels + last_verified
│  └─ quiz.json                        # Phase 7 only
│
├─ data/
│  ├─ raw/                             # SMSSpamCollection + source note + dataset license
│  ├─ processed/                       # train/val/test CSVs (gitignored)
│  └─ README.md                        # provenance, checksum, column meanings
│
├─ docs/
│  ├─ superpowers/specs/2026-11-18-cyber-scam-detection-design.md
│  ├─ superpowers/plans/               # phase plans
│  ├─ dataset.md
│  ├─ model_card.md
│  ├─ licensing.md
│  └─ demo-script.md
│
├─ models/
│  ├─ scam_clf.joblib                  # committed only if < 10 MB
│  ├─ scam_clf.joblib.sha256           # committed checksum
│  └─ README.md                        # regeneration steps, size record, load-safety note
│
├─ reports/
│  ├─ metrics.json                     # generated, never hand-edited
│  ├─ split_summary.json
│  ├─ conflict_report.json
│  ├─ baseline_comparison.csv
│  └─ confusion_matrix.csv
│
├─ scripts/
│  ├─ get_data.py                      # fetch/verify dataset, print checksum
│  ├─ prepare_data.py                  # build splits
│  ├─ train_model.py                   # produce artifact + metrics + checksum
│  └─ verify_artifact.py               # checksum + version/hash compatibility
│
├─ src/scamdetector/
│  ├─ __init__.py                      # __version__
│  ├─ config.py
│  ├─ data.py
│  ├─ features.py
│  ├─ train.py
│  ├─ evaluate.py
│  ├─ predict.py
│  ├─ explain.py
│  ├─ present.py
│  └─ awareness.py
│
└─ tests/
   ├─ conftest.py                      # toy dataset + fitted tiny pipeline fixtures
   ├─ test_environment.py              # interpreter + package smoke tests (Phase 0)
   ├─ unit/
   │  ├─ test_features.py
   │  ├─ test_data.py
   │  ├─ test_explain.py
   │  ├─ test_present.py
   │  └─ test_awareness.py
   ├─ integration/
   │  ├─ test_leakage.py               # zero overlap across splits
   │  ├─ test_pipeline_roundtrip.py    # train → save → load → predict
   │  ├─ test_artifact_checksum.py     # tampered artifact refused
   │  └─ test_metric_regression.py     # regression guard against the recorded run
   └─ app/test_app_smoke.py            # streamlit.testing.v1.AppTest
```

---

## 5. Feature prioritization

| ID | Feature | Priority | Phase | Notes |
|---|---|---|---|---|
| F1 | Dataset acquisition, validation, dedupe, conflict report, splits | P0 | 1 | Blocks everything |
| F2 | TF-IDF + Logistic Regression training pipeline | P0 | 2 | Core |
| F3 | Baseline comparison + evaluation report | P0 | 2 | Comparison required, no target imposed |
| F4 | Model artifact + checksum + metadata | P0 | 2 | Enables separate app |
| F5 | Message analysis UI (input → predicted class) | P0 | 3 | Core UX |
| F6 | Labelled uncalibrated output, no risk percentage | P0 | 3 | Amendment A |
| F7 | Model explanation panel (top ±k n-grams) | P0 | 4 | Differentiator A, layer 1 |
| F8 | Rule-based message characteristics panel | P0 | 4 | Layer 2, kept separate |
| F9 | Seven Indian scam categories, EN + KN | P0 | 4 | Differentiator C, layer 3 |
| F10 | Official reporting information + real `last_verified` dates | P0 | 4 | Amendment F |
| F11 | Model performance dashboard | P0 | 3–4 | Differentiator E |
| F12 | Cybersecurity-themed, accessible UI | P0 | 5 | Differentiator F |
| F13 | Full test suite + privacy safeguards | P0 | 5 | Quality gate |
| F14 | README, model card, licensing docs, demo script | P0 | 6 | Deliverable |
| F15 | Interactive awareness quiz | P1 | 7 | Decision 3: Phase 7 only |
| F16 | Docker deployment | P1 | 7 | Decision 5: optional |
| F17 | Suspicious URL analysis | P1 | 7 | Labelled experimental |
| F18 | Additional model comparison (NB / LinearSVC) | P2 | 7 | Optional depth |
| F19 | Multilingual detection | P2 | beyond | Requires new labelled data; out of scope |
| F20 | Probability calibration + validated risk bands | P2 | 5 | Only with reliability diagram, Brier score, and displayed evidence |

**Rule:** Phase 7 items are attempted only after Phase 6 acceptance is fully met. Nothing in Phase 7 is described as done until tested.

---

## 6. Phases, tasks and acceptance criteria

### Phase 0 — Environment and project skeleton

**Tasks**
1. Verify `uv`, create the venv from the uv-managed 3.12.16 build, record `.python-version`.
2. Resolve and pin dependencies (`requirements.txt`, `requirements-dev.txt`); versions come from the resolver, never typed from memory.
3. Create the folder skeleton, `.gitignore`, `pyproject.toml`, `__init__.py` with `__version__`.
4. Editable install of `src/scamdetector`; confirm imports work from the repo root.
5. **At least one genuine smoke test** (Amendment D): interpreter is 3.12.x and the installed package exposes `__version__`. No placeholder or trivially-true test.
6. Configure pytest (`testpaths`, markers `unit`/`integration`/`app`/`slow`); confirm the suite exits 0.
7. `LICENSE` (MIT), `THIRD_PARTY_NOTICES.md` stub, README setup section using the exact commands that were actually run.
8. Meaningful commits (`chore:` scoped to setup).

**Acceptance criteria**
- `.venv\Scripts\python -V` reports **3.12.16**; `tests/test_environment.py` asserts `sys.version_info[:2] == (3, 12)`.
- Fresh install from the two requirements files succeeds in a clean venv without manual version guessing.
- `python -m pytest` exits **0** with at least two genuine tests collected and passed.
- `python -c "import scamdetector; print(scamdetector.__version__)"` prints `0.1.0`.
- `git status` is clean after commits; no `.venv`, `__pycache__`, or processed data tracked.
- README setup section reproduces the venv from scratch using CMD commands copied verbatim from a successful run.
- `LICENSE` is MIT; dataset attribution is referenced but not merged into the MIT text.

**Deliverables:** working venv, pinned requirements, skeleton, green smoke suite, initial commits.

---

### Phase 1 — Dataset acquisition and preparation

**Tasks**
1. `scripts/get_data.py`: fetch or verify the committed UCI SMS Spam Collection; print row count and SHA-256.
2. Record provenance, license, citation and checksum in `data/README.md`, `docs/dataset.md`, `THIRD_PARTY_NOTICES.md`.
3. `data.py`: parse tab-separated label/text, build `text`/`label`/`label_name`, validate schema and label vocabulary.
4. Normalise for dedupe: lowercase, strip, collapse whitespace, mask URLs/emails/phones/amounts → `dedupe_key`.
5. **Remove exact normalised duplicates before splitting**; report removed rows and groups (Amendment C).
6. **Detect conflicting labels** on identical `dedupe_key`: write `reports/conflict_report.json`, resolve by majority, disclose counts.
7. Group-aware stratified 80/10/10 split, `random_state=42`; write CSVs + `reports/split_summary.json`.
8. Unit tests: loader, validation failures, determinism, dedupe, conflict detection.

**Acceptance criteria**
- Loader reproduces the official row count (5,574 total; 4,827 ham, 747 spam) or documents every exclusion with a reason.
- Duplicates and conflicts are counted in reports, never silently dropped or silently relabelled.
- Zero `dedupe_key` appears in more than one split (asserted by an automated leakage test).
- Running split creation twice produces byte-identical output.
- Malformed input raises a specific, readable exception (not a bare `KeyError`).
- No row is labelled or re-labelled by hand.

**Deliverables:** raw dataset + attribution, processed splits, split summary, conflict report, data tests.

---

### Phase 2 — Model development and evaluation

**Tasks**
1. `features.py`: masking normalisation and TF-IDF factory; masking documented and tested.
2. `train.py`: pipeline, small grid search on train folds, threshold selection on validation, final fit on train+val.
3. Baselines: majority class and single-keyword rule, scored on the identical test split.
4. `evaluate.py`: precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, per-class support; write `reports/metrics.json`, `baseline_comparison.csv`.
5. Artifact: `joblib` dump with pipeline, threshold, metrics, config hash, dataset hash, sklearn version, `trained_at`; write `.sha256`.
6. `scripts/verify_artifact.py`: checksum + version/hash compatibility, failing loudly on mismatch.
7. Tests: round-trip, threshold sanity, prediction contract, checksum refusal, metric regression guard.

**Acceptance criteria**
- Same seed and data ⇒ identical metrics (verified by rerun comparison in a test).
- Baseline comparison is generated from the same test split and reported **as measured**. If the model does not outperform a baseline, that is documented and investigated rather than hidden; no target metric is imposed (Amendment B).
- Test set evaluated once, after all tuning; `metrics.json` generated by script, never edited.
- Threshold chosen from validation data only; recorded in config and artifact.
- Artifact under 10 MB is committed with its `.sha256`; a size exceeding 10 MB is recorded in `models/README.md` and the artifact is not committed.
- Artifact loads in a fresh process; a modified artifact fails checksum verification before `joblib.load` is called.
- Every metric shown in the UI traces to `reports/metrics.json`.

**Deliverables:** trained pipeline, artifact + checksum, metrics + baselines report, model tests.

---

### Phase 3 — Streamlit application core

**Tasks**
1. Multipage shell: landing, Check a Message, Learn (stub), Model Performance (real data), About & Privacy.
2. `predict.py` + `present.py`: typed `PredictionResult`, predicted class, raw uncalibrated output, disclaimer.
3. UI components: verdict card, labelled output display, signal chips, metric tiles, confusion-matrix table.
4. Model loaded once via `st.cache_resource`; checksum verified at load; missing artifact gives instructions, not a traceback.
5. Input validation: empty, whitespace-only, >5,000 characters, non-English advisory notice.
6. Session-state-only results; no message text in logs or files.

**Acceptance criteria**
- `streamlit run app/streamlit_app.py` launches after documented setup only.
- Model loads once; first prediction after load is well under one second on the demo laptop.
- **No 0–100 risk percentage and no risk band is displayed** (Amendment A). Only predicted class + raw output, labelled "uncalibrated, not a probability".
- Missing or tampered artifact produces a clear instruction, not a traceback.
- Empty/oversized input handled with readable messages.
- Repository and runtime directory contain no submitted message text after a session.

**Deliverables:** working end-to-end analysis flow, presentation module, core UI.

---

### Phase 4 — Explainability and Indian cyber awareness

**Tasks**
1. `explain.py`: top-k positive and negative contributions; handle "no strong signal".
2. UI: "Words the model weighted" panel with association-not-evidence caveat; separate "Patterns detected in the message" panel.
3. Content: 7 categories (UPI, OTP, phishing, fake jobs, lottery/prize, investment, banking impersonation) — example, warning signs, prevention, checklist — English **and** Kannada.
4. `reporting.json`: National Cyber Crime Reporting Portal, helpline 1930, RBI Sachet, CERT-In; each with URL, purpose, and a **genuine** verification date recorded on the day it was checked (Amendment F).
5. `awareness.py`: schema validation, language/category filtering, `reviewed` flag per entry.
6. `quiz.json` structure only; no UI before Phase 7.

**Acceptance criteria**
- Explanation panel shows k terms with sign and magnitude, stating they are model associations.
- A message with no meaningful tokens shows an explicit "no strong signals" state.
- All 7 categories exist in both languages with identical keys; a test enforces schema parity.
- Every reported channel cites an official source and a real `last_verified` date; no uncited helplines; no invented dates.
- Kannada content carries `reviewed: false` until a named reviewer verifies it, and the UI displays that status.
- The three layers never share a visual container or label.
- No Kannada detection capability is claimed in UI, README or model card.

**Deliverables:** explanation layer, rule-based signal panel, bilingual awareness content, reporting module, content tests.

---

### Phase 5 — Testing, optimisation, UI refinement

**Tasks**
1. Complete unit + integration coverage; `AppTest` smoke tests per page.
2. Leakage test, checksum test, metric regression test, artifact compatibility test.
3. Accessibility pass: labels, keyboard reachability, contrast, no colour-only meaning, plain language.
4. Performance pass: cache model/content; no recomputation across reruns.
5. Optional calibration experiment (reliability diagram, Brier score). Adopt only if measurably better, and only then introduce validated risk bands.
6. Optional model comparison (Multinomial NB, LinearSVC) in `baseline_comparison.csv`.

**Acceptance criteria**
- `python -m pytest` all green; coverage of `src/scamdetector` reported as measured (no invented percentage).
- AppTest passes for every page, including the full analyse flow.
- No raw traceback reaches the user; every failure path has a human-readable message.
- Text contrast meets WCAG AA; verdicts are never conveyed by colour alone.
- Regression test fails only if test-set F1 drops below the value recorded in the accepted run.
- Calibration, if adopted, is evidenced in the dashboard; if rejected, the uncalibrated label remains.

**Deliverables:** green suite, accessibility/performance notes, optional calibration or comparison results.

---

### Phase 6 — Documentation, publication, demonstration

**Tasks**
1. `README.md`: overview, screenshots, architecture, setup with CMD commands, train/run/test commands, measured metrics, limitations, privacy, attribution.
2. `docs/model_card.md`: intended use, training data, metrics, biases, out-of-scope uses.
3. `docs/licensing.md` + `THIRD_PARTY_NOTICES.md`: MIT for code, UCI CC BY 4.0 for data, kept distinct.
4. `docs/demo-script.md`: 5-minute laptop walkthrough.
5. Commit hygiene, tag `v1.0.0`, publish to the owner's public GitHub account.
6. Verify a fresh clone works end to end.

**Acceptance criteria**
- Fresh clone + documented commands trains and launches without Docker.
- Every number in the README matches `reports/metrics.json`.
- Model card states corpus age, English-only SMS scope, and uncalibrated output.
- MIT license covers code only; dataset terms and citation are separately and correctly stated.
- Demo runs offline; no secrets, absolute local paths, or personal data in the repo or history.

**Deliverables:** published repository, README, model card, licensing docs, demo script, release tag.

---

### Phase 7 — Optional enhancements

Ordered by value/risk: quiz → Docker → URL analysis → model comparison. Each is gated by: implemented, tested, documented, and only then described as available.

**Acceptance criteria (per item)**
- **Quiz:** question bank validated, answers checked, score kept in session, AppTest covers it.
- **Docker:** multi-stage slim image, non-root user, artifact built at image build time, documented `docker run`, Kannada renders.
- **URL analysis:** labelled experimental, deterministic structural checks only, never a silent verdict input.
- **Comparison:** identical splits and fair tuning; results appended to the existing table.

**Deliverables:** only completed features, each tested and documented.

---

## 7. Testing strategy

| Level | Scope | Examples |
|---|---|---|
| Environment | Toolchain contract | interpreter is 3.12.x, package imports, `__version__` |
| Unit | Pure functions | masking idempotence, schema rejection, presentation wording, content parity |
| Integration | Module boundaries | split→train→save→load→predict, zero cross-split overlap, tampered artifact refused |
| Regression | Accepted-run guard | test-set F1 must not drop below the recorded value |
| UI | Real app, headless | `AppTest` per page and full analyse flow |
| Manual | Human judgement | Kannada rendering, readability, false-positive UX, demo rehearsal |

Determinism: one seed in `config.py`; all randomness routed through it. No test depends on the network.

---

## 8. Privacy and ethics

- Message text lives in Streamlit session state only and is discarded on reload.
- No logging of raw text; exception handlers log types and lengths only.
- No third-party analytics, no telemetry, no external calls at inference.
- UI never says "this message is safe"; only "no scam pattern detected by this model".
- Kannada content is educational translation, labelled unreviewed until verified. Detection stays English-only.
- Reporting information is sourced from official portals with real verification dates; guidance, not legal advice.
- `joblib` artifacts are loaded only from the project's own `models/` directory after checksum verification.

---

## 9. Technical risks and mitigations

| # | Risk | Impact | Mitigation |
|---|---|---|---|
| 1 | Python 3.14 is the PATH default | Environment breaks mid-project | `.python-version`, venv from the 3.12.16 uv build, `test_environment.py` fails on 3.14, version assertion at train time |
| 2 | `pip.exe` blocked by policy | Setup stalls | `uv pip` / `python -m pip`; documented and used in every command |
| 3 | Corpus is 2005–2011 SMS; modern UPI/OTP scams underrepresented | Overstated real-world accuracy | Stated as the primary limitation in dashboard, model card, README; no claims beyond the corpus |
| 4 | Class imbalance (747 vs 4,827) | High accuracy hides poor recall | `class_weight='balanced'`, PR-AUC, recall-first threshold on validation, both classes reported |
| 5 | Near-duplicate leakage | Inflated metrics | Exact normalised dedupe before splitting + automated leakage test + conflict report |
| 6 | Conflicting labels on identical text | Contradictory training signal | Majority resolution, counts disclosed in `conflict_report.json` |
| 7 | Uncalibrated output read as certainty | User harm | Raw output labelled "uncalibrated, not a probability"; no percentages or bands before validated calibration |
| 8 | Explanation mistaken for evidence | False assurance | Three-layer separation with fixed UI labels |
| 9 | Kannada content errors | Credibility and safety | `reviewed: false`, named faculty/native review, source citations, status visible in UI |
| 10 | Cherry-picked or edited metrics | Academic integrity failure | Script-generated `metrics.json`, single test-set evaluation, baseline comparison, commit history |
| 11 | Artifact/code drift | Wrong predictions in demo | Hash + version metadata, `.sha256` verification, `verify_artifact.py` fails loudly |
| 12 | Malicious `joblib` payload | Code execution | Project-local artifacts only, checksum verified before load, pickle warning documented |
| 13 | Streamlit rerun resets or recomputes | Slow or confusing UX | `st.cache_resource` for model/content, `st.session_state` for results |
| 14 | Scope creep into Phase 7 | Unfinished core | Priority table, Phase 6 gate, "only claim what is tested" |
| 15 | Kannada font rendering on some systems | Broken UI | Browser-side font fallback verified on Windows; documented |
| 16 | Docker size/time (Phase 7) | Poor demo experience | Multi-stage slim build; core app demonstrably works without Docker |

---

## 10. Approved decisions (2026-11-18)

| # | Decision | Resolution |
|---|---|---|
| 1 | Kannada review | All Kannada content stays `reviewed: false` until verified by a Kannada speaker or faculty reviewer. No premature verification claims. |
| 2 | Model artifact | Commit the artifact if under 10 MB, with metadata and a committed SHA-256. Document regeneration. Never load untrusted `joblib` files. |
| 3 | Awareness quiz | Phase 7 only. |
| 4 | Publication | Public GitHub repo under the owner's account. MIT for original code; UCI CC BY 4.0 attribution kept separate. |
| 5 | Demonstration | Laptop-only Windows demo. Docker optional in Phase 7. Core app must work without Docker. |

---

## Appendix A — Command reference (documented, not executed)

```cmd
cd /d "C:\Users\Charan A M\CyberScamDetection"
uv venv --python 3.12.16 .venv
.venv\Scripts\activate
python -V
uv pip install -r requirements.txt -r requirements-dev.txt
uv pip install -e .
python -m pytest
python scripts\train_model.py
streamlit run app\streamlit_app.py
```

Fallback if uv cannot resolve the version by number:

```cmd
uv venv --python "%APPDATA%\uv\python\cpython-3.12.16-windows-x86_64-none\python.exe" .venv
```

If `uv pip` is unavailable, use `python -m pip install ...` (never bare `pip`). `uv venv` does not install `pip` by default; create the venv with `--seed` if the module entry point is required.

---

## Appendix B — Amendment log

Amendments A–F were supplied by the owner with the Revision 2 approval and are binding on all later phases.

| ID | Amendment | Where it lands |
|---|---|---|
| A | No misleading 0–100 risk percentages or arbitrary bands; show predicted class + labelled uncalibrated output; bands only after validated calibration | §1 non-goals, §2 output display, F6, Phase 3 AC, risk 7 |
| B | Compare against baselines; no fabricated results; no arbitrary performance targets | §2 baselines, F3, Phase 2 AC, risk 10 |
| C | Deduplicate exact normalised messages before stratified splitting; automated leakage test; document conflicting labels | §2 dedupe/conflict rows, Phase 1 tasks 5–6, risk 5–6 |
| D | Phase 0 must contain at least one genuine smoke test; pytest must exit successfully | Phase 0 task 5, Phase 0 AC, §7 environment level |
| E | Model explanations, rule-based indicators and verified guidance remain three separate layers | §3 layer table, F7–F9, Phase 4 AC |
| F | Official sources for Indian cybercrime reporting, with genuine verification dates | Phase 4 task 4 and AC, risk 9 |

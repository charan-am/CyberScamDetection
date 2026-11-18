# Model card — SMS scam classifier v0.1.0

## Overview

| Item | Value |
|---|---|
| Task | Binary text classification: `ham` (ordinary SMS) vs `spam` (unwanted or fraudulent SMS) |
| Model | scikit-learn `Pipeline`: TF-IDF vectoriser + `LogisticRegression` |
| Version | 0.1.0 |
| Trained at | 2026-11-18T08:48:38+00:00 |
| Trained by | Charan A M, for a PBL submission |
| Artifact | `models/scam_clf.joblib` (194,174 bytes) |
| Artifact SHA-256 | `e0b3ee1bf70c4e5aa6c3bf08c673d4161c8324f3c0f51e076c710c8a18987a3d` |
| Environment | Python 3.12.16, scikit-learn 1.9.1 |
| Language | English SMS only |

## Intended use

- Demonstrating an explainable classical NLP pipeline for SMS spam detection.
- Educational awareness: showing a user *why* a message looks suspicious, next to
  independent safety guidance and official reporting channels.
- Coursework evaluation and local demonstration on a Windows laptop.

## Out-of-scope use

- Any decision with legal, financial or safety consequences.
- Screening WhatsApp, RCS, email, voice calls, screenshots or images.
- Kannada, Hindi or other Indian-language messages: the model has never seen them.
- Detecting every kind of cyber scam, or confirming that a message is a crime.
- Production deployment, multi-tenant use, or autonomous blocking of messages.

## Training data

UCI SMS Spam Collection (Almeida, Gómez Hidalgo and Yamakami, 2011), CC BY 4.0.

- Raw: 5,574 messages (4,827 ham, 747 spam), SHA-256
  `7d039a24a6083ed9ef0f806ebad56bbb976e3aeb8de05669173bfdc4996c239d`.
- After removing 430 exact duplicate messages (whitespace/case-normalised): 5,144.
- Conflicting-label groups: 0.
- Stratified split with seed 42: train 4,115, validation 514, test 515.
- Verified zero overlap between all split pairs.

Messages are masked before vectorisation: URLs, emails, amounts and phone numbers are
replaced with `<url>`, `<email>`, `<amount>`, `<phone>`. This reduces the chance of the
model memorising particular phone numbers or links.

## Model configuration

| Setting | Value |
|---|---|
| Vectoriser | TF-IDF, `ngram_range=(1,2)`, `min_df=2`, `max_features=20,000`, `sublinear_tf=True` |
| Classifier | LogisticRegression, `liblinear`, `class_weight="balanced"`, `max_iter=1000` |
| Regularisation | `C=10.0`, chosen by 5-fold cross-validated spam F1 on train |
| Decision threshold | 0.27, chosen on validation only to maximise F2 |
| Final refit | train + validation, then evaluated once on test |

The threshold favours recall (missing a scam is treated as worse than a false alarm).
The raw score is uncalibrated and is never presented as a probability.

## Evaluation

Test split, 515 messages (452 ham, 63 spam), evaluated once after all choices were fixed.

| Metric | Value |
|---|---|
| Accuracy | 0.9845 |
| Precision (spam) | 0.8986 |
| Recall (spam) | 0.9841 |
| F1 (spam) | 0.9394 |
| ROC-AUC | 0.9981 |
| PR-AUC | 0.9889 |

Confusion matrix: TN 445, FP 7, FN 1, TP 62.

Validation F1 at the chosen threshold: 0.9254.

### Baselines (same test split)

| Baseline | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|
| Majority class (always ham) | 0.0000 | 0.0000 | 0.0000 | 0.1223 |
| Single-keyword rule | 0.5843 | 0.8254 | 0.6842 | 0.5036 |

The keyword rule flags a message when it contains any of: free, win, claim, prize,
urgent, txt, call. It is a deliberately simple reference point, not a competing product.

## Explainability

For a linear model with TF-IDF features, the contribution of a term to the spam score is
exactly `coefficient × TF-IDF value`. `explain.top_contributions` reports the eight
largest positive and eight largest negative contributions for the message. This is an
exact decomposition of the model's input–output relationship for the terms present in the
message; it is *not* a claim about the sender's intent.

Rule-based indicators (links, OTP requests, prize words, urgency, amounts, job and
investment terms, shouting, exclamation marks) are computed separately and shown in a
different layer from the model explanation.

## Ethical considerations

- **False negatives matter.** A message labelled "ham" may still be a scam. The interface
  never says a message is safe; it reports a prediction with limits and always offers
  awareness guidance and reporting channels.
- **False positives matter.** A legitimate message flagged as spam could unfairly
  discredit a sender. The interface labels the result "likely" and explains it.
- **No message retention.** Submitted text is processed in memory only.
- **Sensitive-token masking.** Links, numbers, emails and amounts are masked before
  classification.
- **Unreviewed Kannada content.** Kannada guidance is machine-drafted and marked
  `reviewed: false`; the app displays that warning before the content.
- **Dataset bias.** The corpus is English, SMS-based, and from 2011. Its ham messages may
  not represent current legitimate messaging, and its spam messages do not represent
  modern scam tactics.

## Caveats

- Small test set: each spam error moves recall by about 1.6 percentage points.
- Single dataset, single split, single seed; no external validation set.
- No probability calibration; the raw score ordering is meaningful, absolute values are not.
- No drift monitoring; the artifact is a fixed snapshot.
- Modern scam categories in the awareness content are not represented in the training
  labels; awareness content and model capability are intentionally separate.

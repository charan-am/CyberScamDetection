# Dataset Documentation

## What the data is

The UCI SMS Spam Collection is a public corpus of 5,574 English SMS messages, each
labelled `ham` (legitimate) or `spam`. It was assembled in 2011 from Grumbletext,
Caroline Tagg's PhD corpus, the NUS SMS Corpus and the SMS Spam Corpus v0.1 Big.

- 4,827 ham (86.6 %)
- 747 spam (13.4 %)

Messages are British and Singaporean in origin, dominated by 2003–2011 SMS
vocabulary: ringtone downloads, "claim your prize", premium-rate reply numbers,
competition lines. The corpus contains **no** UPI, OTP, KYC, fake-job or investment
scam messages in modern Indian form. That gap is the single most important fact about
this dataset for the project (see `docs/model_card.md`).

## Why this dataset

- Genuine, human-labelled, publicly downloadable and citable.
- Small enough to train on a laptop in seconds, which makes the pipeline inspectable.
- Binary labels match the approved scope: no invented scam sub-categories.
- Widely used as a baseline in SMS spam literature, so the measured performance can be
  placed in context honestly.

## Licence and attribution

The UCI repository distributes the collection under **CC BY 4.0**. The archive's own
`readme` states that Tiago A. Almeida and José María Gómez Hidalgo hold copyright,
asks users to cite the DocEng 2011 paper and to notify the authors of use, and
provides the corpus "as is" without warranty.

The project code is MIT licensed; the dataset is **not** covered by that licence.
See `THIRD_PARTY_NOTICES.md` and `docs/licensing.md`.

## Integrity

| Check | Value |
|---|---|
| SHA-256 | `7d039a24a6083ed9ef0f806ebad56bbb976e3aeb8de05669173bfdc4996c239d` |
| Rows | 5,574 |
| Unique after normalised dedupe | 5,144 |
| Duplicate rows removed | 430 |
| Conflicting-label groups found | 0 |

## Known data limitations

1. **Age.** The newest messages are from around 2011. Vocabulary, sender behaviour and
   scam tactics have changed.
2. **Geography.** Mostly UK and Singapore; Indian numbering, UPI handles and local
   brands are absent.
3. **Class imbalance.** 13.4 % spam. Accuracy alone is misleading, so the project
   reports precision, recall, F1, PR-AUC and the confusion matrix, and uses
   `class_weight="balanced"`.
4. **Near duplicates.** 430 exact normalised duplicates existed and were removed
   before splitting. Paraphrased duplicates may remain; the leakage test covers exact
   normalised keys only.
5. **Label quality.** Labels are taken as published. Zero conflicts were detected in
   this release, but the mechanism for detecting them is tested separately.
6. **English only.** There is no Kannada (or any other language) training data, so no
   multilingual detection claim is made anywhere in the project.

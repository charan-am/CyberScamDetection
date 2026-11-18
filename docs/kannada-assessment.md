# Kannada detection — dataset assessment and decision

Date: 2026-11-18
Status: **Kannada machine-learning detection is not implemented.** Kannada awareness
content is available and remains marked unreviewed.

## The question

Can this project add a genuine Kannada spam/scam classifier — trained on real, labelled
Kannada messages, evaluated on a held-out split, with an honest comparison to a baseline
— without inventing data or borrowing English results?

## What was searched

- Hugging Face dataset search for Kannada text classification and spam datasets
  (`search=kannada`, and `search=spam` filtered to `language:kn`).
- GitHub repository search for `kannada sms spam`, `kannada spam dataset`, `SMSDHL` and
  `RevisedIndianDataset`.
- General web search for labelled Kannada SMS spam corpora.
- The published descriptions of the two most relevant academic datasets (below).

## What was found

### 1. Spam SMS in Dravidian Languages — IEEE DataPort

- Authors: Ramanujam Elangovan (NIT Silchar) and Abirami A M (Thiagarajar College of
  Engineering).
- DOI: `10.21227/dcym-pd69`; file: `Spam Message Collection.xlsx` (1.26 MB).
- Contents: spam and ham messages in English, Tamil, Telugu, **Kannada** and Malayalam;
  nearly 7,700 messages collected through a Google form and labelled by language
  experts. The authors state the data contains no personal or banking information.
- Licence: **CC BY 4.0** — reuse with attribution is permitted.
- Blocker: the file can only be downloaded after logging in to an IEEE account
  ("LOGIN TO ACCESS DATASET FILES"). The public page does not expose the data directly,
  and this project did not create an account or accept terms on the owner's behalf.
- What is still unknown: the Kannada row count, ham/spam balance, duplicate rate,
  label conflicts, and whether the Kannada rows are Kannada script or romanised.

### 2. SMSDHL / "RevisedIndianDataset" — TechRxiv preprint

- Described as 4,565 labelled SMS messages across English, Hindi, Telugu and Kannada
  (3,354 ham, 1,211 spam) for multilingual spam detection.
- Blocker: no public download location was found. GitHub searches for `SMSDHL` and
  `RevisedIndianDataset` returned zero repositories, and the preprint describes the
  corpus without exposing it.
- Without the actual rows, no claim about Kannada performance can be made.

### 3. Everything else

- Hugging Face returned Kannada news, ASR, Wikipedia, translation and instruction
  datasets — no labelled Kannada SMS spam dataset.
- GitHub returned one multilingual SMS API project, which is an application rather than
  a dataset, and gives no licence or provenance for its training data.
- No dataset was found that is both openly downloadable and documented well enough to
  train and evaluate without further verification.

## Decision

Implement the **honest fallback** specified in the enhancement brief:

- The interface detects Kannada script and **does not score** those messages.
- It states that Kannada detection is not available yet, and why.
- Kannada awareness guidance stays available, still marked `reviewed: false`.
- No machine-translated English corpus is used as a substitute, because translated
  ham/spam labels would measure translation artefacts, not Kannada scam language.
- No Kannada metrics, counts or accuracies are claimed anywhere.

## What would enable Kannada detection

1. Obtain a licensed Kannada SMS corpus (the IEEE DataPort dataset is the strongest
   candidate) with the owner's own IEEE account, or obtain written permission from the
   SMSDHL authors.
2. Isolate Kannada-script rows; record the exact counts and class balance.
3. Have a native Kannada speaker review a sample of labels and the preprocessing rules
   (punctuation, matras, transliteration, mixed-script messages).
4. Normalise and de-duplicate exactly as for English; report conflicts instead of
   silently resolving them.
5. Compare word-level and character-level TF-IDF with at least a majority-class
   baseline; choose the threshold on validation data only; evaluate the test split once.
6. Report precision, recall, F1, PR-AUC and the confusion matrix from that run.
7. Only then route Kannada messages to the new model, and remove the current
   "not available" notice.

Scope note: only Kannada **script** is in scope. Romanised Kannada ("nimma account
block aagide") is Latin script and is currently treated as English; it would need its
own labelled data and evaluation before any claim is made.

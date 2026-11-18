# Five-minute demonstration script

Pre-requisites: setup completed from the README, terminal in the project root, venv active.

## 1. Start the app (30 s)

```cmd
python -m streamlit run app\streamlit_app.py
```

Say: *"The app loads a pre-trained model from disk. It verifies the file's SHA-256
checksum before loading it, so a corrupted or swapped model is refused. Training is a
separate script and never happens inside the interface."*

## 2. Detect a scam SMS (90 s)

Paste this into **Message detection**:

> Congratulations! Your mobile number won Rs 25,00,000 in the KBC lucky draw. To claim
> the prize, pay Rs 6,500 processing fee and send your bank details and OTP on WhatsApp
> within 24 hours.

Click **Analyse message**. Point out, in order:

1. The prediction says *likely spam*, not "definitely a scam".
2. The score is labelled **uncalibrated model output, not a probability** — no invented
   percentage or risk band.
3. The explanation table shows the exact words that pushed the score up (for example
   `won`, `prize`, `otp`) and down.
4. The rule-based layer independently lists the link/OTP/prize patterns; it is not part
   of the model.
5. The guidance layer recommends what to do, and the page footer repeats the 1930
   helpline and the national portal.

## 3. Detect a normal message (30 s)

Replace the text with:

> I will be home late tonight, please start dinner without me.

Analyse again: *likely ham*, with ordinary words in the explanation and no rule
indicators.

## 4. Show the awareness area (60 s)

Open **Cyber awareness**. Switch language to **ಕನ್ನಡ** and point out the warning banner:
the translation is marked unreviewed until a Kannada speaker verifies it. Switch back to
English, choose *UPI payment scams*, and show the example, warning signs, prevention
steps and the official reporting block with its verification date.

## 5. Show the performance area (60 s)

Open **Model performance**:

- F1 0.9394, precision 0.8986, recall 0.9841 on 515 held-out messages.
- Confusion matrix: 1 missed spam, 7 false alarms.
- Baselines: majority class F1 0, keyword rule F1 0.6842.
- Read the limitations list aloud: English SMS only, 2011 data, uncalibrated score.

## 6. Show the evidence (60 s)

In the terminal:

```cmd
python -m pytest
```

Expect `99 passed`. Then:

```cmd
git log --oneline
python scripts\verify_artifact.py
```

Say: *"Every committed phase has its own commit, the suite covers data leakage, the
artifact checksum, predictions, explanations, content validation and the interface
itself, and the artifact is traceable to the dataset hash."*

## Likely questions

**Why logistic regression?** It is fast, works well on sparse TF-IDF text, and its
coefficients give an exact explanation, so no approximation library is needed.

**Why is the threshold 0.27?** It was chosen on the validation split to maximise F2,
which weights recall twice as heavily as precision: missing a scam is worse than a false
alarm. The test split was untouched during that choice.

**Can it detect Kannada scams?** No. The model is English-only and the report says so.
Kannada exists only as awareness content.

**Is the score a probability?** No. It is uncalibrated model output and the interface
labels it that way. Calibration is listed as future work.

**What happens to the message?** It stays in memory, is masked for links/numbers, and is
never stored or sent anywhere.

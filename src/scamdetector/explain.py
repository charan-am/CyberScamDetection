"""Two explainability layers that are deliberately kept apart.

1. :func:`top_contributions` explains the *model*: for a linear classifier over
   TF-IDF, ``coefficient x feature value`` is that feature's exact additive
   contribution to the decision score.
2. :func:`detect_signals` reports *rule-based message characteristics*. These are
   deterministic regex/lexicon checks and say nothing about what the model used.

Neither layer is evidence that a message is fraudulent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from . import config
from .features import normalise_text

_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_SHORT_URL_RE = re.compile(r"(?:bit\.ly|tinyurl\.com|t\.co|goo\.gl|cutt\.ly|rb\.gy|is\.gd)", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?91[\s-]?)?(?:\d[\s-]?){9,12}\d(?!\d)")
_AMOUNT_RE = re.compile(r"(?:rs\.?|inr|₹|\$|£|€)\s?\d[\d,]*", re.IGNORECASE)


@dataclass(frozen=True)
class Contribution:
    """One TF-IDF feature and its signed contribution to the spam score."""

    term: str
    weight: float


@dataclass(frozen=True)
class Signal:
    """One deterministic pattern found in the message text."""

    code: str
    label: str
    detail: str


def top_contributions(pipeline, text: str, k: int = config.TOP_K_TERMS):
    """Return the ``k`` most positive and ``k`` most negative contributions.

    Returns two empty lists when the message contains no known vocabulary, which
    the UI renders as an explicit "no strong signals" state.
    """
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["clf"]
    matrix = vectorizer.transform([normalise_text(text)])
    coefficients = classifier.coef_[0]
    names = vectorizer.get_feature_names_out()

    row = matrix.tocoo()
    entries = [
        Contribution(term=str(names[column]), weight=float(value * coefficients[column]))
        for column, value in zip(row.col, row.data)
        if abs(value * coefficients[column]) > 1e-12
    ]
    if not entries:
        return [], []

    positive = sorted((item for item in entries if item.weight > 0), key=lambda item: -item.weight)
    negative = sorted((item for item in entries if item.weight < 0), key=lambda item: item.weight)
    return positive[:k], negative[:k]


def _matching_terms(text: str, terms: tuple[str, ...]) -> list[str]:
    lowered = text.lower()
    return [term for term in terms if term in lowered]


def _caps_ratio(text: str) -> float:
    letters = [character for character in text if character.isalpha()]
    if len(letters) < 20:
        return 0.0
    return sum(1 for character in letters if character.isupper()) / len(letters)


def detect_signals(text: str) -> list[Signal]:
    """Return deterministic, independent-of-the-model message characteristics."""
    signals: list[Signal] = []

    urls = _URL_RE.findall(text)
    if urls:
        signals.append(
            Signal("url", "Contains a web link", f"{len(urls)} link(s) found; check the domain before opening.")
        )
    short_links = _SHORT_URL_RE.findall(text)
    if short_links:
        signals.append(
            Signal("short_url", "Uses a shortened link", "Short links hide the real destination.")
        )
    if _EMAIL_RE.search(text):
        signals.append(Signal("email", "Contains an e-mail address", "Unexpected sender addresses are a warning sign."))
    if _PHONE_RE.search(text):
        signals.append(Signal("phone", "Contains a phone number", "Scam messages often push you to call a number."))
    if _AMOUNT_RE.search(text):
        signals.append(Signal("amount", "Mentions a money amount", "Money requests or offers deserve extra checks."))

    urgency = _matching_terms(text, config.URGENCY_TERMS)
    if urgency:
        signals.append(
            Signal("urgency", "Uses pressure or urgency", f"Terms: {', '.join(urgency[:4])}.")
        )
    prizes = _matching_terms(text, config.PRIZE_TERMS)
    if prizes:
        signals.append(
            Signal("prize", "Mentions a prize or lottery", f"Terms: {', '.join(prizes[:4])}.")
        )
    otp = _matching_terms(text, config.OTP_TERMS)
    if otp:
        signals.append(
            Signal("otp", "Asks about a code, PIN or password", "Never share an OTP, PIN, CVV or password.")
        )
    payments = _matching_terms(text, config.PAYMENT_TERMS)
    if payments:
        signals.append(
            Signal("payment", "References a payment method or bank process", f"Terms: {', '.join(payments[:4])}.")
        )
    jobs = _matching_terms(text, config.JOB_TERMS)
    if jobs:
        signals.append(
            Signal("job", "Looks like a job or earning offer", f"Terms: {', '.join(jobs[:4])}.")
        )
    investments = _matching_terms(text, config.INVESTMENT_TERMS)
    if investments:
        signals.append(
            Signal("investment", "Looks like an investment pitch", f"Terms: {', '.join(investments[:4])}.")
        )
    ratio = _caps_ratio(text)
    if ratio > 0.5:
        signals.append(
            Signal("caps", "Mostly written in capitals", f"{ratio:.0%} of letters are upper-case; unusual for genuine conversations.")
        )
    if text.count("!") >= 3:
        signals.append(Signal("exclamation", "Repeated exclamation marks", "Excessive punctuation is common in bulk promotional messages."))

    return signals

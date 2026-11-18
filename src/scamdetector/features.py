"""Text masking, normalisation and TF-IDF construction.

Masking replaces volatile identifiers (links, e-mail addresses, money amounts and
phone numbers) with stable tokens. The same function is used for training, for
inference and for building the deduplication key, so the model never sees a
memorised phone number and duplicate detection is not fooled by a changed link.
"""

from __future__ import annotations

import re

from sklearn.feature_extraction.text import TfidfVectorizer

from . import config

URL_TOKEN = "<url>"
EMAIL_TOKEN = "<email>"
AMOUNT_TOKEN = "<amount>"
PHONE_TOKEN = "<phone>"

_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_AMOUNT_RE = re.compile(
    r"(?:(?:rs\.?|inr|₹|\$|£|€)\s?\d[\d,]*(?:\.\d+)?)"
    r"|(?:\b\d[\d,]*(?:\.\d+)?\s?(?:rs|inr|rupees|/-))",
    re.IGNORECASE,
)
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?91[\s-]?)?(?:\d[\s-]?){9,12}\d(?!\d)")
_WHITESPACE_RE = re.compile(r"\s+")

_MASK_ORDER = (
    (URL_TOKEN, _URL_RE),
    (EMAIL_TOKEN, _EMAIL_RE),
    (AMOUNT_TOKEN, _AMOUNT_RE),
    (PHONE_TOKEN, _PHONE_RE),
)


class TextInputError(ValueError):
    """Raised when a value cannot be treated as message text."""


def ensure_text(value: object) -> str:
    """Return ``value`` as text, refusing non-text input with a clear message."""
    if isinstance(value, str):
        return value
    raise TextInputError(f"Message must be text, received {type(value).__name__}.")


def mask_sensitive(text: str) -> str:
    """Replace links, e-mails, money amounts and phone numbers with tokens."""
    masked = ensure_text(text)
    for token, pattern in _MASK_ORDER:
        masked = pattern.sub(f" {token} ", masked)
    return masked


def normalise_text(text: str) -> str:
    """Mask, collapse whitespace and strip. Idempotent."""
    return _WHITESPACE_RE.sub(" ", mask_sensitive(text)).strip()


def build_vectorizer(
    ngram_range: tuple[int, int] = (1, 2),
) -> TfidfVectorizer:
    """Create the TF-IDF vectoriser with the project's fixed settings."""
    return TfidfVectorizer(
        lowercase=True,
        preprocessor=normalise_text,
        strip_accents="unicode",
        ngram_range=tuple(ngram_range),
        min_df=config.TFIDF_MIN_DF,
        max_features=config.TFIDF_MAX_FEATURES,
        sublinear_tf=config.TFIDF_SUBLINEAR_TF,
    )


def transform_texts(vectorizer: TfidfVectorizer, texts):
    """Transform texts with an already-fitted vectoriser.

    The vectoriser's own preprocessor applies the same masking used in training,
    so callers pass raw message text.
    """
    return vectorizer.transform([ensure_text(t) for t in texts])

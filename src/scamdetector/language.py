"""Script-based language routing for the detection interface.

The classifier was trained on English SMS only. This module decides which
script a message is written in so that the interface never sends Kannada or
another Indic script through the English model and calls the result reliable.

Only Kannada *script* is recognised. Romanised Kannada (for example
"nimma account block aagide") is Latin script and is therefore reported as
English; that limitation is documented for users.
"""

from __future__ import annotations

__all__ = ["detect_language", "kannada_ratio", "SUPPORTED_LANGUAGES"]

KANNADA_START = 0x0C80
KANNADA_END = 0x0CFF

#: Languages the interface recognises for routing purposes. Only English has a
#: trained classifier; Kannada content exists for awareness guidance.
SUPPORTED_LANGUAGES = ("en", "kn")


def _is_kannada(character: str) -> bool:
    return KANNADA_START <= ord(character) <= KANNADA_END


def _is_latin(character: str) -> bool:
    return character.isascii() and character.isalpha()


def kannada_ratio(text: str) -> float:
    """Return the share of letters in *text* that are Kannada script.

    Digits, punctuation, emoji and spaces are ignored. An empty string, or one
    with no letters at all, returns 0.0.
    """
    letters = [character for character in text if character.isalpha()]
    if not letters:
        return 0.0
    kannada = sum(1 for character in letters if _is_kannada(character))
    return kannada / len(letters)


def detect_language(text: str) -> str:
    """Classify the script of *text* as ``en``, ``kn``, ``other`` or ``unknown``.

    Kannada wins from a 30% share of letters so that messages mixing Kannada
    with English brand names, "OTP" or "KYC" are still routed as Kannada.
    """
    letters = [character for character in text if character.isalpha()]
    if not letters:
        return "unknown"

    total = len(letters)
    kannada = sum(1 for character in letters if _is_kannada(character))
    latin = sum(1 for character in letters if _is_latin(character))
    other = total - kannada - latin

    if kannada and kannada / total >= 0.30:
        return "kn"
    if latin / total >= 0.50:
        return "en"
    if other:
        return "other"
    return "en"

"""Map message characteristics to related awareness categories.

This is part of presentation only. It answers "which awareness topics are
worth reading next?" and must never be shown as a predicted scam category or
as a verified judgement about the message.
"""

from __future__ import annotations

from typing import Iterable

__all__ = ["related_category_ids", "SIGNAL_TOPICS"]

#: Rule-based signal codes from :func:`scamdetector.explain.detect_signals`.
SIGNAL_TOPICS = {
    "prize": "lottery",
    "otp": "otp",
    "job": "fake_jobs",
    "investment": "investment",
    "url": "phishing",
    "short_url": "phishing",
    "email": "phishing",
    "payment": "upi",
    "amount": "upi",
}

#: Plain-language cues for topics that have no dedicated signal code.
UPI_CUES = ("upi", "qr code", "collect request", "gpay", "google pay", "phonepe", "paytm")
BANKING_CUES = (
    "bank",
    "account",
    "kyc",
    "verify",
    "blocked",
    "suspended",
    "frozen",
    "debit card",
    "credit card",
    "net banking",
    "netbanking",
)

#: Most specific topics first; links alone fall through to phishing last.
PRIORITY = ("lottery", "otp", "fake_jobs", "investment", "banking", "upi", "phishing")


def related_category_ids(
    signal_codes: Iterable[str], text: str, limit: int = 3
) -> list[str]:
    """Return awareness category ids worth showing for this message.

    The result is ordered by how specific the topic is, is de-duplicated, and
    never contains more than *limit* entries.
    """
    found: set[str] = set()
    for code in signal_codes:
        category_id = SIGNAL_TOPICS.get(code)
        if category_id:
            found.add(category_id)

    lowered = text.lower()
    if any(cue in lowered for cue in UPI_CUES):
        found.add("upi")
    if any(cue in lowered for cue in BANKING_CUES):
        found.add("banking")

    if not found:
        return ["phishing", "otp"][: max(limit, 1)]

    ordered = [category_id for category_id in PRIORITY if category_id in found]
    return ordered[: max(limit, 1)]

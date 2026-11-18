"""Presentation helpers: turn model output into honest, plain-language text.

This module is deliberately free of Streamlit imports so that every string the
user sees can be unit tested. It must never invent a confidence percentage or a
risk band: the model output is uncalibrated and is presented as such.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from .explain import Contribution, Signal

__all__ = [
    "LABEL_TEXT",
    "label_text",
    "score_caption",
    "contribution_rows",
    "signal_rows",
    "format_metric",
    "disclaimer_text",
    "privacy_note",
    "explanation_intro",
]

LABEL_TEXT = {
    "spam": "Likely spam (scam-like or unwanted SMS)",
    "ham": "Likely ham (ordinary message)",
}


def label_text(label: str) -> str:
    """Return the plain-language heading for a predicted label."""
    try:
        return LABEL_TEXT[label]
    except KeyError as exc:
        raise ValueError(f"unknown label: {label!r}") from exc


def score_caption(raw_score: float, threshold: float) -> str:
    """Describe the raw model score without presenting it as a probability."""
    return (
        f"Raw model score {raw_score:.2f} against a decision threshold of "
        f"{threshold:.2f}. This is uncalibrated model output, not a probability "
        "of the message being a scam."
    )


def contribution_rows(contributions: Iterable[Contribution]) -> list[dict[str, str]]:
    """Turn contributions into table rows with direction and signed weight."""
    rows: list[dict[str, str]] = []
    for item in contributions:
        direction = "pushes towards spam" if item.weight > 0 else "pushes towards ham"
        rows.append(
            {
                "term": item.term,
                "direction": direction,
                "weight": f"{item.weight:+.2f}",
            }
        )
    return rows


def signal_rows(signals: Sequence[Signal]) -> list[dict[str, str]]:
    """Turn rule-based signals into table rows for the interface."""
    return [
        {"indicator": signal.label, "detail": signal.detail, "code": signal.code}
        for signal in signals
    ]


def format_metric(value: float | None) -> str:
    """Format a metric to four decimals, or 'n/a' when it is undefined."""
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def explanation_intro() -> str:
    """Explain what the contribution list does and does not mean."""
    return (
        "These are the words that moved the model's score most, computed exactly "
        "from the logistic-regression coefficients multiplied by the TF-IDF value "
        "of each word in this message. They explain the model, not the criminal."
    )


def disclaimer_text() -> str:
    """State the limits of the classifier plainly."""
    return (
        "This tool classifies English SMS text as ham or spam using a model trained "
        "on the UCI SMS Spam Collection. It cannot verify whether a message is a "
        "crime, it does not cover every kind of cyber scam, and it is not a "
        "substitute for official advice or for reporting fraud to the authorities."
    )


def privacy_note() -> str:
    """State what happens to a submitted message."""
    return (
        "Your message is processed in memory to produce this result and is not "
        "stored by the application. Sensitive parts such as links, phone numbers, "
        "amounts and email addresses are masked before the text reaches the model. "
        "If you run the app on a shared machine, avoid pasting real account details."
    )

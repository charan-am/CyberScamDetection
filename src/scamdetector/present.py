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
    "GAUGE_MAX",
    "label_text",
    "result_caveat",
    "score_caption",
    "gauge_score",
    "gauge_threshold",
    "gauge_heading",
    "gauge_ends",
    "contribution_rows",
    "signal_rows",
    "format_metric",
    "disclaimer_text",
    "privacy_note",
    "hosted_privacy_note",
    "local_privacy_note",
    "explanation_intro",
]

LABEL_TEXT = {
    "spam": "Likely spam",
    "ham": "No spam pattern detected",
}

#: The visual scale is 0-10. It is a linear rescaling of raw model output for
#: display only, never a calibrated probability or a risk band.
GAUGE_MAX = 10.0


def label_text(label: str) -> str:
    """Return the plain-language heading for a predicted label."""
    try:
        return LABEL_TEXT[label]
    except KeyError as exc:
        raise ValueError(f"unknown label: {label!r}") from exc


def result_caveat(label: str) -> str:
    """Return the safety caveat that must accompany a predicted label."""
    if label == "spam":
        return (
            "This is a model flag, not proof that the sender is a criminal. "
            "Check the message yourself before acting on it."
        )
    if label == "ham":
        return (
            "The model did not flag this message. That is not a guarantee that "
            "it is safe, because scams it has never seen can still get through."
        )
    raise ValueError(f"unknown label: {label!r}")


def gauge_score(raw_score: float) -> float:
    """Rescale raw model output in [0, 1] onto the 0-10 visual scale."""
    clamped = min(max(float(raw_score), 0.0), 1.0)
    return round(clamped * GAUGE_MAX, 1)


def gauge_threshold(threshold: float) -> float:
    """Place the decision threshold on the same 0-10 visual scale."""
    return gauge_score(threshold)


def gauge_heading() -> str:
    """Name the visual meter without implying a probability."""
    return "Model Spam Signal"


def gauge_ends() -> tuple[str, str]:
    """Label the two ends of the visual meter."""
    return ("Lower model signal", "Higher model signal")


def score_caption(raw_score: float, threshold: float) -> str:
    """Describe the score and threshold without presenting them as risk."""
    return (
        f"{gauge_heading()} {gauge_score(raw_score):.1f}/{GAUGE_MAX:.0f} "
        f"(raw model output {raw_score:.2f}) against a decision threshold of "
        f"{gauge_threshold(threshold):.1f}/{GAUGE_MAX:.0f} (raw {threshold:.2f}). "
        "The 0-10 figure is a linear rescaling of the raw output for display "
        "only. It is uncalibrated model output, not a probability of the "
        "message being a scam."
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
    """State what the application's own code does with a submitted message."""
    return (
        "The application code does not intentionally store submitted messages. "
        "A message is held in memory for the current session only, to produce the "
        "result you see. Sensitive parts such as links, phone numbers, amounts and "
        "email addresses are masked before the text reaches the model. Never paste "
        "a real OTP, PIN, card number or bank account number into any online tool."
    )


def local_privacy_note() -> str:
    """Explain what local execution means for a submitted message."""
    return (
        "Local execution: when you start the app on your own machine, the message "
        "is processed by the Python process on that machine. Inference uses only "
        "the bundled model file; there is no external AI service and no analytics."
    )


def hosted_privacy_note() -> str:
    """Explain hosted execution honestly, including what is not verified."""
    return (
        "Hosted execution: when this app runs on Streamlit Community Cloud, the "
        "message you submit is sent to Streamlit's servers so that the app can "
        "render a result. The application code does not write messages to disk and "
        "does not send message content to an external AI inference API. Streamlit's "
        "own infrastructure logging, retention and access controls are outside this "
        "project's code and have not been independently verified here. If that "
        "matters for your situation, run the app locally instead."
    )

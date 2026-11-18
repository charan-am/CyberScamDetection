"""Tests for presentation helpers (no Streamlit imports)."""

from __future__ import annotations

from scamdetector import present
from scamdetector.explain import Contribution, Signal


def test_label_text_for_spam_is_plain_language():
    text = present.label_text("spam")
    assert "spam" in text.lower()
    assert text != present.label_text("ham")


def test_label_text_rejects_unknown_label():
    try:
        present.label_text("maybe")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown label should raise ValueError")


def test_score_caption_states_uncalibrated_and_has_no_percent():
    caption = present.score_caption(0.83, 0.27)
    assert "not a probability" in caption
    assert "%" not in caption
    assert "0.83" in caption
    assert "0.27" in caption


def test_contribution_rows_mark_direction_and_format_weight():
    rows = present.contribution_rows(
        [Contribution("win", 0.5123), Contribution("free", -0.2)]
    )
    assert rows[0]["term"] == "win"
    assert rows[0]["direction"] == "pushes towards spam"
    assert rows[0]["weight"] == "+0.51"
    assert rows[1]["direction"] == "pushes towards ham"
    assert rows[1]["weight"] == "-0.20"


def test_signal_rows_preserve_code_and_detail():
    rows = present.signal_rows([Signal("url", "Link in message", "http://x.example")])
    assert rows == [
        {"indicator": "Link in message", "detail": "http://x.example", "code": "url"}
    ]


def test_format_metric_handles_missing_values():
    assert present.format_metric(0.93939) == "0.9394"
    assert present.format_metric(None) == "n/a"


def test_disclaimer_mentions_not_a_scam_detector_for_all_crimes():
    text = present.disclaimer_text().lower()
    assert "not a substitute" in text or "cannot" in text
    assert "sms" in text


def test_privacy_note_mentions_no_storage():
    text = present.privacy_note().lower()
    assert "not stored" in text or "not saved" in text or "no message" in text

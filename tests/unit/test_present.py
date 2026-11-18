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


def test_privacy_note_does_not_claim_hosting_guarantees():
    text = present.privacy_note().lower()
    assert "does not intentionally" in text
    assert "entirely on the local machine" not in text


def test_gauge_score_rescales_raw_output_to_ten():
    assert present.gauge_score(0.93) == 9.3
    assert present.gauge_score(0.27) == 2.7
    assert present.gauge_score(0.0) == 0.0


def test_gauge_score_is_clamped_to_the_scale():
    assert present.gauge_score(1.4) == 10.0
    assert present.gauge_score(-0.3) == 0.0


def test_gauge_threshold_uses_the_same_scale():
    assert present.gauge_threshold(0.27) == 2.7
    assert present.gauge_threshold(1.2) == 10.0


def test_gauge_heading_and_ends_are_plain_language():
    assert present.gauge_heading() == "Model Spam Signal"
    assert present.gauge_ends() == ("Lower model signal", "Higher model signal")


def test_score_caption_includes_visual_score_and_disclaimer():
    caption = present.score_caption(0.93, 0.27)
    assert "9.3/10" in caption
    assert "2.7/10" in caption
    assert "uncalibrated" in caption.lower()
    assert "not a probability" in caption
    assert "%" not in caption


def test_ham_label_is_no_spam_pattern_detected():
    assert present.label_text("ham") == "No spam pattern detected"


def test_each_label_carries_a_safety_caveat():
    assert "not a guarantee" in present.result_caveat("ham").lower()
    assert "not proof" in present.result_caveat("spam").lower()


def test_hosted_privacy_note_names_streamlit_cloud_and_limits():
    text = present.hosted_privacy_note()
    assert "Streamlit Community Cloud" in text
    assert "Streamlit" in text
    assert "external" in text.lower()
    assert "independently verified" in text


def test_local_privacy_note_describes_local_execution():
    assert "your own machine" in present.local_privacy_note().lower()


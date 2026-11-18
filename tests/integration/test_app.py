"""Integration tests that drive the real Streamlit app through AppTest."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[2] / "app" / "streamlit_app.py"

SPAM_TEXT = (
    "Congratulations! Your number won Rs 25,00,000 in the lucky draw. "
    "Pay Rs 6,500 processing fee and share your OTP on WhatsApp within 24 hours."
)
HAM_TEXT = "I will be home late tonight, please start dinner without me."


def _app() -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=180).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_app_renders_four_areas():
    at = _app()
    assert len(at.tabs) == 4
    labels = [tab.label for tab in at.tabs]
    assert labels == [
        "Message detection",
        "Cyber awareness",
        "Model performance",
        "About & privacy",
    ]


def test_spam_message_receives_prediction_and_three_layers():
    at = _app()
    at.text_area[0].set_value(SPAM_TEXT)
    at.button[0].click().run()

    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    rendered = "\n".join(m.value for m in at.markdown)
    assert "Likely spam" in rendered
    assert "Layer 1" in rendered
    assert "Layer 2" in rendered
    assert "Layer 3" in rendered
    assert "not a probability" in rendered
    assert "%" not in next(m.value for m in at.markdown if "Raw model score" in m.value)


def test_ham_message_is_not_flagged():
    at = _app()
    at.text_area[0].set_value(HAM_TEXT)
    at.button[0].click().run()

    assert not at.exception, [e.value for e in at.exception]
    rendered = "\n".join(m.value for m in at.markdown)
    assert "Likely ham" in rendered


def test_empty_message_shows_validation_warning():
    at = _app()
    at.button[0].click().run()
    assert not at.exception
    assert any("Enter a message" in w.value for w in at.warning)
    assert not at.error


def test_whitespace_only_message_is_rejected():
    at = _app()
    at.text_area[0].set_value("   \n  ")
    at.button[0].click().run()
    assert not at.exception
    assert at.warning


def test_clear_button_removes_result():
    at = _app()
    at.text_area[0].set_value(SPAM_TEXT)
    at.button[0].click().run()
    assert any("Likely spam" in m.value for m in at.markdown)

    at.button[1].click().run()
    assert not at.exception
    assert at.text_area[0].value == ""
    assert not any("Likely spam" in m.value for m in at.markdown)


def test_kannada_content_is_flagged_as_unreviewed():
    at = _app()
    at.radio[0].set_value("ಕನ್ನಡ (Kannada)").run()

    assert not at.exception
    warnings = "\n".join(w.value for w in at.warning)
    assert warnings
    assert "ಪರಿಶೀಲಿಸಿಲ್ಲ" in warnings


def test_performance_tab_reports_real_metrics():
    at = _app()
    values = [metric.value for metric in at.metric]
    assert "0.9394" in values
    assert "0.8986" in values
    assert "0.9841" in values


def test_non_english_message_gets_limitation_notice():
    at = _app()
    at.text_area[0].set_value("ನಿಮ್ಮ ಖಾತೆಗೆ ಸಂಬಂಧಿಸಿದ ಮಾಹಿತಿ ಬೇಕು ಎಂದು ಕರೆ ಮಾಡಿ")
    at.button[0].click().run()

    assert not at.exception
    infos = "\n".join(i.value for i in at.info)
    assert "English" in infos


def test_about_tab_exposes_dataset_and_privacy():
    at = _app()
    rendered = "\n".join(m.value for m in at.markdown)
    assert "CC BY 4.0" in rendered
    assert "MIT" in rendered
    assert not at.exception

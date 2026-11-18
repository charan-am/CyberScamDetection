"""Tests for script-based language routing (no Streamlit imports)."""

from __future__ import annotations

from scamdetector.language import detect_language, kannada_ratio


def test_english_message_is_detected():
    assert detect_language("Your OTP is 4821. Do not share it with anyone.") == "en"


def test_kannada_script_message_is_detected():
    text = "ನಿಮ್ಮ ಖಾತೆ ಬ್ಲಾಕ್ ಆಗಿದೆ. ಈ ಲಿಂಕ್ ಒತ್ತಿ ಮಾಹಿತಿ ನವೀಕರಿಸಿ."
    assert detect_language(text) == "kn"


def test_mixed_kannada_and_english_is_still_kannada():
    text = "ನಿಮ್ಮ KYC ಗೆ OTP ಕಳುಹಿಸಿ, account verify ಮಾಡಿ"
    assert detect_language(text) == "kn"


def test_other_indic_script_is_not_treated_as_kannada():
    assert detect_language("உங்கள் கணக்கு முடக்கப்பட்டது") == "other"


def test_text_without_letters_is_unknown():
    assert detect_language("12345 + 67890") == "unknown"


def test_romanised_kannada_is_reported_as_latin_script():
    # Documented limitation: romanised Kannada is not Kannada-script detection.
    assert detect_language("nimma account block aagide, otp kalsi") == "en"


def test_kannada_ratio_counts_only_kannada_letters():
    assert kannada_ratio("ಕನ್ನಡ") == 1.0
    assert kannada_ratio("abc") == 0.0
    assert kannada_ratio("") == 0.0
    ratio = kannada_ratio("ಕನ್ನಡ text")
    assert 0.0 < ratio < 1.0

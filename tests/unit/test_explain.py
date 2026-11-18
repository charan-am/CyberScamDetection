import pytest

from scamdetector import config
from scamdetector.explain import detect_signals, top_contributions


@pytest.mark.unit
def test_contributions_are_signed_and_ranked(tiny_pipeline):
    positive, negative = top_contributions(tiny_pipeline, "win free prize", k=5)
    assert positive, "expected at least one spam-leaning term"
    assert positive[0].weight > 0
    assert all(item.weight > 0 for item in positive)
    assert all(item.weight < 0 for item in negative)
    assert positive == sorted(positive, key=lambda item: -item.weight)
    assert negative == sorted(negative, key=lambda item: item.weight)


@pytest.mark.unit
def test_unknown_vocabulary_returns_no_terms(tiny_pipeline):
    positive, negative = top_contributions(tiny_pipeline, "zzz qqq", k=5)
    assert positive == [] and negative == []


@pytest.mark.unit
def test_known_ham_word_leans_negative(tiny_pipeline):
    _, negative = top_contributions(tiny_pipeline, "station tomorrow", k=5)
    assert {item.term for item in negative} & {"station", "tomorrow"}


@pytest.mark.unit
def test_detects_link_phone_amount_and_code_request():
    text = "URGENT! Share your OTP, pay Rs 5000 at http://x.example or call 9876543210"
    codes = {signal.code for signal in detect_signals(text)}
    assert {"url", "phone", "amount", "urgency", "otp"} <= codes


@pytest.mark.unit
def test_detects_shortened_links():
    codes = {signal.code for signal in detect_signals("claim at bit.ly/abc")}
    assert "short_url" in codes


@pytest.mark.unit
def test_detects_caps_abuse():
    codes = {signal.code for signal in detect_signals("CONGRATULATIONS YOU HAVE WON A FREE PRIZE TODAY")}
    assert "caps" in codes


@pytest.mark.unit
def test_plain_message_has_no_signals():
    assert detect_signals("see you at lunch") == []


@pytest.mark.unit
def test_signal_details_are_human_readable():
    for signal in detect_signals("http://x.example"):
        assert signal.label and signal.label[0].isupper()
        assert signal.detail.endswith(".")


@pytest.mark.unit
def test_positive_terms_are_capped_at_k(tiny_pipeline):
    positive, negative = top_contributions(tiny_pipeline, "win free prize claim now meet station", k=2)
    assert len(positive) <= 2 and len(negative) <= 2

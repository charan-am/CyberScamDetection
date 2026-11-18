import pytest

from scamdetector.features import (
    AMOUNT_TOKEN,
    EMAIL_TOKEN,
    PHONE_TOKEN,
    URL_TOKEN,
    TextInputError,
    build_vectorizer,
    mask_sensitive,
    normalise_text,
)


@pytest.mark.unit
def test_mask_replaces_url_email_amount_and_phone():
    text = "Pay Rs.5000 now at http://bit.ly/x or mail a@b.com from 9876543210"
    masked = mask_sensitive(text)
    assert URL_TOKEN in masked
    assert EMAIL_TOKEN in masked
    assert AMOUNT_TOKEN in masked
    assert PHONE_TOKEN in masked
    assert "bit.ly" not in masked
    assert "a@b.com" not in masked
    assert "9876543210" not in masked


@pytest.mark.unit
def test_masking_is_idempotent():
    text = "Win Rs 10,000 at www.prize.example call +91 98765 43210"
    once = mask_sensitive(text)
    assert mask_sensitive(once) == once


@pytest.mark.unit
def test_normalise_collapses_whitespace_and_strips():
    assert normalise_text("  hello   world \n\n") == "hello world"


@pytest.mark.unit
def test_masking_keeps_ordinary_words():
    assert mask_sensitive("meet me at the station") == "meet me at the station"


@pytest.mark.unit
def test_non_text_input_is_rejected():
    with pytest.raises(TextInputError, match="must be text"):
        mask_sensitive(12345)


@pytest.mark.unit
def test_vectorizer_produces_unigrams_and_bigrams():
    texts = ["free prize claim now", "free prize win big", "meet me tomorrow"]
    vectorizer = build_vectorizer((1, 2)).fit(texts)
    names = vectorizer.get_feature_names_out()
    assert "free" in names
    assert "free prize" in names
    assert vectorizer.transform(["free prize"]).shape[0] == 1


@pytest.mark.unit
def test_vectorizer_applies_masking_before_tokenising():
    vectorizer = build_vectorizer((1, 1)).fit(["call 9876543210 now", "call 9999999999 now"])
    names = set(vectorizer.get_feature_names_out())
    assert "phone" in names
    assert "9876543210" not in names

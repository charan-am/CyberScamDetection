"""Tests for mapping message characteristics to awareness topics."""

from __future__ import annotations

from scamdetector.guidance import related_category_ids


def test_prize_language_maps_to_lottery_first():
    topics = related_category_ids(["prize"], "You won a lucky draw prize")
    assert topics[0] == "lottery"


def test_link_maps_to_phishing():
    assert related_category_ids(["url"], "Open this link")[0] == "phishing"


def test_otp_request_maps_to_otp():
    assert related_category_ids(["otp"], "Share the OTP now")[0] == "otp"


def test_payment_instruction_maps_to_upi():
    assert related_category_ids(["payment"], "Scan this QR code to receive money")[0] == "upi"


def test_job_offer_maps_to_fake_jobs():
    assert related_category_ids(["job"], "Work from home, pay registration fee")[0] == "fake_jobs"


def test_investment_language_maps_to_investment():
    topics = related_category_ids(["investment"], "Guaranteed 30% monthly return")
    assert topics[0] == "investment"


def test_bank_account_warning_maps_to_banking_impersonation():
    topics = related_category_ids(
        ["urgency"], "Your bank account will be blocked. Verify KYC immediately."
    )
    assert topics[0] == "banking"


def test_default_topics_when_no_signal_matches():
    assert related_category_ids([], "Hello, see you at six") == ["phishing", "otp"]


def test_topics_are_unique_and_limited():
    topics = related_category_ids(
        ["prize", "otp", "payment", "url"], "You won, share OTP and scan the QR code"
    )
    assert len(topics) == 3
    assert len(set(topics)) == 3
    assert related_category_ids(["prize", "otp"], "prize OTP", limit=1) == ["lottery"]


def test_unknown_signal_codes_are_ignored():
    assert related_category_ids(["mystery"], "plain message") == ["phishing", "otp"]

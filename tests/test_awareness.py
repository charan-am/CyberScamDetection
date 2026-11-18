"""Tests for curated awareness content and reporting channels."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from scamdetector import awareness

EXPECTED_IDS = [
    "upi",
    "otp",
    "phishing",
    "fake_jobs",
    "lottery",
    "investment",
    "banking",
]


def test_english_content_has_seven_reviewed_categories():
    content = awareness.load_awareness("en")
    assert content.language == "en"
    assert content.reviewed is True
    assert [c.id for c in content.categories] == EXPECTED_IDS
    for category in content.categories:
        assert category.title.strip()
        assert category.summary.strip()
        assert category.example.strip()
        assert category.warning_signs
        assert category.prevention
        assert category.checklist


def test_kannada_content_is_marked_unreviewed():
    content = awareness.load_awareness("kn")
    assert content.language == "kn"
    assert content.reviewed is False
    assert content.review_note.strip()
    assert [c.id for c in content.categories] == EXPECTED_IDS


def test_both_languages_cover_the_same_categories():
    en = awareness.load_awareness("en")
    kn = awareness.load_awareness("kn")
    assert [c.id for c in en.categories] == [c.id for c in kn.categories]


def test_unknown_language_raises():
    with pytest.raises(awareness.ContentError):
        awareness.load_awareness("fr")


def test_invalid_json_raises_content_error(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(awareness.ContentError):
        awareness.load_awareness("en", path=path)


def test_missing_required_key_raises(tmp_path):
    path = tmp_path / "partial.json"
    path.write_text(
        json.dumps(
            {
                "language": "en",
                "language_name": "English",
                "reviewed": True,
                "last_updated": "2026-11-18",
                "categories": [
                    {
                        "id": "upi",
                        "title": "T",
                        "summary": "S",
                        "example": "E",
                        "warning_signs": ["a"],
                        "checklist": ["b"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(awareness.ContentError):
        awareness.load_awareness("en", path=path)


def test_duplicate_category_id_raises(tmp_path):
    category = {
        "id": "upi",
        "title": "T",
        "summary": "S",
        "example": "E",
        "warning_signs": ["a"],
        "prevention": ["b"],
        "checklist": ["c"],
    }
    path = tmp_path / "dup.json"
    path.write_text(
        json.dumps(
            {
                "language": "en",
                "language_name": "English",
                "reviewed": True,
                "last_updated": "2026-11-18",
                "categories": [category, category],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(awareness.ContentError):
        awareness.load_awareness("en", path=path)


def test_reporting_channels_are_official_https_with_dates():
    info = awareness.load_reporting()
    assert info.channels
    ids = [channel.id for channel in info.channels]
    assert "cybercrime_portal" in ids
    assert "helpline_1930" in ids
    for channel in info.channels:
        assert channel.name.strip()
        assert channel.purpose.strip()
        assert channel.url.startswith("https://")
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", channel.last_verified)
    helpline = next(c for c in info.channels if c.id == "helpline_1930")
    assert helpline.phone == "1930"


def test_reporting_requires_verification_date(tmp_path):
    path = tmp_path / "reporting.json"
    path.write_text(
        json.dumps({"channels": [{"id": "x", "name": "X", "purpose": "P", "url": "https://x.gov.in"}]}),
        encoding="utf-8",
    )
    with pytest.raises(awareness.ContentError):
        awareness.load_reporting(path=path)


def test_content_files_are_valid_utf8_json():
    for name in ("awareness_en.json", "awareness_kn.json", "reporting.json"):
        path = Path(awareness.CONTENT_DIR) / name
        assert path.is_file()
        json.loads(path.read_text(encoding="utf-8"))

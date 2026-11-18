"""Curated cyber-awareness content and verified reporting channels.

Content lives in ``content/*.json`` so that a reviewer can correct wording without
touching code. Every load validates the schema; nothing is served unvalidated.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import config

__all__ = [
    "CONTENT_DIR",
    "ContentError",
    "Category",
    "AwarenessContent",
    "Channel",
    "ReportingInfo",
    "load_awareness",
    "load_reporting",
    "available_languages",
]

CONTENT_DIR = config.CONTENT_DIR

LANGUAGE_FILES = {"en": config.AWARENESS_EN_PATH, "kn": config.AWARENESS_KN_PATH}
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


class ContentError(RuntimeError):
    """Raised when curated content is missing, unreadable or malformed."""


@dataclass(frozen=True)
class Category:
    id: str
    title: str
    summary: str
    example: str
    warning_signs: tuple[str, ...]
    prevention: tuple[str, ...]
    checklist: tuple[str, ...]


@dataclass(frozen=True)
class AwarenessContent:
    language: str
    language_name: str
    reviewed: bool
    reviewer: str | None
    review_note: str
    last_updated: str
    categories: tuple[Category, ...]


@dataclass(frozen=True)
class Channel:
    id: str
    name: str
    operator: str
    purpose: str
    url: str
    phone: str | None
    last_verified: str
    note: str


@dataclass(frozen=True)
class ReportingInfo:
    last_verified: str
    verification_method: str
    disclaimer: str
    channels: tuple[Channel, ...]


def available_languages() -> tuple[str, ...]:
    """Return the language codes with a bundled content file."""
    return tuple(LANGUAGE_FILES)


def _read_json(path: Path, what: str) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ContentError(f"cannot read {what} at {path}: {exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ContentError(f"{what} at {path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ContentError(f"{what} at {path} must be a JSON object")
    return data


def _text(data: dict[str, Any], key: str, where: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ContentError(f"{where}: '{key}' must be a non-empty string")
    return value


def _text_list(data: dict[str, Any], key: str, where: str) -> tuple[str, ...]:
    value = data.get(key)
    if not isinstance(value, list) or not value:
        raise ContentError(f"{where}: '{key}' must be a non-empty list")
    items: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ContentError(f"{where}: '{key}' item {index} must be a non-empty string")
        items.append(item)
    return tuple(items)


def _check_date(value: str, where: str) -> str:
    if not _DATE_RE.search(value):
        raise ContentError(f"{where}: expected a date like YYYY-MM-DD, got {value!r}")
    return value


def load_awareness(language: str, path: Path | None = None) -> AwarenessContent:
    """Load one language's awareness content, validating its structure."""
    if path is None:
        try:
            path = LANGUAGE_FILES[language]
        except KeyError as exc:
            raise ContentError(
                f"no awareness content for language {language!r}; "
                f"available: {', '.join(LANGUAGE_FILES)}"
            ) from exc

    data = _read_json(path, "awareness content")
    declared = _text(data, "language", str(path))
    if declared != language:
        raise ContentError(
            f"{path}: declares language {declared!r} but was loaded as {language!r}"
        )

    raw_categories = data.get("categories")
    if not isinstance(raw_categories, list) or not raw_categories:
        raise ContentError(f"{path}: 'categories' must be a non-empty list")

    categories: list[Category] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_categories):
        where = f"{path}: categories[{index}]"
        if not isinstance(raw, dict):
            raise ContentError(f"{where} must be an object")
        category_id = _text(raw, "id", where)
        if category_id in seen:
            raise ContentError(f"{where}: duplicate category id {category_id!r}")
        seen.add(category_id)
        categories.append(
            Category(
                id=category_id,
                title=_text(raw, "title", where),
                summary=_text(raw, "summary", where),
                example=_text(raw, "example", where),
                warning_signs=_text_list(raw, "warning_signs", where),
                prevention=_text_list(raw, "prevention", where),
                checklist=_text_list(raw, "checklist", where),
            )
        )

    raw_reviewed = data.get("reviewed")
    if not isinstance(raw_reviewed, bool):
        raise ContentError(f"{path}: 'reviewed' must be true or false")
    reviewer = data.get("reviewer")
    if reviewer is not None and not isinstance(reviewer, str):
        raise ContentError(f"{path}: 'reviewer' must be a string or null")

    return AwarenessContent(
        language=declared,
        language_name=_text(data, "language_name", str(path)),
        reviewed=raw_reviewed,
        reviewer=reviewer,
        review_note=str(data.get("review_note", "")),
        last_updated=_check_date(_text(data, "last_updated", str(path)), str(path)),
        categories=tuple(categories),
    )


def load_reporting(path: Path | None = None) -> ReportingInfo:
    """Load the reporting channels with their verification dates."""
    path = path or config.REPORTING_PATH
    data = _read_json(path, "reporting information")

    raw_channels = data.get("channels")
    if not isinstance(raw_channels, list) or not raw_channels:
        raise ContentError(f"{path}: 'channels' must be a non-empty list")

    channels: list[Channel] = []
    for index, raw in enumerate(raw_channels):
        where = f"{path}: channels[{index}]"
        if not isinstance(raw, dict):
            raise ContentError(f"{where} must be an object")
        url = _text(raw, "url", where)
        if not url.startswith("https://"):
            raise ContentError(f"{where}: 'url' must start with https://, got {url!r}")
        phone = raw.get("phone")
        if phone is not None and not isinstance(phone, str):
            raise ContentError(f"{where}: 'phone' must be a string or null")
        channels.append(
            Channel(
                id=_text(raw, "id", where),
                name=_text(raw, "name", where),
                operator=_text(raw, "operator", where),
                purpose=_text(raw, "purpose", where),
                url=url,
                phone=phone,
                last_verified=_check_date(
                    _text(raw, "last_verified", where), where
                ),
                note=str(raw.get("note", "")),
            )
        )

    return ReportingInfo(
        last_verified=_check_date(
            _text(data, "last_verified", str(path)), str(path)
        ),
        verification_method=str(data.get("verification_method", "")),
        disclaimer=str(data.get("disclaimer", "")),
        channels=tuple(channels),
    )

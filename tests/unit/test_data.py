import json

import pandas as pd
import pytest

from scamdetector import config
from scamdetector.data import (
    DatasetError,
    add_dedupe_key,
    load_split,
    parse_dataset_text,
    prepare_dataset,
    resolve_duplicates,
    split_dataset,
    verify_no_leakage,
)


def make_frame(rows):
    frame = pd.DataFrame(rows, columns=["label", "text"])
    frame.index = range(1, len(frame) + 1)
    frame.index.name = "source_line"
    return frame


@pytest.mark.unit
def test_parse_valid_rows():
    frame = parse_dataset_text("ham\tHello there\nspam\tWin a prize\n")
    assert list(frame["label"]) == ["ham", "spam"]
    assert list(frame["text"]) == ["Hello there", "Win a prize"]
    assert list(frame.index) == [1, 2]


@pytest.mark.unit
def test_parse_ignores_blank_lines():
    frame = parse_dataset_text("ham\tHello\n\nspam\tWin\n")
    assert len(frame) == 2


@pytest.mark.unit
@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("ham Hello\n", "no tab separator"),
        ("spammy\tText\n", "unknown label"),
        ("ham\t   \n", "empty message"),
    ],
)
def test_parse_rejects_malformed_rows_with_line_number(payload, message):
    with pytest.raises(DatasetError) as error:
        parse_dataset_text(payload)
    assert message in str(error.value)
    assert "line 1" in str(error.value)


@pytest.mark.unit
def test_parse_rejects_empty_dataset():
    with pytest.raises(DatasetError, match="no usable rows"):
        parse_dataset_text("\n\n")


@pytest.mark.unit
def test_dedupe_key_normalises_case_whitespace_and_identifiers():
    frame = make_frame(
        [
            ("ham", "Call  9876543210 now"),
            ("ham", "call 9999999999 NOW"),
        ]
    )
    keyed = add_dedupe_key(frame)
    assert keyed["dedupe_key"].nunique() == 1


@pytest.mark.unit
def test_duplicates_are_removed_and_counted():
    frame = make_frame(
        [
            ("ham", "same message"),
            ("ham", "SAME   message"),
            ("spam", "unrelated text"),
        ]
    )
    resolved, report = resolve_duplicates(frame)
    assert len(resolved) == 2
    assert report["duplicates_removed"] == 1
    assert report["conflict_groups"] == 0


@pytest.mark.unit
def test_conflicting_labels_resolved_by_majority_and_reported():
    frame = make_frame(
        [
            ("spam", "identical text"),
            ("spam", "IDENTICAL text"),
            ("ham", "identical   text"),
        ]
    )
    resolved, report = resolve_duplicates(frame)
    assert report["conflict_groups"] == 1
    assert report["conflicts_resolved_by_majority"] == 1
    assert report["conflicts"][0]["kept_label"] == "spam"
    assert list(resolved["label"]) == ["spam"]


@pytest.mark.unit
def test_conflicting_labels_with_tie_are_dropped_not_guessed():
    frame = make_frame([("ham", "identical text"), ("spam", "IDENTICAL text")])
    resolved, report = resolve_duplicates(frame)
    assert report["conflicts_dropped_as_ties"] == 1
    assert resolved.empty


@pytest.mark.unit
def test_split_is_deterministic_and_leak_free():
    rows = []
    for index in range(200):
        label = "spam" if index % 5 == 0 else "ham"
        rows.append((label, f"message number {index} unique"))
    frame, _ = resolve_duplicates(make_frame(rows))

    first = split_dataset(frame)
    second = split_dataset(frame)

    for name in config.SPLIT_NAMES:
        assert first[name].equals(second[name])
    assert verify_no_leakage(first) == {"train__val": 0, "train__test": 0, "val__test": 0}


@pytest.mark.unit
def test_split_rejects_duplicate_keys():
    frame = make_frame([("ham", "a b"), ("ham", "a b"), ("spam", "c d")])
    frame = add_dedupe_key(frame)
    with pytest.raises(DatasetError, match="Duplicate dedupe_key"):
        split_dataset(frame)


@pytest.mark.unit
def test_split_rejects_bad_ratios():
    frame, _ = resolve_duplicates(make_frame([("ham", f"msg {i}") for i in range(20)]))
    with pytest.raises(DatasetError, match="sum to 1.0"):
        split_dataset(frame, ratios=(0.5, 0.1, 0.1))


@pytest.mark.unit
def test_load_split_reports_missing_file(tmp_path):
    with pytest.raises(DatasetError, match="prepare_data"):
        load_split("train", processed_dir=tmp_path)


@pytest.mark.integration
def test_prepare_dataset_writes_reports_and_is_byte_stable(tmp_path):
    splits_dir = tmp_path / "splits"
    reports_dir = tmp_path / "reports"

    summary = prepare_dataset(out_dir=splits_dir, reports_dir=reports_dir)
    first = {name: (splits_dir / f"{name}.csv").read_bytes() for name in config.SPLIT_NAMES}
    prepare_dataset(out_dir=splits_dir, reports_dir=reports_dir)
    second = {name: (splits_dir / f"{name}.csv").read_bytes() for name in config.SPLIT_NAMES}

    assert first == second
    assert summary["raw_rows"] == config.EXPECTED_RAW_ROWS
    assert summary["leakage_check"] == {"train__val": 0, "train__test": 0, "val__test": 0}

    conflict_report = json.loads((reports_dir / "conflict_report.json").read_text(encoding="utf-8"))
    assert conflict_report["duplicates_removed"] > 0
    assert conflict_report["rows_after_resolution"] + conflict_report["duplicates_removed"] == 5574

    total = sum(part["rows"] for part in summary["splits"].values())
    assert total == conflict_report["rows_after_resolution"]

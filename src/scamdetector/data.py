"""Dataset loading, validation, deduplication and reproducible splitting.

The official corpus ships as tab-separated ``label<TAB>text`` lines. Nothing here
fits a model: this module only turns the raw file into clean, leak-free splits.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config
from .features import ensure_text, normalise_text


class DatasetError(Exception):
    """Raised when the dataset is missing, malformed or cannot be split."""


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_dataset_bytes(raw: bytes) -> str:
    """Decode the corpus, which is UTF-8 for this release and cp1252 in older copies."""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("cp1252")


def parse_dataset_text(text: str) -> pd.DataFrame:
    """Parse raw corpus text into a validated ``label``/``text`` frame.

    Raises
    ------
    DatasetError
        With the offending line numbers when a row has no tab separator, an
        unknown label, or empty text.
    """
    records: list[tuple[int, str, str]] = []
    problems: list[str] = []

    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        label, separator, body = line.partition("\t")
        label = label.strip().lower()
        body = body.strip()
        if not separator:
            problems.append(f"line {line_number}: no tab separator between label and text")
        elif label not in config.LABELS:
            problems.append(f"line {line_number}: unknown label {label!r}")
        elif not body:
            problems.append(f"line {line_number}: empty message text")
        else:
            records.append((line_number, label, body))

    if problems:
        preview = "; ".join(problems[:5])
        more = f" (+{len(problems) - 5} more)" if len(problems) > 5 else ""
        raise DatasetError(f"{len(problems)} malformed row(s): {preview}{more}")

    if not records:
        raise DatasetError("Dataset contains no usable rows.")

    frame = pd.DataFrame(
        {"label": [r[1] for r in records], "text": [r[2] for r in records]},
        index=[r[0] for r in records],
    )
    frame.index.name = "source_line"
    return frame


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Load and validate the raw UCI corpus."""
    path = Path(path or config.RAW_DATASET)
    if not path.exists():
        raise DatasetError(
            f"Dataset not found at {path}. Run: python scripts\\get_data.py"
        )
    frame = parse_dataset_text(decode_dataset_bytes(path.read_bytes()))
    frame.attrs["source_path"] = str(path)
    frame.attrs["sha256"] = sha256_file(path)
    return frame


def add_dedupe_key(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the normalised exact-match key used for deduplication and leakage checks."""
    result = frame.copy()
    result["dedupe_key"] = [
        normalise_text(ensure_text(text)).lower() for text in result["text"]
    ]
    return result


def resolve_duplicates(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Collapse rows sharing a ``dedupe_key`` and report every conflict.

    Majority label wins inside a key. A tie is not guessed: the whole key is
    dropped and recorded, so the model never trains on a contradictory example.
    """
    keyed = add_dedupe_key(frame) if "dedupe_key" not in frame.columns else frame.copy()

    kept_rows: list[int] = []
    conflicts: list[dict] = []
    majority_resolved = 0
    tie_drops = 0

    for dedupe_key, group in keyed.groupby("dedupe_key", sort=True):
        counts = group["label"].value_counts().to_dict()
        if len(counts) > 1:
            ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
            is_tie = len(ranked) > 1 and ranked[0][1] == ranked[1][1]
            if is_tie:
                tie_drops += 1
                resolution = "dropped_tie"
                kept_label = None
            else:
                majority_resolved += 1
                resolution = "majority"
                kept_label = ranked[0][0]
            conflicts.append(
                {
                    "dedupe_key": dedupe_key,
                    "label_counts": counts,
                    "resolution": resolution,
                    "kept_label": kept_label,
                    "sample_text": group["text"].iloc[0][:160],
                }
            )
            if kept_label is None:
                continue
            candidates = group[group["label"] == kept_label]
            kept_rows.append(int(candidates.index[0]))
        else:
            kept_rows.append(int(group.index[0]))

    resolved = keyed.loc[sorted(kept_rows)].reset_index(drop=True)
    removed = len(frame) - len(resolved)
    report = {
        "raw_rows": int(len(frame)),
        "unique_dedupe_keys": int(keyed["dedupe_key"].nunique()),
        "duplicates_removed": int(removed),
        "rows_after_resolution": int(len(resolved)),
        "conflict_groups": len(conflicts),
        "conflicts_resolved_by_majority": majority_resolved,
        "conflicts_dropped_as_ties": tie_drops,
        "conflicts": conflicts,
    }
    return resolved, report


def split_dataset(
    frame: pd.DataFrame,
    ratios: tuple[float, float, float] = config.SPLIT_RATIOS,
    seed: int = config.SEED,
) -> dict[str, pd.DataFrame]:
    """Group-aware stratified 80/10/10 split.

    Rows sharing a ``dedupe_key`` must already be collapsed, so the stratification
    unit and the leakage unit are the same: each key lands in exactly one split.
    """
    if "dedupe_key" not in frame.columns:
        raise DatasetError("split_dataset requires a dedupe_key column; call resolve_duplicates first.")
    if len(frame) != frame["dedupe_key"].nunique():
        raise DatasetError("Duplicate dedupe_key values remain; resolve duplicates before splitting.")
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise DatasetError(f"Split ratios must sum to 1.0, received {sum(ratios)}.")

    train_ratio, val_ratio, test_ratio = ratios
    holdout_ratio = val_ratio + test_ratio
    if any(count < 1 for count in frame["label"].value_counts()):
        raise DatasetError("Every class needs at least one row before splitting.")

    train_frame, holdout = train_test_split(
        frame,
        test_size=holdout_ratio,
        stratify=frame["label"],
        random_state=seed,
        shuffle=True,
    )
    val_share = test_ratio / holdout_ratio
    val_frame, test_frame = train_test_split(
        holdout,
        test_size=1.0 - val_share,
        stratify=holdout["label"],
        random_state=seed,
        shuffle=True,
    )
    splits = {"train": train_frame, "val": val_frame, "test": test_frame}
    for name, part in splits.items():
        if part.empty:
            raise DatasetError(f"Split {name!r} is empty; adjust ratios or collect more data.")
        splits[name] = part.sort_values("dedupe_key").reset_index(drop=True)
    return splits


def verify_no_leakage(splits: dict[str, pd.DataFrame]) -> dict[str, int]:
    """Return pairwise overlap counts of ``dedupe_key`` across splits (all must be 0)."""
    keys = {name: set(part["dedupe_key"]) for name, part in splits.items()}
    names = list(keys)
    overlaps: dict[str, int] = {}
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlaps[f"{left}__{right}"] = len(keys[left] & keys[right])
    return overlaps


def save_splits(splits: dict[str, pd.DataFrame], out_dir: Path | None = None) -> dict[str, str]:
    """Write each split to CSV with a fixed column order and row order."""
    out_dir = Path(out_dir or config.PROCESSED_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for name in config.SPLIT_NAMES:
        part = splits[name].loc[:, ["label", "text", "dedupe_key"]]
        destination = out_dir / f"{name}.csv"
        part.to_csv(destination, index=False, encoding="utf-8", lineterminator="\n")
        paths[name] = str(destination)
    return paths


def load_split(name: str, processed_dir: Path | None = None) -> pd.DataFrame:
    """Load one processed split, with a readable error when it is missing."""
    if name not in config.SPLIT_NAMES:
        raise DatasetError(f"Unknown split {name!r}; expected one of {config.SPLIT_NAMES}.")
    path = Path(processed_dir or config.PROCESSED_DIR) / f"{name}.csv"
    if not path.exists():
        raise DatasetError(f"Split file missing: {path}. Run: python scripts\\prepare_data.py")
    frame = pd.read_csv(path, keep_default_na=False)
    missing = {"label", "text", "dedupe_key"} - set(frame.columns)
    if missing:
        raise DatasetError(f"Split file {path} is missing columns: {sorted(missing)}.")
    return frame


def load_all_splits(processed_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    return {name: load_split(name, processed_dir) for name in config.SPLIT_NAMES}


def split_summary(
    splits: dict[str, pd.DataFrame],
    raw_rows: int,
    dataset_sha256: str | None,
    dedupe_report: dict,
) -> dict:
    """Build the JSON-serialisable description written to ``reports/split_summary.json``."""
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": config.SEED,
        "ratios": dict(zip(config.SPLIT_NAMES, config.SPLIT_RATIOS)),
        "raw_rows": raw_rows,
        "expected_raw_rows": config.EXPECTED_RAW_ROWS,
        "expected_label_counts": config.EXPECTED_LABEL_COUNTS,
        "dataset_sha256": dataset_sha256,
        "dedupe": {key: dedupe_report[key] for key in dedupe_report if key != "conflicts"},
        "splits": {},
        "leakage_check": verify_no_leakage(splits),
    }
    for name, part in splits.items():
        counts = part["label"].value_counts().to_dict()
        summary["splits"][name] = {
            "rows": int(len(part)),
            "ham": int(counts.get("ham", 0)),
            "spam": int(counts.get("spam", 0)),
        }
    return summary


def prepare_dataset(
    raw_path: Path | None = None,
    out_dir: Path | None = None,
    reports_dir: Path | None = None,
) -> dict:
    """Run the full preparation pipeline and write splits + reports."""
    raw_frame = load_raw(raw_path)
    resolved, dedupe_report = resolve_duplicates(raw_frame)
    splits = split_dataset(resolved)
    save_splits(splits, out_dir)

    reports_dir = Path(reports_dir or config.REPORTS_DIR)
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary = split_summary(
        splits,
        raw_rows=len(raw_frame),
        dataset_sha256=raw_frame.attrs.get("sha256"),
        dedupe_report=dedupe_report,
    )
    (reports_dir / "conflict_report.json").write_text(
        json.dumps(dedupe_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (reports_dir / "split_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary

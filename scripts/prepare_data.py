"""Build leak-free train/validation/test splits and report duplicates.

Usage::

    python scripts\\prepare_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scamdetector.data import DatasetError, prepare_dataset  # noqa: E402


def main() -> int:
    try:
        summary = prepare_dataset()
    except DatasetError as error:
        print(f"ERROR: {error}")
        return 1

    print(json.dumps(summary["dedupe"], indent=2))
    print("Splits:")
    for name, part in summary["splits"].items():
        print(f"  {name:<6} rows={part['rows']:>5}  ham={part['ham']:>5}  spam={part['spam']:>4}")
    print(f"Leakage: {summary['leakage_check']}")
    print("Reports: reports/split_summary.json, reports/conflict_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

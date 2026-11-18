"""Verify (or fetch) the UCI SMS Spam Collection and print its fingerprint.

Usage::

    python scripts\\get_data.py
    python scripts\\get_data.py --force-refresh
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scamdetector import config  # noqa: E402
from scamdetector.data import DatasetError, load_raw, sha256_file  # noqa: E402

DATASET_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"


def download_dataset(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {DATASET_URL}")
    with urllib.request.urlopen(DATASET_URL, timeout=60) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = {name.lower(): name for name in archive.namelist()}
        for wanted in ("smsspamcollection", "readme"):
            if wanted not in names:
                raise DatasetError(f"Downloaded archive does not contain {wanted!r}.")
            target = config.RAW_DIR / names[wanted]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(names[wanted]))
            print(f"Extracted {target.name} ({target.stat().st_size:,} bytes)")


def report(path: Path) -> int:
    frame = load_raw(path)
    digest = sha256_file(path)
    counts = frame["label"].value_counts().to_dict()
    print(f"File:      {path}")
    print(f"SHA-256:   {digest}")
    print(f"Rows:      {len(frame):,}")
    print(f"ham:       {counts.get('ham', 0):,}")
    print(f"spam:      {counts.get('spam', 0):,}")

    expected_digest = config.DATASET_SHA256
    status = 0
    if digest.lower() != expected_digest.lower():
        print(f"WARNING:   checksum differs from the recorded {expected_digest}")
        status = 1
    if len(frame) != config.EXPECTED_RAW_ROWS:
        print(f"WARNING:   row count differs from the recorded {config.EXPECTED_RAW_ROWS:,}")
        status = 1
    if status == 0:
        print("Status:    matches the recorded UCI release")
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-refresh", action="store_true", help="re-download the archive")
    args = parser.parse_args(argv)

    path = config.RAW_DATASET
    if args.force_refresh or not path.exists():
        download_dataset(path)
    try:
        return report(path)
    except DatasetError as error:
        print(f"ERROR: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

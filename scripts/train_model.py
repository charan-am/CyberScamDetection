"""Train the model, freeze the threshold and write every artifact.

Usage::

    python scripts\\train_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scamdetector.data import DatasetError  # noqa: E402
from scamdetector.train import train_and_persist  # noqa: E402


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def main() -> int:
    try:
        result = train_and_persist()
    except DatasetError as error:
        print(f"ERROR: {error}")
        return 1
    except FileNotFoundError as error:
        print(f"ERROR: {error}")
        return 1

    print("Training complete")
    print(f"  chosen: C={result['chosen_params']['C']} ngram={result['chosen_params']['ngram_range']}")
    print(f"  threshold (validation F2): {result['threshold']:.4f}")
    print(f"  validation F1: {percent(result['validation']['f1'])}")
    print(f"  test precision: {percent(result['test']['precision'])}")
    print(f"  test recall:    {percent(result['test']['recall'])}")
    print(f"  test F1:        {percent(result['test']['f1'])}")
    print(f"  test PR-AUC:    {percent(result['test']['pr_auc'])}")
    print(f"  artifact: {result['artifact']} ({result['size_bytes']:,} bytes)")
    print(f"  sha256:   {result['sha256']}")
    print(f"  reports:  {result['metrics_path']}")
    if not result["committable"]:
        print("  NOTE: artifact exceeds 10 MB and must not be committed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

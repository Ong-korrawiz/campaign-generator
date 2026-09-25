"""Verify generated Day 1 artifacts without loading a model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..constants import SPLITS
from ..schemas import validate_training_record
from .pipeline import brief_fingerprint, read_jsonl, sha256_file


def verify_run(directory: Path) -> dict:
    """Check output hashes, schema, counts, and exact brief overlap."""
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    errors: list[str] = []
    counts = {}
    fingerprints: dict[str, set[str]] = {}
    for split in SPLITS:
        path = directory / f"{split}.jsonl"
        if sha256_file(path) != manifest["outputs_sha256"][split]:
            errors.append(f"checksum_mismatch:{split}")
        fingerprints[split] = set()
        count = 0
        for line_number, row in read_jsonl(path):
            count += 1
            row_errors = validate_training_record(row)
            if row_errors:
                errors.append(f"{split}:{line_number}:{','.join(row_errors)}")
            if row["source"]["split"] != split:
                errors.append(f"source_split_mismatch:{split}:{line_number}")
            fingerprint = brief_fingerprint(row)
            if fingerprint in fingerprints[split]:
                errors.append(f"duplicate_within_split:{split}:{line_number}")
            fingerprints[split].add(fingerprint)
        counts[split] = count
        if count != manifest["zarn"]["kept_counts"][split]:
            errors.append(f"count_mismatch:{split}")
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1 :]:
            overlap = fingerprints[left] & fingerprints[right]
            if overlap:
                errors.append(f"exact_brief_leakage:{left}:{right}:{len(overlap)}")
    return {
        "status": "pass" if not errors else "fail",
        "counts": counts,
        "trainability": manifest["trainability"]["status"],
        "errors": errors[:20],
        "error_count": len(errors),
    }


def main() -> None:
    """Print verification results and fail when structural checks fail."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    result = verify_run(args.run_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

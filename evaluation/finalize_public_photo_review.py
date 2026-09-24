"""Join a hand-reviewed decision table with immutable fetched-photo evidence."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from scanner.catalog import ROOT, sha256_file

FIELDS = (
    "slug",
    "status",
    "reason",
    "photo_kind",
    "page_url",
    "image_url",
    "fetched_at_utc",
    "page_sha256",
    "image_sha256",
    "reference_sha256",
    "phash_hamming",
)


def finalize(audit_path: Path, decisions_path: Path, output_path: Path) -> dict:
    audit = [
        json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()
    ]
    downloaded = {
        (row["slug"], row["image_sha256"]): row
        for row in audit
        if row.get("local_image")
    }
    with decisions_path.open(newline="", encoding="utf-8") as stream:
        decisions = list(csv.DictReader(stream, delimiter="\t"))
    if len(decisions) != len(downloaded):
        raise ValueError("Every downloaded image needs exactly one decision")
    reviewed = []
    seen = set()
    for decision in decisions:
        key = decision["slug"], decision["image_sha256"]
        if key not in downloaded or key in seen:
            raise ValueError(f"Unknown or repeated decision: {key}")
        seen.add(key)
        if decision["status"] not in {"verified", "ambiguous", "reject"}:
            raise ValueError(f"Invalid status: {decision['status']}")
        if not decision["reason"] or not decision["photo_kind"]:
            raise ValueError("Reason and photo kind are required")
        source = downloaded[key]
        reviewed.append(
            {
                field: decision[field] if field in decision else source.get(field, "")
                for field in FIELDS
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=FIELDS, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(reviewed)
    statuses = Counter(row["status"] for row in reviewed)
    return {
        "reviewed_images": len(reviewed),
        "status": dict(statuses),
        "source_images_unique": len(set(row["image_sha256"] for row in reviewed)),
        "audit_sha256": sha256_file(audit_path),
        "decisions_sha256": sha256_file(decisions_path),
        "review_sha256": sha256_file(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit",
        type=Path,
        default=ROOT / "data/external/catalog_photo_pilot_audit.jsonl",
    )
    parser.add_argument(
        "--decisions",
        type=Path,
        default=ROOT
        / "evaluation/manifests/catalog_public_photo_decisions_2026-09-24.tsv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "evaluation/manifests/catalog_public_photo_review_2026-09-24.tsv",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            finalize(args.audit, args.decisions, args.output), ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()

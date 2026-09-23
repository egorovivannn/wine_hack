"""Check complete organizer-photo labels against source bytes and catalog slugs.

Run from the repository root with ``uv run --locked python evaluation/validate_official_labels.py``.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "evaluation" / "all_official_labels.tsv"
PHOTOS = ROOT / "data" / "official_real_photos"
CATALOG = ROOT / "df_2.csv"
FIELDS = ["image_path", "image_sha256", "status", "slug", "evidence"]


def main() -> int:
    with LABELS.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != FIELDS:
            raise ValueError(f"Expected columns {FIELDS}")
        rows = list(reader)

    photo_paths = {path.name: path for path in PHOTOS.glob("*.webp")}
    names = [row["image_path"] for row in rows]
    if len(rows) != 100 or len(photo_paths) != 100 or len(set(names)) != 100:
        raise ValueError("Expected exactly 100 unique labels and 100 source photos")
    if set(names) != set(photo_paths):
        raise ValueError(f"Coverage mismatch: missing={set(photo_paths)-set(names)}, extra={set(names)-set(photo_paths)}")

    with CATALOG.open(newline="", encoding="utf-8") as stream:
        slugs = {row["slug"] for row in csv.DictReader(stream)}

    counts = {"verified": 0, "unknown": 0, "ambiguous": 0}
    for row in rows:
        name = row["image_path"]
        digest = hashlib.sha256(photo_paths[name].read_bytes()).hexdigest()
        if row["image_sha256"] != digest:
            raise ValueError(f"Photo SHA-256 mismatch: {name}")
        status = row["status"]
        if status not in counts:
            raise ValueError(f"Invalid status for {name}: {status}")
        if not row["evidence"].strip():
            raise ValueError(f"Missing evidence: {name}")
        if status == "verified":
            if row["slug"] not in slugs:
                raise ValueError(f"Slug missing from catalog: {name}: {row['slug']}")
        elif row["slug"]:
            raise ValueError(f"Unverified row has slug: {name}")
        counts[status] += 1

    labels_hash = hashlib.sha256(LABELS.read_bytes()).hexdigest()
    print(f"100/100 photos covered; {counts}; labels SHA-256={labels_hash}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

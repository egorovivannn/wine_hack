"""Validated, photo-hash-bound manual labels for offline scoring."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ManualLabel:
    image_sha256: str
    status: str
    slug: str
    evidence: str


def read_labels(path: Path | None) -> dict[str, ManualLabel]:
    if path is None:
        return {}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != ["image_path", "image_sha256", "status", "slug", "evidence"]:
            raise ValueError("Labels need image_path, image_sha256, status, slug, evidence columns")
        labels = {}
        for row in reader:
            name = row["image_path"]
            status = row["status"]
            digest = row["image_sha256"] or ""
            if (not name or name in labels or len(digest) != 64 or
                    any(char not in "0123456789abcdef" for char in digest) or
                    status not in {"verified", "unknown", "ambiguous"} or
                    bool(row["slug"]) != (status == "verified") or
                    not row["evidence"]):
                raise ValueError(f"Invalid or duplicate label: {name}")
            labels[name] = ManualLabel(digest, status, row["slug"], row["evidence"])
    return labels

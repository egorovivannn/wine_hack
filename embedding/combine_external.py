"""Combine separately prepared external views and frozen features without re-encoding."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wine-views", type=Path, required=True)
    parser.add_argument("--wine-features", type=Path, required=True)
    parser.add_argument("--norwegian-views", type=Path, required=True)
    parser.add_argument("--norwegian-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists")
    records = []
    matrices = []
    sources = (
        (args.wine_views, args.wine_features),
        (args.norwegian_views, args.norwegian_features),
    )
    for root, features in sources:
        rows = [
            json.loads(line) for line in (root / "views.jsonl").read_text().splitlines()
        ]
        with np.load(features) as saved:
            if saved["views"].tolist() != [r["view"] for r in rows]:
                raise ValueError("Feature/view order differs")
            matrices.append(saved["vectors"].astype(np.float32))
        records.extend((root, row) for row in rows)
    if len({row["view"] for _, row in records}) != len(records):
        raise ValueError("View path collision")
    args.output.mkdir(parents=True)
    for root, row in records:
        source = root / row["view"]
        if sha256(source) != row["view_sha256"]:
            raise ValueError(f"View changed: {source}")
        target = args.output / row["view"]
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)
    manifest = args.output / "views.jsonl"
    manifest.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for _, row in records
        )
    )
    output_features = args.output / "features.npz"
    np.savez_compressed(
        output_features,
        vectors=np.concatenate(matrices),
        views=np.asarray([row["view"] for _, row in records]),
    )
    report = dict(
        sources=[
            dict(
                views_sha256=sha256(root / "views.jsonl"),
                features_sha256=sha256(features),
            )
            for root, features in sources
        ],
        views_sha256=sha256(manifest),
        features_sha256=sha256(output_features),
        count=len(records),
    )
    (args.output / "provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Select a reproducible catalog-photo search pilot without touching query holdouts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from scanner.catalog import ROOT, sha256_file


def select(manifest_path: Path, index_path: Path, seed: int = 20260924) -> list[dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with np.load(index_path, allow_pickle=False) as archive:
        names = archive["filenames"].tolist()
        embeddings = archive["embeddings"].astype(np.float32)
    if names != [item["filename"] for item in manifest["images"]]:
        raise ValueError("Index does not match catalog image order")
    cards = {card["slug"]: card for card in manifest["cards"]}
    image_by_slug = {
        slug: index
        for index, item in enumerate(manifest["images"])
        for slug in item["slugs"]
    }
    winery_groups: dict[str, list[str]] = {}
    for slug in sorted(image_by_slug):
        winery_groups.setdefault(cards[slug]["winery"].casefold().strip(), []).append(
            slug
        )
    vectors = embeddings[:, 1]
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    pairs = []
    for winery, slugs in winery_groups.items():
        for pos, left in enumerate(slugs):
            for right in slugs[pos + 1 :]:
                if image_by_slug[left] == image_by_slug[right]:
                    continue
                similarity = float(
                    vectors[image_by_slug[left]] @ vectors[image_by_slug[right]]
                )
                pairs.append((similarity, winery, left, right))
    pairs.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))
    selected: dict[str, tuple[str, str]] = {}
    used_wineries = set()
    for similarity, winery, left, right in pairs:
        if winery in used_wineries or left in selected or right in selected:
            continue
        selected[left] = ("hard_pair", right)
        selected[right] = ("hard_pair", left)
        used_wineries.add(winery)
        if len(selected) == 50:
            break
    if len(selected) != 50:
        raise ValueError("Not enough visually similar winery pairs")
    rng = np.random.default_rng(seed)
    pool = sorted(set(image_by_slug) - set(selected))
    for slug in rng.choice(pool, size=50, replace=False):
        selected[str(slug)] = ("random", "")
    return [
        {
            "slug": slug,
            "selection": kind,
            "paired_slug": pair,
            "name": cards[slug]["name"],
            "winery": cards[slug]["winery"],
            "category": cards[slug]["category"],
            "color": cards[slug]["color"],
            "reference_filename": cards[slug]["image_name"],
            "reference_sha256": manifest["images"][image_by_slug[slug]]["sha256"],
        }
        for slug, (kind, pair) in sorted(selected.items())
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "data/catalog_manifest.json"
    )
    parser.add_argument("--index", type=Path, default=ROOT / "data/index/siglip2.npz")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "evaluation/manifests/catalog_photo_pilot_100.tsv",
    )
    args = parser.parse_args()
    rows = select(args.manifest, args.index)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            {
                "selected": len(rows),
                "hard_pair": 50,
                "random": 50,
                "manifest_sha256": sha256_file(args.manifest),
                "index_sha256": sha256_file(args.index),
                "output_sha256": sha256_file(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()

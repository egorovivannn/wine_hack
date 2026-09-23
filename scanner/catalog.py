"""Build a deterministic, checked catalog manifest from df_2.csv and Strapi images."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_COLUMNS = {
    "vine_name", "category", "color", "region", "grape_type", "desc",
    "winery", "slug", "fname", "is_valid_img",
}


@dataclass(frozen=True)
class WineCard:
    slug: str
    name: str
    category: str
    color: str
    region: str
    grape: str
    description: str
    winery: str
    image_name: str
    reference_available: bool


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _card_from_row(row: dict[str, str]) -> WineCard:
    name = row["fname"].strip()
    if not name or Path(name).name != name or "\\" in name or "\0" in name:
        raise ValueError(f"Unsafe or empty image filename: {name!r}")
    valid = row["is_valid_img"].strip().lower()
    if valid not in {"true", "false"}:
        raise ValueError(f"Expected true/false is_valid_img for {row['slug']!r}")
    slug = row["slug"].strip()
    if not slug:
        raise ValueError("Empty slug in catalog")
    return WineCard(
        slug=slug,
        name=row["vine_name"].strip(),
        category=row["category"].strip(),
        color=row["color"].strip(),
        region=row["region"].strip(),
        grape=row["grape_type"].strip(),
        description=row["desc"].strip(),
        winery=row["winery"].strip(),
        image_name=name,
        reference_available=valid == "true",
    )


def load_cards(csv_path: Path) -> tuple[list[WineCard], int]:
    """Return unique cards in slug order, rejecting inconsistent duplicate rows."""
    cards: dict[str, WineCard] = {}
    rows = 0
    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing catalog columns: {sorted(missing)}")
        for row in reader:
            rows += 1
            card = _card_from_row(row)
            previous = cards.setdefault(card.slug, card)
            if previous != card:
                raise ValueError(f"Conflicting rows for slug {card.slug!r}")
    if not cards:
        raise ValueError("Catalog is empty")
    return [cards[slug] for slug in sorted(cards)], rows


def build_manifest(csv_path: Path, images_dir: Path) -> dict[str, Any]:
    """Hash every usable reference once; preserve shared files and missing cards."""
    cards, source_rows = load_cards(csv_path)
    grouped: dict[str, list[str]] = {}
    excluded: list[str] = []
    for card in cards:
        if not card.reference_available:
            excluded.append(card.slug)
            continue
        image = images_dir / card.image_name
        if not image.is_file():
            raise FileNotFoundError(f"Catalog marks {card.slug} valid, but {image} is missing")
        grouped.setdefault(card.image_name, []).append(card.slug)
    images = [
        {"filename": filename, "sha256": sha256_file(images_dir / filename),
         "slugs": sorted(slugs)}
        for filename, slugs in sorted(grouped.items())
    ]
    return {
        "schema_version": 1,
        "source_csv_sha256": sha256_file(csv_path),
        "source_rows": source_rows,
        "cards": [asdict(card) for card in cards],
        "images": images,
        "excluded_slugs": excluded,
        "counts": {
            "unique_cards": len(cards),
            "indexed_cards": sum(len(image["slugs"]) for image in images),
            "reference_images": len(images),
            "shared_reference_images": sum(len(image["slugs"]) > 1 for image in images),
            "excluded_cards": len(excluded),
        },
    }


def write_manifest(manifest: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    os.replace(temporary, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=ROOT / "df_2.csv")
    parser.add_argument("--images-dir", type=Path, default=ROOT / "data/competition_imgs")
    parser.add_argument("--output", type=Path, default=ROOT / "data/catalog_manifest.json")
    args = parser.parse_args()
    manifest = build_manifest(args.csv, args.images_dir)
    write_manifest(manifest, args.output)
    print(json.dumps({"manifest": str(args.output), "sha256": sha256_file(args.output),
                      **manifest["counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

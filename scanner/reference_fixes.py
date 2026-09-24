"""Build a separate, hash-checked gallery manifest with reviewed catalog repairs."""

from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from .catalog import ROOT, sha256_file, write_manifest

DEFAULT_BASELINE = ROOT / "data/catalog_manifest.json"
DEFAULT_CORRECTIONS = (
    ROOT / "evaluation/manifests/catalog_reference_fixes_2026-09-24.json"
)
DEFAULT_ARCHIVE = ROOT / "data/source/unpacked/prod-svoe-vino-strapi.part1.rar"
DEFAULT_IMAGES = ROOT / "data/competition_imgs"
DEFAULT_OUTPUT = ROOT / "data/catalog_manifest_corrected.json"
ARCHIVE_PREFIX = "prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads/"


def _safe_filename(filename: str) -> bool:
    return bool(
        filename
        and Path(filename).name == filename
        and "\\" not in filename
        and "\0" not in filename
    )


def _ensure_target(image_path: Path, item: dict, archive_path: Path) -> None:
    expected = item["target_sha256"]
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError(f"Invalid target SHA-256 for {item['slug']}")
    member = item["archive_member"]
    if (
        not member.startswith(ARCHIVE_PREFIX)
        or Path(member).name != item["target_image"]
        or ".." in Path(member).parts
    ):
        raise ValueError(f"Unsafe archive member for {item['slug']}")
    if not image_path.is_file():
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            include = root / "members.txt"
            include.write_text(member + "\n", encoding="utf-8")
            subprocess.run(
                [
                    "7z",
                    "e",
                    "-y",
                    "-scsUTF-8",
                    f"-o{root}",
                    f"-i@{include}",
                    str(archive_path),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
            )
            extracted = root / item["target_image"]
            if not extracted.is_file():
                raise FileNotFoundError(f"Archive member not extracted: {member}")
            if sha256_file(extracted) != expected:
                raise ValueError(f"Archive member SHA-256 mismatch: {member}")
            image_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(extracted, image_path)
    if sha256_file(image_path) != expected:
        raise ValueError(f"Target image SHA-256 mismatch: {image_path}")


def build_corrected_manifest(
    baseline_path: Path, corrections_path: Path, images_dir: Path, archive_path: Path
) -> dict:
    spec = json.loads(corrections_path.read_text(encoding="utf-8"))
    if sha256_file(baseline_path) != spec["baseline_manifest_sha256"]:
        raise ValueError("Baseline manifest differs from reviewed source")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if baseline.get("schema_version") != 1:
        raise ValueError("Unsupported baseline manifest schema")
    for record in baseline["images"]:
        if sha256_file(images_dir / record["filename"]) != record["sha256"]:
            raise ValueError(f"Baseline image changed: {record['filename']}")
    manifest = copy.deepcopy(baseline)
    cards = {card["slug"]: card for card in manifest["cards"]}
    seen: set[str] = set()
    for item in spec["fixes"]:
        slug = item["slug"]
        if slug in seen or slug not in cards:
            raise ValueError(f"Unknown or duplicate corrected slug: {slug}")
        seen.add(slug)
        card = cards[slug]
        if not card["reference_available"] or card["image_name"] != item["from_image"]:
            raise ValueError(f"Source reference differs for {slug}")
        if not _safe_filename(item["target_image"]):
            raise ValueError(f"Unsafe target filename for {slug}")
        _ensure_target(images_dir / item["target_image"], item, archive_path)
        card["image_name"] = item["target_image"]
    grouped: dict[str, list[str]] = {}
    for card in manifest["cards"]:
        if card["reference_available"]:
            grouped.setdefault(card["image_name"], []).append(card["slug"])
    manifest["images"] = [
        {
            "filename": filename,
            "sha256": sha256_file(images_dir / filename),
            "slugs": sorted(slugs),
        }
        for filename, slugs in sorted(grouped.items())
    ]
    manifest["source_manifest_sha256"] = spec["baseline_manifest_sha256"]
    manifest["reference_fixes_sha256"] = sha256_file(corrections_path)
    manifest["counts"] = {
        "unique_cards": len(cards),
        "indexed_cards": sum(map(len, grouped.values())),
        "reference_images": len(grouped),
        "shared_reference_images": sum(len(slugs) > 1 for slugs in grouped.values()),
        "excluded_cards": len(manifest["excluded_slugs"]),
    }
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--corrections", type=Path, default=DEFAULT_CORRECTIONS)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.resolve() == args.baseline.resolve():
        parser.error("Corrected manifest must not overwrite the source manifest")
    manifest = build_corrected_manifest(
        args.baseline, args.corrections, args.images_dir, args.archive
    )
    write_manifest(manifest, args.output)
    print(
        json.dumps(
            {
                "manifest": str(args.output),
                "sha256": sha256_file(args.output),
                **manifest["counts"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

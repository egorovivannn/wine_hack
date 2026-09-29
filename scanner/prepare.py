"""Recreate catalog references and query photos from the exact source archives."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from zipfile import ZipFile, ZipInfo
import zlib

from .catalog import ROOT, build_manifest, load_cards, sha256_file, write_manifest


SOURCE_HASHES = {
    "source/Датасет.zip": "c3fb7cabe06e4d9ef1e3719787c32d01446ec2c6f4fac86f87de7790fa9beeda",
    "source/official_real_photos.zip": "5ffa77f7c8fdc82e1f9aa123ded91cdb2df435b7e3990ce074cba53b6be0e4c9",
    "field/real_photos.zip": "f24f4c47a0aa43ba7a541963298e9a949516949bc428e3e21c1eb2101c40947b",
}
# Photo archives are only needed to evaluate on labelled photos; the service needs just Датасет.zip.
OPTIONAL_SOURCES = {"source/official_real_photos.zip", "field/real_photos.zip"}
EXPECTED_CSV_SHA256 = "7b219c23afbeba1e03c6297d0b8098ca8560bb6af53df13ebcbb7f46e6dbbb2b"
EXPECTED_MANIFEST_SHA256 = "5654d67ef71dfd3462aad1a00d949f5f5465a4c30f63667b95f388d68289a21d"
VOLUME_NAMES = [f"prod-svoe-vino-strapi.part{part}.rar" for part in (1, 2, 3)]


def _copy_member(archive: ZipFile, info: ZipInfo, destination: Path,
                 verify_only: bool = False) -> None:
    if destination.is_file() and destination.stat().st_size == info.file_size:
        crc = 0
        with destination.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                crc = zlib.crc32(block, crc)
        if crc == info.CRC:
            return
    if verify_only:
        raise ValueError(f"Extracted file missing or changed: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with archive.open(info) as source, temporary.open("wb") as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)
    os.replace(temporary, destination)


def _extract_photos(archive_path: Path, destination: Path, prefix: str,
                    expected_count: int, verify_only: bool = False) -> None:
    count = 0
    with ZipFile(archive_path) as archive:
        for info in archive.infolist():
            name = info.filename
            if info.is_dir() or name.startswith("__MACOSX/"):
                continue
            if not name.startswith(prefix):
                continue
            filename = name[len(prefix):]
            if not filename or Path(filename).name != filename:
                raise ValueError(f"Unsafe query photo path: {name}")
            if Path(filename).suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            _copy_member(archive, info, destination / filename, verify_only)
            count += 1
    if count != expected_count:
        raise ValueError(f"Expected {expected_count} photos in {archive_path}, found {count}")


def _rar_paths(first_volume: Path) -> dict[str, str]:
    result = subprocess.run(["7z", "l", "-slt", str(first_volume)],
                            text=True, capture_output=True, check=True)
    entries = result.stdout.split("----------", 1)[1]
    mapping: dict[str, str] = {}
    for line in entries.splitlines():
        if not line.startswith("Path = "):
            continue
        archive_path = line.removeprefix("Path = ")
        filename = archive_path.rsplit("/", 1)[-1]
        if filename in mapping:
            raise ValueError(f"RAR contains duplicate basename: {filename}")
        mapping[filename] = archive_path
    return mapping


def _reference_files_match(manifest_path: Path, images_dir: Path,
                           filenames: set[str]) -> bool:
    if not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if {item["filename"] for item in manifest["images"]} != filenames:
        return False
    return all((images_dir / item["filename"]).is_file() and
               sha256_file(images_dir / item["filename"]) == item["sha256"]
               for item in manifest["images"])


def prepare(data_dir: Path, csv_path: Path, verify_only: bool = False) -> dict:
    data_dir = data_dir.resolve()
    if sha256_file(csv_path) != EXPECTED_CSV_SHA256:
        raise ValueError("df_2.csv differs from the pinned catalog")
    for relative, expected in SOURCE_HASHES.items():
        path = data_dir / relative
        if relative in OPTIONAL_SOURCES and not path.exists():
            print(f"Skipping optional {relative}: not found (only needed for evaluation on photos)")
            continue
        if sha256_file(path) != expected:
            raise ValueError(f"Source archive hash differs: {path}")
    images_dir = data_dir / "competition_imgs"
    manifest_path = data_dir / "catalog_manifest.json"
    cards, _ = load_cards(csv_path)
    filenames = {card.image_name for card in cards if card.reference_available}
    references_valid = _reference_files_match(manifest_path, images_dir, filenames)
    if not verify_only:
        unpacked = data_dir / "source/unpacked"
        with ZipFile(data_dir / "source/Датасет.zip") as archive:
            for name in VOLUME_NAMES + ["eval.zip"]:
                info = archive.getinfo(f"Датасет/{name}")
                _copy_member(archive, info, unpacked / name)
        if not references_valid:
            mapping = _rar_paths(unpacked / VOLUME_NAMES[0])
            missing = filenames - mapping.keys()
            if missing:
                raise FileNotFoundError(f"RAR lacks {len(missing)} catalog references: {sorted(missing)[:3]}")
            images_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".txt") as path_list:
                path_list.write("\n".join(mapping[name] for name in sorted(filenames)) + "\n")
                path_list.flush()
                subprocess.run(["7z", "e", "-y", "-scsUTF-8", f"-o{images_dir}",
                                f"-i@{path_list.name}", str(unpacked / VOLUME_NAMES[0])],
                               check=True, stdout=subprocess.DEVNULL)
            write_manifest(build_manifest(csv_path, images_dir), manifest_path)
        with ZipFile(unpacked / "eval.zip") as archive:
            for info in archive.infolist():
                if info.is_dir() or info.filename.startswith("__MACOSX/"):
                    continue
                name = info.filename
                if Path(name).is_absolute() or ".." in Path(name).parts:
                    raise ValueError(f"Unsafe evaluator file path: {name}")
                _copy_member(archive, info, data_dir / "source/eval" / name)
    for archive, destination, prefix, count in (
            ("source/official_real_photos.zip", "official_real_photos", "", 100),
            ("field/real_photos.zip", "field/photos", "real_test/", 13)):
        if (data_dir / archive).exists():
            _extract_photos(data_dir / archive, data_dir / destination, prefix=prefix,
                            expected_count=count, verify_only=verify_only)
    if not _reference_files_match(manifest_path, images_dir, filenames):
        raise ValueError("Catalog references do not match their manifest")
    digest = sha256_file(manifest_path)
    if digest != EXPECTED_MANIFEST_SHA256:
        raise ValueError(f"Catalog manifest differs from pinned version: {digest}")
    return {"reference_images": len(filenames), "manifest_sha256": digest,
            "verified_only": verify_only}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--csv", type=Path, default=ROOT / "df_2.csv")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args.data_dir, args.csv, args.verify_only), ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Download verified shelf-to-product pairs from a pinned Norwegian Grocery revision.

The published COCO annotations have category IDs and names, but no product_code.
Only unique, exact category-name to reference-metadata joins are accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests

from .clean_winesensed import Groups, near_pairs, perceptual_hash

REPO = "valiantlynxz/norwegian-grocery"
API = f"https://huggingface.co/api/datasets/{REPO}"
DEFAULT_REVISION = "17cacfcb3ed0a0a24e757fa92c5c546cb3a19f57"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_split(key: str) -> str:
    value = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) / 2**32
    return "test" if value < 0.15 else "val" if value < 0.30 else "train"


def exact_joins(categories: list[dict], products: list[dict]) -> dict[int, dict]:
    by_name = defaultdict(list)
    for product in products:
        if product.get("has_images") and product.get("product_name"):
            by_name[product["product_name"]].append(product)
    return {
        category["id"]: by_name[category["name"]][0]
        for category in categories
        if len(by_name[category["name"]]) == 1
    }


def fetch_file(
    revision: str, remote_path: str, root: Path, size: int | None = None
) -> dict:
    path = root / remote_path
    if path.exists():
        if size is not None and path.stat().st_size != size:
            raise ValueError(f"Existing file size differs: {path}")
        return {
            "path": remote_path,
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/{remote_path}"
    for attempt in range(5):
        try:
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            content = response.content
            if size is not None and len(content) != size:
                raise ValueError(f"Downloaded size differs: {remote_path}")
            temporary = path.with_suffix(path.suffix + ".part")
            temporary.write_bytes(content)
            temporary.replace(path)
            return {"path": remote_path, "sha256": sha256(path), "bytes": len(content)}
        except (requests.RequestException, ValueError):
            if attempt == 4:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/external/norwegian"))
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    revision = args.revision
    essentials = ["train/annotations.json", "NM_NGD_product_images/metadata.json"]
    records = [fetch_file(revision, path, args.output) for path in essentials]
    annotations = json.loads((args.output / essentials[0]).read_text())
    metadata = json.loads((args.output / essentials[1]).read_text())
    joins = exact_joins(annotations["categories"], metadata["products"])
    scene_listing = requests.get(
        f"{API}/tree/{revision}/train/images", timeout=30
    ).json()
    scene_sizes = {
        item["path"]: item.get("size")
        for item in scene_listing
        if item["type"] == "file"
    }
    # The supplied numeric image filenames determine only conservative capture blocks,
    # never product identity. Entire ten-image blocks stay in one split.
    images = {item["id"]: item for item in annotations["images"]}
    product_info = {product["product_code"]: product for product in joins.values()}
    product_paths = [
        f"NM_NGD_product_images/{code}/{kind}.jpg"
        for code, product in sorted(product_info.items())
        for kind in ("front", "main")
        if kind in product["image_types"]
    ]
    files = [(path, scene_sizes[path]) for path in sorted(scene_sizes)] + [
        (path, None) for path in product_paths
    ]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(fetch_file, revision, path, args.output, size)
            for path, size in files
        ]
        for number, future in enumerate(as_completed(futures), 1):
            records.append(future.result())
            if number % 50 == 0:
                print(f"Downloaded/verified {number}/{len(files)} files", flush=True)
    digest_by_path = {row["path"]: row["sha256"] for row in records}
    inventory = args.output / "files.json"
    file_records = sorted(records, key=lambda row: row["path"])
    if inventory.exists() and json.loads(inventory.read_text()) != file_records:
        raise ValueError("Existing file inventory differs from downloaded files")
    inventory.write_text(json.dumps(file_records, indent=2) + "\n")
    reference_groups = Groups(sorted(product_info))
    reference_hashes = []
    reference_codes = []
    for code, product in sorted(product_info.items()):
        reference = next(
            (
                f"NM_NGD_product_images/{code}/{kind}.jpg"
                for kind in ("front", "main")
                if kind in product["image_types"]
            ),
            None,
        )
        if reference and (args.output / reference).exists():
            reference_codes.append(code)
            reference_hashes.append(perceptual_hash(args.output / reference))
    near_duplicate_pairs = 0
    for a, b, _ in near_pairs(np.asarray(reference_hashes, dtype=np.uint64)):
        reference_groups.union(reference_codes[a], reference_codes[b])
        near_duplicate_pairs += 1
    product_splits = {
        code: stable_split("product-" + reference_groups.find(code))
        for code in product_info
    }
    rows = []
    for annotation in sorted(annotations["annotations"], key=lambda a: a["id"]):
        product = joins.get(annotation["category_id"])
        if product is None or annotation.get("iscrowd"):
            continue
        image = images[annotation["image_id"]]
        scene = f"train/images/{image['file_name']}"
        if scene not in digest_by_path:
            raise ValueError(f"Missing scene {scene}")
        code = product["product_code"]
        reference = next(
            (
                f"NM_NGD_product_images/{code}/{kind}.jpg"
                for kind in ("front", "main")
                if kind in product["image_types"]
            ),
            None,
        )
        if reference is None or reference not in digest_by_path:
            continue
        scene_number = int(image["file_name"].split("_")[-1].split(".")[0])
        scene_group = f"scene-block-{scene_number // 10}"
        scene_split = stable_split(scene_group)
        product_split = product_splits[code]
        if scene_split != product_split:
            continue
        x, y, width, height = annotation["bbox"]
        if (
            min(width, height) < 12
            or x < 0
            or y < 0
            or x + width > image["width"]
            or y + height > image["height"]
        ):
            continue
        rows.append(
            dict(
                annotation_id=annotation["id"],
                scene=scene,
                scene_sha256=digest_by_path[scene],
                image_id=image["id"],
                scene_group=scene_group,
                split=scene_split,
                product_code=code,
                product_name=product["product_name"],
                reference=reference,
                reference_sha256=digest_by_path[reference],
                bbox=[x, y, width, height],
                category_id=annotation["category_id"],
            )
        )
    if not rows or set(row["split"] for row in rows) != {"train", "val", "test"}:
        raise ValueError("No usable pairs in every split")
    manifest = args.output / "pairs.jsonl"
    manifest.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        )
    )
    (args.output / "product_splits.json").write_text(
        json.dumps(product_splits, indent=2, sort_keys=True) + "\n"
    )
    report = dict(
        source=f"https://huggingface.co/datasets/{REPO}",
        revision=revision,
        license="cc-by-nc-4.0",
        annotations_sha256=digest_by_path[essentials[0]],
        metadata_sha256=digest_by_path[essentials[1]],
        manifest_sha256=sha256(manifest),
        files_sha256=sha256(inventory),
        product_splits_sha256=sha256(args.output / "product_splits.json"),
        exact_category_joins=len(joins),
        pairs=len(rows),
        near_duplicate_reference_pairs=near_duplicate_pairs,
        split_pairs=dict(Counter(row["split"] for row in rows)),
        split_products={
            s: len({r["product_code"] for r in rows if r["split"] == s})
            for s in ("train", "val", "test")
        },
        split_scenes={
            s: len({r["scene"] for r in rows if r["split"] == s})
            for s in ("train", "val", "test")
        },
        split_protocol="SHA256 product_code groups connected by reference pHash Hamming<=5, and ten-consecutive-image scene blocks; keep only equal split pairs. No source capture-session metadata.",
    )
    (args.output / "provenance.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

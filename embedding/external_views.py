"""Build reproducible query/reference views for the external retrieval experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from ultralytics import YOLO

from scanner.image import decode_image, query_views, reference_views

from .prepare_norwegian import exact_joins


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def detector_crop(detector: YOLO, image: Image.Image) -> tuple[Image.Image, int]:
    """Same central-label selector, confidence and fallback as the earlier cascade."""
    result = detector.predict(image, imgsz=640, conf=0.25, device=0, verbose=False)[0]
    boxes = result.boxes.xyxy.cpu().numpy()
    if not len(boxes):
        return query_views(image)[1], 0
    width, height = image.size
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    distance = ((centers / np.array([width, height]) - 0.5) ** 2).sum(axis=1)
    area = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1) / (width * height)
    selected = int(np.argmin(distance - 0.05 * np.sqrt(area)))
    x1, y1, x2, y2 = boxes[selected]
    margin = 0.04 * max(x2 - x1, y2 - y1)
    crop = image.crop(
        (
            max(0, int(x1 - margin)),
            max(0, int(y1 - margin)),
            min(width, int(x2 + margin)),
            min(height, int(y2 + margin)),
        )
    )
    return ImageOps.pad(
        crop, (384, 384), method=Image.Resampling.BICUBIC, color="white"
    ), len(boxes)


def box_crop(image: Image.Image, bbox: list[float]) -> Image.Image:
    x, y, width, height = bbox
    margin = 0.04 * max(width, height)
    crop = image.crop(
        (
            max(0, int(x - margin)),
            max(0, int(y - margin)),
            min(image.width, int(x + width + margin)),
            min(image.height, int(y + height + margin)),
        )
    )
    return ImageOps.pad(
        crop, (384, 384), method=Image.Resampling.BICUBIC, color="white"
    )


def write_view(view: Image.Image, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        view.save(path, quality=94, subsampling=0)
    return sha256(path)


def wine_views(manifest: Path, output: Path, checkpoint: Path) -> list[dict]:
    frame = pd.read_csv(manifest, dtype=str, keep_default_na=False).sort_values(
        ["vintage_id", "filename"]
    )
    detector = YOLO(str(checkpoint))
    rows = []
    for number, (vintage, members) in enumerate(frame.groupby("vintage_id"), 1):
        for role, records in (
            ("reference", members.iloc[:1]),
            ("query", members.iloc[1:]),
        ):
            for record in records.itertuples():
                source = Path(record.path)
                if sha256(source) != record.sha256:
                    raise ValueError(f"Source image changed: {record.filename}")
                image = decode_image(source)
                if role == "reference":
                    view = reference_views(image)[1]
                    detections = None
                else:
                    view, detections = detector_crop(detector, image)
                relative = f"wine/{role}/{Path(record.filename).stem}.jpg"
                digest = write_view(view, output / relative)
                rows.append(
                    dict(
                        source="winesensed",
                        role=role,
                        identity=str(vintage),
                        winery_id=record.winery_id,
                        split=record.split,
                        source_sha256=record.sha256,
                        view=relative,
                        view_sha256=digest,
                        crop_type="reference_fraction"
                        if role == "reference"
                        else "detector_center_label",
                        detector_boxes=detections,
                    )
                )
        if number % 100 == 0:
            print(
                f"Prepared {number}/{frame.vintage_id.nunique()} WineSensed vintages",
                flush=True,
            )
    return rows


def norwegian_views(source_root: Path, output: Path) -> list[dict]:
    pairs_path = source_root / "pairs.jsonl"
    pairs = [json.loads(line) for line in pairs_path.read_text().splitlines()]
    chosen = {}
    for row in pairs:
        key = row["scene"], row["product_code"]
        # One real package per product and source scene. Prefer the largest legible box.
        if key not in chosen or np.prod(row["bbox"][2:]) > np.prod(
            chosen[key]["bbox"][2:]
        ):
            chosen[key] = row
    rows = []
    annotations = json.loads((source_root / "train/annotations.json").read_text())
    metadata = json.loads(
        (source_root / "NM_NGD_product_images/metadata.json").read_text()
    )
    joins = exact_joins(annotations["categories"], metadata["products"])
    product_splits = json.loads((source_root / "product_splits.json").read_text())
    references = {product["product_code"]: product for product in joins.values()}
    for code, product in sorted(references.items()):
        reference = next(
            (
                f"NM_NGD_product_images/{code}/{kind}.jpg"
                for kind in ("front", "main")
                if kind in product["image_types"]
            ),
            None,
        )
        if reference is None:
            continue
        path = source_root / reference
        if not path.exists():
            continue
        if code in {r["product_code"] for r in chosen.values()} and sha256(
            path
        ) != next(
            r["reference_sha256"] for r in chosen.values() if r["product_code"] == code
        ):
            raise ValueError(f"Reference changed: {path}")
        view = reference_views(decode_image(path))[0]
        relative = f"norwegian/reference/{code}.jpg"
        digest = write_view(view, output / relative)
        rows.append(
            dict(
                source="norwegian",
                role="reference",
                identity=code,
                product_name=product["product_name"],
                split=product_splits[code],
                source_sha256=sha256(path),
                view=relative,
                view_sha256=digest,
                crop_type="product_reference_full",
                scene_group=None,
            )
        )
    previous_scene = None
    scene_image = None
    for number, ((scene, code), row) in enumerate(sorted(chosen.items()), 1):
        path = source_root / scene
        if scene != previous_scene:
            if sha256(path) != row["scene_sha256"]:
                raise ValueError(f"Scene changed: {path}")
            scene_image = decode_image(path)
            previous_scene = scene
        view = box_crop(scene_image, row["bbox"])
        relative = f"norwegian/query/{row['annotation_id']}.jpg"
        digest = write_view(view, output / relative)
        rows.append(
            dict(
                source="norwegian",
                role="query",
                identity=code,
                product_name=row["product_name"],
                split=row["split"],
                source_sha256=row["scene_sha256"],
                view=relative,
                view_sha256=digest,
                crop_type="ground_truth_product_box",
                scene_group=row["scene_group"],
                annotation_id=row["annotation_id"],
            )
        )
        if number % 200 == 0:
            print(
                f"Prepared {number}/{len(chosen)} Norwegian shelf queries", flush=True
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wine-manifest", type=Path)
    parser.add_argument("--norwegian-root", type=Path)
    parser.add_argument("--detector", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.wine_manifest and not args.norwegian_root:
        parser.error("At least one external dataset is required")
    if args.wine_manifest and not args.detector:
        parser.error("WineSensed queries require --detector")
    if args.output.exists():
        parser.error("Output exists")
    args.output.mkdir(parents=True)
    wine = (
        wine_views(args.wine_manifest, args.output, args.detector)
        if args.wine_manifest
        else []
    )
    norwegian = (
        norwegian_views(args.norwegian_root, args.output) if args.norwegian_root else []
    )
    manifest = args.output / "views.jsonl"
    manifest.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in wine + norwegian
        )
    )
    report = dict(
        wine_manifest_sha256=sha256(args.wine_manifest) if args.wine_manifest else None,
        norwegian_pairs_sha256=sha256(args.norwegian_root / "pairs.jsonl")
        if args.norwegian_root
        else None,
        detector_sha256=sha256(args.detector) if args.detector else None,
        manifest_sha256=sha256(manifest),
        views=len(wine) + len(norwegian),
        counts=dict(
            Counter(f"{r['source']}_{r['role']}_{r['split']}" for r in wine + norwegian)
        ),
        wine_detector_no_box=sum(
            r.get("detector_boxes") == 0 for r in wine if r["role"] == "query"
        ),
        view_policy="Wine queries: central YOLO label or center fallback. Wine references: fixed label fraction. Norwegian queries: verified GT product box; references: full product photo.",
    )
    (args.output / "provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

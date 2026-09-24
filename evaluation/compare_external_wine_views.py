"""Compare full, central and detected query views on external wine validation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from embedding.train_external_adapter import load_data, permitted_gallery_splits, sha256
from scanner.catalog import sha256_file
from scanner.image import decode_image, query_views
from scanner.vision import DEFAULT_MODEL_DIR, VisionEncoder


def metrics(scores: np.ndarray, truth: list[str], gallery: list[str]) -> dict:
    order = np.argsort(-scores, axis=1, kind="stable")
    ranked = np.asarray(gallery, dtype=object)[order]
    correct = ranked == np.asarray(truth, dtype=object)[:, None]
    return {
        "top1": int(correct[:, 0].sum()),
        "hit5": int(correct[:, :5].any(axis=1).sum()),
        "recall20": int(correct[:, :20].any(axis=1).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--views", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--wine-manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists")
    rows, vectors = load_data(args.views, args.features)
    query_ids = [
        i
        for i, row in enumerate(rows)
        if row["source"] == "winesensed"
        and row["role"] == "query"
        and row["split"] == args.split
    ]
    gallery_ids = [
        i
        for i, row in enumerate(rows)
        if row["source"] == "winesensed"
        and row["role"] == "reference"
        and row["split"] in permitted_gallery_splits(args.split)
    ]
    with args.wine_manifest.open(newline="", encoding="utf-8") as stream:
        source_by_sha = {
            row["sha256"]: Path(row["path"]) for row in csv.DictReader(stream)
        }
    query_views_array = []
    encoder = VisionEncoder(args.model_dir)
    for start in range(0, len(query_ids), args.batch_size // 2):
        batch = query_ids[start : start + args.batch_size // 2]
        images = []
        for index in batch:
            expected = rows[index]["source_sha256"]
            path = source_by_sha[expected]
            if sha256_file(path) != expected:
                raise ValueError(f"Wine source changed: {path}")
            images.extend(query_views(decode_image(path)))
        query_views_array.extend(
            encoder.encode(images, batch_size=args.batch_size).reshape(
                len(batch), 2, -1
            )
        )
        if (start + len(batch)) % 100 < len(batch):
            print(f"Encoded {start + len(batch)}/{len(query_ids)} queries", flush=True)
    query_vectors = np.stack(query_views_array)
    gallery_vectors = vectors[gallery_ids]
    detector_scores = vectors[query_ids] @ gallery_vectors.T
    full_scores = query_vectors[:, 0] @ gallery_vectors.T
    center_scores = query_vectors[:, 1] @ gallery_vectors.T
    combined_scores = 0.35 * full_scores + 0.65 * center_scores
    truth = [rows[i]["identity"] for i in query_ids]
    gallery = [rows[i]["identity"] for i in gallery_ids]
    measures = {
        name: metrics(scores, truth, gallery)
        for name, scores in (
            ("detector_crop", detector_scores),
            ("full_frame", full_scores),
            ("central_crop", center_scores),
            ("full_center_blend", combined_scores),
        )
    }
    detector_order = np.argsort(-detector_scores, axis=1, kind="stable")[:, :20]
    blend_order = np.argsort(-combined_scores, axis=1, kind="stable")[:, :20]
    union_hits = sum(
        identity in {gallery[j] for j in detector_order[i]}
        or identity in {gallery[j] for j in blend_order[i]}
        for i, identity in enumerate(truth)
    )
    report = {
        "split": args.split,
        "queries": len(query_ids),
        "gallery": len(gallery_ids),
        "measures": measures,
        "detector_plus_blend_union_recall40": union_hits,
        "views_sha256": sha256(args.views),
        "features_sha256": sha256(args.features),
        "wine_manifest_sha256": sha256(args.wine_manifest),
        "model_weights_sha256": sha256(args.model_dir / "model.safetensors"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

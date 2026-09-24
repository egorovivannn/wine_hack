"""Measure single-query latency on held-out external real photographs."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

from embedding.external_views import detector_crop
from embedding.train_external_adapter import load_data, sha256
from embedding.train_external_reranker import lexical_score, tokens
from embedding.train_label_head import LabelHead
from scanner.image import decode_image
from scanner.ocr import OCRReader
from scanner.vision import DEFAULT_MODEL_DIR, VisionEncoder


def summary(values: list[float]) -> dict:
    return dict(
        count=len(values),
        p50_ms=float(np.percentile(values, 50)),
        p95_ms=float(np.percentile(values, 95)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--views", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--wine-manifest", type=Path, required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--reranker", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-queries", type=int, default=100)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists")
    rows, vectors = load_data(args.views, args.features)
    state = torch.load(args.head, map_location="cpu", weights_only=False)
    head = LabelHead(**state["config"])
    head.load_state_dict(state["state_dict"])
    head.eval()
    rerank = torch.load(args.reranker, map_location="cpu", weights_only=False)
    encoder = VisionEncoder(args.model_dir)
    detector = YOLO(str(args.detector))
    reader = OCRReader()
    wine_sources = (
        pd.read_csv(args.wine_manifest, dtype=str).set_index("sha256").path.to_dict()
    )
    source_results = {}
    for source in ("winesensed", "norwegian"):
        gallery = np.array(
            [
                i
                for i, row in enumerate(rows)
                if row["source"] == source and row["role"] == "reference"
            ]
        )
        queries = [
            row
            for row in rows
            if row["source"] == source
            and row["role"] == "query"
            and row["split"] == "test"
        ][: args.max_queries]
        gallery_vectors = vectors[gallery]
        with torch.no_grad():
            head_gallery = head(torch.from_numpy(gallery_vectors)).numpy()
        names = [rows[i].get("product_name", "") for i in gallery]
        frequency = Counter(word for name in names for word in tokens(name))
        total = []
        stages = {"detector": [], "encoder": [], "ocr": [], "ranking": []}
        for row in queries:
            begin = time.perf_counter()
            if source == "winesensed":
                path = Path(wine_sources[row["source_sha256"]])
                if sha256(path) != row["source_sha256"]:
                    raise ValueError(f"Wine source changed: {path}")
                view, _ = detector_crop(detector, decode_image(path))
            else:
                path = args.views.parent / row["view"]
                view = decode_image(path)
            after_crop = time.perf_counter()
            query = encoder.encode([view], batch_size=1)[0]
            after_encode = time.perf_counter()
            base = gallery_vectors @ query
            if source == "norwegian":
                lines = [
                    dict(text=line.text, confidence=line.confidence)
                    for line in reader.read(view)
                ]
                after_ocr = time.perf_counter()
                with torch.no_grad():
                    head_query = head(torch.from_numpy(query)).numpy()
                all_adapted = head_gallery @ head_query
                top = np.argsort(-all_adapted, kind="stable")[:20]
                adapted = all_adapted[top]
                values = np.array(
                    [
                        [
                            base[j],
                            adapted[pos],
                            base[j] - base[top].max(),
                            adapted[pos] - adapted.max(),
                            pos / 19.0,
                            lexical_score(lines, names[j], frequency, len(gallery)),
                        ]
                        for pos, j in enumerate(top)
                    ],
                    dtype=np.float32,
                )
                correction = ((values - rerank["mean"]) / rerank["std"]) @ rerank[
                    "weights"
                ]
                _ = np.argmax(values[:, 1] + rerank["scale"] * correction)
            else:
                after_ocr = after_encode
                top = np.argsort(-base, kind="stable")[:20]
                _ = top[0]
            end = time.perf_counter()
            total.append((end - begin) * 1000)
            stages["detector"].append((after_crop - begin) * 1000)
            stages["encoder"].append((after_encode - after_crop) * 1000)
            stages["ocr"].append((after_ocr - after_encode) * 1000)
            stages["ranking"].append((end - after_ocr) * 1000)
        source_results[source] = dict(
            total=summary(total),
            stages={name: summary(values) for name, values in stages.items()},
            localization="YOLO central label"
            if source == "winesensed"
            else "oracle ground-truth box, crop preparation excluded",
        )
    report = dict(
        results=source_results,
        views_sha256=sha256(args.views),
        features_sha256=sha256(args.features),
        wine_manifest_sha256=sha256(args.wine_manifest),
        detector_sha256=sha256(args.detector),
        base_sha256=sha256(args.model_dir / "model.safetensors"),
        head_sha256=sha256(args.head),
        reranker_sha256=sha256(args.reranker),
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        method="Sequential warmed-process measurements; first query includes any remaining model warm-up; OCR on all Norwegian queries.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

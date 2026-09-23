"""Record reproducible predictions and score only independently verified labels."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch
import transformers

from .catalog import ROOT, sha256_file
from .labels import read_labels
from .server import SearchEngine
from .ocr import MODEL_DIR as DEFAULT_OCR_MODEL_DIR, MODEL_SHA256 as OCR_MODEL_SHA256
from .vision import DEFAULT_INDEX, DEFAULT_MANIFEST, DEFAULT_MODEL_DIR


def evaluate(images_dir: Path, output: Path, labels_path: Path | None,
             engine: SearchEngine, index_path: Path, manifest_path: Path) -> dict:
    paths = sorted(path for path in images_dir.iterdir()
                   if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    if not paths:
        raise ValueError(f"No images in {images_dir}")
    labels = read_labels(labels_path)
    if set(labels) - {path.name for path in paths}:
        raise ValueError("Some labels have no corresponding image")
    for filename, label in labels.items():
        if label.status == "verified" and label.slug not in engine.cards:
            raise ValueError(f"Unknown label slug for {filename}: {label.slug}")
    digests = {path.name: sha256_file(path) for path in paths}
    for filename, label in labels.items():
        if digests[filename] != label.image_sha256:
            raise ValueError(f"Image changed after labeling: {filename}")
    output.parent.mkdir(parents=True, exist_ok=True)
    correct_top1 = correct_top5 = 0
    latencies = []
    ocr_invocations = 0
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for count, path in enumerate(paths, 1):
            started = time.perf_counter()
            prediction = engine.predict(path.read_bytes())
            ocr_invocations += prediction["ocr_used"]
            latency_ms = round((time.perf_counter() - started) * 1000)
            latencies.append(latency_ms)
            slugs = [prediction["slug"]] + [item["slug"] for item in prediction["alternatives"]]
            row = {
                "image_path": path.name,
                "image_sha256": digests[path.name],
                "predicted_slug": slugs[0],
                "top5_slugs": slugs[:5],
                "visual_similarity": round(prediction["visual_similarity"], 6),
                "margin_to_second": prediction["margin_to_second"],
                "latency_ms": latency_ms,
                "ocr_used": prediction["ocr_used"],
            }
            if path.name in labels:
                label = labels[path.name]
                row["label_status"] = label.status
                if label.status == "verified":
                    row["verified_slug"] = label.slug
                    correct_top1 += slugs[0] == label.slug
                    correct_top5 += label.slug in slugs[:5]
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            if count % 10 == 0 or count == len(paths):
                print(f"Predicted {count}/{len(paths)} images", flush=True)
    os.replace(temporary, output)
    verified_count = sum(label.status == "verified" for label in labels.values())
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              text=True, capture_output=True, check=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                                text=True, capture_output=True, check=True).stdout.strip())
    summary = {
        "images": len(paths),
        "verified_labels": verified_count,
        "unknown_labels": sum(label.status == "unknown" for label in labels.values()),
        "ambiguous_labels": sum(label.status == "ambiguous" for label in labels.values()),
        "ocr_invocations": ocr_invocations,
        "top1_correct": correct_top1,
        "top5_correct": correct_top5,
        "top1_accuracy": correct_top1 / verified_count if verified_count else None,
        "top5_accuracy": correct_top5 / verified_count if verified_count else None,
        "p50_latency_ms": round(float(np.percentile(latencies, 50))),
        "p95_latency_ms": round(float(np.percentile(latencies, 95))),
        "mean_latency_ms": round(sum(latencies) / len(latencies)),
        "index_sha256": sha256_file(index_path),
        "manifest_sha256": sha256_file(manifest_path),
        "model_weights_sha256": sha256_file(engine.encoder.model_dir / "model.safetensors"),
        "ocr_enabled": engine.ocr_reader is not None,
        "ocr_weights_sha256": {name: sha256_file(engine.ocr_model_dir / name) for name in OCR_MODEL_SHA256}
        if engine.ocr_reader is not None else None,
        "labels_sha256": sha256_file(labels_path) if labels_path else None,
        "predictions_sha256": sha256_file(output),
        "git_revision": revision,
        "git_dirty": dirty,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "device": str(engine.encoder.device),
        "gpu": torch.cuda.get_device_name(engine.encoder.device) if engine.encoder.device.type == "cuda" else None,
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--catalog-images-dir", type=Path, default=ROOT / "data/competition_imgs")
    parser.add_argument("--ocr-model-dir", type=Path, default=DEFAULT_OCR_MODEL_DIR)
    parser.add_argument("--use-ocr", action="store_true")
    args = parser.parse_args()
    engine = SearchEngine(args.manifest, args.index, args.model_dir, args.catalog_images_dir,
                          use_ocr=args.use_ocr, ocr_model_dir=args.ocr_model_dir)
    print(json.dumps(evaluate(args.images_dir, args.output, args.labels, engine,
                              args.index, args.manifest), ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Measure wine-label localization and central-box selection on GRAIN test."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from ultralytics import YOLO


def iou(box: np.ndarray, target: np.ndarray) -> float:
    intersection = np.maximum(
        0, np.minimum(box[2:], target[2:]) - np.maximum(box[:2], target[:2])
    ).prod()
    union = (
        np.maximum(0, box[2:] - box[:2]).prod()
        + np.maximum(0, target[2:] - target[:2]).prod()
        - intersection
    )
    return float(intersection / union) if union else 0.0


def choose_center(boxes: np.ndarray, width: int, height: int) -> int:
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    distance = ((centers / np.array([width, height]) - 0.5) ** 2).sum(axis=1)
    area = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1) / (width * height)
    return int(np.argmin(distance - 0.05 * np.sqrt(area)))


def target_boxes(label_file: Path, width: int, height: int) -> np.ndarray:
    rows = []
    for line in label_file.read_text().splitlines():
        cls, cx, cy, bw, bh = map(float, line.split())
        if cls != 0:
            continue
        rows.append(
            [
                (cx - bw / 2) * width,
                (cy - bh / 2) * height,
                (cx + bw / 2) * width,
                (cy + bh / 2) * height,
            ]
        )
    return np.asarray(rows, dtype=np.float32).reshape(-1, 4)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = yaml.safe_load(args.data.read_text())
    root = Path(data["path"])
    paths = sorted((root / data["test"]).glob("*.jpg"))
    model = YOLO(str(args.checkpoint))
    # Warm-up is excluded from steady-state latency.
    model.predict(str(paths[0]), imgsz=640, conf=0.25, device=0, verbose=False)
    samples = []
    for image_path in paths:
        begin = time.perf_counter()
        result = model.predict(
            str(image_path), imgsz=640, conf=0.25, device=0, verbose=False
        )[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        height, width = result.orig_shape
        truth = target_boxes(
            root / "labels/test" / (image_path.stem + ".txt"), width, height
        )
        if not len(truth):
            continue
        selected = choose_center(boxes, width, height) if len(boxes) else -1
        target = choose_center(truth, width, height)
        samples.append(
            dict(
                image=image_path.name,
                annotations=len(truth),
                predictions=len(boxes),
                any_match=any(iou(box, gt) >= 0.5 for box in boxes for gt in truth),
                central_match=selected >= 0
                and iou(boxes[selected], truth[target]) >= 0.5,
                central_iou=iou(boxes[selected], truth[target])
                if selected >= 0
                else 0.0,
                latency_ms=(time.perf_counter() - begin) * 1000,
            )
        )
    latencies = np.array([r["latency_ms"] for r in samples])
    report = dict(
        population=len(samples),
        annotated_boxes=sum(r["annotations"] for r in samples),
        any_box_success=sum(r["any_match"] for r in samples),
        central_box_success=sum(r["central_match"] for r in samples),
        central_box_success_rate=float(np.mean([r["central_match"] for r in samples])),
        no_prediction=sum(r["predictions"] == 0 for r in samples),
        p50_ms=float(np.percentile(latencies, 50)),
        p95_ms=float(np.percentile(latencies, 95)),
        checkpoint_sha256=sha256(args.checkpoint),
        data_manifest_sha256=sha256(root / "manifest.jsonl"),
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        git_revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        python=platform.python_version(),
        torch=torch.__version__,
        definition="IoU>=0.5 to annotation nearest image center under same selector; not bottle identity",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    args.output.with_suffix(".errors.json").write_text(
        json.dumps([r for r in samples if not r["central_match"]], indent=2) + "\n"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

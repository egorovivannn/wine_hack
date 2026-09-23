"""Compare the trained detector, embedding head and reranker on reviewed photos."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import platform
import subprocess
import time

import cv2
import numpy as np
from PIL import Image, ImageOps
import torch
from ultralytics import YOLO

from embedding.train_label_head import LabelHead, sha256
from embedding.train_top20_reranker import orb_features
from scanner.catalog import ROOT
from scanner.image import decode_image, query_views, reference_views
from scanner.labels import read_labels
from scanner.vision import DEFAULT_INDEX, DEFAULT_MANIFEST, DEFAULT_MODEL_DIR, VisionEncoder, load_index


def orb_description(orb, image):
    gray = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    return orb.detectAndCompute(gray, None)


def crop_label(detector, image):
    result = detector.predict(image, imgsz=640, conf=.25, device=0, verbose=False)[0]
    boxes = result.boxes.xyxy.cpu().numpy()
    if not len(boxes):
        return query_views(image)[1], 0
    w, h = image.size
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    distance = ((centers / np.array([w, h]) - .5)**2).sum(axis=1)
    area = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1) / (w*h)
    index = int(np.argmin(distance - .05*np.sqrt(area)))
    x1, y1, x2, y2 = boxes[index]
    margin = .04 * max(x2-x1, y2-y1)
    crop = image.crop((max(0, int(x1-margin)), max(0, int(y1-margin)),
                       min(w, int(x2+margin)), min(h, int(y2+margin))))
    return ImageOps.pad(crop, (384, 384), method=Image.Resampling.BICUBIC, color="white"), len(boxes)


def score_rows(rows, labels):
    verified = [r for r in rows if labels.get(r["image_path"]) and labels[r["image_path"]].status == "verified"]
    top1 = sum(r["predicted_slug"] == labels[r["image_path"]].slug for r in verified)
    top5 = sum(labels[r["image_path"]].slug in r["top5_slugs"] for r in verified)
    return dict(verified=len(verified), top1=top1, top5=top5,
                top1_accuracy=top1/len(verified), top5_accuracy=top5/len(verified))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images-dir", type=Path, default=ROOT / "data/official_real_photos")
    p.add_argument("--labels", type=Path, default=ROOT / "evaluation/all_official_labels.tsv")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--catalog-images", type=Path, default=ROOT / "data/competition_imgs")
    p.add_argument("--detector", type=Path, required=True)
    p.add_argument("--head", type=Path, required=True)
    p.add_argument("--reranker", type=Path, required=True)
    p.add_argument("--baseline", type=Path, default=ROOT / "data/evaluation/repro_hybrid.jsonl")
    p.add_argument("--output", type=Path, default=ROOT / "data/evaluation/trained_cascade.jsonl")
    args = p.parse_args()
    labels = read_labels(args.labels)
    baseline = [json.loads(line) for line in args.baseline.read_text().splitlines()]
    baseline_by_name = {r["image_path"]: r for r in baseline}
    paths = sorted(p for p in args.images_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
    if set(labels) != {p.name for p in paths} or set(baseline_by_name) != set(labels):
        p.error("Baseline, labels and images must contain identical populations")
    for path in paths:
        if sha256(path) != labels[path.name].image_sha256:
            raise ValueError(f"Image changed after labeling: {path}")
    detector = YOLO(str(args.detector))
    encoder = VisionEncoder(args.model_dir)
    gallery = load_index(args.index, args.manifest)
    head_state = torch.load(args.head, map_location="cpu", weights_only=False)
    head = LabelHead(**head_state["config"])
    head.load_state_dict(head_state["state_dict"])
    head.eval()
    rerank_state = torch.load(args.reranker, map_location="cpu", weights_only=False)
    if rerank_state["head_sha256"] != sha256(args.head):
        raise ValueError("Reranker/head checkpoint mismatch")
    weights = rerank_state["state_dict"]["weight"].numpy().ravel()
    mean, std = rerank_state["mean"], rerank_state["std"]
    bias = float(rerank_state["state_dict"]["bias"].item())
    with torch.no_grad():
        projected_refs = head(torch.from_numpy(gallery.embeddings)).numpy()
    orb = cv2.ORB_create(nfeatures=500, fastThreshold=12)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    ref_cache = {}

    def reference_orb(index):
        if index not in ref_cache:
            image = decode_image(args.catalog_images / gallery.filenames[index])
            ref_cache[index] = orb_description(orb, reference_views(image)[1])
        return ref_cache[index]

    rows = []
    for number, path in enumerate(paths, 1):
        start = time.perf_counter()
        image = decode_image(path)
        label_image, detections = crop_label(detector, image)
        full = query_views(image)[0]
        query = encoder.encode([full, label_image], batch_size=2)
        with torch.no_grad():
            projected_query = head(torch.from_numpy(query)).numpy()
        full_scores = gallery.embeddings[:, 0] @ query[0]
        label_scores = gallery.embeddings[:, 1] @ query[1]
        base_scores = .35*full_scores + .65*label_scores
        order = np.argsort(-base_scores, kind="stable")[:20]
        visual_slugs = [slug for i in order for slug in gallery.image_slugs[i]][:5]
        head_scores = .35*(projected_refs[:, 0] @ projected_query[0]) + .65*(projected_refs[:, 1] @ projected_query[1])
        head_order = sorted(order, key=lambda i: -head_scores[i])
        head_slugs = [slug for i in head_order for slug in gallery.image_slugs[i]][:5]
        query_orb = orb_description(orb, label_image)
        label_base = label_scores[order]
        label_head = projected_refs[order, 1] @ projected_query[1]
        pair_features = []
        for rank, index in enumerate(order):
            pair_features.append([float(label_base[rank]), float(label_head[rank]),
                                  float(label_base[rank]-label_base.max()),
                                  float(label_head[rank]-label_head.max()), rank/19.,
                                  *orb_features(query_orb, reference_orb(int(index)), matcher)])
        pair_features = np.array(pair_features, dtype=np.float32)
        rerank_scores = ((pair_features-mean)/std) @ weights + bias
        rerank_order = order[np.argsort(-rerank_scores, kind="stable")]
        ranked = [slug for i in rerank_order for slug in gallery.image_slugs[i]]
        rows.append(dict(image_path=path.name, image_sha256=labels[path.name].image_sha256,
                         label_status=labels[path.name].status,
                         predicted_slug=ranked[0], top5_slugs=ranked[:5],
                         detector_boxes=detections, detector_fallback=detections == 0,
                         visual_top5_slugs=visual_slugs, head_top5_slugs=head_slugs,
                         latency_ms=round((time.perf_counter()-start)*1000)))
        if number % 10 == 0 or number == len(paths):
            print(f"Predicted {number}/{len(paths)}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    latencies = [r["latency_ms"] for r in rows]
    report = dict(population=len(paths), verified=sum(l.status == "verified" for l in labels.values()),
                  baseline=score_rows(baseline, labels),
                  detector_visual=score_rows([dict(image_path=r["image_path"], predicted_slug=r["visual_top5_slugs"][0],
                                                   top5_slugs=r["visual_top5_slugs"]) for r in rows], labels),
                  detector_head=score_rows([dict(image_path=r["image_path"], predicted_slug=r["head_top5_slugs"][0],
                                                 top5_slugs=r["head_top5_slugs"]) for r in rows], labels),
                  cascade=score_rows(rows, labels),
                  p50_ms=float(np.percentile(latencies, 50)), p95_ms=float(np.percentile(latencies, 95)),
                  detector_fallbacks=sum(r["detector_fallback"] for r in rows),
                  checkpoint_sha256={"detector": sha256(args.detector), "head": sha256(args.head),
                                     "reranker": sha256(args.reranker)},
                  index_sha256=sha256(args.index), catalog_manifest_sha256=sha256(args.manifest),
                  labels_sha256=sha256(args.labels), predictions_sha256=sha256(args.output),
                  baseline_predictions_sha256=sha256(args.baseline),
                  git_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                  python=platform.python_version(), errors=[dict(image=r["image_path"], truth=labels[r["image_path"]].slug,
                         predicted=r["predicted_slug"], baseline=baseline_by_name[r["image_path"]]["predicted_slug"])
                         for r in rows if labels[r["image_path"]].status == "verified" and r["predicted_slug"] != labels[r["image_path"]].slug])
    args.output.with_suffix(".summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "errors"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

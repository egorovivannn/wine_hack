"""Train a Top-20 pairwise label reranker with external hard negatives."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

from .train_label_head import LabelHead, sha256


FEATURES = ["base_cos", "head_cos", "base_relative", "head_relative", "base_rank",
            "orb_good_fraction", "orb_good_log", "orb_inliers_log"]


def orb_features(a, b, matcher):
    points_a, desc_a = a
    points_b, desc_b = b
    if desc_a is None or desc_b is None or len(desc_b) < 2:
        return (0., 0., 0.)
    matches = matcher.knnMatch(desc_a, desc_b, k=2)
    good = [m for pair in matches if len(pair) == 2 for m, n in [pair] if m.distance < .75*n.distance]
    inliers = 0
    if len(good) >= 5:
        from_pts = np.float32([points_a[m.queryIdx].pt for m in good])
        to_pts = np.float32([points_b[m.trainIdx].pt for m in good])
        _, mask = cv2.findHomography(from_pts, to_pts, cv2.RANSAC, 5.)
        inliers = int(mask.sum()) if mask is not None else 0
    return (len(good) / max(1, min(len(desc_a), len(desc_b))),
            np.log1p(len(good)), np.log1p(inliers))


def build_examples(frame, base, head, split, descriptors):
    ids = np.flatnonzero(frame.split.to_numpy() == split)
    permitted = {"train"} if split == "train" else {"train", "val"} if split == "val" else {"train", "val", "test"}
    pool = np.flatnonzero(frame.split.isin(permitted).to_numpy())
    labels = frame.vintage_id.to_numpy()
    gallery = np.array([pool[np.flatnonzero(labels[pool] == cls)[0]] for cls in sorted(set(labels[pool]))])
    queries = np.setdiff1d(ids, gallery)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    examples = []
    for query in queries:
        base_scores = base[gallery] @ base[query]
        order = np.argsort(-base_scores, kind="stable")[:20]
        candidates = gallery[order]
        head_scores = head[candidates] @ head[query]
        values = []
        for rank, (candidate, hs) in enumerate(zip(candidates, head_scores)):
            orb = orb_features(descriptors[query], descriptors[candidate], matcher)
            values.append([float(base_scores[order[rank]]), float(hs),
                           float(base_scores[order[rank]] - base_scores[order[0]]),
                           float(hs - head_scores.max()), rank/19., *orb])
        truth = np.flatnonzero(labels[candidates] == labels[query])
        examples.append((np.asarray(values, dtype=np.float32), int(truth[0]) if len(truth) else -1,
                         str(frame.filename.iloc[query]), str(labels[query])))
    return examples


def metrics(examples, scorer):
    n = len(examples)
    top1 = top5 = hit20 = 0
    failures = []
    for values, truth, filename, label in examples:
        ranks = np.argsort(-scorer(values), kind="stable")
        top1 += bool(truth >= 0 and ranks[0] == truth)
        top5 += bool(truth >= 0 and truth in ranks[:5])
        hit20 += bool(truth >= 0)
        if truth < 0 or ranks[0] != truth:
            failures.append(dict(image=filename, vintage_id=label, truth_candidate=truth,
                                 predicted_candidate=int(ranks[0])))
    return dict(queries=n, top1=top1/n, top5=top5/n, hit20=hit20/n,
                top1_count=top1, top5_count=top5, hit20_count=hit20,
                errors=failures[:30])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=Path("data/external/winesensed/sample_v2/manifest.csv"))
    p.add_argument("--head", type=Path, default=Path("data/models/training/label_head_v1/best.pt"))
    p.add_argument("--output", type=Path, default=Path("data/models/training/reranker_v1"))
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.output.exists():
        p.error("Output exists; choose a fresh run directory")
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    cv2.setNumThreads(1)
    torch.manual_seed(args.seed)
    frame = pd.read_csv(args.manifest, dtype=str).sort_values("filename").reset_index(drop=True)
    for path, expected in zip(frame.path, frame.sha256):
        if sha256(path) != expected:
            raise ValueError(f"Image changed: {path}")
    with np.load(args.manifest.parent / "siglip_features.npz") as cache:
        if cache["filenames"].tolist() != frame.filename.tolist():
            raise ValueError("Feature cache order differs")
        base = cache["vectors"].astype(np.float32)
    state = torch.load(args.head, map_location="cpu", weights_only=False)
    if state["fingerprint"]["manifest_sha256"] != sha256(args.manifest):
        raise ValueError("Head trained with different image manifest")
    head_model = LabelHead(**state["config"])
    head_model.load_state_dict(state["state_dict"])
    head_model.eval()
    with torch.no_grad():
        head = head_model(torch.from_numpy(base)).numpy()
    orb = cv2.ORB_create(nfeatures=500, fastThreshold=12)
    descriptors = []
    for path in frame.path:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Decode failed: {path}")
        h, w = image.shape
        image = cv2.resize(image, (round(w*min(1., 640/max(h,w))), round(h*min(1., 640/max(h,w)))))
        descriptors.append(orb.detectAndCompute(image, None))
    groups = {split: build_examples(frame, base, head, split, descriptors)
              for split in ("train", "val", "test")}
    matrix = np.concatenate([v for v, _, _, _ in groups["train"]])
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0).clip(min=1e-4)
    model = nn.Linear(len(FEATURES), 1)
    nn.init.zeros_(model.weight)
    nn.init.zeros_(model.bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.1)
    train_examples = [(v, truth) for v, truth, _, _ in groups["train"] if truth >= 0]
    if len(train_examples) < 30:
        raise ValueError("Too few real positive Top-20 pairs")
    train_x = torch.from_numpy(np.stack([(v-mean)/std for v, _ in train_examples]))
    train_y = torch.tensor([truth for _, truth in train_examples])
    base_metrics = {split: metrics(examples, lambda v: v[:, 0]) for split, examples in groups.items()}
    history = []
    best = -1.
    for epoch in range(args.epochs):
        model.train()
        logits = model(train_x).squeeze(-1)
        objective = F.cross_entropy(logits, train_y)
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            scorer = lambda v: model(torch.from_numpy((v-mean)/std)).flatten().numpy()
            val = metrics(groups["val"], scorer)
        history.append(dict(epoch=epoch+1, train_loss=float(objective),
                            val_top1=val["top1"], val_top5=val["top5"]))
        if val["top1"] > best:
            best = val["top1"]
            torch.save(dict(state_dict=model.state_dict(), mean=mean, std=std,
                            feature_names=FEATURES, head_sha256=sha256(args.head),
                            manifest_sha256=sha256(args.manifest), epoch=epoch+1),
                       args.output / "best.pt")
        if (epoch+1) % 20 == 0:
            print(json.dumps(history[-1]), flush=True)
    state = torch.load(args.output / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"])
    model.eval()
    with torch.no_grad():
        scorer = lambda v: model(torch.from_numpy((v-mean)/std)).flatten().numpy()
        trained_metrics = {split: metrics(examples, scorer) for split, examples in groups.items()}
    report = dict(baseline=base_metrics, trained=trained_metrics,
                  best_epoch=state["epoch"], checkpoint_sha256=sha256(args.output / "best.pt"),
                  head_sha256=sha256(args.head), manifest_sha256=sha256(args.manifest),
                  elapsed_seconds=round(time.perf_counter()-started, 1),
                  git_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("baseline", "trained")}, indent=2), flush=True)


if __name__ == "__main__":
    main()

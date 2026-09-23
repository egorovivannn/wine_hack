"""Train a residual SigLIP2 label embedding head on real WineSensed repeats."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

from scanner.image import decode_image
from scanner.vision import DEFAULT_MODEL_DIR, VisionEncoder


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class LabelHead(nn.Module):
    def __init__(self, width=768, hidden=256):
        super().__init__()
        self.first = nn.Linear(width, hidden)
        self.last = nn.Linear(hidden, width)
        nn.init.zeros_(self.last.weight)
        nn.init.zeros_(self.last.bias)

    def forward(self, x):
        return F.normalize(x + self.last(F.gelu(self.first(x))), dim=-1)


def features(manifest_path, model_dir, cache_path, batch):
    frame = pd.read_csv(manifest_path, dtype=str).sort_values("filename").reset_index(drop=True)
    if frame.filename.duplicated().any():
        raise ValueError("Duplicate WineSensed image filename")
    for path, expected in zip(frame.path, frame.sha256):
        if sha256(path) != expected:
            raise ValueError(f"Source image changed: {path}")
    fingerprint = dict(manifest_sha256=sha256(manifest_path),
                       base_weights_sha256=sha256(model_dir / "model.safetensors"))
    meta_path = cache_path.with_suffix(".json")
    if cache_path.exists():
        if json.loads(meta_path.read_text()) != fingerprint:
            raise ValueError("Feature cache provenance differs")
        with np.load(cache_path) as data:
            vectors = data["vectors"]
            if data["filenames"].tolist() != frame.filename.tolist():
                raise ValueError("Feature cache order differs")
        return frame, vectors.astype(np.float32), fingerprint
    encoder = VisionEncoder(model_dir)
    vectors = []
    for start in range(0, len(frame), batch):
        images = [decode_image(path) for path in frame.path.iloc[start:start + batch]]
        vectors.append(encoder.encode(images, batch_size=batch))
        if (start // batch) % 20 == 0:
            print(f"Encoded {min(start+batch, len(frame))}/{len(frame)}", flush=True)
    vectors = np.concatenate(vectors).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, filenames=np.array(frame.filename.tolist()), vectors=vectors)
    meta_path.write_text(json.dumps(fingerprint, indent=2) + "\n")
    del encoder
    torch.cuda.empty_cache()
    return frame, vectors, fingerprint


@torch.no_grad()
def retrieval(model, frame, x, split):
    ids = np.flatnonzero(frame.split.to_numpy() == split)
    classes = frame.vintage_id.to_numpy()[ids]
    gallery = np.array([ids[np.flatnonzero(classes == cls)[0]] for cls in sorted(set(classes))])
    queries = np.setdiff1d(ids, gallery)
    if not len(queries):
        raise ValueError(f"No positive pairs in {split}")
    embeddings = model(x) if model is not None else F.normalize(x, dim=-1)
    scores = (embeddings[queries] @ embeddings[gallery].T).cpu().numpy()
    order = np.argsort(-scores, axis=1, kind="stable")
    truth = frame.vintage_id.to_numpy()[gallery][order] == frame.vintage_id.to_numpy()[queries, None]
    return dict(top1=float(truth[:, 0].mean()), top5=float(truth[:, :5].any(axis=1).mean()),
                queries=len(queries), gallery=len(gallery))


def supcon(z, labels, temperature=.08):
    scores = z @ z.T / temperature
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    positives = labels[:, None].eq(labels[None, :]) & ~eye
    scores = scores.masked_fill(eye, -1e4)
    log_prob = scores - torch.logsumexp(scores, dim=1, keepdim=True)
    return -(log_prob * positives).sum(dim=1).div(positives.sum(dim=1).clamp_min(1)).mean()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=Path("data/external/winesensed/sample_v2/manifest.csv"))
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--output", type=Path, default=Path("data/models/training/label_head_v1"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--steps-per-epoch", type=int, default=50)
    p.add_argument("--batch-classes", type=int, default=16)
    p.add_argument("--images-per-class", type=int, default=4)
    p.add_argument("--encode-batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.output.exists():
        p.error("Output exists; choose a fresh run directory")
    args.output.mkdir(parents=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    started = time.perf_counter()
    frame, vectors, fingerprint = features(args.manifest, args.model_dir,
                                           args.manifest.parent / "siglip_features.npz", args.encode_batch)
    if frame.groupby("vintage_id").split.nunique().max() != 1:
        raise ValueError("Vintage identity crosses splits")
    if frame[frame.winery_id.notna() & frame.winery_id.ne("")].groupby("winery_id").split.nunique().max() > 1:
        raise ValueError("Winery identity crosses splits")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor(vectors, device=device)
    labels = frame.vintage_id.to_numpy()
    members = {k: np.flatnonzero((labels == k) & (frame.split.to_numpy() == "train")) for k in sorted(set(labels[frame.split == "train"]))}
    members = {k: v for k, v in members.items() if len(v) >= args.images_per_class}
    if len(members) < args.batch_classes:
        raise ValueError("Insufficient real multi-image classes")
    classes = list(members)
    prototypes = np.stack([vectors[v].mean(axis=0) for v in members.values()])
    prototypes /= np.linalg.norm(prototypes, axis=1, keepdims=True)
    hard_order = np.argsort(-(prototypes @ prototypes.T), axis=1)
    model = LabelHead(width=vectors.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    rng = np.random.default_rng(args.seed)
    base = {s: retrieval(None, frame, x, s) for s in ("val", "test")}
    history = []
    best = -1.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for _ in range(args.steps_per_epoch):
            anchor = int(rng.integers(len(classes)))
            choices = hard_order[anchor, :max(args.batch_classes * 3, 40)]
            sampled = rng.choice(choices, size=args.batch_classes, replace=False)
            ids = np.concatenate([rng.choice(members[classes[i]], size=args.images_per_class, replace=False) for i in sampled])
            targets = torch.arange(len(sampled), device=device).repeat_interleave(args.images_per_class)
            z = model(x[ids])
            loss = supcon(z, targets) + .03 * (z - x[ids]).square().sum(dim=1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss))
        model.eval()
        val = retrieval(model, frame, x, "val")
        history.append(dict(epoch=epoch+1, train_loss=float(np.mean(losses)), **val))
        if val["top1"] > best:
            best = val["top1"]
            torch.save(dict(state_dict=model.cpu().state_dict(), config=dict(width=vectors.shape[1], hidden=256),
                            fingerprint=fingerprint, epoch=epoch+1), args.output / "best.pt")
            model.to(device)
        print(json.dumps(history[-1]), flush=True)
    checkpoint = torch.load(args.output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    report = dict(base=base, trained={s: retrieval(model, frame, x, s) for s in ("val", "test")},
                  best_epoch=checkpoint["epoch"], checkpoint_sha256=sha256(args.output / "best.pt"),
                  fingerprint=fingerprint, manifest_sha256=sha256(args.manifest),
                  elapsed_seconds=round(time.perf_counter()-started, 1),
                  peak_allocated_gib=round(torch.cuda.max_memory_allocated()/2**30, 3) if device.type == "cuda" else 0,
                  git_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  args=vars(args) | {"manifest": str(args.manifest), "model_dir": str(args.model_dir), "output": str(args.output)})
    (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

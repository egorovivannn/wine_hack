"""Train a small SigLIP2 residual head on real external query-reference pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .train_label_head import LabelHead


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_data(views: Path, features: Path) -> tuple[list[dict], np.ndarray]:
    rows = [json.loads(line) for line in views.read_text().splitlines()]
    with np.load(features) as saved:
        vectors = saved["vectors"].astype(np.float32)
        if saved["views"].tolist() != [r["view"] for r in rows]:
            raise ValueError("Feature/view order differs")
    if len(vectors) != len(rows):
        raise ValueError("Feature count differs")
    for source in ("winesensed", "norwegian"):
        reference = [
            r for r in rows if r["source"] == source and r["role"] == "reference"
        ]
        keys = [(r["identity"], r["split"]) for r in reference]
        if len(set(r["identity"] for r in reference)) != len(reference):
            raise ValueError("Multiple references per identity")
        for row in (r for r in rows if r["source"] == source):
            if (row["identity"], row["split"]) not in keys:
                raise ValueError("Query/reference identity or split mismatch")
    scene_splits = defaultdict(set)
    winery_splits = defaultdict(set)
    for row in rows:
        if row["source"] == "norwegian" and row["role"] == "query":
            scene_splits[row["scene_group"]].add(row["split"])
        if row["source"] == "winesensed" and row.get("winery_id"):
            winery_splits[row["winery_id"]].add(row["split"])
    if any(len(splits) > 1 for splits in scene_splits.values()):
        raise ValueError("Norwegian scene group crosses splits")
    if any(len(splits) > 1 for splits in winery_splits.values()):
        raise ValueError("Known WineSensed winery crosses splits")
    return rows, vectors


def retrieval(
    rows: list[dict],
    vectors: np.ndarray,
    source: str,
    split: str,
    head: LabelHead | None = None,
) -> dict:
    queries = np.array(
        [
            i
            for i, row in enumerate(rows)
            if row["source"] == source
            and row["role"] == "query"
            and row["split"] == split
        ]
    )
    gallery = np.array(
        [
            i
            for i, row in enumerate(rows)
            if row["source"] == source and row["role"] == "reference"
        ]
    )
    device = next(head.parameters()).device if head is not None else torch.device("cpu")
    x = torch.from_numpy(vectors).to(device)
    with torch.no_grad():
        if head is not None:
            head.eval()
            q = head(x[queries]).cpu().numpy()
            g = head(x[gallery]).cpu().numpy()
        else:
            q, g = vectors[queries], vectors[gallery]
    scores = q @ g.T
    order = np.argsort(-scores, axis=1, kind="stable")
    truth = np.array([r["identity"] for r in rows], dtype=object)
    matches = truth[gallery][order] == truth[queries, None]
    return dict(
        queries=len(queries),
        gallery=len(gallery),
        classes=len(set(truth[gallery])),
        top1=int(matches[:, 0].sum()),
        hit5=int(matches[:, :5].any(axis=1).sum()),
        recall20=int(matches[:, :20].any(axis=1).sum()),
    )


def train(
    rows: list[dict],
    vectors: np.ndarray,
    loss_name: str,
    epochs: int,
    seed: int,
    output: Path,
    views: Path,
    features: Path,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.from_numpy(vectors).to(device)
    train_sources = {}
    for source in ("winesensed", "norwegian"):
        refs = np.array(
            [
                i
                for i, r in enumerate(rows)
                if r["source"] == source
                and r["role"] == "reference"
                and r["split"] == "train"
            ]
        )
        queries = np.array(
            [
                i
                for i, r in enumerate(rows)
                if r["source"] == source
                and r["role"] == "query"
                and r["split"] == "train"
            ]
        )
        codes = [rows[i]["identity"] for i in refs]
        lookup = {code: j for j, code in enumerate(codes)}
        targets = np.array([lookup[rows[i]["identity"]] for i in queries])
        scores = vectors[queries] @ vectors[refs].T
        hard = np.argsort(-scores, axis=1, kind="stable")[:, :64]
        # Positive is always first, followed by the nearest other products.
        candidates = np.concatenate(
            [
                targets[:, None],
                np.array(
                    [
                        [j for j in order if j != target][:63]
                        for order, target in zip(hard, targets)
                    ]
                ),
            ],
            axis=1,
        )
        train_sources[source] = dict(refs=refs, queries=queries, candidates=candidates)
    model = LabelHead(width=vectors.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    rng = np.random.default_rng(seed)
    history = []
    best = -1.0
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        losses = []
        for step in range(120):
            source = ("winesensed", "norwegian")[step % 2]
            data = train_sources[source]
            sampled = rng.integers(0, len(data["queries"]), size=96)
            qi = data["queries"][sampled]
            ci = data["refs"][data["candidates"][sampled]]
            q = model(x[qi])
            g = model(x[ci.reshape(-1)]).reshape(len(qi), 64, -1)
            scores = torch.einsum("bd,bkd->bk", q, g)
            if loss_name == "ce":
                objective = F.cross_entropy(
                    scores / 0.07, torch.zeros(len(qi), dtype=torch.long, device=device)
                )
            elif loss_name == "margin":
                objective = F.softplus(
                    (scores[:, 1:] - scores[:, :1] + 0.08) * 12
                ).mean()
            else:
                raise ValueError(loss_name)
            objective = objective + 0.03 * (q - x[qi]).square().sum(dim=1).mean()
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            optimizer.step()
            losses.append(float(objective))
        val = {
            source: retrieval(rows, vectors, source, "val", model)
            for source in train_sources
        }
        score = sum(
            val[source]["top1"] / val[source]["queries"] * weight
            for source, weight in (("winesensed", 0.4), ("norwegian", 0.6))
        )
        history.append(
            dict(epoch=epoch + 1, loss=float(np.mean(losses)), val=val, val_score=score)
        )
        if score > best:
            best = score
            torch.save(
                dict(
                    state_dict={k: v.cpu() for k, v in model.state_dict().items()},
                    config=dict(width=vectors.shape[1], hidden=256),
                    loss=loss_name,
                    epoch=epoch + 1,
                    seed=seed,
                    views_sha256=sha256(views),
                    features_sha256=sha256(features),
                ),
                output / "best.pt",
            )
        print(json.dumps(history[-1]), flush=True)
    (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    (output / "report.json").write_text(
        json.dumps(
            dict(
                loss=loss_name,
                best_epoch=max(history, key=lambda v: v["val_score"])["epoch"],
                best_val_score=best,
                checkpoint_sha256=sha256(output / "best.pt"),
                views_sha256=sha256(views),
                features_sha256=sha256(features),
                elapsed_seconds=round(time.perf_counter() - started, 1),
                seed=seed,
            ),
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--views", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--loss", choices=("ce", "margin"), default="ce")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists")
    rows, vectors = load_data(args.views, args.features)
    args.output.mkdir(parents=True)
    train(
        rows,
        vectors,
        args.loss,
        args.epochs,
        args.seed,
        args.output,
        args.views,
        args.features,
    )


if __name__ == "__main__":
    main()

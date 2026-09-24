"""Train visual and OCR Top-20 scorers on product/scene-disjoint shelf pairs."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from rapidfuzz.fuzz import ratio
from torch import nn
from unidecode import unidecode

from scanner.ocr import VISUAL_MARGIN_FOR_OCR, OCRLine, OCRReranker
from scanner.vision import Candidate

from .train_external_adapter import load_data, permitted_gallery_splits, sha256
from .train_label_head import LabelHead

FEATURES = (
    "base_cos",
    "head_cos",
    "base_relative",
    "head_relative",
    "candidate_rank",
    "ocr_match",
)


def tokens(text: str) -> set[str]:
    return {
        word
        for word in re.findall(r"[a-z0-9]+", unidecode(text).lower())
        if len(word) >= 3
    }


def lexical_score(
    lines: list[dict], name: str, frequency: Counter, population: int
) -> float:
    names = tokens(name)
    hits = []
    for line in lines:
        if line["confidence"] < 0.3:
            continue
        for word in tokens(line["text"]):
            matches = [(ratio(word, candidate), candidate) for candidate in names]
            if matches:
                similarity, matched = max(matches)
                if similarity >= (90 if len(word) <= 4 else 80):
                    hits.append(
                        float(line["confidence"])
                        * similarity
                        / 100
                        * math.log((population + 1) / (frequency[matched] + 1))
                    )
    return sum(sorted(hits, reverse=True)[:3]) / max(1.0, 3 * math.log(population + 1))


def build_examples(
    rows: list[dict],
    vectors: np.ndarray,
    head: LabelHead,
    ocr: dict,
    split: str,
    candidate_source: str = "adapter",
) -> list[dict]:
    gallery = np.array(
        [
            i
            for i, r in enumerate(rows)
            if r["source"] == "norwegian"
            and r["role"] == "reference"
            and r["split"] in permitted_gallery_splits(split)
        ]
    )
    queries = np.array(
        [
            i
            for i, r in enumerate(rows)
            if r["source"] == "norwegian"
            and r["role"] == "query"
            and r["split"] == split
        ]
    )
    with torch.no_grad():
        needed = np.union1d(gallery, queries)
        projected = np.empty_like(vectors)
        projected[needed] = head(torch.from_numpy(vectors[needed])).numpy()
    names = [rows[i]["product_name"] for i in gallery]
    frequency = Counter(word for name in names for word in tokens(name))
    examples = []
    for qi in queries:
        base_scores = vectors[gallery] @ vectors[qi]
        all_head_scores = projected[gallery] @ projected[qi]
        if candidate_source == "adapter":
            top = np.argsort(-all_head_scores, kind="stable")[:20]
        elif candidate_source == "frozen":
            top = np.argsort(-base_scores, kind="stable")[:20]
        else:
            raise ValueError(candidate_source)
        chosen = gallery[top]
        head_scores = all_head_scores[top]
        ocr_lines = ocr[rows[qi]["view"]]
        values = np.array(
            [
                [
                    base_scores[index],
                    head_scores[rank],
                    base_scores[index] - base_scores[top].max(),
                    head_scores[rank] - max(head_scores),
                    rank / 19.0,
                    lexical_score(
                        ocr_lines,
                        rows[candidate]["product_name"],
                        frequency,
                        len(gallery),
                    ),
                ]
                for rank, (index, candidate) in enumerate(zip(top, chosen))
            ],
            dtype=np.float32,
        )
        matching = [
            rank
            for rank, candidate in enumerate(chosen)
            if rows[candidate]["identity"] == rows[qi]["identity"]
        ]
        examples.append(
            dict(
                view=rows[qi]["view"],
                identity=rows[qi]["identity"],
                values=values,
                truth=matching[0] if matching else -1,
                candidates=[rows[i]["identity"] for i in chosen],
            )
        )
    return examples


def metrics(examples: list[dict], scorer) -> dict:
    top1 = top5 = hit20 = 0
    errors = []
    for example in examples:
        ranking = np.argsort(-scorer(example["values"]), kind="stable")
        truth = example["truth"]
        hit20 += int(truth >= 0)
        top1 += int(truth >= 0 and ranking[0] == truth)
        top5 += int(truth >= 0 and truth in ranking[:5])
        if truth < 0 or ranking[0] != truth:
            errors.append(
                dict(
                    view=example["view"],
                    truth=example["identity"],
                    predicted=example["candidates"][ranking[0]],
                    candidate_hit=truth >= 0,
                )
            )
    return dict(
        queries=len(examples),
        top1=top1,
        hit5=top5,
        recall20=hit20,
        top1_rate=top1 / len(examples),
        hit5_rate=top5 / len(examples),
        recall20_rate=hit20 / len(examples),
        errors=errors[:30],
    )


def current_hybrid_proxy(
    rows: list[dict], vectors: np.ndarray, ocr: dict, split: str
) -> dict:
    """Run the current scanner's OCR trigger/scorer with available grocery names."""
    gallery = [
        i
        for i, row in enumerate(rows)
        if row["source"] == "norwegian"
        and row["role"] == "reference"
        and row["split"] in permitted_gallery_splits(split)
    ]
    queries = [
        i
        for i, row in enumerate(rows)
        if row["source"] == "norwegian"
        and row["role"] == "query"
        and row["split"] == split
    ]
    cards = {
        rows[i]["identity"]: dict(
            slug=rows[i]["identity"],
            name=rows[i]["product_name"],
            winery="",
            grape="",
            reference_available=True,
        )
        for i in gallery
    }
    scorer = OCRReranker(cards)
    counts = Counter()
    errors = []
    for qi in queries:
        scores = vectors[gallery] @ vectors[qi]
        top = np.argsort(-scores, kind="stable")[:20]
        candidates = [
            Candidate(
                slug=rows[gallery[j]]["identity"],
                score=float(scores[j]),
                full_score=float(scores[j]),
                center_score=float(scores[j]),
                image_name=rows[gallery[j]]["view"],
            )
            for j in top
        ]
        truth = rows[qi]["identity"]
        counts["ocr_triggered"] += int(
            len(candidates) >= 2
            and candidates[0].score - candidates[1].score < VISUAL_MARGIN_FOR_OCR
        )
        lines = [
            OCRLine(line["text"], line["confidence"]) for line in ocr[rows[qi]["view"]]
        ]
        ranked = scorer.rerank(candidates, lines)
        slugs = [candidate.slug for candidate in ranked]
        counts["top1"] += int(slugs[0] == truth)
        counts["hit5"] += int(truth in slugs[:5])
        counts["recall20"] += int(truth in slugs)
        if slugs[0] != truth:
            errors.append(
                dict(
                    view=rows[qi]["view"],
                    truth=truth,
                    predicted=slugs[0],
                    candidate_hit=truth in slugs,
                )
            )
    return dict(
        queries=len(queries),
        gallery=len(gallery),
        top1=counts["top1"],
        hit5=counts["hit5"],
        recall20=counts["recall20"],
        ocr_triggered=counts["ocr_triggered"],
        errors=errors[:30],
    )


def fit(
    examples: list[dict], val: list[dict], feature_count: int, output: Path
) -> dict:
    matrices = np.concatenate([e["values"][:, :feature_count] for e in examples])
    mean = matrices.mean(axis=0)
    std = matrices.std(axis=0).clip(min=1e-4)
    x = torch.from_numpy(
        np.stack(
            [
                (e["values"][:, :feature_count] - mean) / std
                for e in examples
                if e["truth"] >= 0
            ]
        )
    )
    y = torch.tensor([e["truth"] for e in examples if e["truth"] >= 0])
    if len(y) < 50:
        raise ValueError("Too few train queries with positive Top-20 candidate")
    torch.manual_seed(42)
    model = nn.Linear(feature_count, 1)
    nn.init.zeros_(model.weight)
    nn.init.zeros_(model.bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.1)
    raw_heads = torch.from_numpy(
        np.stack([e["values"][:, 1] for e in examples if e["truth"] >= 0])
    )
    scale = 0.03
    best = metrics(val, lambda values: values[:, 1])["top1"]
    history = [dict(epoch=0, loss=None, val_top1=best)]
    torch.save(
        dict(
            weights=np.zeros(feature_count, dtype=np.float32),
            mean=mean,
            std=std,
            features=FEATURES[:feature_count],
            epoch=0,
            scale=scale,
        ),
        output,
    )
    for epoch in range(150):
        logits = (raw_heads + scale * model(x).squeeze(-1)) / 0.07
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            weights = model.weight.detach().numpy().ravel().copy()

            def scorer(values):
                return values[:, 1] + scale * (
                    ((values[:, :feature_count] - mean) / std) @ weights
                )

            measured = metrics(val, scorer)
        history.append(
            dict(epoch=epoch + 1, loss=float(loss), val_top1=measured["top1"])
        )
        if measured["top1"] > best:
            best = measured["top1"]
            torch.save(
                dict(
                    weights=weights,
                    mean=mean,
                    std=std,
                    features=FEATURES[:feature_count],
                    epoch=epoch + 1,
                    scale=scale,
                ),
                output,
            )
    torch.save(
        dict(
            weights=model.weight.detach().numpy().ravel().copy(),
            mean=mean,
            std=std,
            features=FEATURES[:feature_count],
            epoch=150,
            scale=scale,
        ),
        output.with_name(output.stem + "_last.pt"),
    )
    return dict(
        best_val_top1=best,
        best_epoch=max(history, key=lambda r: r["val_top1"])["epoch"],
        last_val_top1=history[-1]["val_top1"],
        train_queries=len(examples),
        train_positive_candidates=len(y),
        history=history,
    )


def checkpoint_score(checkpoint: dict):
    return lambda v: (
        v[:, 1]
        + checkpoint["scale"]
        * (
            (
                (v[:, : len(checkpoint["weights"])] - checkpoint["mean"])
                / checkpoint["std"]
            )
            @ checkpoint["weights"]
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--views", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--ocr", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists")
    rows, vectors = load_data(args.views, args.features)
    state = torch.load(args.head, map_location="cpu", weights_only=False)
    if state["views_sha256"] != sha256(args.views) or state[
        "features_sha256"
    ] != sha256(args.features):
        raise ValueError("Adapter/checkpoint provenance differs")
    head = LabelHead(**state["config"])
    head.load_state_dict(state["state_dict"])
    head.eval()
    ocr_rows = [json.loads(line) for line in args.ocr.read_text().splitlines()]
    ocr = {r["view"]: r["lines"] for r in ocr_rows}
    expected = {
        r["view"]: r["view_sha256"]
        for r in rows
        if r["source"] == "norwegian" and r["role"] == "query"
    }
    if set(ocr) != set(expected) or any(
        r["view_sha256"] != expected[r["view"]] for r in ocr_rows
    ):
        raise ValueError("OCR/view identity differs")
    groups = {
        split: build_examples(rows, vectors, head, ocr, split)
        for split in ("train", "val")
    }
    frozen_groups = {
        split: build_examples(
            rows, vectors, head, ocr, split, candidate_source="frozen"
        )
        for split in ("train", "val")
    }
    args.output.mkdir(parents=True)
    visual = fit(groups["train"], groups["val"], 5, args.output / "visual.pt")
    text = fit(groups["train"], groups["val"], 6, args.output / "ocr.pt")
    comparison = {}
    for split, examples in groups.items():
        comparison[split] = {
            "frozen": metrics(frozen_groups[split], lambda v: v[:, 0]),
            "adapter": metrics(examples, lambda v: v[:, 1]),
            "hybrid_0.03": metrics(
                frozen_groups[split], lambda v: v[:, 0] + 0.03 * v[:, 5]
            ),
        }
        for name in ("visual", "ocr", "visual_last", "ocr_last"):
            checkpoint = torch.load(
                args.output / f"{name}.pt", map_location="cpu", weights_only=False
            )
            comparison[split][name] = metrics(examples, checkpoint_score(checkpoint))
    report = dict(
        comparison=comparison,
        training={
            "visual": {k: v for k, v in visual.items() if k != "history"},
            "ocr": {k: v for k, v in text.items() if k != "history"},
        },
        checkpoint_sha256={
            name: sha256(args.output / f"{name}.pt") for name in ("visual", "ocr")
        },
        trained_last_checkpoint_sha256={
            name: sha256(args.output / f"{name}_last.pt") for name in ("visual", "ocr")
        },
        views_sha256=sha256(args.views),
        features_sha256=sha256(args.features),
        head_sha256=sha256(args.head),
        ocr_sha256=sha256(args.ocr),
        cpu_cores=os.cpu_count(),
        python=platform.python_version(),
        torch=torch.__version__,
        git_revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "history.json").write_text(
        json.dumps({"visual": visual["history"], "ocr": text["history"]}, indent=2)
        + "\n"
    )
    print(
        json.dumps(
            {"validation": comparison["val"], "training": report["training"]}, indent=2
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

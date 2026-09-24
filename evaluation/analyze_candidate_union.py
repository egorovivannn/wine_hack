"""Diagnose Top-K losses on external validation with immutable feature caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from embedding.train_external_adapter import load_data, permitted_gallery_splits, sha256
from embedding.train_label_head import LabelHead


def analyze(
    rows: list[dict],
    vectors: np.ndarray,
    head: LabelHead,
    source: str,
    split: str,
    k: int = 20,
) -> dict:
    query_ids = [
        i
        for i, row in enumerate(rows)
        if row["source"] == source and row["role"] == "query" and row["split"] == split
    ]
    gallery_ids = [
        i
        for i, row in enumerate(rows)
        if row["source"] == source
        and row["role"] == "reference"
        and row["split"] in permitted_gallery_splits(split)
    ]
    with torch.no_grad():
        query_head = head(torch.from_numpy(vectors[query_ids])).numpy()
        gallery_head = head(torch.from_numpy(vectors[gallery_ids])).numpy()
    frozen_scores = vectors[query_ids] @ vectors[gallery_ids].T
    adapter_scores = query_head @ gallery_head.T
    frozen_order = np.argsort(-frozen_scores, axis=1, kind="stable")
    adapter_order = np.argsort(-adapter_scores, axis=1, kind="stable")
    gallery_truth = [rows[i]["identity"] for i in gallery_ids]
    counts = {
        "frozen_top1": 0,
        "frozen_hit5": 0,
        "frozen_recall20": 0,
        "adapter_top1": 0,
        "adapter_hit5": 0,
        "adapter_recall20": 0,
        "union_recall40": 0,
        "frozen_only": 0,
        "adapter_only": 0,
        "neither_top20": 0,
    }
    lost = []
    for index, query_id in enumerate(query_ids):
        truth = rows[query_id]["identity"]
        base = [gallery_truth[j] for j in frozen_order[index, :k]]
        adapted = [gallery_truth[j] for j in adapter_order[index, :k]]
        base_hit = truth in base
        adapted_hit = truth in adapted
        counts["frozen_top1"] += int(base[0] == truth)
        counts["frozen_hit5"] += int(truth in base[:5])
        counts["frozen_recall20"] += int(base_hit)
        counts["adapter_top1"] += int(adapted[0] == truth)
        counts["adapter_hit5"] += int(truth in adapted[:5])
        counts["adapter_recall20"] += int(adapted_hit)
        counts["union_recall40"] += int(base_hit or adapted_hit)
        counts["frozen_only"] += int(base_hit and not adapted_hit)
        counts["adapter_only"] += int(adapted_hit and not base_hit)
        counts["neither_top20"] += int(not base_hit and not adapted_hit)
        if base_hit and not adapted_hit:
            lost.append(
                {
                    "view": rows[query_id]["view"],
                    "identity": truth,
                    "frozen_rank": base.index(truth) + 1,
                    "adapter_truth_rank": next(
                        (
                            rank + 1
                            for rank, j in enumerate(adapter_order[index])
                            if gallery_truth[j] == truth
                        ),
                        None,
                    ),
                }
            )
    return {
        "source": source,
        "split": split,
        "queries": len(query_ids),
        "gallery": len(gallery_ids),
        "candidate_limit_per_method": k,
        "union_max_candidates": 2 * k,
        "counts": counts,
        "frozen_only_examples": lost[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--views", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists")
    rows, vectors = load_data(args.views, args.features)
    state = torch.load(args.head, map_location="cpu", weights_only=False)
    if state["views_sha256"] != sha256(args.views) or state[
        "features_sha256"
    ] != sha256(args.features):
        raise ValueError("Adapter feature provenance changed")
    head = LabelHead(**state["config"])
    head.load_state_dict(state["state_dict"])
    head.eval()
    report = {
        "views_sha256": sha256(args.views),
        "features_sha256": sha256(args.features),
        "head_sha256": sha256(args.head),
        "results": [
            analyze(rows, vectors, head, source, args.split)
            for source in ("winesensed", "norwegian")
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            [
                {k: v for k, v in result.items() if k != "frozen_only_examples"}
                for result in report["results"]
            ],
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

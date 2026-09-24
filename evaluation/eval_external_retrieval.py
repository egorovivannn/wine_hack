"""One-shot external validation/test of frozen, adapted and OCR reranked retrieval."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path

import torch

from embedding.train_external_adapter import load_data, retrieval, sha256
from embedding.train_external_reranker import build_examples, checkpoint_score, metrics
from embedding.train_label_head import LabelHead


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--views", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--reranker-dir", type=Path, required=True)
    parser.add_argument("--ocr", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--splits", nargs="+", choices=("val", "test"), default=["val", "test"]
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists")
    rows, vectors = load_data(args.views, args.features)
    state = torch.load(args.head, map_location="cpu", weights_only=False)
    if state["views_sha256"] != sha256(args.views) or state[
        "features_sha256"
    ] != sha256(args.features):
        raise ValueError("Adapter input provenance differs")
    head = LabelHead(**state["config"])
    head.load_state_dict(state["state_dict"])
    head.eval()
    ocr_records = [json.loads(line) for line in args.ocr.read_text().splitlines()]
    expected = {
        r["view"]: r["view_sha256"]
        for r in rows
        if r["source"] == "norwegian" and r["role"] == "query"
    }
    if set(r["view"] for r in ocr_records) != set(expected) or any(
        r["view_sha256"] != expected[r["view"]] for r in ocr_records
    ):
        raise ValueError("OCR cache provenance differs")
    ocr = {r["view"]: r["lines"] for r in ocr_records}
    output = {}
    for split in args.splits:
        wine = {
            name: retrieval(rows, vectors, "winesensed", split, model)
            for name, model in (("frozen", None), ("adapter", head))
        }
        norwegian = {
            name: retrieval(rows, vectors, "norwegian", split, model)
            for name, model in (("frozen", None), ("adapter_full_gallery", head))
        }
        examples = build_examples(rows, vectors, head, ocr, split)
        frozen_examples = build_examples(
            rows, vectors, head, ocr, split, candidate_source="frozen"
        )
        norwegian.update(
            {
                "hybrid_0.03": metrics(
                    frozen_examples, lambda v: v[:, 0] + 0.03 * v[:, 5]
                ),
                "adapter_top20": metrics(examples, lambda v: v[:, 1]),
            }
        )
        for name in ("visual", "ocr", "visual_last", "ocr_last"):
            checkpoint = torch.load(
                args.reranker_dir / f"{name}.pt", map_location="cpu", weights_only=False
            )
            norwegian[f"reranker_{name}"] = metrics(
                examples, checkpoint_score(checkpoint)
            )
        output[split] = dict(winesensed=wine, norwegian=norwegian)
    report = dict(
        results=output,
        views_sha256=sha256(args.views),
        features_sha256=sha256(args.features),
        checkpoints={
            "adapter": sha256(args.head),
            "reranker_visual": sha256(args.reranker_dir / "visual.pt"),
            "reranker_ocr": sha256(args.reranker_dir / "ocr.pt"),
            "reranker_visual_last": sha256(args.reranker_dir / "visual_last.pt"),
            "reranker_ocr_last": sha256(args.reranker_dir / "ocr_last.pt"),
        },
        ocr_sha256=sha256(args.ocr),
        git_revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        python=platform.python_version(),
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        split_protocol="Wine vintage/known winery/near duplicate group disjoint; Norwegian product and source scene-block disjoint. Wine capture sessions unavailable.",
        gallery_policy="Same eligible gallery per split and method: train only; train+val; full train+val+test.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                split: {
                    source: {
                        name: {k: v for k, v in measure.items() if k != "errors"}
                        for name, measure in methods.items()
                    }
                    for source, methods in output[split].items()
                }
                for split in output
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

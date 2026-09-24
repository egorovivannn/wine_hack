"""Encode audited external views with the frozen SigLIP2 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from scanner.image import decode_image
from scanner.vision import DEFAULT_MODEL_DIR, VisionEncoder


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--views", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists")
    rows = [json.loads(line) for line in args.views.read_text().splitlines()]
    encoder = VisionEncoder(args.model_dir)
    started = time.perf_counter()
    vectors = []
    for start in range(0, len(rows), args.batch):
        batch = rows[start : start + args.batch]
        images = []
        for row in batch:
            path = args.views.parent / row["view"]
            if sha256(path) != row["view_sha256"]:
                raise ValueError(f"View changed: {path}")
            images.append(decode_image(path))
        vectors.append(encoder.encode(images, batch_size=args.batch))
        if (start // args.batch) % 100 == 0:
            print(
                f"Encoded {min(start + args.batch, len(rows))}/{len(rows)} views",
                flush=True,
            )
    matrix = np.concatenate(vectors).astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, vectors=matrix, views=np.asarray([r["view"] for r in rows])
    )
    report = dict(
        views_manifest_sha256=sha256(args.views),
        base_checkpoint_sha256=sha256(args.model_dir / "model.safetensors"),
        features_sha256=sha256(args.output),
        count=len(rows),
        dimension=matrix.shape[1],
        elapsed_seconds=round(time.perf_counter() - started, 1),
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        peak_vram_gib=round(torch.cuda.max_memory_allocated() / 2**30, 3)
        if torch.cuda.is_available()
        else 0,
    )
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

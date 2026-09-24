"""Cache EasyOCR lines on real Norwegian shelf-product query crops."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from scanner.image import decode_image
from scanner.ocr import MODEL_SHA256, OCRReader


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--views", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists")
    rows = [json.loads(line) for line in args.views.read_text().splitlines()]
    queries = [r for r in rows if r["source"] == "norwegian" and r["role"] == "query"]
    reader = OCRReader()
    results = []
    for number, row in enumerate(queries, 1):
        path = args.views.parent / row["view"]
        if sha256(path) != row["view_sha256"]:
            raise ValueError(f"Crop changed: {path}")
        start = time.perf_counter()
        lines = reader.read(decode_image(path))
        results.append(
            dict(
                view=row["view"],
                view_sha256=row["view_sha256"],
                lines=[
                    dict(text=line.text, confidence=line.confidence) for line in lines
                ],
                latency_ms=(time.perf_counter() - start) * 1000,
            )
        )
        if number % 100 == 0:
            print(f"OCR {number}/{len(queries)}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results)
    )
    report = dict(
        views_sha256=sha256(args.views),
        ocr_checkpoint_sha256={
            name: sha256(Path("data/models/easyocr") / name) for name in MODEL_SHA256
        },
        output_sha256=sha256(args.output),
        queries=len(results),
        p50_ms=sorted(r["latency_ms"] for r in results)[len(results) // 2],
        p95_ms=sorted(r["latency_ms"] for r in results)[int(0.95 * (len(results) - 1))],
    )
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

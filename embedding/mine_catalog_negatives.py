"""Mine catalog lookalikes with visual, reference OCR and card metadata evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from unidecode import unidecode

from scanner.image import decode_image, reference_views
from scanner.ocr import OCRReader
from scanner.vision import load_index


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def words(value: str) -> set[str]:
    return {
        word
        for word in re.findall(r"[a-z0-9]+", unidecode(value).lower())
        if len(word) >= 4
    }


def similarity(left: set[str], right: set[str]) -> float:
    return len(left & right) / len(left | right) if left or right else 0.0


def card_words(cards: list[dict]) -> set[str]:
    return set().union(
        *(
            words(
                " ".join(
                    str(card.get(field, ""))
                    for field in ("name", "winery", "grape", "category")
                )
            )
            for card in cards
        )
    )


def ocr_cache(index, images: Path, output: Path, manifest: dict) -> list[set[str]]:
    if output.exists():
        saved = json.loads(output.read_text())
        if (
            saved["index_filenames"] != index.filenames
            or saved["manifest_sha256"] != manifest["manifest_sha256"]
        ):
            raise ValueError("OCR cache provenance differs")
        return [set(group) for group in saved["tokens"]]
    reader = OCRReader()
    tokens = []
    for number, (name, record) in enumerate(
        zip(index.filenames, manifest["images"]), 1
    ):
        path = images / name
        if sha256(path) != record["sha256"]:
            raise ValueError(f"Reference changed: {name}")
        lines = reader.read(reference_views(decode_image(path))[1])
        tokens.append(
            sorted(
                set().union(
                    *(words(line.text) for line in lines if line.confidence >= 0.4)
                )
            )
        )
        if number % 200 == 0:
            print(f"Catalog OCR {number}/{len(index.filenames)}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            dict(
                index_filenames=index.filenames,
                manifest_sha256=manifest["manifest_sha256"],
                tokens=tokens,
            )
        )
        + "\n"
    )
    return [set(group) for group in tokens]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/catalog_manifest.json")
    )
    parser.add_argument("--index", type=Path, default=Path("data/index/siglip2.npz"))
    parser.add_argument("--images", type=Path, default=Path("data/competition_imgs"))
    parser.add_argument(
        "--ocr-cache", type=Path, default=Path("data/external/catalog_ocr.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists")
    index = load_index(args.index, args.manifest)
    manifest = json.loads(args.manifest.read_text())
    manifest["manifest_sha256"] = sha256(args.manifest)
    cards = {card["slug"]: card for card in manifest["cards"]}
    metadata = [
        card_words([cards[slug] for slug in slugs]) for slugs in index.image_slugs
    ]
    ocr = ocr_cache(index, args.images, args.ocr_cache, manifest)
    embeddings = index.embeddings[:, 1]
    visual = embeddings @ embeddings.T
    np.fill_diagonal(visual, -1)
    ocr_scores = np.array(
        [[similarity(a, b) for b in ocr] for a in ocr], dtype=np.float32
    )
    metadata_scores = np.array(
        [[similarity(a, b) for b in metadata] for a in metadata], dtype=np.float32
    )
    np.fill_diagonal(ocr_scores, -1)
    np.fill_diagonal(metadata_scores, -1)
    candidate_pairs = set()
    for matrix in (visual, ocr_scores, metadata_scores):
        for i in range(len(index.filenames)):
            for j in np.argsort(-matrix[i], kind="stable")[:20]:
                candidate_pairs.add(tuple(sorted((i, int(j)))))
    rows = []
    for i, j in sorted(candidate_pairs):
        left, right = index.image_slugs[i], index.image_slugs[j]
        names_left = {" ".join(sorted(words(cards[slug]["name"]))) for slug in left}
        names_right = {" ".join(sorted(words(cards[slug]["name"]))) for slug in right}
        ambiguous = (
            bool(names_left & names_right)
            or manifest["images"][i]["sha256"] == manifest["images"][j]["sha256"]
        )
        rows.append(
            dict(
                left=index.filenames[i],
                right=index.filenames[j],
                left_slugs=left,
                right_slugs=right,
                visual=float(visual[i, j]),
                ocr=float(ocr_scores[i, j]),
                metadata=float(metadata_scores[i, j]),
                ambiguous=ambiguous,
            )
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    )
    report = dict(
        catalog_images=len(index.filenames),
        candidate_pairs=len(rows),
        ambiguous_pairs=sum(r["ambiguous"] for r in rows),
        unambiguous_pairs=sum(not r["ambiguous"] for r in rows),
        index_sha256=sha256(args.index),
        manifest_sha256=sha256(args.manifest),
        ocr_cache_sha256=sha256(args.ocr_cache),
        pairs_sha256=sha256(args.output),
        caution="Unverified negative candidates; ambiguous names or identical files excluded from training. No catalog query positives are asserted.",
    )
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

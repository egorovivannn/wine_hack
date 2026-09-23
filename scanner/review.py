"""Create a reproducible Top-K reference sheet for manual photo labeling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import textwrap

from PIL import Image, ImageDraw, ImageFont, ImageOps

from .catalog import ROOT, sha256_file
from .image import PREPROCESSING_VERSION, decode_image, query_views
from .vision import (
    DEFAULT_INDEX,
    DEFAULT_MANIFEST,
    DEFAULT_MODEL_DIR,
    VisionEncoder,
    load_index,
)


def _thumbnail(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    rgba = image.convert("RGBA")
    white = Image.new("RGBA", rgba.size, "white")
    white.alpha_composite(rgba)
    return ImageOps.contain(white.convert("RGB"), size)


def render_sheet(query: Image.Image, references: list[tuple[Image.Image, str]],
                 output: Path) -> None:
    """Show the query alongside the ranked reference images."""
    columns, width, image_height, text_height = 5, 220, 250, 72
    cell_height = image_height + text_height
    items = [(query, "QUERY")] + references
    rows = (len(items) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * width, rows * cell_height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for index, (image, label) in enumerate(items):
        left = (index % columns) * width
        top = (index // columns) * cell_height
        thumb = _thumbnail(image, (width - 12, image_height - 12))
        sheet.paste(thumb, (left + (width - thumb.width) // 2,
                            top + (image_height - thumb.height) // 2))
        draw.rectangle((left, top, left + width - 1, top + cell_height - 1),
                       outline="#dddddd")
        for line_number, line in enumerate(textwrap.wrap(label, width=30)[:4]):
            draw.text((left + 6, top + image_height + 3 + 15 * line_number),
                      line, fill="black", font=font)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.png")
    sheet.save(temporary)
    temporary.replace(output)


def review(image_path: Path, output_dir: Path, top_k: int,
           manifest_path: Path, index_path: Path, model_dir: Path,
           references_dir: Path) -> tuple[Path, Path]:
    """Rank visual candidates and record the evidence needed to inspect them."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cards = {card["slug"]: card for card in manifest["cards"]}
    images = {item["filename"]: item for item in manifest["images"]}
    gallery = load_index(index_path, manifest_path)
    encoder = VisionEncoder(model_dir)
    query = decode_image(image_path)
    vectors = encoder.encode(list(query_views(query)))
    ranked = gallery.search(vectors, top_k=top_k)

    records = []
    reference_tiles = []
    for rank, candidate in enumerate(ranked, 1):
        path = references_dir / candidate.image_name
        if sha256_file(path) != images[candidate.image_name]["sha256"]:
            raise ValueError(f"Reference changed: {path}")
        card = cards[candidate.slug]
        records.append({
            "rank": rank,
            "slug": candidate.slug,
            "name": card["name"],
            "winery": card["winery"],
            "category": card["category"],
            "grape": card["grape"],
            "score": candidate.score,
            "full_score": candidate.full_score,
            "center_score": candidate.center_score,
            "reference_file": candidate.image_name,
            "reference_sha256": images[candidate.image_name]["sha256"],
        })
        reference_tiles.append((
            decode_image(path),
            f"{rank} {candidate.score:.3f} {candidate.slug}",
        ))

    output_dir.mkdir(parents=True, exist_ok=True)
    query_sha256 = sha256_file(image_path)
    index_sha256 = sha256_file(index_path)
    prefix = f"{image_path.stem}_{query_sha256[:12]}_{index_sha256[:12]}_top{top_k}"
    report_path = output_dir / f"{prefix}.json"
    sheet_path = output_dir / f"{prefix}.png"
    report = {
        "query_file": image_path.name,
        "query_sha256": query_sha256,
        "index_sha256": index_sha256,
        "manifest_sha256": sha256_file(manifest_path),
        "model_weights_sha256": sha256_file(model_dir / "model.safetensors"),
        "preprocessing_version": PREPROCESSING_VERSION,
        "git_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            capture_output=True, check=True).stdout.strip(),
        "git_dirty": bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True,
            capture_output=True, check=True).stdout.strip()),
        "ranking": "visual_similarity_before_ocr",
        "candidates": records,
    }
    temporary = report_path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(report_path)
    render_sheet(query, reference_tiles, sheet_path)
    return report_path, sheet_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "data/evaluation/manual_review")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--references-dir", type=Path,
                        default=ROOT / "data/competition_imgs")
    args = parser.parse_args()
    report, sheet = review(args.image, args.output_dir, args.top_k,
                           args.manifest, args.index, args.model_dir,
                           args.references_dir)
    print(json.dumps({"report": str(report), "sheet": str(sheet)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

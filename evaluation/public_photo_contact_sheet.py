"""Make local reference/query contact sheets for manual photo provenance review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlsplit

from PIL import Image, ImageDraw, ImageFont, ImageOps

from scanner.catalog import ROOT


def make_sheets(
    audit_path: Path, reference_dir: Path, output_dir: Path, per_sheet: int = 20
) -> int:
    rows = [
        json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()
    ]
    rows = [row for row in rows if row.get("local_image")]
    manifest = json.loads(
        (ROOT / "data/catalog_manifest.json").read_text(encoding="utf-8")
    )
    reference = {card["slug"]: card["image_name"] for card in manifest["cards"]}
    output_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    tile_w, tile_h = 370, 285
    for start in range(0, len(rows), per_sheet):
        batch = rows[start : start + per_sheet]
        sheet = Image.new("RGB", (tile_w * 4, tile_h * 5), "#f4f4f4")
        draw = ImageDraw.Draw(sheet)
        for offset, row in enumerate(batch):
            x = offset % 4 * tile_w
            y = offset // 4 * tile_h
            ref = Image.open(reference_dir / reference[row["slug"]]).convert("RGB")
            candidate = Image.open(row["local_image"]).convert("RGB")
            sheet.paste(ImageOps.contain(ref, (165, 230)), (x + 10, y + 42))
            sheet.paste(ImageOps.contain(candidate, (165, 230)), (x + 190, y + 42))
            label = f"{start + offset + 1}: {row['slug'][:29]}"
            source = urlsplit(row["page_url"]).hostname or ""
            draw.text((x + 8, y + 3), label, font=font, fill="black")
            draw.text(
                (x + 8, y + 22),
                f"{source[:29]} ph={row['phash_hamming']}",
                font=font,
                fill="black",
            )
            draw.rectangle((x, y, x + tile_w - 1, y + tile_h - 1), outline="#999999")
        sheet.save(output_dir / f"sheet_{start // per_sheet + 1:02}.jpg", quality=90)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit",
        type=Path,
        default=ROOT / "data/external/catalog_photo_pilot_audit.jsonl",
    )
    parser.add_argument(
        "--references", type=Path, default=ROOT / "data/competition_imgs"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/external/catalog_photo_pilot_sheets"
    )
    args = parser.parse_args()
    print(
        f"Contact sheet images: {make_sheets(args.audit, args.references, args.output)}"
    )


if __name__ == "__main__":
    main()

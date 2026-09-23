"""Verify GRAIN Wine Labels and prepare a session-disjoint YOLO dataset."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import random
import re
from zipfile import ZipFile

from PIL import Image, ImageOps
import yaml


EXPECTED_MD5 = "65b2ce83492f7c77249b7d852d8fa1c9"


def digest(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def session(name: str) -> str:
    """Conservative source session: day if known; otherwise camera filename block."""
    stem = name.split(".rf.")[0].lower()
    date = re.search(r"20\d{6}", stem)
    if date:
        return date.group()
    whatsapp = re.search(r"whatsapp.image.(20\d{2}.\d{2}.\d{2})", stem)
    if whatsapp:
        return whatsapp.group(1)
    img = re.search(r"img[_-](\d+)", stem)
    if img:
        return "img_block_" + str(int(img.group(1)) // 100)
    return stem.split("_jpg")[0].split("_jpeg")[0]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, default=Path("data/external/grain/Wine_Labels.zip"))
    p.add_argument("--output", type=Path, default=Path("data/external/grain/yolo"))
    p.add_argument("--max-side", type=int, default=1280)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--reuse", type=Path, help="Reuse validated resized images from a previous preparation")
    args = p.parse_args()
    if digest(args.archive, "md5") != EXPECTED_MD5:
        p.error("GRAIN archive MD5 mismatch")
    if args.output.exists():
        p.error("Output already exists; choose another directory")
    records = []
    previous = None
    if args.reuse:
        previous_provenance = json.loads((args.reuse / "provenance.json").read_text())
        if previous_provenance["archive_md5"] != EXPECTED_MD5 or previous_provenance["max_side"] != args.max_side:
            p.error("Reused images were prepared from different source/settings")
        if digest(args.reuse / "manifest.jsonl", "sha256") != previous_provenance["manifest_sha256"]:
            p.error("Reused split manifest changed")
        previous = [json.loads(line) for line in (args.reuse / "manifest.jsonl").read_text().splitlines()]
    with ZipFile(args.archive) as z:
        names = set(z.namelist())
        if any(PurePosixPath(n).is_absolute() or ".." in PurePosixPath(n).parts for n in names):
            p.error("Unsafe ZIP entry")
        for lighting in ("Flash", "Good Lighting", "Low Lighting"):
            prefix = f"Wine Labels/{lighting}/"
            coco = json.loads(z.read(prefix + "annotations/instances.json"))
            by_image = defaultdict(list)
            for ann in coco["annotations"]:
                if ann["category_id"] != 1 or ann.get("iscrowd"):
                    continue
                by_image[ann["image_id"]].append(ann["bbox"])
            for image in coco["images"]:
                source = prefix + "images/" + image["file_name"]
                if source not in names:
                    raise FileNotFoundError(source)
                records.append(dict(source=source, lighting=lighting, image=image,
                                    boxes=by_image[image["id"]],
                                    session=session(image["file_name"])))
        groups = defaultdict(list)
        for row in records:
            groups[row["session"]].append(row)
        rng = random.Random(args.seed)
        keys = sorted(groups)
        rng.shuffle(keys)
        total = len(records)
        targets = {"test": .10 * total, "val": .10 * total}
        sizes = Counter()
        assignments = {}
        for key in keys:
            split = min(targets, key=lambda s: sizes[s] / targets[s]) if any(sizes[s] < targets[s] for s in targets) else "train"
            # Do not let one huge source session consume almost the entire holdout.
            if split != "train" and len(groups[key]) > targets[split] * .6:
                split = "train"
            assignments[key] = split
            sizes[split] += len(groups[key])
        if not all(sizes[s] for s in ("train", "val", "test")):
            raise ValueError(f"Empty split: {sizes}")
        args.output.mkdir(parents=True)
        manifest = []
        ordered = sorted(records, key=lambda r: r["source"])
        if previous and [r["source"] for r in previous] != [r["source"] for r in ordered]:
            p.error("Reused source ordering differs")
        for index, row in enumerate(ordered):
            split = assignments[row["session"]]
            image_path = args.output / "images" / split / f"{index:05d}.jpg"
            label_path = args.output / "labels" / split / f"{index:05d}.txt"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.parent.mkdir(parents=True, exist_ok=True)
            if previous:
                old_split = previous[index]["split"]
                os.link(args.reuse / "images" / old_split / image_path.name, image_path)
                os.link(args.reuse / "labels" / old_split / label_path.name, label_path)
                label_count = previous[index]["boxes"]
            else:
                with z.open(row["source"]) as stream:
                    im = ImageOps.exif_transpose(Image.open(stream)).convert("RGB")
                    im.load()
                w, h = im.size
                if (w, h) != (row["image"]["width"], row["image"]["height"]):
                    raise ValueError(f"Annotation/image dimensions disagree: {row['source']}")
                im.thumbnail((args.max_side, args.max_side), Image.Resampling.LANCZOS)
                im.save(image_path, quality=90)
                labels = []
                for x, y, bw, bh in row["boxes"]:
                    x1, y1 = max(0, x), max(0, y)
                    x2, y2 = min(w, x + bw), min(h, y + bh)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    labels.append(f"0 {(x1+x2)/(2*w):.7f} {(y1+y2)/(2*h):.7f} {(x2-x1)/w:.7f} {(y2-y1)/h:.7f}")
                label_path.write_text("\n".join(labels) + ("\n" if labels else ""))
                label_count = len(labels)
            manifest.append(dict(source=row["source"], session=row["session"],
                                 split=split, boxes=label_count))
    yaml_path = args.output / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(dict(path=str(args.output.resolve()),
        train="images/train", val="images/val", test="images/test",
        names={0: "wine-labels"}), sort_keys=False))
    (args.output / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in manifest))
    report = dict(source_url="https://zenodo.org/records/16410628", archive_md5=EXPECTED_MD5,
                  archive_sha256=digest(args.archive, "sha256"), seed=args.seed,
                  max_side=args.max_side, images=dict(Counter(r["split"] for r in manifest)),
                  sessions=dict(Counter(assignments.values())),
                  manifest_sha256=digest(args.output / "manifest.jsonl", "sha256"))
    (args.output / "provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

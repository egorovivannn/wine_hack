"""Verify the final manifest, class separation and decoded crop dimensions."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps, ImageDraw

from .core import ROOT, validate_manifest
from .prepare import sha_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv', type=Path, default=ROOT / 'data/winesensed_embedding/dataset.csv')
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    frame = pd.read_csv(args.csv)
    validate_manifest(frame)
    assert frame.groupby('label').wine_id.nunique().max() == 1
    assert frame.groupby('wine_id').label.nunique().max() == 1
    assert (frame.wine_id == 'winesensed:' + frame.vintage_id.astype(str)).all()
    assert frame.image_path.map(lambda p: Path(p).is_absolute()).all()
    counts = frame.groupby('split').label.nunique().to_dict()
    expected = round(frame.label.nunique() * .05)
    assert counts['val'] == counts['test'] == expected
    assert (frame[frame.split != 'train'].groupby('label').size() >= 2).all()
    failures = []

    def check(row):
        try:
            with Image.open(row.image_path) as im:
                im.load()
                if im.size != (row.width, row.height):
                    return dict(image_path=row.image_path, error=f'Wrong dimensions: {im.size}')
        except Exception as exc:
            return dict(image_path=row.image_path, error=str(exc))
        return None

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for offset in range(0, len(frame), 4096):
            failures.extend(x for x in pool.map(check, frame.iloc[offset:offset+4096].itertuples()) if x)
            if offset % (4096*10) == 0:
                print(f'Decoded {min(offset+4096,len(frame)):,}/{len(frame):,}', flush=True)
    report = dict(dataset_sha256=sha_file(args.csv), images=len(frame), classes=frame.label.nunique(),
                  class_splits=counts, image_splits=frame.split.value_counts().to_dict(),
                  class_overlap=0, duplicate_paths=0, duplicate_source_hashes=0,
                  duplicate_crop_hashes=0, missing_positive_queries=0,
                  image_decode_errors=len(failures), errors=failures,
                  protocol='class-disjoint; all-vs-all within val and within test; self excluded')
    (args.csv.parent / 'audit.json').write_text(json.dumps(report, indent=2))
    if failures:
        raise ValueError(f'{len(failures)} invalid images; see audit.json')
    # Fixed, representative sample for human inspection; not a quality score.
    samples = pd.concat([part.sample(min(8, len(part)), random_state=42)
                         for _, part in frame.groupby('split')]).reset_index(drop=True)
    canvas = Image.new('RGB', (6*180, 4*240), '#f2f2f2')
    draw = ImageDraw.Draw(canvas)
    for i, row in enumerate(samples.itertuples()):
        x, y = (i % 6)*180, (i // 6)*240
        with Image.open(row.image_path) as im:
            thumb = ImageOps.contain(im.convert('RGB'), (174, 212))
            canvas.paste(thumb, (x+(180-thumb.width)//2, y+25))
        draw.text((x+3, y+3), f'{row.split} / {row.vintage_id}', fill='black')
    canvas.save(args.csv.parent / 'crop_samples.jpg', quality=92)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

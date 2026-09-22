"""Build a class-disjoint manifest from an existing WineSensed crop export."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sqlite3
import time

import cv2
import numpy as np
import pandas as pd

from .prepare import ROOT, assign_splits, sha_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--confidence', type=float, default=.5)
    p.add_argument('--min-size', type=int, default=24)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--workers', type=int, default=12)
    args = p.parse_args()
    args.input, args.output = args.input.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    metadata_path = ROOT / 'data/WineSensed/metadata/images_reviews_attributes.csv'
    config = dict(input=str(args.input), crops_csv_sha256=sha_file(args.input / 'crops.csv'),
                  metadata_sha256=sha_file(metadata_path), confidence=args.confidence,
                  min_size=args.min_size, seed=args.seed, policy='one detection per source; exact dedup; class-disjoint 90/5/5')
    config_path = args.output / 'import_config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Import settings changed; choose a new output directory.')
    config_path.write_text(json.dumps(config, indent=2))
    metadata = pd.read_csv(metadata_path, usecols=['image', 'vintage_id'], dtype=str).dropna()
    metadata['source_id'] = metadata.image.str.rsplit('/').str[-1]
    conflicts = set(metadata.groupby('source_id').vintage_id.nunique().loc[lambda x: x > 1].index)
    mapping = metadata.drop_duplicates('source_id').set_index('source_id').vintage_id
    frame = pd.read_csv(args.input / 'crops.csv')
    total = len(frame)
    frame['source_id'] = frame.source_path.str.rsplit('/').str[-1]
    frame['vintage_id'] = frame.source_id.map(mapping)
    frame['reason'] = ''
    frame.loc[frame.source_path.duplicated(False), 'reason'] = 'multiple_detections'
    frame.loc[frame.vintage_id.isna(), 'reason'] = 'missing_metadata'
    frame.loc[frame.source_id.isin(conflicts), 'reason'] = 'conflicting_metadata'
    frame.loc[(frame.reason == '') & (frame.confidence < args.confidence), 'reason'] = 'low_confidence'
    rejected = frame[frame.reason != ''].copy()
    frame = frame[frame.reason == ''].copy()
    frame['image_path'] = frame.crop_path.map(lambda x: str(Path(x).resolve()))
    frame['item_id'] = frame.image_path.map(lambda x: hashlib.sha256(x.encode()).hexdigest()[:24])
    frame['wine_id'] = 'winesensed:' + frame.vintage_id
    frame['bbox'] = frame.crop_xyxy
    db = sqlite3.connect(args.output / 'verified.sqlite')
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS verified (item_id TEXT PRIMARY KEY, status TEXT, source_sha256 TEXT, crop_sha256 TEXT, width INTEGER, height INTEGER)')
    done = {row[0] for row in db.execute('SELECT item_id FROM verified')}
    pending = frame[~frame.item_id.isin(done)]

    def check(row):
        try:
            crop = cv2.imdecode(np.frombuffer(Path(row.image_path).read_bytes(), np.uint8), cv2.IMREAD_COLOR)
            if crop is None:
                raise ValueError('Cannot decode crop')
            h, w = crop.shape[:2]
            x1, y1, x2, y2 = json.loads(row.bbox)
            if (w, h) != (x2-x1, y2-y1):
                return row.item_id, 'bbox_size_mismatch', '', '', w, h
            if min(w, h) < args.min_size:
                return row.item_id, 'crop_too_small', '', '', w, h
            source_hash = hashlib.sha256(Path(row.source_path).read_bytes()).hexdigest()
            crop_hash = hashlib.sha256(str(crop.shape).encode() + crop.tobytes()).hexdigest()
            return row.item_id, 'ok', source_hash, crop_hash, w, h
        except (OSError, ValueError, cv2.error):
            return row.item_id, 'unreadable_image_or_source', '', '', 0, 0

    print(f'{total:,} exported crops; {len(frame):,} candidates; {len(pending):,} to verify', flush=True)
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for offset in range(0, len(pending), 2048):
            rows = list(pool.map(check, pending.iloc[offset:offset+2048].itertuples()))
            db.executemany('INSERT INTO verified VALUES (?,?,?,?,?,?)', rows)
            db.commit()
            if offset % (2048*10) == 0 or offset+2048 >= len(pending):
                print(f'Verified {min(offset+2048,len(pending)):,}/{len(pending):,}; {time.monotonic()-started:.0f}s', flush=True)
    verified = pd.read_sql_query('SELECT * FROM verified', db)
    db.close()
    frame = frame.merge(verified, on='item_id', validate='one_to_one')
    bad = frame.status != 'ok'
    frame.loc[bad, 'reason'] = frame.loc[bad, 'status']
    rejected = pd.concat([rejected, frame[bad]], ignore_index=True)
    frame = frame[~bad].sort_values('image_path').copy()
    for column in ('source_sha256', 'crop_sha256'):
        conflict = set(frame.groupby(column).wine_id.nunique().loc[lambda x: x > 1].index)
        mask = frame[column].isin(conflict)
        frame.loc[mask, 'reason'] = 'conflicting_identity_' + column
        rejected = pd.concat([rejected, frame[mask]], ignore_index=True)
        frame = frame[~mask].copy()
        mask = frame.duplicated(column)
        frame.loc[mask, 'reason'] = 'duplicate_' + column
        rejected = pd.concat([rejected, frame[mask]], ignore_index=True)
        frame = frame[~mask].copy()
    frame['label'] = frame.wine_id.map({wine: i for i, wine in enumerate(sorted(frame.wine_id.unique()))})
    frame = assign_splits(frame, args.seed)
    assert len(frame)+len(rejected) == total
    assert (frame.groupby('label').split.nunique() == 1).all()
    front = ['image_path', 'label', 'split', 'wine_id', 'vintage_id', 'source_path', 'source_id',
             'item_id', 'confidence', 'bbox', 'width', 'height', 'source_sha256', 'crop_sha256']
    frame = frame[front]
    temp = args.output / 'dataset.csv.tmp'
    frame.to_csv(temp, index=False)
    temp.replace(args.output / 'dataset.csv')
    frame[['label', 'wine_id', 'vintage_id', 'split']].drop_duplicates().to_csv(args.output / 'labels.csv', index=False)
    rejected.to_csv(args.output / 'rejected.csv', index=False)
    report = dict(input_crops=total, images=len(frame), classes=frame.label.nunique(),
                  image_splits=frame.split.value_counts().to_dict(),
                  class_splits=frame.groupby('split').label.nunique().to_dict(),
                  rejected=rejected.reason.value_counts().to_dict(), seed=args.seed,
                  singleton_wines=int((frame.groupby('label').size() == 1).sum()),
                  dataset_sha256=sha_file(args.output / 'dataset.csv'),
                  decoded_and_hashed_all_accepted=True, class_overlap=0,
                  protocol='90/5/5 by class; singleton classes train only; val-vs-val and test-vs-test, self excluded')
    (args.output / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()

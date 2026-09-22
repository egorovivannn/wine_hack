"""Index metadata, detect unambiguous bottles, deduplicate, split. Resumable SQLite journal."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / 'data/winesensed_embedding'


def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def index_sources(args):
    out = args.output
    index_path = out / 'sources.csv'
    index_report = out / 'index_report.json'
    if index_path.exists() and index_report.exists():
        if json.loads(index_report.read_text())['limit'] != args.limit:
            raise ValueError('--limit differs from the existing index; use a different output directory.')
        return index_path
    print('Indexing metadata and local JPEGs...', flush=True)
    records, rejects = [], []
    root = args.data / 'WineSensed'
    meta = root / 'metadata/images_reviews_attributes.csv'
    df = pd.read_csv(meta, usecols=['vintage_id', 'image'], dtype=str).dropna()
    df['file'] = df.image.str.rsplit('/').str[-1]
    conflicts = set(df.groupby('file').vintage_id.nunique().loc[lambda x: x > 1].index)
    mapping = df.drop_duplicates('file').set_index('file').vintage_id.to_dict()
    seen = set()
    for path in sorted(root.glob('chunk_*/output_images/chunk_*/*.jpg')):
        if path.name in conflicts:
            rejects.append((str(path), 'metadata_conflicting_vintage_ids'))
        elif path.name not in mapping:
            rejects.append((str(path), 'missing_metadata'))
        else:
            vintage = mapping[path.name]
            records.append(dict(source_path=str(path.resolve()), source='winesensed',
                                source_id=path.name, source_key=f'winesensed:{vintage}',
                                wine_name='', vintage_id=vintage))
        seen.add(path.name)
    for name in sorted(set(mapping) - seen):
        rejects.append((name, 'missing_source_file'))
    indexed = pd.DataFrame(records)
    indexed['item_id'] = indexed.source_path.map(lambda s: hashlib.sha256(s.encode()).hexdigest()[:24])
    if args.limit:
        indexed = indexed.sample(min(args.limit, len(indexed)), random_state=args.seed)
    indexed = indexed.sort_values(['source', 'source_id']).reset_index(drop=True)
    temporary_index = index_path.with_suffix('.csv.tmp')
    indexed.to_csv(temporary_index, index=False)
    temporary_index.replace(index_path)
    pd.DataFrame(rejects, columns=['source_path', 'reason']).to_csv(out / 'metadata_rejected.csv', index=False)
    (out / 'index_report.json').write_text(json.dumps(dict(
        indexed=len(indexed), metadata_rejected=len(rejects),
        rejection_reasons=dict(Counter(x[1] for x in rejects)),
        winesensed_metadata_sha256=sha_file(meta),
        limit=args.limit, identity='WineSensed vintage_id: distinct vintages remain distinct wines',
    ), indent=2))
    print(f'Indexed {len(indexed):,}; metadata rejected {len(rejects):,}', flush=True)
    return index_path


def connect(out):
    db = sqlite3.connect(out / 'detections.sqlite')
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('''CREATE TABLE IF NOT EXISTS detections (
        item_id TEXT PRIMARY KEY, status TEXT, image_path TEXT, source_sha256 TEXT,
        crop_sha256 TEXT, bbox TEXT, confidence REAL, width INTEGER, height INTEGER)''')
    return db


def central_box_index(boxes, width, height):
    """Nearest bbox center to image center, with distances normalized by image size."""
    boxes = np.asarray(boxes, dtype=float)
    if boxes.ndim != 2 or boxes.shape[1] != 4 or not len(boxes):
        raise ValueError('Expected a nonempty Nx4 array of boxes')
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    distance = ((centers / np.array([width, height]) - .5) ** 2).sum(axis=1)
    return int(np.argmin(distance))


def detect(args, index_path):
    (args.output / 'ultralytics').mkdir(exist_ok=True)
    os.environ.setdefault('YOLO_CONFIG_DIR', str(args.output / 'ultralytics'))
    import cv2
    import torch
    from ultralytics import YOLO
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    if args.device != 'cpu' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Run on the GPU host or explicitly pass --device cpu.')
    config = dict(weights=str(args.detector), weights_sha256=sha_file(args.detector),
                  conf=args.conf, imgsz=args.imgsz, min_size=args.min_size,
                  policy=f"{getattr(args, 'selection', 'exactly_one_detection')}; floor/ceil clipped xyxy; no added margin; JPEG quality 95",
                  sources_sha256=sha_file(index_path))
    config_path = args.output / 'crop_config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Crop settings or sources changed: use a new --output directory.')
    config_path.write_text(json.dumps(config, indent=2))
    db = connect(args.output)
    done = {r[0] for r in db.execute('SELECT item_id FROM detections')}
    df = pd.read_csv(index_path, dtype=str, keep_default_na=False)
    pending = df.loc[~df.item_id.isin(done), ['item_id', 'source_path']].values.tolist()
    model = YOLO(str(args.detector))
    if len(model.names) != 1 or str(model.names[0]).lower() not in ('bottle', 'wine', 'product', 'wine_bottle', 'wine-labels'):
        raise ValueError(f'Expected the local single-class bottle detector, got {model.names}')
    print(f'Detector classes {model.names}; pending {len(pending):,}; already journaled {len(done):,}', flush=True)

    def read(row):
        item, path = row
        try:
            content = Path(path).read_bytes()
            im = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
            if im is None:
                raise ValueError('decode failed')
            return item, im, hashlib.sha256(content).hexdigest()
        except (OSError, ValueError):
            return item, None, ''

    def save(entry):
        item, im, digest, boxes, scores = entry
        empty = (item, '', '', digest, '', '', 0., 0, 0)
        if im is None:
            return (item, 'decode_error', *empty[2:])
        if len(boxes) > 1 and getattr(args, 'selection', '') == 'highest_confidence':
            best = int(np.argmax(scores))
            boxes, scores = boxes[best:best+1], scores[best:best+1]
        if len(boxes) > 1 and getattr(args, 'selection', '') == 'central_box':
            best = central_box_index(boxes, im.shape[1], im.shape[0])
            boxes, scores = boxes[best:best+1], scores[best:best+1]
        if len(boxes) != 1:
            return (item, 'no_bottle' if not len(boxes) else 'multiple_bottles', *empty[2:])
        x1, y1, x2, y2 = boxes[0]
        h, w = im.shape[:2]
        x1, y1, x2, y2 = max(0, math.floor(x1)), max(0, math.floor(y1)), min(w, math.ceil(x2)), min(h, math.ceil(y2))
        if min(x2 - x1, y2 - y1) < args.min_size:
            return (item, 'crop_too_small', *empty[2:])
        crop = np.ascontiguousarray(im[y1:y2, x1:x2])
        crop_hash = hashlib.sha256(str(crop.shape).encode() + crop.tobytes()).hexdigest()
        dest = args.output / 'crops' / item[:2] / f'{item}.jpg'
        dest.parent.mkdir(parents=True, exist_ok=True)
        temp = dest.with_suffix('.tmp.jpg')
        if not cv2.imwrite(str(temp), crop, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise OSError(f'Could not write {temp}')
        temp.replace(dest)
        return (item, 'ok', str(dest.resolve()), digest, crop_hash, json.dumps([x1, y1, x2, y2]), float(scores[0]), x2-x1, y2-y1)

    start = time.monotonic()
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        def read_batch(offset):
            return list(pool.map(read, pending[offset:offset + args.batch_size]))
        # Separate coordinator avoids deadlocking the worker pool while prefetching.
        with ThreadPoolExecutor(max_workers=1) as prefetch:
            future = prefetch.submit(read_batch, 0)
            for offset in range(0, len(pending), args.batch_size):
                batch = future.result()
                future = prefetch.submit(read_batch, offset + args.batch_size)
                valid = [row for row in batch if row[1] is not None]
                predictions = model.predict([row[1] for row in valid], device=args.device,
                    batch=args.batch_size, imgsz=args.imgsz, conf=args.conf, half=args.device != 'cpu',
                    rect=False, verbose=False, max_det=10) if valid else []
                packed = [(row[0], row[1], row[2], result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy())
                          for row, result in zip(valid, predictions, strict=True)]
                packed += [(item, im, digest, [], []) for item, im, digest in batch if im is None]
                rows = list(pool.map(save, packed))
                db.executemany('INSERT INTO detections VALUES (?,?,?,?,?,?,?,?,?)', rows)
                db.commit()
                completed += len(batch)
                if completed % (args.batch_size * 20) == 0 or completed == len(pending):
                    elapsed = time.monotonic() - start
                    print(f'{completed:,}/{len(pending):,} | {completed/max(elapsed,1):.1f} images/s | '
                          f'ETA {(len(pending)-completed)*elapsed/max(completed,1)/60:.1f} min', flush=True)
    counts = dict(db.execute('SELECT status, count(*) FROM detections GROUP BY status'))
    db.close()
    print(json.dumps(counts, indent=2), flush=True)


def assign_splits(df, seed):
    """90/5/5 by wine class, class-disjoint; held-out classes need >=2 real images."""
    rng = np.random.default_rng(seed)
    df = df.sort_values(['label', 'item_id']).reset_index(drop=True).copy()
    counts = df.groupby('label').size()
    candidates = rng.permutation(counts[counts >= 2].index.to_numpy())
    nval = ntest = int(round(len(counts) * .05))
    if nval < 1:
        raise ValueError('Need at least 11 wine classes for nonempty 5% validation and test splits.')
    if len(candidates) < nval + ntest:
        raise ValueError('Not enough multi-image wine classes for 90/5/5 class-disjoint retrieval.')
    df['split'] = 'train'
    df.loc[df.label.isin(candidates[:nval]), 'split'] = 'val'
    df.loc[df.label.isin(candidates[nval:nval+ntest]), 'split'] = 'test'
    return df


def finalize(args, index_path):
    df = pd.read_csv(index_path, dtype=str, keep_default_na=False)
    db = connect(args.output)
    det = pd.read_sql_query('SELECT * FROM detections', db)
    db.close()
    if len(det) != len(df) or set(det.item_id) != set(df.item_id):
        raise ValueError(f'Detection incomplete: {len(det):,}/{len(df):,}; resume crop stage first.')
    df = df.merge(det, on='item_id', validate='one_to_one')
    rejected = df[df.status != 'ok'].copy()
    df = df[df.status == 'ok'].copy()
    weak = df.confidence < args.crop_confidence
    low_confidence = df[weak].copy()
    low_confidence['status'] = 'low_crop_confidence'
    rejected = pd.concat([rejected, low_confidence], ignore_index=True)
    df = df[~weak].copy()
    df['wine_id'] = df.source_key
    # Exact duplicate images carrying contradictory identities are unusable supervision.
    for column in ['source_sha256', 'crop_sha256']:
        conflicts = set(df.groupby(column).wine_id.nunique().loc[lambda x: x > 1].index)
        bad = df[column].isin(conflicts)
        conflict = df[bad].copy()
        conflict['status'] = f'conflicting_identity_{column}'
        rejected = pd.concat([rejected, conflict], ignore_index=True)
        df = df[~bad].copy()
        duplicate = df.duplicated(column)
        dup = df[duplicate].copy()
        dup['status'] = f'duplicate_{column}'
        rejected = pd.concat([rejected, dup], ignore_index=True)
        df = df[~duplicate].copy()
    keys = sorted(df.wine_id.unique())
    labels = {key: i for i, key in enumerate(keys)}
    df['label'] = df.wine_id.map(labels)
    df = assign_splits(df, args.seed)
    if len(df) + len(rejected) != len(det):
        raise AssertionError('Accepted and rejected rows do not account for every source image')
    front = ['image_path', 'label', 'split', 'wine_id', 'source', 'source_id', 'source_key', 'wine_name', 'vintage_id']
    df = df[front + [x for x in df if x not in front]]
    temp = args.output / 'dataset.csv.tmp'
    df.to_csv(temp, index=False)
    temp.replace(args.output / 'dataset.csv')
    df[front[1:2] + ['wine_id', 'source_key', 'wine_name', 'vintage_id']].drop_duplicates().to_csv(args.output / 'labels.csv', index=False)
    rejected.to_csv(args.output / 'rejected.csv', index=False)
    counts = df.groupby('label').size()
    report = dict(source_images_processed=len(det),
                  metadata_exclusions=json.loads((args.output / 'index_report.json').read_text())['metadata_rejected'],
                  images=len(df), wines=len(keys), singleton_wines=int((counts == 1).sum()),
                  splits=df.split.value_counts().to_dict(),
                  proportions=df.split.value_counts(normalize=True).to_dict(),
                  by_source_split=df.groupby(['source', 'split']).size().unstack(fill_value=0).to_dict(),
                  rejected=rejected.status.value_counts().to_dict(),
                  class_splits=df.groupby('split').label.nunique().to_dict(),
                  class_proportions=(df.groupby('split').label.nunique()/len(keys)).to_dict(),
                  seed=args.seed, crop_confidence=args.crop_confidence,
                  identity_limitation='Labels follow WineSensed vintage_id; different vintages remain distinct. Dataset annotation noise may remain. Exact conflicting duplicates excluded, not guessed.',
                  protocol='class-disjoint 90/5/5. Val against all val; test against all test; self-match excluded. Held-out wines have >=2 distinct images, singletons train only. Exact duplicate source/crop hashes removed before split. Near duplicates not automatically resolved.')
    (args.output / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=ROOT / 'data')
    p.add_argument('--output', type=Path, default=DEFAULT_OUT)
    p.add_argument('--detector', type=Path, default=ROOT / 'detection/runs/grain_yolo26s/weights/best.pt')
    p.add_argument('--stage', choices=['all', 'index', 'crop', 'finalize'], default='all')
    p.add_argument('--device', default='0')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--conf', type=float, default=.25)
    p.add_argument('--crop-confidence', type=float, default=.5,
                   help='Final CSV confidence floor; detection still uses --conf to find extra bottles.')
    p.add_argument('--min-size', type=int, default=24)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--limit', type=int, default=0, help='Smoke test only; use a separate output directory.')
    args = p.parse_args()
    args.output = args.output.resolve()
    args.data = args.data.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    index = index_sources(args)
    if args.stage in ('all', 'crop'):
        detect(args, index)
    if args.stage in ('all', 'finalize'):
        finalize(args, index)


if __name__ == '__main__':
    main()

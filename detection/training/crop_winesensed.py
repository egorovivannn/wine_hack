"""Resumable full WineSensed detection. Save ALL boxes, journal every source."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import time

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic_json(path, data):
    tmp = path.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(data, indent=2), encoding='utf-8')
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'data/winesensed_crops_yolo26l')
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--limit', type=int, default=0, help='Pilot only; zero processes everything')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out/'ultralytics').mkdir(exist_ok=True)
    os.environ['YOLO_CONFIG_DIR'] = str(out/'ultralytics')
    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    source_root = ROOT/'data/WineSensed'
    weights = ROOT/'detection/runs/grain_yolo26l-2/weights/best.pt'
    config = dict(weights=str(weights), weights_sha256=digest(weights), source_root=str(source_root),
                  conf=.25, imgsz=640, max_det=300, jpeg_quality=95, limit=args.limit,
                  policy='all detections; all source images; no identity/dedup/confidence-0.5/min-size filtering',
                  bbox='floor/ceil xyxy, clipped, right/bottom exclusive; OpenCV orientation')
    if (out/'config.json').exists() and json.loads((out/'config.json').read_text()) != config:
        raise ValueError('Configuration differs from previous run; use a new output directory')
    atomic_json(out/'config.json', config)
    # Enumerate physical sources, including the 32 ambiguous metadata identities.
    sources = sorted(p for p in source_root.glob('chunk_*/output_images/chunk_*/*')
                     if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'})
    if args.limit:
        # Uniform coverage of chunks for a representative pilot.
        sources = [sources[i] for i in np.linspace(0, len(sources)-1, min(args.limit, len(sources)), dtype=int)]
    if not sources:
        raise RuntimeError('No source images found')
    source_hash = hashlib.sha256('\n'.join(str(p) for p in sources).encode()).hexdigest()
    index = dict(images=len(sources), source_paths_sha256=source_hash)
    if (out/'index.json').exists() and json.loads((out/'index.json').read_text()) != index:
        raise ValueError('Source inventory changed since previous run')
    atomic_json(out/'index.json', index)
    db = sqlite3.connect(out/'detections.sqlite')
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS images(source_path TEXT PRIMARY KEY, status TEXT, width INT, height INT, detections INT, saved INT, error TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS crops(source_path TEXT, box_index INT, crop_path TEXT UNIQUE, xyxy TEXT, crop_xyxy TEXT, confidence REAL, PRIMARY KEY(source_path,box_index))')
    done = {r[0] for r in db.execute('SELECT source_path FROM images')}
    pending = [p for p in sources if str(p) not in done]
    model = YOLO(str(weights))
    if model.names != {0: 'wine_bottle'}:
        raise ValueError(f'Unexpected classes: {model.names}')
    print(f'Model: {weights}; classes={model.names}; sources={len(sources):,}; pending={len(pending):,}', flush=True)
    counts = Counter(dict(db.execute('SELECT status,count(*) FROM images GROUP BY status')))
    saved = db.execute('SELECT count(*) FROM crops').fetchone()[0]
    started = time.monotonic()
    processed = 0
    torch.cuda.reset_peak_memory_stats()

    def read(path):
        try:
            im = cv2.imread(str(path))
            return path, im, '' if im is not None else 'OpenCV decode failed'
        except Exception as error:
            return path, None, str(error)

    def save(entry):
        path, im, boxes, scores, error = entry
        if im is None:
            return (str(path), 'decode_error', 0, 0, 0, 0, error), []
        h, w = im.shape[:2]
        key = hashlib.sha256(str(path.relative_to(source_root)).encode()).hexdigest()[:24]
        rows = []
        for i, (box, score) in enumerate(zip(boxes, scores, strict=True)):
            if not np.isfinite(box).all():
                raise ValueError(f'Nonfinite detection: {path}')
            x1, y1, x2, y2 = box
            exact = [max(0,math.floor(x1)),max(0,math.floor(y1)),min(w,math.ceil(x2)),min(h,math.ceil(y2))]
            if exact[2] <= exact[0] or exact[3] <= exact[1]:
                continue
            dest = out/'crops'/key[:2]/f'{key}_{i:03d}.jpg'
            dest.parent.mkdir(parents=True,exist_ok=True)
            tmp = dest.with_suffix('.tmp.jpg')
            crop = im[exact[1]:exact[3],exact[0]:exact[2]]
            if not cv2.imwrite(str(tmp), crop, [cv2.IMWRITE_JPEG_QUALITY,95]):
                raise OSError(f'Could not save crop: {tmp}')
            tmp.replace(dest)
            rows.append((str(path),i,str(dest),json.dumps([float(v) for v in box]),json.dumps(exact),float(score)))
        status = 'ok' if rows else ('invalid_boxes' if len(boxes) else 'no_bottle')
        return (str(path), status,w,h,len(boxes),len(rows),''),rows

    def report(complete=False):
        elapsed = time.monotonic()-started
        progress = dict(complete=complete, total_images=len(sources), processed_images=len(done)+processed,
                        saved_crops=saved, status_counts=dict(counts), elapsed_seconds=elapsed,
                        images_per_second=processed/max(elapsed,1),
                        eta_minutes=(len(pending)-processed)*elapsed/max(processed,1)/60,
                        peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                        peak_gpu_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        atomic_json(out/'progress.json',progress)
        print(json.dumps(progress),flush=True)
        return progress

    with ThreadPoolExecutor(max_workers=args.workers) as pool, ThreadPoolExecutor(max_workers=1) as prefetch:
        def read_batch(offset):
            return list(pool.map(read,pending[offset:offset+args.batch]))
        future = prefetch.submit(read_batch,0)
        for offset in range(0,len(pending),args.batch):
            batch = future.result()
            future = prefetch.submit(read_batch,offset+args.batch)
            valid = [r for r in batch if r[1] is not None]
            predictions = model.predict([r[1] for r in valid], device=0,batch=args.batch,imgsz=640,
                        conf=.25,quantize=16,rect=False,max_det=300,verbose=False) if valid else []
            entries = [(p,im,r.boxes.xyxy.cpu().numpy(),r.boxes.conf.cpu().numpy(),'')
                       for (p,im,_),r in zip(valid,predictions,strict=True)]
            entries += [(p,im,[],[],err) for p,im,err in batch if im is None]
            for image_row, crop_rows in pool.map(save,entries):
                db.execute('INSERT INTO images VALUES (?,?,?,?,?,?,?)',image_row)
                db.executemany('INSERT INTO crops VALUES (?,?,?,?,?,?)',crop_rows)
                counts[image_row[1]] += 1
                saved += len(crop_rows)
            db.commit()
            processed += len(batch)
            if processed % (args.batch*100) == 0 or processed == len(pending):
                report()
    # Flat manifests make every crop and every no-detection/error auditable.
    for table in ['images','crops']:
        cursor = db.execute(f'SELECT * FROM {table} ORDER BY source_path')
        tmp = out/f'{table}.csv.tmp'
        with tmp.open('w',newline='',encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow([c[0] for c in cursor.description])
            writer.writerows(cursor)
        tmp.replace(out/f'{table}.csv')
    assert sum(counts.values()) == len(sources)
    atomic_json(out/'report.json',report(complete=True))
    db.close()
    print('CROPPING_COMPLETE',out,flush=True)


if __name__ == '__main__':
    main()

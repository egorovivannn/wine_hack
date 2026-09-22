"""Resumeable label crops and embeddings for data/competition_imgs."""
import argparse
import csv
import json
import math
import sqlite3
from pathlib import Path

import cv2
import numpy as np
import torch

from .inference import WineEmbedder

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/competition_imgs'
MANIFEST = ROOT / 'df_2.csv'
OUT = ROOT / 'data/competition_label_crops'
JSON = ROOT / 'competition_labels_embeddings.json'
WEIGHTS = ROOT / 'detection/runs/wine_labels_yolo26l/weights/best.pt'
CHECKPOINT = ROOT / 'embedding/runs/dinov2_large_labels/best.pt'
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}


def database():
    OUT.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(OUT / 'progress.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS images (fname TEXT PRIMARY KEY, status TEXT, error TEXT)')
    db.execute('''CREATE TABLE IF NOT EXISTS crops
        (fname TEXT, idx INT, path TEXT, xyxy TEXT, crop_xyxy TEXT,
         confidence REAL, embedding TEXT, PRIMARY KEY(fname, idx))''')
    return db


def crop(db, files, batch_size, device):
    from ultralytics import YOLO
    model = YOLO(str(WEIGHTS))
    if model.names != {0: 'wine-labels'}:
        raise ValueError(f'Unexpected detector classes: {model.names}')
    done = {row[0] for row in db.execute('SELECT fname FROM images')}
    pending = [p for p in files if p.name not in done]
    print(f'Crops: {len(done)}/{len(files)} already processed; device={device}', flush=True)
    for start in range(0, len(pending), batch_size):
        paths = pending[start:start + batch_size]
        decoded = [(p, cv2.imread(str(p))) for p in paths]
        valid = [(p, im) for p, im in decoded if im is not None]
        results = model.predict([im for _, im in valid], imgsz=1024, conf=.25,
                                device=device, batch=batch_size, verbose=False) if valid else []
        for (path, im), result in zip(valid, results, strict=True):
            height, width = im.shape[:2]
            count = 0
            for idx, (box, score) in enumerate(zip(result.boxes.xyxy.cpu().numpy(),
                                                    result.boxes.conf.cpu().numpy(), strict=True)):
                x1, y1, x2, y2 = (float(v) for v in box)
                exact = [max(0, math.floor(x1)), max(0, math.floor(y1)),
                         min(width, math.ceil(x2)), min(height, math.ceil(y2))]
                if exact[2] <= exact[0] or exact[3] <= exact[1]:
                    continue
                target = OUT / 'crops' / f'{path.stem}__{idx:03d}.jpg'
                temporary = target.with_suffix('.tmp.jpg')
                target.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(temporary), im[exact[1]:exact[3], exact[0]:exact[2]],
                                   [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise OSError(f'Cannot save {temporary}')
                temporary.replace(target)
                db.execute('INSERT INTO crops VALUES (?,?,?,?,?,?,NULL)',
                           (path.name, idx, str(target.relative_to(ROOT)),
                            json.dumps([x1, y1, x2, y2]), json.dumps(exact), float(score)))
                count += 1
            db.execute('INSERT INTO images VALUES (?,?,?)',
                       (path.name, 'ok' if count else 'no_label', ''))
        for path, im in decoded:
            if im is None:
                db.execute('INSERT INTO images VALUES (?,?,?)', (path.name, 'decode_error', 'OpenCV decode failed'))
        db.commit()
        if (start // batch_size) % 20 == 0 or start + batch_size >= len(pending):
            print(f'Crops: {len(done) + min(start + batch_size, len(pending))}/{len(files)}', flush=True)


def embed(db, batch_size, device):
    rows = db.execute('SELECT fname,idx,path FROM crops WHERE embedding IS NULL ORDER BY fname,idx').fetchall()
    print(f'Embeddings pending: {len(rows)}; device={device}', flush=True)
    if not rows:
        return
    model = WineEmbedder(CHECKPOINT, device=device, precision='fp32' if device == 'cpu' else 'bf16')
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        vectors = model.embed([ROOT / row[2] for row in batch], batch_size=batch_size)
        for (fname, idx, _), vector in zip(batch, vectors, strict=True):
            db.execute('UPDATE crops SET embedding=? WHERE fname=? AND idx=?',
                       (json.dumps(vector.tolist(), separators=(',', ':')), fname, idx))
        db.commit()
        if (start // batch_size) % 20 == 0 or start + batch_size >= len(rows):
            print(f'Embeddings: {min(start + batch_size, len(rows))}/{len(rows)}', flush=True)


def export(db, files):
    if db.execute('SELECT count(*) FROM images').fetchone()[0] != len(files):
        raise RuntimeError('Cropping is incomplete')
    if db.execute('SELECT count(*) FROM crops WHERE embedding IS NULL').fetchone()[0]:
        raise RuntimeError('Embedding is incomplete')
    data = {p.name: [] for p in files}
    for fname, idx, path, xyxy, exact, confidence, vector in db.execute(
            'SELECT fname,idx,path,xyxy,crop_xyxy,confidence,embedding FROM crops ORDER BY fname,idx'):
        data[fname].append(dict(crop_path=path, xyxy=json.loads(xyxy),
                                crop_xyxy=json.loads(exact), confidence=confidence,
                                embedding=json.loads(vector)))
    temporary = JSON.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    temporary.replace(JSON)
    print(f'Saved {JSON}: {len(data)} images, {sum(map(len, data.values()))} labels', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['all', 'crop', 'embed', 'export'], default='all')
    parser.add_argument('--crop-batch', type=int, default=4)
    parser.add_argument('--embed-batch', type=int, default=2)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    with MANIFEST.open(newline='', encoding='utf-8') as stream:
        names = {row['fname'] for row in csv.DictReader(stream) if row.get('fname')}
    files = sorted(SOURCE / name for name in names
                   if (SOURCE / name).is_file() and Path(name).suffix.lower() in EXTENSIONS)
    print(f'Manifest: {len(names)} unique filenames, {len(files)} files present, '
          f'{len(names) - len(files)} missing', flush=True)
    if not files:
        raise RuntimeError(f'No images in {SOURCE}')
    db = database()
    if args.stage in ('all', 'crop'):
        crop(db, files, args.crop_batch, args.device)
    if args.stage in ('all', 'embed'):
        if db.execute('SELECT count(*) FROM images').fetchone()[0] != len(files):
            raise RuntimeError('Finish cropping before embedding')
        embed(db, args.embed_batch, args.device)
    if args.stage in ('all', 'export'):
        export(db, files)
    db.close()


if __name__ == '__main__':
    main()

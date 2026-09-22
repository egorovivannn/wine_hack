"""Create label crops from the current bottle dataset, inheriting its exact split/IDs."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from .core import ROOT, validate_manifest
from .prepare import detect, connect, sha_file


def finalize(base, det):
    if len(det) != len(base) or set(det.item_id) != set(base.item_id):
        raise ValueError('Label detection is incomplete')
    base = base.rename(columns={'image_path':'bottle_image_path','bbox':'bottle_bbox',
                                'confidence':'bottle_confidence','crop_sha256':'bottle_crop_sha256'})
    # Keep the original source hash and split; detect() hashes its bottle-crop input.
    det = det.drop(columns='source_sha256')
    frame = base.drop(columns=['width','height']).merge(det,on='item_id',validate='one_to_one')
    rejected = frame[frame.status != 'ok'].copy()
    frame = frame[frame.status == 'ok'].copy()
    frame['label_bbox_in_bottle'] = frame['bbox']
    def source_box(row):
        x,y,_,_ = json.loads(row.bottle_bbox)
        a,b,c,d = json.loads(row.label_bbox_in_bottle)
        return json.dumps([x+a,y+b,x+c,y+d])
    frame['bbox'] = frame.apply(source_box,axis=1)
    # New crops may collapse previously distinct images. Keep the source split
    # immutable: remove conflicts/cross-split duplicates rather than move rows.
    for column in ['source_sha256','crop_sha256']:
        group = frame.groupby(column)
        bad_hashes = set(group.wine_id.nunique().loc[lambda s:s>1].index)
        bad_hashes |= set(group.split.nunique().loc[lambda s:s>1].index)
        bad = frame[column].isin(bad_hashes)
        rejected = pd.concat([rejected,frame[bad].assign(status=f'conflicting_{column}')])
        frame = frame[~bad]
        duplicate = frame.duplicated(column)
        rejected = pd.concat([rejected,frame[duplicate].assign(status=f'duplicate_{column}')])
        frame = frame[~duplicate]
    sizes = frame.groupby('label').size()
    singleton = (frame.split != 'train') & frame.label.map(sizes).lt(2)
    rejected = pd.concat([rejected,frame[singleton].assign(status='heldout_without_positive')])
    frame = frame[~singleton].copy()
    validate_manifest(frame)
    expected = base.set_index('item_id')
    for col in ['split','label','wine_id','source_path']:
        assert (frame[col].to_numpy()==expected.loc[frame.item_id,col].to_numpy()).all(), col
    assert len(frame)+len(rejected)==len(base)
    return frame,rejected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv',type=Path,default=ROOT/'data/winesensed_labels_embedding/source_dataset.csv')
    parser.add_argument('--output',type=Path,default=ROOT/'data/winesensed_labels_embedding')
    parser.add_argument('--detector',type=Path,default=ROOT/'detection/runs/wine_labels_yolo26l/weights/best.pt')
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True,exist_ok=True)
    base = pd.read_csv(args.csv)
    validate_manifest(base)
    assert base.item_id.is_unique
    sources = base[['item_id','image_path']].rename(columns={'image_path':'source_path'})
    index = out/'sources.csv'
    if not index.exists():
        sources.to_csv(index,index=False)
    else:
        assert pd.read_csv(index).equals(sources.reset_index(drop=True))
    detect(SimpleNamespace(output=out,detector=args.detector,conf=.25,imgsz=1024,min_size=24,
                           device='0',batch_size=8,workers=8,selection='central_box'),index)
    db = connect(out)
    det = pd.read_sql_query('SELECT * FROM detections',db)
    db.close()
    frame,rejected = finalize(base,det)
    # Decode every final crop before allowing training to start.
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image
    def verify(row):
        with Image.open(row.image_path) as image:
            image.load()
            assert image.size == (row.width,row.height), row.image_path
    with ThreadPoolExecutor(max_workers=8) as pool:
        for _ in pool.map(verify,frame.itertuples()):
            pass
    temp = out/'dataset.csv.tmp'
    frame.to_csv(temp,index=False)
    temp.replace(out/'dataset.csv')
    rejected.to_csv(out/'rejected.csv',index=False)
    report = dict(complete=True,source_images=len(base),images=len(frame),
                  split_images=frame.split.value_counts().to_dict(),
                  split_classes=frame.groupby('split').label.nunique().to_dict(),
                  rejected=rejected.status.value_counts().to_dict(),
                  preserved_split=True,preserved_labels=True,decode_errors=0,
                  source_manifest_sha256=sha_file(args.csv),dataset_sha256=sha_file(out/'dataset.csv'))
    (out/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()

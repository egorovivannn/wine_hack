"""Convert GRAIN Products COCO to wine-only YOLO; group bursts to avoid leakage."""
import collections, hashlib, json, pathlib, re
from PIL import Image
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / 'grain_yolo'
records=[]
for source in sorted((ROOT/'Products').glob('*/annotations/instances.json')):
    data=json.loads(source.read_text()); anns=collections.defaultdict(list)
    wine={c['id'] for c in data['categories'] if c['name']=='wine_bottle'}
    for a in data['annotations']:
        if a['category_id'] in wine: anns[a['image_id']].append(a)
    for im in data['images']:
        path=source.parent.parent/'images'/im['file_name']
        with Image.open(path) as img:
            assert img.size==(im['width'],im['height']), path
            img.verify()
        name=im.get('extra',{}).get('name',im['file_name'])
        # Photos in the same capture minute stay in the same split, across lighting folders.
        match=re.search(r'(\d{8})_(\d{4})\d{2}', name)
        group='_'.join(match.groups()) if match else name.split('_jpg.rf.')[0]
        value=int(hashlib.sha256(('42:'+group).encode()).hexdigest()[:8],16)/2**32
        split='train' if value<.8 else 'val' if value<.9 else 'test'
        key=source.parent.parent.name.replace(' ','_')+'_'+path.stem
        image_dir=OUT/'images'/split; label_dir=OUT/'labels'/split
        image_dir.mkdir(parents=True,exist_ok=True);label_dir.mkdir(parents=True,exist_ok=True)
        dst=image_dir/(key+path.suffix)
        if not dst.exists(): dst.symlink_to(path)
        lines=[]
        for a in anns[im['id']]:
            x,y,w,h=a['bbox']; W,H=im['width'],im['height']
            x1,y1=max(0,x),max(0,y);x2,y2=min(W,x+w),min(H,y+h)
            assert x2>x1 and y2>y1, a
            lines.append(f'0 {(x1+x2)/2/W:.8f} {(y1+y2)/2/H:.8f} {(x2-x1)/W:.8f} {(y2-y1)/H:.8f}')
        (label_dir/(key+'.txt')).write_text('\n'.join(lines)+'\n' if lines else '')
        records.append(dict(file=str(path),split=split,group=group,boxes=len(lines),lighting=source.parent.parent.name))
(OUT/'data.yaml').write_text(f'path: {OUT}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: wine_bottle\n')
(OUT/'manifest.json').write_text(json.dumps(records,indent=2))
summary={s:dict(images=sum(r['split']==s for r in records),boxes=sum(r['boxes'] for r in records if r['split']==s),backgrounds=sum(r['boxes']==0 for r in records if r['split']==s)) for s in ['train','val','test']}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

"""Convert all GRAIN text-field classes to YOLO with grouped train/val/test splits."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'text_fields_yolo'


def main(source='Text fields', output=None):
    global OUT
    if output is not None:
        OUT = Path(output)
    records = []
    categories = None
    for annotation in sorted((ROOT / source).glob('*/annotations/instances.json')):
        data = json.loads(annotation.read_text())
        current = {c['id']: c['name'] for c in data['categories']}
        if categories is None:
            categories = current
        assert categories == current
        ids = {old: new for new, old in enumerate(sorted(categories))}
        annotations = defaultdict(list)
        for box in data['annotations']:
            annotations[box['image_id']].append(box)
        for im in data['images']:
            path = annotation.parent.parent / 'images' / im['file_name']
            with Image.open(path) as image:
                assert image.size == (im['width'], im['height']), path
                image.verify()
            name = im.get('extra', {}).get('name', im['file_name'])
            match = re.search(r'(\d{8})_(\d{4})\d{2}', name)
            group = '_'.join(match.groups()) if match else name.split('.rf.')[0]
            # Same original capture and nearby shots stay together across lighting conditions.
            value = int(hashlib.sha256(('42:' + group).encode()).hexdigest()[:8], 16) / 2**32
            split = 'train' if value < .8 else 'val' if value < .9 else 'test'
            lines, classes = [], []
            for box in annotations[im['id']]:
                x, y, w, h = box['bbox']
                W, H = im['width'], im['height']
                x1, y1, x2, y2 = max(0, x), max(0, y), min(W, x+w), min(H, y+h)
                assert x2 > x1 and y2 > y1, (path, box)
                cls = ids[box['category_id']]
                lines.append(f'{cls} {(x1+x2)/2/W:.9f} {(y1+y2)/2/H:.9f} {(x2-x1)/W:.9f} {(y2-y1)/H:.9f}')
                classes.append(cls)
            records.append(dict(source=str(path), group=group, split=split, classes=classes, lines=lines,
                                key=annotation.parent.parent.name.replace(' ', '_')+'_'+path.stem))
    # Identical image bytes cannot land in different splits, even under different names.
    parent = {r['group']: r['group'] for r in records}
    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    hashes = {}
    for r in records:
        digest = hashlib.sha256(Path(r['source']).read_bytes()).hexdigest()
        if digest in hashes:
            a, b = root(r['group']), root(hashes[digest])
            parent[max(a,b)] = min(a,b)
        hashes[digest] = r['group']
    for r in records:
        r['group'] = root(r['group'])
        value = int(hashlib.sha256(('42:' + r['group']).encode()).hexdigest()[:8],16)/2**32
        r['split'] = 'train' if value < .8 else 'val' if value < .9 else 'test'
        images = OUT/'images'/r['split']
        labels = OUT/'labels'/r['split']
        images.mkdir(parents=True,exist_ok=True)
        labels.mkdir(parents=True,exist_ok=True)
        dest = images/(r['key']+Path(r['source']).suffix)
        if not dest.exists():
            dest.symlink_to(r['source'])
        (labels/(r['key']+'.txt')).write_text('\n'.join(r.pop('lines'))+'\n')
    names = {i: categories[c] for i,c in enumerate(sorted(categories))}
    summary = {}
    for split in ['train','val','test']:
        rows = [r for r in records if r['split']==split]
        counts = Counter(c for r in rows for c in r['classes'])
        assert set(counts)==set(names), (split, counts)
        summary[split] = dict(images=len(rows),boxes=sum(counts.values()),classes={names[c]:counts[c] for c in names})
    (OUT/'data.yaml').write_text(yaml.safe_dump(dict(path=str(OUT),train='images/train',val='images/val',test='images/test',names=names)))
    (OUT/'manifest.json').write_text(json.dumps(records,indent=2))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()

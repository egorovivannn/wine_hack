import os, pathlib, shutil, json, hashlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'training'/'config')
from ultralytics import YOLO
import torch
out=ROOT/'real_test_predictions'
out.mkdir(exist_ok=False)
src=ROOT/'runs/grain_yolo26s/weights/best.pt'
# Snapshot the checkpoint so continuing training cannot change this prediction run.
checkpoint=out/'model_snapshot.pt'
for attempt in range(5):
    before=src.stat()
    shutil.copy2(src,checkpoint)
    after=src.stat()
    if before.st_mtime_ns==after.st_mtime_ns and before.st_size==after.st_size:
        break
else: raise RuntimeError('Checkpoint changed repeatedly while copying')
model=YOLO(str(checkpoint))
metadata={'checkpoint_source':str(src),'epoch':int(model.ckpt.get('epoch',-1))+1,'sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'imgsz':640,'conf':0.25,'images':[]}
for p in sorted((ROOT/'real_test').iterdir()):
    if p.suffix.lower() not in {'.jpg','.jpeg','.png','.webp','.bmp','.tif','.tiff'}: continue
    result=model.predict(str(p),imgsz=640,conf=.25,device=0,verbose=False)[0]
    result.save(filename=str(out/p.name))
    (out/'labels').mkdir(exist_ok=True)
    result.save_txt(str(out/'labels'/(p.stem+'.txt')),save_conf=True)
    metadata['images'].append({'file':p.name,'detections':len(result.boxes),'boxes':json.loads(result.to_json())})
    print(p.name,len(result.boxes),flush=True)
(out/'predictions.json').write_text(json.dumps(metadata,indent=2))
print('DONE',out,'epoch',metadata['epoch'],flush=True)

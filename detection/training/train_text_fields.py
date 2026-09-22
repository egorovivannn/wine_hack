"""Train a separate YOLO26 Large text-field detector and evaluate best weights."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ['YOLO_CONFIG_DIR'] = str(ROOT/'training/config')
os.environ['WANDB_MODE'] = 'disabled'
from ultralytics import YOLO
import torch


if __name__ == '__main__':
    assert torch.cuda.is_available(), 'CUDA GPU required'
    data = str(ROOT/'text_fields_yolo/data.yaml')
    model = YOLO(str(ROOT/'training/yolo26l.pt'))
    model.train(data=data,epochs=100,patience=20,imgsz=1024,batch=8,device=0,
                workers=8,seed=42,project=str(ROOT/'runs'),name='text_fields_yolo26l',
                exist_ok=False,pretrained=True,plots=True,save=True,cache=False,
                fliplr=0.0,flipud=0.0)
    best = Path(model.trainer.best)
    metrics = YOLO(str(best)).val(data=data,split='test',device=0,imgsz=1024,batch=8,
                                 project=str(ROOT/'runs'),name='text_fields_yolo26l_test')
    report = dict(metrics.results_dict)
    report['per_class'] = {metrics.names[int(c)]: dict(zip(['precision','recall','mAP50','mAP50-95'],
                                map(float,metrics.box.class_result(i))))
                           for i,c in enumerate(metrics.box.ap_class_index)}
    (best.parent.parent/'test_metrics.json').write_text(json.dumps(report,indent=2))
    print('TRAINING_COMPLETE',best,flush=True)

"""Train YOLO26 Large for full wine labels (one class)."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ['YOLO_CONFIG_DIR'] = str(ROOT/'training/config')
os.environ['WANDB_MODE'] = 'disabled'
from ultralytics import YOLO
import torch
import yaml

if __name__ == '__main__':
    assert torch.cuda.is_available(), 'CUDA GPU required'
    data = ROOT/'wine_labels_yolo/data.yaml'
    assert yaml.safe_load(data.read_text())['names'] == {0: 'wine-labels'}
    model = YOLO(str(ROOT/'training/yolo26l.pt'))
    model.train(data=str(data),epochs=100,patience=20,imgsz=1024,batch=8,device=0,
                workers=8,seed=42,project=str(ROOT/'runs'),name='wine_labels_yolo26l',
                exist_ok=False,pretrained=True,plots=True,save=True,cache=False,
                fliplr=0.0,flipud=0.0)
    best = Path(model.trainer.best)
    metrics = YOLO(str(best)).val(data=str(data),split='test',device=0,imgsz=1024,batch=8,
                                 project=str(ROOT/'runs'),name='wine_labels_yolo26l_test')
    (best.parent.parent/'test_metrics.json').write_text(json.dumps(metrics.results_dict,indent=2))
    print('TRAINING_COMPLETE',best,flush=True)

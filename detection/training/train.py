import json, os, pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'training'/'config')
os.environ['WANDB_MODE']='disabled'
from ultralytics import YOLO
import torch
if __name__=='__main__':
    assert torch.cuda.is_available(), 'CUDA GPU required'
    model=YOLO(str(ROOT/'training'/'yolo26l.pt'))
    model.train(data=str(ROOT/'grain_yolo'/'data.yaml'), epochs=100, patience=20,
                imgsz=640, batch=64, device=0, workers=8, seed=42,
                project=str(ROOT/'runs'),name='grain_yolo26l',exist_ok=False,
                pretrained=True,plots=True,save=True,cache=False)
    best=pathlib.Path(model.trainer.best)
    metrics=YOLO(str(best)).val(data=str(ROOT/'grain_yolo'/'data.yaml'),split='test',
                device=0,imgsz=640,batch=16,project=str(ROOT/'runs'),name='grain_yolo26s_test')
    (best.parent.parent/'test_metrics.json').write_text(json.dumps(metrics.results_dict,indent=2))
    print('TRAINING_COMPLETE',best,flush=True)

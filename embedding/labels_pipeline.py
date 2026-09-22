"""Run after successful label-detector training: crops, audit, new DINOv2 training."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'embedding/runs/dinov2_large_labels'
DATA = ROOT/'data/winesensed_labels_embedding'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    status = OUT/'pipeline_status.json'
    def stage(name,**extra):
        temp = status.with_suffix('.tmp')
        temp.write_text(json.dumps(dict(stage=name,updated_unix=time.time(),**extra),indent=2))
        temp.replace(status)
        print(name,flush=True)
    detector = ROOT/'detection/runs/wine_labels_yolo26l'
    # best.pt exists during training: it alone is NOT a completion signal.
    if not (detector/'test_metrics.json').is_file():
        raise RuntimeError('Detector training and test evaluation have not completed')
    try:
        if not (DATA/'report.json').exists():
            stage('cropping_labels')
            subprocess.run([sys.executable,'-u','-m','embedding.prepare_labels'],cwd=ROOT,check=True)
        report = json.loads((DATA/'report.json').read_text())
        if not report.get('complete') or not report.get('preserved_split'):
            raise RuntimeError('Label dataset not verified')
        if (OUT/'training_complete.json').exists():
            stage('complete')
            return
        stage('training_embedder')
        cmd = [sys.executable,'-u','-m','embedding.train','--csv',str(DATA/'dataset.csv'),
               '--output',str(OUT),'--epochs','20','--steps-per-epoch','2000',
               '--batch-size','64','--accumulation','2','--precision','bf16',
               '--height','336','--width','336']
        if (OUT/'last.pt').exists():
            cmd += ['--resume',str(OUT/'last.pt')]
        with (OUT/'train.log').open('a') as log:
            subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        (OUT/'training_complete.json').write_text(json.dumps(dict(completed_unix=time.time())))
        stage('complete')
    except Exception as error:
        stage('failed',error=str(error))
        raise


if __name__ == '__main__':
    main()

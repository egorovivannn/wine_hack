"""Small, reproducible GRAIN wine-label detector experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import torch
from ultralytics import YOLO, __version__ as ultralytics_version
import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True, help='Prepared YOLO data.yaml')
    parser.add_argument('--output', type=Path, required=True, help='New run directory')
    parser.add_argument('--weights', default='yolo26s.pt')
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f'Output directory already exists: {args.output}')
    if not torch.cuda.is_available():
        parser.error('CUDA GPU is required')
    data = yaml.safe_load(args.data.read_text())
    if list(data['names'].values()) != ['wine-labels']:
        parser.error(f'Expected one wine-labels class, got {data["names"]}')
    for split in ('train', 'val', 'test'):
        if not (Path(data['path']) / data[split]).is_dir():
            parser.error(f'Missing {split} image directory in {args.data}')
    dataset_manifest = Path(data['path']) / 'manifest.jsonl'
    if not dataset_manifest.is_file():
        parser.error(f'Missing split/provenance manifest: {dataset_manifest}')
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    model = YOLO(args.weights)
    pretrained = Path(model.ckpt_path)
    provenance = {
        'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'data_yaml_sha256': sha256(args.data),
        'dataset_manifest_sha256': sha256(dataset_manifest),
        'pretrained_path': str(pretrained.resolve()),
        'pretrained_sha256': sha256(pretrained),
        'pretrained_model': args.weights,
        'ultralytics_version': ultralytics_version,
        'torch_version': torch.__version__,
        'python_version': platform.python_version(),
        'gpu': torch.cuda.get_device_name(0),
        'seed': args.seed,
        'epochs': args.epochs,
        'batch': args.batch,
        'imgsz': args.imgsz,
        'workers': args.workers,
    }
    train = model.train(
        data=str(args.data.resolve()), epochs=args.epochs, patience=5,
        imgsz=args.imgsz, batch=args.batch, device=0, workers=args.workers,
        seed=args.seed, deterministic=True, project=str(args.output.parent.resolve()),
        name=args.output.name, exist_ok=False, pretrained=True, plots=True,
        save=True, cache=False, fliplr=0.0, flipud=0.0,
    )
    best = args.output / 'weights' / 'best.pt'
    if not best.exists():
        raise FileNotFoundError(best)
    test = YOLO(str(best)).val(
        data=str(args.data.resolve()), split='test', device=0,
        imgsz=args.imgsz, batch=args.batch, workers=args.workers,
        project=str(args.output.parent.resolve()), name=args.output.name + '_test',
        exist_ok=False, plots=True,
    )
    provenance.update(
        best_checkpoint_sha256=sha256(best),
        val_metrics={k: float(v) for k, v in train.results_dict.items()},
        test_metrics={k: float(v) for k, v in test.results_dict.items()},
        peak_allocated_gib=round(torch.cuda.max_memory_allocated() / 2**30, 3),
        peak_reserved_gib=round(torch.cuda.max_memory_reserved() / 2**30, 3),
        elapsed_seconds=round(time.time() - started, 1),
    )
    (args.output / 'experiment.json').write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    print(json.dumps(provenance, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()

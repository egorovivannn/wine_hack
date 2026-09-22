"""Run the trained bottle detector for every unique fname in df_2.csv."""

import csv
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DETECTION = ROOT / "detection"
CSV_PATH = ROOT / "df_2.csv"
IMAGE_DIR = ROOT / "imgs"
MODEL_PATH = DETECTION / "runs/grain_yolo26s/weights/best.pt"
OUTPUT_PATH = ROOT / "df_2_bboxes.json"

os.environ["YOLO_CONFIG_DIR"] = str(DETECTION / "training/config")
os.environ["WANDB_MODE"] = "disabled"

from ultralytics import YOLO


def main() -> None:
    with CSV_PATH.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))

    # dict.fromkeys preserves the order of first appearance in the CSV.
    filenames = list(dict.fromkeys(row["fname"] for row in rows))
    existing = [IMAGE_DIR / name for name in filenames if (IMAGE_DIR / name).is_file()]
    output: dict[str, list[dict[str, object]]] = {name: [] for name in filenames}

    model = YOLO(str(MODEL_PATH))
    chunk_size = 16
    for offset in range(0, len(existing), chunk_size):
        paths = existing[offset : offset + chunk_size]
        predictions = model.predict(
            source=[str(path) for path in paths],
            imgsz=640,
            conf=0.25,
            device=0,
            batch=chunk_size,
            verbose=False,
        )
        for path, result in zip(paths, predictions, strict=True):
            boxes = []
            for xyxy, confidence in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist()):
                boxes.append(
                    {
                        "xyxy": [round(value, 2) for value in xyxy],
                        "confidence": round(confidence, 6),
                    }
                )
            output[path.name] = boxes
        processed = min(offset + chunk_size, len(existing))
        if processed % 100 < chunk_size or processed == len(existing):
            print(f"Processed {processed}/{len(existing)}", flush=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2)

    missing = [name for name in filenames if not (IMAGE_DIR / name).is_file()]
    print(f"Saved {OUTPUT_PATH}")
    print(f"Unique keys: {len(output)}; missing files: {len(missing)}")


if __name__ == "__main__":
    main()

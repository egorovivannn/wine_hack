"""Embed every detection crop; preserve original fname -> boxes layout."""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bottle_embeddings import BottleEmbedder, crop_bottle, read_image, MODEL_ID, INPUT_SIZE
import torch


def main():
    source = ROOT / "df_2_bboxes.json"
    output = ROOT / "df_2_bboxes_embeddings_dinov2_large_yolo26l.json"
    data = json.loads(source.read_text())
    model = BottleEmbedder(device="cuda")

    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    total = sum(map(len, data.values()))
    pending, targets = [], []
    completed = 0

    def flush():
        nonlocal completed
        for target, vector in zip(targets, model.embed_batch(pending), strict=True):
            target["embedding"] = vector.tolist()
        completed += len(pending)
        pending.clear()
        targets.clear()
        if completed % 100 == 0 or completed == total:
            print(f"Embedded {completed}/{total}; elapsed {time.monotonic()-started:.0f}s", flush=True)

    for fname, boxes in data.items():
        if not boxes:
            continue
        try:
            image = read_image(ROOT / "data" / "competition_imgs" / fname)
        except:
            continue
        for box in boxes:
            crop, exact_box = crop_bottle(image, box["xyxy"])
            box["crop_xyxy"] = exact_box
            pending.append(crop)
            targets.append(box)
            if len(pending) == 4:
                flush()
    if pending:
        flush()
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    temporary.replace(output)
    meta = dict(model=MODEL_ID, dimension=512, pooling="trained projection", normalization="L2",
                input_size=INPUT_SIZE, resize="aspect-preserving letterbox; RGB fill 124,116,104",
                mean=[.485,.456,.406], std=[.229,.224,.225], dtype="float16 inference / float32 output",
                crop_convention="floor left/top, ceil right/bottom, clipped; right/bottom exclusive",
                orientation="OpenCV imread (same as YOLO)", source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                embeddings=completed, images=len(data), elapsed_seconds=time.monotonic()-started,
                peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                peak_gpu_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
    output.with_suffix(".meta.json").write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta,indent=2), flush=True)
    print(f"Saved {output}",flush=True)


if __name__ == "__main__":
    main()

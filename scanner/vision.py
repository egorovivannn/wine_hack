"""Offline SigLIP 2 image index and exact catalog retrieval."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel

from .catalog import ROOT, sha256_file
from .image import PREPROCESSING_VERSION, decode_image, query_views, reference_views


MODEL_ID = "google/siglip2-base-patch16-384"
MODEL_REVISION = "f775b65a79762255128c981547af89addcfe0f88"
DEFAULT_MODEL_DIR = ROOT / "data/models/siglip2-base-patch16-384"
DEFAULT_MANIFEST = ROOT / "data/catalog_manifest_corrected.json"
DEFAULT_INDEX = ROOT / "data/index/siglip2_multiview.npz"


def _unit_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if not np.isfinite(values).all() or np.any(norms <= 0):
        raise ValueError("Embedding has non-finite or zero vectors")
    return values / norms


class VisionEncoder:
    def __init__(self, model_dir: Path = DEFAULT_MODEL_DIR, device: str | None = None):
        model_dir = Path(model_dir)
        self.model_dir = model_dir
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.processor = AutoImageProcessor.from_pretrained(
            model_dir, use_fast=False, local_files_only=True)
        self.model = AutoModel.from_pretrained(
            model_dir, dtype=dtype, local_files_only=True).to(self.device).eval()

    @torch.inference_mode()
    def encode(self, images: list[Image.Image], batch_size: int = 16) -> np.ndarray:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        vectors = []
        for start in range(0, len(images), batch_size):
            inputs = self.processor(images=images[start:start + batch_size],
                                    return_tensors="pt").to(self.device)
            features = self.model.get_image_features(**inputs)
            if not isinstance(features, torch.Tensor):  # transformers 5 returns a model output
                features = features.pooler_output
            vectors.append(F.normalize(features.float(), dim=-1).cpu().numpy())
        if not vectors:
            raise ValueError("No images to encode")
        return _unit_rows(np.concatenate(vectors))


@dataclass(frozen=True)
class Candidate:
    slug: str
    score: float
    image_name: str


class GalleryIndex:
    def __init__(self, filenames: list[str], embeddings: np.ndarray,
                 image_slugs: list[list[str]]):
        if embeddings.ndim != 3:
            raise ValueError("Expected [reference, view, feature] embeddings")
        if len(filenames) != len(embeddings) or len(image_slugs) != len(filenames):
            raise ValueError("Index rows, filenames and slug groups differ")
        if len(set(filenames)) != len(filenames):
            raise ValueError("Duplicate filenames in index")
        if any(not slugs for slugs in image_slugs):
            raise ValueError("Reference without a slug")
        self.filenames = filenames
        self.embeddings = _unit_rows(embeddings)
        self.image_slugs = image_slugs

    def search(self, query: np.ndarray, top_k: int = 5) -> list[Candidate]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        query = _unit_rows(query)
        if query.ndim != 2 or query.shape[1] != self.embeddings.shape[2]:
            raise ValueError("Query and gallery feature dimensions differ")
        # Each query crop keeps its best reference view; the crops are then averaged.
        scores = np.einsum("qd,rvd->qrv", query, self.embeddings).max(axis=2).mean(axis=0)
        ranked = np.argsort(-scores, kind="stable")
        candidates: list[Candidate] = []
        for index in ranked:
            for slug in self.image_slugs[int(index)]:
                candidates.append(Candidate(slug=slug, score=float(scores[index]),
                                            image_name=self.filenames[int(index)]))
                if len(candidates) >= top_k:
                    return candidates
        return candidates


def build_index(manifest_path: Path, images_dir: Path, model_dir: Path,
                output: Path, batch_size: int = 16) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported catalog manifest")
    expected_files = manifest["images"]
    for record in expected_files:
        path = images_dir / record["filename"]
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"Reference changed: {path}")
    encoder = VisionEncoder(model_dir)
    vectors: list[np.ndarray] = []
    started = time.perf_counter()
    # Decode only a bounded batch: full-resolution product photos may be large.
    for start in range(0, len(expected_files), batch_size):
        records = expected_files[start:start + batch_size]
        views = []
        for record in records:
            views.extend(reference_views(decode_image(images_dir / record["filename"])))
        embeddings = encoder.encode(views, batch_size=batch_size)
        vectors.extend(embeddings.reshape(len(records), len(views) // len(records), -1))
        if start == 0 or (start + len(records)) % 200 < batch_size:
            print(f"Indexed {start + len(records)}/{len(expected_files)} references", flush=True)
    matrix = np.stack(vectors).astype(np.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, filenames=np.asarray([r["filename"] for r in expected_files]),
                        embeddings=matrix)
    os.replace(temporary, output)
    metadata = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_weights_sha256": sha256_file(model_dir / "model.safetensors"),
        "preprocessing_version": PREPROCESSING_VERSION,
        "manifest_sha256": sha256_file(manifest_path),
        "index_sha256": sha256_file(output),
        "references": len(expected_files),
        "embedding_dimension": int(matrix.shape[-1]),
        "build_seconds": round(time.perf_counter() - started, 2),
    }
    meta_path = output.with_suffix(".json")
    temp_meta = meta_path.with_suffix(".json.tmp")
    temp_meta.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    os.replace(temp_meta, meta_path)
    return metadata


def load_index(index_path: Path, manifest_path: Path) -> GalleryIndex:
    metadata = json.loads(index_path.with_suffix(".json").read_text())
    if (metadata["preprocessing_version"] != PREPROCESSING_VERSION or
            metadata["manifest_sha256"] != sha256_file(manifest_path) or
            metadata["index_sha256"] != sha256_file(index_path)):
        raise ValueError("Index, preprocessing, or catalog manifest changed; rebuild index")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with np.load(index_path, allow_pickle=False) as archive:
        filenames = archive["filenames"].tolist()
        embeddings = archive["embeddings"]
    expected = [record["filename"] for record in manifest["images"]]
    if filenames != expected:
        raise ValueError("Index filename order differs from catalog manifest")
    return GalleryIndex(filenames, embeddings,
                        [record["slugs"] for record in manifest["images"]])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build", "search"])
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--images-dir", type=Path, default=ROOT / "data/competition_imgs")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.command == "build":
        print(json.dumps(build_index(args.manifest, args.images_dir,
                                     args.model_dir, args.index, args.batch_size)))
        return
    if args.image is None:
        parser.error("--image is required for search")
    encoder = VisionEncoder(args.model_dir)
    gallery = load_index(args.index, args.manifest)
    query = encoder.encode(list(query_views(decode_image(args.image))))
    for candidate in gallery.search(query, top_k=10):
        print(json.dumps(candidate.__dict__, ensure_ascii=False))


if __name__ == "__main__":
    main()

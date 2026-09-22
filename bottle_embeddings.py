"""Current 512-D wine-label embeddings and NumPy cosine search."""
from pathlib import Path
import json
import math

import cv2
import numpy as np
from PIL import Image, ImageOps
import torch
from embedding.inference import WineEmbedder

ROOT = Path(__file__).resolve().parent
MODEL_ID = "embedding/runs/dinov2_large_labels/best.pt"
MODEL_DIR = ROOT / MODEL_ID
INPUT_SIZE = 336
EMBEDDING_DIM = 512


def read_image(image):
    """Accept a path, PIL image or HWC RGB uint8 array; paths match YOLO orientation."""
    if isinstance(image, (str, Path)):
        # Same decoder as detection: applies EXIF orientation and returns RGB.
        bgr = cv2.imread(str(image))
        if bgr is None:
            raise ValueError(f"Cannot decode image: {image}")
        return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    if isinstance(image, Image.Image):
        return ImageOps.exif_transpose(image).convert("RGB")
    if isinstance(image, np.ndarray) and image.dtype == np.uint8 and image.ndim == 3 and image.shape[2] == 3:
        return Image.fromarray(image)
    raise TypeError("Expected path, PIL image or HWC RGB uint8 numpy array")


def crop_bottle(image, bbox):
    """Return crop and exact integer xyxy (right/bottom exclusive), clipped to image."""
    image = read_image(image)
    coords = np.asarray(bbox, dtype=float)
    if coords.shape != (4,) or not np.isfinite(coords).all():
        raise ValueError("bbox must contain four finite xyxy coordinates")
    x1, y1, x2, y2 = coords
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox has no positive area")
    w, h = image.size
    box = [max(0, math.floor(x1)), max(0, math.floor(y1)), min(w, math.ceil(x2)), min(h, math.ceil(y2))]
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("bbox is outside the image")
    return image.crop(box), box


class BottleEmbedder:
    """Embed label crops with the model used for competition_labels_embeddings.json."""
    def __init__(self, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.encoder = WineEmbedder(MODEL_DIR, device=str(self.device),
                                   precision="bf16" if self.device.type == "cuda" else "fp32")
        self.model = self.encoder.model

    @torch.inference_mode()
    def embed_batch(self, images):
        if not images:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)
        return self.encoder.embed([read_image(im) for im in images]).astype(np.float32)

    def embed_image(self, image, bbox=None):
        """Return L2-normalized float32 vector (512,). Supply bbox or a label crop."""
        if bbox is not None:
            image, _ = crop_bottle(image, bbox)
        return self.embed_batch([image])[0]


def load_database(path=ROOT / "competition_labels_embeddings.json"):
    """Load JSON once into a small float32 matrix and parallel crop records."""
    with Path(path).open(encoding="utf-8") as stream:
        data = json.load(stream)
    records, vectors = [], []
    for fname, boxes in data.items():
        for index, box in enumerate(boxes):
            if "embedding" not in box:
                raise ValueError(f"Missing embedding: {fname}, box {index}")
            vectors.append(box["embedding"])
            records.append({"fname": fname, "box_index": index,
                            **{k: v for k, v in box.items() if k != "embedding"}})
    matrix = np.asarray(vectors, dtype=np.float32).reshape(-1, EMBEDDING_DIM)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not np.isfinite(matrix).all() or np.any(norms == 0):
        raise ValueError("Database contains invalid vectors")
    return matrix / np.maximum(norms, 1e-12), records


def search_embeddings(embedding, database, top_k=5):
    """Exact cosine search on CPU; returns crop records with score (not probability)."""
    matrix, records = database
    query = np.asarray(embedding, dtype=np.float32)
    if query.shape != (matrix.shape[1],) or not np.isfinite(query).all() or np.linalg.norm(query) == 0:
        raise ValueError("Query must be a finite nonzero embedding of matching dimension")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    scores = matrix @ (query / np.linalg.norm(query))
    indices = np.argsort(-scores, kind="stable")[:top_k]
    return [{**records[i], "score": float(scores[i])} for i in indices]

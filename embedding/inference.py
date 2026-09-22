"""Small reusable API: crop -> 512-D vector -> nearest wine labels."""
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch

from .core import BottleTransform, amp_context, exact_topk, load_checkpoint


class WineEmbedder:
    def __init__(self, checkpoint, device='cuda', precision=None):
        self.device = torch.device(device)
        self.model, config = load_checkpoint(checkpoint, self.device)
        self.precision = precision or config['precision']
        self.transform = BottleTransform(config['height'], config['width'])

    @torch.inference_mode()
    def embed(self, crops, batch_size=32):
        """Paths/PIL crops, already cut by the same detector used for training."""
        vectors = []
        for offset in range(0, len(crops), batch_size):
            pixels = []
            for crop in crops[offset:offset+batch_size]:
                if isinstance(crop, (str, Path)):
                    with Image.open(crop) as image:
                        pixels.append(self.transform(image))
                else:
                    pixels.append(self.transform(crop))
            with amp_context(self.device, self.precision):
                vectors.append(self.model(torch.stack(pixels).to(self.device)).cpu().numpy())
        return np.concatenate(vectors) if vectors else np.empty((0, self.model.head[-1].out_features), np.float32)

    def search(self, crops, gallery_vectors, gallery_csv, k=5):
        if isinstance(gallery_vectors, (str, Path)):
            gallery_vectors = np.load(gallery_vectors, mmap_mode='r')
        gallery = pd.read_csv(gallery_csv) if isinstance(gallery_csv, (str, Path)) else gallery_csv
        if len(gallery) != len(gallery_vectors):
            raise ValueError('Gallery table and vectors have different lengths')
        indices, scores = exact_topk(self.embed(crops), gallery_vectors, self.device, k)
        return [[dict(label=int(gallery.iloc[index].label), wine_id=gallery.iloc[index].wine_id,
                      image_path=gallery.iloc[index].image_path, score=float(score))
                 for index, score in zip(row, values)] for row, values in zip(indices, scores)]

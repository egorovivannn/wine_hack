"""Shared preprocessing, metric model, balanced sampling and exact retrieval."""
from contextlib import nullcontext
import math
from pathlib import Path
import random

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms
from transformers import AutoConfig, AutoModel

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / 'detection/models/dinov2-large'


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(1)


def amp_context(device, precision):
    if device.type != 'cuda' or precision == 'fp32':
        return nullcontext()
    return torch.autocast('cuda', dtype=torch.float16 if precision == 'fp16' else torch.bfloat16)


class BottleTransform:
    def __init__(self, height=518, width=224, train=False):
        self.size = (width, height)
        self.jitter = transforms.ColorJitter(.15, .15, .10, .02) if train else None
        self.tensor = transforms.Compose([transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])])

    def __call__(self, image):
        image = ImageOps.exif_transpose(image).convert('RGB')
        if self.jitter:
            image = self.jitter(image)
        image = ImageOps.pad(image, self.size, method=Image.Resampling.BICUBIC,
                             color=(124, 116, 104))
        return self.tensor(image)


class WineDataset(Dataset):
    def __init__(self, frame, height=518, width=224, train=False):
        self.paths = frame.image_path.tolist()
        self.labels = frame.label.to_numpy(dtype=np.int64)
        self.transform = BottleTransform(height, width, train)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            pixels = self.transform(image)
        return pixels, int(self.labels[index])


class PKSampler(Sampler):
    """Uniform classes, K images per class. Singleton views only allowed for train."""
    def __init__(self, labels, batch_size=32, k=2, steps=None, seed=42, allow_repeat=True):
        if k < 2 or batch_size % k or batch_size // k < 2:
            raise ValueError('Need K >= 2, batch_size divisible by K, and at least two classes/batch.')
        groups = pd.Series(np.arange(len(labels))).groupby(np.asarray(labels))
        self.groups = [g.to_numpy() for _, g in groups if allow_repeat or len(g) >= k]
        self.p, self.k = batch_size // k, k
        if len(self.groups) < self.p:
            raise ValueError(f'Need {self.p} eligible wine classes; found {len(self.groups)}')
        self.steps = steps or math.ceil(len(labels) / batch_size)
        self.seed, self.epoch = seed, 0

    def __len__(self):
        return self.steps

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        for _ in range(self.steps):
            selected = rng.choice(len(self.groups), self.p, replace=False)
            yield [int(index) for group in selected
                   for index in rng.choice(self.groups[group], self.k, replace=len(self.groups[group]) < self.k)]


class WineModel(nn.Module):
    def __init__(self, model_path=MODEL_PATH, embedding_dim=512, train_blocks=4,
                 checkpointing=True, pretrained=True, config=None):
        super().__init__()
        if config is None:
            config = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
        else:
            config = AutoConfig.for_model(config['model_type'], **{k: v for k, v in config.items() if k != 'model_type'})
        config._attn_implementation = 'sdpa'
        self.backbone = (AutoModel.from_pretrained(str(model_path), config=config, local_files_only=True)
                         if pretrained else AutoModel.from_config(config))
        self.backbone.requires_grad_(False)
        if not 0 <= train_blocks <= len(self.backbone.encoder.layer):
            raise ValueError('Invalid number of trainable blocks')
        if train_blocks == len(self.backbone.encoder.layer):
            self.backbone.requires_grad_(True)
        elif train_blocks:
            for block in self.backbone.encoder.layer[-train_blocks:]:
                block.requires_grad_(True)
            self.backbone.layernorm.requires_grad_(True)
        if checkpointing and train_blocks:
            self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        hidden = config.hidden_size
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, embedding_dim))

    def forward(self, pixels):
        hidden = self.backbone(pixel_values=pixels).last_hidden_state[:, 0]
        return F.normalize(self.head(hidden).float(), dim=-1)


def supervised_contrastive(embeddings, labels, temperature=.07):
    """Stable SupCon, exclude self from numerator and denominator (FP32)."""
    z = embeddings.float()
    logits = z @ z.T / temperature
    eye = torch.eye(len(z), device=z.device, dtype=torch.bool)
    positive = labels[:, None].eq(labels[None, :]) & ~eye
    counts = positive.sum(1)
    if not torch.all(counts > 0):
        raise ValueError('Each anchor needs a positive; use PKSampler.')
    logits = logits.masked_fill(eye, -torch.inf)
    log_denominator = torch.logsumexp(logits, dim=1)
    positive_logits = logits.masked_fill(~positive, 0).sum(1) / counts
    return (log_denominator - positive_logits).mean()


def make_loader(dataset, workers, batch_size=64, sampler=None, seed=42):
    generator = torch.Generator().manual_seed(seed)
    options = dict(dataset=dataset, num_workers=workers, pin_memory=torch.cuda.is_available(),
                   worker_init_fn=seed_worker, generator=generator)
    if workers:
        options.update(persistent_workers=True, prefetch_factor=2)
    if sampler is not None:
        return DataLoader(**options, batch_sampler=sampler)
    return DataLoader(**options, batch_size=batch_size, shuffle=False)


@torch.inference_mode()
def encode(model, frame, device, precision='fp16', batch_size=64, workers=8, height=518, width=224):
    model.eval()
    loader = make_loader(WineDataset(frame, height, width), workers, batch_size)
    vectors = np.empty((len(frame), model.head[-1].out_features), dtype=np.float32)
    offset = 0
    for pixels, _ in loader:
        with amp_context(device, precision):
            z = model(pixels.to(device, non_blocking=True))
        batch_vectors = z.cpu().numpy()
        if not np.isfinite(batch_vectors).all() or np.any(np.linalg.norm(batch_vectors, axis=1) == 0):
            raise FloatingPointError(f'Invalid embedding at batch offset {offset}')
        vectors[offset:offset+len(z)] = batch_vectors
        offset += len(z)
        if offset % (batch_size * 200) == 0:
            print(f'  embedded {offset:,}/{len(frame):,}', flush=True)
    return vectors


@torch.inference_mode()
def exact_topk(queries, gallery, device, k=5, query_batch=256, gallery_batch=32768, exclude_indices=None):
    """Exact cosine top-k, FP32 matmul, bounded GPU memory (no NxN matrix)."""
    if not len(gallery) or not len(queries):
        raise ValueError('Query and gallery must be nonempty')
    if k < 1:
        raise ValueError('k must be positive')
    k = min(k, len(gallery) - (exclude_indices is not None))
    if k < 1:
        raise ValueError('No gallery candidates after self-exclusion')
    if exclude_indices is not None:
        exclude_indices = np.asarray(exclude_indices)
        if len(exclude_indices) != len(queries) or np.any(exclude_indices < 0) or np.any(exclude_indices >= len(gallery)):
            raise ValueError('Invalid self-exclusion indices')
    indices = np.empty((len(queries), k), dtype=np.int64)
    values = np.empty((len(queries), k), dtype=np.float32)
    for qi in range(0, len(queries), query_batch):
        query = torch.as_tensor(queries[qi:qi+query_batch], device=device, dtype=torch.float32)
        best_values = torch.full((len(query), k), -torch.inf, device=device)
        best_indices = torch.zeros((len(query), k), dtype=torch.long, device=device)
        for gi in range(0, len(gallery), gallery_batch):
            refs = torch.as_tensor(gallery[gi:gi+gallery_batch], device=device, dtype=torch.float32)
            scores = query @ refs.T
            if exclude_indices is not None:
                own = torch.as_tensor(exclude_indices[qi:qi+len(query)] - gi, device=device)
                mask = (own >= 0) & (own < len(refs))
                rows = torch.arange(len(query), device=device)[mask]
                scores[rows, own[mask]] = -torch.inf
            local_values, local_indices = scores.topk(min(k, len(refs)), dim=1)
            candidates = torch.cat([best_values, local_values], dim=1)
            candidates_idx = torch.cat([best_indices, local_indices + gi], dim=1)
            best_values, select = candidates.topk(k, dim=1)
            best_indices = candidates_idx.gather(1, select)
        indices[qi:qi+len(query)] = best_indices.cpu().numpy()
        values[qi:qi+len(query)] = best_values.cpu().numpy()
    return indices, values


def retrieval_metrics(query_labels, gallery_labels, indices, exclude_indices=None):
    query_labels, gallery_labels = np.asarray(query_labels), np.asarray(gallery_labels)
    counts = pd.Series(gallery_labels).value_counts()
    available = pd.Series(query_labels).map(counts).fillna(0).to_numpy()
    if exclude_indices is not None:
        excluded = np.asarray(exclude_indices)
        if (indices == excluded[:, None]).any():
            raise ValueError('Self-match found in top-k')
        available -= gallery_labels[excluded] == query_labels
    eligible = available > 0
    if not eligible.all():
        raise ValueError(f'{(~eligible).sum()} queries have no positive in gallery; invalid evaluation protocol.')
    hits = gallery_labels[indices] == query_labels[:, None]
    result = {'queries': len(query_labels), 'gallery_images': len(gallery_labels), 'gallery_coverage': 1.0}
    for k in (1, 5):
        correct = hits[:, :k].any(1)
        result[f'Recall@{k}'] = float(correct.mean())
        result[f'macro_Recall@{k}'] = float(pd.Series(correct).groupby(query_labels).mean().mean())
    return result


def validate_manifest(frame):
    required = {'image_path', 'label', 'split', 'source_sha256', 'crop_sha256'}
    if not required.issubset(frame):
        raise ValueError(f'Missing columns: {required-set(frame)}')
    if set(frame.split) != {'train', 'val', 'test'}:
        raise ValueError('Manifest must have train, val and test rows')
    if frame.image_path.duplicated().any():
        raise ValueError('Duplicate paths in manifest')
    for column in ('source_sha256', 'crop_sha256'):
        if frame[column].duplicated().any():
            raise ValueError(f'Duplicate {column}: run preparation deduplication')
    if (frame.groupby('label').split.nunique() != 1).any():
        raise ValueError('Wine classes overlap between train/val/test')
    for split in ('val', 'test'):
        if (frame.loc[frame.split == split].groupby('label').size() < 2).any():
            raise ValueError(f'{split} contains singleton classes: no positive after self-exclusion')


def load_checkpoint(path, device):
    # Only load checkpoints produced by this project (torch pickle is not for untrusted files).
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    args = {**checkpoint['args'], 'dataset_sha256': checkpoint['dataset_sha256']}
    model = WineModel(embedding_dim=args['embedding_dim'], train_blocks=0,
                      checkpointing=False, pretrained=False, config=checkpoint['backbone_config'])
    model.load_state_dict(checkpoint['model'])
    return model.to(device).eval(), args

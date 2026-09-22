"""Class-disjoint all-vs-all retrieval within val or test, excluding each query itself."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from .core import ROOT, encode, exact_topk, load_checkpoint, retrieval_metrics, validate_manifest
from .prepare import sha_file


def evaluate(model, queries, gallery, device, precision, batch_size, workers, height, width):
    print(f'Encoding gallery: {len(gallery):,}; queries: {len(queries):,}', flush=True)
    gallery_vectors = encode(model, gallery, device, precision, batch_size, workers, height, width)
    lookup = pd.Series(np.arange(len(gallery)), index=gallery.image_path)
    if gallery.image_path.duplicated().any() or not queries.image_path.isin(gallery.image_path).all():
        raise ValueError('All-vs-all queries must be unique members of the evaluation split gallery')
    self_indices = lookup.loc[queries.image_path].to_numpy()
    query_vectors = gallery_vectors[self_indices]
    # TF32 is helpful for training but use FP32 similarity for reproducible ranking.
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        indices, scores = exact_topk(query_vectors, gallery_vectors, device, exclude_indices=self_indices)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    metrics = retrieval_metrics(queries.label.values, gallery.label.values, indices, self_indices)
    return metrics, indices, scores, query_vectors, gallery_vectors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv', type=Path, default=ROOT / 'data/winesensed_embedding/dataset.csv')
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--split', choices=['val', 'test'], default='test')
    p.add_argument('--output', type=Path)
    p.add_argument('--device', default='cuda')
    p.add_argument('--precision', choices=['fp16', 'bf16', 'fp32'],
                   help='Defaults to the training precision recorded in the checkpoint.')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--save-embeddings', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    model, config = load_checkpoint(args.checkpoint, device)
    args.precision = args.precision or config['precision']
    if sha_file(args.csv) != config['dataset_sha256']:
        raise ValueError('Evaluation CSV differs from the manifest used to train this checkpoint.')
    frame = pd.read_csv(args.csv)
    validate_manifest(frame)
    queries = frame[frame.split == args.split].reset_index(drop=True)
    gallery = queries.copy()
    metrics, indices, scores, qv, gv = evaluate(model, queries, gallery, device, args.precision,
        args.batch_size, args.workers, config['height'], config['width'])
    output = args.output or args.checkpoint.parent / args.split
    output.mkdir(parents=True, exist_ok=True)
    metrics.update(split=args.split, checkpoint=str(args.checkpoint.resolve()), precision=args.precision,
                   protocol='all-vs-all within the held-out split; exact cosine top-5; self-match excluded; classes disjoint from training')
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    result = queries[['image_path', 'label', 'wine_id']].copy()
    for rank in range(indices.shape[1]):
        result[f'label_{rank+1}'] = gallery.label.values[indices[:, rank]]
        result[f'score_{rank+1}'] = scores[:, rank]
        result[f'path_{rank+1}'] = gallery.image_path.values[indices[:, rank]]
    result.to_csv(output / 'predictions.csv', index=False)
    if args.save_embeddings:
        np.save(output / 'query_embeddings.npy', qv)
        np.save(output / 'gallery_embeddings.npy', gv)
        queries.to_csv(output / 'queries.csv', index=False)
        gallery.to_csv(output / 'gallery.csv', index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(['Recall@1', 'Recall@5'], [metrics['Recall@1'], metrics['Recall@5']])
    ax.set(ylim=(0, 1), ylabel='Recall', title=f'Wine retrieval — {args.split}')
    fig.tight_layout()
    fig.savefig(output / 'recall.png', dpi=160)
    plt.close(fig)
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()

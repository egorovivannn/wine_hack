"""DINOv2-L wine embedding baseline: mixed-precision fine-tuning with SupCon."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch

from .core import (MODEL_PATH, ROOT, PKSampler, WineDataset, WineModel, amp_context,
                   make_loader, seed_everything, supervised_contrastive,
                   validate_manifest)
from .evaluate import evaluate
from .prepare import sha_file


def plot_history(history, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    frame = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
    axes[0].plot(frame.epoch, frame.train_loss, label='train SupCon', marker='o', markersize=3)
    axes[0].plot(frame.epoch, frame.val_loss, label='val SupCon (real pairs)', marker='o', markersize=3)
    axes[0].set(xlabel='Epoch', ylabel='Loss')
    axes[1].plot(frame.epoch, frame['Recall@1'], label='val Recall@1', marker='o', markersize=3)
    axes[1].plot(frame.epoch, frame['Recall@5'], label='val Recall@5', marker='o', markersize=3)
    axes[1].set(xlabel='Epoch', ylabel='Recall', ylim=(0, 1))
    axes[2].plot(frame.epoch, frame.backbone_lr, label='backbone LR', marker='o', markersize=3)
    axes[2].plot(frame.epoch, frame.head_lr, label='head LR', marker='o', markersize=3)
    axes[2].set(xlabel='Epoch', ylabel='Learning rate')
    for ax in axes:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlim(.5, max(frame.epoch)+.5)
        ax.grid(alpha=.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / 'curves.png', dpi=160)
    plt.close(fig)


@torch.inference_mode()
def validation_loss(model, loader, device, precision, temperature):
    if loader is None:
        return float('nan')
    model.eval()
    losses = []
    for pixels, labels in loader:
        with amp_context(device, precision):
            z = model(pixels.to(device, non_blocking=True))
        losses.append(float(supervised_contrastive(z, labels.to(device), temperature)))
    return float(np.mean(losses))


def validation_subset(frame, args):
    queries = frame[frame.split == 'val'].copy()
    if args.val_query_limit:
        queries = queries.sample(min(len(queries), args.val_query_limit), random_state=args.seed)
    gallery = frame[frame.split == 'val'].copy()
    return queries.reset_index(drop=True), gallery.reset_index(drop=True)


def save_checkpoint(path, model, optimizer, scheduler, scaler, args, epoch, best, history, dataset_hash,
                    inference_only=False):
    payload = dict(model=model.state_dict(),
                   args={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                   backbone_config=model.backbone.config.to_dict(), epoch=epoch,
                   best_recall=best, history=history, dataset_sha256=dataset_hash)
    if not inference_only:
        payload.update(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                       scaler=scaler.state_dict(),
                       rng=dict(python=random.getstate(), numpy=np.random.get_state(),
                                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all()))
    temp = path.with_suffix('.tmp')
    torch.save(payload, temp)
    temp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv', type=Path, default=ROOT / 'data/winesensed_embedding/dataset.csv')
    p.add_argument('--model-path', type=Path, default=MODEL_PATH)
    p.add_argument('--output', type=Path, default=ROOT / 'embedding/runs/dinov2_large')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--images-per-wine', type=int, default=2)
    p.add_argument('--accumulation', type=int, default=2)
    p.add_argument('--steps-per-epoch', type=int, default=2000,
                   help='Class-balanced sampled batches per epoch; 0 means ceil(train images/batch).')
    p.add_argument('--train-blocks', type=int, default=24)
    p.add_argument('--embedding-dim', type=int, default=512)
    p.add_argument('--height', type=int, default=518)
    p.add_argument('--width', type=int, default=224)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--head-lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=.01)
    p.add_argument('--temperature', type=float, default=.07)
    p.add_argument('--warmup-epochs', type=float, default=1.)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--eval-batch-size', type=int, default=64)
    p.add_argument('--val-query-limit', type=int, default=0, help='0 = full all-vs-all validation. Optional fixed query subset still searches the full val gallery.')
    p.add_argument('--val-loss-batches', type=int, default=100)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--precision', choices=['fp16', 'bf16', 'fp32'], default='bf16')
    p.add_argument('--device', default='cuda')
    p.add_argument('--no-checkpointing', action='store_true')
    p.add_argument('--resume', type=Path)
    args = p.parse_args()
    if min(args.epochs, args.accumulation, args.batch_size, args.height, args.width) < 1:
        p.error('Epochs, accumulation, batch size and input size must be positive')
    if args.height % 14 or args.width % 14:
        p.error('DINOv2 input sides must be multiples of 14')
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; run on your GPU host.')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    seed_everything(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'last.pt').exists() and args.resume is None:
        raise ValueError('Output contains a previous run: pass --resume or choose a new --output.')
    (args.output / 'train.pid').write_text(str(os.getpid()) + '\n')
    frame = pd.read_csv(args.csv)
    validate_manifest(frame)
    dataset_hash = sha_file(args.csv)
    train_frame = frame[frame.split == 'train'].reset_index(drop=True)
    train_data = WineDataset(train_frame, args.height, args.width, train=True)
    sampler = PKSampler(train_data.labels, args.batch_size, args.images_per_wine,
                        args.steps_per_epoch, args.seed)
    train_loader = make_loader(train_data, args.workers, sampler=sampler, seed=args.seed)
    queries, gallery = validation_subset(frame, args)
    queries.to_csv(args.output / 'monitor_queries.csv', index=False)
    gallery.to_csv(args.output / 'monitor_gallery.csv', index=False)
    val_data = WineDataset(queries, args.height, args.width)
    try:
        eligible_classes = int((pd.Series(val_data.labels).value_counts() >= args.images_per_wine).sum())
        val_batch_size = min(args.batch_size, eligible_classes * args.images_per_wine)
        val_sampler = PKSampler(val_data.labels, val_batch_size, args.images_per_wine,
                                args.val_loss_batches, args.seed, allow_repeat=False)
        val_loader = make_loader(val_data, args.workers, sampler=val_sampler)
    except ValueError:
        val_loader = None
        print('Too few real repeated wine photos for val SupCon; retrieval metrics remain valid.', flush=True)
    model = WineModel(args.model_path, args.embedding_dim, args.train_blocks,
                      not args.no_checkpointing).to(device)
    optimizer = torch.optim.AdamW([
        dict(params=[p for p in model.backbone.parameters() if p.requires_grad], lr=args.lr),
        dict(params=model.head.parameters(), lr=args.head_lr)], weight_decay=args.weight_decay,
        fused=device.type == 'cuda')
    updates_per_epoch = math.ceil(len(train_loader) / args.accumulation)
    total_updates = updates_per_epoch * args.epochs
    warmup = min(round(updates_per_epoch * args.warmup_epochs), max(total_updates-1, 0))

    def lr_schedule(step):
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        progress = min((step - warmup) / max(total_updates-warmup, 1), 1.)
        return .05 + .95 * .5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
    scaler = torch.amp.GradScaler('cuda', init_scale=1024., enabled=device.type == 'cuda' and args.precision == 'fp16')
    start_epoch, best, history = 0, -1., []
    if args.resume:
        state = torch.load(args.resume, map_location='cpu', weights_only=False)
        if 'optimizer' not in state:
            raise ValueError('Resume requires last.pt; best.pt is an inference-only export.')
        if state['dataset_sha256'] != dataset_hash:
            raise ValueError('Resume dataset hash differs from checkpoint')
        for key in ('epochs', 'batch_size', 'images_per_wine', 'accumulation', 'steps_per_epoch',
                    'train_blocks', 'embedding_dim', 'height', 'width', 'lr', 'head_lr',
                    'weight_decay', 'temperature', 'warmup_epochs', 'seed', 'precision',
                    'val_query_limit'):
            if state['args'][key] != getattr(args, key):
                raise ValueError(f'Resume option changed: {key}')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        start_epoch, best, history = state['epoch'], state['best_recall'], state['history']
        random.setstate(state['rng']['python'])
        np.random.set_state(state['rng']['numpy'])
        torch.set_rng_state(state['rng']['torch'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(state['rng']['cuda'])
        del state
    run_config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    run_config.update(dataset_sha256=dataset_hash, train_images=len(train_data),
                      total_parameters=sum(p.numel() for p in model.parameters()),
                      trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                      monitor_queries=len(queries), monitor_gallery=len(gallery),
                      negatives_per_anchor=args.batch_size-args.images_per_wine,
                      notes='Accumulation increases optimizer batch only, not contrastive negatives. Train singleton positives are two mild color views. No horizontal flips or random label-cutting crops.')
    (args.output / 'config.json').write_text(json.dumps(run_config, indent=2))
    print(json.dumps(run_config, indent=2), flush=True)
    for epoch in range(start_epoch, args.epochs):
        started = time.monotonic()
        sampler.epoch = epoch
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses, skipped_updates = [], 0
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        for step, (pixels, labels) in enumerate(train_loader):
            pixels, labels = pixels.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            group_start = (step // args.accumulation) * args.accumulation
            divisor = min(args.accumulation, len(train_loader) - group_start)
            with amp_context(device, args.precision):
                z = model(pixels)
            loss = supervised_contrastive(z, labels, args.temperature)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss at epoch {epoch+1}, step {step+1}')
            scaler.scale(loss / divisor).backward()
            losses.append(float(loss.detach()))
            if (step + 1) % args.accumulation == 0 or step + 1 == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= previous_scale:
                    scheduler.step()
                else:
                    skipped_updates += 1
                optimizer.zero_grad(set_to_none=True)
            if (step + 1) % 50 == 0 or step + 1 == len(train_loader):
                print(f'epoch {epoch+1}/{args.epochs} step {step+1}/{len(train_loader)} '
                      f'loss {np.mean(losses[-50:]):.4f} | {(step+1)*args.batch_size/(time.monotonic()-started):.1f} images/s', flush=True)
        if skipped_updates >= updates_per_epoch:
            raise FloatingPointError('All optimizer updates skipped due to AMP overflow')
        val_loss = validation_loss(model, val_loader, device, args.precision, args.temperature)
        metrics, _, _, _, _ = evaluate(model, queries, gallery, device, args.precision,
            args.eval_batch_size, args.workers, args.height, args.width)
        record = dict(epoch=epoch+1, train_loss=float(np.mean(losses)), val_loss=val_loss,
                      **metrics, backbone_lr=optimizer.param_groups[0]['lr'],
                      head_lr=optimizer.param_groups[1]['lr'], seconds=time.monotonic()-started,
                      amp_skipped_updates=skipped_updates,
                      peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30 if device.type == 'cuda' else 0.)
        history.append(record)
        pd.DataFrame(history).to_csv(args.output / 'history.csv', index=False)
        plot_history(history, args.output)
        improved = metrics['Recall@1'] > best
        best = max(best, metrics['Recall@1'])
        save_checkpoint(args.output / 'last.pt', model, optimizer, scheduler, scaler, args,
                        epoch+1, best, history, dataset_hash)
        if improved:
            save_checkpoint(args.output / 'best.pt', model, optimizer, scheduler, scaler, args,
                            epoch+1, best, history, dataset_hash, inference_only=True)
        print(json.dumps(record, indent=2), flush=True)
    print(f'Completed. Best checkpoint: {args.output / "best.pt"}', flush=True)


if __name__ == '__main__':
    main()

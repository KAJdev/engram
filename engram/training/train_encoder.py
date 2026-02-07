"""phase 1: multi head encoder training with multi task contrastive loss."""

from __future__ import annotations

import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from engram.config import EngramConfig
from engram.models.encoder import MultiHeadEncoder
from engram.training.datasets import EncoderContrastiveDataset
from engram.training.losses import MultiTaskEncoderLoss


def _param_count(model: torch.nn.Module) -> str:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"{total:,} total, {trainable:,} trainable"


def _gpu_mem() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    alloc = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    return f"{alloc:.2f}GB allocated, {reserved:.2f}GB reserved"


def train_encoder(config: EngramConfig, data_dir: Path) -> MultiHeadEncoder:
    device = torch.device(config.training.device)
    tc = config.training

    print(f"  device: {device}")
    print(f"  backbone: {config.encoder.backbone}")
    print(f"  backbone_dim: {config.encoder.backbone_dim}")
    print(f"  total_dim: {config.encoder.total_dim}")
    print(f"  head_dims: {config.encoder.head_dims}")
    print(f"  epochs: {tc.encoder_epochs}, batch_size: {tc.encoder_batch_size}")
    print(f"  lr: {tc.encoder_lr}, backbone_lr: {tc.encoder_backbone_lr}")
    print(f"  temperature: {tc.contrastive_temperature}")

    encoder = MultiHeadEncoder(config.encoder).to(device)
    print(f"  encoder params: {_param_count(encoder)}")
    print(f"  gpu memory after model load: {_gpu_mem()}")

    head_names = list(config.encoder.head_dims.keys())
    criterion = MultiTaskEncoderLoss(head_names, temperature=tc.contrastive_temperature).to(device)

    # one dataset per head type
    datasets = {
        "semantic": EncoderContrastiveDataset(data_dir, head_type="semantic"),
        "entity": EncoderContrastiveDataset(data_dir, head_type="entity"),
        "theme": EncoderContrastiveDataset(data_dir, head_type="theme"),
    }

    for name, ds in datasets.items():
        print(f"  dataset '{name}': {len(ds)} triplets")

    primary_ds = datasets["semantic"]
    entity_ds = datasets["entity"]
    theme_ds = datasets["theme"]

    effective_batch_size = min(tc.encoder_batch_size, len(primary_ds))
    if effective_batch_size < tc.encoder_batch_size:
        print(f"  clamping batch size {tc.encoder_batch_size} -> {effective_batch_size} (dataset too small)")

    primary_loader = DataLoader(primary_ds, batch_size=effective_batch_size, shuffle=True, drop_last=len(primary_ds) > effective_batch_size)
    print(f"  effective batch size: {effective_batch_size}")
    print(f"  batches per epoch: {len(primary_loader)}")

    # differential lr: backbone slower than heads
    backbone_params = list(encoder.backbone.parameters())
    head_params = [p for n, p in encoder.named_parameters() if "backbone" not in n]
    loss_params = list(criterion.parameters())

    optimizer = AdamW([
        {"params": backbone_params, "lr": tc.encoder_backbone_lr},
        {"params": head_params, "lr": tc.encoder_lr},
        {"params": loss_params, "lr": tc.encoder_lr},
    ], weight_decay=0.01)

    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, tc.encoder_epochs * len(primary_loader)))

    best_loss = float("inf")
    save_dir = tc.output_dir / "encoder"
    save_dir.mkdir(parents=True, exist_ok=True)

    total_steps = tc.encoder_epochs * len(primary_loader)
    print(f"  total training steps: {total_steps}")
    print(f"  save dir: {save_dir}")
    print()

    train_start = time.time()

    for epoch in range(tc.encoder_epochs):
        epoch_start = time.time()
        encoder.train()
        criterion.train()
        epoch_losses = []
        head_losses: dict[str, list[float]] = {name: [] for name in head_names}

        pbar = tqdm(primary_loader, desc=f"Encoder epoch {epoch+1}/{tc.encoder_epochs}")
        for batch_idx, (anchors, positives, negatives) in enumerate(pbar):
            anchor_tok = encoder.tokenizer(anchors, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            pos_tok = encoder.tokenizer(positives, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            neg_tok = encoder.tokenizer(negatives, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)

            anchor_heads = encoder(anchor_tok["input_ids"], anchor_tok["attention_mask"])
            pos_heads = encoder(pos_tok["input_ids"], pos_tok["attention_mask"])
            neg_heads = encoder(neg_tok["input_ids"], neg_tok["attention_mask"])

            loss, loss_dict = criterion(anchor_heads, pos_heads, neg_heads)

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_losses.append(loss_dict["total"])
            for name in head_names:
                if name in loss_dict:
                    head_losses[name].append(loss_dict[name])

            pbar.set_postfix(
                loss=f"{loss_dict['total']:.4f}",
                sem=f"{loss_dict.get('semantic', 0):.3f}",
                grad=f"{grad_norm:.3f}",
            )

        epoch_time = time.time() - epoch_start
        if not epoch_losses:
            print(f"  epoch {epoch+1}: no batches (dataset too small for batch size)")
            continue
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        lr_backbone = optimizer.param_groups[0]["lr"]
        lr_heads = optimizer.param_groups[1]["lr"]

        head_summary = ", ".join(
            f"{name}={sum(v)/len(v):.4f}" for name, v in head_losses.items() if v
        )

        print(f"  epoch {epoch+1}/{tc.encoder_epochs}: loss={avg_loss:.4f} | {head_summary}")
        print(f"    lr: backbone={lr_backbone:.2e}, heads={lr_heads:.2e} | grad_norm={grad_norm:.3f}")
        print(f"    time: {epoch_time:.1f}s | gpu: {_gpu_mem()}")

        if avg_loss < best_loss:
            improvement = best_loss - avg_loss if best_loss != float("inf") else 0
            best_loss = avg_loss
            torch.save({
                "encoder": encoder.state_dict(),
                "criterion": criterion.state_dict(),
                "epoch": epoch,
                "loss": best_loss,
            }, save_dir / "best.pt")
            print(f"    saved new best (improvement: {improvement:.4f})")

    total_time = time.time() - train_start
    torch.save(encoder.state_dict(), save_dir / "final.pt")
    print(f"  encoder training complete in {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"  best loss: {best_loss:.4f}")
    return encoder

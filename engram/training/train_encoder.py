"""phase 1: multi head encoder training with multi task contrastive loss."""

from __future__ import annotations

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


def train_encoder(config: EngramConfig, data_dir: Path) -> MultiHeadEncoder:
    device = torch.device(config.training.device)
    tc = config.training

    encoder = MultiHeadEncoder(config.encoder).to(device)
    head_names = list(config.encoder.head_dims.keys())
    criterion = MultiTaskEncoderLoss(head_names, temperature=tc.contrastive_temperature).to(device)

    # one dataset per head type
    datasets = {
        "semantic": EncoderContrastiveDataset(data_dir, head_type="semantic"),
        "entity": EncoderContrastiveDataset(data_dir, head_type="entity"),
        "theme": EncoderContrastiveDataset(data_dir, head_type="theme"),
    }

    # semantic dataset is the primary one
    primary_ds = datasets["semantic"]
    entity_ds = datasets["entity"]
    theme_ds = datasets["theme"]

    primary_loader = DataLoader(primary_ds, batch_size=tc.encoder_batch_size, shuffle=True, drop_last=True)

    # differential lr: backbone slower than heads
    backbone_params = list(encoder.backbone.parameters())
    head_params = [p for n, p in encoder.named_parameters() if "backbone" not in n]
    loss_params = list(criterion.parameters())

    optimizer = AdamW([
        {"params": backbone_params, "lr": tc.encoder_backbone_lr},
        {"params": head_params, "lr": tc.encoder_lr},
        {"params": loss_params, "lr": tc.encoder_lr},
    ], weight_decay=0.01)

    scheduler = CosineAnnealingLR(optimizer, T_max=tc.encoder_epochs * len(primary_loader))

    best_loss = float("inf")
    save_dir = tc.output_dir / "encoder"
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(tc.encoder_epochs):
        encoder.train()
        criterion.train()
        epoch_losses = []

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
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_losses.append(loss_dict["total"])
            pbar.set_postfix(loss=f"{loss_dict['total']:.4f}", sem=f"{loss_dict.get('semantic', 0):.3f}")

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch+1}: avg_loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "encoder": encoder.state_dict(),
                "criterion": criterion.state_dict(),
                "epoch": epoch,
                "loss": best_loss,
            }, save_dir / "best.pt")

    torch.save(encoder.state_dict(), save_dir / "final.pt")
    print(f"Encoder training complete. Best loss: {best_loss:.4f}")
    return encoder

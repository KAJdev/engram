"""phase 2: edge classifier training on frozen encoder representations."""

from __future__ import annotations

import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from engram.config import EngramConfig
from engram.models.encoder import MultiHeadEncoder
from engram.models.edge_classifier import EdgeClassifier
from engram.training.datasets import EdgeClassifierDataset
from engram.training.losses import EdgeClassifierLoss


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


def _collate_edge_batch(batch: list[dict]) -> dict:
    return {
        "text_a": [b["text_a"] for b in batch],
        "text_b": [b["text_b"] for b in batch],
        "edge_exists": torch.tensor([b["edge_exists"] for b in batch], dtype=torch.float32),
        "edge_type_idx": torch.tensor([b["edge_type_idx"] for b in batch], dtype=torch.long),
        "edge_weight": torch.tensor([b["edge_weight"] for b in batch], dtype=torch.float32),
    }


def train_edge_classifier(
    config: EngramConfig,
    data_dir: Path,
    encoder: MultiHeadEncoder | None = None,
) -> EdgeClassifier:
    device = torch.device(config.training.device)
    tc = config.training

    print(f"  device: {device}")
    print(f"  hidden_dims: {config.edge_classifier.hidden_dims}")
    print(f"  num_edge_types: {config.edge_classifier.num_edge_types}")
    print(f"  epochs: {tc.edge_epochs}, batch_size: {tc.edge_batch_size}")
    print(f"  lr: {tc.edge_lr}")

    # load frozen encoder
    if encoder is None:
        encoder = MultiHeadEncoder(config.encoder)
        ckpt_path = tc.output_dir / "encoder" / "best.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
            encoder.load_state_dict(ckpt["encoder"])
            print(f"  loaded encoder from {ckpt_path}")
        else:
            print(f"  warning: no encoder checkpoint at {ckpt_path}, using random init")
    else:
        print("  using encoder passed from previous phase")
    encoder = encoder.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    classifier = EdgeClassifier(config.encoder, config.edge_classifier).to(device)
    print(f"  classifier params: {_param_count(classifier)}")
    print(f"  input dim: {config.encoder.total_dim * 2}")

    criterion = EdgeClassifierLoss()

    dataset = EdgeClassifierDataset(data_dir)
    print(f"  dataset: {len(dataset)} edge pairs")

    effective_batch_size = min(tc.edge_batch_size, len(dataset))
    if effective_batch_size < tc.edge_batch_size:
        print(f"  clamping batch size {tc.edge_batch_size} -> {effective_batch_size} (dataset too small)")

    loader = DataLoader(
        dataset, batch_size=effective_batch_size, shuffle=True,
        collate_fn=_collate_edge_batch, drop_last=len(dataset) > effective_batch_size,
    )
    print(f"  effective batch size: {effective_batch_size}")
    print(f"  batches per epoch: {len(loader)}")

    optimizer = AdamW(classifier.parameters(), lr=tc.edge_lr, weight_decay=0.01)

    save_dir = tc.output_dir / "edge_classifier"
    save_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    total_steps = tc.edge_epochs * len(loader)
    print(f"  total training steps: {total_steps}")
    print(f"  save dir: {save_dir}")
    print(f"  gpu memory after setup: {_gpu_mem()}")
    print()

    train_start = time.time()

    for epoch in range(tc.edge_epochs):
        epoch_start = time.time()
        classifier.train()
        epoch_losses = []
        exists_losses = []
        type_losses = []
        weight_losses = []

        pbar = tqdm(loader, desc=f"EdgeClassifier epoch {epoch+1}/{tc.edge_epochs}")
        for batch in pbar:
            with torch.no_grad():
                tok_a = encoder.tokenizer(batch["text_a"], padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
                tok_b = encoder.tokenizer(batch["text_b"], padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
                heads_a = encoder(tok_a["input_ids"], tok_a["attention_mask"])
                heads_b = encoder(tok_b["input_ids"], tok_b["attention_mask"])
                repr_a = encoder.concat_representations(heads_a)
                repr_b = encoder.concat_representations(heads_b)

            preds = classifier(repr_a, repr_b)

            loss, loss_dict = criterion(
                preds["edge_exists"],
                preds["edge_type"],
                preds["edge_weight"],
                batch["edge_exists"].to(device),
                batch["edge_type_idx"].to(device),
                batch["edge_weight"].to(device),
            )

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
            optimizer.step()

            epoch_losses.append(loss_dict["total"])
            exists_losses.append(loss_dict.get("exists", 0))
            type_losses.append(loss_dict.get("type", 0))
            weight_losses.append(loss_dict.get("weight", 0))
            pbar.set_postfix(
                loss=f"{loss_dict['total']:.4f}",
                exists=f"{loss_dict.get('exists', 0):.3f}",
                grad=f"{grad_norm:.3f}",
            )

        epoch_time = time.time() - epoch_start
        if not epoch_losses:
            print(f"  epoch {epoch+1}: no batches (dataset too small for batch size)")
            continue
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        avg_exists = sum(exists_losses) / len(exists_losses) if exists_losses else 0
        avg_type = sum(type_losses) / len(type_losses) if type_losses else 0
        avg_weight = sum(weight_losses) / len(weight_losses) if weight_losses else 0
        lr_now = optimizer.param_groups[0]["lr"]

        print(f"  epoch {epoch+1}/{tc.edge_epochs}: loss={avg_loss:.4f} | exists={avg_exists:.4f}, type={avg_type:.4f}, weight={avg_weight:.4f}")
        print(f"    lr={lr_now:.2e} | grad_norm={grad_norm:.3f} | time: {epoch_time:.1f}s | gpu: {_gpu_mem()}")

        if avg_loss < best_loss:
            improvement = best_loss - avg_loss if best_loss != float("inf") else 0
            best_loss = avg_loss
            torch.save({
                "classifier": classifier.state_dict(),
                "epoch": epoch,
                "loss": best_loss,
            }, save_dir / "best.pt")
            print(f"    saved new best (improvement: {improvement:.4f})")

    total_time = time.time() - train_start
    torch.save(classifier.state_dict(), save_dir / "final.pt")
    print(f"  edge classifier training complete in {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"  best loss: {best_loss:.4f}")
    return classifier

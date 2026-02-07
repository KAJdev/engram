"""phase 3: synthesis encoder training with contrastive triplets."""

from __future__ import annotations

import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from engram.config import EngramConfig
from engram.models.encoder import MultiHeadEncoder
from engram.models.synthesis import SynthesisEncoder
from engram.training.datasets import SynthesisDataset
from engram.training.losses import SynthesisContrastiveLoss


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


def _collate_synthesis_batch(batch: list[tuple[str, str, str]]) -> dict:
    texts_a, texts_b, queries = zip(*batch)
    return {
        "text_a": list(texts_a),
        "text_b": list(texts_b),
        "query": list(queries),
    }


def train_synthesis(
    config: EngramConfig,
    data_dir: Path,
    encoder: MultiHeadEncoder | None = None,
) -> SynthesisEncoder:
    device = torch.device(config.training.device)
    tc = config.training

    print(f"  device: {device}")
    print(f"  synthesis config: layers={config.synthesis.num_layers}, heads={config.synthesis.num_heads}, hidden={config.synthesis.hidden_dim}")
    print(f"  output_dim: {config.synthesis.output_dim}")
    print(f"  epochs: {tc.synthesis_epochs}, batch_size: {tc.synthesis_batch_size}")
    print(f"  lr: {tc.synthesis_lr}, temperature: {tc.synthesis_temperature}")

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

    synthesis = SynthesisEncoder(config.encoder, config.synthesis).to(device)
    print(f"  synthesis params: {_param_count(synthesis)}")

    criterion = SynthesisContrastiveLoss(temperature=tc.synthesis_temperature)

    dataset = SynthesisDataset(data_dir)
    print(f"  dataset: {len(dataset)} synthesis triplets")

    effective_batch_size = min(tc.synthesis_batch_size, len(dataset))
    if effective_batch_size < tc.synthesis_batch_size:
        print(f"  clamping batch size {tc.synthesis_batch_size} -> {effective_batch_size} (dataset too small)")

    loader = DataLoader(
        dataset, batch_size=effective_batch_size, shuffle=True,
        collate_fn=_collate_synthesis_batch, drop_last=len(dataset) > effective_batch_size,
    )
    print(f"  effective batch size: {effective_batch_size}")
    print(f"  batches per epoch: {len(loader)}")

    optimizer = AdamW(synthesis.parameters(), lr=tc.synthesis_lr, weight_decay=0.01)

    save_dir = tc.output_dir / "synthesis"
    save_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    total_steps = tc.synthesis_epochs * len(loader)
    print(f"  total training steps: {total_steps}")
    print(f"  save dir: {save_dir}")
    print(f"  gpu memory after setup: {_gpu_mem()}")
    print()

    train_start = time.time()

    for epoch in range(tc.synthesis_epochs):
        epoch_start = time.time()
        synthesis.train()
        epoch_losses = []

        pbar = tqdm(loader, desc=f"Synthesis epoch {epoch+1}/{tc.synthesis_epochs}")
        for batch in pbar:
            with torch.no_grad():
                tok_a = encoder.tokenizer(batch["text_a"], padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
                tok_b = encoder.tokenizer(batch["text_b"], padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
                tok_q = encoder.tokenizer(batch["query"], padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)

                heads_a = encoder(tok_a["input_ids"], tok_a["attention_mask"])
                heads_b = encoder(tok_b["input_ids"], tok_b["attention_mask"])
                heads_q = encoder(tok_q["input_ids"], tok_q["attention_mask"])

                repr_a = encoder.concat_representations(heads_a)
                repr_b = encoder.concat_representations(heads_b)
                query_semantic = heads_q["semantic"]

            synth_vec = synthesis(repr_a, repr_b)
            loss = criterion(synth_vec, query_semantic)

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(synthesis.parameters(), 1.0)
            optimizer.step()

            epoch_losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}", grad=f"{grad_norm:.3f}")

        epoch_time = time.time() - epoch_start
        if not epoch_losses:
            print(f"  epoch {epoch+1}: no batches (dataset too small for batch size)")
            continue
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        lr_now = optimizer.param_groups[0]["lr"]

        print(f"  epoch {epoch+1}/{tc.synthesis_epochs}: loss={avg_loss:.4f}")
        print(f"    lr={lr_now:.2e} | grad_norm={grad_norm:.3f} | time: {epoch_time:.1f}s | gpu: {_gpu_mem()}")

        if avg_loss < best_loss:
            improvement = best_loss - avg_loss if best_loss != float("inf") else 0
            best_loss = avg_loss
            torch.save({
                "synthesis": synthesis.state_dict(),
                "epoch": epoch,
                "loss": best_loss,
            }, save_dir / "best.pt")
            print(f"    saved new best (improvement: {improvement:.4f})")

    total_time = time.time() - train_start
    torch.save(synthesis.state_dict(), save_dir / "final.pt")
    print(f"  synthesis training complete in {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"  best loss: {best_loss:.4f}")
    return synthesis

"""phase 3: synthesis encoder training with contrastive triplets."""

from __future__ import annotations

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

    # load frozen encoder
    if encoder is None:
        encoder = MultiHeadEncoder(config.encoder)
        ckpt_path = tc.output_dir / "encoder" / "best.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
            encoder.load_state_dict(ckpt["encoder"])
    encoder = encoder.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    synthesis = SynthesisEncoder(config.encoder, config.synthesis).to(device)
    criterion = SynthesisContrastiveLoss(temperature=tc.synthesis_temperature)

    dataset = SynthesisDataset(data_dir)
    loader = DataLoader(
        dataset, batch_size=tc.synthesis_batch_size, shuffle=True,
        collate_fn=_collate_synthesis_batch, drop_last=True,
    )

    optimizer = AdamW(synthesis.parameters(), lr=tc.synthesis_lr, weight_decay=0.01)

    save_dir = tc.output_dir / "synthesis"
    save_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(tc.synthesis_epochs):
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
                query_semantic = heads_q["semantic"]  # synthesis targets semantic space

            synth_vec = synthesis(repr_a, repr_b)
            loss = criterion(synth_vec, query_semantic)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(synthesis.parameters(), 1.0)
            optimizer.step()

            epoch_losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch+1}: avg_loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "synthesis": synthesis.state_dict(),
                "epoch": epoch,
                "loss": best_loss,
            }, save_dir / "best.pt")

    torch.save(synthesis.state_dict(), save_dir / "final.pt")
    print(f"Synthesis training complete. Best loss: {best_loss:.4f}")
    return synthesis

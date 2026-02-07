"""phase 2: edge classifier training on frozen encoder representations."""

from __future__ import annotations

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

    classifier = EdgeClassifier(config.encoder, config.edge_classifier).to(device)
    criterion = EdgeClassifierLoss()

    dataset = EdgeClassifierDataset(data_dir)
    loader = DataLoader(
        dataset, batch_size=tc.edge_batch_size, shuffle=True,
        collate_fn=_collate_edge_batch, drop_last=True,
    )

    optimizer = AdamW(classifier.parameters(), lr=tc.edge_lr, weight_decay=0.01)

    save_dir = tc.output_dir / "edge_classifier"
    save_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(tc.edge_epochs):
        classifier.train()
        epoch_losses = []

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
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
            optimizer.step()

            epoch_losses.append(loss_dict["total"])
            pbar.set_postfix(loss=f"{loss_dict['total']:.4f}", exists=f"{loss_dict['exists']:.3f}")

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch+1}: avg_loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "classifier": classifier.state_dict(),
                "epoch": epoch,
                "loss": best_loss,
            }, save_dir / "best.pt")

    torch.save(classifier.state_dict(), save_dir / "final.pt")
    print(f"Edge classifier training complete. Best loss: {best_loss:.4f}")
    return classifier

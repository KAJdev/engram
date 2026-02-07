"""multi head encoder: shared backbone with multiple projection heads."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from engram.config import EncoderConfig


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


class MultiHeadEncoder(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        self.backbone = AutoModel.from_pretrained(config.backbone)
        self.tokenizer = AutoTokenizer.from_pretrained(config.backbone)

        if config.freeze_backbone_layers > 0:
            for i, layer in enumerate(self.backbone.encoder.layer):
                if i < config.freeze_backbone_layers:
                    for param in layer.parameters():
                        param.requires_grad = False

        dim = config.backbone_dim

        self.semantic_head = ProjectionHead(dim, config.semantic_dim)
        self.entity_head = ProjectionHead(dim, config.entity_dim)
        self.theme_head = ProjectionHead(dim, config.theme_dim)
        self.temporal_head = ProjectionHead(dim, config.temporal_dim)

        # learned heads, optimized end to end
        self.learned_heads = nn.ModuleList(
            [ProjectionHead(dim, config.learned_dim) for _ in range(config.num_learned_heads)]
        )

        self._head_names = list(config.head_dims.keys())

    def _pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.config.pooling == "cls":
            return hidden_states[:, 0]
        mask = attention_mask.unsqueeze(-1).float()
        return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._pool(outputs.last_hidden_state, attention_mask)

        result = {
            "semantic": self.semantic_head(pooled),
            "entity": self.entity_head(pooled),
            "theme": self.theme_head(pooled),
            "temporal": self.temporal_head(pooled),
        }
        for i, head in enumerate(self.learned_heads):
            result[f"learned_{i}"] = head(pooled)

        return result

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        device: str | torch.device = "cpu",
    ) -> dict[str, torch.Tensor]:
        """encode a list of texts into multi head representations."""
        self.eval()
        all_results: dict[str, list[torch.Tensor]] = {name: [] for name in self._head_names}

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            tokens = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                heads = self.forward(tokens["input_ids"], tokens["attention_mask"])

            for name in self._head_names:
                all_results[name].append(heads[name].cpu())

        return {name: torch.cat(vecs, dim=0) for name, vecs in all_results.items()}

    def encode_single(
        self, text: str, device: str | torch.device = "cpu"
    ) -> dict[str, torch.Tensor]:
        """encode a single text, returns dict of 1d tensors."""
        result = self.encode([text], device=device)
        return {name: vec[0] for name, vec in result.items()}

    def concat_representations(self, heads: dict[str, torch.Tensor]) -> torch.Tensor:
        """concatenate all head outputs into a single vector."""
        parts = [heads[name] for name in self._head_names]
        return torch.cat(parts, dim=-1)

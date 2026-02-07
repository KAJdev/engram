"""synthesis encoder: produces a vector for what two complementary memories imply together."""

from __future__ import annotations

import torch
import torch.nn as nn

from engram.config import EncoderConfig, SynthesisConfig


class SynthesisEncoder(nn.Module):
    def __init__(self, encoder_config: EncoderConfig, config: SynthesisConfig):
        super().__init__()
        self.config = config
        input_dim = encoder_config.total_dim

        # project each memory's full representation into the hidden dim
        self.proj_a = nn.Linear(input_dim, config.hidden_dim)
        self.proj_b = nn.Linear(input_dim, config.hidden_dim)

        # cross attention layers
        self.layers = nn.ModuleList()
        for _ in range(config.num_layers):
            self.layers.append(
                SynthesisCrossAttentionLayer(
                    hidden_dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    dropout=config.dropout,
                )
            )

        # projection to semantic space
        self.output_proj = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.output_dim),
            nn.GELU(),
            nn.Linear(config.output_dim, config.output_dim),
            nn.LayerNorm(config.output_dim),
        )

    def forward(self, repr_a: torch.Tensor, repr_b: torch.Tensor) -> torch.Tensor:
        """repr_a, repr_b: [batch, total_dim] -> synthesis_vec: [batch, output_dim]."""
        h_a = self.proj_a(repr_a).unsqueeze(1)
        h_b = self.proj_b(repr_b).unsqueeze(1)

        for layer in self.layers:
            h_a, h_b = layer(h_a, h_b)

        h_a = h_a.squeeze(1)
        h_b = h_b.squeeze(1)
        combined = torch.cat([h_a, h_b], dim=-1)

        return self.output_proj(combined)


class SynthesisCrossAttentionLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.cross_attn_a = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_a1 = nn.LayerNorm(hidden_dim)
        self.ffn_a = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm_a2 = nn.LayerNorm(hidden_dim)

        self.cross_attn_b = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_b1 = nn.LayerNorm(hidden_dim)
        self.ffn_b = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm_b2 = nn.LayerNorm(hidden_dim)

    def forward(
        self, h_a: torch.Tensor, h_b: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn_out_a, _ = self.cross_attn_a(h_a, h_b, h_b)
        h_a = self.norm_a1(h_a + attn_out_a)
        h_a = self.norm_a2(h_a + self.ffn_a(h_a))

        attn_out_b, _ = self.cross_attn_b(h_b, h_a, h_a)
        h_b = self.norm_b1(h_b + attn_out_b)
        h_b = self.norm_b2(h_b + self.ffn_b(h_b))

        return h_a, h_b

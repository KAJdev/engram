"""edge classifier: predicts relationships between memory pairs."""

from __future__ import annotations

import torch
import torch.nn as nn

from engram.config import EdgeClassifierConfig, EncoderConfig

EDGE_TYPES = [
    "complementary",
    "causal",
    "temporal_sequence",
    "contradictory",
    "elaborative",
    "entity_overlap",
]


class EdgeClassifier(nn.Module):
    def __init__(self, encoder_config: EncoderConfig, config: EdgeClassifierConfig):
        super().__init__()
        self.config = config
        input_dim = encoder_config.total_dim * 2

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in config.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
            ])
            prev_dim = hidden_dim

        self.shared = nn.Sequential(*layers)
        self.edge_exists_head = nn.Linear(prev_dim, 1)
        self.edge_type_head = nn.Linear(prev_dim, config.num_edge_types)
        self.edge_weight_head = nn.Linear(prev_dim, 1)

    def forward(
        self, repr_a: torch.Tensor, repr_b: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """repr_a, repr_b: [batch, total_dim] concatenated multi head vecs."""
        combined = torch.cat([repr_a, repr_b], dim=-1)
        features = self.shared(combined)

        return {
            "edge_exists": torch.sigmoid(self.edge_exists_head(features)),
            "edge_type": self.edge_type_head(features),
            "edge_weight": torch.sigmoid(self.edge_weight_head(features)),
        }

    def predict(
        self, repr_a: torch.Tensor, repr_b: torch.Tensor
    ) -> dict[str, torch.Tensor | str | float]:
        """single pair prediction with decoded edge type."""
        self.eval()
        with torch.no_grad():
            out = self.forward(repr_a.unsqueeze(0), repr_b.unsqueeze(0))

        exists_prob = out["edge_exists"].item()
        type_idx = out["edge_type"].argmax(dim=-1).item()
        weight = out["edge_weight"].item()

        return {
            "edge_exists": exists_prob,
            "edge_type": EDGE_TYPES[type_idx],
            "edge_type_idx": type_idx,
            "edge_weight": weight,
        }

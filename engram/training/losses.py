"""loss functions for all training phases."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """contrastive loss for encoder training, supports in batch negatives."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor | None = None
    ) -> torch.Tensor:
        """if negative is none, uses in batch negatives. shapes: [batch, dim]."""
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)

        if negative is None:
            # in batch negatives: each positive is negative for every other anchor
            logits = torch.matmul(anchor, positive.T) / self.temperature  # [B, B]
            labels = torch.arange(len(anchor), device=anchor.device)
            return F.cross_entropy(logits, labels)
        else:
            negative = F.normalize(negative, dim=-1)
            pos_sim = (anchor * positive).sum(dim=-1) / self.temperature  # [B]
            neg_sim = (anchor * negative).sum(dim=-1) / self.temperature  # [B]
            logits = torch.stack([pos_sim, neg_sim], dim=-1)  # [B, 2]
            labels = torch.zeros(len(anchor), dtype=torch.long, device=anchor.device)
            return F.cross_entropy(logits, labels)


class MultiTaskEncoderLoss(nn.Module):
    """combines losses from all encoder heads with learned uncertainty weights.
    based on Kendall et al. 2018."""

    def __init__(self, head_names: list[str], temperature: float = 0.07):
        super().__init__()
        self.head_losses = nn.ModuleDict({
            name: InfoNCELoss(temperature) for name in head_names
            if name != "temporal"
        })
        # temporal head uses mse for time regression
        self.temporal_loss = nn.MSELoss()

        # learnable log variance per task
        self.log_vars = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(1)) for name in head_names
        })

    def forward(
        self,
        head_outputs_anchor: dict[str, torch.Tensor],
        head_outputs_positive: dict[str, torch.Tensor],
        head_outputs_negative: dict[str, torch.Tensor] | None = None,
        temporal_targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        total_loss = torch.tensor(0.0, device=next(iter(head_outputs_anchor.values())).device)
        loss_dict = {}

        for name in head_outputs_anchor:
            if name == "temporal":
                if temporal_targets is not None:
                    loss = self.temporal_loss(head_outputs_anchor[name], temporal_targets)
                else:
                    continue
            else:
                neg = head_outputs_negative[name] if head_outputs_negative else None
                loss = self.head_losses[name](
                    head_outputs_anchor[name],
                    head_outputs_positive[name],
                    neg,
                )

            # uncertainty weighting: L_total = sum(exp(-log_var) * L + log_var)
            precision = torch.exp(-self.log_vars[name])
            weighted = precision * loss + self.log_vars[name]
            total_loss = total_loss + weighted.squeeze()
            loss_dict[name] = loss.item()

        loss_dict["total"] = total_loss.item()
        return total_loss, loss_dict


class EdgeClassifierLoss(nn.Module):
    """combined loss for edge existence, type, and weight prediction."""

    def __init__(self, type_weight: float = 1.0, weight_weight: float = 0.5):
        super().__init__()
        self.exists_loss = nn.BCELoss()
        self.type_loss = nn.CrossEntropyLoss()
        self.weight_loss = nn.MSELoss()
        self.type_weight = type_weight
        self.weight_weight = weight_weight

    def forward(
        self,
        pred_exists: torch.Tensor,
        pred_type: torch.Tensor,
        pred_weight: torch.Tensor,
        target_exists: torch.Tensor,
        target_type: torch.Tensor,
        target_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        l_exists = self.exists_loss(pred_exists.squeeze(), target_exists)

        # type and weight loss only for positive edges
        pos_mask = target_exists > 0.5
        if pos_mask.any():
            l_type = self.type_loss(pred_type[pos_mask], target_type[pos_mask])
            l_weight = self.weight_loss(pred_weight[pos_mask].squeeze(), target_weight[pos_mask])
        else:
            l_type = torch.tensor(0.0, device=pred_exists.device)
            l_weight = torch.tensor(0.0, device=pred_exists.device)

        total = l_exists + self.type_weight * l_type + self.weight_weight * l_weight

        return total, {
            "exists": l_exists.item(),
            "type": l_type.item(),
            "weight": l_weight.item(),
            "total": total.item(),
        }


class SynthesisContrastiveLoss(nn.Module):
    """contrastive loss for synthesis encoder against relevant queries."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self, synthesis_vec: torch.Tensor, query_vec: torch.Tensor
    ) -> torch.Tensor:
        """synthesis_vec and query_vec: [batch, dim]. uses in batch negatives."""
        synth = F.normalize(synthesis_vec, dim=-1)
        query = F.normalize(query_vec, dim=-1)
        logits = torch.matmul(synth, query.T) / self.temperature
        labels = torch.arange(len(synth), device=synth.device)
        return F.cross_entropy(logits, labels)

"""config dataclasses for all engram components."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EncoderConfig:
    backbone: str = "intfloat/e5-large-v2"
    backbone_dim: int = 1024
    semantic_dim: int = 768
    entity_dim: int = 256
    theme_dim: int = 256
    temporal_dim: int = 64
    learned_dim: int = 256
    num_learned_heads: int = 3
    freeze_backbone_layers: int = 0
    pooling: str = "mean"  # mean, cls

    @property
    def total_dim(self) -> int:
        return (
            self.semantic_dim
            + self.entity_dim
            + self.theme_dim
            + self.temporal_dim
            + self.learned_dim * self.num_learned_heads
        )

    @property
    def head_dims(self) -> dict[str, int]:
        d = {
            "semantic": self.semantic_dim,
            "entity": self.entity_dim,
            "theme": self.theme_dim,
            "temporal": self.temporal_dim,
        }
        for i in range(self.num_learned_heads):
            d[f"learned_{i}"] = self.learned_dim
        return d


@dataclass
class EdgeClassifierConfig:
    hidden_dims: list[int] = field(default_factory=lambda: [2048, 1024, 512])
    dropout: float = 0.1
    num_edge_types: int = 6
    edge_threshold: float = 0.5
    synthesis_threshold: float = 0.7


@dataclass
class SynthesisConfig:
    num_layers: int = 3
    num_heads: int = 8
    hidden_dim: int = 512
    output_dim: int = 768  # must match semantic_dim
    dropout: float = 0.1


@dataclass
class GraphConfig:
    ann_backend: str = "faiss"
    ann_metric: str = "cosine"
    ann_nprobe: int = 10
    candidate_k: int = 50
    ppr_damping: float = 0.5
    ppr_max_iter: int = 100
    ppr_tol: float = 1e-6
    ppr_top_n: int = 20


@dataclass
class TrainingConfig:
    # general
    output_dir: Path = Path("./outputs")
    seed: int = 42
    device: str = "cuda"
    num_workers: int = 4

    # encoder training
    encoder_epochs: int = 20
    encoder_lr: float = 2e-5
    encoder_backbone_lr: float = 2e-6
    encoder_batch_size: int = 64
    encoder_warmup_steps: int = 500
    contrastive_temperature: float = 0.07

    # edge classifier training
    edge_epochs: int = 30
    edge_lr: float = 1e-3
    edge_batch_size: int = 256

    # synthesis training
    synthesis_epochs: int = 20
    synthesis_lr: float = 1e-4
    synthesis_batch_size: int = 64
    synthesis_temperature: float = 0.07


@dataclass
class DataGenConfig:
    llm_provider: str = "anthropic"  # anthropic, openai, vllm
    llm_model: str = "claude-sonnet-4-5-20250929"
    num_synthetic_users: int = 100
    memories_per_user: int = 200
    pairs_per_user: int = 500
    queries_per_user: int = 30
    output_dir: Path = Path("./data")
    max_concurrent: int = 10
    vllm_url: str = (
        "http://localhost:8000"  # runpod: https://api.runpod.ai/v2/{endpoint_id}
    )
    vllm_api_key: str = "not-needed"  # set to RUNPOD_API_KEY for runpod endpoints


@dataclass
class ConsolidationConfig:
    leiden_resolution: float = 1.0
    edge_decay_rate: float = 0.01
    min_edge_weight: float = 0.1
    synthesis_prune_threshold: float = 0.2
    bridge_detection_threshold: float = 0.3


@dataclass
class EngramConfig:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    edge_classifier: EdgeClassifierConfig = field(default_factory=EdgeClassifierConfig)
    synthesis: SynthesisConfig = field(default_factory=SynthesisConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    datagen: DataGenConfig = field(default_factory=DataGenConfig)
    consolidation: ConsolidationConfig = field(default_factory=ConsolidationConfig)


# small config for fast dev/testing
def dev_config() -> EngramConfig:
    return EngramConfig(
        encoder=EncoderConfig(
            backbone="intfloat/e5-small-v2",
            backbone_dim=384,
            semantic_dim=256,
            entity_dim=128,
            theme_dim=128,
            temporal_dim=32,
            learned_dim=128,
            num_learned_heads=2,
        ),
        edge_classifier=EdgeClassifierConfig(
            hidden_dims=[512, 256],
        ),
        synthesis=SynthesisConfig(
            num_layers=2,
            num_heads=4,
            hidden_dim=256,
            output_dim=256,
        ),
        training=TrainingConfig(
            device="cpu",
            encoder_epochs=3,
            encoder_batch_size=16,
            edge_epochs=5,
            edge_batch_size=32,
            synthesis_epochs=3,
            synthesis_batch_size=16,
        ),
    )

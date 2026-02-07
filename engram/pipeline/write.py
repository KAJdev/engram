"""write pipeline: encode, index, find candidates, score edges, synthesize, assign community."""

from __future__ import annotations

import time

import numpy as np
import torch

from engram.config import EngramConfig
from engram.models.encoder import MultiHeadEncoder
from engram.models.edge_classifier import EdgeClassifier, EDGE_TYPES
from engram.models.synthesis import SynthesisEncoder
from engram.graph.memory_graph import MemoryGraph, MemoryNode, SynthesisNode, Edge
from engram.graph.ann_index import MultiHeadIndex


class WritePipeline:
    def __init__(
        self,
        config: EngramConfig,
        encoder: MultiHeadEncoder,
        edge_classifier: EdgeClassifier,
        synthesis_encoder: SynthesisEncoder,
        graph: MemoryGraph,
        index: MultiHeadIndex,
        device: str | torch.device = "cpu",
    ):
        self.config = config
        self.encoder = encoder.to(device)
        self.edge_classifier = edge_classifier.to(device)
        self.synthesis_encoder = synthesis_encoder.to(device)
        self.graph = graph
        self.index = index
        self.device = device

        self.encoder.eval()
        self.edge_classifier.eval()
        self.synthesis_encoder.eval()

    @torch.no_grad()
    def insert(self, text: str, metadata: dict | None = None) -> str:
        """insert a new memory, returns the memory id."""
        metadata = metadata or {}

        # encode
        heads = self.encoder.encode_single(text, device=self.device)
        heads_np = {k: v.cpu().numpy() for k, v in heads.items()}

        # create node
        memory_id = self.graph.new_memory_id()
        node = MemoryNode(
            id=memory_id,
            text=text,
            heads=heads_np,
            metadata=metadata,
            timestamp=time.time(),
        )
        self.graph.add_memory(node)

        # index
        self.index.add(memory_id, heads_np)

        # find candidates
        candidates = self.index.search(heads_np, k=self.config.graph.candidate_k)
        # remove self from results
        candidates = [(mid, score) for mid, score in candidates if mid != memory_id]

        # score edges
        if candidates:
            full_repr = self.encoder.concat_representations(heads).unsqueeze(0)  # [1, total_dim]
            full_repr_device = full_repr.to(self.device)

            for cand_id, _ in candidates:
                if cand_id.startswith("s_"):
                    continue

                cand_node = self.graph.memories.get(cand_id)
                if cand_node is None:
                    continue

                cand_heads = {k: torch.tensor(v) for k, v in cand_node.heads.items()}
                cand_repr = self.encoder.concat_representations(cand_heads).unsqueeze(0).to(self.device)

                pred = self.edge_classifier.predict(full_repr_device.squeeze(0), cand_repr.squeeze(0))

                if pred["edge_exists"] > self.config.edge_classifier.edge_threshold:
                    edge = Edge(
                        source=memory_id,
                        target=cand_id,
                        edge_type=pred["edge_type"],
                        weight=pred["edge_weight"],
                        confidence=pred["edge_exists"],
                    )
                    self.graph.add_edge(edge)

                    # synthesize strong complementary edges
                    if (
                        pred["edge_type"] == "complementary"
                        and pred["edge_weight"] > self.config.edge_classifier.synthesis_threshold
                    ):
                        synth_vec = self.synthesis_encoder(
                            full_repr_device, cand_repr
                        ).squeeze(0).cpu().numpy()

                        synth_id = self.graph.new_synthesis_id()
                        synth_node = SynthesisNode(
                            id=synth_id,
                            parent_a=memory_id,
                            parent_b=cand_id,
                            vector=synth_vec,
                        )
                        self.graph.add_synthesis(synth_node)
                        self.index.add_to_semantic(synth_id, synth_vec)

        # community assignment
        self.graph.assign_community_by_neighbors(memory_id)

        return memory_id

    def insert_batch(self, texts: list[str], metadata: list[dict] | None = None) -> list[str]:
        """insert multiple memories."""
        metadata = metadata or [{}] * len(texts)
        return [self.insert(text, meta) for text, meta in zip(texts, metadata)]

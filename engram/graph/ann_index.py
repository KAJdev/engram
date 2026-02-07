"""faiss ann indices, one per encoder head."""

from __future__ import annotations

import numpy as np
import faiss

from engram.config import EncoderConfig, GraphConfig


class ANNIndex:
    """single faiss index for one vector space."""

    def __init__(self, dim: int, metric: str = "cosine"):
        self.dim = dim
        self.metric = metric
        if metric == "cosine":
            self.index = faiss.IndexFlatIP(dim)
        else:
            self.index = faiss.IndexFlatL2(dim)
        self.id_map: list[str] = []

    def add(self, memory_id: str, vector: np.ndarray) -> None:
        vec = vector.reshape(1, -1).astype(np.float32)
        if self.metric == "cosine":
            faiss.normalize_L2(vec)
        self.index.add(vec)
        self.id_map.append(memory_id)

    def add_batch(self, memory_ids: list[str], vectors: np.ndarray) -> None:
        vecs = vectors.astype(np.float32)
        if self.metric == "cosine":
            faiss.normalize_L2(vecs)
        self.index.add(vecs)
        self.id_map.extend(memory_ids)

    def search(self, query_vector: np.ndarray, k: int = 50) -> list[tuple[str, float]]:
        if self.index.ntotal == 0:
            return []
        k = min(k, self.index.ntotal)
        vec = query_vector.reshape(1, -1).astype(np.float32)
        if self.metric == "cosine":
            faiss.normalize_L2(vec)
        scores, indices = self.index.search(vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append((self.id_map[idx], float(score)))
        return results

    def remove(self, memory_id: str) -> None:
        """rebuilds entire index, expensive."""
        if memory_id not in self.id_map:
            return
        idx = self.id_map.index(memory_id)
        n = self.index.ntotal
        if n == 0:
            return
        all_vecs = np.zeros((n, self.dim), dtype=np.float32)
        for i in range(n):
            all_vecs[i] = self.index.reconstruct(i)
        keep_mask = list(range(n))
        keep_mask.pop(idx)
        self.id_map.pop(idx)
        if self.metric == "cosine":
            new_index = faiss.IndexFlatIP(self.dim)
        else:
            new_index = faiss.IndexFlatL2(self.dim)
        if len(keep_mask) > 0:
            new_index.add(all_vecs[keep_mask])
        self.index = new_index

    @property
    def size(self) -> int:
        return self.index.ntotal


class MultiHeadIndex:
    """one ann index per encoder head."""

    def __init__(self, encoder_config: EncoderConfig, graph_config: GraphConfig):
        self.config = graph_config
        self.indices: dict[str, ANNIndex] = {}
        for head_name, dim in encoder_config.head_dims.items():
            self.indices[head_name] = ANNIndex(dim, metric=graph_config.ann_metric)

    def add(self, memory_id: str, heads: dict[str, np.ndarray]) -> None:
        for head_name, vec in heads.items():
            if head_name in self.indices:
                self.indices[head_name].add(memory_id, vec)

    def search(
        self, query_heads: dict[str, np.ndarray], k: int | None = None
    ) -> list[tuple[str, float]]:
        """search all indices, union and deduplicate by max score."""
        k = k or self.config.candidate_k
        score_map: dict[str, float] = {}
        for head_name, query_vec in query_heads.items():
            if head_name not in self.indices:
                continue
            results = self.indices[head_name].search(query_vec, k=k)
            for mem_id, score in results:
                if mem_id not in score_map or score > score_map[mem_id]:
                    score_map[mem_id] = score
        ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
        return ranked

    def add_to_semantic(self, node_id: str, vector: np.ndarray) -> None:
        """add a synthesis node to the semantic index only."""
        self.indices["semantic"].add(node_id, vector)

    def remove(self, memory_id: str) -> None:
        for index in self.indices.values():
            index.remove(memory_id)

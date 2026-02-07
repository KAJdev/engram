"""query pipeline: encode, ann lookup, ppr traversal, rank, return subgraph."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import numpy as np

from engram.config import EngramConfig
from engram.models.encoder import MultiHeadEncoder
from engram.graph.memory_graph import MemoryGraph
from engram.graph.ann_index import MultiHeadIndex


@dataclass
class QueryResult:
    query: str
    memories: list[dict] = field(default_factory=list)
    subgraph: dict = field(default_factory=dict)


class QueryPipeline:
    def __init__(
        self,
        config: EngramConfig,
        encoder: MultiHeadEncoder,
        graph: MemoryGraph,
        index: MultiHeadIndex,
        device: str | torch.device = "cpu",
    ):
        self.config = config
        self.encoder = encoder.to(device)
        self.graph = graph
        self.index = index
        self.device = device
        self.encoder.eval()

    @torch.no_grad()
    def query(self, text: str, top_n: int | None = None) -> QueryResult:
        top_n = top_n or self.config.graph.ppr_top_n

        heads = self.encoder.encode_single(text, device=self.device)
        heads_np = {k: v.cpu().numpy() for k, v in heads.items()}

        candidates = self.index.search(heads_np, k=self.config.graph.candidate_k)

        if not candidates:
            return QueryResult(query=text)

        source_nodes = {}
        for node_id, score in candidates:
            if node_id in self.graph.graph:
                source_nodes[node_id] = max(score, 0.01)

        if not source_nodes:
            return QueryResult(query=text)

        ppr_scores = self.graph.personalized_pagerank(source_nodes, top_n=top_n * 2)

        # multi source convergence bonus
        ann_hit_counts: dict[str, int] = {}
        for head_name, ann_idx in self.index.indices.items():
            head_results = ann_idx.search(heads_np.get(head_name, np.zeros(ann_idx.dim)), k=20)
            for nid, _ in head_results:
                ann_hit_counts[nid] = ann_hit_counts.get(nid, 0) + 1

        for nid in ppr_scores:
            hits = ann_hit_counts.get(nid, 0)
            if hits > 1:
                ppr_scores[nid] *= 1.0 + 0.2 * (hits - 1)

        # resolve synthesis nodes to their parent memories
        memory_scores: dict[str, float] = {}
        for nid, score in ppr_scores.items():
            if nid in self.graph.memories:
                memory_scores[nid] = max(memory_scores.get(nid, 0), score)
            elif nid in self.graph.synthesis_nodes:
                sn = self.graph.synthesis_nodes[nid]
                for parent_id in [sn.parent_a, sn.parent_b]:
                    if parent_id in self.graph.memories:
                        memory_scores[parent_id] = max(
                            memory_scores.get(parent_id, 0), score * 0.8
                        )
                sn.retrieval_count += 1

        # rank and build result
        ranked = sorted(memory_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

        memories = []
        node_ids = []
        for mid, score in ranked:
            mem = self.graph.memories[mid]
            memories.append({
                "id": mid,
                "text": mem.text,
                "score": score,
                "community": mem.community,
                "metadata": mem.metadata,
            })
            node_ids.append(mid)

        # include synthesis nodes in subgraph
        for nid in ppr_scores:
            if nid in self.graph.synthesis_nodes:
                sn = self.graph.synthesis_nodes[nid]
                if sn.parent_a in node_ids or sn.parent_b in node_ids:
                    node_ids.append(nid)

        subgraph = self.graph.get_subgraph(node_ids)

        return QueryResult(query=text, memories=memories, subgraph=subgraph)

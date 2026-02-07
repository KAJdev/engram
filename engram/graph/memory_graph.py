"""memory graph with personalized pagerank and community detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np

from engram.config import GraphConfig, ConsolidationConfig


@dataclass
class MemoryNode:
    id: str
    text: str
    heads: dict[str, np.ndarray]
    metadata: dict[str, Any] = field(default_factory=dict)
    community: int = -1
    timestamp: float = 0.0


@dataclass
class SynthesisNode:
    id: str
    parent_a: str
    parent_b: str
    vector: np.ndarray
    retrieval_count: int = 0


@dataclass
class Edge:
    source: str
    target: str
    edge_type: str
    weight: float
    confidence: float = 1.0


class MemoryGraph:
    def __init__(self, graph_config: GraphConfig, consolidation_config: ConsolidationConfig | None = None):
        self.config = graph_config
        self.consolidation_config = consolidation_config or ConsolidationConfig()
        self.graph = nx.DiGraph()
        self.memories: dict[str, MemoryNode] = {}
        self.synthesis_nodes: dict[str, SynthesisNode] = {}
        self._next_memory_id = 0
        self._next_synthesis_id = 0

    def new_memory_id(self) -> str:
        mid = f"m_{self._next_memory_id}"
        self._next_memory_id += 1
        return mid

    def new_synthesis_id(self) -> str:
        sid = f"s_{self._next_synthesis_id}"
        self._next_synthesis_id += 1
        return sid

    def add_memory(self, node: MemoryNode) -> None:
        self.memories[node.id] = node
        self.graph.add_node(node.id, type="memory")

    def add_synthesis(self, node: SynthesisNode) -> None:
        self.synthesis_nodes[node.id] = node
        self.graph.add_node(node.id, type="synthesis")
        self.graph.add_edge(node.parent_a, node.id, edge_type="synthesis", weight=1.0)
        self.graph.add_edge(node.parent_b, node.id, edge_type="synthesis", weight=1.0)

    def add_edge(self, edge: Edge) -> None:
        self.graph.add_edge(
            edge.source,
            edge.target,
            edge_type=edge.edge_type,
            weight=edge.weight,
            confidence=edge.confidence,
        )
        # symmetric relationship types get a reverse edge
        if edge.edge_type in ("complementary", "contradictory", "entity_overlap", "elaborative"):
            self.graph.add_edge(
                edge.target,
                edge.source,
                edge_type=edge.edge_type,
                weight=edge.weight,
                confidence=edge.confidence,
            )

    def personalized_pagerank(
        self,
        source_nodes: dict[str, float],
        top_n: int | None = None,
    ) -> dict[str, float]:
        """personalized pagerank from weighted source nodes."""
        if not source_nodes or len(self.graph) == 0:
            return {}

        top_n = top_n or self.config.ppr_top_n

        total_weight = sum(source_nodes.values())
        personalization = {}
        for node in self.graph.nodes():
            if node in source_nodes:
                personalization[node] = source_nodes[node] / total_weight
            else:
                personalization[node] = 0.0

        try:
            ppr = nx.pagerank(
                self.graph,
                alpha=1.0 - self.config.ppr_damping,
                personalization=personalization,
                max_iter=self.config.ppr_max_iter,
                tol=self.config.ppr_tol,
                weight="weight",
            )
        except nx.PowerIterationFailedConvergence:
            return dict(sorted(source_nodes.items(), key=lambda x: x[1], reverse=True)[:top_n])

        ranked = sorted(ppr.items(), key=lambda x: x[1], reverse=True)
        return dict(ranked[:top_n])

    def get_neighbors(self, node_id: str) -> list[tuple[str, dict]]:
        """get all neighbors with edge data."""
        if node_id not in self.graph:
            return []
        neighbors = []
        for _, target, data in self.graph.edges(node_id, data=True):
            neighbors.append((target, data))
        return neighbors

    def get_community_members(self, community_id: int) -> list[str]:
        return [
            mid for mid, mem in self.memories.items()
            if mem.community == community_id
        ]

    def assign_community_by_neighbors(self, memory_id: str) -> int:
        """assign to the plurality community of neighbors."""
        neighbors = self.get_neighbors(memory_id)
        if not neighbors:
            community_id = max((m.community for m in self.memories.values()), default=-1) + 1
            self.memories[memory_id].community = community_id
            return community_id

        community_votes: dict[int, float] = {}
        for neighbor_id, edge_data in neighbors:
            if neighbor_id in self.memories:
                c = self.memories[neighbor_id].community
                w = edge_data.get("weight", 1.0)
                community_votes[c] = community_votes.get(c, 0.0) + w

        if community_votes:
            best = max(community_votes, key=community_votes.get)
            self.memories[memory_id].community = best
            return best
        else:
            community_id = max((m.community for m in self.memories.values()), default=-1) + 1
            self.memories[memory_id].community = community_id
            return community_id

    def decay_edges(self, decay_rate: float | None = None) -> int:
        """decay all edge weights, returns number of edges pruned."""
        rate = decay_rate or self.consolidation_config.edge_decay_rate
        min_weight = self.consolidation_config.min_edge_weight
        to_remove = []
        for u, v, data in self.graph.edges(data=True):
            if data.get("edge_type") == "synthesis":
                continue
            new_weight = data["weight"] * (1.0 - rate)
            if new_weight < min_weight:
                to_remove.append((u, v))
            else:
                data["weight"] = new_weight
        for u, v in to_remove:
            self.graph.remove_edge(u, v)
        return len(to_remove)

    def get_subgraph(self, node_ids: list[str]) -> dict:
        """extract subgraph for the given node ids."""
        node_set = set(node_ids)
        nodes = []
        for nid in node_ids:
            if nid in self.memories:
                nodes.append({"id": nid, "type": "memory", "text": self.memories[nid].text})
            elif nid in self.synthesis_nodes:
                sn = self.synthesis_nodes[nid]
                nodes.append({
                    "id": nid,
                    "type": "synthesis",
                    "parents": [sn.parent_a, sn.parent_b],
                })

        edges = []
        for u, v, data in self.graph.edges(data=True):
            if u in node_set and v in node_set:
                edges.append({"source": u, "target": v, **data})

        return {"nodes": nodes, "edges": edges}

    @property
    def num_memories(self) -> int:
        return len(self.memories)

    @property
    def num_edges(self) -> int:
        return self.graph.number_of_edges()

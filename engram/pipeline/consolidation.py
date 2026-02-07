"""consolidation: periodic background job for long range connections."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from engram.config import EngramConfig
from engram.models.encoder import MultiHeadEncoder
from engram.models.edge_classifier import EdgeClassifier
from engram.models.synthesis import SynthesisEncoder
from engram.graph.memory_graph import MemoryGraph, Edge, SynthesisNode
from engram.graph.ann_index import MultiHeadIndex


class ConsolidationEngine:
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
        self.cc = config.consolidation
        self.encoder = encoder.to(device)
        self.edge_classifier = edge_classifier.to(device)
        self.synthesis_encoder = synthesis_encoder.to(device)
        self.graph = graph
        self.index = index
        self.device = device

    def run(self) -> dict:
        """run full consolidation, returns stats."""
        stats = {}

        stats["communities"] = self._redetect_communities()
        stats["new_bridges"] = self._detect_bridges()
        stats["decayed_edges"] = self.graph.decay_edges()
        stats["pruned_synthesis"] = self._prune_synthesis()

        return stats

    def _redetect_communities(self) -> int:
        """leiden community detection."""
        try:
            import igraph as ig
            import leidenalg
        except ImportError:
            # fallback: keep existing communities
            return 0

        if self.graph.num_memories < 3:
            return 0

        # build igraph from networkx
        node_list = list(self.graph.memories.keys())
        node_idx = {nid: i for i, nid in enumerate(node_list)}

        edges = []
        weights = []
        for u, v, data in self.graph.graph.edges(data=True):
            if u in node_idx and v in node_idx:
                edges.append((node_idx[u], node_idx[v]))
                weights.append(data.get("weight", 1.0))

        if not edges:
            return 0

        g = ig.Graph(n=len(node_list), edges=edges, directed=False)
        g.es["weight"] = weights

        partition = leidenalg.find_partition(
            g,
            leidenalg.ModularityVertexPartition,
            weights="weight",
            resolution_parameter=self.cc.leiden_resolution,
        )

        for i, nid in enumerate(node_list):
            self.graph.memories[nid].community = partition.membership[i]

        return len(set(partition.membership))

    @torch.no_grad()
    def _detect_bridges(self) -> int:
        """find unlinked cross community connections."""
        communities: dict[int, list[str]] = defaultdict(list)
        for mid, mem in self.graph.memories.items():
            communities[mem.community].append(mid)

        if len(communities) < 2:
            return 0

        # community centroids in semantic space
        centroids: dict[int, np.ndarray] = {}
        for cid, members in communities.items():
            vecs = [self.graph.memories[m].heads["semantic"] for m in members]
            centroids[cid] = np.mean(vecs, axis=0)

        new_bridges = 0
        community_ids = list(communities.keys())

        for i in range(len(community_ids)):
            for j in range(i + 1, len(community_ids)):
                ci, cj = community_ids[i], community_ids[j]

                # skip if already connected
                has_edge = False
                for m_i in communities[ci]:
                    for m_j in communities[cj]:
                        if self.graph.graph.has_edge(m_i, m_j):
                            has_edge = True
                            break
                    if has_edge:
                        break
                if has_edge:
                    continue

                # centroid similarity filter
                sim = np.dot(centroids[ci], centroids[cj]) / (
                    np.linalg.norm(centroids[ci]) * np.linalg.norm(centroids[cj]) + 1e-9
                )
                if sim < self.cc.bridge_detection_threshold:
                    continue

                # best pair across communities
                best_pair = None
                best_score = 0.0

                for m_i in communities[ci][:10]:  # limit search
                    for m_j in communities[cj][:10]:
                        mem_i = self.graph.memories[m_i]
                        mem_j = self.graph.memories[m_j]
                        heads_i = {k: torch.tensor(v) for k, v in mem_i.heads.items()}
                        heads_j = {k: torch.tensor(v) for k, v in mem_j.heads.items()}
                        repr_i = self.encoder.concat_representations(heads_i).to(self.device)
                        repr_j = self.encoder.concat_representations(heads_j).to(self.device)

                        pred = self.edge_classifier.predict(repr_i, repr_j)
                        if pred["edge_exists"] > best_score:
                            best_score = pred["edge_exists"]
                            best_pair = (m_i, m_j, pred)

                if best_pair and best_score > self.config.edge_classifier.edge_threshold:
                    m_i, m_j, pred = best_pair
                    self.graph.add_edge(Edge(
                        source=m_i, target=m_j,
                        edge_type=pred["edge_type"],
                        weight=pred["edge_weight"],
                        confidence=best_score,
                    ))
                    new_bridges += 1

        return new_bridges

    def _prune_synthesis(self) -> int:
        """prune unretrieved synthesis nodes with weak parents."""
        to_remove = []
        for sid, sn in self.graph.synthesis_nodes.items():
            parent_a_edges = [
                d for _, _, d in self.graph.graph.edges(sn.parent_a, data=True)
                if d.get("edge_type") == "synthesis"
            ]
            parent_b_edges = [
                d for _, _, d in self.graph.graph.edges(sn.parent_b, data=True)
                if d.get("edge_type") == "synthesis"
            ]

            if sn.retrieval_count == 0:
                has_direct = self.graph.graph.has_edge(sn.parent_a, sn.parent_b)
                if has_direct:
                    edge_data = self.graph.graph.get_edge_data(sn.parent_a, sn.parent_b)
                    if edge_data and edge_data.get("weight", 0) < self.cc.synthesis_prune_threshold:
                        to_remove.append(sid)
                else:
                    to_remove.append(sid)

        for sid in to_remove:
            if sid in self.graph.graph:
                self.graph.graph.remove_node(sid)
            self.index.indices["semantic"].remove(sid)
            del self.graph.synthesis_nodes[sid]

        return len(to_remove)

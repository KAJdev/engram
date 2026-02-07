"""pytorch datasets for all three training phases."""

from __future__ import annotations

import json
import random
from pathlib import Path
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


def _load_jsonl(path: Path) -> list[dict]:
    items = []
    with open(path) as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


class EncoderContrastiveDataset(Dataset):
    """anchor/positive/negative triplets for contrastive encoder training.
    each head type defines "positive" differently: semantic uses any edge,
    entity uses shared entities, theme uses shared themes with different entities."""

    def __init__(self, data_dir: Path, head_type: str = "semantic"):
        self.memories = _load_jsonl(data_dir / "memories.jsonl")
        self.edges = _load_jsonl(data_dir / "edges.jsonl")
        self.head_type = head_type

        self._by_user: dict[int, list[dict]] = {}
        for m in self.memories:
            self._by_user.setdefault(m["user_id"], []).append(m)

        self.pairs: list[tuple[str, str]] = []
        self._build_pairs()

    def _build_pairs(self):
        if self.head_type == "semantic":
            # any edge = positive pair
            for edge in self.edges:
                if edge["edge_exists"]:
                    user_mems = self._by_user.get(edge["user_id"], [])
                    a_idx, b_idx = edge["memory_a_idx"], edge["memory_b_idx"]
                    if a_idx < len(user_mems) and b_idx < len(user_mems):
                        self.pairs.append((user_mems[a_idx]["text"], user_mems[b_idx]["text"]))

        elif self.head_type == "entity":
            # memories sharing entities
            for uid, mems in self._by_user.items():
                entity_map: dict[str, list[str]] = {}
                for m in mems:
                    for ent in m.get("entities", []):
                        entity_map.setdefault(ent.lower(), []).append(m["text"])
                for ent, texts in entity_map.items():
                    if len(texts) >= 2:
                        for i in range(len(texts)):
                            for j in range(i + 1, len(texts)):
                                self.pairs.append((texts[i], texts[j]))

        elif self.head_type == "theme":
            # same theme, different entities
            for uid, mems in self._by_user.items():
                theme_map: dict[str, list[dict]] = {}
                for m in mems:
                    for theme in m.get("themes", []):
                        theme_map.setdefault(theme.lower(), []).append(m)
                for theme, group in theme_map.items():
                    if len(group) >= 2:
                        for i in range(len(group)):
                            for j in range(i + 1, len(group)):
                                ents_i = set(e.lower() for e in group[i].get("entities", []))
                                ents_j = set(e.lower() for e in group[j].get("entities", []))
                                if not ents_i & ents_j:  # different entities
                                    self.pairs.append((group[i]["text"], group[j]["text"]))

        # deduplicate
        self.pairs = list(set(self.pairs))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[str, str, str]:
        anchor, positive = self.pairs[idx]
        negative = random.choice(self.memories)["text"]
        while negative == anchor or negative == positive:
            negative = random.choice(self.memories)["text"]
        return anchor, positive, negative


class EdgeClassifierDataset(Dataset):
    """text pairs with edge labels. encoded by frozen encoder at training time."""

    EDGE_TYPES = ["complementary", "causal", "temporal_sequence", "contradictory", "elaborative", "entity_overlap"]

    def __init__(self, data_dir: Path):
        self.memories = _load_jsonl(data_dir / "memories.jsonl")
        self.edges = _load_jsonl(data_dir / "edges.jsonl")

        self._by_user: dict[int, list[dict]] = {}
        for m in self.memories:
            self._by_user.setdefault(m["user_id"], []).append(m)

        self.samples: list[dict] = []
        for edge in self.edges:
            user_mems = self._by_user.get(edge["user_id"], [])
            a_idx, b_idx = edge["memory_a_idx"], edge["memory_b_idx"]
            if a_idx < len(user_mems) and b_idx < len(user_mems):
                edge_type = edge["edge_type"]
                type_idx = self.EDGE_TYPES.index(edge_type) if edge_type in self.EDGE_TYPES else 0
                self.samples.append({
                    "text_a": user_mems[a_idx]["text"],
                    "text_b": user_mems[b_idx]["text"],
                    "edge_exists": 1.0 if edge["edge_exists"] else 0.0,
                    "edge_type_idx": type_idx,
                    "edge_weight": edge["edge_weight"],
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


class SynthesisDataset(Dataset):
    """triplets where the query requires both memories to answer."""

    def __init__(self, data_dir: Path):
        self.memories = _load_jsonl(data_dir / "memories.jsonl")
        synthesis_labels = _load_jsonl(data_dir / "synthesis.jsonl")

        self._by_user: dict[int, list[dict]] = {}
        for m in self.memories:
            self._by_user.setdefault(m["user_id"], []).append(m)

        self.triplets: list[tuple[str, str, str]] = []
        for s in synthesis_labels:
            user_mems = self._by_user.get(s["user_id"], [])
            a_idx, b_idx = s["memory_a_idx"], s["memory_b_idx"]
            if a_idx < len(user_mems) and b_idx < len(user_mems):
                text_a = user_mems[a_idx]["text"]
                text_b = user_mems[b_idx]["text"]
                for query in s["queries"]:
                    self.triplets.append((text_a, text_b, query))

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> tuple[str, str, str]:
        return self.triplets[idx]


class RetrievalEvalDataset(Dataset):
    """evaluation dataset: queries with known relevant memory sets."""

    def __init__(self, data_dir: Path):
        self.memories = _load_jsonl(data_dir / "memories.jsonl")
        self.retrieval = _load_jsonl(data_dir / "retrieval.jsonl")

        self._by_user: dict[int, list[dict]] = {}
        for m in self.memories:
            self._by_user.setdefault(m["user_id"], []).append(m)

        self.samples = []
        for r in self.retrieval:
            user_mems = self._by_user.get(r["user_id"], [])
            self.samples.append({
                "user_id": r["user_id"],
                "query": r["query"],
                "memory_texts": [m["text"] for m in user_mems],
                "directly_relevant": r["directly_relevant"],
                "jointly_relevant": r["jointly_relevant"],
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]

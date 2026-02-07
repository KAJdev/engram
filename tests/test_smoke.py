"""end to end smoke test: generate data, train, write memories, query."""

import shutil
from pathlib import Path

import torch
import pytest

from engram.config import dev_config
from engram.training.datagen import generate_demo_dataset
from engram.training.train_encoder import train_encoder
from engram.training.train_edge_classifier import train_edge_classifier
from engram.training.train_synthesis import train_synthesis
from engram.models.encoder import MultiHeadEncoder
from engram.models.edge_classifier import EdgeClassifier
from engram.models.synthesis import SynthesisEncoder
from engram.graph.memory_graph import MemoryGraph
from engram.graph.ann_index import MultiHeadIndex
from engram.pipeline.write import WritePipeline
from engram.pipeline.query import QueryPipeline


DATA_DIR = Path("./test_data")
OUTPUT_DIR = Path("./test_outputs")


@pytest.fixture(scope="module", autouse=True)
def setup_data():
    """generate demo data once for all tests."""
    generate_demo_dataset(DATA_DIR, seed=42)
    yield
    shutil.rmtree(DATA_DIR, ignore_errors=True)
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


def _make_config():
    config = dev_config()
    config.training.output_dir = OUTPUT_DIR
    config.training.device = "cpu"
    config.training.encoder_epochs = 1
    config.training.encoder_batch_size = 4
    config.training.edge_epochs = 1
    config.training.edge_batch_size = 4
    config.training.synthesis_epochs = 1
    config.training.synthesis_batch_size = 4
    return config


def test_encoder_training():
    config = _make_config()
    encoder = train_encoder(config, DATA_DIR)
    assert isinstance(encoder, MultiHeadEncoder)
    assert (OUTPUT_DIR / "encoder" / "best.pt").exists()

    # verify multi head output
    heads = encoder.encode_single("test memory about health", device="cpu")
    assert "semantic" in heads
    assert "entity" in heads
    assert heads["semantic"].shape[0] == config.encoder.semantic_dim
    assert heads["entity"].shape[0] == config.encoder.entity_dim


def test_edge_classifier_training():
    config = _make_config()
    classifier = train_edge_classifier(config, DATA_DIR)
    assert isinstance(classifier, EdgeClassifier)
    assert (OUTPUT_DIR / "edge_classifier" / "best.pt").exists()


def test_synthesis_training():
    config = _make_config()
    synthesis = train_synthesis(config, DATA_DIR)
    assert isinstance(synthesis, SynthesisEncoder)
    assert (OUTPUT_DIR / "synthesis" / "best.pt").exists()


def test_end_to_end_pipeline():
    """full pipeline: load trained models, write memories, query."""
    config = _make_config()
    device = "cpu"

    encoder = MultiHeadEncoder(config.encoder)
    enc_ckpt = torch.load(OUTPUT_DIR / "encoder" / "best.pt", map_location="cpu", weights_only=True)
    encoder.load_state_dict(enc_ckpt["encoder"])

    edge_clf = EdgeClassifier(config.encoder, config.edge_classifier)
    edge_ckpt = torch.load(OUTPUT_DIR / "edge_classifier" / "best.pt", map_location="cpu", weights_only=True)
    edge_clf.load_state_dict(edge_ckpt["classifier"])

    synthesis = SynthesisEncoder(config.encoder, config.synthesis)
    synth_ckpt = torch.load(OUTPUT_DIR / "synthesis" / "best.pt", map_location="cpu", weights_only=True)
    synthesis.load_state_dict(synth_ckpt["synthesis"])

    # lower thresholds for undertrained smoke test model
    config.edge_classifier.edge_threshold = 0.1
    config.edge_classifier.synthesis_threshold = 0.3
    graph = MemoryGraph(config.graph)
    index = MultiHeadIndex(config.encoder, config.graph)

    writer = WritePipeline(config, encoder, edge_clf, synthesis, graph, index, device=device)

    # the motivating example
    memories = [
        "My doctor put me on warfarin for my blood clot",
        "I've been taking St. John's Wort for my mood",
        "I love going for morning runs in the park",
        "My sister's wedding is next month",
        "I switched to a vegetarian diet three weeks ago",
    ]
    ids = writer.insert_batch(memories)
    assert len(ids) == 5
    assert graph.num_memories == 5
    assert graph.num_edges > 0

    querier = QueryPipeline(config, encoder, graph, index, device=device)

    result = querier.query("I have a terrible headache, should I just take ibuprofen?")
    assert len(result.memories) > 0
    assert result.memories[0]["text"] in memories

    retrieved_texts = [m["text"] for m in result.memories]
    print(f"\nQuery: {result.query}")
    print(f"Retrieved {len(result.memories)} memories:")
    for m in result.memories:
        print(f"  [{m['score']:.3f}] {m['text']}")
    print(f"Graph: {graph.num_memories} memories, {graph.num_edges} edges, "
          f"{len(graph.synthesis_nodes)} synthesis nodes")

"""train all engram models: encoder, edge classifier, synthesis."""

import argparse
from pathlib import Path

import torch

from engram.config import EngramConfig, dev_config
from engram.training.train_encoder import train_encoder
from engram.training.train_edge_classifier import train_edge_classifier
from engram.training.train_synthesis import train_synthesis


def main():
    parser = argparse.ArgumentParser(description="Train all Engram models")
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs"))
    parser.add_argument("--dev", action="store_true", help="Use small dev config for fast iteration")
    parser.add_argument("--device", type=str, default=None, help="cuda, cpu, or mps")
    parser.add_argument("--phase", choices=["all", "encoder", "edge", "synthesis"], default="all")
    parser.add_argument("--encoder-epochs", type=int, default=None)
    parser.add_argument("--edge-epochs", type=int, default=None)
    parser.add_argument("--synthesis-epochs", type=int, default=None)
    args = parser.parse_args()

    config = dev_config() if args.dev else EngramConfig()
    config.training.output_dir = args.output_dir

    if args.device:
        config.training.device = args.device
    elif not args.dev and torch.cuda.is_available():
        config.training.device = "cuda"

    if args.encoder_epochs:
        config.training.encoder_epochs = args.encoder_epochs
    if args.edge_epochs:
        config.training.edge_epochs = args.edge_epochs
    if args.synthesis_epochs:
        config.training.synthesis_epochs = args.synthesis_epochs

    print("=" * 60)
    print("ENGRAM TRAINING PIPELINE")
    print("=" * 60)
    print(f"Data dir:   {args.data_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Device:     {config.training.device}")
    print(f"Backbone:   {config.encoder.backbone}")
    print(f"Phase:      {args.phase}")
    print("=" * 60)

    encoder = None

    if args.phase in ("all", "encoder"):
        print("\n[1/3] training multi head encoder...")
        encoder = train_encoder(config, args.data_dir)

    if args.phase in ("all", "edge"):
        print("\n[2/3] training edge classifier...")
        train_edge_classifier(config, args.data_dir, encoder=encoder)

    if args.phase in ("all", "synthesis"):
        print("\n[3/3] training synthesis encoder...")
        train_synthesis(config, args.data_dir, encoder=encoder)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"Models saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

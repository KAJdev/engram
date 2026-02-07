"""generate training data. --demo for small procedural, --llm for production."""

import argparse
import asyncio
from pathlib import Path

from engram.config import DataGenConfig
from engram.training.datagen import generate_demo_dataset, generate_llm_dataset


def main():
    parser = argparse.ArgumentParser(description="Generate Engram training data")
    parser.add_argument("--mode", choices=["demo", "llm"], default="demo",
                       help="demo: small procedural dataset. llm: full LLM-generated dataset.")
    parser.add_argument("--output-dir", type=Path, default=Path("./data"))
    parser.add_argument("--num-users", type=int, default=100)
    parser.add_argument("--provider", choices=["anthropic", "openai", "vllm"], default="anthropic")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--vllm-url", type=str, default="http://localhost:8000/v1",
                       help="vLLM server URL for local model serving")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.mode == "demo":
        print("Generating demo dataset...")
        stats = generate_demo_dataset(args.output_dir, seed=args.seed)
        print(f"Done! Stats: {stats}")
        print(f"Data saved to {args.output_dir}")

    elif args.mode == "llm":
        config = DataGenConfig(
            llm_provider=args.provider,
            num_synthetic_users=args.num_users,
            output_dir=args.output_dir,
        )
        if args.model:
            config.llm_model = args.model

        print(f"Generating LLM dataset with {args.provider}...")
        print(f"  Users: {config.num_synthetic_users}")
        print(f"  Model: {config.llm_model}")
        stats = asyncio.run(generate_llm_dataset(config))
        print(f"Done! Stats: {stats}")


if __name__ == "__main__":
    main()

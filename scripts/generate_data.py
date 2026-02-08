"""generate training data. --demo for small procedural, --llm for production.

for production data generation, first deploy a vllm endpoint:
    python scripts/deploy_vllm.py

then generate data pointing at it:
    python scripts/generate_data.py --mode llm --provider vllm \\
        --vllm-url https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/openai/v1 \\
        --num-users 500
"""

import argparse
import asyncio
import os
from pathlib import Path

# load .env
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from engram.config import DataGenConfig
from engram.training.datagen import generate_demo_dataset, generate_llm_dataset


def main():
    parser = argparse.ArgumentParser(description="Generate Engram training data")
    parser.add_argument(
        "--mode",
        choices=["demo", "llm"],
        default="demo",
        help="demo: small procedural dataset. llm: full LLM-generated dataset.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("./data"))
    parser.add_argument("--num-users", type=int, default=100)
    parser.add_argument(
        "--provider", choices=["anthropic", "openai", "vllm"], default="anthropic"
    )
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument(
        "--vllm-url",
        type=str,
        default="http://localhost:8000/v1",
        help="vllm/runpod endpoint url (openai compatible)",
    )
    parser.add_argument(
        "--vllm-api-key",
        type=str,
        default=None,
        help="api key for the vllm endpoint. for runpod, uses RUNPOD_API_KEY from env",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=10, help="max concurrent llm requests"
    )
    parser.add_argument("--memories-per-user", type=int, default=200)
    parser.add_argument("--pairs-per-user", type=int, default=500)
    parser.add_argument("--queries-per-user", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.mode == "demo":
        print("generating demo dataset...")
        stats = generate_demo_dataset(args.output_dir, seed=args.seed)
        print(f"done! stats: {stats}")
        print(f"data saved to {args.output_dir}")

    elif args.mode == "llm":
        # figure out api key for vllm provider
        vllm_api_key = args.vllm_api_key
        if args.provider == "vllm" and not vllm_api_key:
            # if url looks like a runpod endpoint, use the runpod api key
            if "runpod.ai" in args.vllm_url:
                vllm_api_key = os.environ.get("RUNPOD_API_KEY", "")
                if not vllm_api_key:
                    print(
                        "error: runpod endpoint detected but no RUNPOD_API_KEY in env"
                    )
                    print("set it in .env or pass --vllm-api-key")
                    return
                print(f"  using RUNPOD_API_KEY for authentication")
            else:
                vllm_api_key = "not-needed"

        config = DataGenConfig(
            llm_provider=args.provider,
            num_synthetic_users=args.num_users,
            output_dir=args.output_dir,
            vllm_url=args.vllm_url,
            vllm_api_key=vllm_api_key or "not-needed",
            max_concurrent=args.max_concurrent,
            memories_per_user=args.memories_per_user,
            pairs_per_user=args.pairs_per_user,
            queries_per_user=args.queries_per_user,
        )
        if args.model:
            config.llm_model = args.model

        print(f"generating llm dataset with {args.provider}...")
        print(f"  users: {config.num_synthetic_users}")
        print(f"  model: {config.llm_model}")
        print(f"  memories/user: {config.memories_per_user}")
        print(f"  pairs/user: {config.pairs_per_user}")
        print(f"  queries/user: {config.queries_per_user}")
        print(f"  max concurrent: {config.max_concurrent}")
        if args.provider == "vllm":
            print(f"  vllm url: {config.vllm_url}")
        print()
        stats = asyncio.run(generate_llm_dataset(config))
        print(f"done! stats: {stats}")


if __name__ == "__main__":
    main()

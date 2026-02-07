"""train engram on runpod flash serverless gpus.

prerequisites:
    1. uv add runpod-flash
    2. RUNPOD_API_KEY env var set
    3. repo pushed to github so remote workers can clone it

usage:
    python scripts/runpod_train.py --phase datagen --datagen-mode demo
    python scripts/runpod_train.py --phase all
    python scripts/runpod_train.py --phase encoder
    python scripts/runpod_train.py --phase all --dev
    python scripts/runpod_train.py --phase datagen --datagen-mode llm --num-users 500
"""

import argparse
import asyncio
from pathlib import Path

from runpod_flash import remote, LiveServerless, GpuGroup, PodTemplate

# sdk doesnt have blackwell yet, monkey patch it in
GpuGroup._value2member_map_["BLACKWELL_180"] = None
BLACKWELL_180 = object.__new__(GpuGroup)
BLACKWELL_180._name_ = "BLACKWELL_180"
BLACKWELL_180._value_ = "BLACKWELL_180"
GpuGroup._value2member_map_["BLACKWELL_180"] = BLACKWELL_180
GpuGroup._member_map_["BLACKWELL_180"] = BLACKWELL_180

REPO_URL = "https://github.com/kajdev/engram.git"
REPO_BRANCH = "main"

# gpu config for training
train_gpu = LiveServerless(
    name="engram-train",
    gpus=[BLACKWELL_180],
    workersMax=1,
    template=PodTemplate(
        containerDiskInGb=100,
        volumeInGb=50,
    ),
)

# gpu for datagen with open source llm via vllm
datagen_gpu = LiveServerless(
    name="engram-datagen",
    gpus=[BLACKWELL_180],
    workersMax=1,
    template=PodTemplate(
        containerDiskInGb=100,
        volumeInGb=50,
    ),
)


def _install_engram(repo_url: str = REPO_URL, branch: str = REPO_BRANCH):
    """clone and install engram on the remote worker."""
    import subprocess, os

    workspace = "/workspace"
    engram_dir = f"{workspace}/engram"
    if not os.path.exists(engram_dir):
        subprocess.run(["git", "clone", "-b", branch, repo_url, engram_dir], check=True)
    subprocess.run(["pip", "install", "-e", engram_dir], check=True, cwd=engram_dir)


# data generation


@remote(
    resource_config=datagen_gpu,
    dependencies=["torch", "vllm", "openai"],
)
def generate_data_remote(
    mode: str = "demo",
    num_users: int = 100,
    model: str = "openai/gpt-oss-120b",
    repo_url: str = "https://github.com/kajdev/engram.git",
):
    """generate training data on a gpu worker."""
    import subprocess, os, sys
    engram_dir = "/workspace/engram"
    if not os.path.exists(engram_dir):
        subprocess.run(["git", "clone", "-b", "main", repo_url, engram_dir], check=True)
    subprocess.run(["pip", "install", "-e", engram_dir], check=True, cwd=engram_dir)
    if engram_dir not in sys.path:
        sys.path.insert(0, engram_dir)

    from pathlib import Path

    data_dir = Path("/workspace/data")

    if mode == "demo":
        from engram.training.datagen import generate_demo_dataset

        stats = generate_demo_dataset(data_dir, seed=42)
        return {"status": "ok", "mode": "demo", "stats": stats}

    # llm mode: start vllm server, generate with open source model
    import subprocess, time

    print(f"Starting vLLM server with {model}...")
    proc = subprocess.Popen(
        [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model,
            "--max-model-len",
            "8192",
            "--gpu-memory-utilization",
            "0.9",
            "--port",
            "8000",
        ]
    )

    # wait for server
    import urllib.request

    for attempt in range(120):  # up to 2 min
        time.sleep(1)
        try:
            urllib.request.urlopen("http://localhost:8000/health")
            print(f"vLLM ready after {attempt+1}s")
            break
        except Exception:
            pass
    else:
        proc.terminate()
        return {"status": "error", "message": "vLLM server failed to start"}

    try:
        import asyncio
        from engram.config import DataGenConfig
        from engram.training.datagen import generate_llm_dataset

        config = DataGenConfig(
            llm_provider="vllm",
            llm_model=model,
            num_synthetic_users=num_users,
            output_dir=data_dir,
            vllm_url="http://localhost:8000/v1",
        )
        stats = asyncio.run(generate_llm_dataset(config))
        return {"status": "ok", "mode": "llm", "model": model, "stats": stats}
    finally:
        proc.terminate()


# training


@remote(
    resource_config=train_gpu,
    dependencies=[
        "torch",
        "transformers",
        "faiss-cpu",
        "networkx",
        "leidenalg",
        "igraph",
    ],
)
def train_all_remote(
    dev: bool = False,
    datagen_mode: str = "demo",
    num_users: int = 100,
    model: str = "openai/gpt-oss-120b",
    repo_url: str = "https://github.com/kajdev/engram.git",
):
    """generate data and run all three training phases on gpu."""
    import subprocess, os, sys
    engram_dir = "/workspace/engram"
    if not os.path.exists(engram_dir):
        subprocess.run(["git", "clone", "-b", "main", repo_url, engram_dir], check=True)
    subprocess.run(["pip", "install", "-e", engram_dir], check=True, cwd=engram_dir)
    if engram_dir not in sys.path:
        sys.path.insert(0, engram_dir)

    import torch
    from pathlib import Path
    from engram.config import EngramConfig, dev_config
    from engram.training.train_encoder import train_encoder
    from engram.training.train_edge_classifier import train_edge_classifier
    from engram.training.train_synthesis import train_synthesis

    data_dir = Path("/workspace/data")

    # generate data on the same worker
    if datagen_mode == "demo":
        from engram.training.datagen import generate_demo_dataset
        print("[0/3] generating demo data...")
        stats = generate_demo_dataset(data_dir, seed=42)
        print(f"  {stats}")
    else:
        import subprocess, time, urllib.request
        print(f"[0/3] starting vllm server with {model}...")
        proc = subprocess.Popen([
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model, "--max-model-len", "8192",
            "--gpu-memory-utilization", "0.9", "--port", "8000",
        ])
        for attempt in range(120):
            time.sleep(1)
            try:
                urllib.request.urlopen("http://localhost:8000/health")
                print(f"  vllm ready after {attempt+1}s")
                break
            except Exception:
                pass
        else:
            proc.terminate()
            return {"status": "error", "message": "vllm server failed to start"}

        try:
            import asyncio
            from engram.config import DataGenConfig
            from engram.training.datagen import generate_llm_dataset
            config_dg = DataGenConfig(
                llm_provider="vllm", llm_model=model,
                num_synthetic_users=num_users, output_dir=data_dir,
                vllm_url="http://localhost:8000/v1",
            )
            stats = asyncio.run(generate_llm_dataset(config_dg))
            print(f"  {stats}")
        finally:
            proc.terminate()

    config = dev_config() if dev else EngramConfig()
    config.training.output_dir = Path("/workspace/outputs")
    config.training.device = "cuda" if torch.cuda.is_available() else "cpu"

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"training on: {gpu_name}")

    print("[1/3] training encoder...")
    encoder = train_encoder(config, data_dir)

    print("[2/3] training edge classifier...")
    train_edge_classifier(config, data_dir, encoder=encoder)

    print("[3/3] training synthesis encoder...")
    train_synthesis(config, data_dir, encoder=encoder)

    return {
        "status": "ok",
        "gpu": gpu_name,
        "output_path": "/workspace/outputs",
    }


@remote(
    resource_config=train_gpu,
    dependencies=[
        "torch",
        "transformers",
        "faiss-cpu",
        "networkx",
        "leidenalg",
        "igraph",
    ],
)
def train_phase_remote(
    phase: str,
    dev: bool = False,
    repo_url: str = "https://github.com/kajdev/engram.git",
):
    """run a single training phase on gpu."""
    import subprocess, os, sys
    engram_dir = "/workspace/engram"
    if not os.path.exists(engram_dir):
        subprocess.run(["git", "clone", "-b", "main", repo_url, engram_dir], check=True)
    subprocess.run(["pip", "install", "-e", engram_dir], check=True, cwd=engram_dir)
    if engram_dir not in sys.path:
        sys.path.insert(0, engram_dir)

    import torch
    from pathlib import Path
    from engram.config import EngramConfig, dev_config

    config = dev_config() if dev else EngramConfig()
    config.training.output_dir = Path("/workspace/outputs")
    config.training.device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = Path("/workspace/data")

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

    if phase == "encoder":
        from engram.training.train_encoder import train_encoder

        train_encoder(config, data_dir)
    elif phase == "edge":
        from engram.training.train_edge_classifier import train_edge_classifier

        train_edge_classifier(config, data_dir)
    elif phase == "synthesis":
        from engram.training.train_synthesis import train_synthesis

        train_synthesis(config, data_dir)
    else:
        return {"status": "error", "message": f"Unknown phase: {phase}"}

    return {"status": "ok", "phase": phase, "gpu": gpu_name}


# cli


async def main():
    parser = argparse.ArgumentParser(description="Train Engram on RunPod Flash")
    parser.add_argument("--dev", action="store_true", help="Use small dev config")
    parser.add_argument(
        "--phase",
        choices=["all", "encoder", "edge", "synthesis", "datagen"],
        default="all",
    )
    parser.add_argument("--datagen-mode", choices=["demo", "llm"], default="demo")
    parser.add_argument("--num-users", type=int, default=100)
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-oss-120b",
        help="Open-source model for LLM data generation. "
        "Options: openai/gpt-oss-120b, "
        "meta-llama/Llama-4-Scout-17B-16E-Instruct, "
        "Qwen/Qwen2.5-72B-Instruct-AWQ",
    )
    parser.add_argument(
        "--repo-url",
        type=str,
        default=REPO_URL,
        help="Git repo URL for remote workers to clone",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("ENGRAM — RunPod Flash Training")
    print("=" * 60)

    if args.phase == "datagen":
        print(f"Generating training data ({args.datagen_mode})...")
        if args.datagen_mode == "llm":
            print(f"  Model: {args.model}")
            print(f"  Users: {args.num_users}")
        result = await generate_data_remote(
            mode=args.datagen_mode,
            num_users=args.num_users,
            model=args.model,
            repo_url=args.repo_url,
        )
        print(f"Result: {result}")

    elif args.phase == "all":
        print("running full pipeline: datagen + train")
        print()
        train_result = await train_all_remote(
            dev=args.dev,
            datagen_mode=args.datagen_mode,
            num_users=args.num_users,
            model=args.model,
            repo_url=args.repo_url,
        )
        print(f"  {train_result}")

    else:
        print(f"Training phase: {args.phase}")
        result = await train_phase_remote(
            phase=args.phase,
            dev=args.dev,
            repo_url=args.repo_url,
        )
        print(f"Result: {result}")

    print()
    print("=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

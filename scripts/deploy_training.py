"""deploy a training pod on runpod with a network volume.

spawns a single-instance gpu pod that:
1. clones the engram repo
2. installs dependencies
3. runs the full training pipeline
4. saves everything to the network volume

usage:
    # deploy training pod (creates network volume if needed)
    python scripts/deploy_training.py

    # deploy with specific gpu
    python scripts/deploy_training.py --gpu "NVIDIA H100 80GB HBM3"

    # dev mode (small model, few epochs, quick validation)
    python scripts/deploy_training.py --dev

    # generate data on the pod first (pass vllm endpoint url)
    python scripts/deploy_training.py --datagen-url https://api.runpod.ai/v2/xxx/openai/v1

    # check pod status
    python scripts/deploy_training.py --status POD_ID

    # terminate pod
    python scripts/deploy_training.py --terminate POD_ID
"""

import argparse
import os
import sys

# load .env before anything else
from pathlib import Path
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from engram.runpod_api import RunPodClient

REPO_URL = "https://github.com/kajdev/engram.git"
REPO_BRANCH = "main"

# pytorch image with cuda support
TRAIN_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"

# default gpu for training (e5-large is ~330M params, a100 is plenty)
DEFAULT_GPU = "NVIDIA A100 80GB PCIe"

# network volume config
VOLUME_NAME = "engram-data-v2"
VOLUME_SIZE_GB = 200
VOLUME_DATACENTER = "US-TX-3"


def _build_startup_script(
    repo_url: str,
    branch: str,
    dev: bool = False,
    datagen_url: str | None = None,
    datagen_model: str | None = None,
    num_users: int = 500,
    runpod_api_key: str = "",
) -> str:
    """build the bash script that runs on pod startup."""
    lines = [
        "#!/bin/bash",
        "set -e",
        "",
        "echo '============================================'",
        "echo 'ENGRAM TRAINING POD'",
        "echo '============================================'",
        "",
        "# setup",
        "cd /workspace",
        f"git clone -b {branch} {repo_url} engram 2>/dev/null || (cd engram && git pull)",
        "cd engram",
        "pip install -e '.[datagen]' 2>&1 | tail -5",
        "",
        "# check gpu",
        "python -c \"import torch; print(f'cuda: {torch.cuda.is_available()}, gpus: {torch.cuda.device_count()}')\"",
        "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader",
        "",
    ]

    # data generation phase (if vllm endpoint provided)
    if datagen_url:
        model = datagen_model or "meta-llama/Llama-3.3-70B-Instruct"
        lines.extend([
            "# generate training data via vllm endpoint",
            f"echo 'generating training data ({num_users} users)...'",
            f"export RUNPOD_API_KEY='{runpod_api_key}'",
            f"python scripts/generate_data.py \\",
            f"  --mode llm \\",
            f"  --provider vllm \\",
            f"  --vllm-url '{datagen_url}' \\",
            f"  --model '{model}' \\",
            f"  --num-users {num_users} \\",
            f"  --output-dir /runpod-volume/data",
            "",
        ])
    else:
        lines.extend([
            "# check for existing data on network volume",
            "if [ -f /runpod-volume/data/memories.jsonl ]; then",
            "  echo 'found existing data on network volume'",
            "  ls -la /runpod-volume/data/",
            "else",
            "  echo 'no data found, generating demo dataset...'",
            "  python scripts/generate_data.py --mode demo --output-dir /runpod-volume/data",
            "fi",
            "",
        ])

    # training phase
    dev_flag = "--dev" if dev else ""
    lines.extend([
        "# run training",
        "echo '============================================'",
        "echo 'STARTING TRAINING'",
        "echo '============================================'",
        f"python scripts/train.py \\",
        f"  --data-dir /runpod-volume/data \\",
        f"  --output-dir /runpod-volume/outputs \\",
        f"  --device cuda \\",
        f"  {dev_flag}",
        "",
        "echo '============================================'",
        "echo 'TRAINING COMPLETE'",
        "echo 'models saved to /runpod-volume/outputs/'",
        "ls -la /runpod-volume/outputs/",
        "echo '============================================'",
        "",
        "# keep pod alive for inspection (will auto-stop based on idle timeout)",
        "echo 'pod staying alive for inspection. terminate when done.'",
        "sleep infinity",
    ])

    return "\n".join(lines)


def deploy(args):
    """deploy training pod."""
    client = RunPodClient()

    print("=" * 60)
    print("ENGRAM — deploy training pod")
    print("=" * 60)

    me = client.get_myself()
    print(f"  account: {me.get('email', 'unknown')}")
    print(f"  spend/hr: ${me.get('currentSpendPerHr', 0):.2f}")
    print()

    # get or create network volume
    print(f"  setting up network volume '{VOLUME_NAME}'...")
    volume = client.get_or_create_network_volume(
        name=VOLUME_NAME,
        size_gb=VOLUME_SIZE_GB,
        datacenter_id=VOLUME_DATACENTER,
    )
    volume_id = volume["id"]
    print(f"  volume: {volume_id} ({volume['name']}, {volume['size']}GB, {volume['dataCenterId']})")
    print()

    # build startup script
    startup_script = _build_startup_script(
        repo_url=args.repo_url,
        branch=args.branch,
        dev=args.dev,
        datagen_url=args.datagen_url,
        datagen_model=args.datagen_model,
        num_users=args.num_users,
        runpod_api_key=client.api_key,
    )

    print(f"  gpu: {args.gpu}")
    print(f"  image: {TRAIN_IMAGE}")
    print(f"  dev mode: {args.dev}")
    if args.datagen_url:
        print(f"  datagen url: {args.datagen_url}")
        print(f"  datagen model: {args.datagen_model or 'default'}")
        print(f"  num users: {args.num_users}")
    print()

    # deploy pod
    print("  deploying pod...")
    pod = client.create_pod(
        name="engram-train",
        image_name=TRAIN_IMAGE,
        gpu_type_id=args.gpu,
        gpu_count=1,
        volume_gb=100,
        container_disk_gb=50,
        min_vcpu=8,
        min_memory_gb=32,
        network_volume_id=volume_id,
        ports="22/tcp,8888/http",
        env={
            "JUPYTER_PASSWORD": "engram",
            "RUNPOD_API_KEY": client.api_key,
        },
        docker_args=f"bash -c '{startup_script}'",
    )

    pod_id = pod.get("id")
    print(f"  pod deployed: {pod_id}")
    print()

    # wait for pod to be ready
    ready_pod = client.wait_for_pod(pod_id, timeout=300)
    if ready_pod:
        runtime = ready_pod.get("runtime", {})
        ports = runtime.get("ports", [])
        print()
        print("=" * 60)
        print("TRAINING POD DEPLOYED")
        print("=" * 60)
        print(f"  pod id:     {pod_id}")
        for port_info in ports:
            if port_info.get("isIpPublic"):
                print(f"  {port_info['privatePort']}/tcp:  {port_info['ip']}:{port_info['publicPort']}")
        print()
        print("  to check status:")
        print(f"    python scripts/deploy_training.py --status {pod_id}")
        print()
        print("  to terminate when done:")
        print(f"    python scripts/deploy_training.py --terminate {pod_id}")
        print()
        print("  models will be saved to /runpod-volume/outputs/ on the network volume")
        print("=" * 60)
    else:
        print(f"  pod {pod_id} is starting but not ready yet.")
        print(f"  check status with: python scripts/deploy_training.py --status {pod_id}")


def check_status(args):
    """check pod status."""
    client = RunPodClient()
    pod_id = args.status
    pod = client.get_pod(pod_id)
    print(f"pod {pod_id}:")
    print(f"  name: {pod.get('name')}")
    print(f"  status: {pod.get('desiredStatus')}")
    runtime = pod.get("runtime", {})
    if runtime:
        print(f"  uptime: {runtime.get('uptimeInSeconds', 0)}s")
        for port_info in runtime.get("ports", []):
            if port_info.get("isIpPublic"):
                print(f"  {port_info['privatePort']}/tcp: {port_info['ip']}:{port_info['publicPort']}")


def terminate(args):
    """terminate pod."""
    client = RunPodClient()
    pod_id = args.terminate
    print(f"terminating pod {pod_id}...")
    ok = client.terminate_pod(pod_id)
    if ok:
        print("  terminated. network volume data is preserved.")
    else:
        print("  failed to terminate (may already be gone)")


def main():
    parser = argparse.ArgumentParser(description="Deploy Engram training pod on RunPod")
    parser.add_argument("--gpu", type=str, default=DEFAULT_GPU,
                       help=f"gpu type. default: {DEFAULT_GPU}")
    parser.add_argument("--dev", action="store_true",
                       help="dev mode: small model, few epochs")
    parser.add_argument("--repo-url", type=str, default=REPO_URL)
    parser.add_argument("--branch", type=str, default=REPO_BRANCH)

    # datagen on the pod
    parser.add_argument("--datagen-url", type=str, default=None,
                       help="vllm endpoint url for data generation on the pod")
    parser.add_argument("--datagen-model", type=str, default=None,
                       help="model name for datagen")
    parser.add_argument("--num-users", type=int, default=500,
                       help="number of synthetic users to generate")

    # management
    parser.add_argument("--status", type=str, metavar="POD_ID",
                       help="check pod status")
    parser.add_argument("--terminate", type=str, metavar="POD_ID",
                       help="terminate a pod")
    parser.add_argument("--list-volumes", action="store_true",
                       help="list network volumes")

    args = parser.parse_args()

    if args.status:
        check_status(args)
    elif args.terminate:
        terminate(args)
    elif args.list_volumes:
        client = RunPodClient()
        volumes = client.get_network_volumes()
        for vol in volumes:
            print(f"  {vol['id']}: {vol['name']} ({vol['size']}GB, {vol['dataCenterId']})")
    else:
        deploy(args)


if __name__ == "__main__":
    main()


"""deploy a vllm serverless endpoint on runpod for synthetic data generation.

build the worker image first, then deploy it here.

usage:
    # build and push the image first
    .\\docker\\build_and_push.ps1 -DockerUser yourusername

    # deploy (defaults to qwen 2.5 72b on 2x a100 80gb)
    python scripts/deploy_vllm.py --image yourusername/engram-vllm-worker:latest

    # check status of existing endpoint
    python scripts/deploy_vllm.py --status ENDPOINT_ID

    # tear down endpoint when done
    python scripts/deploy_vllm.py --delete ENDPOINT_ID
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


# good models for synthetic data gen (vllm compatible, high benchmark scores)
RECOMMENDED_MODELS = {
    "llama-3.3-70b": "meta-llama/Llama-3.3-70B-Instruct",
    "qwen-2.5-72b": "Qwen/Qwen2.5-72B-Instruct",
    "llama-3.1-70b": "meta-llama/Llama-3.1-70B-Instruct",
}

DEFAULT_MODEL = "Qwen/Qwen2.5-72B-Instruct"

# default image — build with docker/build_and_push.ps1 first
DEFAULT_IMAGE = "engram-vllm-worker:latest"

# 70b models need ~140gb vram for bf16, so 2x a100-80gb or 2x h100
DEFAULT_GPU = "NVIDIA A100 80GB PCIe"


def deploy(args):
    """deploy the vllm endpoint."""
    client = RunPodClient()

    print("=" * 60)
    print("ENGRAM — deploy vLLM serverless endpoint")
    print("=" * 60)

    # verify account
    me = client.get_myself()
    print(f"  account: {me.get('email', 'unknown')}")
    print(f"  spend/hr: ${me.get('currentSpendPerHr', 0):.2f}")
    print()

    model = args.model
    image = args.image
    print(f"  model: {model}")
    print(f"  image: {image}")
    print(f"  gpu: {args.gpu}")
    print(f"  max workers: {args.max_workers}")
    print(f"  idle timeout: {args.idle_timeout}s")
    if args.tensor_parallel:
        print(f"  tensor parallel: {args.tensor_parallel}")
    print()

    # build env vars for the worker
    env = {
        "MODEL_NAME": model,
        "MAX_MODEL_LEN": str(args.max_model_len),
        "GPU_MEMORY_UTILIZATION": "0.90",
        "DISABLE_LOG_STATS": "true",
    }
    if args.tensor_parallel:
        env["TENSOR_PARALLEL_SIZE"] = str(args.tensor_parallel)

    # create template for vllm worker
    print("  creating serverless template...")
    template = client.create_template(
        name=f"engram-vllm-{model.split('/')[-1][:30]}",
        image_name=image,
        env=env,
        container_disk_gb=args.container_disk_gb,
        volume_gb=200,
        is_serverless=True,
    )
    template_id = template.get("id")
    print(f"  template created: {template_id}")

    # create serverless endpoint
    print("  creating serverless endpoint...")
    endpoint = client.create_endpoint(
        name=f"engram-datagen-{model.split('/')[-1][:20]}",
        template_id=template_id,
        gpu_ids=args.gpu,
        workers_min=args.min_workers,
        workers_max=args.max_workers,
        idle_timeout=args.idle_timeout,
    )
    endpoint_id = endpoint.get("id")
    print(f"  endpoint created: {endpoint_id}")

    # get the endpoint base url (handler uses /runsync)
    endpoint_url = RunPodClient.endpoint_openai_url(endpoint_id)
    print()
    print("=" * 60)
    print("ENDPOINT DEPLOYED")
    print("=" * 60)
    print(f"  endpoint id:   {endpoint_id}")
    print(f"  endpoint url:  {endpoint_url}")
    print()
    print("  to generate data, run:")
    print(f"    python scripts/generate_data.py --mode llm --provider vllm \\")
    print(f"      --vllm-url {endpoint_url} \\")
    print(f"      --model {model} \\")
    print(f"      --num-users 500")
    print()
    print("  to check status:")
    print(f"    python scripts/deploy_vllm.py --status {endpoint_id}")
    print()
    print("  to tear down when done:")
    print(f"    python scripts/deploy_vllm.py --delete {endpoint_id}")
    print()
    print("  NOTE: the first request may take a few minutes while the")
    print("  worker initializes and downloads the model. subsequent")
    print("  requests will be fast. workers auto-scale to 0 after idle.")
    print("=" * 60)


def status(args):
    """check endpoint status."""
    client = RunPodClient()
    endpoint_id = args.status

    print(f"checking endpoint {endpoint_id}...")
    health = client.endpoint_health(endpoint_id)
    print(f"  health: {health}")

    endpoint = client.get_endpoint(endpoint_id)
    print(f"  config: {endpoint}")


def delete(args):
    """delete an endpoint."""
    client = RunPodClient()
    endpoint_id = args.delete

    print(f"deleting endpoint {endpoint_id}...")
    ok = client.delete_endpoint(endpoint_id)
    if ok:
        print("  deleted successfully")
    else:
        print("  failed to delete (may already be gone)")


def list_gpus(args):
    """list available gpu types."""
    client = RunPodClient()
    gpus = client.get_gpu_types()
    print(f"{'GPU Type':<45} {'VRAM':>8} {'Secure':>8} {'Community':>10}")
    print("-" * 75)
    for gpu in sorted(gpus, key=lambda g: g.get("memoryInGb", 0), reverse=True):
        name = gpu.get("id", "unknown")
        mem = gpu.get("memoryInGb", 0)
        secure = "yes" if gpu.get("secureCloud") else "no"
        community = "yes" if gpu.get("communityCloud") else "no"
        print(f"  {name:<43} {mem:>6}GB {secure:>8} {community:>10}")


def main():
    parser = argparse.ArgumentParser(description="Deploy vLLM endpoint on RunPod")
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"huggingface model id. defaults to {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--image", type=str, default=DEFAULT_IMAGE,
        help=f"docker image for the worker. defaults to {DEFAULT_IMAGE}.",
    )
    parser.add_argument(
        "--gpu", type=str, default=DEFAULT_GPU,
        help=f"gpu type id. defaults to {DEFAULT_GPU}",
    )
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--min-workers", type=int, default=0)
    parser.add_argument("--idle-timeout", type=int, default=300,
                       help="seconds before idle workers scale down")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--tensor-parallel", type=int, default=None,
                       help="number of gpus for tensor parallelism (e.g. 2 for 70b models)")
    parser.add_argument("--container-disk-gb", type=int, default=200,
                       help="container disk size in gb")

    # actions
    parser.add_argument("--status", type=str, metavar="ENDPOINT_ID",
                       help="check status of existing endpoint")
    parser.add_argument("--delete", type=str, metavar="ENDPOINT_ID",
                       help="delete an existing endpoint")
    parser.add_argument("--list-gpus", action="store_true",
                       help="list available gpu types")

    args = parser.parse_args()

    if args.list_gpus:
        list_gpus(args)
    elif args.status:
        status(args)
    elif args.delete:
        delete(args)
    else:
        deploy(args)


if __name__ == "__main__":
    main()


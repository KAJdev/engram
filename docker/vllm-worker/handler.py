"""runpod serverless handler for vllm openai server.

wraps the vllm openai-compatible server in a runpod worker so it
can run on runpod serverless. starts vllm as a subprocess on init,
then proxies incoming requests to it.

env vars:
    MODEL_NAME           - huggingface model id (required)
    MAX_MODEL_LEN        - max sequence length (default: 32768)
    GPU_MEMORY_UTILIZATION - fraction of gpu mem to use (default: 0.90)
    TENSOR_PARALLEL_SIZE - number of gpus for tensor parallelism
    QUANTIZATION         - quantization method (awq, gptq, etc)
    VLLM_PORT            - port for vllm server (default: 8000)
    VLLM_EXTRA_ARGS      - extra cli args for vllm (space separated)
    DISABLE_LOG_STATS    - disable vllm stats logging (default: true)
"""

import os
import subprocess
import time

import requests
import runpod

# ── config from env ──────────────────────────────────────────────────

VLLM_PORT = int(os.environ.get("VLLM_PORT", "8000"))
MODEL_NAME = os.environ.get("MODEL_NAME", "")
MAX_MODEL_LEN = os.environ.get("MAX_MODEL_LEN", "32768")
GPU_MEM_UTIL = os.environ.get("GPU_MEMORY_UTILIZATION", "0.90")
TENSOR_PARALLEL = os.environ.get("TENSOR_PARALLEL_SIZE", "")
QUANTIZATION = os.environ.get("QUANTIZATION", "")
DISABLE_LOG_STATS = os.environ.get("DISABLE_LOG_STATS", "true")
VLLM_EXTRA_ARGS = os.environ.get("VLLM_EXTRA_ARGS", "")

# how long to wait for vllm to start (large models need time to load)
STARTUP_TIMEOUT = int(os.environ.get("STARTUP_TIMEOUT", "900"))


# ── vllm lifecycle ───────────────────────────────────────────────────

def build_vllm_cmd() -> list[str]:
    """build the vllm server command from env vars."""
    if not MODEL_NAME:
        raise RuntimeError("MODEL_NAME env var is required")

    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_NAME,
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--max-model-len", MAX_MODEL_LEN,
        "--gpu-memory-utilization", GPU_MEM_UTIL,
        "--trust-remote-code",
    ]

    if TENSOR_PARALLEL:
        cmd.extend(["--tensor-parallel-size", TENSOR_PARALLEL])
    if QUANTIZATION:
        cmd.extend(["--quantization", QUANTIZATION])
    if DISABLE_LOG_STATS.lower() == "true":
        cmd.append("--disable-log-stats")
    if VLLM_EXTRA_ARGS:
        cmd.extend(VLLM_EXTRA_ARGS.split())

    return cmd


def start_vllm() -> subprocess.Popen:
    """start vllm openai server and wait until it's healthy."""
    cmd = build_vllm_cmd()
    print(f"[handler] starting vllm: {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    # poll until healthy or dead
    for i in range(STARTUP_TIMEOUT):
        # check if process died
        if proc.poll() is not None:
            output = proc.stdout.read().decode() if proc.stdout else ""
            raise RuntimeError(
                f"vllm died with code {proc.returncode}:\n{output[-2000:]}"
            )

        try:
            resp = requests.get(
                f"http://localhost:{VLLM_PORT}/health",
                timeout=5,
            )
            if resp.status_code == 200:
                print(f"[handler] vllm ready after {i + 1}s")
                return proc
        except requests.exceptions.ConnectionError:
            pass
        except Exception as e:
            print(f"[handler] health check error: {e}")

        if i > 0 and i % 30 == 0:
            print(f"[handler] still waiting for vllm... ({i}s)")

        time.sleep(1)

    raise RuntimeError(f"vllm failed to start within {STARTUP_TIMEOUT}s")


# ── start vllm on worker init ───────────────────────────────────────

print("[handler] initializing worker...")
vllm_proc = start_vllm()
print("[handler] worker ready, registering handler")


# ── runpod handler ───────────────────────────────────────────────────

def handler(job):
    """proxy incoming runpod requests to the local vllm server.

    input format: standard openai chat/completions request body, with
    an optional 'openai_route' field to specify the endpoint path.

    example input:
        {
            "openai_route": "/v1/chat/completions",
            "model": "gpt-oss-120b",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 512
        }
    """
    input_data = dict(job["input"])

    # which openai endpoint to hit (default: chat completions)
    route = input_data.pop("openai_route", "/v1/chat/completions")

    # streaming not supported through the runpod handler path
    # (use the /openai/v1 proxy for streaming)
    input_data.pop("stream", None)

    url = f"http://localhost:{VLLM_PORT}{route}"

    try:
        resp = requests.post(url, json=input_data, timeout=300)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.Timeout:
        return {"error": "vllm request timed out after 300s"}
    except requests.exceptions.RequestException as e:
        return {"error": f"vllm request failed: {e}"}


# ── register and start ───────────────────────────────────────────────

runpod.serverless.start({"handler": handler})


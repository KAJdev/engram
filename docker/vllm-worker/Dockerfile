# runpod serverless worker for vllm.
#
# standard vllm from pypi — no custom forks, no cursed nightlies.
#
# build:
#   docker build -t YOUR_DOCKERHUB/engram-vllm-worker:latest .
#
# push:
#   docker push YOUR_DOCKERHUB/engram-vllm-worker:latest
#
# required env vars at runtime (set via runpod template):
#   MODEL_NAME                - huggingface model id
#   TENSOR_PARALLEL_SIZE      - num gpus (optional, for large models)
#   MAX_MODEL_LEN             - max context length (default 32768)
#   GPU_MEMORY_UTILIZATION    - vram fraction (default 0.90)

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# python + vllm + handler deps, all in one layer
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
  python3 python3-pip python3-venv python3-dev curl gcc \
  && ln -sf /usr/bin/python3 /usr/bin/python \
  && rm -rf /var/lib/apt/lists/* \
  && pip install --no-cache-dir vllm runpod requests \
  && rm -rf /root/.cache /tmp/* /var/tmp/*

COPY handler.py /handler.py

EXPOSE 8000

CMD ["python", "-u", "/handler.py"]

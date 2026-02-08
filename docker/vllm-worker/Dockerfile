# custom runpod serverless worker for vllm with gpt-oss support.
#
# base image provides vllm with the gptoss model architecture.
# we add the runpod sdk so it works as a serverless worker.
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

ARG BASE_IMAGE=vllm/vllm-openai:gptoss
FROM ${BASE_IMAGE}

# install runpod sdk and requests for the handler
RUN pip install --no-cache-dir runpod requests

# copy our serverless handler
COPY handler.py /handler.py

# vllm openai server listens here, runpod proxies /openai/v1 to it
EXPOSE 8000

# override the base image entrypoint - our handler manages vllm lifecycle
ENTRYPOINT []
CMD ["python", "-u", "/handler.py"]


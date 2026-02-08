#!/usr/bin/env bash
# build and push the custom vllm worker image to dockerhub.
#
# usage:
#   ./docker/build_and_push.sh yourusername
#   ./docker/build_and_push.sh yourusername v1.0
#   BASE_IMAGE=vllm/vllm-openai:gptoss ./docker/build_and_push.sh yourusername
set -euo pipefail

DOCKER_USER="${1:?usage: $0 <dockerhub_username> [tag]}"
TAG="${2:-latest}"
IMAGE_NAME="engram-vllm-worker"
BASE_IMAGE="${BASE_IMAGE:-vllm/vllm-openai:gptoss}"
FULL_IMAGE="${DOCKER_USER}/${IMAGE_NAME}:${TAG}"

echo "========================================"
echo "  engram vllm worker image builder"
echo "========================================"
echo ""
echo "  base image:  ${BASE_IMAGE}"
echo "  output:      ${FULL_IMAGE}"
echo ""

# build
echo "building image..."
docker build \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    -t "${FULL_IMAGE}" \
    -f docker/vllm-worker/Dockerfile \
    docker/vllm-worker/

echo "build successful!"
echo ""

# push
echo "pushing to dockerhub..."
docker push "${FULL_IMAGE}"

echo ""
echo "========================================"
echo "  pushed: ${FULL_IMAGE}"
echo "========================================"
echo ""
echo "  use this image in deploy_vllm.py:"
echo "    python scripts/deploy_vllm.py --image ${FULL_IMAGE}"
echo ""


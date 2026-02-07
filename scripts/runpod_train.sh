#!/bin/bash
# runpod flash training script
# usage: bash scripts/runpod_train.sh [--dev]

set -e

echo "============================================"
echo "Engram Training on RunPod"
echo "============================================"

# install dependencies
pip install -e ".[gpu]" 2>/dev/null || pip install -e .

# check gpu
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPUs: {torch.cuda.device_count()}')"

# generate demo data if not present
if [ ! -f "./data/memories.jsonl" ]; then
    echo "Generating demo training data..."
    python scripts/generate_data.py --mode demo --output-dir ./data
fi

# train
if [ "$1" = "--dev" ]; then
    echo "Running dev training (small model, few epochs)..."
    python scripts/train.py --data-dir ./data --output-dir ./outputs --dev --device cuda
else
    echo "Running full training..."
    python scripts/train.py --data-dir ./data --output-dir ./outputs --device cuda
fi

echo "============================================"
echo "Training complete! Models in ./outputs/"
echo "============================================"

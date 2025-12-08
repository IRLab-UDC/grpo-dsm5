#!/bin/bash

set -e

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <model_name> [output_dir]"
  echo "  model_name: HuggingFace model name (e.g., google/gemma-3-4b)"
  echo "  output_dir: directory for trained models (default: saved_models)"
  exit 1
fi

MODEL="$1"
OUTPUT_DIR="${2:-saved_models}"
SFT_DIR="${OUTPUT_DIR}/sft"
GRPO_DIR="${OUTPUT_DIR}/grpo"

echo "=== Training Pipeline ==="
echo "Model: $MODEL"
echo "Output: $OUTPUT_DIR"

echo ""
echo "[1/2] Supervised Fine-Tuning..."
python train_sft.py \
  --model "$MODEL" \
  --output-dir "$SFT_DIR" \
  --max-steps 200

echo ""
echo "[2/2] GRPO Reinforcement Learning..."
python train_grpo.py \
  --model "$SFT_DIR" \
  --output-dir "$GRPO_DIR" \
  --max-steps 500

echo ""
echo "=== Training Complete ==="
echo "SFT checkpoint: $SFT_DIR"
echo "GRPO checkpoint: $GRPO_DIR"

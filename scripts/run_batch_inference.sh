#!/bin/bash

OLLAMA_HOST="${OLLAMA_HOST:-localhost}"
OLLAMA_PORT="${OLLAMA_PORT:-11434}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <model_name> [reasoning] [n_shots]"
  echo "  model_name: Ollama model name (e.g., gemma3:4b)"
  echo "  reasoning: 0 or 1 (default: 1)"
  echo "  n_shots: number of few-shot examples (default: 0)"
  exit 1
fi

MODEL="$1"
REASONING="${2:-1}"
SHOTS="${3:-0}"

MODEL_SANITIZED=$(echo "$MODEL" | tr ':' '_')
OUTPUT_DIR="inference_results/reasoning_${REASONING}_shots_${SHOTS}"
EVAL_DIR="eval_outputs/reasoning_${REASONING}_shots_${SHOTS}/${MODEL_SANITIZED}"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$EVAL_DIR"

echo "Running inference: model=$MODEL, reasoning=$REASONING, shots=$SHOTS"

python inference.py \
  --model "$MODEL" \
  --api-base "http://${OLLAMA_HOST}:${OLLAMA_PORT}/v1" \
  --output-file "inference_results.jsonl" \
  --n-shots "$SHOTS"

mv inference_results.jsonl "${OUTPUT_DIR}/${MODEL_SANITIZED}.jsonl"

python evaluate.py \
  --input "${OUTPUT_DIR}/${MODEL_SANITIZED}.jsonl" \
  --output "$EVAL_DIR"

echo "Results saved to $EVAL_DIR"

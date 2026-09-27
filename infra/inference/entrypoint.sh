#!/usr/bin/env bash
set -euo pipefail

exec python3 -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 \
  --port "${PORT:-8080}" \
  --model "${MODEL_DIR}" \
  --served-model-name campaign-baseline \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.90

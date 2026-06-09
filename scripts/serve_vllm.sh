#!/usr/bin/env bash
# Launch a local vLLM server that exposes an OpenAI-compatible API.
#
# The pipeline then talks to it exactly like any hosted API — point
# llm.base_url at http://localhost:8000/v1 and llm.model at $MODEL.
#
# Usage:
#   pip install vllm
#   MODEL=Qwen/Qwen2.5-72B-Instruct TP=2 bash scripts/serve_vllm.sh
#
# Env vars:
#   MODEL   HF model id to serve (default: Qwen/Qwen2.5-7B-Instruct)
#   TP      tensor-parallel size = number of GPUs (default: 1)
#   PORT    server port (default: 8000)
#   MAXLEN  max model context length (default: 8192)
#   GPUUTIL GPU memory utilization 0-1 (default: 0.90)
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
TP="${TP:-1}"
PORT="${PORT:-8000}"
MAXLEN="${MAXLEN:-8192}"
GPUUTIL="${GPUUTIL:-0.90}"

echo "Serving ${MODEL}  (tensor-parallel=${TP}, port=${PORT}, max_len=${MAXLEN})"
echo "OpenAI-compatible endpoint will be: http://localhost:${PORT}/v1"

exec vllm serve "${MODEL}" \
  --tensor-parallel-size "${TP}" \
  --port "${PORT}" \
  --max-model-len "${MAXLEN}" \
  --gpu-memory-utilization "${GPUUTIL}" \
  --enable-prefix-caching \
  --served-model-name "${MODEL}"

# Notes:
# - For strict JSON output, recent vLLM honours response_format={"type":"json_object"};
#   the client already requests it. For older vLLM you can add guided decoding flags.
# - Recommended Russian-capable open models: Qwen2.5-32B/72B-Instruct,
#   Llama-3.1-70B-Instruct, gemma-2-27b-it. Bigger = better generation/judging.

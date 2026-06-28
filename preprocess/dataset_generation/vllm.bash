#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-VL-30B-A3B-Thinking}"
PORT="${PORT:-8099}"
HOST="${HOST:-0.0.0.0}"

VLLM_LOG_LEVEL="${VLLM_LOG_LEVEL:-info}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-102768}"
ALLOWED_LOCAL_MEDIA_PATH="${ALLOWED_LOCAL_MEDIA_PATH:-/}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
VLLM_LOG_LEVEL="${VLLM_LOG_LEVEL}" \
  vllm serve "${MODEL_NAME}" \
  --dtype bfloat16 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --data-parallel-size "${DATA_PARALLEL_SIZE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --limit-mm-per-prompt '{"video": 1}' \
  --media-io-kwargs '{"video": {"num_frames": 64, "fps": -1}}' \
  --mm-processor-kwargs '{"max_pixels": 402144}' \
  --allowed-local-media-path "${ALLOWED_LOCAL_MEDIA_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  ${EXTRA_VLLM_ARGS}

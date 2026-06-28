#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREPROCESS_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PREPROCESS_ROOT}"

POOL_JSONL="${POOL_JSONL:-data/dataset_generation/top1_pool.train.jsonl}"
OUTPUT_JSONL="${OUTPUT_JSONL:-data/dataset_generation/top1_reasoning_grounding.train.jsonl}"
JOBS_JSONL="${JOBS_JSONL:-}"  # optional: jsonl lines with {"pool_jsonl": "...", "output_jsonl": "..."}

VLLM_URL="${VLLM_URL:-http://localhost:8099/v1/chat/completions}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-VL-30B-A3B-Thinking}"
API_KEY="${API_KEY:-${VLLM_API_KEY:-}}"
BACKEND="${BACKEND:-local_vllm}"                  # openai_api | local_vllm

USE_VIDEO="${USE_VIDEO:-1}"
VIDEO_INPUT_TYPE="${VIDEO_INPUT_TYPE:-video_url}"  # video_url | video
VIDEO_URL_PREFIX="${VIDEO_URL_PREFIX:-file://}"
PREFER_NPY="${PREFER_NPY:-1}"                      # 1 => prefer ${VIDEO_NPY_ROOT}/${top1_video_id}.npy
VIDEO_NPY_ROOT="${VIDEO_NPY_ROOT:-data/activitynet/train/video_npy_with_meta}"
VIDEO_NPY_EXT="${VIDEO_NPY_EXT:-.npy}"
VIDEO_META_JSONL="${VIDEO_META_JSONL:-${VIDEO_NPY_ROOT}/meta.jsonl}"  # carries raw_fps/indices for temporal grounding
VIDEO_MAX_FRAMES="${VIDEO_MAX_FRAMES:-0}"
VIDEO_FPS="${VIDEO_FPS:-1.0}"
VIDEO_MIN_PIXELS="${VIDEO_MIN_PIXELS:-0}"
VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-200704}"      # 256 * 28 * 28
VIDEO_TOTAL_PIXELS="${VIDEO_TOTAL_PIXELS:-0}"
TEMPERATURE="${TEMPERATURE:-0.4}"
TOP_P="${TOP_P:-0.9}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
TIMEOUT="${TIMEOUT:-120}"
RETRIES="${RETRIES:-2}"
RETRY_SLEEP="${RETRY_SLEEP:-1.0}"
WORKERS="${WORKERS:-4}"
LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-4}"
LOCAL_IMAGE_PATCH_SIZE="${LOCAL_IMAGE_PATCH_SIZE:-14}"  # 14 * spatial_merge(2) = 28
LOCAL_DTYPE="${LOCAL_DTYPE:-bfloat16}"
LOCAL_TENSOR_PARALLEL_SIZE="${LOCAL_TENSOR_PARALLEL_SIZE:-1}"
LOCAL_GPU_MEMORY_UTILIZATION="${LOCAL_GPU_MEMORY_UTILIZATION:-0.8}"
LOCAL_MAX_MODEL_LEN="${LOCAL_MAX_MODEL_LEN:-102768}"
LOCAL_TRUST_REMOTE_CODE="${LOCAL_TRUST_REMOTE_CODE:-1}"
VISION_PROCESS_PATH="${VISION_PROCESS_PATH:-./videosearch_r1/model/qwen_vl_utils/vision_process.py}"
RESUME="${RESUME:-1}"
INCLUDE_RAW_RESPONSE="${INCLUDE_RAW_RESPONSE:-0}"
LIMIT="${LIMIT:-0}"
SHUFFLE="${SHUFFLE:-0}"
SEED="${SEED:-42}"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

if [[ -z "${JOBS_JSONL}" && ! -f "${POOL_JSONL}" ]]; then
  echo "Missing POOL_JSONL: ${POOL_JSONL}" >&2
  exit 1
fi
if [[ -n "${JOBS_JSONL}" && ! -f "${JOBS_JSONL}" ]]; then
  echo "Missing JOBS_JSONL: ${JOBS_JSONL}" >&2
  exit 1
fi

mkdir -p "$(dirname "${OUTPUT_JSONL}")"

PYTHONPATH=. "${PYTHON_BIN}" -m dataset_generation.generate_match_grounding \
  --pool_jsonl "${POOL_JSONL}" \
  --output_jsonl "${OUTPUT_JSONL}" \
  --jobs_jsonl "${JOBS_JSONL}" \
  --vllm_url "${VLLM_URL}" \
  --model "${MODEL_NAME}" \
  --api_key "${API_KEY}" \
  --backend "${BACKEND}" \
  --use_video "${USE_VIDEO}" \
  --video_input_type "${VIDEO_INPUT_TYPE}" \
  --video_url_prefix "${VIDEO_URL_PREFIX}" \
  --prefer_npy "${PREFER_NPY}" \
  --video_npy_root "${VIDEO_NPY_ROOT}" \
  --video_npy_ext "${VIDEO_NPY_EXT}" \
  --video_meta_jsonl "${VIDEO_META_JSONL}" \
  --video_max_frames "${VIDEO_MAX_FRAMES}" \
  --video_fps "${VIDEO_FPS}" \
  --video_min_pixels "${VIDEO_MIN_PIXELS}" \
  --video_max_pixels "${VIDEO_MAX_PIXELS}" \
  --video_total_pixels "${VIDEO_TOTAL_PIXELS}" \
  --temperature "${TEMPERATURE}" \
  --top_p "${TOP_P}" \
  --max_tokens "${MAX_TOKENS}" \
  --timeout "${TIMEOUT}" \
  --retries "${RETRIES}" \
  --retry_sleep "${RETRY_SLEEP}" \
  --workers "${WORKERS}" \
  --local_batch_size "${LOCAL_BATCH_SIZE}" \
  --local_image_patch_size "${LOCAL_IMAGE_PATCH_SIZE}" \
  --local_dtype "${LOCAL_DTYPE}" \
  --local_tensor_parallel_size "${LOCAL_TENSOR_PARALLEL_SIZE}" \
  --local_gpu_memory_utilization "${LOCAL_GPU_MEMORY_UTILIZATION}" \
  --local_max_model_len "${LOCAL_MAX_MODEL_LEN}" \
  --local_trust_remote_code "${LOCAL_TRUST_REMOTE_CODE}" \
  --vision_process_path "${VISION_PROCESS_PATH}" \
  --include_raw_response "${INCLUDE_RAW_RESPONSE}" \
  --resume "${RESUME}" \
  --limit "${LIMIT}" \
  --shuffle "${SHUFFLE}" \
  --seed "${SEED}"

echo "[dataset_generation] generated: ${OUTPUT_JSONL}"

#!/usr/bin/env bash
set -euo pipefail

_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${_repo_root}/scripts/common/env.bash"
DATASET_ROOT="$(videosearch_dataset_dir charades)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXTRACT_PY="${ROOT_DIR}/extract_npy/extract_npy.py"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATASET_JSONL="${DATASET_JSONL:-${DATASET_ROOT}/test/test_queries.jsonl}"
VIDEO_ID_KEY="${VIDEO_ID_KEY:-video}"
VIDEO_SOURCE_ROOT="${VIDEO_SOURCE_ROOT:-${VIDEOSEARCH_DATA_ROOT}/raw_videos/charades_sta/videos}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATASET_ROOT}/test/video_npy_with_meta}"
VIDEO_FPS="${VIDEO_FPS:-1.0}"
VIDEO_MAXLEN="${VIDEO_MAXLEN:-64}"
VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-200704}" #256 * 28 * 28
VIDEO_MIN_PIXELS="${VIDEO_MIN_PIXELS:-256}"
NUM_WORKERS="${NUM_WORKERS:-10}"
LOG_EVERY="${LOG_EVERY:-50}"
SAVE_META="${SAVE_META:-1}"
META_PATH="${META_PATH:-${OUTPUT_DIR}/meta.jsonl}"
FAILED_PATH="${FAILED_PATH:-${OUTPUT_DIR}/failed_videos.jsonl}"
OVERWRITE="${OVERWRITE:-0}"

DATASET_JSONL="${DATASET_JSONL}" \
VIDEO_ID_KEY="${VIDEO_ID_KEY}" \
VIDEO_SOURCE_ROOT="${VIDEO_SOURCE_ROOT}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
VIDEO_FPS="${VIDEO_FPS}" \
VIDEO_MAXLEN="${VIDEO_MAXLEN}" \
VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS}" \
VIDEO_MIN_PIXELS="${VIDEO_MIN_PIXELS}" \
NUM_WORKERS="${NUM_WORKERS}" \
LOG_EVERY="${LOG_EVERY}" \
SAVE_META="${SAVE_META}" \
META_PATH="${META_PATH}" \
FAILED_PATH="${FAILED_PATH}" \
OVERWRITE="${OVERWRITE}" \
"${PYTHON_BIN}" "${EXTRACT_PY}"

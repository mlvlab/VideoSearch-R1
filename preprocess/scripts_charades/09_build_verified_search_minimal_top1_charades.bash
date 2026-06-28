#!/usr/bin/env bash
set -euo pipefail

_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${_repo_root}/scripts/common/env.bash"
DATASET_ROOT="$(videosearch_dataset_dir charades)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREPROCESS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PREPROCESS_ROOT}"

VIDEOSEARCH_ROOT="${VIDEOSEARCH_ROOT:-${VIDEOSEARCH_REPO_ROOT}}"

RAW_ANNOTATION="${RAW_ANNOTATION:-${DATASET_ROOT}/raw_annotation/train.jsonl}"
QUERY_EMBEDDINGS_PATH="${QUERY_EMBEDDINGS_PATH:-${DATASET_ROOT}/train/query_embedding/query_embeddings.train.npy}"
QUERY_META_PATH="${QUERY_META_PATH:-${DATASET_ROOT}/train/query_embedding/query_meta.train.jsonl}"
INDEX_FAISS_PATH="${INDEX_FAISS_PATH:-${DATASET_ROOT}/train/index/index.faiss}"
INDEX_ID_MAP_PATH="${INDEX_ID_MAP_PATH:-${DATASET_ROOT}/train/index/id_map.json}"
VIDEO_ROOT="${VIDEO_ROOT:-${DATASET_ROOT}/train/video_npy_with_meta}"

OUTPUT_JSONL="${OUTPUT_JSONL:-${VIDEOSEARCH_ROOT}/data/verified_search/charades.train.minimal_top1_system_prompt.jsonl}"
OUTPUT_STATS_JSON="${OUTPUT_STATS_JSON:-${VIDEOSEARCH_ROOT}/data/verified_search/stats.charades.minimal_top1.json}"
DATA_CONFIG_PATH="${DATA_CONFIG_PATH:-${VIDEOSEARCH_ROOT}/data/data_config.yaml}"
DATASET_NAME="${DATASET_NAME:-Charades}"
DATASET_KEY="${DATASET_KEY:-Verified-Search-Minimal-Top1-Charades}"
ANNO_PATH_IN_CONFIG="${ANNO_PATH_IN_CONFIG:-data/verified_search/charades.train.minimal_top1_system_prompt.jsonl}"

QUERY_KEY="${QUERY_KEY:-fig_desc}"
GT_KEY="${GT_KEY:-video}"
TIME_KEY="${TIME_KEY:-time}"
DURATION_KEY="${DURATION_KEY:-duration}"
BOOTSTRAP_KEY="${BOOTSTRAP_KEY:-retrieved_video}"
HARD_NEGATIVE_TOPK="${HARD_NEGATIVE_TOPK:-24}"
HARD_NEGATIVE_DEPTH="${HARD_NEGATIVE_DEPTH:-200}"
DROP_MISSING_VIDEO_NPY="${DROP_MISSING_VIDEO_NPY:-1}"
AUGMENT_MATCH_TO_BALANCE="${AUGMENT_MATCH_TO_BALANCE:-1}"
TARGET_MATCH_RATIO="${TARGET_MATCH_RATIO:-1.0}"
AUGMENT_SEED="${AUGMENT_SEED:-42}"
MAX_ROWS="${MAX_ROWS:-0}"
REGISTER_CONFIG="${REGISTER_CONFIG:-1}"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

PYTHONPATH=. "${PYTHON_BIN}" -m data_annotation_process.build_verified_search_minimal_top1 \
  --dataset-name "${DATASET_NAME}" \
  --dataset-key "${DATASET_KEY}" \
  --videosearch-root "${VIDEOSEARCH_ROOT}" \
  --raw-annotation "${RAW_ANNOTATION}" \
  --query-embeddings-path "${QUERY_EMBEDDINGS_PATH}" \
  --query-meta-path "${QUERY_META_PATH}" \
  --index-faiss-path "${INDEX_FAISS_PATH}" \
  --index-id-map-path "${INDEX_ID_MAP_PATH}" \
  --video-root "${VIDEO_ROOT}" \
  --output-jsonl "${OUTPUT_JSONL}" \
  --output-stats-json "${OUTPUT_STATS_JSON}" \
  --data-config-path "${DATA_CONFIG_PATH}" \
  --anno-path-in-config "${ANNO_PATH_IN_CONFIG}" \
  --query-key "${QUERY_KEY}" \
  --gt-key "${GT_KEY}" \
  --time-key "${TIME_KEY}" \
  --duration-key "${DURATION_KEY}" \
  --bootstrap-key "${BOOTSTRAP_KEY}" \
  --hard-negative-topk "${HARD_NEGATIVE_TOPK}" \
  --hard-negative-depth "${HARD_NEGATIVE_DEPTH}" \
  --drop-missing-video-npy "${DROP_MISSING_VIDEO_NPY}" \
  --augment-match-to-balance "${AUGMENT_MATCH_TO_BALANCE}" \
  --target-match-ratio "${TARGET_MATCH_RATIO}" \
  --augment-seed "${AUGMENT_SEED}" \
  --max-rows "${MAX_ROWS}" \
  --register-config "${REGISTER_CONFIG}"

echo "[charades][06_build_verified_search_minimal_top1] done: ${OUTPUT_JSONL}"

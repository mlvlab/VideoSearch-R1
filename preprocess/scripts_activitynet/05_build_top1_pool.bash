#!/usr/bin/env bash
set -euo pipefail

_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${_repo_root}/scripts/common/env.bash"
DATASET_ROOT="$(videosearch_dataset_dir activitynet)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREPROCESS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PREPROCESS_ROOT}"

STRUCTURED_ROOT="${STRUCTURED_ROOT:-${DATASET_ROOT}/train}"
VIDEO_EMBED_SUBDIR="${VIDEO_EMBED_SUBDIR:-video_embedding_1fps}"
FALLBACK_VIDEO_EMBED_SUBDIR="${FALLBACK_VIDEO_EMBED_SUBDIR:-video_embedding}"

QUERY_EMBEDDINGS="${QUERY_EMBEDDINGS:-${STRUCTURED_ROOT}/query_embedding/query_embeddings.train.npy}"
QUERY_META="${QUERY_META:-${STRUCTURED_ROOT}/query_embedding/query_meta.train.jsonl}"
VIDEO_EMBEDDINGS="${VIDEO_EMBEDDINGS:-}"
VIDEO_META="${VIDEO_META:-}"
DOCID2ROW="${DOCID2ROW:-}"
INDEX_DIR="${INDEX_DIR:-${STRUCTURED_ROOT}/index}"
INDEX_PATH="${INDEX_PATH:-}"
ID_MAP_JSON="${ID_MAP_JSON:-}"

USE_FAISS="${USE_FAISS:-1}"
NORMALIZE_QUERIES="${NORMALIZE_QUERIES:-1}"
NORMALIZE_VIDEOS="${NORMALIZE_VIDEOS:-1}"
QUERY_BATCH_SIZE="${QUERY_BATCH_SIZE:-64}"
VIDEO_CHUNK_SIZE="${VIDEO_CHUNK_SIZE:-4096}"
NUM_MATCH="${NUM_MATCH:-1000}"
NUM_NOT_MATCH="${NUM_NOT_MATCH:-1000}"
ALLOW_SMALLER="${ALLOW_SMALLER:-1}"
EXCLUDE_MISSING_POS="${EXCLUDE_MISSING_POS:-1}"
LIMIT="${LIMIT:-0}"
SEED="${SEED:-42}"
SHUFFLE_OUTPUT="${SHUFFLE_OUTPUT:-1}"

OUTPUT_DIR="${OUTPUT_DIR:-${DATASET_ROOT}/sft_data}"
OUTPUT_JSONL="${OUTPUT_JSONL:-${OUTPUT_DIR}/top1_pool.train.jsonl}"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

mkdir -p "${OUTPUT_DIR}"

PYTHONPATH=. "${PYTHON_BIN}" -m dataset_generation.build_top1_pool \
  --structured_root "${STRUCTURED_ROOT}" \
  --video_embed_subdir "${VIDEO_EMBED_SUBDIR}" \
  --fallback_video_embed_subdir "${FALLBACK_VIDEO_EMBED_SUBDIR}" \
  --query_embeddings "${QUERY_EMBEDDINGS}" \
  --query_meta "${QUERY_META}" \
  --video_embeddings "${VIDEO_EMBEDDINGS}" \
  --video_meta "${VIDEO_META}" \
  --docid2row "${DOCID2ROW}" \
  --index_dir "${INDEX_DIR}" \
  --index_path "${INDEX_PATH}" \
  --id_map_json "${ID_MAP_JSON}" \
  --use_faiss "${USE_FAISS}" \
  --normalize_queries "${NORMALIZE_QUERIES}" \
  --normalize_videos "${NORMALIZE_VIDEOS}" \
  --query_batch_size "${QUERY_BATCH_SIZE}" \
  --video_chunk_size "${VIDEO_CHUNK_SIZE}" \
  --num_match "${NUM_MATCH}" \
  --num_not_match "${NUM_NOT_MATCH}" \
  --allow_smaller "${ALLOW_SMALLER}" \
  --exclude_missing_pos "${EXCLUDE_MISSING_POS}" \
  --limit "${LIMIT}" \
  --seed "${SEED}" \
  --shuffle_output "${SHUFFLE_OUTPUT}" \
  --output_jsonl "${OUTPUT_JSONL}"

echo "[dataset_generation][activitynet] pool built: ${OUTPUT_JSONL}"

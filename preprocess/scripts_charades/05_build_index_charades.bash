#!/usr/bin/env bash
set -euo pipefail

_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${_repo_root}/scripts/common/env.bash"
DATASET_ROOT="$(videosearch_dataset_dir charades)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export DEBUG="${DEBUG:-1}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${VIDEOSEARCH_DATA_ROOT}}"
STRUCTURED_ROOT="${STRUCTURED_ROOT:-${DATASET_ROOT}}"
SPLITS="${SPLITS:-train,test}"

OUT_SUFFIX="${OUT_SUFFIX:-1fps}"
EMBED_SUBDIR="${EMBED_SUBDIR:-video_embedding_${OUT_SUFFIX}}"

EVAL_TEST="${EVAL_TEST:-1}"
EVAL_TOPKS="${EVAL_TOPKS:-1,5,10,100}"
EMBED_MODEL="${EMBED_MODEL:-${VIDEOSEARCH_EMBED_MODEL}}"
EMBED_DEVICE="${EMBED_DEVICE:-0}"
USE_DEBUG_QUERIES="${USE_DEBUG_QUERIES:-0}"
DEBUG_QUERIES="${DEBUG_QUERIES:-}"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

if [[ ! -f "${REPO_ROOT}/annotation_process/build_faiss.py" ]]; then
  echo "[build_index][error] missing ${REPO_ROOT}/annotation_process/build_faiss.py" >&2
  exit 1
fi
if [[ ! -f "${REPO_ROOT}/annotation_process/eval_retrieval.py" ]]; then
  echo "[build_index][error] missing ${REPO_ROOT}/annotation_process/eval_retrieval.py" >&2
  exit 1
fi

resolve_query_jsonl() {
  local split_dir="$1"
  local split_name="$2"
  local candidates=()
  if [[ "${split_name}" == "train" ]]; then
    candidates=("train_queries.jsonl" "test_queries.jsonl" "val_queries.jsonl")
  elif [[ "${split_name}" == "val" ]]; then
    candidates=("val_queries.jsonl" "test_queries.jsonl" "train_queries.jsonl")
  else
    candidates=("test_queries.jsonl" "val_queries.jsonl" "train_queries.jsonl")
  fi
  local c
  for c in "${candidates[@]}"; do
    if [[ -f "${split_dir}/${c}" ]]; then
      echo "${split_dir}/${c}"
      return 0
    fi
  done
  return 1
}

for SPLIT in $(echo "${SPLITS}" | tr "," " "); do
  SPLIT_DIR="${STRUCTURED_ROOT}/${SPLIT}"
  EMBEDS="${SPLIT_DIR}/${EMBED_SUBDIR}/segment_embeds.npy"
  DOCID2ROW="${SPLIT_DIR}/${EMBED_SUBDIR}/docid2row.json"
  OUT_DIR="${SPLIT_DIR}/index"

  if ! SPLIT_QUERY_JSON="$(resolve_query_jsonl "${SPLIT_DIR}" "${SPLIT}")"; then
    echo "[build_index][error] no query jsonl found in ${SPLIT_DIR} for split=${SPLIT}" >&2
    exit 1
  fi

  BUILD_ARGS=(
    --embeds "${EMBEDS}"
    --docid2row "${DOCID2ROW}"
    --out_dir "${OUT_DIR}"
  )
  if [[ "${USE_DEBUG_QUERIES}" == "1" ]]; then
    if [[ -n "${DEBUG_QUERIES}" ]]; then
      BUILD_ARGS+=(--debug_queries "${DEBUG_QUERIES}")
    else
      BUILD_ARGS+=(--debug_queries "${SPLIT_QUERY_JSON}")
    fi
  fi

  PYTHONPATH=. "${PYTHON_BIN}" -m annotation_process.build_faiss "${BUILD_ARGS[@]}"

  if [[ "${SPLIT}" == "test" && "${EVAL_TEST}" == "1" ]]; then
    QUERY_EMB_DIR="${SPLIT_DIR}/query_embedding"
    QUERY_EMB="${QUERY_EMB_DIR}/query_embeddings.${SPLIT}.npy"
    QUERY_META="${QUERY_EMB_DIR}/query_meta.${SPLIT}.jsonl"
    EVAL_ARGS=(
      --index_dir "${OUT_DIR}"
      --queries "${SPLIT_QUERY_JSON}"
      --embed_model "${EMBED_MODEL}"
      --embed_device "${EMBED_DEVICE}"
      --topk_list "${EVAL_TOPKS}"
      --out_path "${SPLIT_DIR}/metrics.json"
    )
    if [[ -f "${QUERY_EMB}" && -f "${QUERY_META}" ]]; then
      EVAL_ARGS+=(--query_embeddings "${QUERY_EMB}" --query_meta "${QUERY_META}")
    fi

    PYTHONPATH=. "${PYTHON_BIN}" -m annotation_process.eval_retrieval "${EVAL_ARGS[@]}"
  fi
done

echo "[build_index] done"

#!/usr/bin/env bash
set -euo pipefail

_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${_repo_root}/scripts/common/env.bash"
DATASET_ROOT="$(videosearch_dataset_dir didemo)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export DEBUG="${DEBUG:-1}"

INPUT_STRUCTURED_ROOT="${INPUT_STRUCTURED_ROOT:-${DATASET_ROOT}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${VIDEOSEARCH_DATA_ROOT}}"
OUTPUT_STRUCTURED_ROOT="${OUTPUT_STRUCTURED_ROOT:-${DATASET_ROOT}}"
SPLITS="${SPLITS:-train,val,test}"

EMBED_MODEL="${EMBED_MODEL:-${VIDEOSEARCH_EMBED_MODEL}}"
EMBED_DEVICE="${EMBED_DEVICE:-3}"
INSTRUCTION="${INSTRUCTION:-}"
if [[ -z "${INSTRUCTION}" ]]; then
  INSTRUCTION="Represent the user's input"
fi
BATCH_SIZE="${BATCH_SIZE:-32}"
LIMIT="${LIMIT:-0}"
LOG_EVERY="${LOG_EVERY:-50}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
  else
    PYTHON_BIN="python3"
  fi
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
  IN_SPLIT_DIR="${INPUT_STRUCTURED_ROOT}/${SPLIT}"
  OUT_SPLIT_DIR="${OUTPUT_STRUCTURED_ROOT}/${SPLIT}"

  if ! QUERY_JSONL="$(resolve_query_jsonl "${IN_SPLIT_DIR}" "${SPLIT}")"; then
    echo "[embed_queries][error] no query jsonl found in ${IN_SPLIT_DIR} for split=${SPLIT}" >&2
    exit 1
  fi
  OUT_DIR="${OUT_SPLIT_DIR}/query_embedding"

  SKIP_ARGS=()
  if [[ "${SKIP_EXISTING}" == "1" ]]; then
    SKIP_ARGS+=(--skip_existing)
  fi

  PYTHONPATH=. "${PYTHON_BIN}" -m extract_embed.embed_queries \
    --queries "${QUERY_JSONL}" \
    --out_dir "${OUT_DIR}" \
    --split_name "${SPLIT}" \
    --embed_model "${EMBED_MODEL}" \
    --embed_device "${EMBED_DEVICE}" \
    --instruction "${INSTRUCTION}" \
    --batch_size "${BATCH_SIZE}" \
    --limit "${LIMIT}" \
    --log_every "${LOG_EVERY}" \
    "${SKIP_ARGS[@]}"
done

echo "[embed_queries] done"

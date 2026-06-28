#!/usr/bin/env bash
set -euo pipefail

_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${_repo_root}/scripts/common/env.bash"
DATASET_ROOT="$(videosearch_dataset_dir didemo)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export DEBUG="${DEBUG:-1}"

ANNO_ROOT="${ANNO_ROOT:-${VIDEOSEARCH_DATA_ROOT}/fine-grained-anno/didemo-fig}"
TRAIN_JSONL="${TRAIN_JSONL:-${ANNO_ROOT}/didemo_fig_train.jsonl}"
VAL_JSONL="${VAL_JSONL:-${ANNO_ROOT}/didemo_fig_val.jsonl}"
TEST_JSONL="${TEST_JSONL:-${ANNO_ROOT}/didemo_fig_test.jsonl}"
VIDEO_BASE="${VIDEO_BASE:-${VIDEOSEARCH_DATA_ROOT}/raw_videos/didemo/videos}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${VIDEOSEARCH_DATA_ROOT}}"
STRUCTURED_ROOT="${STRUCTURED_ROOT:-${DATASET_ROOT}}"
RAW_ANNO_DIR="${RAW_ANNO_DIR:-${STRUCTURED_ROOT}/raw_annotation}"

SEED="${SEED:-0}"
USE_FULL_VIDEO="${USE_FULL_VIDEO:-1}"
TRAIN_LIMIT="${TRAIN_LIMIT:-0}"
TEST_LIMIT="${TEST_LIMIT:-0}"
REMOVE_TEST_TRAIN_QUERIES="${REMOVE_TEST_TRAIN_QUERIES:-1}"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

TRAIN_OUT="${STRUCTURED_ROOT}/train"
VAL_OUT="${STRUCTURED_ROOT}/val"
TEST_OUT="${STRUCTURED_ROOT}/test"

if [[ ! -f "${REPO_ROOT}/annotation_process/prepare_activitynet_segments.py" ]]; then
  echo "[data_generate][error] missing ${REPO_ROOT}/annotation_process/prepare_activitynet_segments.py" >&2
  exit 1
fi
for p in "${TRAIN_JSONL}" "${VAL_JSONL}" "${TEST_JSONL}"; do
  if [[ ! -f "${p}" ]]; then
    echo "[data_generate][error] missing jsonl: ${p}" >&2
    exit 1
  fi
done

mkdir -p "${RAW_ANNO_DIR}"
cp -f "${TRAIN_JSONL}" "${RAW_ANNO_DIR}/train.jsonl"
cp -f "${VAL_JSONL}" "${RAW_ANNO_DIR}/val.jsonl"
cp -f "${TEST_JSONL}" "${RAW_ANNO_DIR}/test.jsonl"

# Train split -> train/train_queries.jsonl
PYTHONPATH=. "${PYTHON_BIN}" -m annotation_process.prepare_activitynet_segments \
  --train_jsonl "${TRAIN_JSONL}" \
  --video_base "${VIDEO_BASE}" \
  --out_dir "${TRAIN_OUT}" \
  --seed "${SEED}" \
  --train_limit "${TRAIN_LIMIT}" \
  --test_limit "${TEST_LIMIT}" \
  $(if [[ "${USE_FULL_VIDEO}" == "1" ]]; then echo "--use_full_video"; fi)

# Val split -> val/val_queries.jsonl
PYTHONPATH=. "${PYTHON_BIN}" -m annotation_process.prepare_activitynet_segments \
  --test_jsonl "${VAL_JSONL}" \
  --video_base "${VIDEO_BASE}" \
  --out_dir "${VAL_OUT}" \
  --seed "${SEED}" \
  --train_limit "${TRAIN_LIMIT}" \
  --test_limit "${TEST_LIMIT}" \
  $(if [[ "${USE_FULL_VIDEO}" == "1" ]]; then echo "--use_full_video"; fi)

# Test split -> test/test_queries.jsonl
PYTHONPATH=. "${PYTHON_BIN}" -m annotation_process.prepare_activitynet_segments \
  --test_jsonl "${TEST_JSONL}" \
  --video_base "${VIDEO_BASE}" \
  --out_dir "${TEST_OUT}" \
  --seed "${SEED}" \
  --train_limit "${TRAIN_LIMIT}" \
  --test_limit "${TEST_LIMIT}" \
  $(if [[ "${USE_FULL_VIDEO}" == "1" ]]; then echo "--use_full_video"; fi)

finalize_eval_split_queries() {
  local split_out="$1"
  local target_file="$2"
  if [[ ! -f "${split_out}/${target_file}" ]]; then
    if [[ -f "${split_out}/test_queries.jsonl" ]]; then
      mv "${split_out}/test_queries.jsonl" "${split_out}/${target_file}"
    elif [[ -f "${split_out}/train_queries.jsonl" ]]; then
      mv "${split_out}/train_queries.jsonl" "${split_out}/${target_file}"
    fi
  fi
  if [[ ! -f "${split_out}/${target_file}" ]]; then
    echo "[data_generate][error] missing ${target_file} in ${split_out}" >&2
    exit 1
  fi
  if [[ "${REMOVE_TEST_TRAIN_QUERIES}" == "1" ]]; then
    if [[ "${target_file}" != "train_queries.jsonl" ]]; then
      rm -f "${split_out}/train_queries.jsonl"
    fi
    if [[ "${target_file}" != "test_queries.jsonl" ]]; then
      rm -f "${split_out}/test_queries.jsonl"
    fi
  fi
}

# Normalize query file names for downstream:
# train -> train_queries.jsonl, val -> val_queries.jsonl, test -> test_queries.jsonl
finalize_eval_split_queries "${VAL_OUT}" "val_queries.jsonl"
finalize_eval_split_queries "${TEST_OUT}" "test_queries.jsonl"

echo "[data_generate] done"

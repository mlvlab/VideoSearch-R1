#!/usr/bin/env bash
set -euo pipefail

_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${_repo_root}/scripts/common/env.bash"
DATASET_ROOT="$(videosearch_dataset_dir activitynet)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREPROCESS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PREPROCESS_ROOT}"

INPUT_JSONL="${INPUT_JSONL:-${DATASET_ROOT}/sft_data/top1_reasoning_grounding.train.jsonl}"
OUTPUT_JSON="${OUTPUT_JSON:-${DATASET_ROOT}/sft_data/train_oneturn.json}"
SOURCE_TAG="${SOURCE_TAG:-pairs_with_reasoning}"
EPISODE_TYPE="${EPISODE_TYPE:-T1}"
VIDEO_EXT="${VIDEO_EXT:-.npy}"
SYSTEM_PROMPT_FILE="${SYSTEM_PROMPT_FILE:-}"
LIMIT="${LIMIT:-0}"
INDENT="${INDENT:-2}"
EXCLUDE_MODEL_FINISH_REASONS="${EXCLUDE_MODEL_FINISH_REASONS:-length}"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

if [[ ! -f "${INPUT_JSONL}" ]]; then
  echo "Missing INPUT_JSONL: ${INPUT_JSONL}" >&2
  exit 1
fi

mkdir -p "$(dirname "${OUTPUT_JSON}")"

cmd=(
  "${PYTHON_BIN}" -m dataset_generation.convert_match_grounding_to_oneturn
  --input_jsonl "${INPUT_JSONL}"
  --output_json "${OUTPUT_JSON}"
  --source "${SOURCE_TAG}"
  --episode_type "${EPISODE_TYPE}"
  --video_ext "${VIDEO_EXT}"
  --limit "${LIMIT}"
  --indent "${INDENT}"
  --exclude_model_finish_reasons "${EXCLUDE_MODEL_FINISH_REASONS}"
)
if [[ -n "${SYSTEM_PROMPT_FILE}" ]]; then
  cmd+=(--system_prompt_file "${SYSTEM_PROMPT_FILE}")
fi

PYTHONPATH=. "${cmd[@]}"

echo "[dataset_generation][activitynet] converted: ${OUTPUT_JSON}"

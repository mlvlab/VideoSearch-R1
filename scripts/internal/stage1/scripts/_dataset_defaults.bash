#!/usr/bin/env bash
set -euo pipefail

defaults_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${defaults_dir}/../../../.." && pwd)"
source "${repo_root}/scripts/common/env.bash"

declare -ag SFT_DATASET_ENV=()

_write_sft_dataset_info() {
  local dataset_name="$1"
  local anno_path="$2"
  local data_path="$3"
  local out_dir="${VIDEOSEARCH_OUTPUT_ROOT}/configs"
  mkdir -p "${out_dir}"
  local out="${out_dir}/data_config.${SFT_DATASET_TAG}.yaml"
  cat > "${out}" <<EOF
${dataset_name}:
  anno_path: ${anno_path}
  data_path: ${data_path}
EOF
  echo "${out}"
}

_build_sft_dataset_env() {
  local keys=(
    QUERY_EMBEDDER_INPUT_PREFIX
    SFT_DATASET_TAG
    EXP_NAME
    DATASET_INFO
    DATASET_NAME_CSV
    QUERY_EMBEDDINGS_PATH
    QUERY_META_PATH
    VIDEO_EMBEDDINGS_PATH
    VIDEO_DOCID2ROW_PATH
    VIDEO_META_PATH
    HARD_NEGATIVES_PATH
    EXTERNAL_EVAL_VERIFIED_TEST_JSONL
    EXTERNAL_EVAL_VIDEO_ROOT
    EXTERNAL_EVAL_VIDEO_META_PATH
    EXTERNAL_EVAL_QUERY_EMBEDDINGS_PATH
    EXTERNAL_EVAL_QUERY_META_PATH
    EXTERNAL_EVAL_VIDEO_EMBEDDINGS_PATH
    EXTERNAL_EVAL_VIDEO_DOCID2ROW_PATH
  )
  local key
  SFT_DATASET_ENV=()
  for key in "${keys[@]}"; do
    if [[ -v "${key}" ]]; then
      SFT_DATASET_ENV+=("${key}=${!key}")
    fi
  done
}

_reset_sft_dataset_env() {
  unset DATASET_INFO DATASET_NAME_CSV
  unset QUERY_EMBEDDINGS_PATH QUERY_META_PATH VIDEO_EMBEDDINGS_PATH VIDEO_DOCID2ROW_PATH
  unset VIDEO_META_PATH HARD_NEGATIVES_PATH
  unset EXTERNAL_EVAL_VERIFIED_TEST_JSONL EXTERNAL_EVAL_VIDEO_ROOT EXTERNAL_EVAL_VIDEO_META_PATH
  unset EXTERNAL_EVAL_QUERY_EMBEDDINGS_PATH EXTERNAL_EVAL_QUERY_META_PATH
  unset EXTERNAL_EVAL_VIDEO_EMBEDDINGS_PATH EXTERNAL_EVAL_VIDEO_DOCID2ROW_PATH
}

set_sft_dataset_defaults() {
  local dataset="${1:-}"
  if [[ -z "${dataset}" ]]; then
    echo "[dataset_defaults][error] dataset key is required" >&2
    return 1
  fi

  local force_dataset_defaults
  force_dataset_defaults="$(echo "${SFT_FORCE_DATASET_DEFAULTS:-True}" | tr '[:upper:]' '[:lower:]')"
  case "${force_dataset_defaults}" in
    1|true|yes|on) _reset_sft_dataset_env ;;
  esac

  if [[ -z "${QUERY_EMBEDDER_INPUT_PREFIX:-}" ]]; then
    QUERY_EMBEDDER_INPUT_PREFIX="Represent the user's input"
  fi

  local root train_root test_root anno data_path dataset_name train_json
  case "${dataset}" in
    activitynet)
      SFT_DATASET_TAG="activitynet"
      root="$(videosearch_dataset_dir activitynet)"
      EXP_NAME="${EXP_NAME:-activitynet_sft}"
      DATASET_NAME_CSV="${DATASET_NAME_CSV:-ActivityNet-Oneturn_temporal}"
      dataset_name="ActivityNet-Oneturn_temporal"
      MASTER_PORT="${MASTER_PORT:-29103}"
      ;;
    charades|charades-sta|charades_sta)
      SFT_DATASET_TAG="charades"
      root="$(videosearch_dataset_dir charades)"
      EXP_NAME="${EXP_NAME:-charades_sft}"
      DATASET_NAME_CSV="${DATASET_NAME_CSV:-Charades-Oneturn_temporal}"
      dataset_name="Charades-Oneturn_temporal"
      MASTER_PORT="${MASTER_PORT:-29113}"
      ;;
    didemo)
      SFT_DATASET_TAG="didemo"
      root="$(videosearch_dataset_dir didemo)"
      EXP_NAME="${EXP_NAME:-didemo_sft}"
      DATASET_NAME_CSV="${DATASET_NAME_CSV:-DiDeMo-Oneturn_temporal}"
      dataset_name="DiDeMo-Oneturn_temporal"
      MASTER_PORT="${MASTER_PORT:-29123}"
      ;;
    *)
      echo "[dataset_defaults][error] unknown dataset: ${dataset}" >&2
      return 1
      ;;
  esac

  train_root="${root}/train"
  test_root="${root}/test"
  train_json="${SFT_TRAIN_JSON:-${root}/sft_data/train_oneturn.json}"
  data_path="${SFT_VIDEO_ROOT:-${train_root}/video_npy_with_meta}"

  DATASET_INFO="${DATASET_INFO:-$(_write_sft_dataset_info "${dataset_name}" "${train_json}" "${data_path}")}"
  QUERY_EMBEDDINGS_PATH="${QUERY_EMBEDDINGS_PATH:-${train_root}/query_embedding/query_embeddings.train.npy}"
  QUERY_META_PATH="${QUERY_META_PATH:-${train_root}/query_embedding/query_meta.train.jsonl}"
  VIDEO_EMBEDDINGS_PATH="${VIDEO_EMBEDDINGS_PATH:-${train_root}/video_embedding_1fps/segment_embeds.npy}"
  VIDEO_DOCID2ROW_PATH="${VIDEO_DOCID2ROW_PATH:-${train_root}/video_embedding_1fps/docid2row.json}"
  VIDEO_META_PATH="${VIDEO_META_PATH:-${train_root}/video_npy_with_meta/meta.jsonl}"
  HARD_NEGATIVES_PATH="${HARD_NEGATIVES_PATH:-${train_root}/hard_negatives.json}"

  EXTERNAL_EVAL_VERIFIED_TEST_JSONL="${EXTERNAL_EVAL_VERIFIED_TEST_JSONL:-${root}/raw_annotation/test.jsonl}"
  EXTERNAL_EVAL_VIDEO_ROOT="${EXTERNAL_EVAL_VIDEO_ROOT:-${test_root}/video_npy_with_meta}"
  EXTERNAL_EVAL_VIDEO_META_PATH="${EXTERNAL_EVAL_VIDEO_META_PATH:-${test_root}/video_npy_with_meta/meta.jsonl}"
  EXTERNAL_EVAL_QUERY_EMBEDDINGS_PATH="${EXTERNAL_EVAL_QUERY_EMBEDDINGS_PATH:-${test_root}/query_embedding/query_embeddings.test.npy}"
  EXTERNAL_EVAL_QUERY_META_PATH="${EXTERNAL_EVAL_QUERY_META_PATH:-${test_root}/query_embedding/query_meta.test.jsonl}"
  EXTERNAL_EVAL_VIDEO_EMBEDDINGS_PATH="${EXTERNAL_EVAL_VIDEO_EMBEDDINGS_PATH:-${test_root}/video_embedding_1fps/segment_embeds.npy}"
  EXTERNAL_EVAL_VIDEO_DOCID2ROW_PATH="${EXTERNAL_EVAL_VIDEO_DOCID2ROW_PATH:-${test_root}/video_embedding_1fps/docid2row.json}"

  _build_sft_dataset_env
}

maybe_set_default_report_jsonl() {
  local mode="${1:-standard}"
  if [[ -n "${JSONL:-}" || "$#" -gt 1 ]]; then
    return 0
  fi

  local name_pat
  case "${mode}" in
    temporal) name_pat='external_verified_test_temporal_grounding_*.jsonl' ;;
    standard) name_pat='external_verified_test_*.jsonl' ;;
    *) return 0 ;;
  esac

  local latest
  latest="$(find "${VIDEOSEARCH_OUTPUT_ROOT}" -type f -name "${name_pat}" 2>/dev/null | sort | tail -n 1 || true)"
  if [[ -n "${latest}" ]]; then
    export JSONL="${latest}"
    echo "[dataset_defaults] JSONL defaulted to: ${JSONL}"
  fi
}

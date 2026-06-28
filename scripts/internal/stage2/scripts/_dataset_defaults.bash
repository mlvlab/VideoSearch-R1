#!/usr/bin/env bash
set -euo pipefail

defaults_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${defaults_dir}/../../../.." && pwd)"
source "${repo_root}/scripts/common/env.bash"

set_grpo_dataset_defaults() {
  local dataset="${1:-}"
  if [[ -z "${dataset}" ]]; then
    echo "[grpo_dataset_defaults][error] dataset key is required" >&2
    return 1
  fi

  local root train_root
  case "${dataset}" in
    activitynet)
      export GRPO_DATASET_TAG="activitynet"
      root="$(videosearch_dataset_dir activitynet)"
      export EXP_NAME="${EXP_NAME:-activitynet_grpo}"
      export MASTER_PORT="${MASTER_PORT:-29710}"
      ;;
    charades|charades-sta|charades_sta)
      export GRPO_DATASET_TAG="charades"
      root="$(videosearch_dataset_dir charades)"
      export EXP_NAME="${EXP_NAME:-charades_grpo}"
      export MASTER_PORT="${MASTER_PORT:-29910}"
      ;;
    didemo)
      export GRPO_DATASET_TAG="didemo"
      root="$(videosearch_dataset_dir didemo)"
      export EXP_NAME="${EXP_NAME:-didemo_grpo}"
      export MASTER_PORT="${MASTER_PORT:-29820}"
      ;;
    *)
      echo "[grpo_dataset_defaults][error] unknown dataset: ${dataset}" >&2
      return 1
      ;;
  esac

  train_root="${root}/train"
  export VERIFIED_JSONL="${VERIFIED_JSONL:-${root}/raw_annotation/train.jsonl}"
  export VIDEO_ROOT="${VIDEO_ROOT:-${train_root}/video_npy_with_meta}"
  export VIDEO_META_PATH="${VIDEO_META_PATH:-${train_root}/video_npy_with_meta/meta.jsonl}"
  export QUERY_EMBEDDINGS_PATH="${QUERY_EMBEDDINGS_PATH:-${train_root}/query_embedding/query_embeddings.train.npy}"
  export QUERY_META_PATH="${QUERY_META_PATH:-${train_root}/query_embedding/query_meta.train.jsonl}"
  export INDEX_FAISS_PATH="${INDEX_FAISS_PATH:-${train_root}/index/index.faiss}"
  export INDEX_ID_MAP_PATH="${INDEX_ID_MAP_PATH:-${train_root}/index/id_map.json}"
  export VIDEO_EMBEDDINGS_PATH="${VIDEO_EMBEDDINGS_PATH:-${train_root}/video_embedding_1fps/segment_embeds.npy}"
  export VIDEO_DOCID2ROW_PATH="${VIDEO_DOCID2ROW_PATH:-${train_root}/video_embedding_1fps/docid2row.json}"
  export MINIMAL_JSONL="${MINIMAL_JSONL:-${root}/grpo_data/train.minimal_top1.jsonl}"
  export MINIMAL_STATS_JSON="${MINIMAL_STATS_JSON:-${root}/grpo_data/stats.minimal_top1.json}"

  export QUERY_KEY="${QUERY_KEY:-fig_desc}"
  export GT_KEY="${GT_KEY:-video}"
  export GT_TIME_KEY="${GT_TIME_KEY:-time}"
  export GT_DURATION_KEY="${GT_DURATION_KEY:-duration}"
  export BOOTSTRAP_KEY="${BOOTSTRAP_KEY:-retrieved_video}"
  export AUGMENT_MATCH_TO_BALANCE="${AUGMENT_MATCH_TO_BALANCE:-1}"
  export TARGET_MATCH_RATIO="${TARGET_MATCH_RATIO:-1.0}"
  export AUGMENT_SEED="${AUGMENT_SEED:-42}"
  export AUTO_DATASET_INFO="${AUTO_DATASET_INFO:-True}"
  if [[ -z "${QUERY_EMBEDDER_INPUT_PREFIX:-}" ]]; then
    export QUERY_EMBEDDER_INPUT_PREFIX="Represent the user's input"
  fi
}


#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/data_construct/start_from_scratch.bash didemo
  bash scripts/data_construct/start_from_scratch.bash charades
  bash scripts/data_construct/start_from_scratch.bash activitynet

Set the raw annotation/video env vars required by the dataset scripts before running.
Run scripts/data_construct/prepare_raw_layout.bash DATASET to create the expected folders.
Use RUN_FROM_STEP=N to resume from a later step in the ordered pipeline.
EOF
}

dataset="${1:-}"
if [[ -z "${dataset}" || "${dataset}" == "-h" || "${dataset}" == "--help" ]]; then
  usage
  exit 0
fi

run_from_step="${RUN_FROM_STEP:-1}"
if ! [[ "${run_from_step}" =~ ^[0-9]+$ ]]; then
  echo "[data_construct][error] RUN_FROM_STEP must be an integer" >&2
  exit 1
fi

case "${dataset}" in
  activitynet)
    steps=(
      preprocess/scripts_activitynet/01_data_generate.bash
      preprocess/scripts_activitynet/04_extract_train_npy.bash
      preprocess/scripts_activitynet/04_extract_test_npy.bash
      preprocess/scripts_activitynet/02_embed_queries.bash
      preprocess/scripts_activitynet/02_embed_segments.bash
      preprocess/scripts_activitynet/03_build_index.bash
      preprocess/scripts_activitynet/05_build_top1_pool.bash
      preprocess/scripts_activitynet/05_generate_match_grounding.bash
      preprocess/scripts_activitynet/06_convert_to_train_oneturn.bash
      preprocess/scripts_activitynet/08_build_hard_negatives.bash
      preprocess/scripts_activitynet/09_build_verified_search_minimal_top1.bash
    )
    ;;
  didemo)
    steps=(
      preprocess/scripts_didemo/01_data_generate_didemo.bash
      preprocess/scripts_didemo/04_extract_train_npy_didemo.bash
      preprocess/scripts_didemo/04_extract_test_npy_didemo.bash
      preprocess/scripts_didemo/02_embed_queries_didemo.bash
      preprocess/scripts_didemo/02_embed_segments_didemo.bash
      preprocess/scripts_didemo/03_build_index_didemo.bash
      preprocess/scripts_didemo/05_build_top1_pool_didemo.bash
      preprocess/scripts_didemo/05_generate_match_grounding_didemo.bash
      preprocess/scripts_didemo/06_convert_to_train_oneturn_didemo.bash
      preprocess/scripts_didemo/08_build_hard_negatives_didemo.bash
      preprocess/scripts_didemo/09_build_verified_search_minimal_top1_didemo.bash
    )
    ;;
  charades|charades-sta|charades_sta)
    steps=(
      preprocess/scripts_charades/01_data_generate_charades.bash
      preprocess/scripts_charades/04_extract_train_npy_charades.bash
      preprocess/scripts_charades/04_extract_test_npy_charades.bash
      preprocess/scripts_charades/02_embed_queries_charades.bash
      preprocess/scripts_charades/02_embed_segments_charades.bash
      preprocess/scripts_charades/05_build_index_charades.bash
      preprocess/scripts_charades/05_build_top1_pool_charades.bash
      preprocess/scripts_charades/05_generate_match_grounding_charades.bash
      preprocess/scripts_charades/06_convert_to_train_oneturn_charades.bash
      preprocess/scripts_charades/08_build_hard_negatives_charades.bash
      preprocess/scripts_charades/09_build_verified_search_minimal_top1_charades.bash
    )
    ;;
  *)
    echo "[data_construct][error] unknown dataset: ${dataset}" >&2
    usage
    exit 1
    ;;
esac

idx=0
for step in "${steps[@]}"; do
  idx=$((idx + 1))
  if (( idx < run_from_step )); then
    continue
  fi
  echo "[data_construct] step ${idx}/${#steps[@]}: ${step}"
  bash "${repo_root}/${step}"
done

#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
source "${repo_root}/scripts/common/env.bash"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/data_construct/prepare_raw_layout.bash all
  bash scripts/data_construct/prepare_raw_layout.bash activitynet
  bash scripts/data_construct/prepare_raw_layout.bash didemo
  bash scripts/data_construct/prepare_raw_layout.bash charades

This creates the expected raw annotation/video folders and prints the files
that must be placed there before running start_from_scratch.bash.
Use download_verified_annotations.bash to fetch the VERIFIED FIG annotations.
Raw videos must be downloaded from the original benchmark sources.
EOF
}

dataset="${1:-all}"
if [[ "${dataset}" == "-h" || "${dataset}" == "--help" ]]; then
  usage
  exit 0
fi

make_activitynet() {
  mkdir -p \
    "${VIDEOSEARCH_DATA_ROOT}/fine-grained-anno/activitynet-fig" \
    "${VIDEOSEARCH_DATA_ROOT}/raw_videos/activitynet/videos"
  cat <<EOF
[activitynet]
Annotations:
  ${VIDEOSEARCH_DATA_ROOT}/fine-grained-anno/activitynet-fig/activitynet_fig_train.jsonl
  ${VIDEOSEARCH_DATA_ROOT}/fine-grained-anno/activitynet-fig/activitynet_fig_val_1.jsonl
  ${VIDEOSEARCH_DATA_ROOT}/fine-grained-anno/activitynet-fig/activitynet_fig_val_2.jsonl
Raw videos:
  ${VIDEOSEARCH_DATA_ROOT}/raw_videos/activitynet/videos/<video_id>.mp4
Source:
  VERIFIED FIG annotations:
    bash scripts/data_construct/download_verified_annotations.bash activitynet
  ActivityNet Captions: https://cs.stanford.edu/people/ranjaykrishna/densevid/
  ActivityNet videos: http://activity-net.org/download.html
  Save/link files as v_<youtube_id>.mp4, matching the VERIFIED "video" field.

EOF
}

make_didemo() {
  mkdir -p \
    "${VIDEOSEARCH_DATA_ROOT}/fine-grained-anno/didemo-fig" \
    "${VIDEOSEARCH_DATA_ROOT}/raw_videos/didemo/videos"
  cat <<EOF
[didemo]
Annotations:
  ${VIDEOSEARCH_DATA_ROOT}/fine-grained-anno/didemo-fig/didemo_fig_train.jsonl
  ${VIDEOSEARCH_DATA_ROOT}/fine-grained-anno/didemo-fig/didemo_fig_val.jsonl
  ${VIDEOSEARCH_DATA_ROOT}/fine-grained-anno/didemo-fig/didemo_fig_test.jsonl
Raw videos:
  ${VIDEOSEARCH_DATA_ROOT}/raw_videos/didemo/videos/<video_id>.mp4
Source:
  VERIFIED FIG annotations:
    bash scripts/data_construct/download_verified_annotations.bash didemo
  DiDeMo: https://github.com/LisaAnne/LocalizingMoments
  Use the official DiDeMo download scripts, preferably download/download_videos_AWS.py.
  Save/link files as <video_id>.mp4, matching the VERIFIED "video" field.

EOF
}

make_charades() {
  mkdir -p \
    "${VIDEOSEARCH_DATA_ROOT}/fine-grained-anno/charades-fig" \
    "${VIDEOSEARCH_DATA_ROOT}/raw_videos/charades_sta/videos"
  cat <<EOF
[charades]
Annotations:
  ${VIDEOSEARCH_DATA_ROOT}/fine-grained-anno/charades-fig/charades_fig_train.jsonl
  ${VIDEOSEARCH_DATA_ROOT}/fine-grained-anno/charades-fig/charades_fig_test.jsonl
Raw videos:
  ${VIDEOSEARCH_DATA_ROOT}/raw_videos/charades_sta/videos/<video_id>.mp4
Source:
  VERIFIED FIG annotations:
    bash scripts/data_construct/download_verified_annotations.bash charades
  Charades: https://prior.allenai.org/projects/charades
  Charades videos: https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip
  Charades-STA: https://github.com/jiyanggao/TALL
  Save/link files as <video_id>.mp4, matching the VERIFIED "video" field.

EOF
}

case "${dataset}" in
  all)
    make_activitynet
    make_didemo
    make_charades
    ;;
  activitynet) make_activitynet ;;
  didemo) make_didemo ;;
  charades|charades-sta|charades_sta) make_charades ;;
  *)
    echo "[prepare_raw_layout][error] unknown dataset: ${dataset}" >&2
    usage
    exit 1
    ;;
esac

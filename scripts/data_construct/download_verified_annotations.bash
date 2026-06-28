#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
source "${repo_root}/scripts/common/env.bash"

base_url="${VERIFIED_RAW_BASE_URL:-https://raw.githubusercontent.com/hlchen23/VERIFIED/main/fine-grained-anno}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/data_construct/download_verified_annotations.bash all
  bash scripts/data_construct/download_verified_annotations.bash activitynet
  bash scripts/data_construct/download_verified_annotations.bash didemo
  bash scripts/data_construct/download_verified_annotations.bash charades

Downloads VERIFIED FIG annotation JSONL files into VIDEOSEARCH_DATA_ROOT.
Raw videos must still be downloaded from the original benchmark sources.
EOF
}

download_file() {
  local url="$1"
  local dst="$2"
  mkdir -p "$(dirname "${dst}")"
  if [[ -s "${dst}" && "${OVERWRITE:-0}" != "1" ]]; then
    echo "[verified_annotations] exists: ${dst}"
    return 0
  fi
  echo "[verified_annotations] ${url} -> ${dst}"
  curl -L --fail --retry 3 --retry-delay 2 --connect-timeout 20 --max-time 300 -o "${dst}.tmp" "${url}"
  mv "${dst}.tmp" "${dst}"
}

download_activitynet() {
  local out="${VIDEOSEARCH_DATA_ROOT}/fine-grained-anno/activitynet-fig"
  download_file "${base_url}/activitynet-fig/activitynet_fig_train.jsonl" "${out}/activitynet_fig_train.jsonl"
  download_file "${base_url}/activitynet-fig/activitynet_fig_val_1.jsonl" "${out}/activitynet_fig_val_1.jsonl"
  download_file "${base_url}/activitynet-fig/activitynet_fig_val_2.jsonl" "${out}/activitynet_fig_val_2.jsonl"
}

download_didemo() {
  local out="${VIDEOSEARCH_DATA_ROOT}/fine-grained-anno/didemo-fig"
  download_file "${base_url}/didemo-fig/didemo_fig_train.jsonl" "${out}/didemo_fig_train.jsonl"
  download_file "${base_url}/didemo-fig/didemo_fig_val.jsonl" "${out}/didemo_fig_val.jsonl"
  download_file "${base_url}/didemo-fig/didemo_fig_test.jsonl" "${out}/didemo_fig_test.jsonl"
}

download_charades() {
  local out="${VIDEOSEARCH_DATA_ROOT}/fine-grained-anno/charades-fig"
  download_file "${base_url}/charades-fig/charades_fig_train.jsonl" "${out}/charades_fig_train.jsonl"
  download_file "${base_url}/charades-fig/charades_fig_test.jsonl" "${out}/charades_fig_test.jsonl"
}

dataset="${1:-all}"
case "${dataset}" in
  -h|--help)
    usage
    ;;
  all)
    download_activitynet
    download_didemo
    download_charades
    ;;
  activitynet)
    download_activitynet
    ;;
  didemo)
    download_didemo
    ;;
  charades|charades-sta|charades_sta)
    download_charades
    ;;
  *)
    echo "[verified_annotations][error] unknown dataset: ${dataset}" >&2
    usage
    exit 1
    ;;
esac

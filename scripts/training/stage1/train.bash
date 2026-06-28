#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/training/stage1/train.bash didemo
  bash scripts/training/stage1/train.bash charades
  bash scripts/training/stage1/train.bash activitynet

Useful env:
  GPUS=0
  MODEL_PATH=Qwen/Qwen3-VL-4B-Instruct
  OUTPUT_DIR=/path/to/output
  MAX_SAMPLES=0
EOF
}

dataset="${1:-}"
if [[ -z "${dataset}" || "${dataset}" == "-h" || "${dataset}" == "--help" ]]; then
  usage
  exit 0
fi
shift || true

case "${dataset}" in
  didemo) wrapper="${repo_root}/scripts/internal/stage1/scripts/didemo/verification_setting.bash" ;;
  charades|charades-sta|charades_sta) wrapper="${repo_root}/scripts/internal/stage1/scripts/charades/verification_setting.bash" ;;
  activitynet) wrapper="${repo_root}/scripts/internal/stage1/scripts/activitynet/verification_setting.bash" ;;
  *)
    echo "[stage1][error] unknown dataset: ${dataset}" >&2
    usage
    exit 1
    ;;
esac

exec env GPUS="${GPUS:-0}" bash "${wrapper}" "$@"

#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/training/stage2/train.bash didemo
  bash scripts/training/stage2/train.bash charades
  bash scripts/training/stage2/train.bash activitynet

Useful env:
  GPUS=0
  MODEL_PATH=didemo-stage1
  OUTPUT_DIR=/path/to/output
  NUM_EPOCHS=1
  PER_DEVICE_TRAIN_BATCH_SIZE=1
EOF
}

dataset="${1:-}"
if [[ -z "${dataset}" || "${dataset}" == "-h" || "${dataset}" == "--help" ]]; then
  usage
  exit 0
fi
shift || true

case "${dataset}" in
  didemo) wrapper="${repo_root}/scripts/internal/stage2/scripts/didemo/stage2.bash" ;;
  charades|charades-sta|charades_sta) wrapper="${repo_root}/scripts/internal/stage2/scripts/charades/stage2.bash" ;;
  activitynet) wrapper="${repo_root}/scripts/internal/stage2/scripts/activitynet/stage2.bash" ;;
  *)
    echo "[stage2][error] unknown dataset: ${dataset}" >&2
    usage
    exit 1
    ;;
esac

exec env GPUS="${GPUS:-0}" TRAIN_GPUS="${TRAIN_GPUS:-${GPUS:-0}}" bash "${wrapper}" "$@"

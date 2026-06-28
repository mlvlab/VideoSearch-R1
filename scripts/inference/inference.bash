#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/inference/inference.bash didemo
  bash scripts/inference/inference.bash charades
  bash scripts/inference/inference.bash didemo --checkpoint /path/to/checkpoint
  bash scripts/inference/inference.bash didemo --checkpoint didemo-stage1
  bash scripts/inference/inference.bash didemo --sequential

Datasets:
  didemo | charades | activitynet

Defaults:
  didemo      -> VideoSearchR1/didemo-stage2
  charades    -> VideoSearchR1/charades-stage2
  activitynet -> VideoSearchR1/activitynet-stage2

Useful env:
  EVAL_GPUS=0
  NUM_PROCESSES_PER_GPU=1
  MAX_SAMPLES=0
  MAX_TURN=2
  USE_VLLM=True
  LOGS_DIR=/path/to/logs
EOF
}

dataset="${1:-}"
if [[ -z "${dataset}" || "${dataset}" == "-h" || "${dataset}" == "--help" ]]; then
  usage
  exit 0
fi
shift || true

checkpoint=""
sequential="False"
pass_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint)
      checkpoint="${2:-}"
      if [[ -z "${checkpoint}" ]]; then
        echo "[inference][error] --checkpoint requires a value" >&2
        exit 1
      fi
      shift 2
      ;;
    --sequential)
      sequential="True"
      shift
      ;;
    --use_updated_query|--no-use_updated_query)
      pass_args+=("$1")
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --*)
      echo "[inference][error] unknown option: $1" >&2
      usage
      exit 1
      ;;
    *)
      if [[ -z "${checkpoint}" ]]; then
        checkpoint="$1"
      else
        echo "[inference][error] unexpected positional argument: $1" >&2
        usage
        exit 1
      fi
      shift
      ;;
  esac
done

case "${dataset}" in
  didemo)
    wrapper_dir="${repo_root}/scripts/internal/stage1/scripts/didemo"
    default_checkpoint="didemo-stage2"
    ;;
  charades|charades-sta|charades_sta)
    wrapper_dir="${repo_root}/scripts/internal/stage1/scripts/charades"
    default_checkpoint="charades-stage2"
    ;;
  activitynet)
    wrapper_dir="${repo_root}/scripts/internal/stage1/scripts/activitynet"
    default_checkpoint="activitynet-stage2"
    ;;
  *)
    echo "[inference][error] unknown dataset: ${dataset}" >&2
    usage
    exit 1
    ;;
esac

target="test_checkpoint_temporal_grounding.bash"
if [[ "${sequential}" == "True" ]]; then
  target="test_checkpoint_temporal_grounding_sequential.bash"
fi

args=()
if [[ -n "${checkpoint}" ]]; then
  args+=("${checkpoint}")
fi
args+=("${pass_args[@]}")

exec env \
  EVAL_GPUS="${EVAL_GPUS:-0}" \
  NUM_PROCESSES_PER_GPU="${NUM_PROCESSES_PER_GPU:-1}" \
  MAX_TURN="${MAX_TURN:-2}" \
  DEFAULT_CHECKPOINT="${DEFAULT_CHECKPOINT:-${default_checkpoint}}" \
  bash "${wrapper_dir}/${target}" "${args[@]}"

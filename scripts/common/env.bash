#!/usr/bin/env bash

_videosearch_env_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VIDEOSEARCH_REPO_ROOT="${VIDEOSEARCH_REPO_ROOT:-$(cd "${_videosearch_env_dir}/../.." && pwd)}"

_videosearch_default_workspace() {
  local candidate
  local username="${USER:-$(id -un 2>/dev/null || echo user)}"
  local min_free_gb="${VIDEOSEARCH_MIN_FREE_GB:-20}"
  for candidate in "/hub_data2/${username}/videosearchr1" "/hub_data1/${username}/videosearchr1" "/hub_data3/${username}/videosearchr1"; do
    if mkdir -p "${candidate}" 2>/dev/null; then
      local avail_kb
      avail_kb="$(df -Pk "${candidate}" 2>/dev/null | awk 'NR==2 {print $4}')"
      if [[ -z "${avail_kb}" || "${avail_kb}" -ge $((min_free_gb * 1024 * 1024)) ]]; then
        echo "${candidate}"
        return 0
      fi
    fi
  done
  echo "${VIDEOSEARCH_REPO_ROOT}/.cache"
}

export VIDEOSEARCH_WORKSPACE="${VIDEOSEARCH_WORKSPACE:-$(_videosearch_default_workspace)}"
export VIDEOSEARCH_DATA_ROOT="${VIDEOSEARCH_DATA_ROOT:-${VIDEOSEARCH_WORKSPACE}/data}"
export VIDEOSEARCH_OUTPUT_ROOT="${VIDEOSEARCH_OUTPUT_ROOT:-${VIDEOSEARCH_WORKSPACE}/outputs}"
export VIDEOSEARCH_CACHE_ROOT="${VIDEOSEARCH_CACHE_ROOT:-${VIDEOSEARCH_WORKSPACE}/cache}"
export VIDEOSEARCH_CHECKPOINT_ROOT="${VIDEOSEARCH_CHECKPOINT_ROOT:-${VIDEOSEARCH_WORKSPACE}/checkpoints}"
export VIDEOSEARCH_TMPDIR="${VIDEOSEARCH_TMPDIR:-${VIDEOSEARCH_WORKSPACE}/tmp}"
export VIDEOSEARCH_HF_ORG="${VIDEOSEARCH_HF_ORG:-VideoSearchR1}"
export VIDEOSEARCH_HF_BUCKET="${VIDEOSEARCH_HF_BUCKET:-hf://buckets/VideoSearchR1/data}"
export VIDEOSEARCH_BASE_MODEL="${VIDEOSEARCH_BASE_MODEL:-Qwen/Qwen3-VL-4B-Instruct}"
export VIDEOSEARCH_EMBED_MODEL="${VIDEOSEARCH_EMBED_MODEL:-Qwen/Qwen3-VL-Embedding-2B}"
export VIDEOSEARCH_REASONING_MODEL="${VIDEOSEARCH_REASONING_MODEL:-Qwen/Qwen3-VL-30B-A3B-Thinking}"

mkdir -p \
  "${VIDEOSEARCH_DATA_ROOT}" \
  "${VIDEOSEARCH_OUTPUT_ROOT}" \
  "${VIDEOSEARCH_CACHE_ROOT}/hf" \
  "${VIDEOSEARCH_CACHE_ROOT}/torch_extensions" \
  "${VIDEOSEARCH_CACHE_ROOT}/triton" \
  "${VIDEOSEARCH_CHECKPOINT_ROOT}" \
  "${VIDEOSEARCH_TMPDIR}"

export HF_HOME="${HF_HOME:-${VIDEOSEARCH_CACHE_ROOT}/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${VIDEOSEARCH_CACHE_ROOT}/torch_extensions}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${VIDEOSEARCH_CACHE_ROOT}/triton}"
export WANDB_DIR="${WANDB_DIR:-${VIDEOSEARCH_WORKSPACE}/wandb}"
export TMPDIR="${TMPDIR:-${VIDEOSEARCH_TMPDIR}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

videosearch_dataset_dir() {
  case "${1:-}" in
    activitynet) echo "${VIDEOSEARCH_DATA_ROOT}/activitynet" ;;
    charades|charades-sta|charades_sta) echo "${VIDEOSEARCH_DATA_ROOT}/charades-sta" ;;
    didemo) echo "${VIDEOSEARCH_DATA_ROOT}/didemo" ;;
    *)
      echo "[videosearch_env][error] unknown dataset: ${1:-}" >&2
      return 1
      ;;
  esac
}

videosearch_hf_model_repo() {
  case "${1:-}" in
    didemo-stage1) echo "${VIDEOSEARCH_HF_ORG}/didemo-stage1" ;;
    didemo-stage2) echo "${VIDEOSEARCH_HF_ORG}/didemo-stage2" ;;
    charades-stage1|charades-sta-stage1) echo "${VIDEOSEARCH_HF_ORG}/charades-stage1" ;;
    charades-stage2|charades-sta-stage2) echo "${VIDEOSEARCH_HF_ORG}/charades-stage2" ;;
    activitynet-stage1) echo "${VIDEOSEARCH_HF_ORG}/activitynet-stage1" ;;
    activitynet-stage2) echo "${VIDEOSEARCH_HF_ORG}/activitynet-stage2" ;;
    *)
      echo "${1:-}"
      ;;
  esac
}

videosearch_resolve_hf_model() {
  local value="${1:-}"
  case "${value}" in
    didemo-stage1|didemo-stage2|charades-stage1|charades-stage2|charades-sta-stage1|charades-sta-stage2|activitynet-stage1|activitynet-stage2)
      videosearch_hf_model_repo "${value}"
      ;;
    *)
      echo "${value}"
      ;;
  esac
}

videosearch_local_model_path() {
  local value
  value="$(videosearch_resolve_hf_model "${1:-}")"
  if [[ -z "${value}" || -d "${value}" ]]; then
    echo "${value}"
    return 0
  fi
  if [[ "${value}" != */* ]]; then
    echo "${value}"
    return 0
  fi

  local target="${VIDEOSEARCH_CHECKPOINT_ROOT}/${value//\//__}"
  mkdir -p "${target}"
  echo "[videosearch_env] snapshot model ${value} -> ${target}" >&2
  python - "${value}" "${target}" <<'PY'
import sys
from huggingface_hub import snapshot_download

repo_id, target = sys.argv[1], sys.argv[2]
snapshot_download(
    repo_id=repo_id,
    repo_type="model",
    local_dir=target,
    local_dir_use_symlinks=False,
    resume_download=True,
)
PY
  echo "${target}"
}
